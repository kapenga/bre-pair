//! Fixed-budget vocabulary selection and bounded, parallel corpus scoring.
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use crate::engine::Compressed;
use crate::grammar::{Symbol, BASE_SYMBOLS, EOF};
use crate::tokenizer::{Bpe, Model, Token, Tokenizer};

mod coverage;
pub mod frequency_estimates;
pub mod reverse;

/// Expose frequent raw byte pairs independently of grammar merge order.
/// Uses one 65,536-entry u64 array, never per-worker pair maps. Added entries
/// receive zero grammar-support prior; actual parser usage decides their value.
pub fn augment_byte_pairs(
    pool: &mut Model,
    support: &mut Vec<u64>,
    training: &Path,
    minimum: u64,
) -> Result<usize, String> {
    augment_byte_pairs_with_prior(pool, support, training, minimum, false)
}

/// Optionally replace byte-pair support priors with exact raw replacement counts.
/// This does not replace the final exact parser-usage counts.
pub fn augment_byte_pairs_with_prior(
    pool: &mut Model,
    support: &mut Vec<u64>,
    training: &Path,
    minimum: u64,
    raw_prior: bool,
) -> Result<usize, String> {
    use std::io::Read;
    if minimum == 0 || pool.tokens.len() != support.len() {
        return Err("invalid byte-pair augmentation threshold or support length".into());
    }
    let mut counts = vec![0u64; 65536];
    let mut endings = [0u64; 256];
    let mut input = File::open(training).map_err(|e| e.to_string())?;
    let mut buffer = [0; 65536];
    let mut previous = None;
    let mut run = 0u64;
    loop {
        let read = input.read(&mut buffer).map_err(|e| e.to_string())?;
        if read == 0 {
            break;
        }
        for &b in &buffer[..read] {
            run = if previous == Some(b) { run + 1 } else { 1 };
            if let Some(a) = previous {
                if a != b || run.is_multiple_of(2) {
                    counts[(usize::from(a) << 8) | usize::from(b)] += 1;
                }
            }
            previous = Some(b);
            if b == b'\n' {
                endings[usize::from(b)] += 1;
                previous = None;
                run = 0;
            }
        }
    }
    if let Some(b) = previous {
        endings[usize::from(b)] += 1;
    }
    let mut present = vec![false; 65536];
    let mut present_endings = [false; 256];
    for (token, prior) in pool.tokens.iter().zip(support.iter_mut()) {
        match (token.bytes.as_slice(), token.ends_field) {
            ([a, b], false) => {
                let key = (usize::from(*a) << 8) | usize::from(*b);
                present[key] = true;
                if raw_prior {
                    *prior = counts[key];
                }
            }
            ([b], true) => {
                present_endings[usize::from(*b)] = true;
                if raw_prior {
                    *prior = endings[usize::from(*b)];
                }
            }
            _ => {}
        }
    }
    let before = pool.tokens.len();
    for (key, &count) in counts.iter().enumerate() {
        if count >= minimum && !present[key] {
            pool.tokens.push(Token {
                bytes: vec![(key >> 8) as u8, key as u8],
                ends_field: false,
            });
            support.push(if raw_prior { count } else { 0 });
        }
    }
    for (b, &count) in endings.iter().enumerate() {
        if count >= minimum && !present_endings[b] {
            pool.tokens.push(Token {
                bytes: vec![b as u8],
                ends_field: true,
            });
            support.push(if raw_prior { count } else { 0 });
        }
    }
    Ok(pool.tokens.len() - before)
}

/// Require the archive to represent exactly these raw line fields, not merely
/// a file with the same length. Retains only one source field during comparison.
pub fn verify_training_source(source: &Compressed, path: &Path) -> Result<(), String> {
    let progress = crate::training_progress::enabled();
    let mut next_progress = 64 * 1024 * 1024u64;
    crate::training_progress::phase("verify_source", 0, 0, source.input_bytes, "bytes");
    let mut input = BufReader::new(File::open(path).map_err(|e| e.to_string())?);
    let mut field = Vec::new();
    input
        .read_until(b'\n', &mut field)
        .map_err(|e| e.to_string())?;
    let mut at = 0usize;
    let mut bytes = 0u64;
    let mut fields = 0u64;
    let mut stack = Vec::new();
    for block in &source.blocks {
        for symbol in block.iter() {
            stack.push(symbol);
            while let Some(symbol) = stack.pop() {
                if symbol == EOF {
                    if field.is_empty() || at != field.len() {
                        return Err(format!(
                            "candidate archive field boundary differs at byte {bytes}"
                        ));
                    }
                    fields += 1;
                    if progress && bytes >= next_progress {
                        crate::training_progress::phase(
                            "verify_source",
                            0,
                            bytes,
                            source.input_bytes,
                            "bytes",
                        );
                        next_progress = bytes.saturating_add(64 * 1024 * 1024);
                    }
                    field.clear();
                    input
                        .read_until(b'\n', &mut field)
                        .map_err(|e| e.to_string())?;
                    at = 0;
                } else if symbol < EOF {
                    if field.get(at).copied() != Some(symbol as u8) {
                        return Err(format!(
                            "candidate archive bytes differ from training input at byte {bytes}"
                        ));
                    }
                    at += 1;
                    bytes += 1;
                } else {
                    let rule = source
                        .grammar
                        .rules
                        .get(symbol as usize - BASE_SYMBOLS)
                        .ok_or("undefined candidate archive symbol")?;
                    stack.push(rule.right);
                    stack.push(rule.left);
                }
            }
        }
    }
    if !field.is_empty() || at != 0 || bytes != source.input_bytes || fields != source.field_count {
        return Err("candidate archive does not cover the exact training line fields".into());
    }
    crate::training_progress::phase("verify_source", 0, bytes, source.input_bytes, "bytes");
    Ok(())
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Totals {
    pub bytes: u64,
    pub tokens: u64,
    pub fields: u64,
    pub byte_tokens: u64,
}
impl Totals {
    pub(crate) fn add(&mut self, other: Self) {
        self.bytes += other.bytes;
        self.tokens += other.tokens;
        self.fields += other.fields;
        self.byte_tokens += other.byte_tokens;
    }
    pub fn bytes_per_token(&self) -> f64 {
        self.bytes as f64 / self.tokens.max(1) as f64
    }
}

pub struct Scan {
    pub all: Totals,
    pub heldout: Totals,
    pub uses: Vec<u64>,
    pub seconds: f64,
}

struct Input {
    reader: BufReader<File>,
    offset: u64,
    field: u64,
    progress: Option<InputProgress>,
}
struct InputProgress {
    phase: &'static str,
    round: usize,
    total: u64,
    next: u64,
}
struct Field {
    bytes: Vec<u8>,
    start: u64,
}
impl Input {
    fn batch(&mut self, stride: u64) -> Result<Vec<Field>, String> {
        let mut batch = Vec::new();
        let mut selected = 0;
        while batch.len() < 512 && selected < 64 * 1024 {
            let mut bytes = Vec::new();
            let read = self
                .reader
                .read_until(b'\n', &mut bytes)
                .map_err(|e| e.to_string())?;
            if read == 0 {
                break;
            }
            let start = self.offset;
            self.offset += read as u64;
            let keep = self.field % stride == 0;
            self.field += 1;
            if keep {
                selected += read;
                batch.push(Field { bytes, start });
            }
        }
        if let Some(progress) = &mut self.progress {
            if self.offset >= progress.next || batch.is_empty() {
                crate::training_progress::phase(
                    progress.phase,
                    progress.round,
                    self.offset,
                    progress.total,
                    "bytes",
                );
                progress.next = self.offset.saturating_add(64 * 1024 * 1024);
            }
        }
        Ok(batch)
    }
}

/// Immutable model/index shared by workers. Counts have one central dense
/// array and each worker combines at most 4096 IDs; there are no worker maps.
#[allow(clippy::too_many_arguments)]
pub fn scan(
    path: &Path,
    tokenizer: &Tokenizer,
    bpe: Option<&Bpe>,
    active: Option<&[u8]>,
    workers: usize,
    stride: u64,
    overlap_bytes: u64,
    collect_uses: bool,
    verify: bool,
) -> Result<Scan, String> {
    scan_with_progress(
        path,
        tokenizer,
        bpe,
        active,
        workers,
        stride,
        overlap_bytes,
        collect_uses,
        verify,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn scan_with_progress(
    path: &Path,
    tokenizer: &Tokenizer,
    bpe: Option<&Bpe>,
    active: Option<&[u8]>,
    workers: usize,
    stride: u64,
    overlap_bytes: u64,
    collect_uses: bool,
    verify: bool,
    progress: Option<(&'static str, usize)>,
) -> Result<Scan, String> {
    if workers == 0 || workers > 256 || stride == 0 {
        return Err("invalid workers or sample stride".into());
    }
    if active
        .is_some_and(|a| a.len() != tokenizer.model.tokens.len() || a[..BASE_SYMBOLS].contains(&0))
    {
        return Err("active vocabulary must retain the complete basis".into());
    }
    if let Some(bpe) = bpe {
        if active.is_some() || bpe.model.tokens != tokenizer.model.tokens {
            return Err("BPE scoring requires its complete matching vocabulary".into());
        }
    }
    let started = Instant::now();
    let progress = progress
        .map(|(phase, round)| {
            let total = std::fs::metadata(path).map_err(|e| e.to_string())?.len();
            crate::training_progress::phase(phase, round, 0, total, "bytes");
            Ok::<_, String>(InputProgress {
                phase,
                round,
                total,
                next: 64 * 1024 * 1024,
            })
        })
        .transpose()?;
    let input = Mutex::new(Input {
        reader: BufReader::new(File::open(path).map_err(|e| e.to_string())?),
        offset: 0,
        field: 0,
        progress,
    });
    let uses: Vec<_> = if collect_uses {
        (0..tokenizer.model.tokens.len())
            .map(|_| AtomicU64::new(0))
            .collect()
    } else {
        vec![]
    };
    let (all, heldout) = std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for _ in 0..workers {
            let input = &input;
            let uses = &uses;
            handles.push(scope.spawn(move || {
                let mut all = Totals::default();
                let mut heldout = Totals::default();
                let mut log: Vec<Symbol> = Vec::with_capacity(4096);
                let flush = |log: &mut Vec<Symbol>| {
                    log.sort_unstable();
                    let mut at = 0;
                    while at < log.len() {
                        let mut end = at + 1;
                        while end < log.len() && log[end] == log[at] {
                            end += 1;
                        }
                        uses[log[at] as usize].fetch_add((end - at) as u64, Ordering::Relaxed);
                        at = end;
                    }
                    log.clear();
                };
                loop {
                    let batch = input
                        .lock()
                        .map_err(|_| "poisoned corpus reader")?
                        .batch(stride)?;
                    if batch.is_empty() {
                        break;
                    }
                    for field in batch {
                        let ids = match bpe {
                            Some(bpe) => bpe.encode(&field.bytes),
                            None => tokenizer.encode_active(&field.bytes, active),
                        };
                        if verify && tokenizer.model.decode_field(&ids)? != field.bytes {
                            return Err("tokenizer round trip differs from source".into());
                        }
                        let totals = Totals {
                            bytes: field.bytes.len() as u64,
                            tokens: ids.len() as u64,
                            fields: 1,
                            byte_tokens: ids.iter().filter(|&&id| id < 256).count() as u64,
                        };
                        all.add(totals);
                        // A line spanning the training cutoff belongs to the
                        // seen partition, keeping the held-out range disjoint.
                        if field.start >= overlap_bytes {
                            heldout.add(totals);
                        }
                        if collect_uses {
                            for id in ids {
                                log.push(id);
                                if log.len() == 4096 {
                                    flush(&mut log);
                                }
                            }
                        }
                    }
                }
                if collect_uses {
                    flush(&mut log);
                }
                Ok::<_, String>((all, heldout))
            }));
        }
        let mut all = Totals::default();
        let mut heldout = Totals::default();
        for handle in handles {
            let (a, h) = handle.join().map_err(|_| "tokenizer worker panicked")??;
            all.add(a);
            heldout.add(h);
        }
        Ok::<_, String>((all, heldout))
    })?;
    Ok(Scan {
        all,
        heldout,
        uses: uses.into_iter().map(AtomicU64::into_inner).collect(),
        seconds: started.elapsed().as_secs_f64(),
    })
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum SelectionMethod {
    /// Exact occurrence counts propagated through the complete training grammar.
    #[default]
    Statistics,
    /// Historical removal estimates from repeated parses of a field sample.
    Sampled,
}
impl SelectionMethod {
    pub fn name(self) -> &'static str {
        match self {
            Self::Statistics => "statistics",
            Self::Sampled => "sampled",
        }
    }
}

#[derive(Clone, Debug)]
pub struct SelectionOptions {
    pub vocabulary: usize,
    pub workers: usize,
    pub method: SelectionMethod,
    /// Used only by the explicit sampled selection method.
    pub sample_bytes: u64,
    pub full_refine_at: usize,
    /// Used only by the explicit sampled selection method.
    pub support_weight: f64,
}
impl Default for SelectionOptions {
    fn default() -> Self {
        Self {
            vocabulary: 8192,
            workers: 4,
            method: SelectionMethod::Statistics,
            sample_bytes: 10_000_000,
            full_refine_at: 32768,
            support_weight: 0.02,
        }
    }
}

pub struct SelectionRound {
    pub before: usize,
    pub after: usize,
    pub method: SelectionMethod,
    pub stride: Option<u64>,
    pub totals: Option<Totals>,
    pub seconds: f64,
}

/// Prune by full-grammar occurrence counts and replacement cost by default.
/// Counts include occurrences inside retained parents; these fallback pieces
/// must not appear unused merely because a larger candidate currently masks them.
/// The explicit sampled method retains the historical parser-based heuristic.
/// Neither method guarantees a globally optimal vocabulary.
pub fn select(
    pool: Model,
    support: &[u64],
    training: &Path,
    options: &SelectionOptions,
    on_round: impl FnMut(&SelectionRound),
) -> Result<Model, String> {
    let (tokenizer, active) = select_active(pool, support, training, options, on_round)?;
    tokenizer.model.subset(&active)
}

pub struct EstimatedSelectionRound {
    pub before: usize,
    pub after: usize,
    pub estimated: Totals,
    pub seconds: f64,
}

pub struct EmissionSelectionRound {
    pub before: usize,
    pub after: usize,
    pub totals: Totals,
    pub seconds: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct PruningSchedule {
    pub coarse_keep: f64,
    pub fine_keep: f64,
    pub fine_below: usize,
}
impl Default for PruningSchedule {
    fn default() -> Self {
        Self {
            coarse_keep: 0.6,
            fine_keep: 0.85,
            fine_below: 32768,
        }
    }
}

/// Research-only counterpart to sampled pruning. Uses complete collapsed
/// grammar counts instead of sampled parser counts, with the same 0.02 support
/// prior and pruning fractions. Does not reparse or repair the raw corpus.
pub fn select_estimated(
    pool: Model,
    support: &[u64],
    source: &Compressed,
    vocabulary: usize,
    mut on_round: impl FnMut(&EstimatedSelectionRound),
) -> Result<Model, String> {
    if vocabulary < BASE_SYMBOLS
        || vocabulary > pool.tokens.len()
        || support.len() != pool.tokens.len()
    {
        return Err("invalid estimated-selection vocabulary or support".into());
    }
    let counter = frequency_estimates::Counter::new(source, &pool)?;
    select_from_counts(
        pool,
        support,
        vocabulary,
        PruningSchedule::default(),
        |_, active| counter.count(active),
        |r| {
            on_round(&EstimatedSelectionRound {
                before: r.before,
                after: r.after,
                estimated: r.totals,
                seconds: r.seconds,
            });
        },
    )
}

/// Research control: one exact full-corpus parse before every pruning batch.
/// Score and pruning schedule are shared with the estimated selector. There is
/// no sampling or post-selection repair; the vocabulary search is heuristic.
pub fn select_exact(
    pool: Model,
    support: &[u64],
    training: &Path,
    vocabulary: usize,
    workers: usize,
    on_round: impl FnMut(&EmissionSelectionRound),
) -> Result<Model, String> {
    select_exact_with_schedule(
        pool,
        support,
        training,
        vocabulary,
        workers,
        PruningSchedule::default(),
        on_round,
    )
}

pub fn select_exact_with_schedule(
    pool: Model,
    support: &[u64],
    training: &Path,
    vocabulary: usize,
    workers: usize,
    schedule: PruningSchedule,
    on_round: impl FnMut(&EmissionSelectionRound),
) -> Result<Model, String> {
    if !(1..=256).contains(&workers) {
        return Err("invalid exact-selection workers".into());
    }
    let bytes = std::fs::metadata(training)
        .map_err(|e| e.to_string())?
        .len();
    let mut round = 0;
    select_from_counts(
        pool,
        support,
        vocabulary,
        schedule,
        |tokenizer, active| {
            round += 1;
            let measured = scan_full_active(training, tokenizer, active, workers, round)?;
            if measured.all.bytes != bytes {
                return Err("exact selection did not cover the full training file".into());
            }
            Ok((measured.uses, measured.all))
        },
        on_round,
    )
}

fn scan_full_active(
    path: &Path,
    tokenizer: &Tokenizer,
    active: &[u8],
    workers: usize,
    round: usize,
) -> Result<Scan, String> {
    if !active.contains(&0) {
        return scan_with_progress(
            path,
            tokenizer,
            None,
            None,
            workers,
            1,
            0,
            true,
            false,
            Some(("vocabulary_exact_scan", round)),
        );
    }
    // Removing inactive trie entries preserves token order and all tie breaks,
    // while avoiding traversal of millions of discarded dictionary paths.
    let ids: Vec<_> = active
        .iter()
        .enumerate()
        .filter_map(|(id, &on)| (on != 0).then_some(id))
        .collect();
    let compact = Tokenizer::new(tokenizer.model.subset(active)?)?;
    let mut measured = scan_with_progress(
        path,
        &compact,
        None,
        None,
        workers,
        1,
        0,
        true,
        false,
        Some(("vocabulary_exact_scan", round)),
    )?;
    let mut uses = vec![0; active.len()];
    for (id, n) in ids.into_iter().zip(measured.uses) {
        uses[id] = n;
    }
    measured.uses = uses;
    Ok(measured)
}

fn select_from_counts(
    pool: Model,
    support: &[u64],
    vocabulary: usize,
    schedule: PruningSchedule,
    mut count: impl FnMut(&Tokenizer, &[u8]) -> Result<(Vec<u64>, Totals), String>,
    mut on_round: impl FnMut(&EmissionSelectionRound),
) -> Result<Model, String> {
    if vocabulary < BASE_SYMBOLS
        || vocabulary > pool.tokens.len()
        || support.len() != pool.tokens.len()
        || !(0.0..1.0).contains(&schedule.coarse_keep)
        || schedule.coarse_keep == 0.0
        || !(0.0..1.0).contains(&schedule.fine_keep)
        || schedule.fine_keep == 0.0
        || schedule.fine_below < BASE_SYMBOLS
    {
        return Err("invalid emission-selection vocabulary or support".into());
    }
    let tokenizer = Tokenizer::new(pool)?;
    let mut active = vec![1u8; support.len()];
    let mut remaining = active.len();
    let mut previous_tokens = 0;
    while remaining > vocabulary {
        let timer = Instant::now();
        let (uses, totals) = count(&tokenizer, &active)?;
        if uses.len() != active.len()
            || uses.iter().map(|&n| u128::from(n)).sum::<u128>() != u128::from(totals.tokens)
            || totals.tokens < previous_tokens
        {
            return Err("emission counts violate totals or subset monotonicity".into());
        }
        previous_tokens = totals.tokens;
        let mut losses = Vec::with_capacity(remaining - BASE_SYMBOLS);
        for id in BASE_SYMBOLS..active.len() {
            if active[id] != 0 {
                let alternatives = tokenizer
                    .replacement_cost(id as u32, &active)
                    .saturating_sub(1);
                // (count + support / 50) * replacement cost, scaled by 50.
                // Exact integer arithmetic preserves the historical 0.02 prior.
                let weight = 50 * u128::from(uses[id]) + u128::from(support[id]);
                losses.push((weight * u128::from(alternatives), id));
            }
        }
        losses.sort_unstable();
        let next = if remaining > schedule.fine_below {
            ((remaining as f64 * schedule.coarse_keep) as usize)
                .max(schedule.fine_below)
                .max(vocabulary)
        } else {
            ((remaining as f64 * schedule.fine_keep) as usize).max(vocabulary)
        };
        for &(_, id) in &losses[..remaining - next] {
            active[id] = 0;
        }
        on_round(&EmissionSelectionRound {
            before: remaining,
            after: next,
            totals,
            seconds: timer.elapsed().as_secs_f64(),
        });
        remaining = next;
    }
    tokenizer.model.subset(&active)
}

fn select_active(
    pool: Model,
    support: &[u64],
    training: &Path,
    options: &SelectionOptions,
    mut on_round: impl FnMut(&SelectionRound),
) -> Result<(Tokenizer, Vec<u8>), String> {
    if options.vocabulary < BASE_SYMBOLS
        || options.vocabulary > pool.tokens.len()
        || support.len() != pool.tokens.len()
        || !(1..=256).contains(&options.workers)
        || (options.method == SelectionMethod::Sampled
            && (options.sample_bytes == 0
                || !options.support_weight.is_finite()
                || options.support_weight < 0.0))
    {
        return Err("invalid selection vocabulary or support prior".into());
    }
    let input_bytes = if options.method == SelectionMethod::Sampled {
        std::fs::metadata(training)
            .map_err(|e| e.to_string())?
            .len()
    } else {
        0
    };
    let sample_stride = input_bytes.div_ceil(options.sample_bytes.max(1)).max(1);
    let tokenizer = Tokenizer::new(pool)?;
    let mut active = vec![1u8; tokenizer.model.tokens.len()];
    let mut remaining = active.len();
    while remaining > options.vocabulary {
        let started = Instant::now();
        let next = if remaining > options.full_refine_at {
            ((remaining as f64 * 0.6) as usize)
                .max(options.full_refine_at)
                .max(options.vocabulary)
        } else {
            ((remaining as f64 * 0.85) as usize).max(options.vocabulary)
        };
        let (stride, totals) = match options.method {
            SelectionMethod::Statistics => {
                let mut losses = Vec::with_capacity(remaining - BASE_SYMBOLS);
                for id in BASE_SYMBOLS..active.len() {
                    if active[id] != 0 {
                        let alternatives = tokenizer.replacement_cost(id as u32, &active);
                        // Keep 64-bit corpus counts and their products exact, even
                        // above the integer precision available in f64.
                        let loss =
                            u128::from(support[id]) * u128::from(alternatives.saturating_sub(1));
                        losses.push((loss, id));
                    }
                }
                losses.sort_unstable();
                for &(_, id) in &losses[..remaining - next] {
                    active[id] = 0;
                }
                (None, None)
            }
            SelectionMethod::Sampled => {
                let stride = if remaining <= options.full_refine_at {
                    1
                } else {
                    sample_stride
                };
                let counted = scan(
                    training,
                    &tokenizer,
                    None,
                    Some(&active),
                    options.workers,
                    stride,
                    0,
                    true,
                    false,
                )?;
                let scale = input_bytes as f64 / counted.all.bytes.max(1) as f64;
                let mut losses = Vec::with_capacity(remaining - BASE_SYMBOLS);
                for id in BASE_SYMBOLS..active.len() {
                    if active[id] != 0 {
                        let alternatives = tokenizer.replacement_cost(id as u32, &active);
                        let observed = counted.uses[id] as f64 * scale;
                        let prior = options.support_weight * support[id] as f64;
                        let loss = (observed + prior) * alternatives.saturating_sub(1) as f64;
                        losses.push((loss, id));
                    }
                }
                losses.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
                for &(_, id) in &losses[..remaining - next] {
                    active[id] = 0;
                }
                (Some(stride), Some(counted.all))
            }
        };
        on_round(&SelectionRound {
            before: remaining,
            after: next,
            method: options.method,
            stride,
            totals,
            seconds: started.elapsed().as_secs_f64(),
        });
        remaining = next;
    }
    Ok((tokenizer, active))
}

#[derive(Clone, Debug)]
pub struct RefinementOptions {
    pub workers: usize,
    pub minimum_uses: u64,
    pub max_rounds: usize,
}
impl Default for RefinementOptions {
    fn default() -> Self {
        Self {
            workers: 4,
            minimum_uses: 8,
            max_rounds: 64,
        }
    }
}

#[derive(Clone, Debug)]
pub struct RefinementRound {
    pub index: usize,
    pub removed: usize,
    pub added: usize,
    pub active: usize,
    pub totals: Totals,
    pub seconds: f64,
}

pub struct Refinement {
    pub model: Model,
    pub before: Totals,
    pub after: Totals,
    pub rounds: Vec<RefinementRound>,
    pub minimum_learned_uses: Option<u64>,
    pub minimum_document_groups: Option<usize>,
    pub minimum_learned_document_groups: Option<u32>,
}

/// Keep one candidate index throughout coarse selection and final refinement.
pub fn select_refined(
    pool: Model,
    support: &[u64],
    training: &Path,
    selection: &SelectionOptions,
    refinement: &RefinementOptions,
    on_select: impl FnMut(&SelectionRound),
    on_refine: impl FnMut(&RefinementRound),
) -> Result<Refinement, String> {
    select_refined_with_documents(
        pool,
        support,
        training,
        selection,
        refinement,
        None,
        (on_select, on_refine),
    )
}

/// Document-qualified candidate pools still receive a final exact parser-usage
/// check before freezing IDs. Document work never enters grammar merge passes.
pub fn select_refined_with_documents(
    pool: Model,
    support: &[u64],
    training: &Path,
    selection: &SelectionOptions,
    refinement: &RefinementOptions,
    documents: Option<&crate::document_support::Policy>,
    callbacks: (impl FnMut(&SelectionRound), impl FnMut(&RefinementRound)),
) -> Result<Refinement, String> {
    let (on_select, on_refine) = callbacks;
    let (tokenizer, active) = select_active(pool, support, training, selection, on_select)?;
    refine_active(
        &tokenizer, active, training, refinement, None, documents, on_refine,
    )
}

/// Refine an already selected vocabulary using the original candidate pool.
/// Only TRAIN is scored; no evaluation corpus participates in this operation.
pub fn refine(
    pool: Model,
    initial: &Model,
    training: &Path,
    options: &RefinementOptions,
    on_round: impl FnMut(&RefinementRound),
) -> Result<Refinement, String> {
    refine_bounded(pool, initial, training, options, None, on_round)
}

/// Clean a proposed exchange against an explicit full-training token ceiling.
/// This permits cleanup to cost tokens while the complete exchange still wins.
pub fn refine_bounded(
    pool: Model,
    initial: &Model,
    training: &Path,
    options: &RefinementOptions,
    token_ceiling: Option<u64>,
    on_round: impl FnMut(&RefinementRound),
) -> Result<Refinement, String> {
    refine_with_documents(
        pool,
        initial,
        training,
        options,
        token_ceiling,
        None,
        on_round,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn refine_with_documents(
    pool: Model,
    initial: &Model,
    training: &Path,
    options: &RefinementOptions,
    token_ceiling: Option<u64>,
    documents: Option<&crate::document_support::Policy>,
    on_round: impl FnMut(&RefinementRound),
) -> Result<Refinement, String> {
    let ids: std::collections::HashMap<_, _> = pool
        .tokens
        .iter()
        .enumerate()
        .map(|(id, token)| (token, id))
        .collect();
    let mut active = vec![0u8; pool.tokens.len()];
    for token in &initial.tokens {
        active[*ids
            .get(token)
            .ok_or("initial token is absent from the candidate pool")?] = 1;
    }
    drop(ids);
    let tokenizer = Tokenizer::new(pool)?;
    refine_active(
        &tokenizer,
        active,
        training,
        options,
        token_ceiling,
        documents,
        on_round,
    )
}

fn proposal_scores(
    path: &Path,
    tokenizer: &Tokenizer,
    active: &[u8],
    banned: &[bool],
    workers: usize,
) -> Result<Vec<u64>, String> {
    let input = Mutex::new(Input {
        reader: BufReader::new(File::open(path).map_err(|e| e.to_string())?),
        offset: 0,
        field: 0,
        progress: None,
    });
    // One central score array and bounded 8 KiB worker logs, never worker maps.
    let scores: Vec<_> = (0..active.len()).map(|_| AtomicU64::new(0)).collect();
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for _ in 0..workers {
            let input = &input;
            let scores = &scores;
            handles.push(scope.spawn(move || {
                let mut log = Vec::<(Symbol, u64)>::with_capacity(512);
                let flush = |log: &mut Vec<(Symbol, u64)>| -> Result<(), String> {
                    log.sort_unstable_by_key(|x| x.0);
                    let mut at = 0;
                    while at < log.len() {
                        let (id, mut sum) = log[at];
                        let mut end = at + 1;
                        while end < log.len() && log[end].0 == id {
                            sum = sum
                                .checked_add(log[end].1)
                                .ok_or("proposal score overflow")?;
                            end += 1;
                        }
                        scores[id as usize]
                            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |old| {
                                old.checked_add(sum)
                            })
                            .map_err(|_| "proposal score overflow")?;
                        at = end;
                    }
                    log.clear();
                    Ok(())
                };
                loop {
                    let batch = input
                        .lock()
                        .map_err(|_| "poisoned corpus reader")?
                        .batch(1)?;
                    if batch.is_empty() {
                        break;
                    }
                    for field in batch {
                        let ids = tokenizer.encode_active(&field.bytes, Some(active));
                        tokenizer.insertion_candidates(
                            &field.bytes,
                            &ids,
                            active,
                            |id, gain| {
                                if !banned[id as usize] {
                                    log.push((id, gain));
                                    if log.len() == 512 {
                                        flush(&mut log)?;
                                    }
                                }
                                Ok(())
                            },
                        )?;
                    }
                }
                flush(&mut log)
            }));
        }
        for handle in handles {
            handle.join().map_err(|_| "proposal worker panicked")??;
        }
        Ok::<(), String>(())
    })?;
    Ok(scores.into_iter().map(AtomicU64::into_inner).collect())
}

#[allow(clippy::too_many_arguments)]
fn refine_active(
    tokenizer: &Tokenizer,
    mut active: Vec<u8>,
    training: &Path,
    options: &RefinementOptions,
    token_ceiling: Option<u64>,
    documents: Option<&crate::document_support::Policy>,
    mut on_round: impl FnMut(&RefinementRound),
) -> Result<Refinement, String> {
    if options.workers == 0
        || options.workers > 256
        || options.minimum_uses == 0
        || options.max_rounds == 0
        || active[..BASE_SYMBOLS].contains(&0)
    {
        return Err("invalid refinement options or missing mandatory basis".into());
    }
    let target = active.iter().filter(|&&x| x != 0).count();
    let mut remaining = target;
    let mut banned = vec![false; active.len()];
    let count = |active: &[u8]| {
        if let Some(policy) = documents {
            let found = crate::document_support::scan(
                training,
                tokenizer,
                Some(active),
                options.workers,
                policy,
            )?;
            return Ok::<_, String>((found.scan, Some(found.groups)));
        }
        scan(
            training,
            tokenizer,
            None,
            Some(active),
            options.workers,
            1,
            0,
            true,
            true,
        )
        .map(|scanned| (scanned, None))
    };
    let (mut scored, mut groups) = count(&active)?;
    let before = scored.all;
    let mut rounds = Vec::new();
    for index in 0..=options.max_rounds {
        let started = Instant::now();
        let mut removed = 0;
        loop {
            let mut changed = false;
            let mut used_removed = false;
            for id in BASE_SYMBOLS..active.len() {
                if active[id] != 0
                    && (scored.uses[id] < options.minimum_uses
                        || documents.is_some_and(|policy| {
                            groups.as_ref().unwrap()[id] < policy.minimum_groups as u32
                        }))
                {
                    active[id] = 0;
                    banned[id] = true;
                    remaining -= 1;
                    removed += 1;
                    changed = true;
                    used_removed |= scored.uses[id] != 0;
                }
            }
            if !changed || !used_removed {
                break;
            }
            // Low-use removals can expose other tokens or change boundaries.
            (scored, groups) = count(&active)?;
        }
        if remaining == target {
            if token_ceiling
                .or(documents.is_none().then_some(before.tokens))
                .is_some_and(|limit| scored.all.tokens > limit)
            {
                return Err("refinement would increase the training token count".into());
            }
            let minimum_learned_uses = (BASE_SYMBOLS..active.len())
                .filter(|&id| active[id] != 0)
                .map(|id| scored.uses[id])
                .min();
            return Ok(Refinement {
                model: tokenizer.model.subset(&active)?,
                before,
                after: scored.all,
                rounds,
                minimum_learned_uses,
                minimum_document_groups: documents.map(|p| p.minimum_groups),
                minimum_learned_document_groups: groups.as_ref().and_then(|g| {
                    (BASE_SYMBOLS..active.len())
                        .filter(|&id| active[id] != 0)
                        .map(|id| g[id])
                        .min()
                }),
            });
        }
        if index == options.max_rounds {
            return Err(format!(
                "refinement round limit reached with {remaining}/{target} usable tokens"
            ));
        }
        let scores = proposal_scores(training, tokenizer, &active, &banned, options.workers)?;
        let mut proposals: Vec<_> = scores
            .iter()
            .enumerate()
            .filter(|&(_, &gain)| gain > 0)
            .map(|(id, &gain)| (gain, id))
            .collect();
        proposals.sort_unstable_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        let added = (target - remaining).min(proposals.len());
        if added == 0 {
            if let Some(policy) = documents {
                return Err(format!("candidate pool cannot fill {target} tokens with minimum usage {} and {} training document groups; retained {remaining}",options.minimum_uses,policy.minimum_groups));
            }
            return Err(format!("candidate pool cannot fill {target} tokens at minimum usage {}; retained {remaining}", options.minimum_uses));
        }
        for &(_, id) in &proposals[..added] {
            active[id] = 1;
        }
        remaining += added;
        (scored, groups) = count(&active)?;
        let round = RefinementRound {
            index: index + 1,
            removed,
            added,
            active: remaining,
            totals: scored.all,
            seconds: started.elapsed().as_secs_f64(),
        };
        on_round(&round);
        rounds.push(round);
    }
    unreachable!()
}
