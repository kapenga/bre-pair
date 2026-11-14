//! EOF-aligned parallel experiment. One central registry, immutable grammar
//! during workers, and a global selection barrier. No worker count maps.
use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use crate::batch::{self, Candidate, Counting, Edge, Position};
use crate::engine::{Compressed, Config, Round, Training};
use crate::grammar::{Grammar, EOF};
use crate::packed::{symbol_width, PackedSymbols};
use crate::pair_map::PairMap;
use crate::registry::{Registry, Writer};
use crate::scoring::{Model, Statistics};
use crate::symbol_counts::SymbolCounts;

#[derive(Clone, Copy, Debug, Default)]
pub struct Options {
    pub max_rounds: Option<usize>,
    /// Hard learned-rule budget. The final batch may contain fewer pairs.
    pub max_rules: Option<usize>,
    /// Stop at a batch barrier once all eligible pair counts are <= this value.
    /// The ordinary merge floor is unchanged while any larger pair remains.
    pub stop_max_pair_count: Option<u64>,
    /// Experimental: exact endpoint-frequency filtering, disabled by default.
    pub frequency_cutoff: bool,
    /// Optional random expansion pulses at completed batch barriers.
    pub shake: Option<crate::shake::Options>,
    /// Fixed-input merge-only mode: retire impossible pairs and release slack.
    /// Requires exact dense marginals (`frequency_cutoff`); no random expansion.
    pub memory_saving: bool,
    /// Approximate document support fused into counting/rewrite; one shared ID.
    pub last_seen_documents: bool,
}

pub struct ParallelTraining {
    pub training: Training,
    pub registry_slot_bytes: usize,
    pub registry_peak_slot_bytes: usize,
    pub registry_growths: usize,
    pub registry_retired_pairs: usize,
    pub registry_compactions: usize,
    pub symbol_statistics_bytes: usize,
    pub initial_count_seconds: f64,
    pub unchanged_chunks: usize,
    pub chunks: usize,
    pub stopped_at_round_limit: bool,
    pub stopped_at_rule_limit: bool,
    pub stopped_at_pair_count: bool,
    pub maximum_pair_count_at_stop: Option<u64>,
    pub document_filtered_candidates: u64,
    pub document_support_seconds: f64,
    /// Seeded runs only: selectable entries, excluding construction helpers.
    pub(crate) candidate_membership: Option<Vec<bool>>,
    pub(crate) blocked_candidates: u64,
    pub(crate) block_lifted_early: bool,
    pub shakes: Vec<crate::shake::Event>,
}

pub(crate) struct MergeBlock {
    entries: HashSet<Vec<u32>>,
    lengths: Vec<bool>,
    rounds: usize,
}

impl MergeBlock {
    pub(crate) fn new(entries: Vec<Vec<u32>>, rounds: usize) -> Self {
        let mut lengths = vec![false; entries.iter().map(Vec::len).max().unwrap_or(0) + 1];
        for entry in &entries {
            lengths[entry.len()] = true;
        }
        Self {
            entries: entries.into_iter().collect(),
            lengths,
            rounds,
        }
    }

    fn excludes(&self, grammar: &Grammar, pair: crate::grammar::Pair) -> bool {
        let length = grammar.terminals(pair.0) + grammar.terminals(pair.1);
        if !self.lengths.get(length).copied().unwrap_or(false) {
            return false;
        }
        let mut labels = Vec::with_capacity(length);
        grammar.visit_terminals(pair.0, |s| labels.push(s));
        grammar.visit_terminals(pair.1, |s| labels.push(s));
        self.entries.contains(&labels)
    }
}

enum Input<'a> {
    Bytes(&'a [Vec<u8>]),
    Packed(Compressed),
    Utf8Packed(Compressed, f64),
    LanguagePacked(Compressed, &'a crate::language_prior::Policy),
    Segments(Compressed),
    Seeded(Compressed, Vec<bool>, Option<MergeBlock>),
    Documented(Compressed, Vec<bool>, &'a crate::document_support::Policy),
}

/// Research control: each packed block is an immutable hard merge segment.
/// Only natural field endings carry EOF; internal segments carry no marker.
/// Use complete-registry marginals: the dense fast path assumes EOF-aligned
/// chunks and would give incorrect outgoing marginals for internal segments.
pub fn compress_segments(
    source: Compressed,
    config: &Config,
    workers: usize,
    options: Options,
    on_round: impl FnMut(&Round),
) -> Result<ParallelTraining, String> {
    if options.frequency_cutoff
        || options.memory_saving
        || options.last_seen_documents
        || options.shake.is_some()
    {
        return Err(
            "segmented research input requires complete registry counts and merge-only training"
                .into(),
        );
    }
    compress_inner::<false>(
        Input::Segments(source),
        config,
        workers,
        64,
        1,
        options,
        on_round,
    )
}

/// Stream raw lines directly into the initial packed representation. No raw
/// corpus copy or per-line allocation is retained. EOF remains mergeable.
pub fn compress_file(
    path: &std::path::Path,
    config: &Config,
    workers: usize,
    chunk_symbols: usize,
    options: Options,
    on_round: impl FnMut(&Round),
) -> Result<ParallelTraining, String> {
    compress_file_with_prior(
        path,
        config,
        workers,
        chunk_symbols,
        options,
        true,
        None,
        None,
        on_round,
    )
}

/// Stream one complete byte file as a single field, retaining embedded LF bytes.
pub fn compress_file_single_field(
    path: &std::path::Path,
    config: &Config,
    workers: usize,
    chunk_symbols: usize,
    options: Options,
    on_round: impl FnMut(&Round),
) -> Result<ParallelTraining, String> {
    compress_file_with_prior(
        path,
        config,
        workers,
        chunk_symbols,
        options,
        false,
        None,
        None,
        on_round,
    )
}

/// Continue an already prepared EOF-aligned stream. Round limits apply to
/// additional rounds, not to any batches used to construct the input seed.
pub fn compress_prepacked(
    source: Compressed,
    config: &Config,
    workers: usize,
    chunk_symbols: usize,
    options: Options,
    on_round: impl FnMut(&Round),
) -> Result<ParallelTraining, String> {
    if source.blocks.iter().any(|b| {
        b.is_empty()
            || b.get(b.len() - 1) as usize >= source.grammar.alphabet_size()
            || !source.grammar.ends_field(b.get(b.len() - 1))
    }) {
        return Err("prepacked input chunks must end at a valid EOF symbol".into());
    }
    if options.last_seen_documents || options.shake.is_some() {
        return Err(
            "prepacked initialization requires merge-only input without document gating".into(),
        );
    }
    if options.frequency_cutoff {
        compress_inner::<true>(
            Input::Packed(source),
            config,
            workers,
            64,
            chunk_symbols,
            options,
            on_round,
        )
    } else {
        compress_inner::<false>(
            Input::Packed(source),
            config,
            workers,
            64,
            chunk_symbols,
            options,
            on_round,
        )
    }
}

/// Research-only UTF-8 prior. Validates the source during packing and adds
/// `count * bonus` only to joins inside code points. Zero preserves priorities.
pub fn compress_file_utf8_prior(
    path: &std::path::Path,
    config: &Config,
    workers: usize,
    chunk_symbols: usize,
    options: Options,
    bonus: f64,
    on_round: impl FnMut(&Round),
) -> Result<ParallelTraining, String> {
    if !bonus.is_finite()
        || bonus < 0.0
        || config.score != Model::WeightedPmi
        || config.pmi_alpha != 1.0
    {
        return Err(
            "UTF-8 prior requires nonnegative finite bonus and weighted PMI alpha 1".into(),
        );
    }
    compress_file_with_prior(
        path,
        config,
        workers,
        chunk_symbols,
        options,
        true,
        Some(bonus),
        None,
        on_round,
    )
}

/// Research-only weighted discovery. Counts/eligibility and the shared
/// registry remain raw; a separate bounded pass supplies weighted priorities.
pub fn compress_file_language_weighted(
    path: &std::path::Path,
    policy: &crate::language_prior::Policy,
    config: &Config,
    workers: usize,
    chunk_symbols: usize,
    options: Options,
    on_round: impl FnMut(&Round),
) -> Result<ParallelTraining, String> {
    if config.score != Model::WeightedPmi || options.shake.is_some() {
        return Err("language-weighted discovery requires merge-only weighted PMI".into());
    }
    compress_file_with_prior(
        path,
        config,
        workers,
        chunk_symbols,
        options,
        true,
        None,
        Some(policy),
        on_round,
    )
}

#[allow(clippy::too_many_arguments)]
fn compress_file_with_prior(
    path: &std::path::Path,
    config: &Config,
    workers: usize,
    chunk_symbols: usize,
    options: Options,
    split_lines: bool,
    utf8_bonus: Option<f64>,
    language_prior: Option<&crate::language_prior::Policy>,
    on_round: impl FnMut(&Round),
) -> Result<ParallelTraining, String> {
    use std::io::BufRead;
    if chunk_symbols == 0 {
        return Err("chunk size must be positive".into());
    }
    let mut input = std::io::BufReader::with_capacity(
        65536,
        std::fs::File::open(path).map_err(|e| e.to_string())?,
    );
    let input_total = input.get_ref().metadata().map_err(|e| e.to_string())?.len();
    let progress = crate::training_progress::enabled();
    let mut next_progress = 64 * 1024 * 1024u64;
    crate::training_progress::phase("packing", 0, 0, input_total, "bytes");
    let mut blocks = Vec::new();
    let mut language_chunks = Vec::new();
    let (mut language_start, mut language_first_field) = (0, 0);
    // A single field cannot flush at line boundaries. Its final symbol count
    // is known, so avoid geometric growth and a second full allocation during
    // shrink_to_fit at EOF. Line-delimited ingestion keeps bounded chunks.
    let initial_symbols = if split_lines {
        chunk_symbols
    } else {
        usize::try_from(input_total)
            .ok()
            .and_then(|n| n.checked_add(1))
            .ok_or("single-field symbol count exceeds addressable memory")?
    };
    let mut block = PackedSymbols::with_capacity(9, initial_symbols);
    let mut input_bytes = 0u64;
    let mut field_count = 0u64;
    let mut open = false;
    let mut allocated = 0usize;
    let mut validator = utf8_bonus.map(|_| crate::utf8_prior::Validator::default());
    loop {
        let bytes = input.fill_buf().map_err(|e| e.to_string())?;
        if bytes.is_empty() {
            break;
        }
        if let Some(validator) = &mut validator {
            validator.push(bytes)?;
        }
        for &byte in bytes {
            block.push(byte as u32);
            input_bytes += 1;
            open = true;
            if split_lines && byte == b'\n' {
                block.push(EOF);
                field_count += 1;
                open = false;
                if block.len() >= chunk_symbols {
                    if language_prior.is_some() {
                        language_chunks.push(crate::language_prior::Chunk {
                            start: language_start,
                            end: input_bytes,
                            first_field: language_first_field,
                            fields: field_count - language_first_field,
                        });
                        language_start = input_bytes;
                        language_first_field = field_count;
                    }
                    block.shrink_to_fit();
                    allocated = allocated
                        .checked_add(block.allocated_bytes())
                        .ok_or("input allocation overflow")?;
                    if allocated > config.buffer_bytes {
                        return Err("packed input exceeds buffer budget".into());
                    }
                    blocks.push(std::mem::replace(
                        &mut block,
                        PackedSymbols::with_capacity(9, chunk_symbols),
                    ));
                }
            }
            if allocated.saturating_add(block.allocated_bytes()) > config.buffer_bytes {
                return Err("packed input exceeds buffer budget".into());
            }
        }
        let length = bytes.len();
        input.consume(length);
        if progress && input_bytes >= next_progress {
            crate::training_progress::phase("packing", 0, input_bytes, input_total, "bytes");
            next_progress = input_bytes.saturating_add(64 * 1024 * 1024);
        }
    }
    if let Some(validator) = validator {
        validator.finish()?;
    }
    if open {
        block.push(EOF);
        field_count += 1;
    }
    if !block.is_empty() {
        if language_prior.is_some() {
            language_chunks.push(crate::language_prior::Chunk {
                start: language_start,
                end: input_bytes,
                first_field: language_first_field,
                fields: field_count - language_first_field,
            });
        }
        block.shrink_to_fit();
        blocks.push(block);
    } else {
        drop(block);
    }
    drop(input);
    let source = Compressed {
        grammar: Grammar::default(),
        blocks,
        input_bytes,
        field_count,
    };
    crate::training_progress::phase("packing", 0, input_bytes, input_total, "bytes");
    crate::training_progress::seed(
        0,
        source.blocks.iter().map(|b| b.len() as u64).sum(),
        source.blocks.iter().map(PackedSymbols::storage_bytes).sum(),
        9,
    );
    let source = match utf8_bonus {
        Some(bonus) => Input::Utf8Packed(source, bonus),
        None => {
            if let Some(policy) = language_prior {
                policy.validate_totals(input_bytes, field_count)?;
                policy.set_chunks(language_chunks)?;
                Input::LanguagePacked(source, policy)
            } else {
                Input::Packed(source)
            }
        }
    };
    if options.frequency_cutoff {
        compress_inner::<true>(
            source,
            config,
            workers,
            64,
            chunk_symbols,
            options,
            on_round,
        )
    } else {
        compress_inner::<false>(
            source,
            config,
            workers,
            64,
            chunk_symbols,
            options,
            on_round,
        )
    }
}

/// Internal entry: the seeded module validates and packs complete fields.
/// Here max_rules limits selectable entries including the mandatory basis,
/// rather than helper rules used to represent flat starting tokens.
pub(crate) fn compress_seeded(
    source: Compressed,
    membership: Vec<bool>,
    config: &Config,
    workers: usize,
    options: Options,
    blocked: Option<MergeBlock>,
    on_round: impl FnMut(&Round),
) -> Result<ParallelTraining, String> {
    if options.frequency_cutoff {
        compress_inner::<true>(
            Input::Seeded(source, membership, blocked),
            config,
            workers,
            64,
            1,
            options,
            on_round,
        )
    } else {
        compress_inner::<false>(
            Input::Seeded(source, membership, blocked),
            config,
            workers,
            64,
            1,
            options,
            on_round,
        )
    }
}

pub(crate) fn compress_seeded_documents(
    source: Compressed,
    membership: Vec<bool>,
    config: &Config,
    workers: usize,
    options: Options,
    documents: &crate::document_support::Policy,
    on_round: impl FnMut(&Round),
) -> Result<ParallelTraining, String> {
    if !options.frequency_cutoff {
        return Err("document-gated training requires exact dense statistics".into());
    }
    compress_inner::<true>(
        Input::Documented(source, membership, documents),
        config,
        workers,
        64,
        1,
        options,
        on_round,
    )
}

/// Const specialization keeps experimental bookkeeping out of the default
/// worker loops. The unfiltered state has no dense allocations or updates.
struct CountingState<const FILTER: bool> {
    symbols: SymbolCounts,
    track: bool,
}

impl<const FILTER: bool> CountingState<FILTER> {
    fn new(minimum: u64, track: bool) -> Self {
        Self {
            symbols: if FILTER {
                SymbolCounts::new(minimum, track)
            } else {
                SymbolCounts::disabled()
            },
            track,
        }
    }

    #[inline]
    fn contribution(&self, writer: &mut Writer<'_>, edge: Edge, delta: i64) -> Result<(), String> {
        match edge {
            Edge::None => Ok(()),
            Edge::Pair(pair) => {
                let allowed = !FILTER || self.symbols.allows(pair);
                #[cfg(feature = "profile")]
                if FILTER {
                    writer.record_endpoint_check(allowed);
                }
                if allowed {
                    writer.pair(pair, delta)
                } else {
                    Ok(())
                }
            }
            Edge::Overlap(symbol) => {
                if FILTER {
                    #[cfg(feature = "profile")]
                    writer.record_dense_overlap();
                    self.symbols.change_overlap(symbol, delta)
                } else {
                    writer.overlap(symbol, delta)
                }
            }
        }
    }

    #[inline]
    fn documented(
        &self,
        writer: &mut Writer<'_>,
        edge: Edge,
        delta: i64,
        document: u64,
    ) -> Result<(), String> {
        if let Edge::Pair(pair) = edge {
            let allowed = !FILTER || self.symbols.allows(pair);
            #[cfg(feature = "profile")]
            writer.record_endpoint_check(allowed);
            if allowed {
                writer.pair_document(pair, delta, document)?;
            }
            Ok(())
        } else {
            self.contribution(writer, edge, delta)
        }
    }
}

/// One scoped task queue. Each worker has one bounded update log and produces
/// only its assigned output chunks. Results retain original chunk order.
fn parallel<T: Send>(
    count: usize,
    workers: usize,
    registry: &Registry,
    #[cfg(feature = "profile")] worker_name: &'static str,
    work: impl Fn(usize, &mut Writer<'_>) -> Result<T, String> + Sync,
) -> Result<Vec<T>, String> {
    let next = AtomicUsize::new(0);
    let tasks = std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for _ in 0..workers.min(count.max(1)) {
            let work = &work;
            let next = &next;
            handles.push(scope.spawn(move || {
                #[cfg(feature = "profile")]
                let _worker = crate::profile::Span::new(worker_name);
                let mut writer = registry.writer();
                let mut results = Vec::new();
                loop {
                    let index = next.fetch_add(1, Ordering::Relaxed);
                    if index >= count {
                        break;
                    }
                    results.push((index, work(index, &mut writer)?));
                }
                writer.finish()?;
                Ok::<_, String>(results)
            }));
        }
        let mut results = Vec::with_capacity(count);
        for handle in handles {
            results.extend(handle.join().map_err(|_| "field worker panicked")??);
        }
        Ok::<_, String>(results)
    })?;
    let mut tasks = tasks;
    tasks.sort_unstable_by_key(|(index, _)| *index);
    Ok(tasks.into_iter().map(|(_, output)| output).collect())
}

/// Transfer chunk ownership to workers. Unchanged-width compaction reuses the
/// allocation; width growth retains one old/new chunk per active worker only.
fn parallel_rewrite(
    blocks: Vec<PackedSymbols>,
    workers: usize,
    registry: &Registry,
    work: impl Fn(usize, &mut PackedSymbols, &mut Writer<'_>) -> Result<bool, String> + Sync,
) -> Result<(Vec<PackedSymbols>, usize), String> {
    let count = blocks.len();
    let jobs = Mutex::new(blocks.into_iter().enumerate());
    let (mut results, unchanged) = std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for _ in 0..workers.min(count.max(1)) {
            let jobs = &jobs;
            let work = &work;
            handles.push(scope.spawn(move || {
                #[cfg(feature = "profile")]
                let _worker = crate::profile::Span::new("worker.rewrite");
                let mut writer = registry.writer();
                let mut results = Vec::new();
                let mut unchanged = 0;
                loop {
                    let job = jobs.lock().map_err(|_| "poisoned chunk queue")?.next();
                    let Some((index, mut block)) = job else { break };
                    if !work(index, &mut block, &mut writer)? {
                        unchanged += 1;
                    }
                    results.push((index, block));
                }
                writer.finish()?;
                Ok::<_, String>((results, unchanged))
            }));
        }
        let mut results = Vec::with_capacity(count);
        let mut unchanged = 0;
        for handle in handles {
            let (output, kept) = handle.join().map_err(|_| "field worker panicked")??;
            results.extend(output);
            unchanged += kept;
        }
        Ok::<_, String>((results, unchanged))
    })?;
    results.sort_unstable_by_key(|(index, _)| *index);
    Ok((
        results.into_iter().map(|(_, block)| block).collect(),
        unchanged,
    ))
}

#[inline]
fn read_original(input: &PackedSymbols, next: &mut usize, end: usize) -> Option<u32> {
    if *next == end {
        return None;
    }
    let symbol = input.get(*next);
    *next += 1;
    Some(symbol)
}

fn contribution(writer: &mut Writer<'_>, edge: Edge, delta: i64) -> Result<(), String> {
    match edge {
        Edge::None => Ok(()),
        Edge::Pair(pair) => writer.pair(pair, delta),
        Edge::Overlap(symbol) => writer.overlap(symbol, delta),
    }
}

/// Preserve the measured baseline kernel independently of experimental state.
/// Both kernels are checked against the independent sequential trainer.
fn rewrite_unfiltered(
    input: &mut PackedSymbols,
    replacements: &PairMap<u32>,
    grammar: &Grammar,
    width: u8,
    incremental: bool,
    track: bool,
    writer: &mut Writer<'_>,
) -> Result<bool, String> {
    let mut output = if width == input.width() {
        None
    } else {
        Some(PackedSymbols::with_capacity(width, input.len()))
    };
    let mut old = Position::default();
    let mut new = Position::default();
    let end = input.len();
    let mut read = 0;
    let mut pending = read_original(input, &mut read, end);
    let mut in_place = None;
    let mut consumed = 0;
    while let Some(symbol) = pending {
        let next = read_original(input, &mut read, end);
        let replacement = next.and_then(|b| replacements.get(&(symbol, b)).copied());
        let start = consumed;
        let mut first = if incremental {
            old.advance(symbol, grammar, track)
        } else {
            Edge::None
        };
        let mut second = Edge::None;
        let emitted = if let Some(replacement) = replacement {
            if incremental {
                second = old.advance(next.unwrap(), grammar, track);
            }
            consumed += 2;
            pending = read_original(input, &mut read, end);
            if output.is_none() && in_place.is_none() {
                in_place = Some(input.rewrite_from(start));
            }
            replacement
        } else {
            consumed += 1;
            pending = next;
            symbol
        };
        let mut added = new.advance(emitted, grammar, track);
        if incremental {
            if added == first {
                first = Edge::None;
                added = Edge::None;
            } else if added == second {
                second = Edge::None;
                added = Edge::None;
            }
            contribution(writer, first, -1)?;
            contribution(writer, second, -1)?;
        }
        contribution(writer, added, 1)?;
        if let Some(output) = &mut output {
            output.push(emitted);
        } else if let Some(cursor) = &mut in_place {
            input.push_rewritten(cursor, emitted);
        }
    }
    if let Some(output) = output {
        *input = output;
        Ok(true)
    } else if let Some(cursor) = in_place {
        input.finish_rewrite(cursor);
        Ok(true)
    } else {
        Ok(false)
    }
}

fn rewrite_filtered<const FILTER: bool, const DOCUMENTS: bool>(
    input: &mut PackedSymbols,
    replacements: &PairMap<u32>,
    grammar: &Grammar,
    counts: &CountingState<FILTER>,
    incremental: bool,
    writer: &mut Writer<'_>,
    mut document: Option<crate::document_support::last_seen::Cursor<'_>>,
) -> Result<bool, String> {
    let width = symbol_width(grammar.alphabet_size());
    let track = counts.track;
    // At unchanged widths, overwrite complete output words behind the reader.
    // Width growth uses one replacement allocation, swapped into this chunk.
    let mut output = if width == input.width() {
        None
    } else {
        Some(PackedSymbols::with_capacity(width, input.len()))
    };
    let mut old = Position::default();
    let mut new = Position::default();
    let end = input.len();
    let mut read = 0;
    let mut pending = read_original(input, &mut read, end);
    let mut in_place = None;
    let mut consumed = 0;
    while let Some(symbol) = pending {
        let next = read_original(input, &mut read, end);
        let replacement = next.and_then(|b| replacements.get(&(symbol, b)).copied());
        let start = consumed;
        let mut first = if incremental {
            old.advance(symbol, grammar, track)
        } else {
            Edge::None
        };
        let mut second = Edge::None;
        let emitted = if let Some(replacement) = replacement {
            if incremental {
                second = old.advance(next.unwrap(), grammar, track);
            }
            consumed += 2;
            pending = read_original(input, &mut read, end);
            if output.is_none() && in_place.is_none() {
                in_place = Some(input.rewrite_from(start));
            }
            replacement
        } else {
            consumed += 1;
            pending = next;
            symbol
        };
        let mut added = {
            let (edge, first_in_field) = new.advance_with_start(emitted, grammar, track);
            if first_in_field && emitted != symbol {
                #[cfg(feature = "profile")]
                writer.record_start_change();
                counts.symbols.replace_start(symbol, emitted)?;
            }
            edge
        };
        let observed = added;
        if incremental {
            if added == first {
                first = Edge::None;
                added = Edge::None;
            } else if added == second {
                second = Edge::None;
                added = Edge::None;
            }
            counts.contribution(writer, first, -1)?;
            counts.contribution(writer, second, -1)?;
        }
        if DOCUMENTS {
            let document = document.as_mut().expect("document cursor");
            if matches!(observed, Edge::Pair(_)) {
                counts.documented(
                    writer,
                    observed,
                    i64::from(added == observed),
                    document.document(),
                )?;
            } else {
                counts.contribution(writer, added, 1)?;
            }
            document.advance(grammar.ends_field(emitted))?;
        } else {
            counts.contribution(writer, added, 1)?;
        }
        if let Some(output) = &mut output {
            output.push(emitted);
        } else if let Some(cursor) = &mut in_place {
            input.push_rewritten(cursor, emitted);
        }
    }
    if let Some(output) = output {
        *input = output;
        Ok(true)
    } else if let Some(cursor) = in_place {
        input.finish_rewrite(cursor);
        Ok(true)
    } else {
        Ok(false)
    }
}

pub fn compress(
    fields: &[Vec<u8>],
    config: &Config,
    workers: usize,
    shard_count: usize,
    chunk_symbols: usize,
    on_round: impl FnMut(&Round),
) -> Result<ParallelTraining, String> {
    compress_with_limit(
        fields,
        config,
        workers,
        shard_count,
        chunk_symbols,
        None,
        on_round,
    )
}

/// Stop only at a completed batch boundary. The returned grammar/stream remain
/// decodable; reaching the limit does not certify that training has converged.
pub fn compress_with_limit(
    fields: &[Vec<u8>],
    config: &Config,
    workers: usize,
    shard_count: usize,
    chunk_symbols: usize,
    max_rounds: Option<usize>,
    on_round: impl FnMut(&Round),
) -> Result<ParallelTraining, String> {
    compress_with_options(
        fields,
        config,
        workers,
        shard_count,
        chunk_symbols,
        Options {
            max_rounds,
            ..Options::default()
        },
        on_round,
    )
}

/// Select the implementation once, outside every worker's hot loops.
pub fn compress_with_options(
    fields: &[Vec<u8>],
    config: &Config,
    workers: usize,
    shard_count: usize,
    chunk_symbols: usize,
    options: Options,
    on_round: impl FnMut(&Round),
) -> Result<ParallelTraining, String> {
    if options.frequency_cutoff {
        compress_inner::<true>(
            Input::Bytes(fields),
            config,
            workers,
            shard_count,
            chunk_symbols,
            options,
            on_round,
        )
    } else {
        compress_inner::<false>(
            Input::Bytes(fields),
            config,
            workers,
            shard_count,
            chunk_symbols,
            options,
            on_round,
        )
    }
}

fn compress_inner<const FILTER: bool>(
    input: Input<'_>,
    config: &Config,
    workers: usize,
    shard_count: usize,
    chunk_symbols: usize,
    limits: Options,
    mut on_round: impl FnMut(&Round),
) -> Result<ParallelTraining, String> {
    if workers == 0
        || workers > 256
        || chunk_symbols == 0
        || limits.max_rounds == Some(0)
        || limits.max_rules == Some(0)
    {
        return Err(
            "workers must be 1..=256; chunk symbols and round limits must be positive".into(),
        );
    }
    if config.minimum_count < 2
        || limits
            .stop_max_pair_count
            .is_some_and(|n| n < config.minimum_count)
        || !config.pmi_alpha.is_finite()
        || !(0.0..=4.0).contains(&config.pmi_alpha)
        || (config.score != Model::WeightedPmi && config.pmi_alpha != 1.0)
        || config.pruning.is_some()
    {
        return Err(
            "invalid field experiment configuration (checkpoint pruning is unsupported)".into(),
        );
    }
    if limits.memory_saving && (!FILTER || limits.shake.is_some()) {
        return Err("memory-saving training requires dense statistics and merge-only input".into());
    }
    if let Some(shake) = &limits.shake {
        shake.validate()?;
        if FILTER {
            return Err("random shakes require the unfiltered central registry".into());
        }
    }
    let started = Instant::now();
    #[cfg(feature = "profile")]
    let pack_profile = crate::profile::Span::new("training.initial_pack");
    let track = config.score != Model::Frequency;
    let mut counts = CountingState::<FILTER>::new(config.minimum_count, track);
    let utf8_bonus = match &input {
        Input::Utf8Packed(_, bonus) => *bonus,
        _ => 0.0,
    };
    let mut utf8_prior = (utf8_bonus > 0.0).then(crate::utf8_prior::JoinPrior::new);
    let language_prior = match &input {
        Input::LanguagePacked(_, policy) => Some(*policy),
        _ => None,
    };
    let prepacked = matches!(
        &input,
        Input::Packed(_)
            | Input::Utf8Packed(..)
            | Input::LanguagePacked(..)
            | Input::Segments(_)
            | Input::Seeded(..)
            | Input::Documented(..)
    );
    #[cfg(feature = "audit")]
    let segmented = matches!(&input, Input::Segments(_));
    let (mut grammar, mut blocks, input_bytes, field_count, mut membership, mut blocked, documents) =
        match input {
            Input::Packed(source)
            | Input::Utf8Packed(source, _)
            | Input::LanguagePacked(source, _)
            | Input::Segments(source) => (
                source.grammar,
                source.blocks,
                source.input_bytes,
                source.field_count,
                None,
                None,
                None,
            ),
            Input::Seeded(source, members, blocked) => (
                source.grammar,
                source.blocks,
                source.input_bytes,
                source.field_count,
                Some(members),
                blocked,
                None,
            ),
            Input::Documented(source, members, documents) => (
                source.grammar,
                source.blocks,
                source.input_bytes,
                source.field_count,
                Some(members),
                None,
                Some(documents),
            ),
            Input::Bytes(fields) => {
                let symbols = fields
                    .iter()
                    .try_fold(0usize, |n, f| n.checked_add(f.len())?.checked_add(1))
                    .ok_or("input size overflow")?;
                if (symbols as u128 * 9).div_ceil(8) > config.buffer_bytes as u128 {
                    return Err(
                        "field experiment requires the initial input to fit the active budget"
                            .into(),
                    );
                }
                let mut blocks = Vec::new();
                let mut block = PackedSymbols::with_capacity(9, chunk_symbols.min(symbols));
                for field in fields {
                    for (index, &byte) in field.iter().enumerate() {
                        block.push(byte as u32);
                        if FILTER {
                            counts.symbols.push_initial(byte as u32, index == 0);
                        }
                    }
                    block.push(EOF);
                    if FILTER {
                        counts.symbols.push_initial(EOF, field.is_empty());
                    }
                    if block.len() >= chunk_symbols {
                        blocks.push(block);
                        block = PackedSymbols::new(9);
                    }
                }
                if !block.is_empty() {
                    blocks.push(block);
                }
                (
                    Grammar::default(),
                    blocks,
                    fields.iter().map(|f| f.len() as u64).sum(),
                    fields.len() as u64,
                    None,
                    None,
                    None,
                )
            }
        };
    if limits.last_seen_documents && (documents.is_none() || !FILTER || limits.shake.is_some()) {
        return Err(
            "last-seen support requires indexed documents, dense statistics, and no shake".into(),
        );
    }
    let support_minimum = documents.map_or(1, |p| p.minimum_groups);
    let last_seen = limits.last_seen_documents && support_minimum > 1;
    let document_index = documents
        .filter(|_| last_seen)
        .map(|p| crate::document_support::last_seen::Documents::new(p, input_bytes, field_count))
        .transpose()?;
    let mut document_gate = documents
        .filter(|_| !last_seen)
        .map(|p| crate::document_support::PairGate::new(p, input_bytes, field_count))
        .transpose()?;
    let mut document_filtered_candidates = 0u64;
    let mut document_support_seconds = 0.0;
    let mut chunk_fields = Vec::new();
    if prepacked {
        if membership.as_ref().is_some_and(|members| {
            members.len() != grammar.alphabet_size()
                || limits
                    .max_rules
                    .is_some_and(|n| members.iter().filter(|&&b| b).count() > n)
        }) || blocks
            .iter()
            .map(PackedSymbols::storage_bytes)
            .sum::<usize>()
            > config.buffer_bytes
        {
            return Err("invalid seeded candidate or packed budget".into());
        }
        if FILTER {
            let progress_statistics = crate::training_progress::enabled();
            crate::training_progress::phase(
                "symbol_statistics",
                0,
                0,
                blocks.len() as u64,
                "chunks",
            );
            counts.symbols.resize(grammar.alphabet_size());
            let mut document = document_index.as_ref().map(|index| index.cursor(0));
            for (i, block) in blocks.iter().enumerate() {
                if let Some(document) = &document {
                    chunk_fields.push(document.field);
                }
                let mut first = true;
                for symbol in block.iter() {
                    counts.symbols.push_initial(symbol, first);
                    first = grammar.ends_field(symbol);
                    if let Some(document) = &mut document {
                        document.validate_symbol(symbol, first, &grammar)?;
                    }
                }
                if progress_statistics && (i + 1) % 64 == 0 {
                    crate::training_progress::phase(
                        "symbol_statistics",
                        0,
                        (i + 1) as u64,
                        blocks.len() as u64,
                        "chunks",
                    );
                }
            }
            if let Some(document) = document {
                document.finish(input_bytes, field_count)?;
            }
        }
    }
    let chunks = blocks.len();
    if FILTER {
        counts.symbols.refresh_eligibility();
    }
    let mut registry = Registry::with_documents(
        shard_count,
        if last_seen { support_minimum as u8 } else { 0 },
    )?;
    #[cfg(feature = "profile")]
    drop(pack_profile);
    let count_started = Instant::now();
    let progress_count = crate::training_progress::enabled();
    let counted_chunks = AtomicUsize::new(0);
    crate::training_progress::phase("initial_pair_count", 0, 0, blocks.len() as u64, "chunks");
    parallel(
        blocks.len(),
        workers,
        &registry,
        #[cfg(feature = "profile")]
        "worker.initial_count",
        |i, writer| {
            let mut position = Position::default();
            if let Some(index) = &document_index {
                let mut document = index.cursor(chunk_fields[i]);
                for symbol in blocks[i].iter() {
                    counts.documented(
                        writer,
                        position.advance(symbol, &grammar, track),
                        1,
                        document.document(),
                    )?;
                    document.advance(grammar.ends_field(symbol))?;
                }
            } else {
                for symbol in blocks[i].iter() {
                    counts.contribution(writer, position.advance(symbol, &grammar, track), 1)?;
                }
            }
            if progress_count {
                let done = counted_chunks.fetch_add(1, Ordering::Relaxed) + 1;
                if done % 64 == 0 {
                    crate::training_progress::phase(
                        "initial_pair_count",
                        0,
                        done as u64,
                        blocks.len() as u64,
                        "chunks",
                    );
                }
            }
            Ok(())
        },
    )?;
    let initial_count_seconds = count_started.elapsed().as_secs_f64();
    crate::training_progress::phase(
        "initial_pair_count",
        0,
        blocks.len() as u64,
        blocks.len() as u64,
        "chunks",
    );
    let mut rounds = Vec::new();
    let mut unchanged_chunks = 0;
    let mut stopped_at_round_limit = false;
    let mut stopped_at_rule_limit = false;
    let mut stopped_at_pair_count = false;
    let mut maximum_pair_count_at_stop = None;
    let mut blocked_candidates = 0u64;
    let mut block_lifted_early = false;
    let mut shakes = Vec::new();
    let mut max_buffer_bytes = blocks
        .iter()
        .map(PackedSymbols::storage_bytes)
        .sum::<usize>();
    loop {
        crate::training_progress::phase(
            "candidate_collection",
            rounds.len() + 1,
            0,
            0,
            "candidates",
        );
        let round_started = Instant::now();
        let statistics_started = Instant::now();
        let candidate_capacity = if limits.memory_saving {
            registry.compact_frequent(config.minimum_count, grammar.alphabet_size())?
        } else {
            0
        };
        #[cfg(feature = "profile")]
        let statistics_profile = crate::profile::Span::new("selection.statistics_and_collect");
        // Compaction already counted an upper bound on eligible pairs. Avoid
        // geometric growth, spare capacity, and old/new candidate-array overlap.
        let mut candidates = Vec::with_capacity(candidate_capacity);
        if let Some(index) = &document_index {
            index.validate_file()?;
        }
        let mut collect = |pair: crate::grammar::Pair, count, support: u8| {
            if pair.0 != EOF && count >= config.minimum_count {
                if last_seen && usize::from(support) < support_minimum {
                    document_filtered_candidates += 1;
                    return;
                }
                if blocked.as_ref().is_some_and(|b| b.excludes(&grammar, pair)) {
                    blocked_candidates += 1;
                    return;
                }
                candidates.push(Candidate {
                    pair,
                    count,
                    priority: count as f64,
                });
            }
        };
        if FILTER {
            registry.retain_and_visit_documented(|pair| counts.symbols.allows(pair), &mut collect);
        }
        let frozen = registry.freeze();
        #[cfg(feature = "audit")]
        (if segmented {
            crate::audit::check_segments
        } else {
            crate::audit::check
        })(
            &blocks,
            &grammar,
            &frozen,
            FILTER.then_some(&counts.symbols),
            config.minimum_count,
            track,
        )
        .map_err(|error| format!("count audit before batch {}: {error}", rounds.len() + 1))?;
        let stats = if track {
            Some(if FILTER {
                Statistics::from_symbols(&frozen, &counts.symbols, &grammar)
            } else {
                Statistics::from_registry_with_visit(
                    &frozen,
                    grammar.alphabet_size(),
                    |pair, count| collect(pair, count, 0),
                )
            })
        } else {
            if !FILTER {
                frozen.visit(|pair, count| collect(pair, count, 0));
            }
            None
        };
        #[cfg(feature = "profile")]
        drop(statistics_profile);
        let score_seconds = statistics_started.elapsed().as_secs_f64();
        let selection_started = Instant::now();
        if let Some(gate) = &mut document_gate {
            let document_started = Instant::now();
            #[cfg(feature = "profile")]
            let _document_profile = crate::profile::Span::new("selection.document_support");
            let before = candidates.len();
            gate.filter(&blocks, &grammar, &mut candidates, rounds.len() + 1)?;
            document_filtered_candidates = document_filtered_candidates
                .checked_add((before - candidates.len()) as u64)
                .ok_or("document filter total overflow")?;
            document_support_seconds += document_started.elapsed().as_secs_f64();
        }
        if let Some(threshold) = limits.stop_max_pair_count {
            // Read the already collected exact counts, before PMI ordering or
            // conflicts can hide a frequent pair. No extra corpus/count pass.
            let maximum = candidates.iter().map(|c| c.count).max().unwrap_or(0);
            crate::training_progress::frontier(
                rounds.len() + 1,
                candidates.len(),
                maximum,
                threshold,
            );
            if maximum <= threshold {
                stopped_at_pair_count = true;
                maximum_pair_count_at_stop = Some(maximum);
                break;
            }
        }
        #[cfg(feature = "profile")]
        let candidates_profile = crate::profile::Span::new("selection.score_candidates");
        if let Some(stats) = &stats {
            // The candidate vector already exists; score it once all global
            // marginals are known, without visiting the registry a second time.
            for candidate in &mut candidates {
                candidate.priority = stats.score_with_alpha(
                    config.score,
                    candidate.pair,
                    candidate.count,
                    config.pmi_alpha,
                );
            }
        }
        if let Some(prior) = language_prior {
            prior.score(
                &mut candidates,
                stats.as_ref().expect("weighted PMI statistics"),
                &blocks,
                &grammar,
                rounds.len() + 1,
                config.pmi_alpha,
                workers,
            )?;
        }
        if let Some(prior) = &mut utf8_prior {
            prior.sync(&grammar);
            for candidate in &mut candidates {
                candidate.priority = prior.score(
                    candidate.priority,
                    candidate.pair.1,
                    candidate.count,
                    utf8_bonus,
                );
            }
        }
        #[cfg(feature = "profile")]
        drop(candidates_profile);
        // Dense marginals are no longer needed after scoring. Do not retain
        // them across candidate-index and conflict-graph construction.
        drop(stats);
        crate::training_progress::phase(
            "conflict_selection",
            rounds.len() + 1,
            0,
            candidates.len() as u64,
            "candidates",
        );
        let mut selection = batch::select_candidates_fields(
            &blocks,
            candidates,
            &config.search.for_round(rounds.len()),
            workers,
        );
        if let Some(limit) = limits.max_rules {
            let used = membership
                .as_ref()
                .map_or(grammar.rules.len(), |m| m.iter().filter(|&&b| b).count());
            let available = limit.saturating_sub(used);
            let mut new_rules = 0;
            let mut keep = 0;
            for candidate in &selection.pairs {
                let existing = grammar.find(candidate.pair);
                let adds_entry = membership.as_ref().map_or(existing.is_none(), |m| {
                    existing.is_none_or(|id| !m[id as usize])
                });
                if adds_entry {
                    if new_rules == available {
                        break;
                    }
                    new_rules += 1;
                }
                keep += 1;
            }
            if keep < selection.pairs.len() {
                // Any subset of a compatible batch remains compatible. Keep
                // its existing descending-frequency order, then stop discovery.
                selection.pairs.truncate(keep);
                selection.score = selection.pairs.iter().map(|c| c.count).sum();
                selection.greedy_score = selection.score;
                selection.proven_optimal = false;
            }
        }
        let selection_seconds = selection_started.elapsed().as_secs_f64();
        if selection.pairs.is_empty() {
            // No successful batches remain to exhaust the warm-up. Lift the
            // block and retry selection on the existing exact registry.
            if blocked.take().is_some() {
                block_lifted_early = true;
                continue;
            }
            break;
        }
        let before = blocks.iter().map(PackedSymbols::len).sum::<usize>();
        let removed = usize::try_from(selection.score).map_err(|_| "replacement size overflow")?;
        let rewrite_started = Instant::now();
        crate::training_progress::phase("rewrite", rounds.len() + 1, 0, before as u64, "symbols");
        #[cfg(feature = "profile")]
        let rules_profile = crate::profile::Span::new("rewrite.register_rules");
        let mut replacements = PairMap::default();
        let previous_rules = grammar.rules.len();
        let previous_alphabet = grammar.alphabet_size();
        let mut reactivated = false;
        let mut minimum_selected_count = u64::MAX;
        let mut maximum_selected_count = 0;
        for candidate in &selection.pairs {
            minimum_selected_count = minimum_selected_count.min(candidate.count);
            maximum_selected_count = maximum_selected_count.max(candidate.count);
            let symbol = grammar.add(candidate.pair)?;
            if let Some(members) = &mut membership {
                members.resize(grammar.alphabet_size(), false);
                members[symbol as usize] = true;
            }
            if FILTER {
                reactivated |= (symbol as usize) < previous_alphabet;
                counts.symbols.resize(grammar.alphabet_size());
                counts
                    .symbols
                    .replace(candidate.pair, symbol, candidate.count)?;
            }
            replacements.insert(candidate.pair, symbol);
        }
        let width = symbol_width(grammar.alphabet_size());
        let predicted = before.checked_sub(removed).ok_or("replacement underflow")?;
        if (predicted as u128 * width as u128).div_ceil(8) > config.buffer_bytes as u128 {
            return Err("field rewrite exceeds packed budget; streaming/flush is not implemented in this experiment".into());
        }
        // Merge-only training does not normally re-emit existing rule IDs.
        // If that changes, recount so newly eligible pairs can be discovered.
        let incremental = !reactivated
            && match config.pair_counting {
                Counting::Incremental => true,
                Counting::Adaptive => before >= 1_000_000 && removed as u128 * 8 <= before as u128,
                _ => false,
            };
        #[cfg(feature = "profile")]
        drop(rules_profile);
        if !incremental {
            #[cfg(feature = "profile")]
            let _clear = crate::profile::Span::new("rewrite.clear_registry");
            registry.clear();
            if FILTER {
                counts.symbols.clear_overlaps();
            }
        }
        #[cfg(feature = "profile")]
        let rewrite_profile = crate::profile::Span::new(if incremental {
            "rewrite.parallel_incremental"
        } else {
            "rewrite.parallel_recount"
        });
        let (rewritten_blocks, unchanged) =
            parallel_rewrite(blocks, workers, &registry, |i, block, writer| {
                if FILTER {
                    if let Some(index) = &document_index {
                        rewrite_filtered::<FILTER, true>(
                            block,
                            &replacements,
                            &grammar,
                            &counts,
                            incremental,
                            writer,
                            Some(index.cursor(chunk_fields[i])),
                        )
                    } else {
                        rewrite_filtered::<FILTER, false>(
                            block,
                            &replacements,
                            &grammar,
                            &counts,
                            incremental,
                            writer,
                            None,
                        )
                    }
                } else {
                    rewrite_unfiltered(
                        block,
                        &replacements,
                        &grammar,
                        width,
                        incremental,
                        track,
                        writer,
                    )
                }
            })?;
        #[cfg(feature = "profile")]
        drop(rewrite_profile);
        #[cfg(feature = "profile")]
        let install_profile = crate::profile::Span::new("rewrite.install_outputs");
        unchanged_chunks += unchanged;
        blocks = rewritten_blocks;
        if limits.memory_saving {
            for block in &mut blocks {
                block.shrink_to_fit();
            }
        }
        let rewritten = blocks.iter().map(PackedSymbols::len).sum::<usize>();
        if rewritten != predicted {
            return Err("shared rewrite counts disagree with output".into());
        }
        #[cfg(feature = "profile")]
        drop(install_profile);
        let rewrite_seconds = rewrite_started.elapsed().as_secs_f64();
        let packed_bytes = blocks.iter().map(PackedSymbols::storage_bytes).sum();
        max_buffer_bytes = max_buffer_bytes.max(packed_bytes);
        let round = Round {
            index: rounds.len() + 1,
            candidates: selection.candidates,
            eligible: selection.eligible,
            tie_edges: selection.tie_edges,
            selected: selection.pairs.len(),
            minimum_selected_count,
            maximum_selected_count,
            new_rules: grammar.rules.len() - previous_rules,
            before,
            rewritten,
            after_refill: rewritten,
            removed,
            refill_bytes: 0,
            total_rules: grammar.rules.len(),
            width,
            packed_bytes,
            seconds: round_started.elapsed().as_secs_f64(),
            components: selection.components,
            largest_component: selection.largest_component,
            greedy_score: selection.greedy_score,
            score_upper_bound: selection.score_upper_bound,
            proven_optimal: selection.proven_optimal,
            search_nodes: selection.search_nodes,
            search_seconds: selection.search_seconds,
            improved_components: selection.improved_components,
            score_seconds,
            selection_seconds,
            rewrite_seconds,
            refill_seconds: 0.0,
        };
        on_round(&round);
        rounds.push(round);
        if let Some(block) = &mut blocked {
            block.rounds = block.rounds.saturating_sub(1);
            if block.rounds == 0 {
                blocked = None;
            }
        }
        stopped_at_round_limit = limits.max_rounds.is_some_and(|limit| rounds.len() >= limit);
        stopped_at_rule_limit = limits.max_rules.is_some_and(|limit| {
            membership
                .as_ref()
                .map_or(grammar.rules.len(), |m| m.iter().filter(|&&b| b).count())
                >= limit
        });
        if stopped_at_round_limit || stopped_at_rule_limit {
            #[cfg(feature = "audit")]
            {
                if FILTER {
                    registry.retain_and_visit(|pair| counts.symbols.allows(pair), |_, _| {});
                }
                crate::audit::check(
                    &blocks,
                    &grammar,
                    &registry.freeze(),
                    FILTER.then_some(&counts.symbols),
                    config.minimum_count,
                    track,
                )
                .map_err(|error| format!("count audit after terminal batch: {error}"))?;
            }
            break;
        }
        if let Some(options) = &limits.shake {
            if shakes.len() < options.pulses
                && rounds.len() == options.first_batch + shakes.len() * options.interval
            {
                let started = Instant::now();
                let plan = crate::shake::plan(&blocks, &grammar, options, shakes.len())?;
                let width = symbol_width(grammar.alphabet_size());
                if (plan.after as u128 * width as u128).div_ceil(8) > config.buffer_bytes as u128 {
                    return Err("random shake exceeds the packed memory budget".into());
                }
                registry.clear();
                let (expanded, _) =
                    parallel_rewrite(blocks, workers, &registry, |_, block, writer| {
                        let mut output = PackedSymbols::with_capacity(width, block.len());
                        let mut position = Position::default();
                        for symbol in block.iter() {
                            let index = plan.indices[symbol as usize];
                            let labels = if index == u32::MAX {
                                std::slice::from_ref(&symbol)
                            } else {
                                &plan.expansions[index as usize]
                            };
                            for &label in labels {
                                output.push(label);
                                contribution(writer, position.advance(label, &grammar, track), 1)?;
                            }
                        }
                        *block = output;
                        Ok(true)
                    })?;
                blocks = expanded;
                if blocks.iter().map(PackedSymbols::len).sum::<usize>() != plan.after {
                    return Err("shake expansion length disagrees with prediction".into());
                }
                max_buffer_bytes =
                    max_buffer_bytes.max(blocks.iter().map(PackedSymbols::storage_bytes).sum());
                // Each pulse gets a short warm-up against immediate recreation.
                // Combining an initial targeted block with shaking is rejected
                // by the seeded API, so no initial exclusions are discarded.
                blocked = if plan.expansions.is_empty() {
                    None
                } else {
                    Some(MergeBlock::new(plan.expansions, options.block_rounds))
                };
                shakes.push(crate::shake::Event {
                    after_batch: rounds.len(),
                    selected_symbols: plan.symbols,
                    before: plan.before,
                    after: plan.after,
                    seconds: started.elapsed().as_secs_f64(),
                });
            }
        }
    }
    Ok(ParallelTraining {
        registry_slot_bytes: registry.slot_bytes(),
        registry_peak_slot_bytes: registry.peak_slot_bytes(),
        registry_growths: registry.growths(),
        registry_retired_pairs: registry.retired_pairs(),
        registry_compactions: registry.compactions(),
        symbol_statistics_bytes: counts.symbols.storage_bytes(),
        initial_count_seconds,
        unchanged_chunks,
        chunks,
        stopped_at_round_limit,
        stopped_at_rule_limit,
        stopped_at_pair_count,
        maximum_pair_count_at_stop,
        document_filtered_candidates,
        document_support_seconds,
        candidate_membership: membership,
        blocked_candidates,
        block_lifted_early,
        shakes,
        training: Training {
            compressed: Compressed {
                grammar,
                blocks,
                input_bytes,
                field_count,
            },
            rounds,
            refill_calls: 1,
            width_flushes: 0,
            stalled_flushes: 0,
            max_buffer_bytes,
            seconds: started.elapsed().as_secs_f64(),
            pruning: None,
        },
    })
}
