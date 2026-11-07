//! Dependency-safe reverse batches. Counts describe a literal rewrite of one
//! initial optimal parse. Optional exact refresh reparses before each subsequent
//! batch; a caller can audit the final vocabulary with a separate full scan.
//! Fixed substitutions make count redistribution equivalent to rewriting the
//! corpus, without retaining its token positions or copying a packed stream.
use std::{path::Path, time::Instant};

use crate::{
    grammar::{Symbol, BASE_SYMBOLS},
    tokenizer::{Model, ReplacementScratch, Tokenizer},
};

pub use super::coverage::{
    FrequencyWeight, Normalization as CoverageNormalization, OverlapWeight,
    Settings as CoverageSettings, Signal as CoverageSignal,
};
use super::{scan_full_active, scan_with_progress, PruningSchedule, Scan, Totals};
mod search;
pub use search::{Options as SearchOptions, Stats as SearchStats};

#[derive(Clone, Copy)]
pub struct Options {
    pub vocabulary: usize,
    pub workers: usize,
    /// Consider only this schedule's cheapest candidate prefix per round.
    pub schedule: PruningSchedule,
    /// Research control: refresh counts with a full optimal parse between rounds.
    pub exact_refresh: bool,
    /// Maximum relative coverage discount, in parts per million; zero disables.
    pub coverage_discount_ppm: u32,
    pub coverage_settings: CoverageSettings,
    /// Research only: compare equal-size dependency-safe batches by full parse.
    pub batch_search: Option<SearchOptions>,
}

pub struct Round {
    pub before: usize,
    pub after: usize,
    pub before_totals: Totals,
    pub after_totals: Totals,
    pub added_tokens: u64,
    pub deferred: usize,
    pub replacement_edges: usize,
    /// Reusable dense workspace only; excludes the model/trie and usage array.
    pub workspace_bytes: usize,
    pub replacement_reused: usize,
    pub replacement_rebuilt: usize,
    pub planning_seconds: f64,
    pub ranking_seconds: f64,
    pub conflict_seconds: f64,
    pub update_seconds: f64,
    pub coverage_seconds: f64,
    pub coverage_targets: usize,
    pub coverage_pair_checks: u64,
    pub coverage_index_bytes: usize,
    pub coverage_mean_ratio_ppm: f64,
    pub coverage_max_ratio_ppm: u128,
    pub coverage_saturated_targets: usize,
    pub coverage_zero_support_targets: usize,
    pub coverage_mean_one_overlap_share_ppm: f64,
    pub coverage_mean_internal_share_ppm: f64,
    pub batch_search: SearchStats,
    pub reparse_saved: u64,
    pub refresh_seconds: f64,
    pub seconds: f64,
}

pub struct Selection {
    pub model: Model,
    pub initial: Totals,
    pub maintained: Totals,
    /// Final compact model IDs. Exact for the maintained literal segmentation.
    pub maintained_uses: Vec<u64>,
    /// Research only: compact counts of the weighted language's occurrences.
    pub maintained_language_uses: Option<Vec<u64>>,
    pub initial_scan_seconds: f64,
    pub refresh_scans: usize,
    pub refresh_seconds: f64,
}

/// One labeled language's shares of emission and construction counts. Shared
/// tokens retain both languages' credit; there are no language-specific pools.
pub struct LanguageWeight {
    pub uses: Vec<u64>,
    pub support: Vec<u64>,
    pub numerator: u64,
    pub denominator: u64,
}

/// Research-only weighted reduction from cached exact initial parse counts.
/// Raw floors/support admission and replacement paths remain unchanged.
pub fn select_language_weighted(
    pool: Model,
    support: &[u64],
    uses: Vec<u64>,
    initial: Totals,
    language: LanguageWeight,
    options: Options,
    on_round: impl FnMut(&Round),
) -> Result<Selection, String> {
    let n = pool.tokens.len();
    validate_options(&pool, support, options)?;
    if options.exact_refresh
        || options.batch_search.is_some()
        || options.coverage_discount_ppm != 0
        || uses.len() != n
        || language.uses.len() != n
        || language.support.len() != n
        || language.numerator == 0
        || language.denominator == 0
        || language.numerator > 1_000_000_000
        || language.denominator > 1_000_000_000
        || language.uses.iter().zip(&uses).any(|(a, b)| a > b)
        || language.support.iter().zip(support).any(|(a, b)| a > b)
    {
        return Err("invalid weighted-language counts or reverse options".into());
    }
    run_weighted(
        Tokenizer::new(pool)?,
        support,
        uses,
        initial,
        options,
        Some(language),
        |_, _| Err("weighted refresh is unsupported".into()),
        on_round,
    )
}

/// Research selector. Matches the existing 0.02 construction-support prior,
/// integer score arithmetic and ID ties. No samples or intermediate raw scans.
pub fn select(
    pool: Model,
    support: &[u64],
    training: &Path,
    vocabulary: usize,
    workers: usize,
    on_round: impl FnMut(&Round),
) -> Result<Selection, String> {
    select_with_options(
        pool,
        support,
        training,
        Options {
            vocabulary,
            workers,
            schedule: PruningSchedule::default(),
            exact_refresh: false,
            coverage_discount_ppm: 0,
            coverage_settings: CoverageSettings::default(),
            batch_search: None,
        },
        on_round,
    )
}

pub fn select_with_options(
    pool: Model,
    support: &[u64],
    training: &Path,
    options: Options,
    on_round: impl FnMut(&Round),
) -> Result<Selection, String> {
    validate_options(&pool, support, options)?;
    let tokenizer = Tokenizer::new(pool)?;
    let measured = scan_with_progress(
        training,
        &tokenizer,
        None,
        None,
        options.workers,
        1,
        0,
        true,
        false,
        Some(("vocabulary_reverse_initial", 1)),
    )?;
    let seconds = measured.seconds;
    let mut scan_round = 1;
    let mut selected = run(
        tokenizer,
        support,
        measured.uses,
        measured.all,
        options,
        |tokenizer, active| {
            scan_round += 1;
            scan_full_active(training, tokenizer, active, options.workers, scan_round)
        },
        on_round,
    )?;
    selected.initial_scan_seconds = seconds;
    Ok(selected)
}

fn validate_options(pool: &Model, support: &[u64], options: Options) -> Result<(), String> {
    let schedule = options.schedule;
    if options.vocabulary < BASE_SYMBOLS
        || options.vocabulary > pool.tokens.len()
        || support.len() != pool.tokens.len()
        || !(1..=256).contains(&options.workers)
        || !(0.0..1.0).contains(&schedule.coarse_keep)
        || schedule.coarse_keep == 0.0
        || !(0.0..1.0).contains(&schedule.fine_keep)
        || schedule.fine_keep == 0.0
        || schedule.fine_below < BASE_SYMBOLS
        || options.coverage_discount_ppm > 1_000_000
        || !options.coverage_settings.valid()
        || options.batch_search.is_some_and(|search| {
            !search.valid() || !options.exact_refresh || options.coverage_discount_ppm != 0
        })
    {
        return Err("invalid reverse-selection vocabulary, support or workers".into());
    }
    Ok(())
}

fn totals(tokenizer: &Tokenizer, uses: &[u64], active: &[u8]) -> Result<Totals, String> {
    let (mut bytes, mut fields, mut tokens, mut byte_tokens) = (0u128, 0u128, 0u128, 0u128);
    for (id, (&n, &kept)) in uses.iter().zip(active).enumerate() {
        if n != 0 && kept == 0 {
            return Err("retired reverse-selection token still has occurrences".into());
        }
        let token = &tokenizer.model().tokens[id];
        bytes += u128::from(n) * token.bytes.len() as u128;
        fields += u128::from(n) * u128::from(token.ends_field);
        tokens += u128::from(n);
        if id < 256 {
            byte_tokens += u128::from(n);
        }
    }
    let checked = |n| u64::try_from(n).map_err(|_| "reverse-selection totals overflow".to_string());
    Ok(Totals {
        bytes: checked(bytes)?,
        fields: checked(fields)?,
        tokens: checked(tokens)?,
        byte_tokens: checked(byte_tokens)?,
    })
}

#[derive(Default)]
struct Workspace {
    language: Option<LanguageWeight>,
    // Flat replacement paths; offsets stay usize/64-bit, not symbol IDs.
    offsets: Vec<usize>,
    edges: Vec<Symbol>,
    previous_offsets: Vec<usize>,
    previous_edges: Vec<Symbol>,
    scratch: ReplacementScratch,
    replacement: Vec<Symbol>,
    ranked: Vec<(u128, Symbol)>,
    ranks: Vec<Symbol>,
    selected: Vec<u8>,
    blocked: Vec<u8>,
    batch: Vec<Symbol>,
    round: usize,
    reused: usize,
    rebuilt: usize,
    planning_seconds: f64,
    ranking_seconds: f64,
    conflict_seconds: f64,
    coverage_discount_ppm: u32,
    coverage_settings: CoverageSettings,
    coverage: Option<super::coverage::Index>,
    coverage_seconds: f64,
    coverage_stats: super::coverage::Stats,
}

impl Workspace {
    fn path(&self, id: Symbol) -> &[Symbol] {
        &self.edges[self.offsets[id as usize]..self.offsets[id as usize + 1]]
    }

    fn allocated_bytes(&self) -> usize {
        self.offsets.capacity() * std::mem::size_of::<usize>()
            + self.edges.capacity() * std::mem::size_of::<Symbol>()
            + self.previous_offsets.capacity() * std::mem::size_of::<usize>()
            + self.previous_edges.capacity() * std::mem::size_of::<Symbol>()
            + self.scratch.allocated_bytes()
            + self.replacement.capacity() * std::mem::size_of::<Symbol>()
            + self.ranked.capacity() * std::mem::size_of::<(u128, Symbol)>()
            + self.ranks.capacity() * std::mem::size_of::<Symbol>()
            + self.selected.capacity()
            + self.blocked.capacity()
            + self.batch.capacity() * std::mem::size_of::<Symbol>()
            + self.coverage.as_ref().map_or(0, |c| c.allocated_bytes())
    }

    fn plan(
        &mut self,
        tokenizer: &Tokenizer,
        active: &[u8],
        uses: &[u64],
        support: &[u64],
        limit: usize,
    ) -> Result<(), String> {
        let planning = Instant::now();
        self.round += 1;
        self.reused = 0;
        self.rebuilt = 0;
        let progress = crate::training_progress::enabled();
        crate::training_progress::phase(
            "vocabulary_reverse_plan",
            self.round,
            0,
            active.len() as u64,
            "candidate_ids",
        );
        std::mem::swap(&mut self.offsets, &mut self.previous_offsets);
        std::mem::swap(&mut self.edges, &mut self.previous_edges);
        self.offsets.clear();
        self.edges.clear();
        self.ranked.clear();
        self.batch.clear();
        self.selected.resize(active.len(), 0);
        self.selected.fill(0);
        self.blocked.resize(active.len(), 0);
        self.blocked.fill(0);
        self.ranks.resize(active.len(), Symbol::MAX);
        self.ranks.fill(Symbol::MAX);
        for (id, &kept) in active.iter().enumerate() {
            if progress && id.is_multiple_of(65536) {
                crate::training_progress::phase(
                    "vocabulary_reverse_plan",
                    self.round,
                    id as u64,
                    active.len() as u64,
                    "candidate_ids",
                );
            }
            self.offsets.push(self.edges.len());
            if id < BASE_SYMBOLS || kept == 0 {
                continue;
            }
            // Vocabulary only shrinks. If the old optimal path survives,
            // removing other entries cannot make any cheaper path appear.
            let previous = self
                .previous_offsets
                .get(id + 1)
                .map(|&end| &self.previous_edges[self.previous_offsets[id]..end]);
            if let Some(path) =
                previous.filter(|p| !p.is_empty() && p.iter().all(|&c| active[c as usize] != 0))
            {
                self.edges.extend_from_slice(path);
                self.reused += 1;
            } else {
                tokenizer.replacement(
                    id as Symbol,
                    active,
                    &mut self.scratch,
                    &mut self.replacement,
                );
                if self.replacement.len() < 2 {
                    return Err("learned token needs a proper shorter-piece replacement".into());
                }
                self.edges.extend_from_slice(&self.replacement);
                self.rebuilt += 1;
            }
            let weight = if let Some(language) = &self.language {
                let mixed = |total: u64, part: u64| {
                    u128::from(total - part) * u128::from(language.denominator)
                        + u128::from(part) * u128::from(language.numerator)
                };
                50 * mixed(uses[id], language.uses[id]) + mixed(support[id], language.support[id])
            } else {
                50 * u128::from(uses[id]) + u128::from(support[id])
            };
            self.ranked.push((
                weight * (self.edges.len() - self.offsets[id] - 1) as u128,
                id as Symbol,
            ));
        }
        self.offsets.push(self.edges.len());
        crate::training_progress::phase(
            "vocabulary_reverse_plan",
            self.round,
            active.len() as u64,
            active.len() as u64,
            "candidate_ids",
        );
        self.planning_seconds = planning.elapsed().as_secs_f64();
        self.coverage_seconds = 0.0;
        self.coverage_stats = super::coverage::Stats::default();
        if self.coverage_discount_ppm != 0 {
            let started = Instant::now();
            let index = self
                .coverage
                .get_or_insert_with(|| super::coverage::Index::new(tokenizer.model()));
            index.settings = self.coverage_settings;
            self.coverage_stats = index.adjust(
                tokenizer,
                active,
                uses,
                &mut self.ranked,
                limit,
                self.coverage_discount_ppm,
            );
            self.coverage_seconds = started.elapsed().as_secs_f64();
        }
        let ranking = Instant::now();
        let prefix = limit.min(self.ranked.len());
        if prefix < self.ranked.len() {
            self.ranked.select_nth_unstable(prefix);
        }
        self.ranked[..prefix].sort_unstable();
        for (rank, &(_, id)) in self.ranked[..prefix].iter().enumerate() {
            self.ranks[id as usize] = rank as Symbol;
        }
        self.ranking_seconds = ranking.elapsed().as_secs_f64();
        let conflicts = Instant::now();
        // Match forward BRe-Pair priority: a cheaper conflicting dissolve still
        // blocks its neighbor even if it is itself blocked outside this batch.
        // Candidates outside the cheap prefix all have worse priority. Their
        // dependencies cannot block its members, so their conflict scan is free
        // to omit. Blocked cheaper members within the prefix still have priority.
        for &(_, id) in &self.ranked[..prefix] {
            for at in self.offsets[id as usize]..self.offsets[id as usize + 1] {
                let child = self.edges[at] as usize;
                if self.ranks[child] != Symbol::MAX {
                    let loser = if self.ranks[id as usize] < self.ranks[child] {
                        child
                    } else {
                        id as usize
                    };
                    self.blocked[loser] = 1;
                }
            }
        }
        // Independence is a safety condition, not a reason to admit arbitrarily
        // costly untouched tokens. Restrict removals to the cheapest prefix.
        for &(_, id) in self.ranked.iter().take(limit) {
            if self.blocked[id as usize] != 0 {
                continue;
            }
            self.selected[id as usize] = 1;
            self.batch.push(id);
            if self.batch.len() == limit {
                break;
            }
        }
        if self.batch.is_empty() {
            return Err("reverse-selection batch made no progress".into());
        }
        self.conflict_seconds = conflicts.elapsed().as_secs_f64();
        Ok(())
    }

    fn apply(&mut self, uses: &mut [u64], active: &mut [u8]) -> Result<u64, String> {
        let mut extra = 0u128;
        for &id in &self.batch {
            let path = self.path(id);
            if active[id as usize] == 0
                || path
                    .iter()
                    .any(|&c| active[c as usize] == 0 || self.selected[c as usize] != 0)
            {
                return Err("reverse batch contains a retired replacement dependency".into());
            }
            let n = uses[id as usize];
            let language_n = self.language.as_ref().map_or(0, |l| l.uses[id as usize]);
            extra += u128::from(n) * (path.len() - 1) as u128;
            uses[id as usize] = 0;
            active[id as usize] = 0;
            for &child in path {
                uses[child as usize] = uses[child as usize]
                    .checked_add(n)
                    .ok_or("reverse-selection usage overflow")?;
            }
            if let Some(language) = &mut self.language {
                language.uses[id as usize] = 0;
                for at in self.offsets[id as usize]..self.offsets[id as usize + 1] {
                    let child = self.edges[at] as usize;
                    language.uses[child] = language.uses[child]
                        .checked_add(language_n)
                        .ok_or("language usage overflow")?;
                }
            }
        }
        u64::try_from(extra).map_err(|_| "reverse-selection token increase overflow".into())
    }
}

fn run(
    tokenizer: Tokenizer,
    support: &[u64],
    uses: Vec<u64>,
    initial: Totals,
    options: Options,
    refresh: impl FnMut(&Tokenizer, &[u8]) -> Result<Scan, String>,
    on_round: impl FnMut(&Round),
) -> Result<Selection, String> {
    run_weighted(
        tokenizer, support, uses, initial, options, None, refresh, on_round,
    )
}

#[allow(clippy::too_many_arguments)]
fn run_weighted(
    tokenizer: Tokenizer,
    support: &[u64],
    mut uses: Vec<u64>,
    initial: Totals,
    options: Options,
    language: Option<LanguageWeight>,
    mut refresh: impl FnMut(&Tokenizer, &[u8]) -> Result<Scan, String>,
    mut on_round: impl FnMut(&Round),
) -> Result<Selection, String> {
    let mut active = vec![1u8; tokenizer.model().tokens.len()];
    let mut remaining = active.len();
    let mut current = totals(&tokenizer, &uses, &active)?;
    if current.bytes != initial.bytes
        || current.fields != initial.fields
        || current.tokens != initial.tokens
        || current.byte_tokens != initial.byte_tokens
    {
        return Err("initial reverse counts disagree with the full parse".into());
    }
    let mut work = Workspace {
        language,
        coverage_discount_ppm: options.coverage_discount_ppm,
        coverage_settings: options.coverage_settings,
        ..Default::default()
    };
    let initial_language = work
        .language
        .as_ref()
        .map(|l| totals(&tokenizer, &l.uses, &active))
        .transpose()?;
    let mut refresh_scans = 0;
    let mut refresh_seconds = 0.0;
    let mut previous_exact_tokens = initial.tokens;
    let mut pending_refresh: Option<Scan> = None;
    while remaining > options.vocabulary {
        let timer = Instant::now();
        let mut reparse_saved = 0;
        let mut scan_seconds = 0.0;
        if options.exact_refresh && work.round != 0 {
            let measured = match pending_refresh.take() {
                Some(measured) => measured,
                None => refresh(&tokenizer, &active)?,
            };
            if measured.all.bytes != initial.bytes
                || measured.all.fields != initial.fields
                || measured.all.tokens > current.tokens
                || measured.all.tokens < previous_exact_tokens
            {
                return Err("reverse refresh violates exact-parse bounds".into());
            }
            reparse_saved = current.tokens - measured.all.tokens;
            scan_seconds = measured.seconds;
            uses = measured.uses;
            current = totals(&tokenizer, &uses, &active)?;
            if current.tokens != measured.all.tokens {
                return Err("reverse refresh counts disagree with parse".into());
            }
            previous_exact_tokens = current.tokens;
            refresh_scans += 1;
            refresh_seconds += scan_seconds;
        }
        let next = if remaining > options.schedule.fine_below {
            ((remaining as f64 * options.schedule.coarse_keep) as usize)
                .max(options.schedule.fine_below)
                .max(options.vocabulary)
        } else {
            ((remaining as f64 * options.schedule.fine_keep) as usize).max(options.vocabulary)
        };
        work.plan(&tokenizer, &active, &uses, support, remaining - next)?;
        let batch_search = if let Some(search) = options.batch_search {
            let (best, stats) = search::choose(
                &mut work,
                &tokenizer,
                &mut active,
                &uses,
                (current, remaining - next),
                search,
                &mut refresh,
            )?;
            pending_refresh = Some(best);
            stats
        } else {
            SearchStats::default()
        };
        let updating = Instant::now();
        let extra = work.apply(&mut uses, &mut active)?;
        let after = totals(&tokenizer, &uses, &active)?;
        if let Some(language) = &work.language {
            if language.uses.iter().zip(&uses).any(|(a, b)| a > b) {
                return Err("language counts exceed total counts after reverse batch".into());
            }
            let current_language = totals(&tokenizer, &language.uses, &active)?;
            let initial_language = initial_language.as_ref().expect("language totals");
            if current_language.bytes != initial_language.bytes
                || current_language.fields != initial_language.fields
            {
                return Err("language reverse batch does not conserve bytes/fields".into());
            }
        }
        if after.bytes != initial.bytes
            || after.fields != initial.fields
            || current.tokens.checked_add(extra) != Some(after.tokens)
        {
            return Err(
                "reverse batch does not conserve bytes, EOFs or predicted token cost".into(),
            );
        }
        let next = remaining - work.batch.len();
        on_round(&Round {
            before: remaining,
            after: next,
            before_totals: current,
            after_totals: after,
            added_tokens: extra,
            deferred: work.blocked.iter().filter(|&&flag| flag != 0).count(),
            replacement_edges: work.edges.len(),
            workspace_bytes: work.allocated_bytes() + batch_search.workspace_bytes,
            replacement_reused: work.reused,
            replacement_rebuilt: work.rebuilt,
            planning_seconds: work.planning_seconds,
            ranking_seconds: work.ranking_seconds,
            conflict_seconds: work.conflict_seconds,
            update_seconds: updating.elapsed().as_secs_f64(),
            coverage_seconds: work.coverage_seconds,
            coverage_targets: work.coverage_stats.targets,
            coverage_pair_checks: work.coverage_stats.pair_checks,
            coverage_index_bytes: work.coverage.as_ref().map_or(0, |c| c.allocated_bytes()),
            coverage_mean_ratio_ppm: work.coverage_stats.ratio_ppm_sum as f64
                / work.coverage_stats.targets.max(1) as f64,
            coverage_max_ratio_ppm: work.coverage_stats.ratio_ppm_max,
            coverage_saturated_targets: work.coverage_stats.saturated_targets,
            coverage_zero_support_targets: work.coverage_stats.zero_support_targets,
            coverage_mean_one_overlap_share_ppm: work.coverage_stats.one_overlap_share_ppm_sum
                as f64
                / work.coverage_stats.targets.max(1) as f64,
            coverage_mean_internal_share_ppm: work.coverage_stats.internal_share_ppm_sum as f64
                / work.coverage_stats.targets.max(1) as f64,
            batch_search,
            reparse_saved,
            refresh_seconds: scan_seconds,
            seconds: timer.elapsed().as_secs_f64(),
        });
        remaining = next;
        current = after;
    }
    let maintained_uses = uses
        .into_iter()
        .zip(&active)
        .filter_map(|(n, &kept)| (kept != 0).then_some(n))
        .collect();
    let maintained_language_uses = work.language.map(|l| {
        l.uses
            .into_iter()
            .zip(&active)
            .filter_map(|(n, &kept)| (kept != 0).then_some(n))
            .collect()
    });
    Ok(Selection {
        model: tokenizer.model().subset(&active)?,
        initial,
        maintained: current,
        maintained_uses,
        maintained_language_uses,
        initial_scan_seconds: 0.0,
        refresh_scans,
        refresh_seconds,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{grammar::EOF, tokenizer::Token};

    fn weighted_options(vocabulary: usize) -> Options {
        Options {
            vocabulary,
            workers: 1,
            schedule: PruningSchedule::default(),
            exact_refresh: false,
            coverage_discount_ppm: 0,
            coverage_settings: CoverageSettings::default(),
            batch_search: None,
        }
    }

    #[test]
    fn language_weight_one_reproduces_raw_selection_and_shared_occurrences_keep_both_counts() {
        for scale in [1, 1u64 << 50] {
            let t = tokenizer(&[(b"ab", false), (b"cd", false), (b"xy", false)]);
            let n = t.model().tokens.len();
            let support = vec![0; n];
            let mut uses = vec![0; n];
            uses[257..].copy_from_slice(&[10 * scale, 6 * scale, 20 * scale]);
            let mut chinese = vec![0; n];
            chinese[257..].copy_from_slice(&[scale, 6 * scale, 0]);
            let initial = totals(&t, &uses, &vec![1; n]).unwrap();
            let options = weighted_options(n - 1);
            let baseline = run(
                Tokenizer::new(t.model().clone()).unwrap(),
                &support,
                uses.clone(),
                initial,
                options,
                |_, _| unreachable!(),
                |_| {},
            )
            .unwrap();
            for (num, den) in [(1, 1), (1_000_000, 1_000_000)] {
                let selected = select_language_weighted(
                    t.model().clone(),
                    &support,
                    uses.clone(),
                    initial,
                    LanguageWeight {
                        uses: chinese.clone(),
                        support: vec![0; n],
                        numerator: num,
                        denominator: den,
                    },
                    options,
                    |_| {},
                )
                .unwrap();
                assert_eq!(selected.model.to_bytes(), baseline.model.to_bytes());
            }
            let selected = select_language_weighted(
                t.model().clone(),
                &support,
                uses.clone(),
                initial,
                LanguageWeight {
                    uses: chinese,
                    support: vec![0; n],
                    numerator: 2,
                    denominator: 1,
                },
                options,
                |_| {},
            )
            .unwrap();
            assert!(!selected.model.tokens.iter().any(|t| t.bytes == b"ab"));
            assert!(selected.model.tokens.iter().any(|t| t.bytes == b"cd"));
            let counts = selected.maintained_language_uses.unwrap();
            assert_eq!(counts[b'a' as usize], scale);
            assert_eq!(counts[b'b' as usize], scale);
            assert_eq!(selected.maintained_uses[b'a' as usize], 10 * scale);
            assert_eq!(selected.maintained.bytes, initial.bytes);
        }
    }

    #[test]
    fn language_construction_prior_is_weighted_even_before_emitted_children_reappear() {
        let t = tokenizer(&[(b"ab", false), (b"cd", false)]);
        let n = t.model().tokens.len();
        let uses = vec![0; n];
        let mut support = vec![0; n];
        support[257..].copy_from_slice(&[10, 15]);
        let mut chinese_support = vec![0; n];
        chinese_support[257] = 10;
        let selected = select_language_weighted(
            t.model().clone(),
            &support,
            uses,
            Totals::default(),
            LanguageWeight {
                uses: vec![0; n],
                support: chinese_support,
                numerator: 2,
                denominator: 1,
            },
            weighted_options(n - 1),
            |_| {},
        )
        .unwrap();
        assert!(selected.model.tokens.iter().any(|t| t.bytes == b"ab"));
        assert!(!selected.model.tokens.iter().any(|t| t.bytes == b"cd"));
    }

    fn tokenizer(extra: &[(&[u8], bool)]) -> Tokenizer {
        let mut tokens: Vec<_> = (0..=255)
            .map(|b| Token {
                bytes: vec![b],
                ends_field: false,
            })
            .chain([Token {
                bytes: vec![],
                ends_field: true,
            }])
            .collect();
        tokens.extend(extra.iter().map(|(bytes, ends_field)| Token {
            bytes: bytes.to_vec(),
            ends_field: *ends_field,
        }));
        Tokenizer::new(Model::new(tokens).unwrap()).unwrap()
    }

    fn counts(ids: &[Symbol], size: usize) -> Vec<u64> {
        let mut uses = vec![0; size];
        for &id in ids {
            uses[id as usize] += 1;
        }
        uses
    }

    fn labels(t: &Tokenizer, ids: &[Symbol]) -> Vec<u16> {
        ids.iter()
            .flat_map(|&id| {
                let token = &t.model().tokens[id as usize];
                token
                    .bytes
                    .iter()
                    .map(|&b| u16::from(b))
                    .chain(token.ends_field.then_some(EOF as u16))
            })
            .collect()
    }

    #[test]
    fn batches_protect_dependencies_in_both_directions_and_allow_shared_children() {
        let t = tokenizer(&[
            (b"ab", false),
            (b"abc", false),
            (b"ac", false),
            (b"bc", false),
        ]);
        let active = vec![1; 261];
        let support = vec![0; 261];
        let mut uses = vec![0; 261];
        uses[257..].copy_from_slice(&[1, 2, 3, 4]);
        let mut work = Workspace::default();
        work.plan(&t, &active, &uses, &support, 4).unwrap();
        assert_eq!(work.path(258), [257, b'c' as u32]);
        assert_eq!(work.batch, [257, 259, 260]);
        uses[257..].copy_from_slice(&[2, 1, 3, 4]);
        work.plan(&t, &active, &uses, &support, 4).unwrap();
        assert_eq!(work.batch, [258, 259, 260]);
    }

    #[test]
    fn cheaper_outsiders_keep_priority_even_when_they_are_blocked() {
        let t = tokenizer(&[(b"ab", false), (b"abc", false), (b"abcd", false)]);
        let active = vec![1; 260];
        let mut uses = vec![0; 260];
        uses[257..].copy_from_slice(&[1, 2, 3]);
        let mut work = Workspace::default();
        work.plan(&t, &active, &uses, &vec![0; 260], 3).unwrap();
        assert_eq!(work.path(258), [257, b'c' as u32]);
        assert_eq!(work.path(259), [258, b'd' as u32]);
        assert_eq!(work.batch, [257]);
        assert_eq!(&work.blocked[257..], [0, 1, 1]);
    }

    #[test]
    fn costly_untouched_candidates_do_not_fill_a_blocked_cheap_prefix() {
        let t = tokenizer(&[(b"ab", false), (b"abc", false), (b"xy", false)]);
        let active = vec![1; 260];
        let mut uses = vec![0; 260];
        uses[257..].copy_from_slice(&[1, 2, 1000]);
        let mut work = Workspace::default();
        work.plan(&t, &active, &uses, &vec![0; 260], 2).unwrap();
        assert_eq!(work.batch, [257]);
        assert_eq!(work.blocked[259], 0);
        assert_eq!(work.selected[259], 0);
    }

    #[test]
    fn shrinking_vocabulary_cache_matches_fresh_dp_paths_and_batches() {
        let t = tokenizer(&[
            (b"ab", false),
            (b"abc", false),
            (b"bc", false),
            (b"abcd", false),
            (b"cd", false),
            (b"abcabcd", true),
            (b"bcd", false),
        ]);
        let n = t.model().tokens.len();
        let uses: Vec<_> = (0..n).map(|id| (id % 5 + 1) as u64).collect();
        let support: Vec<_> = (0..n).map(|id| (id % 11) as u64).collect();
        let mut total_reused = 0;
        for seed in 0..16u64 {
            let mut order: Vec<_> = (BASE_SYMBOLS..n).collect();
            order.sort_unstable_by_key(|&id| crate::pair_map::mix64(seed ^ id as u64));
            let mut cached = Workspace::default();
            let mut active = vec![1; n];
            for remaining in (1..=order.len()).rev() {
                for limit in [1, remaining] {
                    cached.plan(&t, &active, &uses, &support, limit).unwrap();
                    let mut fresh = Workspace::default();
                    fresh.plan(&t, &active, &uses, &support, limit).unwrap();
                    assert_eq!(cached.batch, fresh.batch);
                    for (id, &kept) in active.iter().enumerate().skip(BASE_SYMBOLS) {
                        if kept != 0 {
                            assert_eq!(cached.path(id as Symbol), fresh.path(id as Symbol));
                        }
                    }
                    assert_eq!(cached.reused + cached.rebuilt, remaining);
                    total_reused += cached.reused;
                }
                active[order[order.len() - remaining]] = 0;
            }
        }
        assert!(total_reused > 0);
    }

    #[test]
    fn partial_priority_sort_matches_full_sort_and_outsider_conflicts() {
        let t = tokenizer(&[
            (b"ab", false),
            (b"abc", false),
            (b"bc", false),
            (b"abcd", false),
            (b"cd", false),
            (b"abcabcd", true),
            (b"bcd", false),
        ]);
        let n = t.model().tokens.len();
        for seed in 0..32u64 {
            let active = vec![1; n];
            let uses: Vec<_> = (0..n)
                .map(|id| crate::pair_map::mix64(seed ^ id as u64) % 4)
                .collect();
            let support = vec![0; n];
            for limit in 1..n - BASE_SYMBOLS {
                let mut work = Workspace::default();
                work.plan(&t, &active, &uses, &support, limit).unwrap();
                let mut all = work.ranked.clone();
                all.sort_unstable();
                assert_eq!(work.ranked[..limit], all[..limit]);
                let mut ranks = vec![usize::MAX; n];
                let mut blocked = vec![false; n];
                for (rank, &(_, id)) in all.iter().enumerate() {
                    ranks[id as usize] = rank;
                }
                for &(_, id) in &all {
                    for &child in work.path(id) {
                        if ranks[child as usize] != usize::MAX {
                            let loser = if ranks[id as usize] < ranks[child as usize] {
                                child
                            } else {
                                id
                            };
                            blocked[loser as usize] = true;
                        }
                    }
                }
                let expected: Vec<_> = all[..limit]
                    .iter()
                    .filter_map(|&(_, id)| (!blocked[id as usize]).then_some(id))
                    .collect();
                assert_eq!(work.batch, expected);
            }
        }
    }

    #[test]
    fn dense_updates_equal_literal_rewrites_for_every_active_subset_and_batch_limit() {
        let t = tokenizer(&[
            (b"ab", false),
            (b"abc", false),
            (b"ac", false),
            (b"bc", false),
            (b"a", true),
            (b"bc", true),
            (&[255, 0], false),
        ]);
        let n = t.model().tokens.len();
        let support: Vec<_> = (0..n).map(|id| (id % 7) as u64).collect();
        let fields: &[&[u8]] = &[b"abcabc\n", b"abca", &[255, 0, 255, 0], b""];
        for mask in 1..(1usize << (n - BASE_SYMBOLS)) {
            let mut active = vec![1; n];
            for (bit, kept) in active[BASE_SYMBOLS..].iter_mut().enumerate() {
                *kept = u8::from(mask & (1 << bit) != 0);
            }
            let ids: Vec<_> = fields
                .iter()
                .flat_map(|f| t.encode_active(f, Some(&active)))
                .collect();
            let before = counts(&ids, n);
            let initial = totals(&t, &before, &active).unwrap();
            for limit in 1..=mask.count_ones() as usize {
                let mut work = Workspace::default();
                work.plan(&t, &active, &before, &support, limit).unwrap();
                for &id in &work.batch {
                    assert!(work
                        .path(id)
                        .iter()
                        .all(|&child| work.selected[child as usize] == 0));
                    // A replacement preserves the token's bytes and EOF exactly.
                    assert_eq!(labels(&t, work.path(id)), labels(&t, &[id]));
                }
                let rewritten: Vec<_> = ids
                    .iter()
                    .flat_map(|&id| {
                        if work.selected[id as usize] != 0 {
                            work.path(id).to_vec()
                        } else {
                            vec![id]
                        }
                    })
                    .collect();
                let mut uses = before.clone();
                let mut kept = active.clone();
                let extra = work.apply(&mut uses, &mut kept).unwrap();
                assert_eq!(uses, counts(&rewritten, n));
                assert_eq!(labels(&t, &ids), labels(&t, &rewritten));
                let after = totals(&t, &uses, &kept).unwrap();
                assert_eq!(after.tokens, initial.tokens + extra);
                assert_eq!((after.bytes, after.fields), (initial.bytes, initial.fields));
            }
        }
    }

    #[test]
    fn repeated_batches_match_literal_streams_until_every_budget_including_the_basis() {
        let t = tokenizer(&[
            (b"ab", false),
            (b"abc", false),
            (b"bc", false),
            (b"abcabc", true),
        ]);
        let n = t.model().tokens.len();
        let support = vec![0; n];
        let original: Vec<_> = [b"abcabc".as_slice(), b"abcab", b"bcabc"]
            .iter()
            .flat_map(|f| t.encode(f))
            .collect();
        for target in BASE_SYMBOLS..=n {
            let mut ids = original.clone();
            let mut active = vec![1; n];
            let mut uses = counts(&ids, n);
            let mut work = Workspace::default();
            let mut remaining = n;
            while remaining > target {
                work.plan(&t, &active, &uses, &support, remaining - target)
                    .unwrap();
                ids = ids
                    .iter()
                    .flat_map(|&id| {
                        if work.selected[id as usize] != 0 {
                            work.path(id).to_vec()
                        } else {
                            vec![id]
                        }
                    })
                    .collect();
                work.apply(&mut uses, &mut active).unwrap();
                assert_eq!(uses, counts(&ids, n));
                assert_eq!(labels(&t, &ids), labels(&t, &original));
                remaining -= work.batch.len();
            }
            assert_eq!(active.iter().filter(|&&flag| flag != 0).count(), target);
            assert!(active[..BASE_SYMBOLS].iter().all(|&flag| flag == 1));
        }
    }

    #[test]
    fn count_and_score_arithmetic_preserves_more_than_32_and_53_bits() {
        let t = tokenizer(&[(b"ab", false), (b"xy", false)]);
        let mut active = vec![1; 259];
        let mut uses = vec![0; 259];
        uses[257] = (1u64 << 53) + 1;
        uses[258] = 1u64 << 53;
        let mut support = vec![0; 259];
        support[257..].fill(u64::MAX);
        let before = totals(&t, &uses, &active).unwrap();
        let mut work = Workspace::default();
        work.plan(&t, &active, &uses, &support, 1).unwrap();
        assert_eq!(work.batch, [258]);
        assert!(work.ranked[0].0 > u128::from(u64::MAX));
        assert_eq!(work.apply(&mut uses, &mut active).unwrap(), 1u64 << 53);
        let after = totals(&t, &uses, &active).unwrap();
        assert_eq!(after.bytes, before.bytes);
        assert_eq!(after.tokens, before.tokens + (1u64 << 53));
        uses[257] = u64::MAX;
        assert!(totals(&t, &uses, &active).is_err());
    }

    #[test]
    fn final_exact_parse_can_cross_former_boundaries_and_improve_literal_rewrite() {
        let t = tokenizer(&[(b"ab", false), (b"abc", false), (b"bcd", false)]);
        let initial = t.encode(b"abcd");
        assert_eq!(initial, [258, b'd' as u32, EOF]);
        let mut active = vec![1; 260];
        let mut replacement = Vec::new();
        t.replacement(
            258,
            &active,
            &mut ReplacementScratch::default(),
            &mut replacement,
        );
        assert_eq!(replacement, [257, b'c' as u32]);
        let rewritten = [257, b'c' as u32, b'd' as u32, EOF];
        active[258] = 0;
        let optimal = t.encode_active(b"abcd", Some(&active));
        assert_eq!(optimal, [b'a' as u32, 259, EOF]);
        assert_eq!(rewritten.len(), optimal.len() + 1);
        assert_eq!(labels(&t, &rewritten), labels(&t, &optimal));
    }
}
