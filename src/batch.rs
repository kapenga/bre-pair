use std::cmp::Reverse;
use std::collections::{HashMap, HashSet};
use std::hash::BuildHasher;
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Mutex,
};
use std::time::Instant;

use crate::grammar::{Grammar, Pair, Symbol};
use crate::packed::PackedSymbols;
use crate::pair_map::PairMap;
use crate::scoring::{Model, Statistics};
use crate::search::{self, Options};

pub type Counts = PairMap<u64>;

#[cfg(test)]
mod large_count_tests {
    use super::*;

    #[test]
    fn updates_and_raw_marginals_preserve_counts_above_u32() {
        let grammar = Grammar::default();
        let large = u64::from(u32::MAX) + 1;
        let mut counter = PairCounter::default();
        // Seed aggregate counts to exercise the boundary without a multi-GB
        // input allocation. All production updates still use the normal path.
        counter.counts.insert((1, 2), large);
        counter.push(1, &grammar);
        counter.push(2, &grammar);
        assert_eq!(counter.counts[&(1, 2)], large + 1);
        counter.subtract(Edge::Pair((1, 2)));
        assert_eq!(counter.counts[&(1, 2)], large);

        counter.previous = Some(5);
        counter.run = (1usize << 32) + 1;
        counter.counts.insert((5, 5), large);
        counter.self_overlaps.insert(5, large);
        counter.push(5, &grammar); // even run: one replacement contribution
        counter.push(5, &grammar); // odd run: one raw overlap contribution
        assert_eq!(counter.run, (1usize << 32) + 3);
        assert_eq!(counter.counts[&(5, 5)], large + 1);
        assert_eq!(counter.self_overlaps[&5], large + 1);
        let stats = Statistics::from_counter(&counter, grammar.alphabet_size());
        assert_eq!(stats.observed((5, 5)), 2 * large + 2);
        assert_eq!(stats.left[5], 2 * large + 2);
        assert_eq!(stats.right[5], 2 * large + 2);
        assert_eq!(stats.edges, 3 * large + 2);
    }

    #[test]
    fn selection_preserves_large_minimum_counts_priorities_and_totals() {
        let large = u64::from(u32::MAX) + 1;
        let counts = Counts::from_iter([((1, 2), large + 10), ((2, 3), large + 1)]);
        let mut buffer = PackedSymbols::new(9);
        for symbol in [1, 2, 3] {
            buffer.push(symbol);
        }
        let selected = select(&buffer, &counts, large);
        assert_eq!(selected.pairs.len(), 1);
        assert_eq!(selected.pairs[0].pair, (1, 2));
        assert_eq!(selected.pairs[0].count, large + 10);
        assert_eq!(selected.score, large + 10);
        assert_eq!(selected.score_upper_bound, large + 10);
    }
}

/// All modes produce the same counts; only table maintenance differs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Counting {
    Full,
    Reuse,
    Incremental,
    #[default]
    Adaptive,
}

impl Counting {
    pub fn name(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::Reuse => "reuse",
            Self::Incremental => "incremental",
            Self::Adaptive => "adaptive",
        }
    }
}

impl std::str::FromStr for Counting {
    type Err = String;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "full" => Ok(Self::Full),
            "reuse" => Ok(Self::Reuse),
            "incremental" => Ok(Self::Incremental),
            "adaptive" => Ok(Self::Adaptive),
            _ => Err(format!("unknown pair counting mode: {value}")),
        }
    }
}

pub struct PairCounter {
    pub counts: Counts,
    /// Raw overlapping self-pair observations omitted by the replacement count.
    pub self_overlaps: HashMap<Symbol, u64>,
    previous: Option<Symbol>,
    run: usize,
    track_overlaps: bool,
}

impl Default for PairCounter {
    fn default() -> Self {
        Self {
            counts: Counts::default(),
            self_overlaps: HashMap::new(),
            previous: None,
            run: 0,
            track_overlaps: true,
        }
    }
}

impl PairCounter {
    pub fn new(track_overlaps: bool) -> Self {
        Self {
            track_overlaps,
            ..Self::default()
        }
    }
    pub fn tracks_self_overlaps(&self) -> bool {
        self.track_overlaps
    }

    fn clear(&mut self) {
        self.counts.clear();
        self.self_overlaps.clear();
        self.previous = None;
        self.run = 0;
    }

    pub fn push(&mut self, symbol: Symbol, grammar: &Grammar) {
        self.run = if self.previous == Some(symbol) {
            self.run + 1
        } else {
            1
        };
        if let Some(previous) = self.previous {
            if !grammar.ends_field(previous) {
                if previous == symbol && self.run % 2 == 1 {
                    if self.track_overlaps {
                        *self.self_overlaps.entry(symbol).or_default() += 1;
                    }
                } else {
                    *self.counts.entry((previous, symbol)).or_default() += 1;
                }
            }
        }
        self.previous = Some(symbol);
    }

    pub fn from_buffer(buffer: &PackedSymbols, grammar: &Grammar) -> Self {
        Self::from_buffer_with_tracking(buffer, grammar, true)
    }

    pub(crate) fn from_buffer_with_tracking(
        buffer: &PackedSymbols,
        grammar: &Grammar,
        track: bool,
    ) -> Self {
        let mut counter = Self::new(track);
        for symbol in buffer.iter() {
            counter.push(symbol, grammar);
        }
        counter
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Candidate {
    pub pair: Pair,
    pub count: u64,
    pub priority: f64,
}

pub struct Selection {
    pub candidates: usize,
    pub eligible: usize,
    pub tie_edges: usize,
    pub pairs: Vec<Candidate>,
    pub components: usize,
    pub largest_component: usize,
    pub greedy_score: u64,
    pub score: u64,
    pub score_upper_bound: u64,
    pub proven_optimal: bool,
    pub search_nodes: usize,
    pub search_seconds: f64,
    pub improved_components: usize,
}

/// Conflicts are binary and based on observed triples of current symbols.
/// A higher-frequency neighbor blocks a candidate even when that neighbor is blocked.
pub fn select<S: BuildHasher>(
    buffer: &PackedSymbols,
    counts: &HashMap<Pair, u64, S>,
    minimum: u64,
) -> Selection {
    select_with_options(buffer, counts, minimum, &Options::default())
}

pub fn select_with_options<S: BuildHasher>(
    buffer: &PackedSymbols,
    counts: &HashMap<Pair, u64, S>,
    minimum: u64,
    options: &Options,
) -> Selection {
    select_scored(buffer, counts, minimum, options, None)
}

/// Experimental priorities change blocking, while the objective remains the
/// sum of non-overlapping replacement counts among eligible pairs.
pub fn select_scored<S: BuildHasher>(
    buffer: &PackedSymbols,
    counts: &HashMap<Pair, u64, S>,
    minimum: u64,
    options: &Options,
    scoring: Option<(Model, &Statistics)>,
) -> Selection {
    select_scored_with_alpha(buffer, counts, minimum, options, scoring, 1.0)
}

pub fn select_scored_with_alpha<S: BuildHasher>(
    buffer: &PackedSymbols,
    counts: &HashMap<Pair, u64, S>,
    minimum: u64,
    options: &Options,
    scoring: Option<(Model, &Statistics)>,
    alpha: f64,
) -> Selection {
    let candidates: Vec<_> = counts
        .iter()
        .filter(|(_, count)| **count >= minimum)
        .map(|(&pair, &count)| Candidate {
            pair,
            count,
            priority: scoring.map_or(count as f64, |(model, stats)| {
                stats.score_with_alpha(model, pair, count, alpha)
            }),
        })
        .collect();
    select_candidates(buffer.iter(), candidates, options)
}

pub(crate) fn select_candidates(
    mut symbols: impl Iterator<Item = Symbol> + Clone,
    mut candidates: Vec<Candidate>,
    options: &Options,
) -> Selection {
    if options.batch_priority != search::BatchPriority::Score || options.fill_waves != 1 {
        return crate::batch_experiments::select(candidates, options, |index, edge| {
            crate::batch_experiments::scan(symbols.clone(), index, edge);
        });
    }
    candidates.sort_unstable_by_key(|c| (Reverse(c.count), c.pair));
    let index: PairMap<_> = candidates
        .iter()
        .enumerate()
        .map(|(i, c)| (c.pair, i))
        .collect();
    let mut blocked = vec![false; candidates.len()];
    let mut ties = HashSet::new();
    if !index.is_empty() {
        if let (Some(a), Some(mut b)) = (symbols.next(), symbols.next()) {
            let mut previous = index.get(&(a, b)).copied();
            for c in symbols {
                let next = index.get(&(b, c)).copied();
                if let (Some(p), Some(q)) = (previous, next) {
                    if p != q {
                        // Frequency breaks equal association scores, so surviving
                        // tie edges still join equal-count candidates. The existing
                        // unweighted component solvers and count bounds remain valid.
                        match conflict_priority(
                            candidates[p].count,
                            candidates[q].count,
                            options.frequency_override_percent,
                            || candidates[p].priority.total_cmp(&candidates[q].priority),
                        ) {
                            std::cmp::Ordering::Less => blocked[p] = true,
                            std::cmp::Ordering::Greater => blocked[q] = true,
                            std::cmp::Ordering::Equal => {
                                ties.insert((p.min(q), p.max(q)));
                            }
                        }
                    }
                }
                previous = next;
                b = c;
            }
        }
    }
    finish_selection(&candidates, blocked, ties, options)
}

/// An unsigned key with exactly f64::total_cmp's order, including signed zero
/// and NaN payloads. Dense keys keep the scan off the larger Candidate rows;
/// equal keys still use the original full-width count as a secondary priority.
#[inline]
pub(crate) fn priority_key(priority: f64) -> u64 {
    let bits = priority.to_bits();
    bits ^ (((bits as i64 >> 63) as u64) | (1 << 63))
}

#[inline]
fn conflict_priority(
    a: u64,
    b: u64,
    percent: u8,
    score_order: impl FnOnce() -> std::cmp::Ordering,
) -> std::cmp::Ordering {
    if percent != 0 && a != b {
        // Products fit u128 even at u64::MAX; no rounded floating-point counts.
        if u128::from(a.min(b)) * 100 <= u128::from(a.max(b)) * u128::from(percent) {
            return a.cmp(&b);
        }
    }
    score_order().then_with(|| a.cmp(&b))
}

#[cfg(test)]
mod selection_tests {
    use super::*;
    use crate::search::{Strategy, TieBreak};

    #[test]
    fn isolated_selection_keeps_full_counts_order_and_certificates_after_blocked_ties() {
        let high = (1u64 << 54) + 1;
        let candidates =
            [((9, 10), 4), ((2, 3), high), ((1, 2), high), ((3, 4), high)].map(|(pair, count)| {
                Candidate {
                    pair,
                    count,
                    priority: 1.0,
                }
            });
        for strategy in [Strategy::Greedy, Strategy::Beam, Strategy::Hybrid] {
            for certify in [false, true] {
                let options = Options {
                    strategy,
                    certify,
                    ..Options::default()
                };
                // The only tie reaches a vetoed candidate; it must not exclude
                // an otherwise isolated candidate or keep graph storage alive.
                let actual = finish_selection(
                    &candidates,
                    vec![false, false, false, true],
                    [(1, 3)].into_iter().collect(),
                    &options,
                );
                assert_eq!(
                    actual.pairs.iter().map(|c| c.pair).collect::<Vec<_>>(),
                    [(1, 2), (2, 3), (9, 10)]
                );
                assert_eq!(actual.score, high * 2 + 4);
                assert_eq!(actual.greedy_score, actual.score);
                assert_eq!(actual.score_upper_bound, actual.score);
                assert_eq!(actual.proven_optimal, certify);
                assert_eq!(
                    (actual.eligible, actual.components, actual.largest_component),
                    (3, 3, 1)
                );
                assert_eq!(
                    (
                        actual.tie_edges,
                        actual.search_nodes,
                        actual.improved_components
                    ),
                    (0, 0, 0)
                );
            }
        }
    }

    fn bridge_selection(rows: &[(Pair, u64, f64)], fields: &[&[Symbol]], percent: u8) -> Vec<Pair> {
        let candidates: Vec<_> = rows
            .iter()
            .map(|&(pair, count, priority)| Candidate {
                pair,
                count,
                priority,
            })
            .collect();
        let blocks: Vec<_> = (0..4)
            .map(|_| {
                let mut block = PackedSymbols::new(9);
                for field in fields {
                    for &s in *field {
                        block.push(s);
                    }
                    block.push(crate::grammar::EOF);
                }
                block
            })
            .collect();
        let options = Options {
            frequency_override_percent: percent,
            ..Default::default()
        };
        let signature = |s: Selection| s.pairs.iter().map(|c| c.pair).collect::<Vec<_>>();
        let expected = signature(select_candidates(
            blocks.iter().flat_map(PackedSymbols::iter),
            candidates.clone(),
            &options,
        ));
        for workers in [1, 2, 4] {
            let actual = select_candidates_fields(&blocks, candidates.clone(), &options, workers);
            assert_eq!(signature(actual), expected);
        }
        expected
    }

    #[test]
    fn weak_high_pmi_bridge_yields_to_both_frequent_neighbors() {
        let mut rows = [((1, 2), 1000, 1.0), ((2, 3), 400, 10.0), ((3, 4), 800, 2.0)];
        let fields: &[&[Symbol]] = &[&[1, 2, 3, 4]];
        assert_eq!(bridge_selection(&rows, fields, 0), [(2, 3)]);
        assert_eq!(bridge_selection(&rows, fields, 50), [(1, 2), (3, 4)]);
        // At 401, BC exceeds half of CD. Its veto still counts even though AB
        // blocks BC: outsiders retain priority under the thresholded rule.
        rows[1].1 = 401;
        assert_eq!(bridge_selection(&rows, fields, 50), [(1, 2)]);
        for row in &mut rows {
            row.2 = row.1 as f64;
        }
        assert_eq!(
            bridge_selection(&rows, fields, 0),
            bridge_selection(&rows, fields, 50)
        );
    }

    #[test]
    fn count_overrides_need_actual_overlap_and_never_select_both_sides() {
        let rows = [((1, 2), 1000, 1.0), ((2, 3), 400, 10.0)];
        assert_eq!(bridge_selection(&rows, &[&[1, 2, 3]], 50), [(1, 2)]);
        // No ABC triple: shared symbol identity alone creates no conflict.
        assert_eq!(
            bridge_selection(&rows, &[&[1, 2], &[2, 3]], 50),
            [(1, 2), (2, 3)]
        );
        // Equal counts still obey score; exact score/count ties still use MIS.
        let equal = [((1, 2), 8, 1.0), ((2, 3), 8, 2.0)];
        assert_eq!(bridge_selection(&equal, &[&[1, 2, 3]], 100), [(2, 3)]);
        let tied = [((1, 2), 8, 1.0), ((2, 3), 8, 1.0)];
        assert_eq!(bridge_selection(&tied, &[&[1, 2, 3]], 50).len(), 1);
    }

    #[test]
    fn count_override_cycles_cannot_cause_false_convergence() {
        // A > C by count, C > B by score, B > A by score. The deterministic
        // single-pair escape must run instead of declaring convergence.
        let rows = [((1, 1), 100, 1.0), ((1, 2), 70, 2.0), ((2, 1), 40, 3.0)];
        assert_eq!(
            bridge_selection(&rows, &[&[1, 1, 2], &[1, 2, 1], &[2, 1, 1]], 50),
            [(1, 1)]
        );
        // Exercise every local score order/count combination on this triangle:
        // a batch must make progress and cannot contain adjacent pair types.
        for a in [4, 7, 10] {
            for b in [4, 7, 10] {
                for c in [4, 7, 10] {
                    for scores in [
                        [1., 2., 3.],
                        [1., 3., 2.],
                        [2., 1., 3.],
                        [2., 3., 1.],
                        [3., 1., 2.],
                        [3., 2., 1.],
                    ] {
                        let rows = [
                            ((1, 1), a, scores[0]),
                            ((1, 2), b, scores[1]),
                            ((2, 1), c, scores[2]),
                        ];
                        assert_eq!(
                            bridge_selection(&rows, &[&[1, 1, 2], &[1, 2, 1], &[2, 1, 1]], 50)
                                .len(),
                            1
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn count_ratio_comparison_is_exact_above_float_and_u32_ranges() {
        use std::cmp::Ordering::{Greater, Less};
        let high = 1u64 << 55;
        assert_eq!(conflict_priority(high / 2, high, 50, || Greater), Less);
        assert_eq!(
            conflict_priority(high / 2 + 1, high, 50, || Greater),
            Greater
        );
        assert_eq!(
            conflict_priority(u64::MAX / 2, u64::MAX, 50, || Greater),
            Less
        );
        assert_eq!(
            conflict_priority(u64::MAX / 2 + 1, u64::MAX, 50, || Greater),
            Greater
        );
        assert_eq!(
            conflict_priority(u64::MAX, u64::MAX - 1, 100, || Less),
            Greater
        );
    }

    #[test]
    fn priority_keys_preserve_float_total_order_and_full_width_count_ties() {
        let priorities = [
            f64::from_bits(0xfff8_0000_0000_0001),
            f64::NEG_INFINITY,
            -1.0,
            -0.0,
            0.0,
            1.0,
            f64::INFINITY,
            f64::from_bits(0x7ff8_0000_0000_0001),
            f64::from_bits(0x7ff8_0000_0000_0002),
            1.0,
        ];
        let counts = [
            0,
            1,
            u64::from(u32::MAX) + 1,
            1 << 53,
            (1 << 53) + 1,
            u64::MAX,
        ];
        let candidates: Vec<_> = priorities
            .iter()
            .flat_map(|&priority| {
                counts.iter().map(move |&count| Candidate {
                    pair: (0, 1),
                    count,
                    priority,
                })
            })
            .collect();
        let keys: Vec<_> = candidates
            .iter()
            .map(|c| priority_key(c.priority))
            .collect();
        for (i, a) in candidates.iter().enumerate() {
            for (j, b) in candidates.iter().enumerate() {
                assert_eq!(
                    keys[i].cmp(&keys[j]).then(a.count.cmp(&b.count)),
                    a.priority
                        .total_cmp(&b.priority)
                        .then(a.count.cmp(&b.count)),
                    "{a:?} / {b:?}"
                );
            }
        }
    }

    #[test]
    fn parallel_selection_preserves_blocked_blockers_ties_and_observed_overlaps() {
        let candidates: Vec<_> = [
            ((1, 2), 10, 3.0),
            ((2, 3), 10, 2.0),
            ((3, 4), 10, 1.0),
            ((10, 11), 10, 7.0),
            ((11, 12), 10, 7.0),
            ((12, 13), 10, 7.0),
            ((20, 21), (1 << 53) + 1, 1.0),
            ((21, 22), 1 << 53, 1.0),
            ((30, 31), 10, -0.0),
            ((31, 32), 10, 0.0),
            ((40, 41), 10, f64::NAN),
            ((41, 42), 10, f64::INFINITY),
            ((50, 51), 10, 1.0),
            ((51, 52), 10, 2.0),
            ((60, 60), 2, 1.0),
        ]
        .into_iter()
        .map(|(pair, count, priority)| Candidate {
            pair,
            count,
            priority,
        })
        .collect();
        let fields: &[&[Symbol]] = &[
            &[1, 2, 3, 4],
            &[10, 11, 12, 13],
            &[20, 21, 22],
            &[30, 31, 32],
            &[40, 41, 42],
            &[50, 51],
            &[51, 52],
            &[60, 60, 60, 60, 60],
            &[],
        ];
        let blocks: Vec<_> = (0..64)
            .map(|_| {
                let mut block = PackedSymbols::new(9);
                for field in fields {
                    for &symbol in *field {
                        block.push(symbol);
                    }
                    block.push(crate::grammar::EOF);
                }
                block
            })
            .collect();
        for strategy in [Strategy::Greedy, Strategy::Beam, Strategy::Hybrid] {
            for tie_break in [TieBreak::Lex, TieBreak::Reverse, TieBreak::Seeded] {
                let options = Options {
                    strategy,
                    tie_break,
                    seed: 79,
                    ..Options::default()
                };
                let expected = select_candidates(
                    blocks.iter().flat_map(PackedSymbols::iter),
                    candidates.clone(),
                    &options,
                );
                let pairs: Vec<_> = expected.pairs.iter().map(|c| c.pair).collect();
                for pair in [
                    (1, 2),
                    (10, 11),
                    (12, 13),
                    (20, 21),
                    (31, 32),
                    (40, 41),
                    (50, 51),
                    (51, 52),
                    (60, 60),
                ] {
                    assert!(pairs.contains(&pair), "missing {pair:?}");
                }
                for pair in [(2, 3), (3, 4), (11, 12), (21, 22), (30, 31), (41, 42)] {
                    assert!(!pairs.contains(&pair), "unexpected {pair:?}");
                }
                let signature = |s: &Selection| {
                    (
                        s.pairs
                            .iter()
                            .map(|c| (c.pair, c.count, c.priority.to_bits()))
                            .collect::<Vec<_>>(),
                        [
                            s.candidates as u64,
                            s.eligible as u64,
                            s.tie_edges as u64,
                            s.components as u64,
                            s.largest_component as u64,
                            s.greedy_score,
                            s.score,
                            s.score_upper_bound,
                            s.proven_optimal as u64,
                            s.search_nodes as u64,
                            s.improved_components as u64,
                        ],
                    )
                };
                for workers in [1, 2, 4, 8] {
                    let actual =
                        select_candidates_fields(&blocks, candidates.clone(), &options, workers);
                    assert_eq!(
                        signature(&actual),
                        signature(&expected),
                        "{strategy:?} {tie_break:?} {workers}"
                    );
                }
            }
        }
    }
}

/// EOF-aligned chunks share one immutable candidate index, one blocked array,
/// and one tie-edge set. No per-worker maps or full graph copies.
pub(crate) fn select_candidates_fields(
    blocks: &[PackedSymbols],
    mut candidates: Vec<Candidate>,
    options: &Options,
    workers: usize,
) -> Selection {
    if options.batch_priority != search::BatchPriority::Score || options.fill_waves != 1 {
        return crate::batch_experiments::select(candidates, options, |index, edge| {
            let next_block = AtomicUsize::new(0);
            std::thread::scope(|scope| {
                for _ in 0..workers.min(blocks.len()).max(1) {
                    scope.spawn(|| loop {
                        let i = next_block.fetch_add(1, Ordering::Relaxed);
                        let Some(block) = blocks.get(i) else { break };
                        crate::batch_experiments::scan(block.iter(), index, edge);
                    });
                }
            });
        });
    }
    if workers == 1 || blocks.len() < 2 || candidates.is_empty() {
        return select_candidates(
            blocks.iter().flat_map(PackedSymbols::iter),
            candidates,
            options,
        );
    }
    #[cfg(feature = "profile")]
    let sort_profile = crate::profile::Span::new("selection.sort_candidates");
    candidates.sort_unstable_by_key(|c| (Reverse(c.count), c.pair));
    let priorities: Vec<_> = candidates
        .iter()
        .map(|c| priority_key(c.priority))
        .collect();
    #[cfg(feature = "profile")]
    drop(sort_profile);
    #[cfg(feature = "profile")]
    let index_profile = crate::profile::Span::new("selection.build_index");
    let index: PairMap<_> = candidates
        .iter()
        .enumerate()
        .map(|(i, c)| (c.pair, i))
        .collect();
    #[cfg(feature = "profile")]
    drop(index_profile);
    #[cfg(feature = "profile")]
    let conflicts_profile = crate::profile::Span::new("selection.parallel_conflicts");
    let blocked: Vec<_> = (0..candidates.len())
        .map(|_| AtomicBool::new(false))
        .collect();
    let ties = Mutex::new(HashSet::new());
    let next_block = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..workers.min(blocks.len()) {
            scope.spawn(|| {
                #[cfg(feature = "profile")]
                let _worker = crate::profile::Span::new("worker.conflicts");
                loop {
                    let i = next_block.fetch_add(1, Ordering::Relaxed);
                    if i >= blocks.len() {
                        break;
                    }
                    let mut symbols = blocks[i].iter();
                    if let (Some(a), Some(mut b)) = (symbols.next(), symbols.next()) {
                        let mut previous = index.get(&(a, b)).copied();
                        for c in symbols {
                            let next = index.get(&(b, c)).copied();
                            if let (Some(p), Some(q)) = (previous, next) {
                                if p != q {
                                    match conflict_priority(
                                        candidates[p].count,
                                        candidates[q].count,
                                        options.frequency_override_percent,
                                        || priorities[p].cmp(&priorities[q]),
                                    ) {
                                        std::cmp::Ordering::Less => {
                                            // Flags only move false -> true. A stale
                                            // read can cause a redundant store, never
                                            // lose a block; scoped joins publish it.
                                            if !blocked[p].load(Ordering::Relaxed) {
                                                blocked[p].store(true, Ordering::Relaxed);
                                            }
                                        }
                                        std::cmp::Ordering::Greater => {
                                            if !blocked[q].load(Ordering::Relaxed) {
                                                blocked[q].store(true, Ordering::Relaxed);
                                            }
                                        }
                                        std::cmp::Ordering::Equal => {
                                            ties.lock()
                                                .expect("poisoned tie set")
                                                .insert((p.min(q), p.max(q)));
                                        }
                                    }
                                }
                            }
                            previous = next;
                            b = c;
                        }
                    }
                }
            });
        }
    });
    #[cfg(feature = "profile")]
    drop(conflicts_profile);
    // These shared scan structures are no longer needed while constructing
    // the tie graph. Release them before its per-candidate allocations.
    drop(index);
    drop(priorities);
    finish_selection(
        &candidates,
        blocked.into_iter().map(AtomicBool::into_inner).collect(),
        ties.into_inner().expect("poisoned tie set"),
        options,
    )
}

pub(crate) fn finish_selection(
    candidates: &[Candidate],
    mut blocked: Vec<bool>,
    mut ties: HashSet<(usize, usize)>,
    options: &Options,
) -> Selection {
    assert!(options.frequency_override_percent <= 100);
    // Pairwise count overrides can create priority cycles (A beats B by score,
    // B beats C by score, C beats A by frequency). Never mistake an all-veto
    // cycle for convergence: choose just the first, most frequent candidate.
    // One pair type is always safe, including left-to-right AA replacements.
    // This deterministic escape adds neither a scan nor a second pair table.
    if options.frequency_override_percent != 0
        && !candidates.is_empty()
        && blocked.iter().all(|&b| b)
    {
        blocked[0] = false;
    }
    #[cfg(feature = "profile")]
    let _finish = crate::profile::Span::new("selection.finish_graph_and_search");
    let eligible = blocked.iter().filter(|&&value| !value).count();
    ties.retain(|&(p, q)| !blocked[p] && !blocked[q]);
    if ties.is_empty() {
        // All remaining vertices are isolated and mandatory. Avoid one Vec
        // header (24 bytes) and one local index (8 bytes) per candidate, plus
        // tiny allocations/searches for every isolated component.
        drop(ties);
        let search_started = Instant::now();
        let mut selected = Vec::with_capacity(eligible);
        selected.extend(
            candidates
                .iter()
                .zip(blocked)
                .filter_map(|(&candidate, blocked)| (!blocked).then_some(candidate)),
        );
        selected.sort_unstable_by_key(|c| (Reverse(c.count), c.pair));
        let score = selected.iter().map(|c| c.count).sum();
        return Selection {
            candidates: candidates.len(),
            eligible,
            tie_edges: 0,
            pairs: selected,
            components: eligible,
            largest_component: usize::from(eligible != 0),
            greedy_score: score,
            score,
            score_upper_bound: score,
            proven_optimal: options.certify,
            search_nodes: 0,
            search_seconds: search_started.elapsed().as_secs_f64(),
            improved_components: 0,
        };
    }
    let mut neighbors = vec![Vec::new(); candidates.len()];
    let mut tie_edges = 0;
    for (p, q) in ties {
        if !blocked[p] && !blocked[q] {
            neighbors[p].push(q);
            neighbors[q].push(p);
            tie_edges += 1;
        }
    }
    for adjacent in &mut neighbors {
        adjacent.sort_unstable();
    }
    let search_started = Instant::now();
    // Each connected component has a single frequency. Solve independently;
    // isolated eligible pairs are mandatory because all weights are positive.
    let mut visited = blocked;
    let mut local_index = vec![usize::MAX; candidates.len()];
    let mut selected = Vec::new();
    let mut components = 0;
    let mut largest_component = 0;
    let mut greedy_score = 0;
    let mut score_upper_bound = 0;
    let mut search_nodes = 0;
    let mut improved_components = 0;
    for root in 0..candidates.len() {
        if visited[root] {
            continue;
        }
        let mut stack = vec![root];
        let mut component = Vec::new();
        visited[root] = true;
        while let Some(v) = stack.pop() {
            component.push(v);
            for &u in &neighbors[v] {
                if !visited[u] {
                    visited[u] = true;
                    stack.push(u);
                }
            }
        }
        component.sort_unstable();
        components += 1;
        largest_component = largest_component.max(component.len());
        for (local, &global) in component.iter().enumerate() {
            local_index[global] = local;
        }
        let graph: Vec<Vec<usize>> = component
            .iter()
            .map(|&v| neighbors[v].iter().map(|&u| local_index[u]).collect())
            .collect();
        let root_pair = candidates[root].pair;
        let local_options = Options {
            seed: options.seed ^ ((root_pair.0 as u64) << 32 | root_pair.1 as u64),
            ..*options
        };
        let solution = search::solve(&graph, &local_options);
        let count = candidates[root].count;
        greedy_score += count * solution.greedy_size as u64;
        score_upper_bound += count * solution.upper_bound as u64;
        search_nodes += solution.nodes;
        if solution.vertices.len() > solution.greedy_size {
            improved_components += 1;
        }
        selected.extend(solution.vertices.iter().map(|&v| candidates[component[v]]));
    }
    selected.sort_unstable_by_key(|c| (Reverse(c.count), c.pair));
    let score = selected.iter().map(|c| c.count).sum();
    Selection {
        candidates: candidates.len(),
        eligible,
        tie_edges,
        pairs: selected,
        components,
        largest_component,
        greedy_score,
        score,
        score_upper_bound,
        proven_optimal: options.certify && score == score_upper_bound,
        search_nodes,
        search_seconds: search_started.elapsed().as_secs_f64(),
        improved_components,
    }
}

pub fn rewrite<S: BuildHasher>(
    buffer: &PackedSymbols,
    replacements: &HashMap<Pair, Symbol, S>,
    grammar: &Grammar,
    width: u8,
) -> (PackedSymbols, PairCounter) {
    rewrite_with_tracking(buffer, replacements, grammar, width, true)
}

pub(crate) fn rewrite_with_tracking<S: BuildHasher>(
    buffer: &PackedSymbols,
    replacements: &HashMap<Pair, Symbol, S>,
    grammar: &Grammar,
    width: u8,
    track: bool,
) -> (PackedSymbols, PairCounter) {
    rewrite_reusing(
        buffer,
        replacements,
        grammar,
        width,
        PairCounter::new(track),
        PackedSymbols::new(width),
    )
}

fn rewrite_reusing<S: BuildHasher>(
    buffer: &PackedSymbols,
    replacements: &HashMap<Pair, Symbol, S>,
    grammar: &Grammar,
    width: u8,
    mut counter: PairCounter,
    mut output: PackedSymbols,
) -> (PackedSymbols, PairCounter) {
    debug_assert_eq!(output.width(), width);
    let mut symbols = buffer.iter();
    let mut pending = symbols.next();
    while let Some(symbol) = pending {
        let next = symbols.next();
        let replacement = next.and_then(|next| replacements.get(&(symbol, next)).copied());
        let emitted = if let Some(replacement) = replacement {
            pending = symbols.next();
            replacement
        } else {
            pending = next;
            symbol
        };
        output.push(emitted);
        counter.push(emitted, grammar);
    }
    (output, counter)
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Edge {
    None,
    Pair(Pair),
    Overlap(Symbol),
}

#[derive(Default)]
pub(crate) struct Position {
    previous: Option<Symbol>,
    run: usize,
}

impl Position {
    pub(crate) fn advance(&mut self, symbol: Symbol, grammar: &Grammar, track: bool) -> Edge {
        self.run = if self.previous == Some(symbol) {
            self.run + 1
        } else {
            1
        };
        let edge = match self.previous {
            Some(previous) if !grammar.ends_field(previous) => {
                if previous == symbol && self.run % 2 == 1 {
                    if track {
                        Edge::Overlap(symbol)
                    } else {
                        Edge::None
                    }
                } else {
                    Edge::Pair((previous, symbol))
                }
            }
            _ => Edge::None,
        };
        self.previous = Some(symbol);
        edge
    }

    pub(crate) fn advance_with_start(
        &mut self,
        symbol: Symbol,
        grammar: &Grammar,
        track: bool,
    ) -> (Edge, bool) {
        let first = self.previous.is_none_or(|s| grammar.ends_field(s));
        (self.advance(symbol, grammar, track), first)
    }
}

fn decrement<K: Eq + std::hash::Hash, S: BuildHasher>(map: &mut HashMap<K, u64, S>, key: K) {
    let count = map.get_mut(&key).expect("missing old pair contribution");
    *count -= 1;
    if *count == 0 {
        map.remove(&key);
    }
}

impl PairCounter {
    fn subtract(&mut self, edge: Edge) {
        match edge {
            Edge::None => (),
            Edge::Pair(pair) => decrement(&mut self.counts, pair),
            Edge::Overlap(symbol) => decrement(&mut self.self_overlaps, symbol),
        }
    }

    fn add(&mut self, edge: Edge) {
        match edge {
            Edge::None => (),
            Edge::Pair(pair) => *self.counts.entry(pair).or_default() += 1,
            Edge::Overlap(symbol) => *self.self_overlaps.entry(symbol).or_default() += 1,
        }
    }
}

/// Rewrite while keeping one count table. Equal old/new contributions cancel
/// without hashing. Tracking both run parities handles self-pairs exactly,
/// including odd tails, merges at a run boundary, and adjacent replacements.
pub fn rewrite_counted<S: BuildHasher>(
    buffer: &PackedSymbols,
    replacements: &HashMap<Pair, Symbol, S>,
    grammar: &Grammar,
    width: u8,
    mut counter: PairCounter,
    mode: Counting,
) -> (PackedSymbols, PairCounter) {
    let removed: u64 = replacements
        .keys()
        .map(|pair| counter.counts.get(pair).copied().unwrap_or(0))
        .sum();
    let expected = buffer.len().saturating_sub(removed as usize);
    let mode = if mode == Counting::Adaptive {
        if buffer.len() < 1_000_000 || removed as u128 * 8 > buffer.len() as u128 {
            Counting::Reuse
        } else {
            Counting::Incremental
        }
    } else {
        mode
    };
    if mode == Counting::Full {
        let track = counter.track_overlaps;
        drop(counter);
        return rewrite_reusing(
            buffer,
            replacements,
            grammar,
            width,
            PairCounter::new(track),
            PackedSymbols::with_capacity(width, expected),
        );
    }
    if mode == Counting::Reuse {
        counter.clear();
        return rewrite_reusing(
            buffer,
            replacements,
            grammar,
            width,
            counter,
            PackedSymbols::with_capacity(width, expected),
        );
    }
    let mut output = PackedSymbols::with_capacity(width, expected);
    let mut old = Position::default();
    let mut new = Position::default();
    let mut symbols = buffer.iter();
    let mut pending = symbols.next();
    while let Some(symbol) = pending {
        let next = symbols.next();
        let replacement = next.and_then(|next| replacements.get(&(symbol, next)).copied());
        let mut first = old.advance(symbol, grammar, counter.track_overlaps);
        let mut second = Edge::None;
        let emitted = if let Some(replacement) = replacement {
            second = old.advance(next.unwrap(), grammar, counter.track_overlaps);
            pending = symbols.next();
            replacement
        } else {
            pending = next;
            symbol
        };
        let mut added = new.advance(emitted, grammar, counter.track_overlaps);
        if added == first {
            first = Edge::None;
            added = Edge::None;
        } else if added == second {
            second = Edge::None;
            added = Edge::None;
        }
        counter.subtract(first);
        counter.subtract(second);
        counter.add(added);
        output.push(emitted);
    }
    counter.previous = new.previous;
    counter.run = new.run;
    (output, counter)
}
