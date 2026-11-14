//! Opt-in batch enlargement using one shared index and dense per-candidate flags.
//! No occurrence lists, stored conflict graph, or per-worker pair maps.
use crate::{
    batch::{finish_selection, priority_key, Candidate, Selection},
    grammar::Symbol,
    pair_map::PairMap,
    search::{BatchPriority, Options},
};
use std::{
    cmp::{Ordering as Cmp, Reverse},
    collections::HashSet,
    sync::{
        atomic::{AtomicBool, Ordering},
        Mutex,
    },
};

pub(crate) fn scan(
    mut symbols: impl Iterator<Item = Symbol>,
    index: &PairMap<usize>,
    edge: &(dyn Fn(usize, usize) + Sync),
) {
    if let (Some(a), Some(mut b)) = (symbols.next(), symbols.next()) {
        let mut previous = index.get(&(a, b)).copied();
        for c in symbols {
            let next = index.get(&(b, c)).copied();
            if let (Some(p), Some(q)) = (previous, next) {
                if p != q {
                    edge(p, q);
                }
            }
            previous = next;
            b = c;
        }
    }
}

pub(crate) fn select(
    candidates: Vec<Candidate>,
    options: &Options,
    scan_edges: impl Fn(&PairMap<usize>, &(dyn Fn(usize, usize) + Sync)),
) -> Selection {
    select_observed(
        candidates,
        options,
        scan_edges,
        crate::training_progress::wave,
    )
}

fn select_observed(
    mut candidates: Vec<Candidate>,
    options: &Options,
    scan_edges: impl Fn(&PairMap<usize>, &(dyn Fn(usize, usize) + Sync)),
    mut on_wave: impl FnMut(usize, &[Candidate]),
) -> Selection {
    assert_eq!(
        options.frequency_override_percent, 0,
        "local overrides cannot be combined with consistent priorities/fill"
    );
    assert!((1..=8).contains(&options.fill_waves));
    let cap = match options.batch_priority {
        BatchPriority::Bounded2 => Some(2.0),
        BatchPriority::Bounded4 => Some(4.0),
        BatchPriority::Bounded8 => Some(8.0),
        _ => None,
    };
    if let Some(cap) = cap {
        for c in &mut candidates {
            // For weighted PMI alpha=1, priority/count is PMI in bits.
            // A multiplier in [1,2] limits association's influence globally.
            let association = c.priority / c.count as f64;
            c.priority = c.count as f64 * (1.0 + association.clamp(0.0, cap) / cap);
        }
    }
    candidates.sort_unstable_by_key(|c| (Reverse(c.count), c.pair));
    let index: PairMap<_> = candidates
        .iter()
        .enumerate()
        .map(|(i, c)| (c.pair, i))
        .collect();
    let keys: Vec<_> = candidates
        .iter()
        .map(|c| priority_key(c.priority))
        .collect();
    let band_bits = match options.batch_priority {
        BatchPriority::Bands2 => 1,
        BatchPriority::Bands4 => 2,
        _ => 0,
    };
    let bands: Vec<_> = if band_bits == 0 {
        Vec::new()
    } else {
        candidates
            .iter()
            .map(|c| (c.count.checked_ilog2().unwrap_or(0) / band_bits) as u8)
            .collect()
    };
    let unavailable: Vec<_> = candidates.iter().map(|_| AtomicBool::new(false)).collect();
    let mut chosen = vec![false; candidates.len()];
    let mut result: Option<Selection> = None;
    for wave in 0..options.fill_waves {
        if unavailable.iter().all(|b| b.load(Ordering::Relaxed)) {
            break;
        }
        let blocked: Vec<_> = candidates.iter().map(|_| AtomicBool::new(false)).collect();
        let ties = Mutex::new(HashSet::new());
        scan_edges(&index, &|p, q| {
            if unavailable[p].load(Ordering::Relaxed) || unavailable[q].load(Ordering::Relaxed) {
                return;
            }
            let order = if bands.is_empty() {
                Cmp::Equal
            } else {
                bands[p].cmp(&bands[q])
            }
            .then_with(|| keys[p].cmp(&keys[q]))
            .then_with(|| candidates[p].count.cmp(&candidates[q].count));
            match order {
                Cmp::Less => {
                    blocked[p].store(true, Ordering::Relaxed);
                }
                Cmp::Greater => {
                    blocked[q].store(true, Ordering::Relaxed);
                }
                Cmp::Equal => {
                    ties.lock()
                        .expect("poisoned tie set")
                        .insert((p.min(q), p.max(q)));
                }
            }
        });
        let blocked = blocked
            .into_iter()
            .enumerate()
            .map(|(i, b)| b.into_inner() || unavailable[i].load(Ordering::Relaxed))
            .collect();
        let next = finish_selection(
            &candidates,
            blocked,
            ties.into_inner().expect("poisoned tie set"),
            options,
        );
        on_wave(usize::from(wave) + 1, &next.pairs);
        if next.pairs.is_empty() {
            break;
        }
        for c in &next.pairs {
            let i = index[&c.pair];
            assert!(!chosen[i]);
            chosen[i] = true;
            unavailable[i].store(true, Ordering::Relaxed);
        }
        if let Some(total) = &mut result {
            total.eligible += next.eligible;
            total.tie_edges += next.tie_edges;
            total.components += next.components;
            total.largest_component = total.largest_component.max(next.largest_component);
            total.greedy_score += next.greedy_score;
            total.score += next.score;
            total.search_nodes += next.search_nodes;
            total.search_seconds += next.search_seconds;
            total.improved_components += next.improved_components;
            total.pairs.extend(next.pairs);
        } else {
            result = Some(next);
        }
        if wave + 1 < options.fill_waves {
            // Counts are sorted descending. If no unchosen candidate reaches
            // the extra-wave floor, avoid an otherwise pointless corpus scan.
            if !candidates
                .iter()
                .enumerate()
                .take_while(|(_, c)| c.count >= options.fill_min_count)
                .any(|(i, _)| !unavailable[i].load(Ordering::Relaxed))
            {
                break;
            }
            // Remove actual neighbors of the accumulated selection. Unselected
            // blocked outsiders may then cease vetoing the remaining candidates.
            // The next wave cannot conflict with anything already chosen.
            scan_edges(&index, &|p, q| {
                if chosen[p] {
                    unavailable[q].store(true, Ordering::Relaxed);
                }
                if chosen[q] {
                    unavailable[p].store(true, Ordering::Relaxed);
                }
            });
            for (i, c) in candidates.iter().enumerate() {
                if c.count < options.fill_min_count {
                    unavailable[i].store(true, Ordering::Relaxed);
                }
            }
        }
    }
    let mut result = result.unwrap_or_else(|| {
        assert!(
            candidates.is_empty(),
            "a nonempty globally ordered graph must make progress"
        );
        finish_selection(
            &candidates,
            vec![false; candidates.len()],
            HashSet::new(),
            options,
        )
    });
    result
        .pairs
        .sort_unstable_by_key(|c| (Reverse(c.count), c.pair));
    if options.fill_waves > 1 {
        // A sum of per-wave conditional bounds is not a global certificate.
        result.score_upper_bound = candidates.iter().map(|c| c.count).sum();
        result.proven_optimal = false;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn negative_priorities_remain_eligible_and_use_signed_order() {
        let candidates = vec![
            Candidate {
                pair: (1, 2),
                count: 8,
                priority: -1.0,
            },
            Candidate {
                pair: (2, 3),
                count: 16,
                priority: -2.0,
            },
            Candidate {
                pair: (3, 4),
                count: 32,
                priority: -3.0,
            },
        ];
        let options = Options {
            fill_waves: 2,
            fill_min_count: 8,
            ..Options::default()
        };
        let result = select(candidates, &options, |index, edge| {
            scan([1, 2, 3, 4].into_iter(), index, edge);
        });
        // Signed priorities select AB first; wave two releases CD. Neither
        // negative values nor their negative sum imply count exhaustion.
        let mut pairs = result.pairs.iter().map(|c| c.pair).collect::<Vec<_>>();
        pairs.sort_unstable();
        assert_eq!(pairs, [(1, 2), (3, 4)]);
        assert_eq!(result.score, 40);
    }

    #[test]
    fn wave_observer_preserves_selection_and_counts_released_candidates() {
        let large = u64::from(u32::MAX) + 1;
        let candidates: Vec<_> = [(1, 2), (2, 3), (3, 4)]
            .into_iter()
            .enumerate()
            .map(|(i, pair)| Candidate {
                pair,
                count: large,
                priority: (3 - i) as f64,
            })
            .collect();
        let options = Options {
            fill_waves: 2,
            ..Default::default()
        };
        let scan_edges = |index: &PairMap<usize>, edge: &(dyn Fn(usize, usize) + Sync)| {
            scan([1, 2, 3, 4].into_iter(), index, edge);
        };
        let mut waves = Vec::new();
        let observed =
            select_observed(candidates.clone(), &options, scan_edges, |number, pairs| {
                waves.push((
                    number,
                    pairs.len(),
                    pairs.iter().map(|p| p.count).sum::<u64>(),
                ));
            });
        let ordinary = select(candidates, &options, scan_edges);
        assert_eq!(waves, [(1, 1, large), (2, 1, large)]);
        assert_eq!(
            observed.pairs.iter().map(|c| c.pair).collect::<Vec<_>>(),
            ordinary.pairs.iter().map(|c| c.pair).collect::<Vec<_>>()
        );
        assert_eq!(observed.score, ordinary.score);
        assert_eq!(waves.iter().map(|w| w.2).sum::<u64>(), observed.score);
    }

    #[test]
    fn high_fill_floor_needs_no_extra_corpus_scan() {
        let candidates = vec![
            Candidate {
                pair: (1, 2),
                count: 10,
                priority: 3.0,
            },
            Candidate {
                pair: (2, 3),
                count: 10,
                priority: 2.0,
            },
            Candidate {
                pair: (3, 4),
                count: 10,
                priority: 1.0,
            },
        ];
        let calls = std::cell::Cell::new(0);
        let actual = select(
            candidates,
            &Options {
                fill_waves: 2,
                fill_min_count: 11,
                ..Default::default()
            },
            |index, edge| {
                calls.set(calls.get() + 1);
                scan([1, 2, 3, 4].into_iter(), index, edge);
            },
        );
        assert_eq!(calls.get(), 1);
        assert_eq!(
            actual.pairs.iter().map(|c| c.pair).collect::<Vec<_>>(),
            [(1, 2)]
        );
    }

    #[test]
    fn fill_releases_a_blocked_outsider_veto_without_overlapping() {
        let candidates = vec![
            Candidate {
                pair: (1, 2),
                count: 10,
                priority: 3.0,
            },
            Candidate {
                pair: (2, 3),
                count: 10,
                priority: 2.0,
            },
            Candidate {
                pair: (3, 4),
                count: 10,
                priority: 1.0,
            },
        ];
        let mut options = Options::default();
        let baseline =
            crate::batch::select_candidates([1, 2, 3, 4].into_iter(), candidates.clone(), &options);
        assert_eq!(
            baseline.pairs.iter().map(|c| c.pair).collect::<Vec<_>>(),
            [(1, 2)]
        );
        options.fill_waves = 2;
        let filled =
            crate::batch::select_candidates([1, 2, 3, 4].into_iter(), candidates, &options);
        assert_eq!(
            filled.pairs.iter().map(|c| c.pair).collect::<Vec<_>>(),
            [(1, 2), (3, 4)]
        );
        assert_eq!(filled.score, 20);
        assert!(!filled.proven_optimal);
        options.fill_min_count = 11;
        let filtered = crate::batch::select_candidates(
            [1, 2, 3, 4].into_iter(),
            vec![
                Candidate {
                    pair: (1, 2),
                    count: 10,
                    priority: 3.0,
                },
                Candidate {
                    pair: (2, 3),
                    count: 10,
                    priority: 2.0,
                },
                Candidate {
                    pair: (3, 4),
                    count: 10,
                    priority: 1.0,
                },
            ],
            &options,
        );
        assert_eq!(filtered.pairs.len(), 1);
    }

    #[test]
    fn every_five_vertex_graph_preserves_baseline_and_batch_independence() {
        let candidates: Vec<_> = (0..5)
            .map(|i| Candidate {
                pair: (2 * i, 2 * i + 1),
                count: u64::from(7 + i),
                priority: f64::from(7 + i),
            })
            .collect();
        let all_edges: Vec<_> = (0..5)
            .flat_map(|a| (a + 1..5).map(move |b| (a, b)))
            .collect();
        for bits in 0..1usize << all_edges.len() {
            let edges: Vec<_> = all_edges
                .iter()
                .enumerate()
                .filter_map(|(i, &edge)| ((bits >> i) & 1 != 0).then_some(edge))
                .collect();
            let scan_edges = |index: &PairMap<usize>, edge: &(dyn Fn(usize, usize) + Sync)| {
                for &(a, b) in &edges {
                    edge(index[&candidates[a].pair], index[&candidates[b].pair]);
                }
            };
            let mut options = Options::default();
            let baseline = select(candidates.clone(), &options, scan_edges);
            for waves in [2, 4, 8] {
                options.fill_waves = waves;
                let actual = select(candidates.clone(), &options, scan_edges);
                let selected: Vec<_> = actual
                    .pairs
                    .iter()
                    .map(|c| (c.pair.0 / 2) as usize)
                    .collect();
                assert!(!selected.is_empty());
                assert!(actual.score >= baseline.score);
                for c in &baseline.pairs {
                    assert!(actual.pairs.iter().any(|p| p.pair == c.pair));
                }
                for &(a, b) in &edges {
                    assert!(!(selected.contains(&a) && selected.contains(&b)));
                }
                if waves == 8 {
                    for i in 0..5 {
                        assert!(
                            selected.contains(&i)
                                || edges.iter().any(|&(a, b)| (a == i && selected.contains(&b))
                                    || (b == i && selected.contains(&a)))
                        );
                    }
                }
                assert!(actual.score_upper_bound >= actual.score);
            }
        }
    }

    #[test]
    fn consistent_priorities_preserve_progress_and_parallel_decisions() {
        use crate::{
            batch,
            grammar::{Grammar, EOF},
            packed::PackedSymbols,
            scoring::{Model, Statistics},
        };
        let raw = b"aaaaa aabcdaafgcdaacgadaaee abcd abcdef the there qu qu\n".repeat(12);
        let mut block = PackedSymbols::new(9);
        for field in raw.split_inclusive(|&b| b == b'\n') {
            for &b in field {
                block.push(u32::from(b));
            }
            block.push(EOF);
        }
        let grammar = Grammar::default();
        let mut counter = batch::PairCounter::default();
        for s in block.iter() {
            counter.push(s, &grammar);
        }
        let stats = Statistics::from_counter(&counter, 257);
        let candidates: Vec<_> = counter
            .counts
            .iter()
            .filter(|(_, n)| **n >= 2)
            .map(|(&pair, &count)| Candidate {
                pair,
                count,
                priority: stats.score(Model::WeightedPmi, pair, count),
            })
            .collect();
        let blocks = vec![block.clone(), block];
        for batch_priority in [
            BatchPriority::Score,
            BatchPriority::Bands2,
            BatchPriority::Bands4,
            BatchPriority::Bounded2,
            BatchPriority::Bounded4,
            BatchPriority::Bounded8,
        ] {
            for fill_waves in [1, 2, 4] {
                let options = Options {
                    batch_priority,
                    fill_waves,
                    ..Default::default()
                };
                let expected = batch::select_candidates(
                    blocks.iter().flat_map(PackedSymbols::iter),
                    candidates.clone(),
                    &options,
                );
                let pairs: Vec<_> = expected.pairs.iter().map(|c| c.pair).collect();
                assert!(!pairs.is_empty());
                for workers in [1, 2, 4] {
                    let actual = batch::select_candidates_fields(
                        &blocks,
                        candidates.clone(),
                        &options,
                        workers,
                    );
                    assert_eq!(
                        actual.pairs.iter().map(|c| c.pair).collect::<Vec<_>>(),
                        pairs
                    );
                    assert_eq!(actual.score, expected.score);
                }
                let source: Vec<_> = blocks.iter().flat_map(PackedSymbols::iter).collect();
                for triple in source.windows(3) {
                    let a = (triple[0], triple[1]);
                    let b = (triple[1], triple[2]);
                    assert!(a == b || !(pairs.contains(&a) && pairs.contains(&b)));
                }
            }
        }
    }

    #[test]
    fn new_priorities_keep_full_width_count_ties() {
        let candidates = vec![
            Candidate {
                pair: (1, 2),
                count: (1 << 53) + 1,
                priority: 1.0,
            },
            Candidate {
                pair: (2, 3),
                count: 1 << 53,
                priority: 1.0,
            },
        ];
        for batch_priority in [
            BatchPriority::Score,
            BatchPriority::Bands2,
            BatchPriority::Bands4,
            BatchPriority::Bounded2,
            BatchPriority::Bounded4,
            BatchPriority::Bounded8,
        ] {
            let options = Options {
                batch_priority,
                fill_waves: 2,
                ..Default::default()
            };
            let actual = crate::batch::select_candidates(
                [1, 2, 3].into_iter(),
                candidates.clone(),
                &options,
            );
            assert_eq!(actual.pairs.len(), 1);
            assert_eq!(actual.pairs[0].pair, (1, 2));
            assert_eq!(actual.score, (1 << 53) + 1);
        }
    }
}
