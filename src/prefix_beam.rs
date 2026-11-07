//! Bounded beam search over individual early merges. States contain recipes,
//! never copies of the corpus. Independent states own disjoint slices of one
//! dense counter arena; replay needs only one pending symbol per rule.
use crate::{
    engine::{Compressed, Config, Round},
    field_engine::{self, ParallelTraining},
    grammar::{Grammar, Pair, Symbol, BASE_SYMBOLS, EOF},
    packed::PackedSymbols,
};
use std::{collections::BTreeMap, time::Instant};

pub const MAX_RULES: usize = 16;
const LIMIT: usize = BASE_SYMBOLS + MAX_RULES;
const CELLS: usize = LIMIT * LIMIT;

#[derive(Clone, Copy, Debug)]
pub struct Options {
    pub rules: usize,
    pub width: usize,
    /// Propose this many types from EACH frequency and weighted-PMI ranking.
    pub branch: usize,
    pub minimum_count: u64,
    pub workers: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            rules: 16,
            width: 8,
            branch: 4,
            minimum_count: 8,
            workers: 4,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Prefix {
    pub pairs: Vec<Pair>,
    pub symbols: u64,
}

impl Prefix {
    pub fn grammar(&self) -> Result<Grammar, String> {
        if self.pairs.len() > MAX_RULES {
            return Err("prefix exceeds 16 rules".into());
        }
        let mut grammar = Grammar::default();
        for (index, &pair) in self.pairs.iter().enumerate() {
            if grammar.add(pair)? as usize != BASE_SYMBOLS + index {
                return Err("prefix contains a duplicate rule".into());
            }
        }
        Ok(grammar)
    }

    /// Streaming cascade: each stage performs one chronological pair rewrite.
    /// Scratch is O(rules), even for a single field larger than RAM.
    fn visit(&self, fields: &[Vec<u8>], mut emit: impl FnMut(Symbol)) {
        let mut pending = [None; MAX_RULES];
        for field in fields {
            for symbol in field.iter().map(|&b| u32::from(b)).chain([EOF]) {
                cascade(&self.pairs, &mut pending, 0, symbol, &mut emit);
            }
            for stage in 0..self.pairs.len() {
                if let Some(symbol) = pending[stage].take() {
                    cascade(&self.pairs, &mut pending, stage + 1, symbol, &mut emit);
                }
            }
        }
    }

    /// Materialize only the one finalist being continued, in EOF-aligned chunks.
    pub fn pack(&self, fields: &[Vec<u8>]) -> Result<Compressed, String> {
        let grammar = self.grammar()?;
        let mut blocks = Vec::new();
        let mut block = PackedSymbols::with_capacity(9, 65536);
        let mut symbols = 0u64;
        self.visit(fields, |symbol| {
            symbols += 1;
            block.push(symbol);
            if grammar.ends_field(symbol) && block.len() >= 65536 {
                blocks.push(std::mem::replace(
                    &mut block,
                    PackedSymbols::with_capacity(9, 65536),
                ));
            }
        });
        if !block.is_empty() {
            blocks.push(block);
        }
        if symbols != self.symbols {
            return Err("prefix replay disagrees with predicted size".into());
        }
        Ok(Compressed {
            grammar,
            blocks,
            input_bytes: fields.iter().map(|f| f.len() as u64).sum(),
            field_count: fields.len() as u64,
        })
    }
}

fn cascade(
    pairs: &[Pair],
    pending: &mut [Option<Symbol>; MAX_RULES],
    stage: usize,
    symbol: Symbol,
    emit: &mut impl FnMut(Symbol),
) {
    if stage == pairs.len() {
        emit(symbol);
        return;
    }
    match pending[stage].take() {
        None => pending[stage] = Some(symbol),
        Some(previous) if (previous, symbol) == pairs[stage] => cascade(
            pairs,
            pending,
            stage + 1,
            (BASE_SYMBOLS + stage) as Symbol,
            emit,
        ),
        Some(previous) => {
            cascade(pairs, pending, stage + 1, previous, emit);
            pending[stage] = Some(symbol);
        }
    }
}

#[derive(Debug)]
pub struct Level {
    pub depth: usize,
    pub states: usize,
    pub proposed: usize,
    pub distinct: usize,
    pub best_symbols: u64,
    pub seconds: f64,
}

pub struct Discovery {
    pub beam: Vec<Prefix>,
    pub greedy: Prefix,
    /// Dense pair cells plus exact self-overlap counters; no thread maps.
    pub counter_arena_bytes: usize,
}

struct Counts<'a> {
    pairs: &'a mut [u64],
    overlaps: &'a mut [u64],
    left: [u64; LIMIT],
    right: [u64; LIMIT],
    edges: u64,
    symbols: u64,
}

impl<'a> Counts<'a> {
    fn scan(prefix: &Prefix, fields: &[Vec<u8>], arena: &'a mut [u64]) -> Result<Self, String> {
        arena.fill(0);
        let (pairs, overlaps) = arena.split_at_mut(CELLS);
        let mut result = Self {
            pairs,
            overlaps,
            left: [0; LIMIT],
            right: [0; LIMIT],
            edges: 0,
            symbols: 0,
        };
        let grammar = prefix.grammar()?;
        let mut previous = None;
        let mut even = false;
        prefix.visit(fields, |symbol| {
            result.symbols += 1;
            if let Some(left) = previous {
                if !grammar.ends_field(left) {
                    let index = left as usize * LIMIT + symbol as usize;
                    if left != symbol || !even {
                        result.pairs[index] += 1;
                    } else {
                        result.overlaps[symbol as usize] += 1;
                    }
                    even = left == symbol && !even;
                    result.left[left as usize] += 1;
                    result.right[symbol as usize] += 1;
                    result.edges += 1;
                } else {
                    even = false;
                }
            }
            previous = Some(symbol);
        });
        if result.symbols != prefix.symbols {
            return Err("counted prefix size differs from prediction".into());
        }
        Ok(result)
    }

    fn proposals(&self, minimum: u64, branch: usize) -> (Vec<Pair>, Option<Pair>) {
        let mut candidates = Vec::new();
        for (index, &count) in self.pairs.iter().enumerate() {
            if count < minimum {
                continue;
            }
            let pair = ((index / LIMIT) as Symbol, (index % LIMIT) as Symbol);
            let observed = count
                + if pair.0 == pair.1 {
                    self.overlaps[pair.0 as usize]
                } else {
                    0
                };
            let expected = self.left[pair.0 as usize] as f64 * self.right[pair.1 as usize] as f64
                / self.edges as f64;
            let score = count as f64 * (observed as f64 / expected).log2();
            candidates.push((pair, count, score));
        }
        candidates.sort_unstable_by(|a, b| {
            b.2.total_cmp(&a.2)
                .then_with(|| b.1.cmp(&a.1))
                .then_with(|| a.0.cmp(&b.0))
        });
        let greedy = candidates.first().map(|c| c.0);
        let mut proposals: Vec<_> = candidates.iter().take(branch).map(|c| c.0).collect();
        candidates.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        proposals.extend(candidates.iter().take(branch).map(|c| c.0));
        proposals.sort_unstable();
        proposals.dedup();
        (proposals, greedy)
    }
}

/// Canonicalize only PROVABLY commuting operations. Dependencies include rule
/// operands and every possible shared-position overlap, even absent triples.
/// A deterministic topological order also renumbers all rule references.
fn canonical(pairs: &[Pair]) -> Vec<Pair> {
    let n = pairs.len();
    let mut dependencies = [0u32; MAX_RULES];
    for j in 0..n {
        for i in 0..j {
            let (a, b) = pairs[i];
            let (c, d) = pairs[j];
            let id = (BASE_SYMBOLS + i) as Symbol;
            if c == id || d == id || b == c || d == a {
                dependencies[j] |= 1 << i;
            }
        }
    }
    let mut mapping: Vec<_> = (0..BASE_SYMBOLS + n).map(|id| id as Symbol).collect();
    let mut done = 0u32;
    let mut result = Vec::with_capacity(n);
    while result.len() < n {
        let (index, pair) = (0..n)
            .filter(|&i| done & (1 << i) == 0 && dependencies[i] & !done == 0)
            .map(|i| {
                (
                    i,
                    (mapping[pairs[i].0 as usize], mapping[pairs[i].1 as usize]),
                )
            })
            .min_by_key(|&(i, pair)| (pair, i))
            .expect("acyclic merge history");
        mapping[BASE_SYMBOLS + index] = (BASE_SYMBOLS + result.len()) as Symbol;
        done |= 1 << index;
        result.push(pair);
    }
    result
}

pub fn discover(
    fields: &[Vec<u8>],
    options: Options,
    mut on_level: impl FnMut(&Level),
) -> Result<Discovery, String> {
    if options.rules == 0
        || options.rules > MAX_RULES
        || options.width == 0
        || options.width > 64
        || options.branch == 0
        || options.branch > 64
        || options.minimum_count < 2
        || options.workers == 0
        || options.workers > 256
    {
        return Err(
            "prefix needs 1..=16 rules, 1..=64 width/branch, minimum >=2 and 1..=256 workers"
                .into(),
        );
    }
    let root = Prefix {
        pairs: vec![],
        symbols: fields.iter().map(|f| f.len() as u64 + 1).sum(),
    };
    let mut beam = vec![root.clone()];
    let mut greedy = root;
    let stride = CELLS + LIMIT;
    let mut arena = vec![0u64; (options.width + 1) * stride];
    for depth in 1..=options.rules {
        let started = Instant::now();
        let mut parents = beam.clone();
        let greedy_index = match parents.iter().position(|p| p == &greedy) {
            Some(index) => index,
            None => {
                parents.push(greedy.clone());
                parents.len() - 1
            }
        };
        // Disjoint arena slices, one per STATE; no replication per worker.
        let scans = std::thread::scope(|scope| -> Result<Vec<_>, String> {
            let chunk = parents.len().div_ceil(options.workers);
            let mut jobs = Vec::new();
            for (group, memory) in parents.chunks(chunk).zip(arena.chunks_mut(chunk * stride)) {
                jobs.push(scope.spawn(move || -> Result<Vec<_>, String> {
                    group
                        .iter()
                        .zip(memory.chunks_mut(stride))
                        .map(|(prefix, memory)| {
                            let counts = Counts::scan(prefix, fields, memory)?;
                            let (proposals, greedy) =
                                counts.proposals(options.minimum_count, options.branch);
                            let proposals: Vec<_> = proposals
                                .into_iter()
                                .map(|pair| {
                                    (
                                        pair,
                                        counts.pairs[pair.0 as usize * LIMIT + pair.1 as usize],
                                    )
                                })
                                .collect();
                            let greedy = greedy.map(|pair| {
                                (
                                    pair,
                                    counts.pairs[pair.0 as usize * LIMIT + pair.1 as usize],
                                )
                            });
                            Ok((proposals, greedy))
                        })
                        .collect()
                }));
            }
            let mut result = Vec::new();
            for job in jobs {
                result.extend(job.join().map_err(|_| "prefix counter worker panicked")??);
            }
            Ok(result)
        })?;
        let mut children = BTreeMap::new();
        let mut proposed = 0;
        for (parent, (proposals, _)) in beam.iter().zip(&scans) {
            if proposals.is_empty() {
                children.insert(parent.pairs.clone(), parent.symbols);
            }
            for &(pair, count) in proposals {
                let mut pairs = parent.pairs.clone();
                pairs.push(pair);
                let key = canonical(&pairs);
                let symbols = parent.symbols - count;
                if let Some(old) = children.insert(key, symbols) {
                    if old != symbols {
                        return Err("commuting histories disagree on token count".into());
                    }
                }
                proposed += 1;
            }
        }
        if let Some((pair, count)) = scans[greedy_index].1 {
            greedy.pairs.push(pair);
            greedy.symbols -= count;
        }
        let distinct = children.len();
        beam = children
            .into_iter()
            .map(|(pairs, symbols)| Prefix { pairs, symbols })
            .collect();
        beam.sort_unstable_by(|a, b| {
            a.symbols
                .cmp(&b.symbols)
                .then_with(|| a.pairs.cmp(&b.pairs))
        });
        beam.truncate(options.width);
        on_level(&Level {
            depth,
            states: beam.len(),
            proposed,
            distinct,
            best_symbols: beam[0].symbols,
            seconds: started.elapsed().as_secs_f64(),
        });
        if proposed == 0 && scans[greedy_index].1.is_none() {
            break;
        }
    }
    Ok(Discovery {
        beam,
        greedy,
        counter_arena_bytes: arena.len() * 8,
    })
}

/// Continue exactly the supplied segmentation using the ordinary shared-registry
/// trainer. Its seeded limit counts the mandatory basis as selectable entries.
pub fn continue_training(
    prefix: &Prefix,
    fields: &[Vec<u8>],
    config: &Config,
    workers: usize,
    learned_limit: usize,
    on_round: impl FnMut(&Round),
) -> Result<ParallelTraining, String> {
    if learned_limit <= prefix.pairs.len() {
        return Err("learned budget must exceed prefix length".into());
    }
    let source = prefix.pack(fields)?;
    let members = vec![true; source.grammar.alphabet_size()];
    field_engine::compress_seeded(
        source,
        members,
        config,
        workers,
        field_engine::Options {
            max_rules: Some(
                learned_limit
                    .checked_add(BASE_SYMBOLS)
                    .ok_or("rule budget overflow")?,
            ),
            ..Default::default()
        },
        None,
        on_round,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        batch::PairCounter,
        scoring::{Model, Statistics},
    };

    fn reference(fields: &[Vec<u8>], pairs: &[Pair]) -> Vec<Symbol> {
        let mut data: Vec<_> = fields
            .iter()
            .flat_map(|f| f.iter().map(|&b| b as Symbol).chain([EOF]))
            .collect();
        for (i, &pair) in pairs.iter().enumerate() {
            let mut out = Vec::new();
            let mut at = 0;
            while at < data.len() {
                if at + 1 < data.len() && (data[at], data[at + 1]) == pair {
                    out.push((BASE_SYMBOLS + i) as Symbol);
                    at += 2;
                } else {
                    out.push(data[at]);
                    at += 1;
                }
            }
            data = out;
        }
        data
    }

    #[test]
    fn streaming_counts_match_independent_rewrite_and_pmi() {
        let fields = vec![
            b"aaaaa".to_vec(),
            vec![],
            b"ababcabcaaaaab".to_vec(),
            (0..=255).collect(),
        ];
        for pairs in [vec![], vec![(97, 97)], vec![(97, 98), (257, 99), (97, 256)]] {
            let expected = reference(&fields, &pairs);
            let prefix = Prefix {
                pairs,
                symbols: expected.len() as u64,
            };
            let packed = prefix.pack(&fields).unwrap();
            assert_eq!(packed.decode().unwrap(), fields);
            assert_eq!(
                packed
                    .blocks
                    .iter()
                    .flat_map(PackedSymbols::iter)
                    .collect::<Vec<_>>(),
                expected
            );
            let reference = PairCounter::from_buffer(&packed.blocks[0], &packed.grammar);
            let stats = Statistics::from_counter(&reference, packed.grammar.alphabet_size());
            let mut arena = vec![0; CELLS + LIMIT];
            let dense = Counts::scan(&prefix, &fields, &mut arena).unwrap();
            assert_eq!(dense.edges, stats.edges);
            assert_eq!(&dense.left[..stats.left.len()], stats.left);
            assert_eq!(&dense.right[..stats.right.len()], stats.right);
            for (&pair, &count) in &reference.counts {
                assert_eq!(
                    dense.pairs[pair.0 as usize * LIMIT + pair.1 as usize],
                    count
                );
                assert_eq!(
                    count
                        + if pair.0 == pair.1 {
                            dense.overlaps[pair.0 as usize]
                        } else {
                            0
                        },
                    stats.observed(pair)
                );
            }
            let (_, best) = dense.proposals(2, 4);
            let mut sorted: Vec<_> = reference.counts.iter().filter(|&(_, &n)| n >= 2).collect();
            sorted.sort_by(|(a, an), (b, bn)| {
                stats
                    .score(Model::WeightedPmi, **b, **bn)
                    .total_cmp(&stats.score(Model::WeightedPmi, **a, **an))
                    .then_with(|| bn.cmp(an))
                    .then_with(|| a.cmp(b))
            });
            assert_eq!(best, sorted.first().map(|&(pair, _)| *pair));
        }
    }

    #[test]
    fn canonicalization_only_combines_commuting_orders() {
        assert_eq!(
            canonical(&[(97, 98), (99, 100), (257, 258)]),
            canonical(&[(99, 100), (97, 98), (258, 257)])
        );
        assert_ne!(
            canonical(&[(97, 98), (98, 99)]),
            canonical(&[(98, 99), (97, 98)])
        );
        assert_ne!(
            reference(&[b"abc".to_vec()], &[(97, 98), (98, 99)]),
            reference(&[b"abc".to_vec()], &[(98, 99), (97, 98)])
        );
    }

    #[test]
    fn beam_is_deterministic_and_resume_preserves_fields() {
        let fields = vec![
            b"aaaaa ababc abc def def def abc ababc\n".repeat(12),
            vec![],
        ];
        let options = Options {
            rules: 4,
            width: 3,
            branch: 2,
            minimum_count: 2,
            workers: 1,
        };
        let first = discover(&fields, options, |_| {}).unwrap();
        let second = discover(
            &fields,
            Options {
                workers: 4,
                ..options
            },
            |_| {},
        )
        .unwrap();
        assert_eq!(first.beam, second.beam);
        assert_eq!(first.greedy, second.greedy);
        assert_eq!(first.beam.len(), 3);
        for prefix in first.beam.iter().chain([&first.greedy]) {
            assert_eq!(prefix.pairs.len(), 4);
            let output = continue_training(
                prefix,
                &fields,
                &Config {
                    buffer_bytes: 1 << 20,
                    minimum_count: 2,
                    score: Model::WeightedPmi,
                    ..Default::default()
                },
                2,
                20,
                |_| {},
            )
            .unwrap();
            assert_eq!(output.training.compressed.decode().unwrap(), fields);
            assert!(output.training.compressed.grammar.rules.len() <= 20);
        }
    }

    #[test]
    fn canonical_histories_preserve_actual_token_boundaries() {
        // Deterministic varied byte data, including overlapping triples, runs,
        // empty fields and rules which consume earlier rules or EOF.
        let mut seed = 713u64;
        for _ in 0..40 {
            let fields: Vec<Vec<u8>> = (0..7)
                .map(|_| {
                    (0..100)
                        .map(|_| {
                            seed ^= seed << 13;
                            seed ^= seed >> 7;
                            seed ^= seed << 17;
                            b'a' + (seed % 5) as u8
                        })
                        .collect()
                })
                .chain([vec![]])
                .collect();
            let mut pairs = Vec::new();
            for _ in 0..MAX_RULES {
                let data = reference(&fields, &pairs);
                let grammar = Prefix {
                    pairs: pairs.clone(),
                    symbols: data.len() as u64,
                }
                .grammar()
                .unwrap();
                let mut eligible = std::collections::BTreeSet::new();
                for pair in data.windows(2) {
                    if !grammar.ends_field(pair[0]) {
                        eligible.insert((pair[0], pair[1]));
                    }
                }
                if eligible.is_empty() {
                    break;
                }
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                pairs.push(*eligible.iter().nth(seed as usize % eligible.len()).unwrap());
            }
            let signature = |pairs: &[Pair]| {
                let data = reference(&fields, pairs);
                let grammar = Prefix {
                    pairs: pairs.to_vec(),
                    symbols: data.len() as u64,
                }
                .grammar()
                .unwrap();
                data.into_iter()
                    .map(|symbol| {
                        let mut terminals = Vec::new();
                        grammar.visit_terminals(symbol, |s| terminals.push(s));
                        terminals
                    })
                    .collect::<Vec<_>>()
            };
            assert_eq!(signature(&pairs), signature(&canonical(&pairs)));
        }
    }

    #[test]
    fn empty_prefix_continuation_matches_ordinary_trainer() {
        let fields = vec![b"aabcabcaaaaabcbcbc\n".repeat(100), vec![]];
        let config = Config {
            buffer_bytes: 1 << 20,
            minimum_count: 2,
            score: Model::WeightedPmi,
            ..Default::default()
        };
        let prefix = Prefix {
            pairs: vec![],
            symbols: fields.iter().map(|f| f.len() as u64 + 1).sum(),
        };
        let resumed = continue_training(&prefix, &fields, &config, 2, 23, |_| {})
            .unwrap()
            .training
            .compressed;
        let ordinary = field_engine::compress_with_options(
            &fields,
            &config,
            2,
            64,
            65536,
            field_engine::Options {
                max_rules: Some(23),
                ..Default::default()
            },
            |_| {},
        )
        .unwrap()
        .training
        .compressed;
        assert_eq!(resumed.token_count(), ordinary.token_count());
        assert_eq!(
            resumed
                .grammar
                .rules
                .iter()
                .map(|r| (r.left, r.right))
                .collect::<Vec<_>>(),
            ordinary
                .grammar
                .rules
                .iter()
                .map(|r| (r.left, r.right))
                .collect::<Vec<_>>()
        );
        assert_eq!(
            crate::archive::encode(&resumed).unwrap(),
            crate::archive::encode(&ordinary).unwrap()
        );
    }

    #[test]
    fn finished_beam_does_not_stop_the_greedy_control_early() {
        let fields = vec![b"bcaabcbcaabbacacbaaacacacaababbacacbbaabcabaa".to_vec()];
        let result = discover(
            &fields,
            Options {
                width: 1,
                branch: 1,
                minimum_count: 2,
                ..Default::default()
            },
            |_| {},
        )
        .unwrap();
        assert_eq!(result.greedy.pairs.len(), 9);
        assert!(result.beam[0].pairs.len() < result.greedy.pairs.len());
    }
}
