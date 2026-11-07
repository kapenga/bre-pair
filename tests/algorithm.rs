use std::collections::HashMap;

use bre_pair::archive;
use bre_pair::batch::{self, PairCounter};
use bre_pair::engine::{compress, Config};
use bre_pair::grammar::{Grammar, Symbol, TokenTrie, EOF};
use bre_pair::packed::PackedSymbols;

fn packed(symbols: &[Symbol], width: u8) -> PackedSymbols {
    let mut buffer = PackedSymbols::new(width);
    for &symbol in symbols {
        buffer.push(symbol);
    }
    buffer
}

#[test]
fn packed_symbols_cross_word_and_byte_boundaries() {
    for width in [9, 10, 13, 14, 17, 31, 32] {
        let mask = (1u64 << width) - 1;
        let values: Vec<_> = (0..1000u64)
            .map(|i| ((i * 1_234_567) & mask) as u32)
            .collect();
        let buffer = packed(&values, width);
        assert_eq!(buffer.iter().collect::<Vec<_>>(), values);
        let restored = PackedSymbols::from_bytes(width, values.len(), &buffer.to_bytes()).unwrap();
        assert_eq!(restored.iter().collect::<Vec<_>>(), values);
    }
}

#[test]
fn pair_counts_handle_runs_without_skipping_other_pairs() {
    let grammar = Grammar::default();
    let buffer = packed(&[97, 97, 97, 97, 97, 98, EOF, 97, 98, EOF], 9);
    let counter = PairCounter::from_buffer(&buffer, &grammar);
    assert_eq!(counter.counts[&(97, 97)], 2);
    assert_eq!(counter.counts[&(97, 98)], 2);
    assert_eq!(counter.counts[&(98, EOF)], 2);
    assert!(!counter.counts.contains_key(&(EOF, 97)));
    let mut frequency_only = PairCounter::new(false);
    for symbol in buffer.iter() {
        frequency_only.push(symbol, &grammar);
    }
    assert_eq!(frequency_only.counts, counter.counts);
    assert!(frequency_only.self_overlaps.is_empty());
}

#[test]
fn excluded_higher_frequency_pair_still_blocks_a_lower_pair() {
    let buffer = packed(&[1, 2, 3, EOF, 2, 3, 4, EOF], 9);
    let counts = HashMap::from([((1, 2), 10), ((2, 3), 7), ((3, 4), 4)]);
    let selection = batch::select(&buffer, &counts, 4);
    assert_eq!(
        selection.pairs.iter().map(|c| c.pair).collect::<Vec<_>>(),
        [(1, 2)]
    );
}

#[test]
fn shared_symbols_without_an_observed_overlap_are_compatible() {
    let buffer = packed(&[1, 2, EOF, 2, 3, EOF], 9);
    let counts = HashMap::from([((1, 2), 5), ((2, 3), 4)]);
    let selection = batch::select(&buffer, &counts, 4);
    assert_eq!(selection.pairs.len(), 2);
}

#[test]
fn tie_selection_and_rewrite_do_not_cascade() {
    let mut grammar = Grammar::default();
    let buffer = packed(&[1, 2, 3, 1, 2, 3, EOF], 9);
    let counter = PairCounter::from_buffer(&buffer, &grammar);
    let selection = batch::select(&buffer, &counter.counts, 2);
    assert_eq!(selection.tie_edges, 1);
    assert_eq!(selection.pairs.len(), 1);
    let mut replacements = HashMap::new();
    for candidate in selection.pairs {
        replacements.insert(candidate.pair, grammar.add(candidate.pair).unwrap());
    }
    let (output, counted) = batch::rewrite(&buffer, &replacements, &grammar, 9);
    assert_eq!(output.len(), 5);
    assert_eq!(
        counted.counts,
        PairCounter::from_buffer(&output, &grammar).counts
    );
}

#[test]
fn a_self_pair_has_no_self_conflict() {
    let grammar = Grammar::default();
    let buffer = packed(&[97, 97, 97, 97, 97], 9);
    let counter = PairCounter::from_buffer(&buffer, &grammar);
    let selection = batch::select(&buffer, &counter.counts, 2);
    assert_eq!(selection.pairs.len(), 1);
    assert_eq!(selection.pairs[0].count, 2);
}

#[test]
fn absorbed_eof_stays_a_field_boundary() {
    let mut grammar = Grammar::default();
    let ending = grammar.add((97, EOF)).unwrap();
    let longer = grammar.add((98, ending)).unwrap();
    assert!(grammar.ends_field(longer));
    assert!(grammar.add((ending, 99)).is_err());
    assert!(grammar.add((EOF, 99)).is_err());
    let buffer = packed(&[ending, 98, ending], 9);
    let counts = PairCounter::from_buffer(&buffer, &grammar).counts;
    assert!(!counts.contains_key(&(ending, 98)));
    assert_eq!(counts[&(98, ending)], 1);
}

#[test]
fn segmentation_is_exact_when_longest_match_is_suboptimal() {
    let mut grammar = Grammar::default();
    grammar.add((b'a' as u32, b'b' as u32)).unwrap();
    let bc = grammar.add((b'b' as u32, b'c' as u32)).unwrap();
    let bcd = grammar.add((bc, b'd' as u32)).unwrap();
    let bcde = grammar.add((bcd, b'e' as u32)).unwrap();
    let tokens = TokenTrie::new(&grammar).segment(b"abcde");
    assert_eq!(tokens, [b'a' as u32, bcde, EOF]);
}

#[test]
fn exact_segmentation_matches_an_independent_shortest_path_oracle() {
    let mut grammar = Grammar::default();
    let ab = grammar.add((0, 1)).unwrap();
    let ba = grammar.add((1, 0)).unwrap();
    grammar.add((ab, 0)).unwrap();
    grammar.add((ba, 1)).unwrap();
    grammar.add((1, EOF)).unwrap();
    let mut dictionary = Vec::new();
    for symbol in 0..grammar.alphabet_size() as u32 {
        let mut expansion = Vec::new();
        grammar.visit_terminals(symbol, |s| expansion.push(s));
        dictionary.push(expansion);
    }
    let trie = TokenTrie::new(&grammar);
    for pattern in 0..256 {
        let bytes: Vec<_> = (0..8).map(|bit| ((pattern >> bit) & 1) as u8).collect();
        let terminals: Vec<_> = bytes.iter().map(|&b| b as u32).chain([EOF]).collect();
        // Forward shortest paths enumerate all dictionary matches independently.
        let mut distance = vec![usize::MAX; terminals.len() + 1];
        distance[0] = 0;
        for start in 0..terminals.len() {
            for expansion in &dictionary {
                if terminals[start..].starts_with(expansion) {
                    let end = start + expansion.len();
                    distance[end] = distance[end].min(distance[start] + 1);
                }
            }
        }
        let parsed = trie.segment(&bytes);
        assert_eq!(parsed.len(), distance[terminals.len()]);
        for prefix in 0..=parsed.len() {
            assert_eq!(trie.segment_prefix(&bytes, prefix), parsed[..prefix]);
        }
    }
}

#[test]
fn exhaustive_small_batches_have_additive_non_overlapping_replacements() {
    let grammar = Grammar::default();
    for pattern in 0..3usize.pow(8) {
        let mut remaining = pattern;
        let symbols: Vec<_> = (0..8)
            .map(|_| {
                let symbol = (remaining % 3) as u32;
                remaining /= 3;
                symbol
            })
            .collect();
        let buffer = packed(&symbols, 9);
        let counts = PairCounter::from_buffer(&buffer, &grammar).counts;
        // Count each possible pair with a separate naive left-to-right scan.
        for a in 0..3 {
            for b in 0..3 {
                let mut count = 0;
                let mut at = 0;
                while at + 1 < symbols.len() {
                    if symbols[at] == a && symbols[at + 1] == b {
                        count += 1;
                        at += 2;
                    } else {
                        at += 1;
                    }
                }
                assert_eq!(counts.get(&(a, b)).copied().unwrap_or(0), count);
            }
        }
        let selection = batch::select(&buffer, &counts, 2);
        for triple in symbols.windows(3) {
            let p = (triple[0], triple[1]);
            let q = (triple[1], triple[2]);
            if p == q {
                continue;
            }
            let selected_p = selection.pairs.iter().any(|c| c.pair == p);
            let selected_q = selection.pairs.iter().any(|c| c.pair == q);
            assert!(!(selected_p && selected_q));
            if selected_p {
                assert!(counts[&p] >= counts[&q]);
            }
            if selected_q {
                assert!(counts[&q] >= counts[&p]);
            }
        }
        let expected: usize = selection.pairs.iter().map(|c| c.count as usize).sum();
        let mut updated_grammar = Grammar::default();
        let replacements: HashMap<_, _> = selection
            .pairs
            .iter()
            .map(|c| (c.pair, updated_grammar.add(c.pair).unwrap()))
            .collect();
        let (rewritten, _) = batch::rewrite(&buffer, &replacements, &updated_grammar, 9);
        assert_eq!(buffer.len() - rewritten.len(), expected);
    }
}

#[test]
fn round_trip_preserves_bytes_empty_fields_and_small_buffer_refills() {
    let cases = [
        vec![],
        vec![vec![]],
        vec![vec![], vec![]],
        vec![b"aaaaabaaaaabaaaaab".to_vec()],
        vec![
            b"the quick brown fox\n".repeat(50),
            vec![],
            b"the brown fox".repeat(20),
        ],
        vec![(0..=255).cycle().take(3000).map(|b| b as u8).collect()],
    ];
    for fields in cases {
        for buffer_bytes in [8, 32, 128, 4096] {
            for minimum_count in [2, 4] {
                let training = compress(
                    &fields,
                    &Config {
                        buffer_bytes,
                        minimum_count,
                        ..Config::default()
                    },
                    |_| {},
                )
                .unwrap();
                assert!(training.max_buffer_bytes <= buffer_bytes);
                let serialized = archive::encode(&training.compressed).unwrap();
                let restored = archive::decode(&serialized).unwrap().decode().unwrap();
                assert_eq!(
                    restored, fields,
                    "buffer={buffer_bytes}, min={minimum_count}"
                );
            }
        }
    }
}

#[test]
fn deterministic_generated_inputs_preserve_replacement_count_invariants() {
    let mut random = 42u64;
    for case in 0..24 {
        let mut bytes = Vec::new();
        for _ in 0..(200 + case * 19) {
            random ^= random << 13;
            random ^= random >> 7;
            random ^= random << 17;
            bytes.push((random % 5) as u8);
        }
        let fields = vec![bytes];
        let config = Config {
            buffer_bytes: if case % 2 == 0 { 64 } else { 1024 },
            minimum_count: 2,
            ..Config::default()
        };
        let training = compress(&fields, &config, |_| {}).unwrap();
        for round in &training.rounds {
            assert_eq!(round.before - round.rewritten, round.removed);
        }
        assert_eq!(training.compressed.decode().unwrap(), fields);
    }
}

#[test]
fn archive_rejects_truncation_and_invalid_dictionary_references() {
    let fields = vec![b"banana banana banana banana".to_vec()];
    let training = compress(&fields, &Config::default(), |_| {}).unwrap();
    let bytes = archive::encode(&training.compressed).unwrap();
    for length in 0..bytes.len() {
        assert!(archive::decode(&bytes[..length]).is_err());
    }
    let mut invalid = bytes.clone();
    // Rule payload begins at byte 40. Its first 9-bit child cannot refer to itself.
    invalid[40] = 255;
    invalid[41] |= 1;
    assert!(archive::decode(&invalid).is_err());
}

#[test]
fn output_is_reproducible_despite_hash_iteration_order() {
    let fields = vec![b"abc cab bac aba bcb caca ".repeat(100)];
    let config = Config {
        buffer_bytes: 512,
        minimum_count: 4,
        ..Config::default()
    };
    let a = compress(&fields, &config, |_| {}).unwrap();
    let b = compress(&fields, &config, |_| {}).unwrap();
    assert_eq!(
        archive::encode(&a.compressed).unwrap(),
        archive::encode(&b.compressed).unwrap()
    );
}

#[test]
fn all_selectors_and_tie_orders_preserve_constraints_and_round_trips() {
    use bre_pair::search::{Options, Strategy, TieBreak};
    let fields = vec![
        b"abc cab bac aba bcb caca ".repeat(30),
        vec![],
        b"aaaaab".repeat(30),
    ];
    for strategy in [Strategy::Greedy, Strategy::Beam, Strategy::Hybrid] {
        for tie_break in [TieBreak::Lex, TieBreak::Reverse, TieBreak::Seeded] {
            let config = Config {
                buffer_bytes: 512,
                minimum_count: 2,
                search: Options {
                    strategy,
                    tie_break,
                    seed: 42,
                    ..Options::default()
                },
                ..Config::default()
            };
            let result = compress(&fields, &config, |_| {}).unwrap();
            for round in &result.rounds {
                assert!(round.removed as u64 >= round.greedy_score);
                assert!(round.removed as u64 <= round.score_upper_bound);
                assert_eq!(
                    round.proven_optimal,
                    round.removed as u64 == round.score_upper_bound
                );
            }
            let encoded = archive::encode(&result.compressed).unwrap();
            assert_eq!(archive::decode(&encoded).unwrap().decode().unwrap(), fields);
        }
    }
}

#[test]
fn tie_breaking_changes_grammar_without_changing_decoded_content() {
    use bre_pair::search::{Options, TieBreak};
    let fields = vec![b"abc".repeat(20)];
    let a = compress(&fields, &Config::default(), |_| {}).unwrap();
    let alternate = Config {
        search: Options {
            tie_break: TieBreak::Reverse,
            ..Options::default()
        },
        ..Config::default()
    };
    let b = compress(&fields, &alternate, |_| {}).unwrap();
    assert_ne!(
        archive::encode(&a.compressed).unwrap(),
        archive::encode(&b.compressed).unwrap()
    );
    assert_eq!(a.compressed.decode().unwrap(), fields);
    assert_eq!(b.compressed.decode().unwrap(), fields);
    let seeded = Config {
        search: Options {
            tie_break: TieBreak::Seeded,
            seed: 23,
            ..Options::default()
        },
        ..Config::default()
    };
    let x = compress(&fields, &seeded, |_| {}).unwrap();
    let y = compress(&fields, &seeded, |_| {}).unwrap();
    assert_eq!(
        archive::encode(&x.compressed).unwrap(),
        archive::encode(&y.compressed).unwrap()
    );
}

#[test]
fn association_priority_can_override_frequency_while_count_bounds_stay_valid() {
    use bre_pair::scoring::{Model, Statistics};
    use bre_pair::search::Options;
    let buffer = packed(&[1, 2, 3], 9);
    let counts = HashMap::from([((1, 2), 100), ((2, 3), 4)]);
    let all_counts = batch::Counts::from_iter([
        ((1, 2), 100),
        ((2, 3), 4),
        ((1, 4), 9900),
        ((2, 5), 96),
        ((6, 3), 96),
        ((7, 8), 89804),
    ]);
    let overlaps = HashMap::new();
    let statistics = Statistics::from_counts(&all_counts, &overlaps, 9);
    for model in [Model::Conditional, Model::Pmi] {
        let selected = batch::select_scored(
            &buffer,
            &counts,
            4,
            &Options::default(),
            Some((model, &statistics)),
        );
        assert_eq!(selected.pairs[0].pair, (2, 3));
        assert_eq!(selected.score, 4);
        assert!(selected.proven_optimal);
    }
    let selected = batch::select_scored(
        &buffer,
        &counts,
        4,
        &Options::default(),
        Some((Model::WeightedPmi, &statistics)),
    );
    assert_eq!(selected.pairs[0].pair, (1, 2));
}

#[test]
fn all_score_models_preserve_fields_runs_and_refill() {
    use bre_pair::scoring::Model;
    let fields = vec![
        b"the quick queen quacks\n".repeat(12),
        b"aaaaab".repeat(10),
        vec![],
    ];
    for score in [
        Model::Frequency,
        Model::Conditional,
        Model::Pmi,
        Model::WeightedPmi,
        Model::Likelihood,
    ] {
        for buffer_bytes in [128, 4096] {
            let config = Config {
                buffer_bytes,
                minimum_count: 2,
                score,
                ..Config::default()
            };
            let result = compress(&fields, &config, |_| {}).unwrap();
            assert_eq!(
                archive::decode(&archive::encode(&result.compressed).unwrap())
                    .unwrap()
                    .decode()
                    .unwrap(),
                fields
            );
            for round in &result.rounds {
                assert!(round.score_upper_bound >= round.removed as u64);
            }
        }
    }
}

#[test]
fn skipping_greedy_certificates_preserves_every_archive_byte() {
    use bre_pair::scoring::Model;
    use bre_pair::search::TieBreak;
    let fields = vec![
        b"the quick queen quacks\naaaaab abc cab bac ".repeat(12),
        vec![],
    ];
    for score in [
        Model::Frequency,
        Model::Conditional,
        Model::Pmi,
        Model::WeightedPmi,
        Model::Likelihood,
    ] {
        for tie_break in [TieBreak::Lex, TieBreak::Reverse, TieBreak::Seeded] {
            for buffer_bytes in [128, 4096] {
                let mut config = Config {
                    buffer_bytes,
                    minimum_count: 2,
                    score,
                    ..Config::default()
                };
                config.search.tie_break = tie_break;
                let checked = compress(&fields, &config, |_| {}).unwrap();
                config.search.certify = false;
                let fast = compress(&fields, &config, |_| {}).unwrap();
                assert_eq!(
                    archive::encode(&checked.compressed).unwrap(),
                    archive::encode(&fast.compressed).unwrap()
                );
                assert!(fast.rounds.iter().all(|r| !r.proven_optimal));
                assert_eq!(fast.compressed.decode().unwrap(), fields);
            }
        }
    }
}

#[test]
fn pmi_exponent_preserves_round_trips_and_its_zero_and_one_controls() {
    use bre_pair::scoring::Model;
    let fields = vec![
        b"the quick queen quacks\naaaaab abc cab bac ".repeat(12),
        vec![],
    ];
    for buffer_bytes in [128, 4096] {
        let mut config = Config {
            buffer_bytes,
            minimum_count: 2,
            score: Model::WeightedPmi,
            ..Config::default()
        };
        let original = compress(&fields, &config, |_| {}).unwrap();
        let original_bytes = archive::encode(&original.compressed).unwrap();
        for alpha in [0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 4.0] {
            config.pmi_alpha = alpha;
            let result = compress(&fields, &config, |_| {}).unwrap();
            let bytes = archive::encode(&result.compressed).unwrap();
            assert_eq!(archive::decode(&bytes).unwrap().decode().unwrap(), fields);
            if alpha == 1.0 {
                assert_eq!(bytes, original_bytes);
            }
            if alpha == 0.0 {
                let control = compress(
                    &fields,
                    &Config {
                        score: Model::Pmi,
                        pmi_alpha: 1.0,
                        ..config.clone()
                    },
                    |_| {},
                )
                .unwrap();
                assert_eq!(bytes, archive::encode(&control.compressed).unwrap());
            }
        }
    }
}

#[test]
fn invalid_or_inapplicable_pmi_exponents_are_rejected() {
    use bre_pair::scoring::Model;
    for alpha in [-0.1, f64::NAN, f64::INFINITY, 4.1] {
        let config = Config {
            score: Model::WeightedPmi,
            pmi_alpha: alpha,
            ..Config::default()
        };
        assert!(compress(&[vec![0; 10]], &config, |_| {}).is_err());
    }
    assert!(compress(
        &[vec![]],
        &Config {
            pmi_alpha: 0.5,
            ..Config::default()
        },
        |_| {}
    )
    .is_err());
}
