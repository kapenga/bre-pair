use bre_pair::archive;
use bre_pair::engine::{self, Compressed, Config};
use bre_pair::grammar::{Grammar, Symbol, EOF};
use bre_pair::packed::{symbol_width, PackedSymbols};
use bre_pair::prune::{self, Checkpoint};
use bre_pair::scoring::Model;
use bre_pair::sequence_archive;

fn source(grammar: Grammar, symbols: &[Symbol]) -> Compressed {
    let mut block = PackedSymbols::new(symbol_width(grammar.alphabet_size()));
    for &symbol in symbols {
        block.push(symbol);
    }
    let input_bytes = symbols
        .iter()
        .map(|&s| grammar.terminals(s) - usize::from(grammar.ends_field(s)))
        .sum::<usize>() as u64;
    let field_count = symbols.iter().filter(|&&s| grammar.ends_field(s)).count() as u64;
    Compressed {
        grammar,
        blocks: vec![block],
        input_bytes,
        field_count,
    }
}

#[test]
fn post_pruning_inlines_parents_and_expands_direct_occurrences() {
    let mut g = Grammar::default();
    let ab = g.add((97, 98)).unwrap();
    let abc = g.add((ab, 99)).unwrap();
    let abd = g.add((ab, 100)).unwrap();
    let original = source(g, &[abc, abc, abd, abd, ab, EOF]);
    let plan = prune::plan(&original.grammar, &original.blocks, 1).unwrap();
    assert_eq!(plan.retired, [true, false, false]);
    assert_eq!(plan.added_tokens, 1);
    let compact = prune::export(&original, &plan.retired).unwrap();
    assert_eq!(compact.rules, [vec![97, 98, 99], vec![97, 98, 100]]);
    assert_eq!(compact.token_count(), original.token_count() + 1);
    let encoded = sequence_archive::encode(&compact).unwrap();
    assert_eq!(
        sequence_archive::decode(&encoded)
            .unwrap()
            .to_binary()
            .unwrap()
            .decode()
            .unwrap(),
        original.decode().unwrap()
    );
}

#[test]
fn zero_direct_use_pruning_preserves_token_coverage() {
    let mut g = Grammar::default();
    let ab = g.add((97, 98)).unwrap();
    let abc = g.add((ab, 99)).unwrap();
    let original = source(g, &[abc, abc, abc, abc, EOF]);
    let plan = prune::plan(&original.grammar, &original.blocks, 0).unwrap();
    assert_eq!(plan.selected_rules, 1);
    assert_eq!(plan.added_tokens, 0);
    let compact = prune::export(&original, &plan.retired).unwrap();
    assert_eq!(compact.token_count(), original.token_count());
    assert_eq!(compact.rules, [vec![97, 98, 99]]);
    assert_eq!(
        compact.to_binary().unwrap().decode().unwrap(),
        original.decode().unwrap()
    );
}

#[test]
fn cost_pruning_keeps_shared_dictionary_rules_and_removes_profitable_rare_rules() {
    let mut g = Grammar::default();
    let shared = g.add((97, 98)).unwrap();
    let rare = g.add((120, 121)).unwrap();
    let parents: Vec<_> = (1..10).map(|b| g.add((shared, b)).unwrap()).collect();
    let mut data = vec![rare];
    for parent in parents {
        data.extend([parent; 8]);
    }
    data.push(EOF);
    let original = source(g, &data);
    let cost = prune::cost_plan(&original, 7).unwrap();
    assert!(
        !cost.plan.retired[0],
        "zero data use is not enough to justify inlining"
    );
    assert!(cost.plan.retired[1]);
    assert_eq!(cost.choice, "local-cost");
    let result = prune::export(&original, &cost.plan.retired).unwrap();
    assert_eq!(
        sequence_archive::encode(&result).unwrap().len(),
        cost.predicted_bytes
    );
    assert!(cost.predicted_bytes < cost.baseline_bytes);
    assert_eq!(
        result.to_binary().unwrap().decode().unwrap(),
        original.decode().unwrap()
    );
}

#[test]
fn cost_pruning_tries_groups_that_reduce_the_global_symbol_width() {
    let mut g = Grammar::default();
    let shared = g.add((97, 98)).unwrap();
    let mut data = Vec::new();
    for byte in 0..255 {
        let parent = g.add((shared, byte)).unwrap();
        data.extend([parent; 16]);
    }
    data.push(EOF);
    let original = source(g, &data);
    assert_eq!(original.blocks[0].width(), 10);
    let cost = prune::cost_plan(&original, 7).unwrap();
    assert_eq!(cost.choice, "threshold-group");
    assert!(cost.plan.retired[0]);
    assert_eq!(cost.plan.added_tokens, 0);
    let result = prune::export(&original, &cost.plan.retired).unwrap();
    assert_eq!(result.blocks[0].width(), 9);
    assert_eq!(
        sequence_archive::encode(&result).unwrap().len(),
        cost.predicted_bytes
    );
    assert!(cost.predicted_bytes < cost.baseline_bytes);
    assert_eq!(
        result.to_binary().unwrap().decode().unwrap(),
        original.decode().unwrap()
    );
}

#[test]
fn cost_size_predictions_match_exports_across_generated_grammars_and_multiple_blocks() {
    let mut seed = 42u64;
    for example in 0..60 {
        let fields = vec![
            (0..(300 + example))
                .map(|_| {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    b'a' + (seed % 4) as u8
                })
                .collect(),
            b"abcabcabcaaaaabbb".repeat(6),
        ];
        let training = engine::compress(
            &fields,
            &Config {
                buffer_bytes: if example % 2 == 0 { 4096 } else { 64 },
                minimum_count: 2,
                ..Config::default()
            },
            |_| {},
        )
        .unwrap();
        for cutoff in [0, 1, 3, 7] {
            let cost = prune::cost_plan(&training.compressed, cutoff).unwrap();
            let output = prune::export(&training.compressed, &cost.plan.retired).unwrap();
            assert_eq!(
                cost.predicted_bytes,
                sequence_archive::encode(&output).unwrap().len()
            );
            assert!(cost.predicted_bytes <= cost.baseline_bytes);
            assert_eq!(
                output.token_count(),
                cost.plan.before_tokens + cost.plan.added_tokens
            );
            assert_eq!(output.to_binary().unwrap().decode().unwrap(), fields);
        }
    }
}

#[test]
fn nested_pruning_preserves_absorbed_eof_and_empty_fields() {
    let mut g = Grammar::default();
    let ending = g.add((97, EOF)).unwrap();
    let ba = g.add((98, ending)).unwrap();
    let cba = g.add((99, ba)).unwrap();
    let original = source(g, &[cba, cba, cba, cba, ba, ending, EOF]);
    let plan = prune::plan(&original.grammar, &original.blocks, 1).unwrap();
    assert_eq!(plan.retired, [true, true, false]);
    assert_eq!(plan.added_tokens, 3);
    let compact = prune::export(&original, &plan.retired).unwrap();
    assert_eq!(compact.rules, [vec![99, 98, 97, EOF]]);
    assert_eq!(compact.token_count(), original.token_count() + 3);
    assert_eq!(
        sequence_archive::decode(&sequence_archive::encode(&compact).unwrap())
            .unwrap()
            .to_binary()
            .unwrap()
            .decode()
            .unwrap(),
        original.decode().unwrap()
    );
}

#[test]
fn every_retirement_mask_preserves_a_small_nested_grammar() {
    let mut g = Grammar::default();
    let ab = g.add((97, 98)).unwrap();
    let abc = g.add((ab, 99)).unwrap();
    let abcd = g.add((abc, 100)).unwrap();
    let repeat = g.add((abcd, abcd)).unwrap();
    let ending = g.add((repeat, EOF)).unwrap();
    let original = source(g, &[ab, abc, abcd, repeat, ending]);
    for bits in 0..1 << original.grammar.rules.len() {
        let retired: Vec<_> = (0..original.grammar.rules.len())
            .map(|i| bits & (1 << i) != 0)
            .collect();
        let compact = prune::export(&original, &retired).unwrap();
        assert!(compact.token_count() >= original.token_count());
        assert_eq!(
            sequence_archive::decode(&sequence_archive::encode(&compact).unwrap())
                .unwrap()
                .to_binary()
                .unwrap()
                .decode()
                .unwrap(),
            original.decode().unwrap()
        );
    }
}

#[test]
fn checkpoint_pruning_continues_training_and_preserves_fields() {
    let fields = vec![
        b"abcabd abcabe abcabf the quick queen\n".repeat(20),
        vec![],
        b"aaaaaabababababa".repeat(10),
    ];
    for max_uses in [0, 1, 2, 3] {
        for after_round in [1, 3, 5] {
            let config = Config {
                buffer_bytes: 8192,
                score: Model::WeightedPmi,
                pruning: Some(Checkpoint {
                    after_round,
                    max_uses,
                }),
                ..Config::default()
            };
            let result = engine::compress(&fields, &config, |_| {}).unwrap();
            let state = result.pruning.as_ref().expect("checkpoint reached");
            assert!(state.event.after_tokens >= state.event.before_tokens);
            assert_eq!(result.compressed.decode().unwrap(), fields);
            let compact = prune::export(&result.compressed, &state.retired).unwrap();
            assert_eq!(
                sequence_archive::decode(&sequence_archive::encode(&compact).unwrap())
                    .unwrap()
                    .to_binary()
                    .unwrap()
                    .decode()
                    .unwrap(),
                fields
            );
        }
    }
}

#[test]
fn checkpoint_with_only_invisible_rules_keeps_binary_training_identical() {
    let fields = vec![b"abcdefghijklmnopq".repeat(40)];
    let config = Config {
        buffer_bytes: 8192,
        ..Config::default()
    };
    let control = engine::compress(&fields, &config, |_| {}).unwrap();
    let result = engine::compress(
        &fields,
        &Config {
            pruning: Some(Checkpoint {
                after_round: 2,
                max_uses: 0,
            }),
            ..config
        },
        |_| {},
    )
    .unwrap();
    assert!(result.pruning.as_ref().unwrap().event.selected_rules > 0);
    assert_eq!(
        archive::encode(&result.compressed).unwrap(),
        archive::encode(&control.compressed).unwrap()
    );
}

#[test]
fn during_pruning_rejects_unsupported_streaming_and_unsafe_checkpoints() {
    for checkpoint in [
        Checkpoint {
            after_round: 0,
            max_uses: 0,
        },
        Checkpoint {
            after_round: 1,
            max_uses: 4,
        },
    ] {
        assert!(engine::compress(
            &[vec![0; 100]],
            &Config {
                pruning: Some(checkpoint),
                ..Config::default()
            },
            |_| {}
        )
        .is_err());
    }
    assert!(engine::compress(
        &[vec![0; 100]],
        &Config {
            buffer_bytes: 8,
            pruning: Some(Checkpoint {
                after_round: 1,
                max_uses: 0
            }),
            ..Config::default()
        },
        |_| {}
    )
    .is_err());
}
