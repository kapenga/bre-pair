use bre_pair::{
    archive,
    batch::Counting,
    engine::{Compressed, Config},
    field_engine,
    packed::{symbol_width, PackedSymbols},
    scoring::Model,
};

fn canonical(source: &Compressed) -> Vec<u8> {
    let mut block = PackedSymbols::new(symbol_width(source.grammar.alphabet_size()));
    for part in &source.blocks {
        for symbol in part.iter() {
            block.push(symbol);
        }
    }
    archive::encode(&Compressed {
        grammar: source.grammar.clone(),
        blocks: vec![block],
        input_bytes: source.input_bytes,
        field_count: source.field_count,
    })
    .unwrap()
}

#[test]
fn hard_rule_budgets_preserve_prefixes_counts_fields_and_unlimited_control() {
    let fields: Vec<_> = (0..64)
        .map(|i| format!("aaaaa aabcdaafgcdaacgadaaee the queen and the king {i}\n").into_bytes())
        .collect();
    for score in [Model::Frequency, Model::WeightedPmi] {
        let mut config = Config {
            minimum_count: 2,
            score,
            ..Config::default()
        };
        let full = field_engine::compress(&fields, &config, 1, 8, 64, |_| {}).unwrap();
        assert!(full.training.rounds[0].selected > 3);
        let rule_count = full.training.compressed.grammar.rules.len();
        let expected = canonical(&full.training.compressed);
        for limit in [1, 3, 7, 31, rule_count + 1] {
            for counting in [Counting::Full, Counting::Incremental] {
                config.pair_counting = counting;
                for workers in [1, 4] {
                    for cutoff in [false, true] {
                        let result = field_engine::compress_with_options(
                            &fields,
                            &config,
                            workers,
                            8,
                            64,
                            field_engine::Options {
                                max_rules: Some(limit),
                                frequency_cutoff: cutoff,
                                ..Default::default()
                            },
                            |_| {},
                        )
                        .unwrap();
                        let source = &result.training.compressed;
                        assert_eq!(source.decode().unwrap(), fields);
                        assert_eq!(source.grammar.rules.len(), limit.min(rule_count));
                        assert_eq!(result.stopped_at_rule_limit, limit <= rule_count);
                        assert!(!result.stopped_at_round_limit);
                        for (a, b) in source
                            .grammar
                            .rules
                            .iter()
                            .zip(&full.training.compressed.grammar.rules)
                        {
                            assert_eq!((a.left, a.right), (b.left, b.right));
                        }
                        for round in &result.training.rounds {
                            assert_eq!(round.before - round.rewritten, round.removed);
                            assert!(round.total_rules <= limit);
                        }
                        if limit > rule_count {
                            assert_eq!(canonical(source), expected);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn zero_budget_is_rejected_and_simultaneous_limits_are_reported() {
    let fields = vec![b"abc abc abc abc abc abc".to_vec()];
    let config = Config::default();
    assert!(field_engine::compress_with_options(
        &fields,
        &config,
        1,
        2,
        64,
        field_engine::Options {
            max_rules: Some(0),
            ..Default::default()
        },
        |_| {}
    )
    .is_err());
    let result = field_engine::compress_with_options(
        &fields,
        &config,
        1,
        2,
        64,
        field_engine::Options {
            max_rules: Some(1),
            max_rounds: Some(1),
            ..Default::default()
        },
        |_| {},
    )
    .unwrap();
    assert!(result.stopped_at_rule_limit && result.stopped_at_round_limit);
    assert_eq!(result.training.compressed.grammar.rules.len(), 1);
    assert_eq!(result.training.compressed.decode().unwrap(), fields);
}
