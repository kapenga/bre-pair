use bre_pair::{
    archive, context_entropy,
    engine::{self, Config},
    entropy_engine, field_engine,
    packed::{symbol_width, PackedSymbols},
    scoring,
};

fn canonical(source: &engine::Compressed) -> Vec<u8> {
    let mut block = PackedSymbols::new(symbol_width(source.grammar.alphabet_size()));
    for part in &source.blocks {
        for symbol in part.iter() {
            block.push(symbol);
        }
    }
    archive::encode(&engine::Compressed {
        grammar: source.grammar.clone(),
        blocks: vec![block],
        input_bytes: source.input_bytes,
        field_count: source.field_count,
    })
    .unwrap()
}

#[test]
fn offline_statistics_include_later_text_and_do_not_adapt_during_annotation() {
    let options = context_entropy::Options {
        order: 3,
        minimum_context_count: 1,
        ..Default::default()
    };
    let prefix = vec![b"ab".to_vec()];
    let small = context_entropy::Model::train(&prefix, options).unwrap();
    assert_eq!(small.field_weights(b"ab")[1], 1.0);
    let fields = vec![b"ab".to_vec(), b"ac".to_vec()];
    let whole = context_entropy::Model::train(&fields, options).unwrap();
    let expected = 1.0f32 - (1.0f64 / 257f64.log2()) as f32;
    assert_eq!(whole.field_weights(b"ab")[1], expected);
    let before = whole.field_weights(b"ab");
    whole.field_weights(b"unseen text");
    assert_eq!(whole.field_weights(b"ab"), before);
    let reversed =
        context_entropy::Model::train(&fields.into_iter().rev().collect::<Vec<_>>(), options)
            .unwrap();
    for field in [b"ab".as_slice(), b"ac", b"unseen text", &[0, 0, 1], &[]] {
        assert_eq!(whole.field_weights(field), reversed.field_weights(field));
        assert!(whole
            .field_weights(field)
            .iter()
            .all(|w| w.is_finite() && (0.0..=1.0).contains(w)));
    }
}

#[test]
fn contexts_reset_at_eof_and_low_support_backs_off_without_dropping_followers() {
    let fields = vec![
        b"ab".to_vec(),
        b"ac".to_vec(),
        b"ab".to_vec(),
        b"ac".to_vec(),
    ];
    let options = context_entropy::Options {
        order: 3,
        minimum_context_count: 3,
        ..Default::default()
    };
    let model = context_entropy::Model::train(&fields, options).unwrap();
    let root = context_entropy::Model::train(
        &fields,
        context_entropy::Options {
            order: 0,
            ..options
        },
    )
    .unwrap();
    assert_eq!(model.field_weights(b"ab")[0], 0.0);
    assert_eq!(
        model.field_weights(b"ab")[1],
        1.0 - (1.0 / 257f64.log2()) as f32
    );
    // Contexts "ab" and "b" each occur twice, below the threshold.
    assert_eq!(model.field_weights(b"ab")[2], root.field_weights(b"ab")[2]);
    assert_eq!(model.field_weights(b"z")[1], root.field_weights(b"z")[1]);
    let repeated: [Vec<u8>; 4] = std::array::from_fn(|_| b"a".to_vec());
    let deterministic = context_entropy::Model::train(
        &repeated,
        context_entropy::Options {
            order: 1,
            minimum_context_count: 1,
            ..options
        },
    )
    .unwrap();
    assert_eq!(deterministic.field_weights(b"a"), vec![0.0, 1.0]);
    assert_eq!(deterministic.statistics.contexts_by_order[1], 1); // Only "a", never EOF.
    for invalid in [
        context_entropy::Options {
            order: 7,
            ..options
        },
        context_entropy::Options {
            minimum_context_count: 0,
            ..options
        },
        context_entropy::Options {
            maximum_ngrams: 1,
            ..options
        },
    ] {
        assert!(context_entropy::Model::train(&fields, invalid).is_err());
    }
}

#[test]
fn conditional_entropies_match_an_independent_suffix_counter() {
    use std::collections::HashMap;
    let fields = vec![
        vec![],
        vec![0, 0, 1],
        vec![0, 1, 0],
        b"abab".to_vec(),
        b"acac".to_vec(),
        b"aaaaaa".to_vec(),
        vec![255, 0, 255],
        b"ab".to_vec(),
    ];
    for order in 0..=6 {
        let mut oracle: HashMap<Vec<u8>, [u64; 257]> = HashMap::new();
        for field in &fields {
            for i in 0..=field.len() {
                let symbol = field.get(i).copied().map_or(256, usize::from);
                for length in 0..=i.min(order) {
                    oracle
                        .entry(field[i - length..i].to_vec())
                        .or_insert([0; 257])[symbol] += 1;
                }
            }
        }
        for minimum in [1, 3] {
            let model = context_entropy::Model::train(
                &fields,
                context_entropy::Options {
                    order,
                    minimum_context_count: minimum,
                    ..Default::default()
                },
            )
            .unwrap();
            for field in &fields {
                let actual = model.field_weights(field);
                assert_eq!(actual[0], 0.0);
                for (i, &actual) in actual.iter().enumerate().skip(1) {
                    let distribution = (0..=i.min(order))
                        .rev()
                        .find_map(|length| {
                            let counts = oracle.get(&field[i - length..i]).unwrap();
                            (length == 0 || counts.iter().sum::<u64>() >= minimum).then_some(counts)
                        })
                        .unwrap();
                    let total = distribution.iter().sum::<u64>();
                    let entropy = distribution
                        .iter()
                        .filter(|&&n| n != 0)
                        .map(|&n| {
                            let p = n as f64 / total as f64;
                            -p * p.log2()
                        })
                        .sum::<f64>();
                    assert_eq!(actual, 1.0f32 - (entropy / 257f64.log2()) as f32);
                }
            }
        }
    }
}

#[test]
fn order_zero_is_the_frequency_control_and_all_workers_reproduce_it() {
    let fields: Vec<_> = (0..48)
        .map(|i| {
            if i % 9 == 0 {
                vec![]
            } else {
                format!("aaaaa aabcdaafgcdaacgadaaee abcabcabc {i}\n").into_bytes()
            }
        })
        .collect();
    let config = Config {
        score: scoring::Model::Frequency,
        minimum_count: 2,
        buffer_bytes: 1048576,
        ..Default::default()
    };
    let reference = field_engine::compress(&fields, &config, 1, 64, 32, |_| {}).unwrap();
    let expected = canonical(&reference.training.compressed);
    for workers in [1, 4] {
        let result = entropy_engine::compress(
            &fields,
            &entropy_engine::Options {
                context: context_entropy::Options {
                    order: 0,
                    minimum_context_count: 1,
                    ..Default::default()
                },
                minimum_count: 2,
                workers,
                chunk_symbols: 32,
                ..Default::default()
            },
            |_| {},
        )
        .unwrap();
        assert_eq!(canonical(&result.training.compressed), expected);
        assert_eq!(result.training.compressed.decode().unwrap(), fields);
    }
}

#[test]
fn weighted_merges_preserve_gaps_runs_eof_and_results_across_workers_and_chunks() {
    let mut fields: Vec<_> = (0..48)
        .map(|i| format!("the queen qu qux aaaaa aabcdaafgcdaacgadaaee {i}\n").into_bytes())
        .collect();
    fields.extend([
        vec![],
        vec![0, 0, 0, 255, 0],
        vec![255; 65],
        b"abcdefghijk".to_vec(),
    ]);
    let mut expected = None;
    for order in [1, 3, 5] {
        for (workers, chunk_symbols) in [(1, 17), (2, 17), (4, 17), (8, 17), (4, 500)] {
            let result = entropy_engine::compress(
                &fields,
                &entropy_engine::Options {
                    context: context_entropy::Options {
                        order,
                        minimum_context_count: 2,
                        ..Default::default()
                    },
                    minimum_count: 2,
                    workers,
                    chunk_symbols,
                    ..Default::default()
                },
                |_| {},
            )
            .unwrap();
            assert_eq!(
                result.initial_weight_bytes,
                4 * (fields.iter().map(Vec::len).sum::<usize>() + fields.len())
            );
            assert_eq!(
                result.initial_weight_capacity_bytes,
                result.initial_weight_bytes
            );
            assert_eq!(
                result.final_weight_bytes,
                4 * result.training.compressed.token_count()
            );
            assert_eq!(result.training.compressed.decode().unwrap(), fields);
            let encoded = canonical(&result.training.compressed);
            assert_eq!(archive::decode(&encoded).unwrap().decode().unwrap(), fields);
            if workers == 1 {
                expected = Some(encoded);
            } else {
                assert_eq!(
                    expected.as_ref().unwrap(),
                    &encoded,
                    "order={order} workers={workers} chunk={chunk_symbols}"
                );
            }
        }
    }
}

#[test]
fn boundary_training_limits_are_explicit_and_terminal_batches_remain_valid() {
    let fields: Vec<_> = (0..32)
        .map(|i| format!("aababababaaaaaaa the queen {i}\n").into_bytes())
        .collect();
    for limit in [1, 7, 29] {
        let options = entropy_engine::Options {
            minimum_count: 2,
            max_rules: Some(limit),
            chunk_symbols: 32,
            ..Default::default()
        };
        let result = entropy_engine::compress(&fields, &options, |_| {}).unwrap();
        assert_eq!(result.training.compressed.grammar.rules.len(), limit);
        assert!(result.stopped_at_rule_limit);
        for round in &result.training.rounds {
            assert!(round.greedy_score <= round.removed as u64);
            assert!(round.score_upper_bound >= round.removed as u64);
        }
        assert_eq!(result.training.compressed.decode().unwrap(), fields);
    }
    let limited = entropy_engine::compress(
        &fields,
        &entropy_engine::Options {
            max_rounds: Some(2),
            minimum_count: 2,
            ..Default::default()
        },
        |_| {},
    )
    .unwrap();
    assert_eq!(limited.training.rounds.len(), 2);
    assert!(limited.stopped_at_round_limit);
    for invalid in [
        entropy_engine::Options {
            workers: 0,
            ..Default::default()
        },
        entropy_engine::Options {
            minimum_count: 1,
            ..Default::default()
        },
        entropy_engine::Options {
            max_rules: Some(0),
            ..Default::default()
        },
        entropy_engine::Options {
            buffer_bytes: 1,
            ..Default::default()
        },
    ] {
        assert!(entropy_engine::compress(&fields, &invalid, |_| {}).is_err());
    }
    let empty = entropy_engine::compress(&[], &entropy_engine::Options::default(), |_| {}).unwrap();
    assert_eq!(
        empty.training.compressed.decode().unwrap(),
        Vec::<Vec<u8>>::new()
    );
}

#[test]
fn boundary_audits_cover_packed_width_growth_and_multiple_complete_fields() {
    let mut state = 73u64;
    let raw: Vec<_> = (0..4096)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state as u8
        })
        .collect();
    let fields = vec![raw; 8];
    let mut expected = None;
    for workers in [1, 4] {
        let result = entropy_engine::compress(
            &fields,
            &entropy_engine::Options {
                context: context_entropy::Options {
                    order: 2,
                    minimum_context_count: 2,
                    ..Default::default()
                },
                workers,
                minimum_count: 4,
                chunk_symbols: 1000,
                max_rules: Some(900),
                ..Default::default()
            },
            |_| {},
        )
        .unwrap();
        assert!(result.training.rounds.iter().any(|r| r.width >= 11));
        assert_eq!(result.training.compressed.decode().unwrap(), fields);
        let bytes = canonical(&result.training.compressed);
        if let Some(expected) = &expected {
            assert_eq!(&bytes, expected);
        } else {
            expected = Some(bytes);
        }
    }
}

#[test]
fn entropy_cli_writes_a_standard_archive_and_preserves_existing_outputs() {
    use std::{
        fs,
        process::Command,
        time::{SystemTime, UNIX_EPOCH},
    };
    let root = std::env::temp_dir().join(format!(
        "bre-pair-entropy-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir(&root).unwrap();
    let input = root.join("input");
    let output = root.join("out.brp");
    let report = root.join("out.json");
    let raw = b"aaaaa the queen\n\nqu qu qux aaaaa\n";
    fs::write(&input, raw).unwrap();
    let run = || {
        Command::new(env!("CARGO_BIN_EXE_entropy-repair"))
            .arg(&input)
            .arg(&output)
            .args([
                "--order",
                "3",
                "--min-context-count",
                "1",
                "--min-count",
                "2",
                "--threads",
                "4",
                "--chunk-symbols",
                "8",
            ])
            .arg("--report")
            .arg(&report)
            .output()
            .unwrap()
    };
    let result = run();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let encoded = fs::read(&output).unwrap();
    assert_eq!(
        archive::decode(&encoded)
            .unwrap()
            .decode()
            .unwrap()
            .concat(),
        raw
    );
    let text = fs::read_to_string(&report).unwrap();
    assert!(
        text.contains("\"weight_type\": \"float32\"") && text.contains("full-training-file-frozen")
    );
    assert!(!run().status.success());
    assert_eq!(fs::read(&output).unwrap(), encoded);
    fs::remove_dir_all(root).unwrap();
}
