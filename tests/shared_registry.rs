use bre_pair::{
    archive,
    batch::Counting,
    engine::{self, Config},
    field_engine,
    grammar::EOF,
    packed::{symbol_width, PackedSymbols},
    registry::Registry,
    scoring::Model,
};

fn canonical(mut source: engine::Compressed) -> Vec<u8> {
    let mut block = PackedSymbols::with_capacity(
        symbol_width(source.grammar.alphabet_size()),
        source.token_count(),
    );
    for b in &source.blocks {
        for s in b.iter() {
            block.push(s);
        }
    }
    source.blocks = vec![block];
    archive::encode(&source).unwrap()
}

#[test]
fn shared_registry_registers_once_under_collisions_growth_and_hot_updates() {
    let mut registry = Registry::new(8).unwrap();
    std::thread::scope(|scope| {
        for worker in 0..8 {
            let registry = &registry;
            scope.spawn(move || {
                let mut writer = registry.writer();
                for i in 0..10000 {
                    writer.pair((1, 2), 1).unwrap();
                    writer.pair((worker + 100, i), 1).unwrap();
                }
                writer.flush().unwrap();
            });
        }
    });
    {
        let frozen = registry.freeze();
        assert_eq!(frozen.get((1, 2)), 80000);
        let mut entries = 0;
        let mut total = 0;
        frozen.visit(|_, n| {
            entries += 1;
            total += n;
        });
        assert_eq!(entries, 80001);
        assert_eq!(total, 160000);
    }
    std::thread::scope(|scope| {
        for worker in 0..8 {
            let registry = &registry;
            scope.spawn(move || {
                let mut writer = registry.writer();
                for i in 0..10000 {
                    writer.pair((1, 2), -1).unwrap();
                    writer.pair((worker + 100, i), -1).unwrap();
                }
                writer.flush().unwrap();
            });
        }
    });
    registry
        .freeze()
        .visit(|_, _| panic!("zero count remained live"));
    // Force cleanup/growth with a fresh generation after concurrent deletion.
    let mut writer = registry.writer();
    for i in 0..100000 {
        writer.pair((u32::MAX, i), 1).unwrap();
    }
    writer.flush().unwrap();
    drop(writer);
    assert_eq!(registry.freeze().get((u32::MAX, 99999)), 1);
    assert!(registry.peak_slot_bytes() > registry.slot_bytes());
    assert_eq!(Registry::worker_log_bytes(), 16384);
}

#[test]
fn full_width_keys_and_counters_never_alias_wrap_or_saturate() {
    let mut registry = Registry::new(2).unwrap();
    let mut writer = registry.writer();
    writer.pair((0, 0), i64::MAX).unwrap();
    writer.pair((0, 0), i64::MAX).unwrap();
    writer.pair((u32::MAX, u32::MAX), 1).unwrap();
    writer.overlap(17, 1i64 << 33).unwrap();
    writer.flush().unwrap();
    drop(writer);
    assert_eq!(registry.freeze().get((0, 0)), u64::MAX - 1);
    let mut writer = registry.writer();
    writer.pair((0, 0), 1).unwrap();
    writer.flush().unwrap();
    writer.pair((0, 0), 1).unwrap();
    assert!(writer.flush().is_err());
    drop(writer);
    let frozen = registry.freeze();
    assert_eq!(frozen.get((0, 0)), u64::MAX);
    assert_eq!(frozen.get((u32::MAX, u32::MAX)), 1);
    assert_eq!(frozen.get((EOF, 17)), 1u64 << 33);
    let mut marginal_registry = Registry::new(2).unwrap();
    let mut writer = marginal_registry.writer();
    writer.pair((17, 17), 1i64 << 33).unwrap();
    writer.overlap(17, 1i64 << 33).unwrap();
    writer.finish().unwrap();
    let marginal_frozen = marginal_registry.freeze();
    let stats = bre_pair::scoring::Statistics::from_registry(&marginal_frozen, 258);
    assert_eq!(stats.observed((EOF, 17)), 0);
    assert_eq!(stats.observed((17, 17)), 1u64 << 34);
}

#[test]
fn last_seen_support_is_capped_approximate_and_reset_without_changing_counts() {
    let plain = Registry::new(2).unwrap();
    let mut registry = Registry::with_documents(2, 64).unwrap();
    assert_eq!(registry.slot_bytes(), plain.slot_bytes() * 3 / 2);
    let mut writer = registry.writer();
    let document = (1u64 << 40) + 7;
    for _ in 0..1024 {
        writer.pair_document((1, 2), 1, document).unwrap();
    }
    writer.finish().unwrap();
    let mut found = Vec::new();
    registry.retain_and_visit_documented(|_| true, |p, n, d| found.push((p, n, d)));
    assert_eq!(found, vec![((1, 2), 1024, 1)]);
    // Even one worker can overcount A/B/A: this is the requested approximation.
    let mut writer = registry.writer();
    for id in [document, document + 1, document] {
        writer.pair_document((1, 2), 0, id).unwrap();
        writer.flush().unwrap();
    }
    writer.finish().unwrap();
    registry.retain_and_visit_documented(
        |_| true,
        |p, n, d| {
            assert_eq!((p, n, d), ((1, 2), 1024, 3));
        },
    );
    // No rewrite observation means no carried-over support in the next epoch.
    registry.retain_and_visit_documented(
        |_| true,
        |_, n, d| {
            assert_eq!((n, d), (1024, 0));
        },
    );
    let mut writer = registry.writer();
    for i in 0..200 {
        writer.pair_document((1, 2), 0, document + i % 2).unwrap();
        writer.flush().unwrap();
    }
    writer.finish().unwrap();
    registry.retain_and_visit_documented(|_| true, |_, _, d| assert_eq!(d, 64));
    let mut writer = registry.writer();
    assert!(writer.pair_document((1, 2), 0, 1 << 56).is_err());
    assert!(writer.pair_document((1, 2), 0, 0).is_err());
}

#[test]
fn last_seen_state_survives_growth_compaction_and_unchanged_pair_observations() {
    let mut registry = Registry::with_documents(4, 8).unwrap();
    std::thread::scope(|scope| {
        for id in 1..=4 {
            let registry = &registry;
            scope.spawn(move || {
                let mut writer = registry.writer();
                for symbol in (0..5000).map(|s| if s >= EOF { s + 1 } else { s }) {
                    writer.pair_document((symbol, 6000), 1, id).unwrap();
                }
                writer.finish().unwrap();
            });
        }
    });
    assert!(registry.growths() > 0);
    assert_eq!(registry.compact_frequent(4, 7000).unwrap(), 5000);
    let mut seen = 0;
    registry.retain_and_visit_documented(
        |_| true,
        |_, n, d| {
            assert_eq!(n, 4);
            assert_eq!(d, 4);
            seen += 1;
        },
    );
    assert_eq!(seen, 5000);
    let mut writer = registry.writer();
    for symbol in (0..5000).map(|s| if s >= EOF { s + 1 } else { s }) {
        writer.pair_document((symbol, 6000), 0, 1u64 << 40).unwrap();
    }
    writer.finish().unwrap();
    registry.retain_and_visit_documented(
        |p| p.0 < 2,
        |_, n, d| {
            assert_eq!((n, d), (4, 1));
        },
    );
    let mut writer = registry.writer();
    writer
        .pair_document((1, 6000), 0, (1u64 << 40) + 1)
        .unwrap();
    writer.finish().unwrap();
    registry.retain_and_visit_documented(
        |_| true,
        |p, n, d| {
            assert_eq!(n, 4);
            assert_eq!(d, u8::from(p.0 == 1));
        },
    );
    registry.clear();
    let mut writer = registry.writer();
    writer.pair_document((1, 6000), 1, 42).unwrap();
    writer.finish().unwrap();
    registry.retain_and_visit_documented(|_| true, |_, n, d| assert_eq!((n, d), (1, 1)));
}

#[test]
fn a_pair_qualifies_from_global_frequency_even_when_each_field_has_one_occurrence() {
    let fields = vec![vec![1, 2]; 8];
    let config = Config {
        score: Model::Frequency,
        minimum_count: 8,
        ..Config::default()
    };
    let result = field_engine::compress(&fields, &config, 8, 8, 3, |_| {}).unwrap();
    assert!(result.training.compressed.grammar.find((1, 2)).is_some());
    assert_eq!(result.training.compressed.decode().unwrap(), fields);
}

#[test]
fn fused_statistics_keep_rare_edges_and_raw_self_overlaps() {
    use bre_pair::{batch::PairCounter, grammar::Grammar, scoring::Statistics};

    let mut fields = vec![b"aaaaabz".to_vec(); 8];
    fields.extend([b"abq".to_vec(), b"abr".to_vec(), vec![]]);
    let grammar = Grammar::default();
    let mut packed = PackedSymbols::new(9);
    let mut left = vec![0u64; 257];
    let mut right = vec![0u64; 257];
    let mut edges = 0u64;
    for field in &fields {
        let symbols: Vec<_> = field.iter().map(|&b| b as u32).chain([EOF]).collect();
        for pair in symbols.windows(2) {
            left[pair[0] as usize] += 1;
            right[pair[1] as usize] += 1;
            edges += 1;
        }
        for symbol in symbols {
            packed.push(symbol);
        }
    }
    let counter = PairCounter::from_buffer(&packed, &grammar);
    let mut registry = Registry::new(2).unwrap();
    let mut writer = registry.writer();
    for (&pair, &count) in &counter.counts {
        writer.pair(pair, count as i64).unwrap();
    }
    for (&symbol, &count) in &counter.self_overlaps {
        writer.overlap(symbol, count as i64).unwrap();
    }
    writer.finish().unwrap();
    let frozen = registry.freeze();
    let mut candidates = Vec::new();
    let stats = Statistics::from_registry_with_visit(&frozen, 257, |pair, count| {
        if pair.0 != EOF && count >= 8 {
            candidates.push((pair, count));
        }
    });
    assert_eq!(stats.left, left);
    assert_eq!(stats.right, right);
    assert_eq!(stats.edges, edges);
    assert_eq!(stats.observed((b'a' as u32, b'a' as u32)), 32);
    assert_eq!(candidates.len(), 4);
    let reference = Statistics::from_counter(&counter, 257);
    for (pair, count) in candidates {
        assert!(pair.0 != EOF && count >= 8);
        for model in [
            Model::Conditional,
            Model::Pmi,
            Model::WeightedPmi,
            Model::Likelihood,
        ] {
            for alpha in [0.0, 0.5, 1.0, 2.0] {
                assert_eq!(
                    stats.score_with_alpha(model, pair, count, alpha).to_bits(),
                    reference
                        .score_with_alpha(model, pair, count, alpha)
                        .to_bits(),
                );
            }
        }
    }
}

#[test]
fn parallel_fields_match_the_original_trainer_for_every_model_and_counting_mode() {
    let fields: Vec<_> = (0..40)
        .map(|i| {
            if i % 7 == 0 {
                vec![]
            } else {
                format!("aaaaa abababc the quick queen {i} abcabd abcabe\n").into_bytes()
            }
        })
        .collect();
    for model in [
        Model::Frequency,
        Model::Conditional,
        Model::Pmi,
        Model::WeightedPmi,
        Model::Likelihood,
    ] {
        let mut config = Config {
            score: model,
            minimum_count: 2,
            buffer_bytes: 1024 * 1024,
            ..Config::default()
        };
        let reference = engine::compress(&fields, &config, |_| {}).unwrap();
        let expected = canonical(reference.compressed);
        for mode in [Counting::Reuse, Counting::Incremental, Counting::Adaptive] {
            config.pair_counting = mode;
            for workers in [1, 2, 4] {
                let actual =
                    field_engine::compress(&fields, &config, workers, 8, 64, |_| {}).unwrap();
                assert_eq!(actual.training.compressed.decode().unwrap(), fields);
                assert_eq!(
                    canonical(actual.training.compressed),
                    expected,
                    "{model:?} {mode:?} {workers}"
                );
            }
        }
    }
}

#[test]
fn bounded_training_stops_at_a_decodable_batch_boundary() {
    let fields: Vec<_> = (0..40)
        .map(|i| format!("aaaaa abababc the quick queen {i} abcabd abcabe\n").into_bytes())
        .collect();
    let config = Config {
        score: Model::WeightedPmi,
        minimum_count: 2,
        ..Config::default()
    };
    let mut expected = Vec::new();
    let full = field_engine::compress(&fields, &config, 4, 8, 64, |round| {
        expected.push((
            round.before,
            round.rewritten,
            round.selected,
            round.total_rules,
        ));
    })
    .unwrap();
    assert!(expected.len() > 2);
    assert!(!full.stopped_at_round_limit);
    for workers in [1, 4] {
        let mut actual = Vec::new();
        let limited =
            field_engine::compress_with_limit(&fields, &config, workers, 8, 64, Some(2), |round| {
                actual.push((
                    round.before,
                    round.rewritten,
                    round.selected,
                    round.total_rules,
                ));
            })
            .unwrap();
        assert!(limited.stopped_at_round_limit);
        assert_eq!(actual, expected[..2]);
        assert_eq!(
            archive::decode(&canonical(limited.training.compressed))
                .unwrap()
                .decode()
                .unwrap(),
            fields
        );
    }
    assert!(
        field_engine::compress_with_limit(&fields, &config, 4, 8, 64, Some(0), |_| {}).is_err()
    );
}

#[test]
fn owned_chunk_rewrites_preserve_width_growth_and_unchanged_fields() {
    let mut fields: Vec<_> = (0..=255u8)
        .map(|symbol| {
            let pattern = [
                symbol,
                symbol.wrapping_add(1),
                symbol.wrapping_mul(7),
                42,
                symbol,
                17,
            ];
            pattern.repeat(8)
        })
        .collect();
    fields.extend([vec![], vec![], vec![19], vec![203, 7, 149]]);
    let mut config = Config {
        score: Model::WeightedPmi,
        minimum_count: 4,
        buffer_bytes: 1024 * 1024,
        ..Config::default()
    };
    let reference = engine::compress(&fields, &config, |_| {}).unwrap();
    assert!(reference.rounds.iter().any(|round| round.width >= 10));
    let expected = canonical(reference.compressed);
    for mode in [Counting::Reuse, Counting::Incremental, Counting::Adaptive] {
        config.pair_counting = mode;
        for workers in [1, 4] {
            let result = field_engine::compress(&fields, &config, workers, 8, 1, |_| {}).unwrap();
            assert_eq!(result.training.compressed.decode().unwrap(), fields);
            assert_eq!(
                canonical(result.training.compressed),
                expected,
                "{mode:?} {workers}"
            );
        }
    }
}

#[test]
fn registry_retirement_preserves_probe_chains_and_shrinks_sparse_shards() {
    let mut registry = Registry::new(2).unwrap();
    let mut writer = registry.writer();
    for i in 0..6000 {
        writer.pair((i + 1000, i), i as i64 + 1).unwrap();
    }
    writer.finish().unwrap();
    let before = registry.slot_bytes();
    let mut visited = 0;
    registry.retain_and_visit(|pair| pair.1 % 2 == 0, |_, _| visited += 1);
    assert_eq!(visited, 3000);
    assert_eq!(registry.slot_bytes(), before);
    {
        let frozen = registry.freeze();
        for i in 0..6000 {
            assert_eq!(
                frozen.get((i + 1000, i)),
                if i % 2 == 0 { i as u64 + 1 } else { 0 }
            );
        }
    }
    registry.retain_and_visit(|pair| pair.1 < 16, |_, _| {});
    assert_eq!(registry.retired_pairs(), 5992);
    assert!(registry.compactions() > 0);
    assert_eq!(registry.slot_bytes(), 2 * 64 * 16);
    assert!(registry.slot_bytes() < before);
    let mut writer = registry.writer();
    for i in 0..6000 {
        writer.pair((i + 1000, i), 1).unwrap();
    }
    writer.finish().unwrap();
    let frozen = registry.freeze();
    for i in 0..6000 {
        assert_eq!(
            frozen.get((i + 1000, i)),
            1 + if i < 16 && i % 2 == 0 {
                i as u64 + 1
            } else {
                0
            }
        );
    }
}

#[test]
fn filtered_pairs_preserve_every_reference_batch_with_rare_symbols_and_self_runs() {
    let mut fields: Vec<Vec<u8>> = (0..48)
        .map(|i| {
            let mut field = [b"aaaaa ababab abc abd aab aac aa".as_slice(), &[0, 255]].concat();
            if i < 7 {
                field.extend([160 + i, 240 - i]);
            }
            field
        })
        .collect();
    fields.extend(vec![b"xy".to_vec(); 8]);
    fields.extend(vec![b"xy!".to_vec(); 7]);
    fields.extend([vec![], vec![], vec![42], b"aaaaaaa".to_vec()]);
    let shape = |round: &engine::Round| {
        [
            round.before as u64,
            round.rewritten as u64,
            round.candidates as u64,
            round.eligible as u64,
            round.selected as u64,
            round.new_rules as u64,
            round.total_rules as u64,
            round.width as u64,
            round.tie_edges as u64,
            round.components as u64,
            round.largest_component as u64,
            round.greedy_score,
            round.score_upper_bound,
        ]
    };
    let mut retired = 0;
    for minimum in [4, 8] {
        for (model, alpha) in [
            (Model::Frequency, 1.0),
            (Model::Conditional, 1.0),
            (Model::Pmi, 1.0),
            (Model::WeightedPmi, 0.5),
            (Model::WeightedPmi, 1.0),
            (Model::WeightedPmi, 2.0),
            (Model::Likelihood, 1.0),
        ] {
            let mut config = Config {
                minimum_count: minimum,
                score: model,
                pmi_alpha: alpha,
                ..Config::default()
            };
            let reference = engine::compress(&fields, &config, |_| {}).unwrap();
            let expected_rounds: Vec<_> = reference.rounds.iter().map(shape).collect();
            let expected = canonical(reference.compressed);
            for mode in [
                Counting::Full,
                Counting::Reuse,
                Counting::Incremental,
                Counting::Adaptive,
            ] {
                config.pair_counting = mode;
                for workers in [1, 4] {
                    let actual = field_engine::compress_with_options(
                        &fields,
                        &config,
                        workers,
                        8,
                        1,
                        field_engine::Options {
                            frequency_cutoff: true,
                            ..Default::default()
                        },
                        |_| {},
                    )
                    .unwrap();
                    retired += actual.registry_retired_pairs;
                    assert_eq!(
                        actual.training.rounds.iter().map(shape).collect::<Vec<_>>(),
                        expected_rounds,
                        "{minimum} {model:?} {alpha} {mode:?} {workers}"
                    );
                    assert_eq!(actual.training.compressed.decode().unwrap(), fields);
                    assert_eq!(canonical(actual.training.compressed), expected);
                }
            }
        }
    }
    assert!(
        retired > 0,
        "exercise pairs becoming impossible after merging"
    );
}

#[test]
fn default_uses_unfiltered_counting_and_both_options_preserve_bounded_batches() {
    let fields: Vec<_> = (0..32)
        .map(|i| format!("aaaaa abababc the quick queen {i}\n").into_bytes())
        .collect();
    let config = Config {
        score: Model::WeightedPmi,
        minimum_count: 4,
        ..Config::default()
    };
    assert!(!field_engine::Options::default().frequency_cutoff);
    let default = field_engine::compress(&fields, &config, 4, 8, 32, |_| {}).unwrap();
    assert_eq!(default.symbol_statistics_bytes, 0);
    assert_eq!(default.registry_retired_pairs, 0);
    assert_eq!(default.registry_compactions, 0);
    let expected_prefix: Vec<_> = default
        .training
        .rounds
        .iter()
        .take(2)
        .map(|r| {
            (
                r.before,
                r.rewritten,
                r.candidates,
                r.selected,
                r.total_rules,
            )
        })
        .collect();
    for cutoff in [false, true] {
        let full = field_engine::compress_with_options(
            &fields,
            &config,
            4,
            8,
            32,
            field_engine::Options {
                frequency_cutoff: cutoff,
                max_rounds: None,
                max_rules: None,
                stop_max_pair_count: None,
                shake: None,
                memory_saving: false,
                last_seen_documents: false,
            },
            |_| {},
        )
        .unwrap();
        assert_eq!(full.symbol_statistics_bytes > 0, cutoff);
        assert_eq!(
            canonical(full.training.compressed),
            canonical(
                engine::compress(&fields, &config, |_| {})
                    .unwrap()
                    .compressed
            )
        );
        let limited = field_engine::compress_with_options(
            &fields,
            &config,
            4,
            8,
            32,
            field_engine::Options {
                frequency_cutoff: cutoff,
                max_rounds: Some(2),
                max_rules: None,
                stop_max_pair_count: None,
                shake: None,
                memory_saving: false,
                last_seen_documents: false,
            },
            |_| {},
        )
        .unwrap();
        assert!(limited.stopped_at_round_limit);
        assert_eq!(limited.training.compressed.decode().unwrap(), fields);
        assert_eq!(
            limited
                .training
                .rounds
                .iter()
                .map(|r| (
                    r.before,
                    r.rewritten,
                    r.candidates,
                    r.selected,
                    r.total_rules
                ))
                .collect::<Vec<_>>(),
            expected_prefix
        );
    }
}
