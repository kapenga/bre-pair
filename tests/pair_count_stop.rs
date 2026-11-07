use bre_pair::{
    engine::{Compressed, Config},
    field_engine,
    grammar::Pair,
    scoring::Model,
};
use std::collections::HashMap;

// Independent recount: distinct adjacencies plus floor(run_length/2) for each
// self run, without PairCounter, dense marginals, registry or PMI ordering.
fn maximum(source: &Compressed, minimum: u64) -> u64 {
    let mut counts = HashMap::<Pair, u64>::new();
    for block in &source.blocks {
        let ids: Vec<_> = block.iter().collect();
        for edge in ids.windows(2) {
            if !source.grammar.ends_field(edge[0]) && edge[0] != edge[1] {
                *counts.entry((edge[0], edge[1])).or_default() += 1;
            }
        }
        let mut at = 0;
        while at < ids.len() {
            let mut end = at + 1;
            while end < ids.len() && ids[end] == ids[at] {
                end += 1;
            }
            if !source.grammar.ends_field(ids[at]) {
                *counts.entry((ids[at], ids[at])).or_default() += (end - at) as u64 / 2;
            }
            at = end;
        }
    }
    counts
        .values()
        .copied()
        .filter(|&n| n >= minimum)
        .max()
        .unwrap_or(0)
}

#[test]
fn inclusive_frontier_stop_checks_all_pairs_and_preserves_ordinary_merge_floor() {
    let config = Config {
        minimum_count: 32,
        ..Default::default()
    };
    for workers in [1, 4] {
        let exact = vec![b"ab".to_vec(); 64];
        let stopped = field_engine::compress_with_options(
            &exact,
            &config,
            workers,
            8,
            64,
            field_engine::Options {
                stop_max_pair_count: Some(64),
                frequency_cutoff: true,
                memory_saving: true,
                ..Default::default()
            },
            |_| {},
        )
        .unwrap();
        assert!(stopped.stopped_at_pair_count);
        assert_eq!(stopped.maximum_pair_count_at_stop, Some(64));
        assert!(stopped.training.rounds.is_empty());
        assert_eq!(maximum(&stopped.training.compressed, 32), 64);
        let mut fields = vec![b"ab".to_vec(); 65];
        fields.extend(vec![b"xy".to_vec(); 32]);
        let merged = field_engine::compress_with_options(
            &fields,
            &config,
            workers,
            8,
            64,
            field_engine::Options {
                stop_max_pair_count: Some(64),
                frequency_cutoff: true,
                memory_saving: true,
                ..Default::default()
            },
            |_| {},
        )
        .unwrap();
        assert!(!merged.training.rounds.is_empty());
        assert!(merged
            .training
            .rounds
            .iter()
            .any(|r| r.minimum_selected_count == 32));
        assert!(
            merged.stopped_at_pair_count
                && !merged.stopped_at_rule_limit
                && !merged.stopped_at_round_limit
        );
        let actual = maximum(&merged.training.compressed, 32);
        assert!(actual <= 64);
        assert_eq!(merged.maximum_pair_count_at_stop, Some(actual));
        assert_eq!(merged.training.compressed.decode().unwrap(), fields);
    }
}

#[test]
fn frontier_counts_match_independent_recount_for_runs_eof_filters_and_scoring() {
    let fields: Vec<_> = (0..128)
        .map(|i| format!("aaaaa common queen {i} {}\n", i % 17).into_bytes())
        .collect();
    for score in [Model::Frequency, Model::WeightedPmi] {
        let config = Config {
            minimum_count: 4,
            score,
            ..Default::default()
        };
        for cutoff in [false, true] {
            for workers in [1, 4] {
                let result = field_engine::compress_with_options(
                    &fields,
                    &config,
                    workers,
                    8,
                    32,
                    field_engine::Options {
                        stop_max_pair_count: Some(8),
                        frequency_cutoff: cutoff,
                        memory_saving: cutoff,
                        ..Default::default()
                    },
                    |_| {},
                )
                .unwrap();
                assert!(result.stopped_at_pair_count);
                let actual = maximum(&result.training.compressed, 4);
                assert!(actual <= 8);
                assert_eq!(result.maximum_pair_count_at_stop, Some(actual));
                assert_eq!(result.training.compressed.decode().unwrap(), fields);
            }
        }
    }
    let runs = vec![b"aaaaa".to_vec(); 64];
    let config = Config {
        minimum_count: 32,
        ..Default::default()
    };
    let result = field_engine::compress_with_options(
        &runs,
        &config,
        4,
        8,
        64,
        field_engine::Options {
            stop_max_pair_count: Some(64),
            ..Default::default()
        },
        |_| {},
    )
    .unwrap();
    assert_eq!(result.training.rounds[0].maximum_selected_count, 128);
    assert_eq!(maximum(&result.training.compressed, 32), 64);
}

#[test]
fn invalid_floor_is_rejected_and_u64_threshold_can_stop_before_any_merge() {
    let fields = vec![b"abc".to_vec(); 10];
    let config = Config {
        minimum_count: 4,
        ..Default::default()
    };
    assert!(field_engine::compress_with_options(
        &fields,
        &config,
        1,
        8,
        64,
        field_engine::Options {
            stop_max_pair_count: Some(3),
            ..Default::default()
        },
        |_| {}
    )
    .is_err());
    let result = field_engine::compress_with_options(
        &fields,
        &config,
        1,
        8,
        64,
        field_engine::Options {
            stop_max_pair_count: Some(u64::MAX),
            ..Default::default()
        },
        |_| {},
    )
    .unwrap();
    assert!(result.training.rounds.is_empty());
    assert_eq!(result.maximum_pair_count_at_stop, Some(10));
    let ordinary =
        field_engine::compress_with_options(&fields, &config, 1, 8, 64, Default::default(), |_| {})
            .unwrap();
    assert!(!ordinary.stopped_at_pair_count);
    assert_eq!(ordinary.maximum_pair_count_at_stop, None);
}
