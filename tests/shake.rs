use bre_pair::{
    archive,
    batch::Counting,
    engine::{Compressed, Config},
    field_engine,
    packed::{symbol_width, PackedSymbols},
    scoring::Model,
    shake,
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
fn shaking_preserves_bytes_eof_counts_and_reproducibility_across_workers_and_count_modes() {
    let fields: Vec<_> = (0..96)
        .map(|i| {
            let mut field = format!(
                "the queen and the king aaaaa aabcdaafgcdaacgadaaee {} {}\n",
                i % 7,
                i % 11
            )
            .into_bytes();
            field.extend([0, 255]);
            field
        })
        .chain([vec![]])
        .collect();
    for score in [Model::Frequency, Model::WeightedPmi] {
        let mut reference = None;
        for mode in [Counting::Full, Counting::Incremental] {
            for workers in [1, 4] {
                let result = field_engine::compress_with_options(
                    &fields,
                    &Config {
                        minimum_count: 3,
                        score,
                        pair_counting: mode,
                        ..Config::default()
                    },
                    workers,
                    8,
                    64,
                    field_engine::Options {
                        shake: Some(shake::Options {
                            fraction: 0.25,
                            seed: 42,
                            ..shake::Options::default()
                        }),
                        ..Default::default()
                    },
                    |_| {},
                )
                .unwrap();
                assert_eq!(result.shakes.len(), 2);
                let source = &result.training.compressed;
                assert_eq!(source.decode().unwrap(), fields);
                assert_eq!(
                    archive::decode(&canonical(source))
                        .unwrap()
                        .decode()
                        .unwrap(),
                    fields
                );
                for event in &result.shakes {
                    assert_eq!(
                        event.before,
                        result.training.rounds[event.after_batch - 1].rewritten
                    );
                    assert!(event.after > event.before);
                    assert!(event.selected_symbols.iter().all(|&id| id >= 257));
                    assert_eq!(
                        result.training.rounds[event.after_batch].before,
                        event.after
                    );
                }
                let original: usize = fields.iter().map(|f| f.len() + 1).sum();
                let removed: usize = result.training.rounds.iter().map(|r| r.removed).sum();
                let expanded: usize = result.shakes.iter().map(|s| s.after - s.before).sum();
                assert_eq!(original + expanded - removed, source.token_count());
                let current = (
                    canonical(source),
                    result
                        .shakes
                        .iter()
                        .map(|e| e.selected_symbols.clone())
                        .collect::<Vec<_>>(),
                );
                if let Some(expected) = &reference {
                    assert_eq!(&current, expected);
                } else {
                    reference = Some(current);
                }
            }
        }
    }
}

#[test]
fn random_seeds_change_the_selection_and_invalid_schedules_fail() {
    let fields: Vec<_> = (0..64)
        .map(|i| format!("abcdefghijk lmnopqrstu repeated words repeated words {i}\n").into_bytes())
        .collect();
    let run = |options| {
        field_engine::compress_with_options(
            &fields,
            &Config::default(),
            4,
            8,
            64,
            field_engine::Options {
                shake: Some(options),
                ..Default::default()
            },
            |_| {},
        )
    };
    let first = run(shake::Options {
        seed: 1,
        fraction: 0.5,
        ..Default::default()
    })
    .unwrap();
    let second = run(shake::Options {
        seed: 2,
        fraction: 0.5,
        ..Default::default()
    })
    .unwrap();
    assert_ne!(
        first.shakes[0].selected_symbols,
        second.shakes[0].selected_symbols
    );
    for fraction in [0.0, -0.1, 1.1, f64::NAN] {
        assert!(run(shake::Options {
            fraction,
            ..Default::default()
        })
        .is_err());
    }
    for options in [
        shake::Options {
            pulses: 0,
            ..Default::default()
        },
        shake::Options {
            first_batch: 0,
            ..Default::default()
        },
        shake::Options {
            interval: 0,
            ..Default::default()
        },
        shake::Options {
            block_rounds: 0,
            ..Default::default()
        },
        shake::Options {
            first_batch: usize::MAX,
            interval: 2,
            ..Default::default()
        },
    ] {
        assert!(run(options).is_err());
    }
    assert!(field_engine::compress_with_options(
        &fields,
        &Config::default(),
        1,
        8,
        64,
        field_engine::Options {
            frequency_cutoff: true,
            shake: Some(Default::default()),
            ..Default::default()
        },
        |_| {}
    )
    .is_err());
}
