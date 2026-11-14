use bre_pair::{
    archive,
    entropy_engine::{self, Options},
    grammar::EOF,
    scalar_bonds, search,
};
use std::collections::HashMap;

#[test]
fn two_wave_constant_gains_match_the_production_counter_with_frequency_retirement() {
    let mut fields: Vec<_> = (0..64)
        .map(|i| format!("aaaaa abcabc the queen 甲乙 {}\n", i % 7).into_bytes())
        .collect();
    fields.extend([
        vec![],
        vec![0, 255, 0, 255, 0],
        b"rare single field".to_vec(),
    ]);
    for workers in [1, 4] {
        let experimental = entropy_engine::compress_boundaries(
            &fields,
            &Options {
                workers,
                ..options()
            },
            |field| {
                let mut weights = vec![1.0; field.len() + 1];
                weights[0] = 0.0;
                Ok(weights)
            },
            |_, _| {},
        )
        .unwrap();
        let production = bre_pair::field_engine::compress_with_options(
            &fields,
            &bre_pair::engine::Config {
                minimum_count: 2,
                score: bre_pair::scoring::Model::Frequency,
                search: options().search,
                ..Default::default()
            },
            workers,
            64,
            23,
            bre_pair::field_engine::Options {
                frequency_cutoff: true,
                memory_saving: true,
                ..Default::default()
            },
            |_| {},
        )
        .unwrap();
        let a = &experimental.training.compressed;
        let b = &production.training.compressed;
        let recipes = |source: &bre_pair::engine::Compressed| {
            source
                .grammar
                .rules
                .iter()
                .map(|r| (r.left, r.right))
                .collect::<Vec<_>>()
        };
        let stream = |source: &bre_pair::engine::Compressed| {
            source
                .blocks
                .iter()
                .flat_map(|b| b.iter())
                .collect::<Vec<_>>()
        };
        assert_eq!(recipes(a), recipes(b));
        assert_eq!(stream(a), stream(b));
        assert_eq!(a.decode().unwrap(), fields);
    }
}

fn options() -> Options {
    Options {
        minimum_count: 2,
        chunk_symbols: 23,
        search: search::Options {
            certify: false,
            fill_waves: 2,
            fill_min_count: 2,
            ..Default::default()
        },
        ..Default::default()
    }
}

#[test]
fn actual_removed_join_wins_and_self_runs_do_not_count_overlaps_twice() {
    let fields = vec![b"abc".to_vec(); 20];
    let result = entropy_engine::compress_boundaries(
        &fields,
        &Options {
            max_rounds: Some(1),
            ..options()
        },
        |_| Ok(vec![0.0, 0.25, 0.75, 0.0]),
        |_, _| {},
    )
    .unwrap();
    assert!(result.training.compressed.grammar.find((98, 99)).is_some());
    assert!(result.training.compressed.grammar.find((97, 98)).is_none());
    assert_eq!(
        result.losses[0].removed_units,
        15 * u128::from(scalar_bonds::SCALE)
    );
    let fields = vec![b"aaaaa".to_vec(); 20];
    let result = entropy_engine::compress_boundaries(
        &fields,
        &Options {
            max_rounds: Some(1),
            ..options()
        },
        |_| Ok(vec![0.0, 0.25, 0.75, 0.125, 1.0, 0.0]),
        |_, _| {},
    )
    .unwrap();
    assert_eq!(result.training.rounds[0].removed, 40);
    assert_eq!(
        result.losses[0].removed_units,
        15 * u128::from(scalar_bonds::SCALE) / 2
    );
    assert_eq!(result.training.compressed.decode().unwrap(), fields);
}

#[test]
fn independent_raw_offset_oracle_checks_every_batch_loss_conflict_and_reconstruction() {
    let mut fields: Vec<_> = (0..36)
        .map(|i| format!("aaaaa abcabc the queen 甲乙😀 {i}\n").into_bytes())
        .collect();
    fields.extend([vec![], vec![0, 255, 0, 255, 0], b"zzzzzzzzzzzzzzz".to_vec()]);
    let model = scalar_bonds::Statistics::count(&fields).unwrap().freeze();
    let weights: Vec<_> = fields
        .iter()
        .map(|f| model.field_weights(f).unwrap())
        .collect();
    let mut expected = None;
    for workers in [1, 4] {
        let result = entropy_engine::compress_boundaries(
            &fields,
            &Options {
                workers,
                ..options()
            },
            |f| model.field_weights(f),
            |_, _| {},
        )
        .unwrap();
        let source = &result.training.compressed;
        let mut state: Vec<Vec<_>> = fields
            .iter()
            .map(|f| {
                f.iter()
                    .map(|&b| u32::from(b))
                    .chain([EOF])
                    .enumerate()
                    .map(|(offset, symbol)| (symbol, offset))
                    .collect()
            })
            .collect();
        let sum = |state: &Vec<Vec<(u32, usize)>>| -> u128 {
            state
                .iter()
                .zip(&weights)
                .map(|(f, w)| {
                    f.iter()
                        .map(|&(_, at)| (w[at] * scalar_bonds::SCALE as f32) as u128)
                        .sum::<u128>()
                })
                .sum()
        };
        assert_eq!(sum(&state), result.initial_loss_units);
        let mut prior_rules = 0;
        for (round, loss) in result.training.rounds.iter().zip(&result.losses) {
            assert_eq!(round.new_rules, round.selected); // This fixture introduces each recipe once.
            let chosen: HashMap<_, _> = source.grammar.rules[prior_rules..round.total_rules]
                .iter()
                .enumerate()
                .map(|(i, r)| ((r.left, r.right), (257 + prior_rules + i) as u32))
                .collect();
            assert_eq!(sum(&state), loss.before_units);
            let before_count: usize = state.iter().map(Vec::len).sum();
            for field in &mut state {
                // Distinct chosen pair types must not share an actual middle symbol.
                for triple in field.windows(3) {
                    let ab = (triple[0].0, triple[1].0);
                    let bc = (triple[1].0, triple[2].0);
                    assert!(!(ab != bc && chosen.contains_key(&ab) && chosen.contains_key(&bc)));
                }
                let mut rewritten = Vec::new();
                let mut at = 0;
                while at < field.len() {
                    if let Some(&symbol) = field
                        .get(at + 1)
                        .and_then(|b| chosen.get(&(field[at].0, b.0)))
                    {
                        rewritten.push((symbol, field[at].1));
                        at += 2;
                    } else {
                        rewritten.push(field[at]);
                        at += 1;
                    }
                }
                *field = rewritten;
            }
            assert_eq!(
                before_count - state.iter().map(Vec::len).sum::<usize>(),
                round.removed
            );
            assert_eq!(sum(&state), loss.after_units);
            assert_eq!(loss.before_units - loss.after_units, loss.removed_units);
            prior_rules = round.total_rules;
        }
        assert_eq!(sum(&state), result.final_loss_units);
        assert_eq!(source.decode().unwrap(), fields);
        // Normalize chunk boundaries before comparing complete serialized output.
        let mut block = bre_pair::packed::PackedSymbols::new(bre_pair::packed::symbol_width(
            source.grammar.alphabet_size(),
        ));
        for part in &source.blocks {
            for symbol in part.iter() {
                block.push(symbol);
            }
        }
        let normalized = bre_pair::engine::Compressed {
            grammar: source.grammar.clone(),
            blocks: vec![block],
            input_bytes: source.input_bytes,
            field_count: source.field_count,
        };
        let encoded = archive::encode(&normalized).unwrap();
        assert_eq!(archive::decode(&encoded).unwrap().decode().unwrap(), fields);
        if let Some(expected) = &expected {
            assert_eq!(expected, &encoded);
        } else {
            expected = Some(encoded);
        }
    }
}

#[test]
fn annotation_contract_rejects_wrong_lengths_nonfinite_and_non_q24_values() {
    for weights in [
        vec![],
        vec![0.0],
        vec![0.5, 0.0],
        vec![0.0, f32::NAN],
        vec![0.0, 2.0],
        vec![0.0, 1e-12],
    ] {
        assert!(entropy_engine::compress_boundaries(
            &[b"a".to_vec()],
            &options(),
            |_| Ok(weights.clone()),
            |_, _| {}
        )
        .is_err());
    }
    let empty = entropy_engine::compress_boundaries(&[], &options(), |_| unreachable!(), |_, _| {})
        .unwrap();
    assert_eq!(empty.initial_loss_units, 0);
    assert_eq!(empty.final_loss_units, 0);
}
