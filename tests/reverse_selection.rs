use bre_pair::{
    grammar::BASE_SYMBOLS,
    tokenizer::{Model, Token, Tokenizer},
    vocabulary::{self, reverse},
};
use std::{
    fs,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

struct Input(PathBuf);
impl Input {
    fn new(raw: &[u8]) -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("bre-reverse-{}-{nonce}", std::process::id()));
        fs::write(&path, raw).unwrap();
        Self(path)
    }
}
impl Drop for Input {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

fn model() -> Model {
    let mut tokens: Vec<_> = (0..=255)
        .map(|b| Token {
            bytes: vec![b],
            ends_field: false,
        })
        .chain([Token {
            bytes: vec![],
            ends_field: true,
        }])
        .collect();
    for (bytes, ends_field) in [
        (b"ab".as_slice(), false),
        (b"abc", false),
        (b"bc", false),
        (b"abc\n", true),
        (b"xy", false),
        (&[255, 0], false),
    ] {
        tokens.push(Token {
            bytes: bytes.to_vec(),
            ends_field,
        });
    }
    Model::new(tokens).unwrap()
}

#[test]
fn full_tail_worker_determinism_final_reconstruction_and_emission_bound() {
    let mut raw = b"abc\n".repeat(19);
    raw.extend(b"xy\n".repeat(200));
    raw.extend([255, 0, 255, 0]);
    let input = Input::new(&raw);
    let pool = model();
    let support = vec![0; pool.tokens.len()];
    let mut expected = None;
    for workers in [1, 4] {
        for budget in [BASE_SYMBOLS, BASE_SYMBOLS + 2, pool.tokens.len()] {
            let mut removed = 0;
            let selected =
                reverse::select(pool.clone(), &support, &input.0, budget, workers, |r| {
                    assert!(r.before > r.after);
                    assert_eq!(
                        r.after_totals.tokens,
                        r.before_totals.tokens + r.added_tokens
                    );
                    assert_eq!(r.after_totals.bytes, raw.len() as u64);
                    assert_eq!(r.after_totals.fields, 220);
                    removed += r.before - r.after;
                })
                .unwrap();
            assert_eq!(selected.model.tokens.len(), budget);
            assert_eq!(removed, pool.tokens.len() - budget);
            assert_eq!(
                selected.maintained_uses.iter().sum::<u64>(),
                selected.maintained.tokens
            );
            assert_eq!(selected.maintained.bytes, raw.len() as u64);
            assert_eq!(selected.initial.bytes, raw.len() as u64);
            let t = Tokenizer::new(selected.model).unwrap();
            let actual =
                vocabulary::scan(&input.0, &t, None, None, workers, 1, 0, true, true).unwrap();
            assert!(actual.all.tokens <= selected.maintained.tokens);
            assert_eq!(actual.all.bytes, raw.len() as u64);
            assert_eq!(actual.all.fields, 220);
            if budget == BASE_SYMBOLS + 2 {
                assert!(t.model().tokens.iter().any(|token| token.bytes == b"xy"));
                let result = (t.model().to_bytes(), selected.maintained_uses);
                if let Some(expected) = &expected {
                    assert_eq!(&result, expected);
                } else {
                    expected = Some(result);
                }
            }
        }
    }
}

#[test]
fn invalid_parameters_fail_before_reading_training_data() {
    let pool = model();
    let missing = PathBuf::from("missing-reverse-selection-input");
    for (budget, workers, len) in [
        (256, 1, pool.tokens.len()),
        (pool.tokens.len() + 1, 1, pool.tokens.len()),
        (258, 0, pool.tokens.len()),
        (258, 257, pool.tokens.len()),
        (258, 1, 1),
    ] {
        assert!(reverse::select(
            pool.clone(),
            &vec![0; len],
            &missing,
            budget,
            workers,
            |_| {}
        )
        .is_err());
    }
    for fine_keep in [0.0, 1.0, f64::NAN] {
        let options = reverse::Options {
            batch_search: None,
            coverage_discount_ppm: 0,
            coverage_settings: Default::default(),
            vocabulary: 258,
            workers: 1,
            exact_refresh: false,
            schedule: vocabulary::PruningSchedule {
                fine_keep,
                ..Default::default()
            },
        };
        let error = reverse::select_with_options(
            pool.clone(),
            &vec![0; pool.tokens.len()],
            &missing,
            options,
            |_| {},
        )
        .err()
        .unwrap();
        assert!(error.contains("invalid reverse-selection"));
    }
}

#[test]
fn exact_refresh_recounts_between_batches_and_accounts_for_every_saved_emission() {
    let input = Input::new(&b"abc\n".repeat(20));
    let pool = model();
    let support = vec![0; pool.tokens.len()];
    let mut last_after = None;
    let (mut added, mut saved, mut rounds) = (0, 0, 0);
    let selected = reverse::select_with_options(
        pool,
        &support,
        &input.0,
        reverse::Options {
            batch_search: None,
            coverage_discount_ppm: 0,
            coverage_settings: Default::default(),
            vocabulary: 258,
            workers: 4,
            exact_refresh: true,
            schedule: vocabulary::PruningSchedule {
                fine_keep: 0.999,
                ..Default::default()
            },
        },
        |r| {
            if let Some(last) = last_after {
                assert_eq!(r.before_totals.tokens + r.reparse_saved, last);
            }
            last_after = Some(r.after_totals.tokens);
            added += r.added_tokens;
            saved += r.reparse_saved;
            rounds += 1;
        },
    )
    .unwrap();
    assert_eq!(rounds, 5);
    assert_eq!(selected.refresh_scans, rounds - 1);
    assert_eq!(
        selected.initial.tokens + added - saved,
        selected.maintained.tokens
    );
    assert_eq!(selected.maintained.bytes, 80);
    assert_eq!(selected.maintained.fields, 20);
    let tokenizer = Tokenizer::new(selected.model).unwrap();
    let actual = vocabulary::scan(&input.0, &tokenizer, None, None, 4, 1, 0, true, true).unwrap();
    assert!(actual.all.tokens <= selected.maintained.tokens);
}

#[test]
fn overlap_discount_preserves_bytes_fields_bounds_and_worker_determinism() {
    let mut raw = b"abc\n".repeat(20);
    raw.extend(b"xyabc\n".repeat(9));
    raw.extend([255, 0]);
    let input = Input::new(&raw);
    let pool = model();
    let support = vec![0; pool.tokens.len()];
    for (exact_refresh, signal) in [false, true].into_iter().flat_map(|exact| {
        [
            reverse::CoverageSignal::Overlap,
            reverse::CoverageSignal::Shuffle,
            reverse::CoverageSignal::Stratified,
            reverse::CoverageSignal::Uniform,
        ]
        .map(|signal| (exact, signal))
    }) {
        let mut expected = None;
        for workers in [1, 4] {
            let mut scored = 0;
            let selected = reverse::select_with_options(
                pool.clone(),
                &support,
                &input.0,
                reverse::Options {
                    batch_search: None,
                    vocabulary: 259,
                    workers,
                    exact_refresh,
                    coverage_discount_ppm: 30000,
                    coverage_settings: reverse::CoverageSettings {
                        signal,
                        seed: 123,
                        ..Default::default()
                    },
                    schedule: Default::default(),
                },
                |r| {
                    scored += r.coverage_targets;
                    assert!(r.coverage_index_bytes > 0);
                },
            )
            .unwrap();
            assert!(scored > 0);
            let encoded = selected.model.to_bytes();
            if let Some(previous) = &expected {
                assert_eq!(&encoded, previous);
            } else {
                expected = Some(encoded);
            }
            let tokenizer = Tokenizer::new(selected.model).unwrap();
            let actual =
                vocabulary::scan(&input.0, &tokenizer, None, None, workers, 1, 0, true, true)
                    .unwrap();
            assert_eq!(actual.all.bytes, raw.len() as u64);
            assert_eq!(actual.all.fields, 30);
            assert!(actual.all.tokens <= selected.maintained.tokens);
        }
    }
    let invalid = reverse::Options {
        batch_search: None,
        vocabulary: 259,
        workers: 1,
        exact_refresh: false,
        coverage_discount_ppm: 1_000_001,
        coverage_settings: Default::default(),
        schedule: Default::default(),
    };
    assert!(reverse::select_with_options(pool, &support, &input.0, invalid, |_| {}).is_err());
}

#[test]
fn measured_batch_search_reuses_winner_and_preserves_one_trial_control_and_workers() {
    let mut raw = b"abc\n".repeat(20);
    raw.extend(b"ab\n".repeat(9));
    raw.extend(b"xyabc\n".repeat(13));
    let input = Input::new(&raw);
    let pool = model();
    let support = vec![0; pool.tokens.len()];
    let options = reverse::Options {
        vocabulary: 259,
        workers: 1,
        exact_refresh: true,
        schedule: Default::default(),
        coverage_discount_ppm: 0,
        coverage_settings: Default::default(),
        batch_search: None,
    };
    let baseline =
        reverse::select_with_options(pool.clone(), &support, &input.0, options, |_| {}).unwrap();
    let single = reverse::select_with_options(
        pool.clone(),
        &support,
        &input.0,
        reverse::Options {
            batch_search: Some(reverse::SearchOptions {
                trials: 1,
                ..Default::default()
            }),
            ..options
        },
        |r| {
            assert_eq!(r.batch_search.evaluated, 1);
            assert_eq!(r.batch_search.chosen_trial, 0);
            assert_eq!(r.batch_search.baseline_tokens, r.batch_search.chosen_tokens);
        },
    )
    .unwrap();
    assert_eq!(single.model.to_bytes(), baseline.model.to_bytes());
    assert_eq!(single.maintained_uses, baseline.maintained_uses);
    let mut previous = None;
    for workers in [1, 4] {
        let mut last = 0;
        let selected = reverse::select_with_options(
            pool.clone(),
            &support,
            &input.0,
            reverse::Options {
                workers,
                batch_search: Some(reverse::SearchOptions::default()),
                ..options
            },
            |r| {
                assert!(r.batch_search.evaluated >= 1 && r.batch_search.evaluated <= 8);
                assert_eq!(r.batch_search.evaluated + r.batch_search.duplicates, 8);
                assert!(r.batch_search.chosen_tokens <= r.batch_search.baseline_tokens);
                last = r.batch_search.chosen_tokens;
            },
        )
        .unwrap();
        if let Some(p) = &previous {
            assert_eq!(&selected.model.to_bytes(), p);
        }
        previous = Some(selected.model.to_bytes());
        let parsed = vocabulary::scan(
            &input.0,
            &Tokenizer::new(selected.model).unwrap(),
            None,
            None,
            workers,
            1,
            0,
            true,
            true,
        )
        .unwrap();
        assert_eq!(parsed.all.tokens, last);
        assert_eq!(parsed.all.bytes, raw.len() as u64);
        assert_eq!(parsed.all.fields, 42);
    }
    for search in [
        reverse::SearchOptions {
            trials: 0,
            ..Default::default()
        },
        reverse::SearchOptions {
            swaps_per_token: 0,
            ..Default::default()
        },
    ] {
        assert!(reverse::select_with_options(
            pool.clone(),
            &support,
            &PathBuf::from("missing-search-input"),
            reverse::Options {
                batch_search: Some(search),
                ..options
            },
            |_| {}
        )
        .is_err());
    }
}
