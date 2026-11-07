use bre_pair::{
    engine::Compressed,
    grammar::{Grammar, BASE_SYMBOLS, EOF},
    packed::{symbol_width, PackedSymbols},
    tokenizer::{Model, Token},
    vocabulary::{self, SelectionMethod, SelectionOptions},
};
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

fn pool(extra: &[(&[u8], bool)]) -> Model {
    let mut tokens: Vec<_> = (0..=255)
        .map(|byte| Token {
            bytes: vec![byte],
            ends_field: false,
        })
        .chain([Token {
            bytes: Vec::new(),
            ends_field: true,
        }])
        .collect();
    tokens.extend(extra.iter().map(|&(bytes, ends_field)| Token {
        bytes: bytes.to_vec(),
        ends_field,
    }));
    Model::new(tokens).unwrap()
}

#[test]
fn statistics_selection_keeps_nested_fallbacks_using_complete_derivation_counts() {
    let mut grammar = Grammar::default();
    let ab = grammar.add((b'a' as u32, b'b' as u32)).unwrap();
    let abab = grammar.add((ab, ab)).unwrap();
    let ending = grammar.add((abab, EOF)).unwrap();
    let mut block = PackedSymbols::new(symbol_width(grammar.alphabet_size()));
    for _ in 0..100 {
        block.push(ending);
    }
    let source = Compressed {
        grammar,
        blocks: vec![block],
        input_bytes: 400,
        field_count: 100,
    };
    let (candidates, support) = Model::candidates(&source, 128).unwrap();
    assert_eq!(support[ab as usize], 200);
    assert_eq!(support[abab as usize], 100);
    // The stream never emits `ab` directly. It is still the most useful
    // fallback according to its complete grammar frequency.
    for workers in [1, 4] {
        let mut rounds = 0;
        let selected = vocabulary::select(
            candidates.clone(),
            &support,
            Path::new("no-training-file-is-needed-for-statistics"),
            &SelectionOptions {
                vocabulary: BASE_SYMBOLS + 1,
                workers,
                sample_bytes: 1,
                ..SelectionOptions::default()
            },
            |r| {
                assert_eq!(r.method, SelectionMethod::Statistics);
                assert!(r.stride.is_none() && r.totals.is_none());
                rounds += 1;
            },
        )
        .unwrap();
        assert!(rounds > 0);
        assert_eq!(selected.tokens.len(), BASE_SYMBOLS + 1);
        assert_eq!(selected.tokens.last().unwrap().bytes, b"ab");
    }
}

#[test]
fn statistics_scores_keep_64_bit_precision_and_do_not_overflow_when_multiplied() {
    type ScoringCase<'a> = (&'a [(&'a [u8], bool)], &'a [u64], &'a [u8]);
    let cases: &[ScoringCase<'_>] = &[
        (
            &[(b"ab", false), (b"ac", false)],
            &[(1 << 53) + 1, 1 << 53],
            b"ab",
        ),
        (
            &[(b"ab", false), (b"cde", false)],
            &[u64::MAX, u64::MAX / 2 + 1],
            b"cde",
        ),
        // EOF is part of replacement cost, including for one-byte tokens.
        (&[(b"q", true), (b"yz", false)], &[90, 80], b"q"),
        // Utility includes saved output tokens, not just raw frequency.
        (&[(b"ab", false), (b"xyz", false)], &[100, 60], b"xyz"),
    ];
    for &(extra, counts, expected) in cases {
        let mut support = vec![0; BASE_SYMBOLS];
        support.extend(counts);
        let selected = vocabulary::select(
            pool(extra),
            &support,
            Path::new("no-training-file-is-needed-for-statistics"),
            &SelectionOptions {
                vocabulary: BASE_SYMBOLS + 1,
                ..SelectionOptions::default()
            },
            |_| {},
        )
        .unwrap();
        assert_eq!(selected.tokens.last().unwrap().bytes, expected);
        assert_eq!(selected.tokens[..BASE_SYMBOLS], pool(&[]).tokens);
    }
}

struct Directory(PathBuf);
impl Drop for Directory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn public_selection_defaults_to_statistics_and_requires_explicit_sampling() {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = Directory(
        std::env::temp_dir().join(format!("bre-statistics-{}-{nonce}", std::process::id())),
    );
    fs::create_dir(&root.0).unwrap();
    let training = root.0.join("train.txt");
    let candidates = root.0.join("candidates.brp");
    fs::write(&training, b"ab\ncd\nef\ngh\nij\nkl\nmn\nop\n".repeat(30)).unwrap();
    let run = |args: &[&str]| {
        Command::new(env!("CARGO_BIN_EXE_brepair"))
            .args(args)
            .output()
            .unwrap()
    };
    let trained = run(&[
        "train",
        training.to_str().unwrap(),
        candidates.to_str().unwrap(),
        "--min-count",
        "2",
    ]);
    assert!(
        trained.status.success(),
        "{}",
        String::from_utf8_lossy(&trained.stderr)
    );
    for (name, extra) in [
        ("statistics", vec![]),
        (
            "sampled",
            vec!["--selection", "sampled", "--sample-bytes", "100"],
        ),
    ] {
        let model = root.0.join(format!("{name}.brt"));
        let mut args = vec![
            "vocab",
            candidates.to_str().unwrap(),
            training.to_str().unwrap(),
            model.to_str().unwrap(),
            "--vocab-size",
            "260",
            "--min-token-uses",
            "2",
        ];
        args.extend(extra);
        let built = run(&args);
        assert!(
            built.status.success(),
            "{}",
            String::from_utf8_lossy(&built.stderr)
        );
        let report = String::from_utf8(built.stdout).unwrap();
        assert!(report.contains(&format!("\"selection_method\":\"{name}\"")));
        assert!(report.contains("\"round_trip_verified\":true"));
        if name == "statistics" {
            assert!(report.contains("\"sample_bytes\":null"));
            assert!(report.contains("\"support_weight\":null"));
            assert!(report.contains("\"corpus\":null"));
        } else {
            assert!(report.contains("\"sample_bytes\":100"));
            assert!(!report.contains("\"corpus\":null"));
        }
    }
    for extra in [
        vec!["--sample-bytes", "100"],
        vec!["--support-weight", "0.02"],
        vec!["--selection", "invalid"],
        vec!["--selection", "sampled", "--sample-bytes", "0"],
    ] {
        let model = root.0.join("rejected.brt");
        let mut args = vec![
            "vocab",
            candidates.to_str().unwrap(),
            training.to_str().unwrap(),
            model.to_str().unwrap(),
        ];
        args.extend(extra);
        assert!(!run(&args).status.success());
        assert!(!model.exists());
    }
}
