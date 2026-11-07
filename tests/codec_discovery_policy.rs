use bre_pair::{
    codec::{DocumentCounting, Frozen, TrainOptions},
    document_support::Policy,
    engine::Config,
    huffman_archive::Reader,
    huffman_dictionary::byte_model,
};
use std::{
    collections::HashSet,
    fs,
    io::Cursor,
    path::PathBuf,
    process::Command,
    sync::atomic::{AtomicU64, Ordering},
};
static NEXT: AtomicU64 = AtomicU64::new(0);
struct Fixture(PathBuf);
impl Fixture {
    fn new(docs: &[(&[u8], u8)]) -> Self {
        let root = std::env::temp_dir().join(format!(
            "brepair-rule-policy-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&root).unwrap();
        let mut raw = vec![];
        let mut index = b"BREDOC01".to_vec();
        index.extend(104u32.to_le_bytes());
        index.extend(0u32.to_le_bytes());
        index.extend((docs.len() as u64).to_le_bytes());
        for &(data, group) in docs {
            let mut record = [0u8; 104];
            record[32..64].fill(group);
            record[76..84].copy_from_slice(&(raw.len() as u64).to_le_bytes());
            record[84..92].copy_from_slice(&(data.len() as u64).to_le_bytes());
            let fields = data.iter().filter(|&&b| b == b'\n').count() as u64
                + u64::from(!data.ends_with(b"\n"));
            record[92..100].copy_from_slice(&fields.to_le_bytes());
            index.extend(record);
            raw.extend(data);
        }
        fs::write(root.join("train"), raw).unwrap();
        fs::write(root.join("index"), index).unwrap();
        Self(root)
    }
    fn policy(&self, minimum: usize) -> Policy {
        Policy {
            index: self.0.join("index"),
            minimum_groups: minimum,
        }
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}
fn config() -> Config {
    Config {
        minimum_count: 4,
        buffer_bytes: 4 << 20,
        ..Default::default()
    }
}

#[test]
fn scaled_occurrences_use_original_bytes_and_integer_u64_bounds() {
    let options = TrainOptions {
        minimum_count_per_gb: 1,
        ..Default::default()
    };
    assert_eq!(options.effective_minimum(4, 768771).unwrap(), 4);
    assert_eq!(options.effective_minimum(5, 100000000).unwrap(), 5);
    assert_eq!(options.effective_minimum(4, 16000004443).unwrap(), 16);
    assert_eq!(
        options.effective_minimum(4, 1_000_000_000_000).unwrap(),
        1000
    );
    assert_eq!(options.effective_minimum(4, 1u64 << 33).unwrap(), 8);
    let large_count = TrainOptions {
        minimum_count_per_gb: 1_000_000_000,
        ..Default::default()
    };
    assert_eq!(
        large_count.effective_minimum(4, 1u64 << 33).unwrap(),
        1u64 << 33
    );
    let excessive = TrainOptions {
        minimum_count_per_gb: u64::MAX,
        ..Default::default()
    };
    assert!(excessive.effective_minimum(4, u64::MAX).is_err());
}

#[test]
fn codec_cli_uses_compatible_fill_waves_in_final_compression_discovery() {
    let mut raw = b"abcd\n".repeat(80);
    raw.extend(b"ab\n".repeat(90));
    raw.extend(b"bc\n".repeat(40));
    let fixture = Fixture::new(&[(&raw, 1), (&raw, 2)]);
    let vocabulary = byte_model();
    fs::write(fixture.0.join("tokens.brt"), vocabulary.to_bytes()).unwrap();
    let mut cfg = Config {
        minimum_count: 8,
        score: bre_pair::scoring::Model::Frequency,
        ..config()
    };
    let options = TrainOptions {
        max_rounds: Some(1),
        documents: Some(fixture.policy(2)),
        document_counting: DocumentCounting::Boundary,
        ..Default::default()
    };
    let (plain, _) = Frozen::train_with_options(
        &vocabulary,
        &fixture.0.join("train"),
        &cfg,
        2,
        options.clone(),
        |_| {},
    )
    .unwrap();
    assert!(!plain.dictionary.rules.contains(&(99, 100)));
    cfg.search.fill_waves = 2;
    cfg.search.fill_min_count = 32;
    let (expected, _) = Frozen::train_with_options(
        &vocabulary,
        &fixture.0.join("train"),
        &cfg,
        2,
        options,
        |_| {},
    )
    .unwrap();
    assert!(
        expected.dictionary.rules.contains(&(99, 100)),
        "second wave did not admit CD after BC was excluded"
    );
    let command = |waves: &str, output: &str| {
        Command::new(env!("CARGO_BIN_EXE_brepair"))
            .current_dir(&fixture.0)
            .args([
                "codec-train",
                "tokens.brt",
                "train",
                output,
                "--document-index",
                "index",
                "--min-rule-documents",
                "2",
                "--document-counting",
                "boundary",
                "--min-count",
                "8",
                "--score",
                "frequency",
                "--max-rounds",
                "1",
                "--threads",
                "2",
                "--fill-waves",
                waves,
                "--fill-min-count",
                "32",
                "--fill-min-count-per-gb",
                "1280",
            ])
            .output()
            .unwrap()
    };
    let result = command("2", "accelerated.brc");
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(
        fs::read(fixture.0.join("accelerated.brc")).unwrap(),
        expected.to_bytes().unwrap()
    );
    let report = String::from_utf8_lossy(&result.stdout);
    for field in [
        "\"fill_waves\":2,",
        "\"fill_min_count\":32,",
        "\"fill_min_count_per_gb\":1280,",
    ] {
        assert!(report.contains(field));
    }
    let invalid = command("0", "invalid-waves.brc");
    assert!(!invalid.status.success());
    assert!(!fixture.0.join("invalid-waves.brc").exists());
    assert_eq!(
        bre_pair::search::scaled_fill_minimum(32, 1280, 16000004443).unwrap(),
        20481
    );
    assert_eq!(
        bre_pair::search::scaled_fill_minimum(32, 1_000_000_000, 1u64 << 33).unwrap(),
        1u64 << 33
    );
    assert!(bre_pair::search::scaled_fill_minimum(32, u64::MAX, u64::MAX).is_err());
}

#[test]
fn fused_last_seen_matches_exact_for_ordered_unique_documents_and_roundtrips_parallel() {
    // Multiple fields per document, multiple packed chunks per document, runs,
    // and EOF-absorbing rules. A local artifact has plenty of occurrences but
    // appears only in the first document.
    let shared = b"aaaaab abc def abc def abc def\n".repeat(3000);
    let mut first = b"XYXYXYXY\n".repeat(1000);
    first.extend(&shared);
    let mut docs = vec![(first.as_slice(), 1)];
    docs.extend((2..=8).map(|id| (shared.as_slice(), id)));
    let fixture = Fixture::new(&docs);
    let options = TrainOptions {
        max_rounds: Some(8),
        documents: Some(fixture.policy(8)),
        ..Default::default()
    };
    let (exact, report) = Frozen::train_with_options(
        &byte_model(),
        &fixture.0.join("train"),
        &config(),
        1,
        options.clone(),
        |_| {},
    )
    .unwrap();
    assert!(report.rounds >= 3);
    let raw = fs::read(fixture.0.join("train")).unwrap();
    for workers in [1, 4] {
        let (model, report) = Frozen::train_with_options(
            &byte_model(),
            &fixture.0.join("train"),
            &config(),
            workers,
            TrainOptions {
                document_counting: DocumentCounting::LastSeen,
                ..options.clone()
            },
            |_| {},
        )
        .unwrap();
        assert_eq!(report.document_support_seconds, 0.0);
        assert!(report.document_filtered_candidates > 0);
        assert!(!model.dictionary.rules.contains(&(b'X' as u32, b'Y' as u32)));
        if workers == 1 {
            assert_eq!(model.to_bytes().unwrap(), exact.to_bytes().unwrap());
        }
        let model = Frozen::from_bytes(&model.to_bytes().unwrap()).unwrap();
        let mut archive = Cursor::new(vec![]);
        model.encode(raw.as_slice(), &mut archive, 4096).unwrap();
        let mut reader = Reader::new(Cursor::new(archive.into_inner())).unwrap();
        let mut actual = vec![];
        reader.decode_to(false, &mut actual).unwrap();
        assert_eq!(actual, raw);
    }
}

#[test]
fn last_seen_preserves_alias_identity_and_rejects_invalid_document_indexes() {
    let text = b"abababab\n".repeat(200);
    let fixture = Fixture::new(&[(&text, 1), (&text, 1), (&text, 1)]);
    let options = TrainOptions {
        max_rounds: Some(2),
        documents: Some(fixture.policy(2)),
        document_counting: DocumentCounting::LastSeen,
        ..Default::default()
    };
    let (model, report) = Frozen::train_with_options(
        &byte_model(),
        &fixture.0.join("train"),
        &config(),
        1,
        options.clone(),
        |_| {},
    )
    .unwrap();
    assert!(model.dictionary.rules.is_empty());
    assert_eq!(report.rounds, 0);
    let mut index = fs::read(fixture.0.join("index")).unwrap();
    index[12] = 1;
    fs::write(fixture.0.join("index"), &index).unwrap();
    assert!(Frozen::train_with_options(
        &byte_model(),
        &fixture.0.join("train"),
        &config(),
        1,
        options.clone(),
        |_| {}
    )
    .err()
    .unwrap()
    .contains("training index"));
    index[12] = 0;
    // Preserve the total bytes but move a document boundary inside a field.
    let a = 24 + 84;
    let b = 24 + 104 + 76;
    let c = 24 + 104 + 84;
    let length = text.len() as u64;
    index[a..a + 8].copy_from_slice(&(length - 1).to_le_bytes());
    index[b..b + 8].copy_from_slice(&(length - 1).to_le_bytes());
    index[c..c + 8].copy_from_slice(&(length + 1).to_le_bytes());
    fs::write(fixture.0.join("index"), &index).unwrap();
    assert!(Frozen::train_with_options(
        &byte_model(),
        &fixture.0.join("train"),
        &config(),
        1,
        options,
        |_| {}
    )
    .is_err());
}

#[test]
fn boundary_cleanup_keeps_ordinary_merge_trajectory_and_frozen_ai_ids() {
    let common = b"abab abc def shared shared\n".repeat(300);
    let mut local = b"XYXYXYXY isolated anomaly\n".repeat(100);
    local.extend(&common);
    let mut docs = vec![(local.as_slice(), 1)];
    docs.extend((2..=8).map(|id| (common.as_slice(), id)));
    let fixture = Fixture::new(&docs);
    let mut tokens = byte_model().tokens;
    tokens.push(bre_pair::tokenizer::Token {
        bytes: b"XY".to_vec(),
        ends_field: false,
    });
    let vocabulary = bre_pair::tokenizer::Model::new(tokens).unwrap();
    let cfg = Config {
        minimum_count: 8,
        score: bre_pair::scoring::Model::WeightedPmi,
        ..config()
    };
    let options = TrainOptions {
        max_rounds: Some(8),
        ..Default::default()
    };
    let mut baseline_rounds = vec![];
    let (baseline, _) = Frozen::train_with_options(
        &vocabulary,
        &fixture.0.join("train"),
        &cfg,
        4,
        options.clone(),
        |r| baseline_rounds.push((r.before, r.rewritten, r.selected, r.new_rules)),
    )
    .unwrap();
    let mut expected = None;
    for workers in [1, 4] {
        let mut rounds = vec![];
        let (model, report) = Frozen::train_with_options(
            &vocabulary,
            &fixture.0.join("train"),
            &cfg,
            workers,
            TrainOptions {
                documents: Some(fixture.policy(8)),
                document_counting: DocumentCounting::Boundary,
                ..options.clone()
            },
            |r| rounds.push((r.before, r.rewritten, r.selected, r.new_rules)),
        )
        .unwrap();
        assert_eq!(rounds, baseline_rounds);
        assert_eq!(report.document_filtered_candidates, 0);
        assert_eq!(report.document_support_seconds, 0.0);
        assert!(report.document_removed_codes > 0);
        assert!(report.document_removed_used_codes > 0);
        assert!(report.document_cleanup_after_symbols > report.document_cleanup_before_symbols);
        assert_eq!(model.dictionary.model.to_bytes(), vocabulary.to_bytes());
        assert!(model.book.symbols.contains(&257)); // A fixed AI ID is mandatory even in one source document.
        let basis = vocabulary.tokens.len() as u32;
        for &code in &model.book.symbols {
            if code < basis {
                continue;
            }
            let mut stack = vec![code];
            while let Some(id) = stack.pop() {
                if id < basis {
                    assert_ne!(id, 257, "one-document compression recipe retained");
                } else {
                    let (a, b) = model.dictionary.rules[(id - basis) as usize];
                    stack.extend([a, b]);
                }
            }
        }
        let bytes = model.to_bytes().unwrap();
        if let Some(reference) = &expected {
            assert_eq!(&bytes, reference);
        } else {
            expected = Some(bytes.clone());
        }
        let model = Frozen::from_bytes(&bytes).unwrap();
        let heldout = b"unseen XYXY shared abc\nshared abab XYZ\n";
        let mut ids = None;
        for codec in [&baseline, &model] {
            let mut archive = Cursor::new(vec![]);
            codec.encode(heldout.as_slice(), &mut archive, 11).unwrap();
            let bytes = archive.into_inner();
            let mut raw = vec![];
            Reader::new(Cursor::new(&bytes))
                .unwrap()
                .decode_to(false, &mut raw)
                .unwrap();
            assert_eq!(raw, heldout);
            let mut tokens = vec![];
            Reader::new(Cursor::new(&bytes))
                .unwrap()
                .decode_to(true, &mut tokens)
                .unwrap();
            if let Some(expected) = &ids {
                assert_eq!(&tokens, expected);
            } else {
                ids = Some(tokens);
            }
        }
    }
    fs::write(fixture.0.join("tokens.brt"), vocabulary.to_bytes()).unwrap();
    let cli = Command::new(env!("CARGO_BIN_EXE_brepair"))
        .current_dir(&fixture.0)
        .args([
            "codec-train",
            "tokens.brt",
            "train",
            "boundary.brc",
            "--document-index",
            "index",
            "--min-rule-documents",
            "8",
            "--document-counting",
            "boundary",
            "--min-count",
            "8",
            "--max-rounds",
            "8",
            "--threads",
            "4",
        ])
        .output()
        .unwrap();
    assert!(
        cli.status.success(),
        "{}",
        String::from_utf8_lossy(&cli.stderr)
    );
    let stdout = String::from_utf8_lossy(&cli.stdout);
    let stderr = String::from_utf8_lossy(&cli.stderr);
    assert!(stdout.contains("\"document_counting\":\"boundary\""));
    assert!(!stderr.contains("\"phase\":\"document_support\""));
    assert!(stderr.contains("\"phase\":\"document_cleanup_count\""));
    let completed = stdout
        .split("\"rounds\":")
        .nth(1)
        .unwrap()
        .split(',')
        .next()
        .unwrap();
    assert!(completed.parse::<usize>().unwrap() > 0);
    assert!(stderr.contains(&format!(
        "\"phase\":\"document_cleanup_count\",\"round\":{completed},"
    )));
    assert_eq!(
        stderr
            .lines()
            .filter(|line| line.contains("\"phase\":\"document_cleanup_count\"")
                && line.contains("\"done\":0,"))
            .count(),
        1
    );
}

#[test]
fn cli_last_seen_has_no_separate_support_phase_and_requires_a_training_index() {
    let raw = b"abababab\n".repeat(30);
    let fixture = Fixture::new(&[(&raw, 1), (&raw, 2)]);
    fs::write(fixture.0.join("bytes.brt"), byte_model().to_bytes()).unwrap();
    let run = |extra: &[&str], output: &str| {
        Command::new(env!("CARGO_BIN_EXE_brepair"))
            .current_dir(&fixture.0)
            .args([
                "codec-train",
                "bytes.brt",
                "train",
                output,
                "--document-counting",
                "last-seen",
                "--max-rounds",
                "1",
                "--min-count",
                "4",
            ])
            .args(extra)
            .output()
            .unwrap()
    };
    let result = run(
        &["--document-index", "index", "--min-rule-documents", "2"],
        "fused.brc",
    );
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let stdout = String::from_utf8_lossy(&result.stdout);
    assert!(stdout.contains("\"document_counting\":\"last-seen\""));
    assert!(stdout.contains("\"document_support_seconds\":0.000000"));
    assert!(!String::from_utf8_lossy(&result.stderr).contains("\"phase\":\"document_support\""));
    let result = run(&[], "invalid.brc");
    assert!(!result.status.success());
    assert!(!fixture.0.join("invalid.brc").exists());
}

#[test]
fn repeated_groups_across_bitmap_words_are_not_mistaken_for_unique_documents() {
    let rare = b"XYXYXYXY\n".repeat(128);
    let common = b"abababab\n".repeat(128);
    let mut docs = vec![(rare.as_slice(), 1)];
    docs.extend((2..=128u8).map(|group| (common.as_slice(), group)));
    docs.push((rare.as_slice(), 1));
    let fixture = Fixture::new(&docs);
    let mut reference = None;
    for workers in [1, 4] {
        let (model, _) = Frozen::train_with_options(
            &byte_model(),
            &fixture.0.join("train"),
            &Config {
                minimum_count: 128,
                ..config()
            },
            workers,
            TrainOptions {
                max_rounds: Some(3),
                documents: Some(fixture.policy(2)),
                ..Default::default()
            },
            |_| {},
        )
        .unwrap();
        assert!(!model.dictionary.rules.is_empty());
        // X and Y occur often, but only in one normalized group at records
        // zero and 128. Neither record may take the unique-group shortcut.
        let mut forbidden = vec![false; 257];
        forbidden[b'X' as usize] = true;
        forbidden[b'Y' as usize] = true;
        for &(left, right) in &model.dictionary.rules {
            let contains = forbidden[left as usize] || forbidden[right as usize];
            assert!(!contains, "one-group-only fragment was admitted");
            forbidden.push(contains);
        }
        let bytes = model.to_bytes().unwrap();
        if let Some(reference) = &reference {
            assert_eq!(&bytes, reference);
        }
        reference = Some(bytes);
    }
}

#[test]
fn selection_requires_distinct_groups_not_lines_or_repeated_group_records() {
    let repeated = b"xyxyxyxy ab\n".repeat(80);
    let shared = b"ab\n".repeat(4);
    let mut docs = vec![
        (repeated.as_slice(), 1),
        (shared.as_slice(), 2),
        (shared.as_slice(), 1),
    ];
    for group in 3..=8 {
        docs.push((shared.as_slice(), group));
    }
    let f = Fixture::new(&docs);
    let mut reference = None;
    for workers in [1, 4] {
        let options = TrainOptions {
            max_rules: Some(128),
            max_rounds: Some(12),
            documents: Some(f.policy(8)),
            ..Default::default()
        };
        let (model, report) = Frozen::train_with_options(
            &byte_model(),
            &f.0.join("train"),
            &config(),
            workers,
            options,
            |_| {},
        )
        .unwrap();
        assert_eq!(report.minimum_count, 8);
        assert!(report.document_filtered_candidates > 0);
        let mut expansions: Vec<Vec<u16>> = (0..257).map(|id| vec![id]).collect();
        for &(a, b) in &model.dictionary.rules {
            expansions.push(
                [
                    expansions[a as usize].as_slice(),
                    expansions[b as usize].as_slice(),
                ]
                .concat(),
            );
        }
        assert!(expansions[257..]
            .iter()
            .any(|e| e.windows(2).any(|p| p == [97, 98])));
        for expansion in &expansions[257..] {
            let mut support = HashSet::new();
            for &(raw, group) in &docs {
                if raw.split_inclusive(|&b| b == b'\n').any(|field| {
                    let labels: Vec<_> = field.iter().map(|&b| u16::from(b)).chain([256]).collect();
                    labels.windows(expansion.len()).any(|w| w == expansion)
                }) {
                    support.insert(group);
                }
            }
            assert!(
                support.len() >= 8,
                "rule {expansion:?} supported by only {} groups",
                support.len()
            );
            assert!(!expansion.contains(&u16::from(b'x')));
        }
        let raw = fs::read(f.0.join("train")).unwrap();
        let mut archive = Cursor::new(vec![]);
        model.encode(raw.as_slice(), &mut archive, 17).unwrap();
        let mut reader = Reader::new(Cursor::new(archive.into_inner())).unwrap();
        let mut actual = vec![];
        reader.decode_to(false, &mut actual).unwrap();
        assert_eq!(actual, raw);
        if let Some(reference) = &reference {
            assert_eq!(&model.to_bytes().unwrap(), reference);
        } else {
            reference = Some(model.to_bytes().unwrap());
        }
    }
}

#[test]
fn ineligible_higher_frequency_pair_cannot_block_a_cross_document_overlap() {
    let mut artifact = b"ab ".repeat(1000);
    artifact.extend(b"abc\n");
    let shared = b"bc\n".repeat(4);
    let mut docs = vec![(artifact.as_slice(), 1)];
    for group in 2..=9 {
        docs.push((shared.as_slice(), group));
    }
    let f = Fixture::new(&docs);
    let options = TrainOptions {
        max_rounds: Some(1),
        documents: Some(f.policy(8)),
        ..Default::default()
    };
    let (model, report) = Frozen::train_with_options(
        &byte_model(),
        &f.0.join("train"),
        &config(),
        4,
        options,
        |_| {},
    )
    .unwrap();
    assert_eq!(report.rounds, 1);
    assert!(!model.dictionary.rules.contains(&(97, 98)));
    assert!(model.dictionary.rules.contains(&(98, 99)));
}

#[test]
fn one_document_policy_is_compatible_and_round_limits_publish_valid_models() {
    let raw = b"abababab\n".repeat(100);
    let f = Fixture::new(&[(&raw, 1)]);
    let options = TrainOptions {
        max_rounds: Some(1),
        ..Default::default()
    };
    let (a, report) = Frozen::train_with_options(
        &byte_model(),
        &f.0.join("train"),
        &config(),
        4,
        options.clone(),
        |_| {},
    )
    .unwrap();
    assert_eq!(report.rounds, 1);
    assert!(report.stopped_at_round_limit);
    let (b, _) = Frozen::train_with_options(
        &byte_model(),
        &f.0.join("train"),
        &config(),
        4,
        TrainOptions {
            documents: Some(f.policy(1)),
            ..options
        },
        |_| {},
    )
    .unwrap();
    assert_eq!(a.to_bytes().unwrap(), b.to_bytes().unwrap());
    assert!(Frozen::from_bytes(&a.to_bytes().unwrap()).is_ok());
    assert!(Frozen::train_with_options(
        &byte_model(),
        &f.0.join("train"),
        &config(),
        1,
        TrainOptions {
            max_rounds: Some(0),
            ..Default::default()
        },
        |_| {}
    )
    .is_err());
}

#[test]
fn rejects_holdout_indexes_and_misaligned_document_boundaries() {
    let f = Fixture::new(&[(b"a\n", 1), (b"b\n", 2)]);
    let mut index = fs::read(f.0.join("index")).unwrap();
    index[12] = 1;
    fs::write(f.0.join("holdout"), &index).unwrap();
    let options = TrainOptions {
        documents: Some(Policy {
            index: f.0.join("holdout"),
            minimum_groups: 2,
        }),
        ..Default::default()
    };
    assert!(Frozen::train_with_options(
        &byte_model(),
        &f.0.join("train"),
        &config(),
        1,
        options,
        |_| {}
    )
    .err()
    .unwrap()
    .contains("training index"));
    index[12] = 0;
    index[24 + 84..24 + 92].copy_from_slice(&1u64.to_le_bytes());
    index[24 + 104 + 76..24 + 104 + 84].copy_from_slice(&1u64.to_le_bytes());
    index[24 + 104 + 84..24 + 104 + 92].copy_from_slice(&3u64.to_le_bytes());
    fs::write(f.0.join("index"), index).unwrap();
    let options = TrainOptions {
        documents: Some(f.policy(2)),
        ..Default::default()
    };
    let config = Config {
        minimum_count: 2,
        ..config()
    };
    assert!(Frozen::train_with_options(
        &byte_model(),
        &f.0.join("train"),
        &config,
        1,
        options,
        |_| {}
    )
    .is_err());
}

#[test]
fn cli_reports_effective_policy_and_does_not_publish_bad_policy() {
    let raw = b"abababab\n".repeat(16);
    let f = Fixture::new(&[(&raw, 1)]);
    fs::write(f.0.join("bytes.brt"), byte_model().to_bytes()).unwrap();
    let run = |args: &[&str]| {
        Command::new(env!("CARGO_BIN_EXE_brepair"))
            .current_dir(&f.0)
            .args(args)
            .output()
            .unwrap()
    };
    let output = run(&[
        "codec-train",
        "bytes.brt",
        "train",
        "bounded.brc",
        "--min-count",
        "4",
        "--score",
        "frequency",
        "--max-rounds",
        "1",
        "--document-index",
        "index",
        "--min-rule-documents",
        "1",
    ]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("\"stopped_at_round_limit\":true"));
    // Each of the 16 fields contains four AB occurrences and one newline/EOF.
    // The independent one-round result has 80 roots in twelve 64-bit words.
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("\"phase\":\"packing\",\"round\":0,\"done\":0,"));
    assert!(stderr.contains("\"event\":\"seed\",\"completed\":{\"round\":0,"));
    assert!(stderr.contains("\"phase\":\"initial_pair_count\""));
    let progress = stderr
        .lines()
        .find_map(|line| line.strip_prefix("codec progress "))
        .unwrap();
    for field in [
        "\"round\":1,",
        "\"grammar_rules\":2,",
        "\"rules_created\":2,",
        "\"symbols\":80,",
        "\"packed_bytes\":96,",
        "\"symbol_bits\":9,",
        "\"minimum_selected_count\":16,",
        "\"maximum_selected_count\":64,",
    ] {
        assert!(progress.contains(field), "missing {field} in {progress}");
    }
    assert!(String::from_utf8_lossy(&output.stdout).contains("\"initial_helper_rules\":0"));
    let output = run(&[
        "codec-train",
        "bytes.brt",
        "train",
        "scaled.brc",
        "--min-count",
        "4",
        "--min-count-per-gb",
        "1000000000",
    ]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let model = Frozen::from_bytes(&fs::read(f.0.join("scaled.brc")).unwrap()).unwrap();
    assert!(model.dictionary.rules.is_empty());
    assert!(String::from_utf8_lossy(&output.stdout)
        .contains(&format!("\"minimum_count\":{}", raw.len())));
    let output = run(&[
        "codec-train",
        "bytes.brt",
        "train",
        "invalid.brc",
        "--min-rule-documents",
        "8",
    ]);
    assert!(!output.status.success());
    assert!(!f.0.join("invalid.brc").exists());
}
