use bre_pair::{
    archive,
    engine::Config,
    repeated, scoring, targeted,
    tokenizer::{Model, Token},
    vocabulary,
};
use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};

static NEXT: AtomicUsize = AtomicUsize::new(0);
struct Corpus(PathBuf);
impl Corpus {
    fn new(bytes: &[u8]) -> Self {
        let path = std::env::temp_dir().join(format!(
            "bre-pair-repeat-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::write(&path, bytes).unwrap();
        Self(path)
    }
}
impl Drop for Corpus {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

fn model(extra: &[(&[u8], bool)]) -> Model {
    Model::new(
        (0..256)
            .map(|b| Token {
                bytes: vec![b as u8],
                ends_field: false,
            })
            .chain([Token {
                bytes: vec![],
                ends_field: true,
            }])
            .chain(extra.iter().map(|(b, end)| Token {
                bytes: b.to_vec(),
                ends_field: *end,
            }))
            .collect(),
    )
    .unwrap()
}

#[test]
fn seed_segmentation_and_discovery_preserve_binary_bytes_fields_and_candidates() {
    let bytes = b"abcdefabcdef abcdef\nabcdef\n\n\0\xffabcdefabcdef\n".repeat(32);
    let fields: Vec<_> = bytes
        .split_inclusive(|&b| b == b'\n')
        .map(<[u8]>::to_vec)
        .collect();
    let corpus = Corpus::new(&bytes);
    let seed = model(&[(b"abcdef", false), (b"abcdef\n", true)]);
    let (source, membership) = repeated::prepare(&corpus.0, &seed, 7, 1_000_000).unwrap();
    assert_eq!(source.decode().unwrap(), fields);
    assert_eq!(membership.iter().filter(|&&b| b).count(), seed.tokens.len());
    let mut baseline = None;
    for workers in [1, 4] {
        let result = repeated::discover(
            &corpus.0,
            &seed,
            &Config {
                minimum_count: 4,
                score: scoring::Model::WeightedPmi,
                ..Config::default()
            },
            workers,
            seed.tokens.len() + 7,
            128,
            |_| {},
        )
        .unwrap();
        let source = &result.training.training.compressed;
        assert_eq!(source.decode().unwrap(), fields);
        assert_eq!(
            archive::decode(&archive::encode(source).unwrap())
                .unwrap()
                .decode()
                .unwrap(),
            fields
        );
        assert!(result.pool.tokens.len() <= seed.tokens.len() + 7);
        assert!(seed.tokens.iter().all(|t| result.pool.tokens.contains(t)));
        assert!(!result.pool.tokens.iter().any(|t| t.bytes == b"abc"));
        assert!(!result.pool.tokens.iter().any(|t| t.bytes == b"def"));
        let current = (result.pool.to_bytes(), result.support);
        if let Some(expected) = &baseline {
            assert_eq!(&current, expected);
        } else {
            baseline = Some(current);
        }
    }
}

#[test]
fn a_reactivated_helper_consumes_a_candidate_slot_but_not_a_new_rule() {
    let bytes = b"abXabYabZ".repeat(64);
    let corpus = Corpus::new(&bytes);
    let seed = model(&[(b"abcd", false)]);
    let result = repeated::discover(
        &corpus.0,
        &seed,
        &Config {
            minimum_count: 8,
            ..Config::default()
        },
        4,
        seed.tokens.len() + 1,
        128,
        |_| {},
    )
    .unwrap();
    assert!(result.training.stopped_at_rule_limit);
    assert_eq!(result.training.training.rounds.len(), 1);
    assert_eq!(result.training.training.rounds[0].new_rules, 0);
    assert_eq!(
        result.training.training.compressed.grammar.rules.len(),
        result.helper_rules
    );
    assert!(result.pool.tokens.iter().any(|t| t.bytes == b"ab"));
    assert!(!result.pool.tokens.iter().any(|t| t.bytes == b"cd"));
    assert_eq!(result.pool.tokens.len(), seed.tokens.len() + 1);
    assert_eq!(
        result.training.training.compressed.decode().unwrap(),
        vec![bytes]
    );
}

#[test]
fn empty_corpus_and_invalid_limits_are_handled_without_losing_the_seed() {
    let corpus = Corpus::new(b"");
    let seed = model(&[(b"abcdef", false)]);
    assert!(repeated::prepare(&corpus.0, &seed, 0, 1_000_000).is_err());
    assert!(repeated::discover(
        &corpus.0,
        &seed,
        &Config::default(),
        1,
        seed.tokens.len(),
        128,
        |_| {}
    )
    .is_err());
    assert!(repeated::discover(
        &corpus.0,
        &seed,
        &Config::default(),
        1,
        seed.tokens.len() + 1,
        2,
        |_| {}
    )
    .is_err());
    let empty = repeated::discover(
        &corpus.0,
        &seed,
        &Config::default(),
        4,
        seed.tokens.len() + 1,
        128,
        |_| {},
    )
    .unwrap();
    assert_eq!(empty.initial_tokens, 0);
    assert!(empty
        .training
        .training
        .compressed
        .decode()
        .unwrap()
        .is_empty());
    assert!(seed.tokens.iter().all(|t| empty.pool.tokens.contains(t)));
    let populated = Corpus::new(b"abcdefabcdef\n");
    assert!(repeated::prepare(&populated.0, &seed, 100, 0).is_err());
}

#[test]
fn cycle_cli_keeps_the_frozen_seed_when_no_improvement_is_available() {
    let bytes = b"abcdef\n".repeat(64);
    let corpus = Corpus::new(&bytes);
    let seed = model(&[(b"abcdef\n", true)]);
    let model_path = corpus.0.with_extension("brt");
    let output = corpus.0.with_extension("cycle");
    fs::write(&model_path, seed.to_bytes()).unwrap();
    let result = std::process::Command::new(env!("CARGO_BIN_EXE_cycle-tokenizer"))
        .arg(&corpus.0)
        .arg(&model_path)
        .arg(&output)
        .args(["--threads", "4"])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(fs::read(output.join("model.brt")).unwrap(), seed.to_bytes());
    let report = fs::read_to_string(output.join("report.json")).unwrap();
    assert!(report.contains("\"accepted\":false"));
    assert!(report.contains("\"selection_failed\":false"));
    assert!(report.contains("\"seed_training_tokens\":64"));
    assert!(report.contains("\"proposal_training_tokens\":64"));
    // The CLI must refuse reuse of an existing output directory.
    assert!(
        !std::process::Command::new(env!("CARGO_BIN_EXE_cycle-tokenizer"))
            .arg(&corpus.0)
            .arg(&model_path)
            .arg(&output)
            .output()
            .unwrap()
            .status
            .success()
    );
    fs::remove_file(model_path).unwrap();
    fs::remove_dir_all(output).unwrap();
}

#[test]
fn exact_removal_ranking_matches_independent_subset_reparsing_and_context_changes() {
    use bre_pair::tokenizer::Tokenizer;
    let bytes = b"abc\nabcd abc\nabc\n\0\xffabc\n".repeat(16);
    let corpus = Corpus::new(&bytes);
    let seed = model(&[(b"ab", false), (b"bc", false), (b"abcd", false)]);
    let baseline = vocabulary::scan(
        &corpus.0,
        &Tokenizer::new(seed.clone()).unwrap(),
        None,
        None,
        4,
        1,
        0,
        true,
        true,
    )
    .unwrap();
    let mut expected = None;
    for workers in [1, 4] {
        let rank = targeted::rank(&corpus.0, &seed, &baseline.uses, 3, workers, u64::MAX).unwrap();
        assert_eq!(rank.sample_bytes, bytes.len() as u64);
        assert_eq!(rank.sample_stride, 1);
        for entry in &rank.entries {
            let mut active = vec![1; seed.tokens.len()];
            active[entry.id as usize] = 0;
            let reduced = Tokenizer::new(seed.subset(&active).unwrap()).unwrap();
            let tokens: u64 = bytes
                .split_inclusive(|&b| b == b'\n')
                .map(|f| reduced.encode(f).len() as u64)
                .sum();
            assert_eq!(entry.sample_uses, baseline.uses[entry.id as usize]);
            assert_eq!(entry.sample_penalty, tokens - baseline.all.tokens);
            assert!(entry.sample_penalty <= entry.upper_bound);
        }
        let ab = rank.entries.iter().find(|r| r.id == 257).unwrap();
        assert_eq!(ab.sample_penalty, 0); // a|bc is as short as ab|c.
        assert!(ab.upper_bound > 0);
        let current: Vec<_> = rank
            .entries
            .iter()
            .map(|r| (r.id, r.sample_uses, r.sample_penalty, r.estimated_penalty))
            .collect();
        if let Some(previous) = &expected {
            assert_eq!(&current, previous);
        } else {
            expected = Some(current);
        }
    }
    assert!(targeted::rank(&corpus.0, &seed, &baseline.uses, 0, 1, 100).is_err());
    assert!(targeted::rank(&corpus.0, &seed, &baseline.uses, 3, 0, 100).is_err());
}

#[test]
fn targeted_discovery_blocks_expansions_across_different_binary_trees_and_restores_candidates() {
    let bytes = b"abcd".repeat(64);
    let corpus = Corpus::new(&bytes);
    let seed = model(&[(b"abcd", false), (b"bcd", false)]);
    let mut previous = None;
    for workers in [1, 4] {
        let result = repeated::discover_with_options(
            &corpus.0,
            &seed,
            &Config {
                minimum_count: 4,
                ..Config::default()
            },
            &repeated::DiscoveryOptions {
                workers,
                candidate_limit: seed.tokens.len() + 8,
                max_token_bytes: 128,
                collapse: vec![257],
                block_rounds: 2,
                shake: None,
            },
            |_| {},
        )
        .unwrap();
        assert!(result.blocked_candidates > 0);
        assert!(result.pool.tokens.contains(&seed.tokens[257]));
        assert!(result.initial_tokens > 64);
        let source = &result.training.training.compressed;
        assert_eq!(source.decode().unwrap(), vec![bytes.clone()]);
        let warm_rules = result
            .training
            .training
            .rounds
            .iter()
            .take(2)
            .next_back()
            .unwrap()
            .total_rules;
        for offset in result.helper_rules..warm_rules {
            let mut labels = Vec::new();
            source
                .grammar
                .visit_terminals((257 + offset) as u32, |s| labels.push(s));
            assert_ne!(
                labels,
                b"abcd".iter().map(|&b| b as u32).collect::<Vec<_>>()
            );
        }
        let current = (result.pool.to_bytes(), result.support);
        if let Some(expected) = &previous {
            assert_eq!(&current, expected);
        } else {
            previous = Some(current);
        }
    }
    for collapse in [vec![0], vec![256], vec![999], vec![257, 257]] {
        assert!(repeated::prepare_collapsed(&corpus.0, &seed, &collapse, 100, 1_000_000).is_err());
    }
}

#[test]
fn blocked_discovery_lifts_the_warmup_when_it_would_stall() {
    let bytes = b"x\n".repeat(32);
    let corpus = Corpus::new(&bytes);
    let seed = model(&[(b"x\n", true)]);
    let result = repeated::discover_with_options(
        &corpus.0,
        &seed,
        &Config::default(),
        &repeated::DiscoveryOptions {
            workers: 4,
            candidate_limit: seed.tokens.len() + 10,
            max_token_bytes: 128,
            collapse: vec![257],
            block_rounds: 10,
            shake: None,
        },
        |_| {},
    )
    .unwrap();
    assert!(result.block_lifted_early);
    assert!(result.blocked_candidates > 0);
    assert_eq!(
        result.training.training.compressed.decode().unwrap(),
        vec![b"x\n".to_vec(); 32]
    );
}

#[test]
fn targeted_cli_validates_the_rank_fingerprint_and_preserves_the_seed_fallback() {
    use bre_pair::tokenizer::fingerprint;
    let bytes = b"x\n".repeat(64);
    let corpus = Corpus::new(&bytes);
    let seed = model(&[(b"x\n", true)]);
    let model_path = corpus.0.with_extension("brt");
    let ranking = corpus.0.with_extension("collapse");
    let output = corpus.0.with_extension("targeted");
    fs::write(&model_path, seed.to_bytes()).unwrap();
    let command = || {
        let mut command = std::process::Command::new(env!("CARGO_BIN_EXE_cycle-tokenizer"));
        command
            .arg(&corpus.0)
            .arg(&model_path)
            .arg(&output)
            .arg("--collapse-ranking")
            .arg(&ranking)
            .args(["--collapse-count", "1", "--block-rounds", "10"]);
        command
    };
    fs::write(&ranking, "BRCOLLAPSE01 0\n257\n").unwrap();
    assert!(!command().output().unwrap().status.success());
    assert!(!output.exists());
    fs::write(
        &ranking,
        format!("BRCOLLAPSE01 {}\n257\n", fingerprint(&seed.to_bytes())),
    )
    .unwrap();
    let result = command().output().unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(fs::read(output.join("model.brt")).unwrap(), seed.to_bytes());
    let pool = Model::from_bytes(&fs::read(output.join("candidate-pool.brt")).unwrap()).unwrap();
    assert!(pool.tokens.contains(&seed.tokens[257]));
    assert_eq!(
        fs::read(output.join("candidate-support.u64"))
            .unwrap()
            .len(),
        pool.tokens.len() * 8
    );
    let report = fs::read_to_string(output.join("report.json")).unwrap();
    assert!(report.contains("\"block_lifted_early\":true"));
    assert!(report.contains("\"accepted\":false"));
    fs::remove_file(model_path).unwrap();
    fs::remove_file(ranking).unwrap();
    fs::remove_dir_all(output).unwrap();
}

#[test]
fn random_expansion_checks_the_packed_budget_before_allocating_outputs() {
    let words: Vec<Vec<u8>> = (0..300)
        .map(|i| format!("w{i:03}\n").into_bytes())
        .collect();
    let mut bytes = Vec::new();
    for word in &words {
        for _ in 0..8 {
            bytes.extend(word);
        }
    }
    let corpus = Corpus::new(&bytes);
    let extra: Vec<_> = words.iter().map(|w| (w.as_slice(), false)).collect();
    let seed = model(&extra);
    let (source, _) = repeated::prepare(&corpus.0, &seed, 65536, 1_000_000).unwrap();
    let budget = source.blocks.iter().map(|b| b.storage_bytes()).sum();
    let error = repeated::discover_with_options(
        &corpus.0,
        &seed,
        &Config {
            buffer_bytes: budget,
            ..Config::default()
        },
        &repeated::DiscoveryOptions {
            workers: 4,
            candidate_limit: 5000,
            max_token_bytes: 128,
            collapse: vec![],
            block_rounds: 0,
            shake: Some(bre_pair::shake::Options {
                fraction: 1.0,
                ..Default::default()
            }),
        },
        |_| {},
    )
    .err()
    .unwrap();
    assert!(error.contains("random shake exceeds"), "{error}");
}

#[test]
fn shaking_cli_reports_real_mid_training_pulses_and_rejects_incomplete_options() {
    let corpus = Corpus::new(&b"abcdefabcdef abcdef\n".repeat(64));
    let seed = model(&[(b"abcdef", false)]);
    let model_path = corpus.0.with_extension("brt");
    let output = corpus.0.with_extension("shaking");
    fs::write(&model_path, seed.to_bytes()).unwrap();
    let mut invalid = std::process::Command::new(env!("CARGO_BIN_EXE_cycle-tokenizer"));
    invalid
        .arg(&corpus.0)
        .arg(&model_path)
        .arg(&output)
        .args(["--shake-percent", "10"]);
    assert!(!invalid.output().unwrap().status.success());
    assert!(!output.exists());
    let result = std::process::Command::new(env!("CARGO_BIN_EXE_cycle-tokenizer"))
        .arg(&corpus.0)
        .arg(&model_path)
        .arg(&output)
        .args(["--shake-seed", "42"])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let report = fs::read_to_string(output.join("report.json")).unwrap();
    assert!(report.contains("\"shake_seed\":42"));
    assert_eq!(report.matches("\"after_batch\":").count(), 2);
    let saved = Model::from_bytes(&fs::read(output.join("model.brt")).unwrap()).unwrap();
    let tokenizer = bre_pair::tokenizer::Tokenizer::new(saved).unwrap();
    let field = b"abcdefabcdef abcdef\n";
    assert_eq!(
        tokenizer
            .model()
            .decode_field(&tokenizer.encode(field))
            .unwrap(),
        field
    );
    fs::remove_file(model_path).unwrap();
    fs::remove_dir_all(output).unwrap();
}
