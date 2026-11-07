use bre_pair::{
    engine::{self, Compressed, Config},
    grammar::{Grammar, BASE_SYMBOLS, EOF},
    packed::{symbol_width, PackedSymbols},
    tokenizer::{Model, Token, Tokenizer},
    vocabulary::{self, RefinementOptions},
};
use std::{
    collections::HashMap,
    fs,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

struct Source(PathBuf);
impl Source {
    fn new(bytes: &[u8]) -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("bre-refinement-{}-{nonce}.raw", std::process::id()));
        fs::write(&path, bytes).unwrap();
        Self(path)
    }
}
impl Drop for Source {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

fn model(extra: &[(&[u8], bool)]) -> Model {
    let mut tokens: Vec<_> = (0..=255)
        .map(|byte| Token {
            bytes: vec![byte],
            ends_field: false,
        })
        .collect();
    tokens.push(Token {
        bytes: vec![],
        ends_field: true,
    });
    tokens.extend(extra.iter().map(|&(bytes, ends_field)| Token {
        bytes: bytes.to_vec(),
        ends_field,
    }));
    Model::new(tokens).unwrap()
}

#[test]
fn source_verification_rejects_same_size_different_bytes_and_field_boundaries() {
    let fields = vec![b"abc\n".to_vec(), b"\n".to_vec(), b"xyz".to_vec()];
    let source = engine::compress(&fields, &Config::default(), |_| {})
        .unwrap()
        .compressed;
    let valid = Source::new(b"abc\n\nxyz");
    vocabulary::verify_training_source(&source, &valid.0).unwrap();
    let wrong = Source::new(b"abc\n\nxYz");
    assert!(vocabulary::verify_training_source(&source, &wrong.0)
        .unwrap_err()
        .contains("bytes differ"));
    let wrong_boundaries = engine::compress(&[b"abc\n\nxyz".to_vec()], &Config::default(), |_| {})
        .unwrap()
        .compressed;
    assert!(vocabulary::verify_training_source(&wrong_boundaries, &valid.0).is_err());
    let longer = Source::new(b"abc\n\nxyzZ");
    assert!(vocabulary::verify_training_source(&source, &longer.0).is_err());
}

#[test]
fn candidate_support_matches_individual_derivation_visits_and_deduplication() {
    let mut grammar = Grammar::default();
    let ab = grammar.add((97, 98)).unwrap();
    let bc = grammar.add((98, 99)).unwrap();
    let abc = grammar.add((ab, 99)).unwrap();
    let duplicate = grammar.add((97, bc)).unwrap();
    let ending = grammar.add((duplicate, EOF)).unwrap();
    let aa = grammar.add((97, 97)).unwrap();
    let aaaa = grammar.add((aa, aa)).unwrap();
    let mut block = PackedSymbols::new(symbol_width(grammar.alphabet_size()));
    for symbol in [abc, ending, aaaa, EOF, EOF] {
        block.push(symbol);
    }
    let source = Compressed {
        grammar,
        blocks: vec![block],
        input_bytes: 10,
        field_count: 3,
    };
    let mut visited = vec![0u64; source.grammar.alphabet_size()];
    let mut stack = Vec::new();
    for symbol in source.blocks[0].iter() {
        stack.push(symbol);
        while let Some(id) = stack.pop() {
            visited[id as usize] += 1;
            if id as usize >= BASE_SYMBOLS {
                let rule = &source.grammar.rules[id as usize - BASE_SYMBOLS];
                stack.push(rule.left);
                stack.push(rule.right);
            }
        }
    }
    assert_eq!(visited[aa as usize], 2);
    assert_eq!(visited[97], 6);
    let mut expected = HashMap::<Token, u64>::new();
    for (id, &count) in visited.iter().enumerate() {
        let mut terminals = Vec::new();
        source
            .grammar
            .visit_terminals(id as u32, |s| terminals.push(s));
        let ends_field = terminals.last() == Some(&EOF);
        let token = Token {
            bytes: terminals
                .into_iter()
                .filter(|&s| s != EOF)
                .map(|s| s as u8)
                .collect(),
            ends_field,
        };
        *expected.entry(token).or_default() += count;
    }
    let (pool, counts) = Model::candidates(&source, 128).unwrap();
    assert_eq!(pool.tokens.len(), expected.len());
    for (token, count) in pool.tokens.iter().zip(counts) {
        assert_eq!(count, expected[token]);
    }
}

#[test]
fn final_refinement_recounts_after_insertions_expose_more_unused_entries() {
    let mut raw = b"abab\n".repeat(8);
    raw.extend(b"xyxy\n".repeat(8));
    let source = Source::new(&raw);
    let pool = model(&[
        (b"ab", false),
        (b"never", false),
        (b"abab\n", true),
        (b"xy", false),
    ]);
    let initial = model(&[(b"ab", false), (b"never", false)]);
    let mut expected = None;
    for workers in [1, 4] {
        let result = vocabulary::refine(
            pool.clone(),
            &initial,
            &source.0,
            &RefinementOptions {
                workers,
                minimum_uses: 8,
                max_rounds: 8,
            },
            |_| {},
        )
        .unwrap();
        assert_eq!(result.model.tokens.len(), 259);
        assert!(result.rounds.len() >= 2); // New whole-field token consumes "ab".
        assert_eq!(result.minimum_learned_uses, Some(8));
        assert!(result.after.tokens < result.before.tokens);
        let bytes = result.model.to_bytes();
        if let Some(expected) = &expected {
            assert_eq!(&bytes, expected);
        } else {
            expected = Some(bytes);
        }
        let tokenizer = Tokenizer::new(result.model).unwrap();
        let counts =
            vocabulary::scan(&source.0, &tokenizer, None, None, workers, 1, 0, true, true).unwrap();
        assert!(counts.uses[257..].iter().all(|&n| n >= 8));
        assert!(counts.uses[..257].contains(&0)); // Unused mandatory bytes survive.
    }
}

#[test]
fn low_usage_is_pruned_using_final_counts_even_when_it_covered_training_text() {
    let mut raw = b"abab\n".repeat(8);
    raw.extend(b"xyxy\n".repeat(8));
    raw.extend(b"unique\n");
    let source = Source::new(&raw);
    let pool = model(&[
        (b"ab", false),
        (b"unique", false),
        (b"abab\n", true),
        (b"xy", false),
    ]);
    let initial = model(&[(b"ab", false), (b"unique", false)]);
    let result = vocabulary::refine(
        pool,
        &initial,
        &source.0,
        &RefinementOptions::default(),
        |_| {},
    )
    .unwrap();
    assert!(result.model.tokens[257..]
        .iter()
        .all(|t| t.bytes != b"unique"));
    assert!(result.minimum_learned_uses.unwrap() >= 8);
    assert!(result.after.tokens <= result.before.tokens);
}

#[test]
fn insufficient_useful_candidates_and_invalid_thresholds_fail_explicitly() {
    let source = Source::new(b"a\na\n");
    let pool = model(&[(b"never", false)]);
    let error = vocabulary::refine(
        pool.clone(),
        &pool,
        &source.0,
        &RefinementOptions::default(),
        |_| {},
    )
    .err()
    .unwrap();
    assert!(error.contains("cannot fill"));
    let error = vocabulary::refine(
        pool.clone(),
        &pool,
        &source.0,
        &RefinementOptions {
            minimum_uses: 0,
            ..RefinementOptions::default()
        },
        |_| {},
    )
    .err()
    .unwrap();
    assert!(error.contains("invalid refinement"));
}

#[test]
fn final_usage_cleanup_rejects_a_net_training_regression() {
    let long = b"abcdefghijklmnopqrstuvwx\n";
    let mut raw = long.to_vec();
    raw.extend(b"ab\n".repeat(8));
    let source = Source::new(&raw);
    let pool = model(&[(long, true), (b"ab", false)]);
    let initial = model(&[(long, true)]);
    let error = vocabulary::refine(
        pool,
        &initial,
        &source.0,
        &RefinementOptions::default(),
        |_| {},
    )
    .err()
    .unwrap();
    assert!(error.contains("increase the training token count"));
}

#[test]
fn exchange_cleanup_obeys_its_explicit_corpus_ceiling() {
    let long = b"abcdefghijklmnopqrstuvwx\n";
    let mut raw = long.to_vec();
    raw.extend(b"ab\n".repeat(8));
    let source = Source::new(&raw);
    let pool = model(&[(long, true), (b"ab", false)]);
    let proposal = model(&[(long, true)]);
    let accepted = vocabulary::refine_bounded(
        pool.clone(),
        &proposal,
        &source.0,
        &RefinementOptions::default(),
        Some(50),
        |_| {},
    )
    .unwrap();
    assert!(accepted.after.tokens > accepted.before.tokens);
    assert!(accepted.after.tokens <= 50);
    assert_eq!(accepted.model.tokens.len(), proposal.tokens.len());
    assert_eq!(accepted.minimum_learned_uses, Some(9));
    assert!(vocabulary::refine_bounded(
        pool,
        &proposal,
        &source.0,
        &RefinementOptions::default(),
        Some(accepted.after.tokens - 1),
        |_| {},
    )
    .is_err());
}
