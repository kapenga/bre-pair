#![cfg(feature = "audit")]
use bre_pair::{
    archive,
    batch::Counting,
    engine::{self, Compressed, Config},
    field_engine,
    grammar::{BASE_SYMBOLS, EOF},
    packed::{symbol_width, PackedSymbols},
    scoring::Model,
    tokenizer,
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
#[ignore = "requires local book1; run --release --features audit --test book1_audit -- --ignored --nocapture"]
fn book1_every_batch_counts_and_models_match_full_recount_reference() {
    let path = std::env::var_os("BRE_PAIR_BOOK1")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| "C:/Data/corpus/calgary/book1".into());
    let raw = std::fs::read(&path).expect("set BRE_PAIR_BOOK1 to the Calgary book1 file");
    assert_eq!(raw.len(), 768_771);
    let lines: Vec<_> = raw
        .split_inclusive(|&b| b == b'\n')
        .map(<[u8]>::to_vec)
        .collect();
    for (policy, fields, minimum) in [
        ("single", vec![raw.clone()], 4),
        ("lines", lines.clone(), 4),
        ("lines", lines, 8),
    ] {
        let mut config = Config {
            buffer_bytes: 8 * 1024 * 1024,
            minimum_count: minimum,
            score: Model::WeightedPmi,
            pair_counting: Counting::Full,
            ..Config::default()
        };
        let reference = engine::compress(&fields, &config, |_| {}).unwrap();
        assert_eq!(reference.compressed.decode().unwrap(), fields);
        let expected = canonical(&reference.compressed);
        let settings = if policy == "single" {
            vec![(4, Counting::Incremental, false)]
        } else {
            vec![
                (1, Counting::Full, false),
                (4, Counting::Incremental, false),
                (4, Counting::Incremental, true),
                (4, Counting::Adaptive, false),
            ]
        };
        for (workers, mode, frequency_cutoff) in settings {
            config.pair_counting = mode;
            let actual = field_engine::compress_with_options(
                &fields,
                &config,
                workers,
                64,
                65536,
                field_engine::Options {
                    frequency_cutoff,
                    memory_saving: false,
                    last_seen_documents: false,
                    shake: None,
                    max_rounds: None,
                    max_rules: None,
                    stop_max_pair_count: None,
                },
                |_| {},
            )
            .unwrap();
            assert_eq!(actual.training.rounds.len(), reference.rounds.len());
            for (a, b) in actual.training.rounds.iter().zip(&reference.rounds) {
                assert_eq!(
                    (
                        a.candidates,
                        a.eligible,
                        a.selected,
                        a.new_rules,
                        a.before,
                        a.rewritten,
                        a.removed
                    ),
                    (
                        b.candidates,
                        b.eligible,
                        b.selected,
                        b.new_rules,
                        b.before,
                        b.rewritten,
                        b.removed
                    )
                );
            }
            assert_eq!(canonical(&actual.training.compressed), expected);
            assert_eq!(actual.training.compressed.decode().unwrap(), fields);
            println!("book1 policy={policy} min={minimum} workers={workers} mode={mode:?} cutoff={frequency_cutoff}: {} batches, {} rules, {} tokens; independent counts and archive match",
                actual.training.rounds.len(), actual.training.compressed.grammar.rules.len(), actual.training.compressed.token_count());
        }
    }
}

#[test]
#[ignore = "requires local book1; run with --features audit and --ignored"]
fn book1_candidate_support_matches_individual_tree_visits() {
    let path = std::env::var_os("BRE_PAIR_BOOK1")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| "C:/Data/corpus/calgary/book1".into());
    let raw = std::fs::read(&path).unwrap();
    assert_eq!(raw.len(), 768_771);
    let fields: Vec<_> = raw
        .split_inclusive(|&b| b == b'\n')
        .map(<[u8]>::to_vec)
        .collect();
    let source = field_engine::compress(
        &fields,
        &Config {
            buffer_bytes: 8 * 1024 * 1024,
            minimum_count: 4,
            score: Model::WeightedPmi,
            pair_counting: Counting::Incremental,
            ..Config::default()
        },
        4,
        64,
        65536,
        |_| {},
    )
    .unwrap()
    .training
    .compressed;
    bre_pair::vocabulary::verify_training_source(&source, &path).unwrap();
    let mut visited = vec![0u64; source.grammar.alphabet_size()];
    let mut stack = Vec::new();
    for block in &source.blocks {
        for id in block.iter() {
            stack.push(id);
            while let Some(id) = stack.pop() {
                visited[id as usize] += 1;
                if id as usize >= BASE_SYMBOLS {
                    let rule = &source.grammar.rules[id as usize - BASE_SYMBOLS];
                    stack.push(rule.left);
                    stack.push(rule.right);
                }
            }
        }
    }
    assert_eq!(visited[..256].iter().sum::<u64>(), raw.len() as u64);
    assert_eq!(visited[256], fields.len() as u64);
    let mut expected = std::collections::HashMap::<tokenizer::Token, u64>::new();
    for (id, &count) in visited.iter().enumerate() {
        let mut terminals = Vec::new();
        source
            .grammar
            .visit_terminals(id as u32, |s| terminals.push(s));
        let ends_field = terminals.last() == Some(&EOF);
        let bytes: Vec<_> = terminals
            .into_iter()
            .filter(|&s| s != EOF)
            .map(|s| s as u8)
            .collect();
        if bytes.len() <= 128 {
            *expected
                .entry(tokenizer::Token { bytes, ends_field })
                .or_default() += count;
        }
    }
    let (pool, support) = tokenizer::Model::candidates(&source, 128).unwrap();
    assert_eq!(pool.tokens.len(), expected.len());
    for (token, count) in pool.tokens.iter().zip(support) {
        assert_eq!(count, expected[token]);
    }
    println!("book1 candidate audit: all {} expansions/support counts match individual tree visits; byte and EOF totals match input", pool.tokens.len());
}
