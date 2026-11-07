use bre_pair::{
    codec::{periodic, Frozen, TrainOptions},
    engine::Config,
    huffman_archive::Reader,
    huffman_dictionary::byte_model,
    scoring::Model as Score,
};
use std::{
    fs,
    io::Cursor,
    sync::atomic::{AtomicU64, Ordering},
};
static NEXT: AtomicU64 = AtomicU64::new(0);

#[test]
fn checkpoints_recount_identically_and_reparsing_preserves_exact_ai_ids() {
    let root = std::env::temp_dir().join(format!(
        "brepair-periodic-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir(&root).unwrap();
    let raw = b"abracadabra abcabcabc banana mississippi\n".repeat(16);
    fs::write(root.join("input"), &raw).unwrap();
    let config = Config {
        buffer_bytes: 2_000_000,
        minimum_count: 2,
        score: Score::WeightedPmi,
        ..Default::default()
    };
    let vocabulary = byte_model();
    let options = TrainOptions {
        max_rounds: Some(12),
        ..Default::default()
    };
    let baseline = periodic::train(
        &vocabulary,
        &root.join("input"),
        &config,
        1,
        &options,
        0,
        false,
        |_, _| {},
        |_| {},
    )
    .unwrap();
    let ordinary = Frozen::train_with_options(
        &vocabulary,
        &root.join("input"),
        &config,
        1,
        options.clone(),
        |_| {},
    )
    .unwrap()
    .0;
    assert_eq!(
        baseline.model.to_bytes().unwrap(),
        ordinary.to_bytes().unwrap()
    );
    let restarted = periodic::train(
        &vocabulary,
        &root.join("input"),
        &config,
        4,
        &options,
        2,
        false,
        |_, _| {},
        |_| {},
    )
    .unwrap();
    assert_eq!(
        baseline.model.to_bytes().unwrap(),
        restarted.model.to_bytes().unwrap()
    );
    let mut hashes = Vec::new();
    for workers in [1, 4] {
        let result = periodic::train(
            &vocabulary,
            &root.join("input"),
            &config,
            workers,
            &options,
            2,
            true,
            |_, _| {},
            |_| {},
        )
        .unwrap();
        assert!(!result.events.is_empty());
        assert!(result.events.iter().all(|e| e.after_bits <= e.before_bits));
        let archive = root.join(format!("output-{workers}.brh"));
        result.write_archive(&archive).unwrap();
        let bytes = fs::read(archive).unwrap();
        let mut reader = Reader::new(Cursor::new(bytes.clone())).unwrap();
        let mut decoded = Vec::new();
        reader.decode_to(false, &mut decoded).unwrap();
        assert_eq!(decoded, raw);
        let mut reader = Reader::new(Cursor::new(bytes)).unwrap();
        let mut ids = Vec::new();
        reader.decode_to(true, &mut ids).unwrap();
        let expected: Vec<_> = raw
            .split_inclusive(|&b| b == b'\n')
            .flat_map(|f| f.iter().map(|&b| b as u32).chain([256]))
            .flat_map(u32::to_le_bytes)
            .collect();
        assert_eq!(ids, expected);
        hashes.push(result.model.to_bytes().unwrap());
    }
    assert_eq!(hashes[0], hashes[1]);
    assert!(periodic::train(
        &vocabulary,
        &root.join("input"),
        &config,
        1,
        &TrainOptions::default(),
        2,
        true,
        |_, _| {},
        |_| {}
    )
    .is_err());
    for path in fs::read_dir(&root).unwrap() {
        fs::remove_file(path.unwrap().path()).unwrap();
    }
    fs::remove_dir(root).unwrap();
}

#[test]
fn periodic_reparsing_preserves_composite_ai_tokens_and_their_boundaries() {
    use bre_pair::tokenizer::{Model, Token, Tokenizer};
    let root = std::env::temp_dir().join(format!(
        "brepair-periodic-ai-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir(&root).unwrap();
    let mut tokens = byte_model().tokens;
    for text in [b"ab".as_slice(), b"bc", b"xa", b"abcabc"] {
        tokens.push(Token {
            bytes: text.to_vec(),
            ends_field: false,
        });
    }
    let vocabulary = Model::new(tokens).unwrap();
    let raw = b"xabc abcabc abcx xabcabcx\nabc xabc\n".repeat(32);
    fs::write(root.join("input"), &raw).unwrap();
    let tokenizer = Tokenizer::new(vocabulary.clone()).unwrap();
    let expected: Vec<_> = raw
        .split_inclusive(|&b| b == b'\n')
        .flat_map(|f| tokenizer.encode(f))
        .flat_map(u32::to_le_bytes)
        .collect();
    let config = Config {
        buffer_bytes: 2_000_000,
        minimum_count: 2,
        score: Score::WeightedPmi,
        ..Default::default()
    };
    let run = periodic::train(
        &vocabulary,
        &root.join("input"),
        &config,
        4,
        &TrainOptions {
            max_rounds: Some(12),
            ..Default::default()
        },
        1,
        true,
        |_, _| {},
        |_| {},
    )
    .unwrap();
    assert!(!run.events.is_empty());
    run.write_archive(&root.join("output.brh")).unwrap();
    let mut reader = Reader::new(Cursor::new(fs::read(root.join("output.brh")).unwrap())).unwrap();
    let mut ids = Vec::new();
    reader.decode_to(true, &mut ids).unwrap();
    assert_eq!(ids, expected);
    for path in fs::read_dir(&root).unwrap() {
        fs::remove_file(path.unwrap().path()).unwrap();
    }
    fs::remove_dir(root).unwrap();
}
