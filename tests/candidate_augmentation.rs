use bre_pair::{
    batch::PairCounter,
    grammar::{Grammar, EOF},
    tokenizer::{Model, Token, Tokenizer},
    vocabulary,
};
use std::{
    collections::HashSet,
    fs,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

struct Fixture(PathBuf);
impl Fixture {
    fn new(raw: &[u8]) -> Self {
        let id = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("bre-augmentation-{}-{id}", std::process::id()));
        fs::write(&path, raw).unwrap();
        Self(path)
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}
fn basis() -> Model {
    Model::new(
        (0..=255)
            .map(|b| Token {
                bytes: vec![b],
                ends_field: false,
            })
            .chain([Token {
                bytes: vec![],
                ends_field: true,
            }])
            .collect(),
    )
    .unwrap()
}

#[test]
fn raw_candidates_match_independent_pair_counter_including_runs_and_fields() {
    let mut raw = b"aaaaa\naabcdaafgcdaacgadaaee\nabcd\n\n".repeat(3);
    raw.extend((0..=255).cycle().take(8000));
    raw.extend([0, 255, 0, 255]);
    let input = Fixture::new(&raw);
    let grammar = Grammar::default();
    let mut counter = PairCounter::default();
    for field in raw.split_inclusive(|&b| b == b'\n') {
        for &b in field {
            counter.push(u32::from(b), &grammar);
        }
        counter.push(EOF, &grammar);
    }
    for minimum in [1, 2, 4, 8, 32] {
        let mut model = basis();
        let mut support = vec![0; 257];
        let count =
            vocabulary::augment_byte_pairs(&mut model, &mut support, &input.0, minimum).unwrap();
        let expected: HashSet<_> = counter
            .counts
            .iter()
            .filter(|(_, n)| **n >= minimum)
            .map(|(&(a, b), _)| Token {
                bytes: if b == EOF {
                    vec![a as u8]
                } else {
                    vec![a as u8, b as u8]
                },
                ends_field: b == EOF,
            })
            .collect();
        assert_eq!(
            model.tokens[257..].iter().cloned().collect::<HashSet<_>>(),
            expected
        );
        assert_eq!(count, expected.len());
        assert_eq!(support.len(), model.tokens.len());
        assert!(support.iter().all(|&n| n == 0));
        let before = model.tokens.clone();
        support[257] = 123456789;
        assert_eq!(
            vocabulary::augment_byte_pairs(&mut model, &mut support, &input.0, minimum).unwrap(),
            0
        );
        assert_eq!(model.tokens, before);
        assert_eq!(support[257], 123456789);
        Tokenizer::new(model).unwrap();
        let mut raw_model = basis();
        let mut raw_support = vec![0; 257];
        vocabulary::augment_byte_pairs_with_prior(
            &mut raw_model,
            &mut raw_support,
            &input.0,
            minimum,
            true,
        )
        .unwrap();
        for (i, token) in raw_model.tokens.iter().enumerate().skip(257) {
            let pair = if token.ends_field {
                (u32::from(token.bytes[0]), EOF)
            } else {
                (u32::from(token.bytes[0]), u32::from(token.bytes[1]))
            };
            assert_eq!(raw_support[i], counter.counts[&pair]);
        }
    }
}

#[test]
fn augmentation_streams_across_buffer_edges_and_preserves_field_endings() {
    let mut raw = vec![b'a'; 65537];
    raw.extend(b"bb\n");
    let input = Fixture::new(&raw);
    let mut model = basis();
    let mut support = vec![0; 257];
    vocabulary::augment_byte_pairs(&mut model, &mut support, &input.0, 32769).unwrap();
    assert_eq!(model.tokens.len(), 257); // floor(65537/2), not all overlaps.
    vocabulary::augment_byte_pairs(&mut model, &mut support, &input.0, 32768).unwrap();
    assert_eq!(model.tokens[257].bytes, b"aa");
    let input = Fixture::new(b"a\nb\n");
    let mut model = basis();
    let mut support = vec![0; 257];
    vocabulary::augment_byte_pairs(&mut model, &mut support, &input.0, 1).unwrap();
    assert!(model
        .tokens
        .iter()
        .any(|t| t.bytes == b"\n" && t.ends_field));
    assert!(!model.tokens.iter().any(|t| t.bytes == b"\nb"));
    assert!(!model.tokens.iter().any(|t| t.bytes == b"a" && t.ends_field));
    assert_eq!(
        vocabulary::augment_byte_pairs(&mut model, &mut support, &input.0, 0).unwrap_err(),
        "invalid byte-pair augmentation threshold or support length"
    );
}
