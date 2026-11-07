use bre_pair::{
    archive,
    engine::{self, Config},
    grammar::EOF,
    joint,
    tokenizer::{Model, Token},
};
use std::{
    collections::BTreeSet,
    fs,
    path::PathBuf,
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("bre-joint-{}-{nonce}", std::process::id()));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn model() -> Model {
    let mut tokens: Vec<_> = (0..=255)
        .map(|b| Token {
            bytes: vec![b],
            ends_field: false,
        })
        .collect();
    tokens.push(Token {
        bytes: vec![],
        ends_field: true,
    });
    for (bytes, ends_field) in [
        (b"ab".as_slice(), false),
        (b"bc", false),
        (b"abc\n", true),
        (b"\xff\0", false),
    ] {
        tokens.push(Token {
            bytes: bytes.to_vec(),
            ends_field,
        });
    }
    Model::new(tokens).unwrap()
}
fn word(raw: &[u8], at: &mut usize) -> u64 {
    let n = u64::from_le_bytes(raw[*at..*at + 8].try_into().unwrap());
    *at += 8;
    n
}

#[test]
fn exported_graph_contains_every_literal_match_and_exact_eof_boundaries() {
    let dir = Fixture::new();
    let fields = [b"abc\n".to_vec(), vec![255, 0, b'\n'], b"\n".to_vec()];
    let raw = fields.concat();
    let source = dir.0.join("source");
    fs::write(&source, &raw).unwrap();
    let pool = model();
    let initial = Model::new(pool.tokens[..257].to_vec()).unwrap();
    joint::prepare(
        pool.clone(),
        &initial,
        &source,
        &dir.0.join("out"),
        &joint::Options {
            sample_bytes: 100,
            maximum_edges: 1000,
            workers: 4,
            minimum_uses: 1,
        },
    )
    .unwrap();
    let graph = fs::read(dir.0.join("out/graph.bin")).unwrap();
    assert_eq!(&graph[..8], b"BRTGRF01");
    let mut at = 8;
    assert_eq!(word(&graph, &mut at), 3);
    let mut offset = 0;
    for (index, field) in fields.iter().enumerate() {
        assert_eq!(word(&graph, &mut at), index as u64);
        assert_eq!(word(&graph, &mut at), offset);
        let len = word(&graph, &mut at) as usize;
        let n = word(&graph, &mut at);
        assert_eq!(&graph[at..at + len], field);
        at += len;
        let mut actual = BTreeSet::new();
        for _ in 0..n {
            let start = word(&graph, &mut at);
            let end = word(&graph, &mut at);
            let id = u32::from_le_bytes(graph[at..at + 4].try_into().unwrap());
            at += 4;
            assert!(actual.insert((start, end, id)));
        }
        let labels: Vec<_> = field.iter().map(|&b| u32::from(b)).chain([EOF]).collect();
        let mut expected = BTreeSet::new();
        for start in 0..labels.len() {
            for (id, token) in pool.tokens.iter().enumerate() {
                let expansion: Vec<_> = token
                    .bytes
                    .iter()
                    .map(|&b| u32::from(b))
                    .chain(token.ends_field.then_some(EOF))
                    .collect();
                if labels[start..].starts_with(&expansion) {
                    expected.insert((start as u64, (start + expansion.len()) as u64, id as u32));
                }
            }
        }
        assert_eq!(actual, expected);
        offset += len as u64;
    }
    assert_eq!(at, graph.len());
    assert!(joint::prepare(
        pool,
        &initial,
        &source,
        &dir.0.join("out"),
        &joint::Options {
            sample_bytes: 100,
            maximum_edges: 1000,
            workers: 1,
            minimum_uses: 1,
        }
    )
    .is_err());
}

#[test]
fn calibration_budgets_skip_whole_fields_and_source_mismatch_is_rejected() {
    let dir = Fixture::new();
    let source = dir.0.join("source");
    fs::write(&source, b"abc\n\n").unwrap();
    let pool = model();
    let initial = Model::new(pool.tokens[..257].to_vec()).unwrap();
    joint::prepare(
        pool.clone(),
        &initial,
        &source,
        &dir.0.join("out"),
        &joint::Options {
            sample_bytes: 100,
            maximum_edges: 3,
            workers: 1,
            minimum_uses: 1,
        },
    )
    .unwrap();
    let graph = fs::read(dir.0.join("out/graph.bin")).unwrap();
    let mut at = 8;
    assert_eq!(word(&graph, &mut at), 1);
    assert_eq!(word(&graph, &mut at), 1);
    assert_eq!(word(&graph, &mut at), 4);
    assert_eq!(word(&graph, &mut at), 1);
    assert_eq!(word(&graph, &mut at), 2);
    assert_eq!(graph[at], b'\n');
    let compressed = engine::compress(
        &[b"Abc\n".to_vec(), b"\n".to_vec()],
        &Config::default(),
        |_| {},
    )
    .unwrap()
    .compressed;
    let archive_path = dir.0.join("wrong.brp");
    fs::write(&archive_path, archive::encode(&compressed).unwrap()).unwrap();
    let initial_path = dir.0.join("initial.brt");
    fs::write(&initial_path, initial.to_bytes()).unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_prepare-joint"))
        .args([
            archive_path.to_str().unwrap(),
            source.to_str().unwrap(),
            initial_path.to_str().unwrap(),
            dir.0.join("rejected").to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("bytes differ"));
    assert!(!dir.0.join("rejected").exists());
}
