use bre_pair::document_support::{self, Policy};
use bre_pair::tokenizer::{Model, Token, Tokenizer};
use bre_pair::vocabulary::{self, RefinementOptions};
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

static ID: AtomicU64 = AtomicU64::new(0);
struct Fixture(PathBuf);
impl Fixture {
    fn new(docs: &[(&[u8], u8)]) -> Self {
        let root = std::env::temp_dir().join(format!(
            "brepair-documents-{}-{}",
            std::process::id(),
            ID.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&root).unwrap();
        let mut raw = Vec::new();
        let mut index = b"BREDOC01".to_vec();
        index.extend(104u32.to_le_bytes());
        index.extend(0u32.to_le_bytes());
        index.extend((docs.len() as u64).to_le_bytes());
        for &(bytes, group) in docs {
            let mut record = [0u8; 104];
            record[32..64].fill(group);
            record[76..84].copy_from_slice(&(raw.len() as u64).to_le_bytes());
            record[84..92].copy_from_slice(&(bytes.len() as u64).to_le_bytes());
            let fields = bytes.iter().filter(|&&b| b == b'\n').count() as u64
                + u64::from(!bytes.ends_with(b"\n"));
            record[92..100].copy_from_slice(&fields.to_le_bytes());
            index.extend(record);
            raw.extend(bytes);
        }
        fs::write(root.join("train"), raw).unwrap();
        fs::write(root.join("index"), index).unwrap();
        Self(root)
    }
    fn train(&self) -> PathBuf {
        self.0.join("train")
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
fn model(words: &[&[u8]]) -> Model {
    let mut tokens = Model::new(
        (0..=255u8)
            .map(|b| Token {
                bytes: vec![b],
                ends_field: false,
            })
            .chain(std::iter::once(Token {
                bytes: vec![],
                ends_field: true,
            }))
            .collect(),
    )
    .unwrap()
    .tokens;
    tokens.extend(words.iter().map(|word| Token {
        bytes: word.to_vec(),
        ends_field: false,
    }));
    Model::new(tokens).unwrap()
}

#[test]
fn repeated_fields_and_normalized_variants_do_not_fake_independent_support() {
    let repeated = b"artifact artifact artifact\n".repeat(800);
    let f = Fixture::new(&[
        (&repeated, 1),
        (b"artifact\n", 1),
        (b"artifact\n", 2),
        (b"artifact\n", 3),
    ]);
    let tokenizer = Tokenizer::new(model(&[b"artifact"])).unwrap();
    for workers in [1, 4] {
        let found =
            document_support::scan(&f.train(), &tokenizer, None, workers, &f.policy(8)).unwrap();
        assert_eq!(found.scan.uses[257], 2403);
        assert_eq!(found.groups[257], 3);
        assert_eq!(found.scan.all.fields, 803);
        let plain = vocabulary::scan(
            &f.train(),
            &tokenizer,
            None,
            None,
            workers,
            1,
            0,
            true,
            true,
        )
        .unwrap();
        assert_eq!(found.scan.uses, plain.uses);
        assert_eq!(found.scan.all.tokens, plain.all.tokens);
    }
}

#[test]
fn active_pool_mapping_and_support_saturation_preserve_eof() {
    let f = Fixture::new(&[(b"word\n", 1), (b"word\n", 2), (b"word\n", 3)]);
    let pool = Tokenizer::new(model(&[b"unused", b"word"])).unwrap();
    let mut active = vec![1; 259];
    active[257] = 0;
    let found = document_support::scan(&f.train(), &pool, Some(&active), 3, &f.policy(2)).unwrap();
    assert_eq!((found.scan.uses[257], found.groups[257]), (0, 0));
    assert_eq!((found.scan.uses[258], found.groups[258]), (3, 2));
    assert_eq!(found.scan.uses[256], 3);
}

#[test]
fn rejects_holdout_index_bad_boundaries_and_trailing_input() {
    let f = Fixture::new(&[(b"word\n", 1)]);
    let tokenizer = Tokenizer::new(model(&[b"word"])).unwrap();
    let mut index = fs::read(f.0.join("index")).unwrap();
    index[12] = 1;
    fs::write(f.0.join("holdout"), index).unwrap();
    let p = Policy {
        index: f.0.join("holdout"),
        minimum_groups: 2,
    };
    assert!(document_support::scan(&f.train(), &tokenizer, None, 1, &p)
        .err()
        .unwrap()
        .contains("training index"));
    fs::write(f.train(), b"word\nextra\n").unwrap();
    assert!(document_support::scan(&f.train(), &tokenizer, None, 1, &f.policy(2)).is_err());
    let broken = Fixture::new(&[(b"word", 1), (b"word\n", 2)]);
    assert!(
        document_support::scan(&broken.train(), &tokenizer, None, 2, &broken.policy(2)).is_err()
    );
}

#[test]
fn document_refinement_replaces_a_high_frequency_one_document_artifact() {
    let noisy = b"artifact common word\n".repeat(30);
    let other = b"common word common word\n".repeat(8);
    let f = Fixture::new(&[(&noisy, 1), (&other, 2), (&other, 3)]);
    let initial = model(&[b"artifact", b"common"]);
    let pool = model(&[b"artifact", b"common", b"word"]);
    let options = RefinementOptions {
        workers: 3,
        minimum_uses: 8,
        max_rounds: 4,
    };
    let old =
        vocabulary::refine_bounded(pool.clone(), &initial, &f.train(), &options, None, |_| {})
            .unwrap();
    assert_eq!(old.model.tokens, initial.tokens);
    let improved = vocabulary::refine_with_documents(
        pool.clone(),
        &initial,
        &f.train(),
        &options,
        None,
        Some(&f.policy(2)),
        |_| {},
    )
    .unwrap();
    assert_eq!(improved.model.tokens.len(), 259);
    assert!(!improved.model.tokens.iter().any(|t| t.bytes == b"artifact"));
    assert!(improved.model.tokens.iter().any(|t| t.bytes == b"word"));
    assert_eq!(improved.minimum_learned_document_groups, Some(2));
    assert!(improved.after.tokens > improved.before.tokens); // An explicit quality constraint may cost training tokens.
    assert!(vocabulary::refine_with_documents(
        pool,
        &initial,
        &f.train(),
        &options,
        Some(old.before.tokens),
        Some(&f.policy(2)),
        |_| {}
    )
    .is_err());
}

#[test]
fn public_cli_document_refinement_is_atomic_and_checks_the_export() {
    use bre_pair::engine::Compressed;
    use bre_pair::grammar::{Grammar, EOF};
    use bre_pair::packed::{symbol_width, PackedSymbols};
    use std::process::Command;
    let noisy = b"artifact common word\n".repeat(30);
    let other = b"common word common word\n".repeat(8);
    let f = Fixture::new(&[(&noisy, 1), (&other, 2), (&other, 3)]);
    let mut grammar = Grammar::default();
    for word in [&b"artifact"[..], &b"common"[..], &b"word"[..]] {
        let mut left = word[0] as u32;
        for &byte in &word[1..] {
            left = grammar.add((left, byte as u32)).unwrap();
        }
    }
    let raw = fs::read(f.train()).unwrap();
    let mut symbols = PackedSymbols::new(symbol_width(grammar.alphabet_size()));
    for &byte in &raw {
        symbols.push(byte as u32);
        if byte == b'\n' {
            symbols.push(EOF);
        }
    }
    let source = Compressed {
        grammar,
        blocks: vec![symbols],
        input_bytes: raw.len() as u64,
        field_count: raw.iter().filter(|&&b| b == b'\n').count() as u64,
    };
    fs::write(
        f.0.join("source.brp"),
        bre_pair::archive::encode(&source).unwrap(),
    )
    .unwrap();
    fs::write(
        f.0.join("initial.brt"),
        model(&[b"artifact", b"common"]).to_bytes(),
    )
    .unwrap();
    let args = vec![
        "refine".into(),
        f.0.join("source.brp").into_os_string(),
        f.train().into_os_string(),
        f.0.join("initial.brt").into_os_string(),
        f.0.join("clean.brt").into_os_string(),
        "--document-index".into(),
        f.0.join("index").into_os_string(),
        "--min-token-documents".into(),
        "2".into(),
        "--min-token-uses".into(),
        "8".into(),
        "--memory".into(),
        "64MB".into(),
        "--threads".into(),
        "3".into(),
    ];
    let result = Command::new(env!("CARGO_BIN_EXE_brepair"))
        .args(&args)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(String::from_utf8_lossy(&result.stdout)
        .contains("\"minimum_learned_document_groups_lower_bound\":2"));
    let bytes = fs::read(f.0.join("clean.brt")).unwrap();
    let final_model = Model::from_bytes(&bytes).unwrap();
    assert_eq!(final_model.tokens.len(), 259);
    assert!(!final_model.tokens.iter().any(|t| t.bytes == b"artifact"));
    assert!(!Command::new(env!("CARGO_BIN_EXE_brepair"))
        .args(&args)
        .output()
        .unwrap()
        .status
        .success());
    assert_eq!(fs::read(f.0.join("clean.brt")).unwrap(), bytes);
}
