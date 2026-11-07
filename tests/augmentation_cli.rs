use bre_pair::tokenizer::{Model, Tokenizer};
use std::{
    fs,
    path::PathBuf,
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

struct Fixture(PathBuf);
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn public_augmented_model_round_trips_and_rejects_invalid_options() {
    let id = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir =
        Fixture(std::env::temp_dir().join(format!("bre-augment-cli-{}-{id}", std::process::id())));
    fs::create_dir(&dir.0).unwrap();
    let input = dir.0.join("input");
    let archive = dir.0.join("source.brp");
    let model = dir.0.join("model.brt");
    let report = dir.0.join("model.json");
    let rejected = dir.0.join("invalid.brt");
    let raw = b"ab\ncd\nef\ngh\nij\nkl\nmn\nop\n".repeat(30);
    fs::write(&input, &raw).unwrap();
    let run = |args: &[&str]| {
        Command::new(env!("CARGO_BIN_EXE_brepair"))
            .args(args)
            .output()
            .unwrap()
    };
    let trained = run(&[
        "train",
        input.to_str().unwrap(),
        archive.to_str().unwrap(),
        "--min-count",
        "2",
    ]);
    assert!(
        trained.status.success(),
        "{}",
        String::from_utf8_lossy(&trained.stderr)
    );
    let built = run(&[
        "vocab",
        archive.to_str().unwrap(),
        input.to_str().unwrap(),
        model.to_str().unwrap(),
        "--vocab-size",
        "264",
        "--min-token-uses",
        "2",
        "--augment-byte-pairs",
        "--raw-byte-pair-support",
        "--report",
        report.to_str().unwrap(),
    ]);
    assert!(
        built.status.success(),
        "{}",
        String::from_utf8_lossy(&built.stderr)
    );
    assert!(fs::read_to_string(&report)
        .unwrap()
        .contains("\"raw_byte_pair_support\":true"));
    let tokenizer = Tokenizer::new(Model::from_bytes(&fs::read(model).unwrap()).unwrap()).unwrap();
    assert_eq!(tokenizer.model().tokens.len(), 264);
    for field in raw.split_inclusive(|&b| b == b'\n') {
        assert_eq!(
            tokenizer
                .model()
                .decode_field(&tokenizer.encode(field))
                .unwrap(),
            field
        );
    }
    let bad = run(&[
        "vocab",
        archive.to_str().unwrap(),
        input.to_str().unwrap(),
        rejected.to_str().unwrap(),
        "--raw-byte-pair-support",
    ]);
    assert!(!bad.status.success());
    assert!(!rejected.exists());
    let bad = run(&[
        "vocab",
        archive.to_str().unwrap(),
        input.to_str().unwrap(),
        rejected.to_str().unwrap(),
        "--augment-byte-pairs",
        "--max-token-bytes",
        "1",
    ]);
    assert!(!bad.status.success());
    assert!(!rejected.exists());
}
