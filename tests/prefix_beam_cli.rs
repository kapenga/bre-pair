use bre_pair::archive;
use std::{
    fs,
    path::PathBuf,
    process::{Command, Output},
    time::{SystemTime, UNIX_EPOCH},
};

struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("bre-prefix-{}-{nonce}", std::process::id()));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn command(&self, mode: &str, name: &str, extra: &[&str]) -> Output {
        Command::new(env!("CARGO_BIN_EXE_prefix-repair"))
            .arg(mode)
            .arg(self.0.join("input"))
            .arg(self.0.join(name))
            .args(extra)
            .output()
            .unwrap()
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn discovery_and_all_continuations_round_trip_without_overwrite() {
    let dir = Fixture::new();
    let mut raw = b"aaaaa abcabcababc abc def def def\n".repeat(10);
    raw.extend([0, 255, b'\n']);
    fs::write(dir.0.join("input"), &raw).unwrap();
    let result = dir.command(
        "discover",
        "prefixes",
        &["--rules", "4", "--width", "3", "--minimum", "2"],
    );
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let report = fs::read(dir.0.join("prefixes/discovery.json")).unwrap();
    let fields: Vec<_> = raw
        .split_inclusive(|&b| b == b'\n')
        .map(<[u8]>::to_vec)
        .collect();
    for name in ["greedy", "beam-00", "beam-01", "beam-02"] {
        let prefix = dir.0.join(format!("prefixes/{name}.rules"));
        let result = dir.command(
            "train",
            name,
            &[
                "--prefix",
                prefix.to_str().unwrap(),
                "--minimum",
                "2",
                "--learned-budget",
                "20",
            ],
        );
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let compressed =
            archive::decode(&fs::read(dir.0.join(name).join("candidates.brp")).unwrap()).unwrap();
        assert_eq!(compressed.decode().unwrap(), fields);
        assert!(compressed.grammar.rules.len() <= 20);
        let saved = fs::read(dir.0.join(name).join("training.json")).unwrap();
        assert!(!dir.command("train", name, &[]).status.success());
        assert_eq!(
            fs::read(dir.0.join(name).join("training.json")).unwrap(),
            saved
        );
    }
    assert!(!dir.command("discover", "prefixes", &[]).status.success());
    assert_eq!(
        fs::read(dir.0.join("prefixes/discovery.json")).unwrap(),
        report
    );
}

#[test]
fn rejects_size_caps_invalid_rules_and_wrong_source_counts() {
    let dir = Fixture::new();
    fs::write(dir.0.join("input"), b"aaaaab\naaaaab\n").unwrap();
    assert!(!dir
        .command("discover", "oversize", &["--max-input-bytes", "1"])
        .status
        .success());
    assert!(!dir.0.join("oversize").exists());
    for (name, contents) in [
        ("eof-left", "BREPAIR-PREFIX 1 12\n256 97\n"),
        ("forward", "BREPAIR-PREFIX 1 12\n257 97\n"),
        ("wrong-size", "BREPAIR-PREFIX 1 999\n97 97\n"),
    ] {
        let path = dir.0.join(format!("{name}.rules"));
        fs::write(&path, contents).unwrap();
        assert!(!dir
            .command("train", name, &["--prefix", path.to_str().unwrap()])
            .status
            .success());
        assert!(!dir.0.join(name).join("candidates.brp").exists());
    }
}
