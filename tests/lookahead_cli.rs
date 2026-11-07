use bre_pair::archive;
use std::{
    collections::BTreeMap,
    fs,
    path::PathBuf,
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        let n = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("bre-worlds-{}-{n}", std::process::id()));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn world_cli_exports_matched_archives_and_preserves_existing_outputs() {
    let dir = Fixture::new();
    let input = dir.0.join("input");
    let output = dir.0.join("worlds");
    let mut raw = b"abcabc\n".repeat(8);
    raw.extend([0, 255, b'\n']);
    fs::write(&input, &raw).unwrap();
    let run = || {
        Command::new(env!("CARGO_BIN_EXE_lookahead-worlds"))
            .args([
                input.to_str().unwrap(),
                output.to_str().unwrap(),
                "--minimum",
                "2",
                "--layer-rules",
                "2",
                "--checkpoints",
                "0,1",
                "--archives",
            ])
            .output()
            .unwrap()
    };
    let result = run();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let report = fs::read(output.join("report.json")).unwrap();
    assert!(String::from_utf8_lossy(&report).contains("\"rule_budget_matched\":true"));
    let fields: Vec<Vec<u8>> = raw
        .split_inclusive(|&b| b == b'\n')
        .map(<[u8]>::to_vec)
        .collect();
    let mut groups = BTreeMap::<String, Vec<usize>>::new();
    for entry in fs::read_dir(&output).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_some_and(|e| e == "brp") {
            let decoded = archive::decode(&fs::read(&path).unwrap()).unwrap();
            assert_eq!(decoded.decode().unwrap(), fields);
            let group = path
                .file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .split('-')
                .next()
                .unwrap()
                .to_string();
            groups
                .entry(group)
                .or_default()
                .push(decoded.grammar.rules.len());
        }
    }
    assert_eq!(groups.len(), 2);
    for rules in groups.values() {
        assert_eq!(rules.len(), 3);
        assert!(rules.iter().all(|&n| n == rules[0]));
    }
    assert!(!run().status.success());
    assert_eq!(fs::read(output.join("report.json")).unwrap(), report);
}

#[test]
fn world_cli_rejects_oversized_inputs_and_unreached_checkpoints() {
    let dir = Fixture::new();
    let input = dir.0.join("input");
    fs::write(&input, b"abcabc\n").unwrap();
    let output = dir.0.join("too-big");
    let result = Command::new(env!("CARGO_BIN_EXE_lookahead-worlds"))
        .args([
            input.to_str().unwrap(),
            output.to_str().unwrap(),
            "--max-input-bytes",
            "1",
        ])
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(!output.exists());
    let output = dir.0.join("too-late");
    let result = Command::new(env!("CARGO_BIN_EXE_lookahead-worlds"))
        .args([
            input.to_str().unwrap(),
            output.to_str().unwrap(),
            "--checkpoints",
            "999",
        ])
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(!output.join("report.json").exists());
    assert!(String::from_utf8_lossy(&result.stderr).contains("before any requested"));
}
