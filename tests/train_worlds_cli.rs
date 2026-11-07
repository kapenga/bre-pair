use bre_pair::archive;
use std::{
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
        let path =
            std::env::temp_dir().join(format!("bre-train-worlds-{}-{n}", std::process::id()));
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
fn rolling_cli_saves_exact_candidate_checkpoint_and_converges_without_overwrite() {
    let dir = Fixture::new();
    let input = dir.0.join("input");
    let raw = b"abcabacababacdabcabc\n".repeat(8);
    fs::write(&input, &raw).unwrap();
    let fields: Vec<_> = raw
        .split_inclusive(|&b| b == b'\n')
        .map(<[u8]>::to_vec)
        .collect();
    for policy in ["standard", "control", "lookahead"] {
        let output = dir.0.join(policy);
        let run = || {
            Command::new(env!("CARGO_BIN_EXE_train-worlds"))
                .args([
                    input.to_str().unwrap(),
                    output.to_str().unwrap(),
                    "--minimum",
                    "2",
                    "--layer-rules",
                    "2",
                    "--candidate-budget",
                    "3",
                    "--policy",
                    policy,
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
        let candidate = archive::decode(&fs::read(output.join("candidates.brp")).unwrap()).unwrap();
        let final_state = archive::decode(&fs::read(output.join("final.brp")).unwrap()).unwrap();
        assert_eq!(candidate.grammar.rules.len(), 3);
        assert_eq!(candidate.decode().unwrap(), fields);
        assert_eq!(final_state.decode().unwrap(), fields);
        let report = fs::read(output.join("training.json")).unwrap();
        let text = String::from_utf8_lossy(&report);
        assert!(text.contains("\"final_converged\":true"));
        assert!(text.contains("\"candidate_stopped_at_rule_limit\":true"));
        let trace = fs::read_to_string(output.join("decisions.jsonl")).unwrap();
        assert!(trace.contains("\"phase\":\"candidates\""));
        assert!(trace.contains("\"phase\":\"convergence\""));
        assert!(!run().status.success());
        assert_eq!(fs::read(output.join("training.json")).unwrap(), report);
    }
}

#[test]
fn candidate_only_is_explicit_and_requires_a_limit() {
    let dir = Fixture::new();
    let input = dir.0.join("input");
    fs::write(&input, b"abcabacababacdabcabc\n".repeat(8)).unwrap();
    let missing = dir.0.join("missing");
    let result = Command::new(env!("CARGO_BIN_EXE_train-worlds"))
        .args([
            input.to_str().unwrap(),
            missing.to_str().unwrap(),
            "--candidate-only",
        ])
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(!missing.exists());
    let output = dir.0.join("limited");
    let result = Command::new(env!("CARGO_BIN_EXE_train-worlds"))
        .args([
            input.to_str().unwrap(),
            output.to_str().unwrap(),
            "--candidate-only",
            "--candidate-budget",
            "3",
            "--minimum",
            "2",
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(
        fs::read(output.join("candidates.brp")).unwrap(),
        fs::read(output.join("final.brp")).unwrap()
    );
    assert!(fs::read_to_string(output.join("training.json"))
        .unwrap()
        .contains("\"final_converged\":false"));
}
