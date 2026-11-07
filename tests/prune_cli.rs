use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use bre_pair::engine::Compressed;
use bre_pair::grammar::{Grammar, EOF};
use bre_pair::packed::{symbol_width, PackedSymbols};
use bre_pair::{archive, sequence_archive};

struct Workspace(PathBuf);

impl Drop for Workspace {
    fn drop(&mut self) {
        if let Ok(entries) = fs::read_dir(&self.0) {
            for entry in entries.flatten() {
                let _ = fs::remove_file(entry.path());
            }
        }
        let _ = fs::remove_dir(&self.0);
    }
}

fn run(mode: &str, input: &Path, output: &Path, options: &[&str]) {
    let result = Command::new(env!("CARGO_BIN_EXE_prune-repair"))
        .arg(mode)
        .arg(input)
        .arg(output)
        .args(options)
        .arg("--verify")
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
}

#[test]
fn pruning_cli_uses_merge_minimum_and_preserves_exact_boundary() {
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    let workspace = Workspace(std::env::temp_dir().join(format!(
        "bre-pair-pruning-{}-{}",
        std::process::id(),
        nonce.as_nanos()
    )));
    fs::create_dir(&workspace.0).unwrap();
    let mut grammar = Grammar::default();
    let mut symbols = Vec::new();
    for (pair, uses) in [
        ((97, 98), 3),
        ((99, 100), 4),
        ((101, 102), 7),
        ((103, 104), 8),
    ] {
        let symbol = grammar.add(pair).unwrap();
        symbols.extend(std::iter::repeat_n(symbol, uses));
    }
    symbols.push(EOF);
    let mut block = PackedSymbols::new(symbol_width(grammar.alphabet_size()));
    for symbol in symbols {
        block.push(symbol);
    }
    let source = Compressed {
        grammar,
        blocks: vec![block],
        input_bytes: 44,
        field_count: 1,
    };
    let input = workspace.0.join("input.brp");
    fs::write(&input, archive::encode(&source).unwrap()).unwrap();
    for (name, options, expected_rules, expected_tokens) in [
        ("default", vec![], 3, 26),
        ("minimum8", vec!["--min-count", "8"], 1, 37),
        (
            "override",
            vec!["--min-count", "8", "--max-uses", "0"],
            4,
            23,
        ),
    ] {
        let output = workspace.0.join(format!("{name}.mrp"));
        run("post", &input, &output, &options);
        let compact = sequence_archive::decode(&fs::read(output).unwrap()).unwrap();
        assert_eq!(compact.rules.len(), expected_rules);
        assert_eq!(compact.token_count(), expected_tokens);
        assert_eq!(
            compact.to_binary().unwrap().decode().unwrap(),
            source.decode().unwrap()
        );
    }

    let raw = workspace.0.join("input.raw");
    fs::write(&raw, b"abcabd abcabe abcabf the quick queen\n".repeat(20)).unwrap();
    for minimum in ["4", "8"] {
        let automatic = workspace.0.join(format!("during-{minimum}.mrp"));
        let explicit = workspace.0.join(format!("explicit-{minimum}.mrp"));
        let cutoff = (minimum.parse::<u64>().unwrap() - 1).to_string();
        run(
            "during",
            &raw,
            &automatic,
            &["--after-round", "1", "--min-count", minimum],
        );
        run(
            "during",
            &raw,
            &explicit,
            &[
                "--after-round",
                "1",
                "--min-count",
                minimum,
                "--max-uses",
                &cutoff,
            ],
        );
        assert_eq!(fs::read(automatic).unwrap(), fs::read(explicit).unwrap());
    }
}
