//! Explicit resource regression: cargo test --test single_field_memory -- --ignored.
use std::{
    fs,
    io::{Read, Write},
    path::PathBuf,
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

struct Scratch(PathBuf);
impl Drop for Scratch {
    fn drop(&mut self) {
        let base = self.0.parent().unwrap().canonicalize().unwrap();
        if let Ok(target) = self.0.canonicalize() {
            assert!(target.starts_with(&base) && target != base);
            let _ = fs::remove_dir_all(target);
        }
    }
}

#[test]
#[ignore = "writes a 100 MB resource fixture; run explicitly"]
fn one_field_fits_without_geometric_packing_peak() {
    let base = std::env::var_os("BREPAIR_TEST_TMP")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    fs::create_dir_all(&base).unwrap();
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = Scratch(base.join(format!(
        "brepair-single-field-{}-{nonce}",
        std::process::id()
    )));
    fs::create_dir(&root.0).unwrap();
    let raw = root.0.join("raw");
    let mut file = fs::File::create(&raw).unwrap();
    let bytes = [b'a'; 65536];
    let mut remaining = 100_000_000usize;
    while remaining != 0 {
        let n = remaining.min(bytes.len());
        file.write_all(&bytes[..n]).unwrap();
        remaining -= n;
    }
    drop(file);
    let binary = env!("CARGO_BIN_EXE_brepair");
    for (name, fields, budget) in [("bounded", "single", "256MB"), ("control", "lines", "2GB")] {
        let output = Command::new(binary)
            .args(["compress"])
            .arg(&raw)
            .arg(root.0.join(name))
            .args([
                "--fields",
                fields,
                "--memory",
                budget,
                "--threads",
                "1",
                "--score",
                "frequency",
            ])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    // The input has no LF, so both policies represent precisely one field.
    assert_eq!(
        fs::read(root.0.join("bounded")).unwrap(),
        fs::read(root.0.join("control")).unwrap()
    );
    let decoded = root.0.join("decoded");
    assert!(Command::new(binary)
        .arg("huffman-decode")
        .arg(root.0.join("bounded"))
        .arg(&decoded)
        .args(["--memory", "256MB"])
        .status()
        .unwrap()
        .success());
    let mut input = fs::File::open(decoded).unwrap();
    let mut actual = [0u8; 65536];
    let mut count = 0u64;
    loop {
        let n = input.read(&mut actual).unwrap();
        if n == 0 {
            break;
        }
        assert_eq!(&actual[..n], &bytes[..n]);
        count += n as u64;
    }
    assert_eq!(count, 100_000_000);
}
