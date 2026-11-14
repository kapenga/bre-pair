use bre_pair::{
    huffman::Codebook,
    huffman_archive::Encoder,
    huffman_dictionary::{byte_model, Dictionary},
};
use std::{
    fs,
    io::Cursor,
    process::Command,
    sync::atomic::{AtomicU64, Ordering},
};
static NEXT: AtomicU64 = AtomicU64::new(0);
#[test]
fn explicit_cli_decoders_preserve_bytes_and_guard_long_codes() {
    let root = std::env::temp_dir().join(format!(
        "bre-backend-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir(&root).unwrap();
    let dictionary = Dictionary::new(byte_model(), vec![], false).unwrap();
    for long in [false, true] {
        let book = if long {
            let mut lengths = (0..31).map(|id| (id, id as u8 + 1)).collect::<Vec<_>>();
            lengths.extend([(31, 32), (256, 32)]);
            Codebook::from_lengths(&lengths).unwrap()
        } else {
            Codebook::from_counts(&vec![1; 257]).unwrap()
        };
        let mut stream = Cursor::new(Vec::new());
        let mut encoder = Encoder::with_codebook(&mut stream, &dictionary, 3, &book).unwrap();
        encoder.extend(&[0, 256]).unwrap();
        encoder.finish(1, 1).unwrap();
        let input = root.join(format!("input-{long}"));
        fs::write(&input, stream.into_inner()).unwrap();
        for decoder in ["scalar", "auto", "avx2"] {
            for workers in [1, 4] {
                let output = root.join(format!("out-{long}-{decoder}-{workers}"));
                let result = Command::new(env!("CARGO_BIN_EXE_brepair"))
                    .args([
                        "huffman-decode",
                        input.to_str().unwrap(),
                        output.to_str().unwrap(),
                        "--decoder",
                        decoder,
                        "--threads",
                        &workers.to_string(),
                        "--arena-mib",
                        if long { "0" } else { "1024" },
                    ])
                    .output()
                    .unwrap();
                assert!(
                    result.status.success(),
                    "{}",
                    String::from_utf8_lossy(&result.stderr)
                );
                assert_eq!(fs::read(output).unwrap(), [0]);
            }
        }
        let output = root.join(format!("invalid-{long}"));
        let status = Command::new(env!("CARGO_BIN_EXE_brepair"))
            .args([
                "huffman-decode",
                input.to_str().unwrap(),
                output.to_str().unwrap(),
                "--arena-mib",
                "1025",
            ])
            .output()
            .unwrap();
        assert!(!status.status.success());
        assert!(!output.exists());
    }
    fs::remove_dir_all(root).unwrap();
}
