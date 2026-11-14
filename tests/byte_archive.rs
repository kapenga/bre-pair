#![cfg(feature = "bytecode-experiments")]
use bre_pair::{
    huffman_archive::Encoder,
    huffman_dictionary::{byte_model, Dictionary},
    shared_huffman::{self, byte_archive},
};
use std::{
    fs,
    io::Cursor,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};
static NEXT: AtomicU64 = AtomicU64::new(0);
struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "bre-byte-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn source(dictionary: &Dictionary, ids: &[u32]) -> Vec<u8> {
    let mut counts = vec![0; dictionary.expansions.len()];
    let (mut bytes, mut fields) = (0, 0);
    for &id in ids {
        counts[id as usize] += 1;
        bytes += dictionary.expansions[id as usize].bytes;
        fields += u64::from(dictionary.expansions[id as usize].ends_field);
    }
    let mut output = Cursor::new(Vec::new());
    let mut encoder = Encoder::with_counts(&mut output, dictionary, 7, &counts).unwrap();
    encoder.extend(ids).unwrap();
    encoder.finish(bytes, fields).unwrap();
    output.into_inner()
}
#[test]
fn ordinary_and_compact_byte_archives_preserve_binary_empty_and_fused_eof_fields() {
    let temp = Temp::new();
    let dictionary = Dictionary::new(byte_model(), vec![(97, 98), (257, 256)], false).unwrap();
    let ids = [258, 256, 0, 255, 10, 256];
    let raw = b"ab\0\xff\n";
    fs::write(temp.0.join("source"), source(&dictionary, &ids)).unwrap();
    let variants = temp.0.join("variants");
    let report =
        shared_huffman::optimize(&temp.0.join("source"), &variants, Default::default()).unwrap();
    for (i, input) in std::iter::once(temp.0.join("source"))
        .chain(
            report
                .variants
                .iter()
                .map(|v| variants.join(format!("{}.brhv", v.name))),
        )
        .enumerate()
    {
        let source_bytes = fs::read(&input).unwrap();
        let source_symbols = u64::from_le_bytes(source_bytes[40..48].try_into().unwrap());
        let packed = temp.0.join(format!("packed-{i}"));
        byte_archive::pack(&input, &packed).unwrap();
        for workers in [1, 4] {
            let output = temp.0.join(format!("raw-{i}-{workers}"));
            let totals = byte_archive::decode(&packed, &output, workers).unwrap();
            assert_eq!(fs::read(&output).unwrap(), raw);
            assert_eq!(totals.fields, 3);
            assert_eq!(totals.symbols, source_symbols);
            assert!(byte_archive::decode(&packed, &output, workers).is_err());
            assert_eq!(fs::read(output).unwrap(), raw);
        }
    }
}
#[test]
fn bounded_chunks_parallelize_a_long_field_and_empty_archives_roundtrip() {
    let temp = Temp::new();
    let mut rules = vec![(97, 97)];
    for i in 0..23 {
        let id = 257 + i;
        rules.push((id, id));
    }
    let dictionary = Dictionary::new(byte_model(), rules, false).unwrap();
    let mut ids = vec![276; 9];
    ids.push(256);
    for (name, sequence, expected) in [
        ("large", ids, vec![97; 9 << 20]),
        ("oversized-symbol", vec![280, 256], vec![97; 16 << 20]),
        ("empty", vec![], vec![]),
    ] {
        let input = temp.0.join(name);
        fs::write(&input, source(&dictionary, &sequence)).unwrap();
        let packed = temp.0.join(format!("{name}.brb"));
        byte_archive::pack(&input, &packed).unwrap();
        for workers in [1, 4] {
            let output = temp.0.join(format!("{name}-{workers}"));
            byte_archive::decode(&packed, &output, workers).unwrap();
            assert_eq!(fs::read(output).unwrap(), expected);
        }
    }
}
#[test]
fn malformed_byte_archives_fail_without_publishing_output() {
    let temp = Temp::new();
    let dictionary = Dictionary::new(byte_model(), vec![], false).unwrap();
    let input = temp.0.join("source");
    fs::write(&input, source(&dictionary, &[97, 256])).unwrap();
    let packed = temp.0.join("good");
    byte_archive::pack(&input, &packed).unwrap();
    let bytes = fs::read(packed).unwrap();
    let mut cases = vec![bytes[..20].to_vec(), bytes[..bytes.len() - 1].to_vec()];
    let mut corrupt = bytes.clone();
    let model_len = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
    corrupt[80 + model_len + 16 + 40] ^= 1;
    cases.push(corrupt);
    let mut corrupt = bytes;
    corrupt[64] = 1;
    cases.push(corrupt);
    for (i, bad) in cases.iter().enumerate() {
        let archive = temp.0.join(format!("bad-{i}"));
        fs::write(&archive, bad).unwrap();
        let out = temp.0.join(format!("out-{i}"));
        assert!(byte_archive::decode(&archive, &out, 4).is_err());
        assert!(!out.exists());
    }
}
