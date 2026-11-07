use bre_pair::{
    codec::{EncodeOptions, Frozen, OptimizeOptions},
    huffman::Codebook,
    huffman_archive::{Encoder, Reader, Totals},
    huffman_dictionary::{byte_model, Dictionary},
    huffman_pipeline,
};
use std::{
    fs,
    io::Cursor,
    process::Command,
    sync::atomic::{AtomicU64, Ordering},
};
static NEXT: AtomicU64 = AtomicU64::new(0);

fn fixture() -> (Frozen, Vec<u64>, Vec<u32>) {
    // cd is an orphan; ab is an uncoded helper still needed by aba.
    let dictionary =
        Dictionary::new(byte_model(), vec![(99, 100), (97, 98), (258, 97)], true).unwrap();
    let stream: Vec<_> = (0..256).flat_map(|_| [259, 256]).collect();
    let mut counts = vec![0u64; dictionary.expansions.len()];
    for &id in &stream {
        counts[id as usize] += 1;
    }
    let training = Totals {
        bytes: 768,
        fields: 256,
        tokens: 1024,
        symbols: 512,
    };
    let weights: Vec<_> = counts.iter().map(|&n| n.max(1)).collect();
    (
        Frozen {
            dictionary,
            book: Codebook::from_counts(&weights).unwrap(),
            training,
        },
        counts,
        stream,
    )
}
#[test]
fn omits_unused_codes_compacts_orphans_keeps_parent_recipes_and_every_ai_id() {
    let (model, counts, _) = fixture();
    let result = model
        .optimize_counts(&counts, OptimizeOptions { max_trials: 16 })
        .unwrap();
    assert!(result.report.accepted > 0);
    assert_eq!(result.report.after_rules, 2);
    assert_eq!(result.report.removed_zero_use_codes, 2);
    assert_eq!(result.mapping[257], u32::MAX);
    assert_eq!(result.mapping[258], u32::MAX);
    assert_eq!(result.mapping[259], 258);
    assert!(!result.model.book.symbols.contains(&257));
    assert!(result.model.book.symbols.contains(&258));
    for id in 0..257 {
        assert!(result.model.book.symbols.contains(&id));
    }
    assert_eq!(
        result.model.dictionary.model.to_bytes(),
        model.dictionary.model.to_bytes()
    );
    let model = Frozen::from_bytes(&result.model.to_bytes().unwrap()).unwrap();
    assert_eq!(
        model
            .prepare()
            .unwrap()
            .segment(&[97, 98, 97, 256])
            .unwrap(),
        vec![258, 256]
    );
    for workers in [1, 4] {
        let raw = b"aba\n\xff";
        let mut archive = Cursor::new(vec![]);
        model
            .encode_with_options(
                raw.as_slice(),
                &mut archive,
                2,
                EncodeOptions {
                    workers,
                    ..Default::default()
                },
            )
            .unwrap();
        for ids in [false, true] {
            let mut reader = Reader::new(Cursor::new(archive.get_ref().clone())).unwrap();
            let mut decoded = vec![];
            reader.decode_to(ids, &mut decoded).unwrap();
            let expected = if ids {
                raw.split_inclusive(|&b| b == b'\n')
                    .flat_map(|f| f.iter().map(|&b| b as u32).chain([256]))
                    .flat_map(u32::to_le_bytes)
                    .collect()
            } else {
                raw.to_vec()
            };
            assert_eq!(decoded, expected);
            let reader = Reader::new(Cursor::new(archive.get_ref().clone())).unwrap();
            let mut decoded = vec![];
            huffman_pipeline::decode(reader, ids, &mut decoded, Default::default()).unwrap();
            assert_eq!(decoded, expected);
        }
    }
}
#[test]
fn positive_use_removal_can_improve_global_huffman_payload() {
    // A:1 B:2 X:1 AB:1 costs 10 bits. Expanding AB changes all lengths:
    // A:2 B:3 X:1 costs 9 bits, despite emitting one additional symbol.
    let before = [1, 2, 1, 1];
    let after = [2, 3, 1, 0];
    let cost = |counts: &[u64]| {
        Codebook::from_counts(counts)
            .unwrap()
            .lengths()
            .map(|(id, n)| counts[id as usize] * u64::from(n))
            .sum::<u64>()
    };
    assert_eq!(cost(&before), 10);
    assert_eq!(cost(&after), 9);
}
#[test]
fn preserves_u64_histograms_rejects_mismatches_and_never_removes_ai_fallback() {
    let (mut model, mut counts, _) = fixture();
    counts[259] = 1u64 << 33;
    counts[256] = 1u64 << 33;
    model.training = Totals {
        bytes: 3 * (1u64 << 33),
        fields: 1u64 << 33,
        tokens: 4 * (1u64 << 33),
        symbols: 2 * (1u64 << 33),
    };
    let r = model
        .optimize_counts(&counts, OptimizeOptions::default())
        .unwrap();
    assert_eq!(r.model.training.tokens, 4 * (1u64 << 33));
    assert!(r.report.after_payload_bits > u128::from(u32::MAX));
    assert!(model
        .optimize_counts(&counts[..counts.len() - 1], OptimizeOptions::default())
        .is_err());
    counts[97] = 1;
    assert!(model
        .optimize_counts(&counts, OptimizeOptions::default())
        .is_err());
    let (mut invalid, _, _) = fixture();
    invalid.book = Codebook::from_counts(&[1, 1]).unwrap();
    assert!(invalid.to_bytes().is_err());
    let (model, counts, _) = fixture();
    let r = model
        .optimize_counts(&counts, OptimizeOptions { max_trials: 0 })
        .unwrap();
    assert_eq!(r.model.to_bytes().unwrap(), model.to_bytes().unwrap());
}
#[test]
fn cli_training_archive_recode_preserves_bytes_ids_and_static_table() {
    let path = std::env::temp_dir().join(format!(
        "brepair-optimize-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir(&path).unwrap();
    let (model, _, stream) = fixture();
    fs::write(path.join("model.brc"), model.to_bytes().unwrap()).unwrap();
    let mut archive = Cursor::new(vec![]);
    let mut encoder =
        Encoder::with_codebook(&mut archive, &model.dictionary, 1048576, &model.book).unwrap();
    encoder.extend(&stream).unwrap();
    encoder
        .finish(model.training.bytes, model.training.fields)
        .unwrap();
    fs::write(path.join("train.brh"), archive.get_ref()).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_brepair"))
        .current_dir(&path)
        .args([
            "codec-optimize",
            "model.brc",
            "train.brh",
            "optimized.brc",
            "--recode",
            "optimized.brh",
            "--report",
            "report.json",
            "--max-trials",
            "16",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let optimized = Frozen::from_bytes(&fs::read(path.join("optimized.brc")).unwrap()).unwrap();
    for ids in [false, true] {
        let mut before = Reader::new(Cursor::new(archive.get_ref().clone())).unwrap();
        let mut a = vec![];
        before.decode_to(ids, &mut a).unwrap();
        let mut after =
            Reader::new(Cursor::new(fs::read(path.join("optimized.brh")).unwrap())).unwrap();
        assert_eq!(
            after.static_codebook().unwrap().to_bytes(),
            optimized.book.to_bytes()
        );
        let mut b = vec![];
        after.decode_to(ids, &mut b).unwrap();
        assert_eq!(a, b);
    }
    assert!(
        fs::metadata(path.join("optimized.brh")).unwrap().len() < archive.get_ref().len() as u64
    );
    fs::remove_dir_all(&path).unwrap();
}
