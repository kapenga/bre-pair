use bre_pair::{
    archive,
    engine::{self, Config},
    huffman_archive::{self, Encoder, Reader},
    huffman_dictionary::{byte_model, Arena, Dictionary},
    huffman_pipeline::{self, Options as PipelineOptions},
    tokenizer::{Model, Token, Tokenizer},
};
use std::{
    fs,
    io::Cursor,
    path::PathBuf,
    process::Command,
    sync::atomic::{AtomicU64, Ordering},
};
static NEXT: AtomicU64 = AtomicU64::new(0);
#[test]
fn eof_records_need_no_counters_and_reject_malformed_payloads() {
    use bre_pair::{huffman::Codebook, huffman_archive::PreparedDecoder};
    use std::sync::Arc;
    for ai in [false, true] {
        let dictionary = Dictionary::new(byte_model(), vec![(97, 98), (257, 256)], ai).unwrap();
        let book = Arc::new(Codebook::from_counts(&vec![1; 259]).unwrap());
        for budget in [0, 65536] {
            let decoder =
                PreparedDecoder::with_codebook(&dictionary, Arc::clone(&book), false, budget)
                    .unwrap();
            for ids in [
                vec![256],
                vec![258],
                vec![97, 257, 258],
                vec![97; 64].into_iter().chain([256]).collect(),
            ] {
                let (payload, bits) = book.encode(&ids).unwrap();
                let expected: Vec<_> = ids
                    .iter()
                    .flat_map(|&id| match id {
                        256 => &b""[..],
                        257 | 258 => &b"ab"[..],
                        _ => &b"a"[..],
                    })
                    .copied()
                    .collect();
                let mut out = vec![7; 3];
                assert_eq!(
                    decoder
                        .decode_field(&dictionary, &payload, expected.len(), &mut out)
                        .unwrap(),
                    (bits, ids.len() as u64)
                );
                assert_eq!(out, expected);
                if !expected.is_empty() {
                    assert!(decoder
                        .decode_field(&dictionary, &payload, expected.len() - 1, &mut out)
                        .is_err());
                }
                for length in 0..payload.len() {
                    assert!(decoder
                        .decode_field(&dictionary, &payload[..length], 1024, &mut out)
                        .is_err());
                }
                let mut extra = payload.clone();
                extra.push(0);
                assert!(decoder
                    .decode_field(&dictionary, &extra, 1024, &mut out)
                    .is_err());
                if bits % 8 != 0 {
                    let mut bad = payload;
                    *bad.last_mut().unwrap() |= 1;
                    assert!(decoder
                        .decode_field(&dictionary, &bad, 1024, &mut out)
                        .is_err());
                }
            }
            let (no_eof, _) = book.encode(&[97, 257]).unwrap();
            assert!(decoder
                .decode_field(&dictionary, &no_eof, 1024, &mut Vec::new())
                .is_err());
            if ai {
                let token_decoder =
                    PreparedDecoder::with_codebook(&dictionary, Arc::clone(&book), true, budget)
                        .unwrap();
                assert!(token_decoder
                    .decode_field(&dictionary, &[0], 1024, &mut Vec::new())
                    .is_err());
            }
        }
    }
    let dictionary = Dictionary::new(byte_model(), vec![], false).unwrap();
    let book = Arc::new(Codebook::from_lengths(&[(256, 1)]).unwrap());
    let decoder = PreparedDecoder::with_codebook(&dictionary, book, false, 65536).unwrap();
    assert_eq!(
        decoder
            .decode_field(&dictionary, &[0], 0, &mut Vec::new())
            .unwrap(),
        (1, 1)
    );
    assert!(decoder
        .decode_field(&dictionary, &[128], 0, &mut Vec::new())
        .is_err());
}
struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "brepair-huffman-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}
fn vocabulary() -> Model {
    let mut tokens = byte_model().tokens;
    tokens.extend([
        Token {
            bytes: b"abc".to_vec(),
            ends_field: false,
        },
        Token {
            bytes: b"\n".to_vec(),
            ends_field: true,
        },
        Token {
            bytes: b"abc\n".to_vec(),
            ends_field: true,
        },
    ]);
    Model::new(tokens).unwrap()
}
fn encoded(d: &Dictionary, ids: &[u32], block: usize, bytes: u64, fields: u64) -> Vec<u8> {
    let mut out = Cursor::new(Vec::new());
    let mut encoder = Encoder::new(&mut out, d, block).unwrap();
    for &id in ids {
        encoder.push(id).unwrap();
    }
    encoder.finish(bytes, fields).unwrap();
    out.into_inner()
}
fn encoded_static(d: &Dictionary, ids: &[u32], block: usize) -> Vec<u8> {
    let mut counts = vec![0u64; d.expansions.len()];
    let (mut bytes, mut fields) = (0u64, 0u64);
    for &id in ids {
        counts[id as usize] += 1;
        bytes += d.expansions[id as usize].bytes;
        fields += u64::from(d.expansions[id as usize].ends_field);
    }
    let mut out = Cursor::new(Vec::new());
    let mut encoder = Encoder::with_counts(&mut out, d, block, &counts).unwrap();
    for &id in ids {
        encoder.push(id).unwrap();
    }
    encoder.finish(bytes, fields).unwrap();
    out.into_inner()
}

#[test]
fn static_table_is_shared_by_all_blocks_and_preserves_both_outputs_and_seeking() {
    use std::sync::Arc;
    let dictionary = Dictionary::new(vocabulary(), vec![], true).unwrap();
    let ids: Vec<_> = [vec![97; 31], vec![257; 19], vec![255, 258, 259, 256]].concat();
    let expected: Vec<_> = ids
        .iter()
        .flat_map(|&id| dictionary.model.tokens[id as usize].bytes.clone())
        .collect();
    for size in [1, 7, 65536] {
        let archive = encoded_static(&dictionary, &ids, size);
        let mut reader = Reader::new(Cursor::new(archive.clone())).unwrap();
        assert_eq!(reader.format(), "BRHUFF02");
        assert!(reader.static_table_bytes() > 0);
        let first = reader.block(0).unwrap();
        let table = first.book.clone();
        let prepared = reader.prepare_decoder(false, 1024).unwrap();
        prepared
            .decode_block(&reader.dictionary, &first, &mut std::io::sink())
            .unwrap();
        let mut other = Reader::new(Cursor::new(archive.clone())).unwrap();
        let foreign = other.block(0).unwrap();
        assert!(prepared
            .decode_block(&reader.dictionary, &foreign, &mut std::io::sink())
            .is_err());
        for i in 0..reader.index.len() {
            let block = reader.block(i).unwrap();
            assert!(Arc::ptr_eq(&table, &block.book));
        }
        for i in 0..ids.len() {
            let count = (ids.len() - i).min(9);
            assert_eq!(
                reader.token_range(i as u64, count).unwrap(),
                ids[i..i + count]
            );
        }
        let mut output = Vec::new();
        reader.decode_to(false, &mut output).unwrap();
        assert_eq!(output, expected);
        let expected_ids: Vec<_> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
        let mut output = Vec::new();
        reader.decode_to(true, &mut output).unwrap();
        assert_eq!(output, expected_ids);
        for token_ids in [false, true] {
            let reader = Reader::new(Cursor::new(archive.clone())).unwrap();
            let mut output = Vec::new();
            let options = PipelineOptions {
                buffers_per_stage: 1,
                references_per_buffer: 3,
                output_buffer_bytes: 7,
                ..PipelineOptions::default()
            };
            huffman_pipeline::decode(reader, token_ids, &mut output, options).unwrap();
            assert_eq!(
                output,
                if token_ids {
                    expected_ids.clone()
                } else {
                    expected.clone()
                }
            );
        }
    }
}

#[test]
fn static_empty_singleton_and_preset_table_allow_unused_codes() {
    let dictionary = Dictionary::new(byte_model(), vec![], true).unwrap();
    for ids in [vec![], vec![256], vec![256; 17], vec![97, 256]] {
        let bytes = encoded_static(&dictionary, &ids, 1);
        let mut reader = Reader::new(Cursor::new(bytes.clone())).unwrap();
        assert_eq!(reader.format(), "BRHUFF02");
        let mut actual = Vec::new();
        reader.decode_to(true, &mut actual).unwrap();
        assert_eq!(
            actual,
            ids.iter()
                .flat_map(|id| id.to_le_bytes())
                .collect::<Vec<_>>()
        );
        let reader = Reader::new(Cursor::new(bytes)).unwrap();
        huffman_pipeline::decode(
            reader,
            false,
            &mut std::io::sink(),
            PipelineOptions::default(),
        )
        .unwrap();
    }
    for ids in [vec![], vec![97, 256]] {
        let mut out = Cursor::new(Vec::new());
        let mut encoder = Encoder::with_counts(&mut out, &dictionary, 1, &[1; 257]).unwrap();
        for &id in &ids {
            encoder.push(id).unwrap();
        }
        encoder
            .finish(
                if ids.is_empty() { 0 } else { 1 },
                if ids.is_empty() { 0 } else { 1 },
            )
            .unwrap();
        let mut reader = Reader::new(Cursor::new(out.into_inner())).unwrap();
        assert_eq!(reader.static_codebook().unwrap().symbols.len(), 257);
        let mut actual = Vec::new();
        reader.decode_to(true, &mut actual).unwrap();
        assert_eq!(
            actual,
            ids.into_iter()
                .flat_map(u32::to_le_bytes)
                .collect::<Vec<_>>()
        );
        let mut piped = Vec::new();
        huffman_pipeline::decode(reader, true, &mut piped, PipelineOptions::default()).unwrap();
        assert_eq!(piped, actual);
    }
}

#[test]
fn static_generic_recipes_support_uncached_expansion_and_detect_late_corruption() {
    let dictionary = Dictionary::new(byte_model(), vec![(97, 98), (257, 256)], true).unwrap();
    let ids = vec![258; 111];
    let archive = encoded_static(&dictionary, &ids, 7);
    for token_ids in [false, true] {
        let mut reader = Reader::new(Cursor::new(archive.clone())).unwrap();
        let mut expected = Vec::new();
        reader.decode_to(token_ids, &mut expected).unwrap();
        for arena_bytes in [0, 8, 1024] {
            let reader = Reader::new(Cursor::new(archive.clone())).unwrap();
            let options = PipelineOptions {
                buffers_per_stage: 1,
                references_per_buffer: 1,
                output_buffer_bytes: 3,
                arena_bytes,
            };
            let mut actual = Vec::new();
            huffman_pipeline::decode(reader, token_ids, &mut actual, options).unwrap();
            assert_eq!(actual, expected);
        }
    }
    let reader = Reader::new(Cursor::new(archive.clone())).unwrap();
    let start = reader.index.last().unwrap().offset as usize;
    let mut bad = archive;
    bad[start + 56] ^= 128;
    let reader = Reader::new(Cursor::new(bad)).unwrap();
    let options = PipelineOptions {
        buffers_per_stage: 1,
        references_per_buffer: 1,
        output_buffer_bytes: 3,
        ..PipelineOptions::default()
    };
    let mut partial = Vec::new();
    assert!(huffman_pipeline::decode(reader, false, &mut partial, options).is_err());
    assert!(!partial.is_empty() && partial.len() < 222);
}

#[test]
fn static_format_rejects_every_single_byte_mutation_and_truncation() {
    let dictionary = Dictionary::new(byte_model(), vec![(97, 98), (257, 256)], false).unwrap();
    let good = encoded_static(&dictionary, &[258; 13], 5);
    let valid = |bytes: Vec<u8>| {
        Reader::new(Cursor::new(bytes))
            .and_then(|mut r| r.decode_to(false, &mut std::io::sink()))
            .is_ok()
    };
    assert!(valid(good.clone()));
    for i in 0..good.len() {
        let mut bad = good.clone();
        bad[i] ^= 1;
        assert!(!valid(bad), "accepted mutation at {i}");
        assert!(!valid(good[..i].to_vec()), "accepted truncation at {i}");
    }
    let mut bad = good;
    bad.push(0);
    assert!(!valid(bad));
}

#[test]
fn pipeline_checks_expansion_counts_and_final_eof_after_valid_checksums() {
    // Recompute the checksum after changing valid one-bit codes. These inputs
    // reach expansion validation, rather than failing the encoded-byte hash.
    for ai_tokens in [false, true] {
        let dictionary = Dictionary::new(byte_model(), vec![(97, 98)], ai_tokens).unwrap();
        for shared in [false, true] {
            for (ids, mutation, expected_error) in [
                (vec![257, 257, 256], 0x80, "exceeds declared totals"),
                (vec![257, 256], 0xc0, "final EOF"),
            ] {
                let mut bytes = if shared {
                    encoded_static(&dictionary, &ids, 10)
                } else {
                    encoded(&dictionary, &ids, 10, (ids.len() as u64 - 1) * 2, 1)
                };
                let reader = Reader::new(Cursor::new(bytes.clone())).unwrap();
                let start = reader.index[0].offset as usize;
                let table_len =
                    u32::from_le_bytes(bytes[start + 40..start + 44].try_into().unwrap()) as usize;
                bytes[start + 56 + table_len] ^= mutation;
                let hash = bytes[start..start + 48]
                    .iter()
                    .chain(&bytes[start + 56..])
                    // Include table and payload only, not the archive index.
                    .take(48 + table_len + 1)
                    .fold(0xcbf29ce484222325u64, |h, &b| {
                        (h ^ u64::from(b)).wrapping_mul(0x100000001b3)
                    });
                bytes[start + 48..start + 56].copy_from_slice(&hash.to_le_bytes());
                for token_ids in [false, true].into_iter().filter(|&ids| ai_tokens || !ids) {
                    for size in [1, 2, 257] {
                        let reader = Reader::new(Cursor::new(bytes.clone())).unwrap();
                        let options = PipelineOptions {
                            buffers_per_stage: 1,
                            references_per_buffer: size,
                            output_buffer_bytes: 3,
                            ..PipelineOptions::default()
                        };
                        let error = huffman_pipeline::decode(
                            reader,
                            token_ids,
                            &mut std::io::sink(),
                            options,
                        )
                        .err()
                        .unwrap();
                        assert!(error.contains(expected_error), "{error}");
                    }
                }
            }
        }
    }
}

#[test]
fn pipeline_lookup_batches_preserve_tails_and_queue_boundaries() {
    let dictionary = Dictionary::new(vocabulary(), vec![], true).unwrap();
    let ids: Vec<u32> = (0..1030)
        .map(|i| (i * 73 + 19) % 256)
        .chain([256])
        .collect();
    let expected_bytes: Vec<_> = ids
        .iter()
        .flat_map(|&id| dictionary.model.tokens[id as usize].bytes.iter().copied())
        .collect();
    let expected_ids: Vec<_> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
    for shared in [false, true] {
        for block in [255, 256, 257, 511, 512, 513] {
            let bytes = if shared {
                encoded_static(&dictionary, &ids, block)
            } else {
                encoded(&dictionary, &ids, block, expected_bytes.len() as u64, 1)
            };
            for size in [17, 257] {
                for token_ids in [false, true] {
                    let reader = Reader::new(Cursor::new(bytes.clone())).unwrap();
                    let mut output = Vec::new();
                    let options = PipelineOptions {
                        buffers_per_stage: 1,
                        references_per_buffer: size,
                        output_buffer_bytes: 31,
                        ..PipelineOptions::default()
                    };
                    huffman_pipeline::decode(reader, token_ids, &mut output, options).unwrap();
                    assert_eq!(
                        output,
                        if token_ids {
                            &expected_ids
                        } else {
                            &expected_bytes
                        }
                        .as_slice()
                    );
                }
            }
        }
    }
}
#[test]
fn frozen_token_ids_and_arbitrary_bytes_survive_block_boundaries() {
    let tokenizer = Tokenizer::new(vocabulary()).unwrap();
    let input: Vec<_> = [
        b"abc\n\n".as_slice(),
        &(0..=255).collect::<Vec<u8>>(),
        b"abc",
    ]
    .concat();
    let ids: Vec<_> = input
        .split_inclusive(|&b| b == b'\n')
        .flat_map(|field| tokenizer.encode(field))
        .collect();
    let fields = input.split_inclusive(|&b| b == b'\n').count() as u64;
    let d = Dictionary::new(tokenizer.model().clone(), vec![], true).unwrap();
    for block in [1, 2, 7, 1024] {
        let bytes = encoded(&d, &ids, block, input.len() as u64, fields);
        let mut reader = Reader::new(Cursor::new(bytes)).unwrap();
        let mut actual = vec![];
        reader.decode_to(false, &mut actual).unwrap();
        assert_eq!(actual, input);
        let mut raw_ids = vec![];
        reader.decode_to(true, &mut raw_ids).unwrap();
        assert_eq!(
            raw_ids,
            ids.iter().flat_map(|n| n.to_le_bytes()).collect::<Vec<_>>()
        );
        for start in [0, 1, ids.len() / 2, ids.len() - 1, ids.len()] {
            let n = (ids.len() - start).min(11);
            assert_eq!(
                reader.token_range(start as u64, n).unwrap(),
                ids[start..start + n]
            );
        }
        assert!(reader.token_range(ids.len() as u64, 1).is_err());
    }
}
#[test]
fn grammar_over_ai_ids_decodes_both_ways_and_seeks_inside_recipes() {
    let model = vocabulary();
    let base = model.tokens.len() as u32;
    let d = Dictionary::new(model, vec![(257, 97), (base, 258)], true).unwrap();
    let bytes = encoded(&d, &[base + 1, 259], 1, 9, 2);
    let mut reader = Reader::new(Cursor::new(bytes)).unwrap();
    let mut actual = vec![];
    reader.decode_to(false, &mut actual).unwrap();
    assert_eq!(actual, b"abca\nabc\n");
    assert_eq!(reader.token_range(0, 4).unwrap(), [257, 97, 258, 259]);
    assert_eq!(reader.token_range(1, 2).unwrap(), [97, 258]);
    let arena = Arena::new(&reader.dictionary, true, 0).unwrap();
    let block = reader.block(0).unwrap();
    let mut raw = vec![];
    Reader::<Cursor<Vec<u8>>>::decode_block(&reader.dictionary, &arena, &block, &mut raw).unwrap();
    assert_eq!(
        raw,
        [257u32, 97, 258]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>()
    );
}
#[test]
fn empty_stream_and_eof_only_stream_are_distinct() {
    let d = Dictionary::new(byte_model(), vec![], true).unwrap();
    for (ids, fields) in [(&[][..], 0), (&[256][..], 1), (&[256, 256][..], 2)] {
        let bytes = encoded(&d, ids, 1, 0, fields);
        let mut reader = Reader::new(Cursor::new(bytes)).unwrap();
        assert_eq!(reader.totals.fields, fields);
        let mut output = vec![];
        reader.decode_to(false, &mut output).unwrap();
        assert!(output.is_empty());
        assert_eq!(reader.token_range(0, ids.len()).unwrap(), ids);
    }
    let mut out = Cursor::new(vec![]);
    let mut encoder = Encoder::new(&mut out, &d, 10).unwrap();
    encoder.push(97).unwrap();
    assert!(encoder.finish(1, 0).is_err());
}
#[test]
fn every_single_byte_corruption_and_truncation_is_rejected() {
    let d = Dictionary::new(byte_model(), vec![(97, 98), (257, 256)], false).unwrap();
    let good = encoded(&d, &[258, 258, 258, 258], 3, 8, 4);
    fn valid(b: Vec<u8>) -> bool {
        Reader::new(Cursor::new(b))
            .and_then(|mut r| r.decode_to(false, &mut std::io::sink()))
            .is_ok()
    }
    assert!(valid(good.clone()));
    for at in 0..good.len() {
        let mut bad = good.clone();
        bad[at] ^= 1;
        assert!(!valid(bad), "accepted mutation at {at}");
        assert!(!valid(good[..at].to_vec()), "accepted truncation at {at}");
    }
    let mut bad = good;
    bad.push(0);
    assert!(!valid(bad));
}
#[test]
fn brpair_transcode_is_streaming_and_byte_exact() {
    let temp = Temp::new();
    let input = vec![b"abracadabra\n".to_vec(); 128];
    let compressed = engine::compress(&input, &Config::default(), |_| {})
        .unwrap()
        .compressed;
    let path = temp.0.join("input.brp");
    fs::write(&path, archive::encode(&compressed).unwrap()).unwrap();
    let packed = temp.0.join("output.brh");
    huffman_archive::pack_file(&path, &packed, 17).unwrap();
    let mut reader = Reader::new(fs::File::open(&packed).unwrap()).unwrap();
    assert!(!reader.dictionary.ai_tokens);
    assert!(!reader.dictionary.rules.is_empty());
    let mut output = vec![];
    reader.decode_to(false, &mut output).unwrap();
    assert_eq!(output, input.concat());
    assert!(reader.decode_to(true, &mut std::io::sink()).is_err());
}

#[test]
fn batched_encoding_keeps_u64_expansion_totals_and_rejects_overflow() {
    for power in [32, 63] {
        let mut rules = vec![(97, 97)];
        for i in 1..power {
            let previous = 256 + i;
            rules.push((previous, previous));
        }
        let huge = 256 + power;
        rules.push((huge, 256));
        let dictionary = Dictionary::new(byte_model(), rules, false).unwrap();
        let last = dictionary.expansions.len() - 1;
        let mut counts = vec![0; dictionary.expansions.len()];
        counts[last] = 2;
        let mut out = Cursor::new(vec![]);
        let mut encoder = Encoder::with_counts(&mut out, &dictionary, 2, &counts).unwrap();
        let result = encoder.extend(&[last as u32, last as u32]);
        if power == 63 {
            assert!(result.unwrap_err().contains("byte count overflow"));
        } else {
            result.unwrap();
            let bytes = 1u64 << (power + 1);
            let totals = encoder.finish(bytes, 2).unwrap();
            assert_eq!(totals.bytes, bytes);
            assert_eq!(totals.tokens, bytes + 2);
            let mut reader = Reader::new(Cursor::new(out.into_inner())).unwrap();
            assert_eq!(reader.totals, totals);
            assert_eq!(reader.block(0).unwrap().totals, totals);
        }
    }
}
#[test]
fn public_commands_export_exact_ids_and_publish_atomically() {
    let temp = Temp::new();
    let root = &temp.0;
    fs::write(root.join("model.brt"), vocabulary().to_bytes()).unwrap();
    fs::write(root.join("input"), b"abc\nabc\n\xff").unwrap();
    let run = |args: &[&str]| {
        Command::new(env!("CARGO_BIN_EXE_brepair"))
            .current_dir(root)
            .args(args)
            .output()
            .unwrap()
    };
    let packed = run(&[
        "huffman-encode",
        "model.brt",
        "input",
        "packed.brh",
        "--block-symbols",
        "2",
    ]);
    assert!(
        packed.status.success(),
        "{}",
        String::from_utf8_lossy(&packed.stderr)
    );
    for args in [
        vec!["huffman-decode", "packed.brh", "decoded"],
        vec!["huffman-ids", "packed.brh", "ids"],
        vec!["huffman-inspect", "packed.brh"],
    ] {
        let result = run(&args);
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
    }
    assert_eq!(
        fs::read(root.join("decoded")).unwrap(),
        fs::read(root.join("input")).unwrap()
    );
    assert!(!run(&["huffman-decode", "packed.brh", "decoded"])
        .status
        .success());
    assert_eq!(
        fs::read(root.join("ids")).unwrap(),
        [259u32, 259, 255, 256]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>()
    );
    let mut bad = fs::read(root.join("packed.brh")).unwrap();
    *bad.last_mut().unwrap() ^= 1;
    fs::write(root.join("bad.brh"), bad).unwrap();
    assert!(!run(&["huffman-decode", "bad.brh", "failed"])
        .status
        .success());
    assert!(!root.join("failed").exists());
    assert!(!fs::read_dir(root).unwrap().any(|e| e
        .unwrap()
        .file_name()
        .to_string_lossy()
        .ends_with(".partial")));
}

#[test]
fn public_decoder_autodetects_compact_archives_in_both_thread_modes() {
    let temp = Temp::new();
    let dictionary = Dictionary::new(vocabulary(), vec![(257, 258)], true).unwrap();
    let sequence = [260, 259];
    fs::write(
        temp.0.join("source.brh"),
        encoded_static(&dictionary, &sequence, 1),
    )
    .unwrap();
    let variants = temp.0.join("variants");
    let report = bre_pair::shared_huffman::optimize(
        &temp.0.join("source.brh"),
        &variants,
        Default::default(),
    )
    .unwrap();
    let compact = variants.join(format!("{}.brhv", report.variants[0].name));
    for threads in ["1", "4"] {
        for (command, suffix, expected) in [
            ("huffman-decode", "bytes", b"abc\nabc\n".to_vec()),
            (
                "huffman-ids",
                "ids",
                [257u32, 258, 259]
                    .into_iter()
                    .flat_map(u32::to_le_bytes)
                    .collect(),
            ),
        ] {
            let out = temp.0.join(format!("{suffix}-{threads}"));
            let result = Command::new(env!("CARGO_BIN_EXE_brepair"))
                .arg(command)
                .arg(&compact)
                .arg(&out)
                .args(["--threads", threads])
                .output()
                .unwrap();
            assert!(
                result.status.success(),
                "{}",
                String::from_utf8_lossy(&result.stderr)
            );
            assert_eq!(fs::read(&out).unwrap(), expected);
            let mut broken = fs::read(&compact).unwrap();
            *broken.last_mut().unwrap() ^= 1;
            let bad = temp.0.join(format!("bad-{suffix}-{threads}"));
            fs::write(&bad, broken).unwrap();
            let failed = temp.0.join(format!("failed-{suffix}-{threads}"));
            let result = Command::new(env!("CARGO_BIN_EXE_brepair"))
                .arg(command)
                .arg(&bad)
                .arg(&failed)
                .args(["--threads", threads])
                .output()
                .unwrap();
            assert!(!result.status.success());
            assert!(!failed.exists());
        }
    }
    let result = Command::new(env!("CARGO_BIN_EXE_brepair"))
        .arg("huffman-inspect")
        .arg(&compact)
        .output()
        .unwrap();
    assert!(result.status.success());
    assert!(String::from_utf8_lossy(&result.stdout).contains("BRHUFV01"));
}

#[test]
fn compressed_rule_lengths_repack_preserves_payload_ids_and_both_public_decoders() {
    use bre_pair::shared_huffman::{self, RuleLengths};
    for ai in [false, true] {
        let temp = Temp::new();
        let model = if ai { vocabulary() } else { byte_model() };
        let basis = model.tokens.len() as u32;
        let mut rules: Vec<_> = (0..128).map(|i| (65 + i / 16, 65 + i % 16)).collect();
        if ai {
            rules[0] = (257, 258);
        }
        let dictionary = Dictionary::new(model, rules, ai).unwrap();
        let sequence: Vec<_> = (basis..basis + 128)
            .cycle()
            .take(384)
            .chain([256])
            .collect();
        let source = temp.0.join("source.brh");
        fs::write(&source, encoded_static(&dictionary, &sequence, 31)).unwrap();
        let variants = temp.0.join("variants");
        shared_huffman::optimize(
            &source,
            &variants,
            shared_huffman::Options {
                rounds: 0,
                rule_lengths: RuleLengths::Varint,
                ..Default::default()
            },
        )
        .unwrap();
        let old = variants.join("binary-shared.brhv");
        let old_bytes = fs::read(&old).unwrap();
        assert_eq!(&old_bytes[..8], b"BRHUFV01");
        let compressed = temp.0.join("compressed.brhv");
        let run = |args: &[&std::ffi::OsStr]| {
            Command::new(env!("CARGO_BIN_EXE_brepair"))
                .args(args)
                .output()
                .unwrap()
        };
        let status = run(&[
            "huffman-repack".as_ref(),
            old.as_os_str(),
            compressed.as_os_str(),
        ]);
        assert!(
            status.status.success(),
            "{}",
            String::from_utf8_lossy(&status.stderr)
        );
        let new_bytes = fs::read(&compressed).unwrap();
        assert_eq!(&new_bytes[..8], b"BRHUFV02");
        assert!(new_bytes.len() < old_bytes.len());
        let num = |b: &[u8], at| u64::from_le_bytes(b[at..at + 8].try_into().unwrap()) as usize;
        let payload = |b: &[u8]| b[128 + num(b, 48)..num(b, 104)].to_vec();
        assert_eq!(payload(&new_bytes), payload(&old_bytes));
        let references = |b: &[u8]| {
            let end = 128 + num(b, 48);
            b[end - num(b, 56).div_ceil(8)..end].to_vec()
        };
        assert_eq!(references(&new_bytes), references(&old_bytes));
        for ids in [false, true] {
            if ids && !ai {
                continue;
            }
            let mut expected = Vec::new();
            shared_huffman::Reader::new(Cursor::new(old_bytes.clone()))
                .unwrap()
                .decode_to(&mut expected, ids)
                .unwrap();
            for threads in ["1", "4"] {
                let output = temp.0.join(format!("decoded-{ids}-{threads}"));
                let command = if ids { "huffman-ids" } else { "huffman-decode" };
                let result = run(&[
                    command.as_ref(),
                    compressed.as_os_str(),
                    output.as_os_str(),
                    "--threads".as_ref(),
                    threads.as_ref(),
                ]);
                assert!(
                    result.status.success(),
                    "{}",
                    String::from_utf8_lossy(&result.stderr)
                );
                assert_eq!(fs::read(output).unwrap(), expected);
            }
        }
        let restored = temp.0.join("restored.brhv");
        let result = run(&[
            "huffman-repack".as_ref(),
            compressed.as_os_str(),
            restored.as_os_str(),
            "--rule-lengths".as_ref(),
            "varint".as_ref(),
        ]);
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert_eq!(fs::read(&restored).unwrap(), old_bytes);
        let result = run(&["huffman-inspect".as_ref(), compressed.as_os_str()]);
        assert!(result.status.success());
        assert!(String::from_utf8_lossy(&result.stdout).contains("BRHUFV02"));
        assert!(!run(&[
            "huffman-repack".as_ref(),
            old.as_os_str(),
            compressed.as_os_str()
        ])
        .status
        .success());
        assert_eq!(fs::read(&compressed).unwrap(), new_bytes);
        let mut broken = new_bytes.clone();
        broken[128] ^= 1;
        let bad = temp.0.join("bad.brhv");
        fs::write(&bad, broken).unwrap();
        let failed = temp.0.join("failed.raw");
        assert!(!run(&[
            "huffman-decode".as_ref(),
            bad.as_os_str(),
            failed.as_os_str(),
            "--threads".as_ref(),
            "4".as_ref()
        ])
        .status
        .success());
        assert!(!failed.exists());
        // Direct new checkpoints choose the same representation automatically.
        let auto = temp.0.join("auto");
        shared_huffman::optimize(
            &source,
            &auto,
            shared_huffman::Options {
                rounds: 0,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(
            fs::read(auto.join("binary-shared.brhv")).unwrap(),
            new_bytes
        );
    }
}

#[test]
fn run_coded_descriptors_restore_old_formats_and_use_both_public_decoders() {
    use bre_pair::shared_huffman::{self, RuleLengths};
    use bre_pair::tokenizer::fingerprint;
    for ai in [false, true] {
        let temp = Temp::new();
        let model = if ai { vocabulary() } else { byte_model() };
        let basis = model.tokens.len() as u32;
        let mut rules: Vec<_> = (0..512).map(|i| (65 + i / 16, 65 + i % 16)).collect();
        if ai {
            rules[0] = (257, 258);
        }
        let dictionary = Dictionary::new(model, rules, ai).unwrap();
        let sequence: Vec<_> = (basis..basis + 512)
            .cycle()
            .take(1536)
            .chain([256])
            .collect();
        let input = temp.0.join("static.brh");
        fs::write(&input, encoded_static(&dictionary, &sequence, 31)).unwrap();
        let variants = temp.0.join("variants");
        shared_huffman::optimize(
            &input,
            &variants,
            shared_huffman::Options {
                rounds: 0,
                ..Default::default()
            },
        )
        .unwrap();
        let old = variants.join("binary-shared.brhv");
        let old_bytes = fs::read(&old).unwrap();
        assert_eq!(&old_bytes[..8], b"BRHUFV02");
        let new = temp.0.join("runs.brhv");
        let result = Command::new(env!("CARGO_BIN_EXE_brepair"))
            .args([
                "huffman-repack",
                old.to_str().unwrap(),
                new.to_str().unwrap(),
                "--rule-lengths",
                "runs",
            ])
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let bytes = fs::read(&new).unwrap();
        assert_eq!(&bytes[..8], b"BRHUFV03");
        assert!(bytes.len() < old_bytes.len());
        let num = |b: &[u8], at| u64::from_le_bytes(b[at..at + 8].try_into().unwrap()) as usize;
        let payload = |b: &[u8]| b[128 + num(b, 48)..num(b, 104)].to_vec();
        assert_eq!(payload(&bytes), payload(&old_bytes));
        let refs = |b: &[u8]| {
            let end = 128 + num(b, 48);
            b[end - num(b, 56).div_ceil(8)..end].to_vec()
        };
        assert_eq!(refs(&bytes), refs(&old_bytes));
        for ids in [false, true] {
            if ids && !ai {
                continue;
            }
            let mut expected = Vec::new();
            shared_huffman::Reader::new(Cursor::new(old_bytes.clone()))
                .unwrap()
                .decode_to(&mut expected, ids)
                .unwrap();
            for threads in ["1", "4"] {
                let out = temp.0.join(format!("out-{ids}-{threads}"));
                let command = if ids { "huffman-ids" } else { "huffman-decode" };
                let status = Command::new(env!("CARGO_BIN_EXE_brepair"))
                    .args([
                        command,
                        new.to_str().unwrap(),
                        out.to_str().unwrap(),
                        "--threads",
                        threads,
                    ])
                    .output()
                    .unwrap();
                assert!(
                    status.status.success(),
                    "{}",
                    String::from_utf8_lossy(&status.stderr)
                );
                assert_eq!(fs::read(out).unwrap(), expected);
            }
        }
        let restored = temp.0.join("restored02");
        shared_huffman::repack_rule_lengths(&new, &restored, RuleLengths::Auto).unwrap();
        assert_eq!(fs::read(restored).unwrap(), old_bytes);
        let raw1 = temp.0.join("raw01");
        let raw2 = temp.0.join("raw01from03");
        shared_huffman::repack_rule_lengths(&old, &raw1, RuleLengths::Varint).unwrap();
        shared_huffman::repack_rule_lengths(&new, &raw2, RuleLengths::Varint).unwrap();
        assert_eq!(fs::read(raw1).unwrap(), fs::read(raw2).unwrap());
        for end in 0..bytes.len() {
            assert!(shared_huffman::Reader::new(Cursor::new(bytes[..end].to_vec())).is_err());
        }
        for at in 0..bytes.len() {
            let mut bad = bytes.clone();
            bad[at] ^= 1;
            let mut output = Vec::new();
            assert!(shared_huffman::Reader::new(Cursor::new(bad))
                .and_then(|mut r| r.decode_to(&mut output, false))
                .is_err());
        }
        // Repair enclosing checksums: malformed run metadata must fail structurally.
        let mut bad = bytes.clone();
        let descriptor = 128 + num(&bad, 48) - num(&bad, 56).div_ceil(8) - num(&bad, 72);
        bad[descriptor] = 255;
        let end = 128 + num(&bad, 48);
        let crc = fingerprint(&bad[128..end]);
        bad[112..120].copy_from_slice(&crc.to_le_bytes());
        let crc = fingerprint(&bad[..120]);
        bad[120..128].copy_from_slice(&crc.to_le_bytes());
        assert!(shared_huffman::Reader::new(Cursor::new(bad)).is_err());
    }
}

#[test]
fn pipeline_matches_reference_with_tiny_recycled_buffers_and_both_outputs() {
    let tokenizer = Tokenizer::new(vocabulary()).unwrap();
    let input: Vec<_> = [
        b"abc\n\n".as_slice(),
        &(0..=255).collect::<Vec<_>>(),
        b"abc",
    ]
    .concat();
    let ids: Vec<_> = input
        .split_inclusive(|&b| b == b'\n')
        .flat_map(|f| tokenizer.encode(f))
        .collect();
    let fields = input.split_inclusive(|&b| b == b'\n').count() as u64;
    let dictionary = Dictionary::new(tokenizer.model().clone(), vec![], true).unwrap();
    let archive = encoded(&dictionary, &ids, 7, input.len() as u64, fields);
    for slots in [1, 2] {
        for size in [1, 3, 17, 65536] {
            for token_ids in [false, true] {
                let options = PipelineOptions {
                    buffers_per_stage: slots,
                    references_per_buffer: size,
                    output_buffer_bytes: size,
                    arena_bytes: 1024,
                };
                let reader = Reader::new(Cursor::new(archive.clone())).unwrap();
                let mut output = Vec::new();
                let report =
                    huffman_pipeline::decode(reader, token_ids, &mut output, options).unwrap();
                assert_eq!(report.totals.tokens, ids.len() as u64);
                let expected = if token_ids {
                    ids.iter().flat_map(|id| id.to_le_bytes()).collect()
                } else {
                    input.clone()
                };
                assert_eq!(output, expected);
            }
        }
    }
}

#[test]
fn pipeline_handles_empty_and_uncached_deep_expansions() {
    let mut rules = vec![(97, 98)];
    for id in 258..2257 {
        rules.push((id - 1, 97));
    }
    rules.push((2256, 256));
    let dictionary = Dictionary::new(byte_model(), rules, true).unwrap();
    let archive = encoded(&dictionary, &[2257, 256], 1, 2001, 2);
    for budget in [0, 32, 4096] {
        for token_ids in [false, true] {
            let mut reader = Reader::new(Cursor::new(archive.clone())).unwrap();
            let mut expected = Vec::new();
            reader.decode_to(token_ids, &mut expected).unwrap();
            let reader = Reader::new(Cursor::new(archive.clone())).unwrap();
            let options = PipelineOptions {
                buffers_per_stage: 1,
                references_per_buffer: 1,
                output_buffer_bytes: 7,
                arena_bytes: budget,
            };
            let mut output = Vec::new();
            huffman_pipeline::decode(reader, token_ids, &mut output, options).unwrap();
            assert_eq!(output, expected);
        }
    }
    for (ids, fields) in [(&[][..], 0), (&[256][..], 1), (&[256, 256][..], 2)] {
        let dictionary = Dictionary::new(byte_model(), vec![], true).unwrap();
        for token_ids in [false, true] {
            let reader = Reader::new(Cursor::new(encoded(&dictionary, ids, 1, 0, fields))).unwrap();
            let mut output = Vec::new();
            huffman_pipeline::decode(reader, token_ids, &mut output, PipelineOptions::default())
                .unwrap();
            assert_eq!(output.len(), if token_ids { ids.len() * 4 } else { 0 });
        }
    }
}

#[test]
fn pipeline_propagates_failed_reads_checksums_writes_and_flushes() {
    use std::io::{self, Read, Seek, SeekFrom, Write};
    use std::sync::{atomic::AtomicBool, Arc};
    struct Source {
        data: Cursor<Vec<u8>>,
        fail: Arc<AtomicBool>,
    }
    impl Read for Source {
        fn read(&mut self, b: &mut [u8]) -> io::Result<usize> {
            if self.fail.load(Ordering::Relaxed) {
                Err(io::Error::other("injected read failure"))
            } else {
                self.data.read(b)
            }
        }
    }
    impl Seek for Source {
        fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
            self.data.seek(pos)
        }
    }
    struct Target {
        remaining: usize,
        fail_flush: bool,
    }
    impl Write for Target {
        fn write(&mut self, b: &[u8]) -> io::Result<usize> {
            if self.remaining == 0 {
                return Err(io::Error::other("injected write failure"));
            }
            let n = b.len().min(self.remaining);
            self.remaining -= n;
            Ok(n)
        }
        fn flush(&mut self) -> io::Result<()> {
            if self.fail_flush {
                Err(io::Error::other("injected flush failure"))
            } else {
                Ok(())
            }
        }
    }
    let dictionary = Dictionary::new(byte_model(), vec![], true).unwrap();
    let ids: Vec<_> = [97; 1000].into_iter().chain([256]).collect();
    let archive = encoded(&dictionary, &ids, 11, 1000, 1);
    let options = PipelineOptions {
        buffers_per_stage: 1,
        references_per_buffer: 3,
        output_buffer_bytes: 5,
        arena_bytes: 1024,
    };
    let fail = Arc::new(AtomicBool::new(false));
    let reader = Reader::new(Source {
        data: Cursor::new(archive.clone()),
        fail: fail.clone(),
    })
    .unwrap();
    fail.store(true, Ordering::Relaxed);
    let e = huffman_pipeline::decode(reader, false, &mut io::sink(), options)
        .err()
        .unwrap();
    assert!(e.contains("injected read failure"), "{e}");
    for (remaining, fail_flush) in [(17, false), (usize::MAX, true)] {
        let reader = Reader::new(Cursor::new(archive.clone())).unwrap();
        let e = huffman_pipeline::decode(
            reader,
            false,
            &mut Target {
                remaining,
                fail_flush,
            },
            options,
        )
        .err()
        .unwrap();
        assert!(
            e.contains(if fail_flush {
                "injected flush failure"
            } else {
                "injected write failure"
            }),
            "{e}"
        );
    }
    let reader = Reader::new(Cursor::new(archive.clone())).unwrap();
    let start = reader.index[0].offset as usize;
    let mut bad = archive;
    bad[start + 56] ^= 1;
    let reader = Reader::new(Cursor::new(bad)).unwrap();
    let e = huffman_pipeline::decode(reader, false, &mut io::sink(), options)
        .err()
        .unwrap();
    assert!(e.contains("checksum mismatch"), "{e}");
    let mut late = encoded(&dictionary, &ids, 11, 1000, 1);
    let reader = Reader::new(Cursor::new(late.clone())).unwrap();
    let offset = reader.index.last().unwrap().offset as usize;
    late[offset + 56] ^= 1;
    let reader = Reader::new(Cursor::new(late)).unwrap();
    let mut partial = Vec::new();
    let error = huffman_pipeline::decode(reader, false, &mut partial, options)
        .err()
        .unwrap();
    assert!(error.contains("checksum mismatch"), "{error}");
    assert!(!partial.is_empty() && partial.len() < 1000);
    // The reference API still works after a pipeline failure elsewhere.
    let mut reader = Reader::new(Cursor::new(encoded(&dictionary, &ids, 11, 1000, 1))).unwrap();
    reader.decode_to(false, &mut io::sink()).unwrap();
}

#[test]
fn public_pipeline_commands_preserve_ids_and_atomic_failure_behavior() {
    let temp = Temp::new();
    let root = &temp.0;
    let dictionary = Dictionary::new(vocabulary(), vec![], true).unwrap();
    let archive = encoded(&dictionary, &[259, 259, 255, 256], 1, 9, 3);
    fs::write(root.join("input.brh"), &archive).unwrap();
    let run = |command: &str, output: &str| {
        Command::new(env!("CARGO_BIN_EXE_brepair"))
            .current_dir(root)
            .args([command, "input.brh", output, "--threads", "4"])
            .output()
            .unwrap()
    };
    for (command, output) in [("huffman-decode", "bytes"), ("huffman-ids", "ids")] {
        let result = run(command, output);
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert!(String::from_utf8_lossy(&result.stdout).contains("pipeline_active_seconds"));
    }
    assert_eq!(fs::read(root.join("bytes")).unwrap(), b"abc\nabc\n\xff");
    assert_eq!(
        fs::read(root.join("ids")).unwrap(),
        [259u32, 259, 255, 256]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>()
    );
    let mut reader = Reader::new(Cursor::new(archive.clone())).unwrap();
    let start = reader.index[0].offset as usize;
    let mut bad = archive;
    bad[start + 56] ^= 1;
    fs::write(root.join("input.brh"), bad).unwrap();
    assert!(!run("huffman-decode", "failed").status.success());
    assert!(!root.join("failed").exists());
    assert!(!fs::read_dir(root).unwrap().any(|e| e
        .unwrap()
        .file_name()
        .to_string_lossy()
        .ends_with(".partial")));
    // Repeated references read an immutable dictionary and never mutate it.
    reader.decode_to(false, &mut std::io::sink()).unwrap();
}

#[test]
fn pipeline_joins_workers_after_an_input_panic_and_rejects_invalid_options() {
    use std::io::{self, Read, Seek, SeekFrom};
    use std::sync::{atomic::AtomicBool, Arc};
    struct PanicSource {
        data: Cursor<Vec<u8>>,
        panic: Arc<AtomicBool>,
    }
    impl Read for PanicSource {
        fn read(&mut self, target: &mut [u8]) -> io::Result<usize> {
            assert!(!self.panic.load(Ordering::Relaxed), "injected input panic");
            self.data.read(target)
        }
    }
    impl Seek for PanicSource {
        fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
            self.data.seek(pos)
        }
    }
    let dictionary = Dictionary::new(byte_model(), vec![], true).unwrap();
    let archive = encoded(&dictionary, &[97, 97, 256], 1, 2, 1);
    let panic = Arc::new(AtomicBool::new(false));
    let source = PanicSource {
        data: Cursor::new(archive.clone()),
        panic: panic.clone(),
    };
    let reader = Reader::new(source).unwrap();
    panic.store(true, Ordering::Relaxed);
    let e = huffman_pipeline::decode(reader, false, &mut io::sink(), PipelineOptions::default())
        .err()
        .unwrap();
    assert!(e.contains("worker panicked"), "{e}");
    let reader = Reader::new(Cursor::new(archive)).unwrap();
    let options = PipelineOptions {
        buffers_per_stage: 0,
        ..PipelineOptions::default()
    };
    assert!(huffman_pipeline::decode(reader, false, &mut io::sink(), options).is_err());
}
