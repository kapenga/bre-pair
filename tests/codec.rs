use bre_pair::{
    codec::{EncodeMode, EncodeOptions, Frozen, MatchScratch},
    engine::Config,
    huffman::Codebook,
    huffman_archive::{Reader, Totals},
    huffman_dictionary::{byte_model, Dictionary},
    huffman_pipeline,
    scoring::Model as Score,
    tokenizer::{GreedyParser, Model, ParseScratch, Token, Tokenizer},
};
use std::{
    fs,
    io::Cursor,
    path::PathBuf,
    process::Command,
    sync::atomic::{AtomicU64, Ordering},
};

static NEXT: AtomicU64 = AtomicU64::new(0);
struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "brepair-codec-{}-{}",
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
    tokens.push(Token {
        bytes: b"abcd".to_vec(),
        ends_field: false,
    });
    Model::new(tokens).unwrap()
}
fn ids(model: &Model, raw: &[u8]) -> Vec<u8> {
    let tokenizer = Tokenizer::new(model.clone()).unwrap();
    raw.split_inclusive(|&b| b == b'\n')
        .flat_map(|f| tokenizer.encode(f))
        .flat_map(u32::to_le_bytes)
        .collect()
}
fn mode_ids(model: &Model, raw: &[u8], mode: EncodeMode) -> Vec<u8> {
    if mode == EncodeMode::Advanced {
        return ids(model, raw);
    }
    let parser = GreedyParser::new(model.clone()).unwrap();
    let mut scratch = ParseScratch::default();
    raw.split_inclusive(|&b| b == b'\n')
        .flat_map(|field| parser.encode(field, &mut scratch).to_vec())
        .flat_map(u32::to_le_bytes)
        .collect()
}

#[test]
fn learned_recipes_preserve_ai_ids_and_freeze_codes_for_unseen_data() {
    let temp = Temp::new();
    let training = temp.0.join("train");
    fs::write(&training, b"ab ab\n".repeat(256)).unwrap();
    for score in [Score::Frequency, Score::WeightedPmi] {
        for workers in [1, 4] {
            let config = Config {
                score,
                minimum_count: 4,
                buffer_bytes: 1 << 20,
                ..Default::default()
            };
            let (model, report) =
                Frozen::train(&vocabulary(), &training, &config, workers, None, |_| {}).unwrap();
            assert!(!report.stopped_at_limit);
            assert!(model.dictionary.rules.contains(&(b'a' as u32, b'b' as u32)));
            assert!(model.training.symbols < model.training.tokens);
            assert!(report.zero_usage_codes > 0);
            let frozen = model.to_bytes().unwrap();
            let model = Frozen::from_bytes(&frozen).unwrap();
            assert_eq!(model.to_bytes().unwrap(), frozen);
            let raw: Vec<_> = [
                b"unseen ab\n\n".as_slice(),
                &(0..=255).collect::<Vec<_>>(),
                b"abcd\xff",
            ]
            .concat();
            let expected_ids = ids(&model.dictionary.model, &raw);
            let mut archive = Cursor::new(vec![]);
            let mut input = raw.as_slice();
            model.encode(&mut input, &mut archive, 3).unwrap();
            assert!(input.is_empty());
            assert_eq!(model.to_bytes().unwrap(), frozen);
            for token_ids in [false, true] {
                let mut reader = Reader::new(Cursor::new(archive.get_ref().clone())).unwrap();
                assert_eq!(reader.dictionary.rules, model.dictionary.rules);
                assert_eq!(
                    reader.static_codebook().unwrap().to_bytes(),
                    model.book.to_bytes()
                );
                let mut output = vec![];
                reader.decode_to(token_ids, &mut output).unwrap();
                assert_eq!(
                    output,
                    if token_ids { &expected_ids } else { &raw }.as_slice()
                );
                let reader = Reader::new(Cursor::new(archive.get_ref().clone())).unwrap();
                let mut output = vec![];
                let options = huffman_pipeline::Options {
                    buffers_per_stage: 1,
                    references_per_buffer: 3,
                    output_buffer_bytes: 7,
                    arena_bytes: 32,
                };
                huffman_pipeline::decode(reader, token_ids, &mut output, options).unwrap();
                assert_eq!(
                    output,
                    if token_ids { &expected_ids } else { &raw }.as_slice()
                );
            }
        }
    }
}

#[test]
fn fresh_codec_rules_preserve_incremental_counts_and_frozen_ai_ids() {
    use bre_pair::batch::Counting;

    let temp = Temp::new();
    let training = temp.0.join("train");
    // `ab` and `cd` already have helper IDs inside the unused `abcd` token.
    // Both are frequent merge candidates, alongside self-runs and field EOFs.
    let raw = b"abXabYabZ cdQcdRcdS aaaaa aabcdaafgcdaacgadaaee\n\n\0\xffab\n".repeat(64);
    fs::write(&training, &raw).unwrap();
    let vocabulary = vocabulary();
    let expected_ids = ids(&vocabulary, &raw);
    for score in [Score::Frequency, Score::WeightedPmi] {
        let mut reference = None;
        for counting in [Counting::Full, Counting::Incremental] {
            for workers in [1, 4] {
                let mut rounds = Vec::new();
                let (model, report) = Frozen::train(
                    &vocabulary,
                    &training,
                    &Config {
                        score,
                        pair_counting: counting,
                        minimum_count: 4,
                        buffer_bytes: 1 << 20,
                        ..Default::default()
                    },
                    workers,
                    None,
                    |round| {
                        assert_eq!(round.selected, round.new_rules);
                        rounds.push((round.selected, round.rewritten));
                    },
                )
                .unwrap();
                assert!(!report.stopped_at_limit && !report.stopped_at_round_limit);
                assert!(rounds.len() > 1);
                assert_eq!(model.dictionary.model.tokens, vocabulary.tokens);
                assert_eq!(model.training.tokens * 4, expected_ids.len() as u64);
                let snapshot = (rounds, model.to_bytes().unwrap());
                if let Some(expected) = &reference {
                    assert_eq!(&snapshot, expected);
                } else {
                    reference = Some(snapshot);
                }
                let model = Frozen::from_bytes(&model.to_bytes().unwrap()).unwrap();
                let mut archive = Cursor::new(Vec::new());
                model.encode(&mut raw.as_slice(), &mut archive, 17).unwrap();
                for ai_ids in [false, true] {
                    let mut decoded = Vec::new();
                    Reader::new(Cursor::new(archive.get_ref().clone()))
                        .unwrap()
                        .decode_to(ai_ids, &mut decoded)
                        .unwrap();
                    assert_eq!(
                        decoded,
                        if ai_ids { &expected_ids } else { &raw }.as_slice()
                    );
                }
            }
        }
    }
}

#[test]
fn matcher_minimizes_frozen_bits_and_never_substitutes_another_ai_segmentation() {
    let mut tokens = byte_model().tokens;
    tokens.extend([
        Token {
            bytes: b"ab".to_vec(),
            ends_field: false,
        },
        Token {
            bytes: b"bc".to_vec(),
            ends_field: false,
        },
    ]);
    let model = Model::new(tokens).unwrap();
    let basis = model.tokens.len() as u32;
    let dictionary =
        Dictionary::new(model, vec![(b'a' as u32, 258), (257, b'c' as u32)], true).unwrap();
    let mut counts = vec![1; dictionary.expansions.len()];
    counts[basis as usize] = 1_000_000;
    let codec = Frozen {
        dictionary,
        book: Codebook::from_counts(&counts).unwrap(),
        training: Totals::default(),
    };
    let matcher = codec.prepare().unwrap();
    let terminals = Tokenizer::new(codec.dictionary.model.clone())
        .unwrap()
        .encode(b"abc");
    assert_eq!(terminals, vec![257, b'c' as u32, 256]);
    assert!(!matcher.segment(&terminals).unwrap().contains(&basis));
    let dictionary = Dictionary::new(byte_model(), vec![(b'a' as u32, b'b' as u32)], true).unwrap();
    let mut counts = vec![1; 258];
    counts[b'a' as usize] = 1_000_000;
    counts[b'b' as usize] = 1_000_000;
    let codec = Frozen {
        dictionary,
        book: Codebook::from_counts(&counts).unwrap(),
        training: Totals::default(),
    };
    assert_eq!(
        codec
            .prepare()
            .unwrap()
            .segment(&[b'a' as u32, b'b' as u32, 256])
            .unwrap(),
        vec![b'a' as u32, b'b' as u32, 256]
    );
    assert_eq!(
        codec
            .prepare()
            .unwrap()
            .segment_greedy_with(
                &[b'a' as u32, b'b' as u32, 256],
                &mut MatchScratch::default()
            )
            .unwrap(),
        &[257, 256]
    );
}

#[test]
fn weighted_grammar_matching_agrees_with_an_independent_forward_oracle() {
    let dictionary = Dictionary::new(
        byte_model(),
        vec![
            (97, 98),
            (98, 99),
            (257, 99),
            (97, 258),
            (97, 97),
            (261, 256),
        ],
        true,
    )
    .unwrap();
    let mut counts: Vec<_> = (0..dictionary.expansions.len())
        .map(|i| (i as u64 * 17 % 97) + 1)
        .collect();
    counts[259] = 1;
    counts[260] = 500;
    let book = Codebook::from_counts(&counts).unwrap();
    let mut bits = vec![0; counts.len()];
    for (id, n) in book.lengths() {
        bits[id as usize] = n as u64;
    }
    let basis = dictionary.model.tokens.len();
    let mut expansions: Vec<Vec<u32>> = (0..basis as u32).map(|id| vec![id]).collect();
    for &(a, b) in &dictionary.rules {
        expansions.push(
            [
                expansions[a as usize].as_slice(),
                expansions[b as usize].as_slice(),
            ]
            .concat(),
        );
    }
    let codec = Frozen {
        dictionary,
        book,
        training: Totals::default(),
    };
    let matcher = codec.prepare().unwrap();
    for n in 0..=5 {
        for mut pattern in 0..3usize.pow(n) {
            let mut input = Vec::new();
            for _ in 0..n {
                input.push(97 + (pattern % 3) as u32);
                pattern /= 3;
            }
            input.push(256);
            let mut cost = vec![u64::MAX; input.len() + 1];
            cost[0] = 0;
            for at in 0..input.len() {
                for (id, expansion) in expansions.iter().enumerate() {
                    let end = at + expansion.len();
                    if end <= input.len() && input[at..end] == *expansion {
                        cost[end] = cost[end].min(cost[at] + bits[id]);
                    }
                }
            }
            let output = matcher.segment(&input).unwrap();
            assert_eq!(
                output.iter().map(|&id| bits[id as usize]).sum::<u64>(),
                *cost.last().unwrap()
            );
            assert_eq!(
                output
                    .iter()
                    .flat_map(|&id| expansions[id as usize].iter().copied())
                    .collect::<Vec<_>>(),
                input
            );
            let mut greedy = Vec::new();
            let mut at = 0;
            while at < input.len() {
                let (id, expansion) = expansions
                    .iter()
                    .enumerate()
                    .filter(|(_, e)| input[at..].starts_with(e))
                    .max_by_key(|(id, e)| {
                        (
                            e.len(),
                            std::cmp::Reverse(bits[*id]),
                            std::cmp::Reverse(*id),
                        )
                    })
                    .unwrap();
                greedy.push(id as u32);
                at += expansion.len();
            }
            let mut scratch = MatchScratch::default();
            assert_eq!(
                matcher.segment_greedy_with(&input, &mut scratch).unwrap(),
                greedy
            );
            assert!(cost.last().unwrap() <= &greedy.iter().map(|&id| bits[id as usize]).sum());
        }
    }
    assert!(matcher
        .segment_greedy_with(&[u32::MAX], &mut MatchScratch::default())
        .is_err());
}

#[test]
fn empty_training_unused_fallbacks_model_damage_and_rule_limits_are_explicit() {
    let temp = Temp::new();
    let training = temp.0.join("train");
    fs::write(&training, b"").unwrap();
    let (model, _) = Frozen::train(
        &vocabulary(),
        &training,
        &Config::default(),
        1,
        None,
        |_| {},
    )
    .unwrap();
    assert_eq!(model.training, Totals::default());
    assert!(model.dictionary.rules.is_empty());
    let good = model.to_bytes().unwrap();
    for i in 0..good.len() {
        let mut bad = good.clone();
        bad[i] ^= 1;
        assert!(
            Frozen::from_bytes(&bad).is_err(),
            "accepted corruption at {i}"
        );
        assert!(
            Frozen::from_bytes(&good[..i]).is_err(),
            "accepted truncation at {i}"
        );
    }
    let mut extra = good;
    extra.push(0);
    assert!(Frozen::from_bytes(&extra).is_err());
    let mut archive = Cursor::new(vec![]);
    model.encode(b"\xff\n".as_slice(), &mut archive, 1).unwrap();
    let mut reader = Reader::new(Cursor::new(archive.into_inner())).unwrap();
    let mut bytes = vec![];
    reader.decode_to(false, &mut bytes).unwrap();
    assert_eq!(bytes, b"\xff\n");
    fs::write(&training, b"ab ab\n".repeat(64)).unwrap();
    let (limited, _) = Frozen::train(
        &vocabulary(),
        &training,
        &Config::default(),
        1,
        Some(1),
        |_| {},
    )
    .unwrap();
    assert!(limited.dictionary.rules.len() <= 1);
}

#[test]
fn public_codec_commands_use_fixed_tables_and_publish_only_complete_outputs() {
    let temp = Temp::new();
    let root = &temp.0;
    fs::write(root.join("model.brt"), vocabulary().to_bytes()).unwrap();
    fs::write(root.join("train"), b"ab ab\n".repeat(64)).unwrap();
    let raw = b"unseen ab\n\xffabcd";
    fs::write(root.join("test"), raw).unwrap();
    let run = |args: &[&str]| {
        Command::new(env!("CARGO_BIN_EXE_brepair"))
            .current_dir(root)
            .args(args)
            .output()
            .unwrap()
    };
    for args in [
        vec![
            "codec-train",
            "model.brt",
            "train",
            "frozen.brc",
            "--threads",
            "4",
            "--min-count",
            "4",
            "--report",
            "train.json",
        ],
        vec!["codec-inspect", "frozen.brc"],
        vec![
            "codec-encode",
            "frozen.brc",
            "test",
            "out.brh",
            "--block-symbols",
            "2",
            "--threads",
            "4",
        ],
        vec!["huffman-decode", "out.brh", "decoded", "--threads", "4"],
        vec!["huffman-ids", "out.brh", "ids", "--threads", "4"],
    ] {
        let result = run(&args);
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
    }
    assert_eq!(fs::read(root.join("decoded")).unwrap(), raw);
    assert_eq!(fs::read(root.join("ids")).unwrap(), ids(&vocabulary(), raw));
    let greedy = run(&[
        "codec-encode",
        "frozen.brc",
        "test",
        "greedy.brh",
        "--mode",
        "greedy",
        "--threads",
        "4",
    ]);
    assert!(
        greedy.status.success(),
        "{}",
        String::from_utf8_lossy(&greedy.stderr)
    );
    assert!(String::from_utf8_lossy(&greedy.stdout).contains("\"mode\":\"greedy\""));
    assert!(run(&["huffman-decode", "greedy.brh", "greedy-bytes"])
        .status
        .success());
    assert!(run(&["huffman-ids", "greedy.brh", "greedy-ids"])
        .status
        .success());
    assert_eq!(fs::read(root.join("greedy-bytes")).unwrap(), raw);
    assert_eq!(
        fs::read(root.join("greedy-ids")).unwrap(),
        mode_ids(&vocabulary(), raw, EncodeMode::Greedy)
    );
    assert!(!run(&[
        "codec-encode",
        "frozen.brc",
        "test",
        "bad-mode",
        "--mode",
        "unknown"
    ])
    .status
    .success());
    assert!(!root.join("bad-mode").exists());
    let verify = || {
        Command::new(env!("CARGO_BIN_EXE_verify-token-ids"))
            .current_dir(root)
            .args(["model.brt", "test", "ids"])
            .output()
            .unwrap()
    };
    assert!(verify().status.success());
    let mut wrong_ids = fs::read(root.join("ids")).unwrap();
    wrong_ids[0] ^= 1;
    fs::write(root.join("ids"), wrong_ids).unwrap();
    assert!(!verify().status.success());
    assert!(!run(&["codec-encode", "frozen.brc", "test", "out.brh"])
        .status
        .success());
    let mut bad = fs::read(root.join("frozen.brc")).unwrap();
    *bad.last_mut().unwrap() ^= 1;
    fs::write(root.join("bad.brc"), bad).unwrap();
    assert!(!run(&["codec-encode", "bad.brc", "test", "failed"])
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
fn greedy_archives_restore_their_own_ai_ids_and_keep_the_frozen_table() {
    let mut tokens = byte_model().tokens;
    tokens.extend([
        Token {
            bytes: b"abcd".to_vec(),
            ends_field: false,
        },
        Token {
            bytes: b"ab".to_vec(),
            ends_field: false,
        },
        Token {
            bytes: b"cdef".to_vec(),
            ends_field: true,
        },
    ]);
    let vocabulary = Model::new(tokens).unwrap();
    let dictionary = Dictionary::new(vocabulary, vec![(257, 101), (258, 259)], true).unwrap();
    let book = Codebook::from_counts(&vec![1; dictionary.expansions.len()]).unwrap();
    let codec = Frozen {
        dictionary,
        book,
        training: Totals::default(),
    };
    let before = codec.to_bytes().unwrap();
    let raw = b"abcdef";
    let advanced_ids = mode_ids(&codec.dictionary.model, raw, EncodeMode::Advanced);
    let greedy_ids = mode_ids(&codec.dictionary.model, raw, EncodeMode::Greedy);
    assert_ne!(advanced_ids, greedy_ids);
    for mode in [EncodeMode::Advanced, EncodeMode::Greedy] {
        let expected = if mode == EncodeMode::Advanced {
            &advanced_ids
        } else {
            &greedy_ids
        };
        let mut reference = None;
        for workers in [1, 4] {
            let mut archive = Cursor::new(vec![]);
            let options = EncodeOptions {
                mode,
                workers,
                batch_bytes: 1,
            };
            let totals = codec
                .encode_with_options(raw.as_slice(), &mut archive, 2, options)
                .unwrap();
            assert_eq!(totals.tokens, expected.len() as u64 / 4);
            if let Some(reference) = &reference {
                assert_eq!(archive.get_ref(), reference);
            } else {
                reference = Some(archive.get_ref().clone());
            }
            for token_ids in [false, true] {
                let mut reader = Reader::new(Cursor::new(archive.get_ref().clone())).unwrap();
                assert_eq!(
                    reader.static_codebook().unwrap().to_bytes(),
                    codec.book.to_bytes()
                );
                let mut decoded = vec![];
                reader.decode_to(token_ids, &mut decoded).unwrap();
                assert_eq!(
                    decoded,
                    if token_ids {
                        expected.as_slice()
                    } else {
                        raw.as_slice()
                    }
                );
                let reader = Reader::new(Cursor::new(archive.get_ref().clone())).unwrap();
                let mut decoded = vec![];
                huffman_pipeline::decode(reader, token_ids, &mut decoded, Default::default())
                    .unwrap();
                assert_eq!(
                    decoded,
                    if token_ids {
                        expected.as_slice()
                    } else {
                        raw.as_slice()
                    }
                );
            }
        }
    }
    let mut original = Cursor::new(vec![]);
    codec.encode(raw.as_slice(), &mut original, 2).unwrap();
    let mut explicit = Cursor::new(vec![]);
    codec
        .encode_with_options(raw.as_slice(), &mut explicit, 2, EncodeOptions::default())
        .unwrap();
    assert_eq!(original.get_ref(), explicit.get_ref());
    assert_eq!(codec.to_bytes().unwrap(), before);
}

#[test]
fn parallel_encoding_is_bit_identical_with_small_batches_and_empty_or_long_fields() {
    let dictionary = Dictionary::new(vocabulary(), vec![(97, 98), (258, 256)], true).unwrap();
    let book = Codebook::from_counts(&vec![1; dictionary.expansions.len()]).unwrap();
    let model = Frozen {
        dictionary,
        book,
        training: Totals::default(),
    };
    let raw = [
        b"abcd ab\n\n".repeat(30),
        vec![b'a'; 4097],
        b"\n\xff\xc0\x00tail".to_vec(),
    ]
    .concat();
    for (raw, mode) in [vec![], b"\n".to_vec(), raw]
        .into_iter()
        .flat_map(|raw| [EncodeMode::Advanced, EncodeMode::Greedy].map(|mode| (raw.clone(), mode)))
    {
        for block in [1, 17, 1024] {
            let mut serial = Cursor::new(vec![]);
            let options = EncodeOptions {
                mode,
                ..Default::default()
            };
            model
                .encode_with_options(raw.as_slice(), &mut serial, block, options)
                .unwrap();
            for workers in [2, 4, 16] {
                for batch_bytes in [1, 31, 65536] {
                    let mut parallel = Cursor::new(vec![]);
                    let options = EncodeOptions {
                        workers,
                        batch_bytes,
                        mode,
                    };
                    model
                        .encode_with_options(raw.as_slice(), &mut parallel, block, options)
                        .unwrap();
                    assert_eq!(parallel.get_ref(), serial.get_ref());
                }
            }
            let mut reader = Reader::new(Cursor::new(serial.get_ref().clone())).unwrap();
            let mut output = vec![];
            reader.decode_to(true, &mut output).unwrap();
            assert_eq!(output, mode_ids(&model.dictionary.model, &raw, mode));
            let mut reader = Reader::new(serial).unwrap();
            let mut bytes = vec![];
            reader.decode_to(false, &mut bytes).unwrap();
            assert_eq!(bytes, raw);
        }
    }
}

#[test]
fn parallel_encoder_cancels_and_joins_after_read_write_or_coordinator_failures() {
    use std::io::{self, BufReader, Read, Seek, SeekFrom, Write};
    struct BadRead {
        input: Cursor<Vec<u8>>,
        remaining: usize,
    }
    impl Read for BadRead {
        fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
            if self.remaining == 0 {
                return Err(io::Error::other("injected input failure"));
            }
            let n = out.len().min(self.remaining);
            let found = self.input.read(&mut out[..n])?;
            self.remaining -= found;
            Ok(found)
        }
    }
    struct BadWrite {
        out: Cursor<Vec<u8>>,
        remaining: usize,
        panic: bool,
    }
    impl Write for BadWrite {
        fn write(&mut self, data: &[u8]) -> io::Result<usize> {
            if self.remaining == 0 {
                assert!(!self.panic, "injected coordinator panic");
                return Err(io::Error::other("injected output failure"));
            }
            let n = data.len().min(self.remaining);
            self.remaining -= n;
            self.out.write(&data[..n])
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    impl Seek for BadWrite {
        fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
            self.out.seek(position)
        }
    }
    let dictionary = Dictionary::new(vocabulary(), vec![], true).unwrap();
    let book = Codebook::from_counts(&vec![1; dictionary.expansions.len()]).unwrap();
    let model = Frozen {
        dictionary,
        book,
        training: Totals::default(),
    };
    let raw = b"abcd ab\n".repeat(2048);
    for workers in [2, 4] {
        let options = EncodeOptions {
            workers,
            batch_bytes: 17,
            ..Default::default()
        };
        let input = BufReader::with_capacity(
            16,
            BadRead {
                input: Cursor::new(raw.clone()),
                remaining: 64,
            },
        );
        let error = model
            .encode_with_options(input, Cursor::new(vec![]), 1, options)
            .unwrap_err();
        assert!(error.contains("injected input failure"), "{error}");
        for panic in [false, true] {
            let output = BadWrite {
                out: Cursor::new(vec![]),
                remaining: 8192,
                panic,
            };
            let error = model
                .encode_with_options(raw.as_slice(), output, 1, options)
                .unwrap_err();
            assert!(
                error.contains(if panic {
                    "coordinator panicked"
                } else {
                    "injected output failure"
                }),
                "{error}"
            );
        }
    }
    for options in [
        EncodeOptions {
            workers: 0,
            batch_bytes: 1,
            ..Default::default()
        },
        EncodeOptions {
            workers: 257,
            batch_bytes: 1,
            ..Default::default()
        },
        EncodeOptions {
            workers: 4,
            batch_bytes: 0,
            ..Default::default()
        },
    ] {
        assert!(model
            .encode_with_options(raw.as_slice(), Cursor::new(vec![]), 1, options)
            .is_err());
    }
}
