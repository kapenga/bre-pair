use bre_pair::{
    engine::{self, Config},
    grammar::{BASE_SYMBOLS, EOF},
    tokenizer::{
        fingerprint, Bpe, ByteParser, GreedyParser, Model, ParseScratch, Token, Tokenizer,
    },
    vocabulary::{self, SelectionMethod, SelectionOptions},
};
use std::{
    fs,
    time::{SystemTime, UNIX_EPOCH},
};

fn basis() -> Vec<Token> {
    (0..=255)
        .map(|b| Token {
            bytes: vec![b],
            ends_field: false,
        })
        .chain([Token {
            bytes: vec![],
            ends_field: true,
        }])
        .collect()
}
fn model(extra: &[(&[u8], bool)]) -> Model {
    let mut tokens = basis();
    tokens.extend(extra.iter().map(|(bytes, ends)| Token {
        bytes: bytes.to_vec(),
        ends_field: *ends,
    }));
    Model::new(tokens).unwrap()
}
fn oracle(model: &Model, bytes: &[u8]) -> u64 {
    let labels: Vec<_> = bytes
        .iter()
        .map(|&b| b as u16)
        .chain([EOF as u16])
        .collect();
    let mut costs = vec![u64::MAX; labels.len() + 1];
    costs[0] = 0;
    for at in 0..labels.len() {
        for token in &model.tokens {
            let expansion: Vec<_> = token
                .bytes
                .iter()
                .map(|&b| b as u16)
                .chain(token.ends_field.then_some(EOF as u16))
                .collect();
            let end = at + expansion.len();
            if end <= labels.len() && labels[at..end] == expansion && costs[at] != u64::MAX {
                costs[end] = costs[end].min(costs[at] + 1);
            }
        }
    }
    costs[labels.len()]
}

#[test]
fn exact_tokenizer_matches_independent_shortest_path_and_preserves_all_bytes() {
    let tokenizer = Tokenizer::new(model(&[
        (b"aa", false),
        (b"ab", false),
        (b"bc", false),
        (b"abc", false),
        (b"aaab", false),
        (b"abca", false),
        (b"a", true),
        (b"bc", true),
    ]))
    .unwrap();
    let prepared = ByteParser::new(tokenizer.model().clone()).unwrap();
    let mut scratch = ParseScratch::default();
    for n in 0..=8 {
        for mut pattern in 0..3usize.pow(n) {
            let mut bytes = Vec::new();
            for _ in 0..n {
                bytes.push(b'a' + (pattern % 3) as u8);
                pattern /= 3;
            }
            let ids = tokenizer.encode(&bytes);
            assert_eq!(prepared.encode(&bytes, &mut scratch), ids);
            assert_eq!(ids.len() as u64, oracle(tokenizer.model(), &bytes));
            assert_eq!(tokenizer.model().decode_field(&ids).unwrap(), bytes);
        }
    }
    for bytes in [
        (0..=255).collect::<Vec<u8>>(),
        vec![0xff, 0xc0, 0, 10, 13, 0x80],
        vec![],
    ] {
        let ids = tokenizer.encode(&bytes);
        assert_eq!(prepared.encode(&bytes, &mut scratch), ids);
        assert_eq!(tokenizer.model().decode_field(&ids).unwrap(), bytes);
    }
}

#[test]
fn exact_parse_preserves_ids_and_ties_across_cost_ring_wraps() {
    for length in [3, 7, 15, 31, 63, 127] {
        let mut tokens = basis();
        for n in [2, length - 1, length, length + 1] {
            let token = Token {
                bytes: vec![b'a'; n],
                ends_field: false,
            };
            if !tokens.contains(&token) {
                tokens.push(token);
            }
        }
        tokens.push(Token {
            bytes: vec![b'a'; length],
            ends_field: true,
        });
        let tokenizer = Tokenizer::new(Model::new(tokens).unwrap()).unwrap();
        let prepared = ByteParser::new(tokenizer.model().clone()).unwrap();
        let mut scratch = ParseScratch::default();
        let expansions: Vec<Vec<_>> = tokenizer
            .model()
            .tokens
            .iter()
            .map(|t| {
                t.bytes
                    .iter()
                    .map(|&b| b as u16)
                    .chain(t.ends_field.then_some(EOF as u16))
                    .collect()
            })
            .collect();
        for size in [0, 1, length - 1, length, length + 1, 3 * length + 2] {
            let bytes = vec![b'a'; size];
            let labels: Vec<_> = bytes
                .iter()
                .map(|&b| b as u16)
                .chain([EOF as u16])
                .collect();
            // Independent forward paths retain the complete best sequence at
            // each position; there is no rolling suffix-cost storage.
            let mut paths: Vec<Option<Vec<u32>>> = vec![None; labels.len() + 1];
            paths[0] = Some(vec![]);
            for at in 0..labels.len() {
                let prefix = paths[at].as_ref().unwrap().clone();
                for (id, expansion) in expansions.iter().enumerate() {
                    let end = at + expansion.len();
                    if end > labels.len() || labels[at..end] != *expansion {
                        continue;
                    }
                    let mut candidate = prefix.clone();
                    candidate.push(id as u32);
                    let key = |id: &u32| (std::cmp::Reverse(expansions[*id as usize].len()), *id);
                    if paths[end].as_ref().is_none_or(|old| {
                        candidate.len() < old.len()
                            || (candidate.len() == old.len()
                                && candidate.iter().map(key).cmp(old.iter().map(key)).is_lt())
                    }) {
                        paths[end] = Some(candidate);
                    }
                }
            }
            assert_eq!(
                prepared.encode(&bytes, &mut scratch),
                tokenizer.encode(&bytes)
            );
            assert_eq!(
                tokenizer.encode(&bytes),
                *paths.last().unwrap().as_ref().unwrap()
            );
        }
    }
}

#[test]
fn flat_vocabulary_needs_no_children_and_exact_parse_beats_longest_match() {
    let tokenizer =
        Tokenizer::new(model(&[(b"abcd", false), (b"ab", false), (b"cdef", true)])).unwrap();
    assert_eq!(tokenizer.encode(b"abcdef"), vec![258, 259]);
    let greedy = GreedyParser::new(tokenizer.model().clone()).unwrap();
    assert_eq!(
        greedy.encode(b"abcdef", &mut ParseScratch::default()),
        &[257, b'e' as u32, b'f' as u32, EOF]
    );
    let independent =
        Tokenizer::new(model(&[(b"long token without construction nodes", true)])).unwrap();
    assert_eq!(
        independent.encode(b"long token without construction nodes"),
        vec![257]
    );
    let tokenized = Tokenizer::new(model(&[(b"ab", false), (b"bc", false)])).unwrap();
    assert_eq!(tokenized.encode(b"abc"), vec![257, b'c' as u32, EOF]);
}

#[test]
fn greedy_matches_independent_longest_prefix_with_eof_binary_and_incomplete_paths() {
    let parser = GreedyParser::new(model(&[
        (b"ab", false),
        (b"abc", false),
        (b"abcxyz", false),
        (b"aaaaa", false),
        (b"aa", true),
        (b"abc", true),
        (b"\x00\xff", true),
    ]))
    .unwrap();
    let expansions: Vec<Vec<_>> = parser
        .model()
        .tokens
        .iter()
        .map(|t| {
            t.bytes
                .iter()
                .map(|&b| u16::from(b))
                .chain(t.ends_field.then_some(EOF as u16))
                .collect()
        })
        .collect();
    let mut inputs = vec![
        vec![],
        (0..=255).collect(),
        b"abcx!\x00\xff".to_vec(),
        vec![b'a'; 4097],
    ];
    for n in 0..=6 {
        for mut pattern in 0..3usize.pow(n) {
            let mut bytes = Vec::new();
            for _ in 0..n {
                bytes.push(b'a' + (pattern % 3) as u8);
                pattern /= 3;
            }
            inputs.push(bytes);
        }
    }
    let mut scratch = ParseScratch::default();
    for bytes in inputs {
        let labels: Vec<_> = bytes
            .iter()
            .map(|&b| u16::from(b))
            .chain([EOF as u16])
            .collect();
        let (mut at, mut expected) = (0, vec![]);
        while at < labels.len() {
            let (id, expansion) = expansions
                .iter()
                .enumerate()
                .filter(|(_, e)| labels[at..].starts_with(e))
                .max_by_key(|(id, e)| (e.len(), std::cmp::Reverse(*id)))
                .unwrap();
            expected.push(id as u32);
            at += expansion.len();
        }
        let actual = parser.encode(&bytes, &mut scratch);
        assert_eq!(actual, expected);
        assert_eq!(parser.model().decode_field(actual).unwrap(), bytes);
    }
}

#[test]
fn models_reject_invalid_basis_duplicates_truncation_and_incomplete_fields() {
    let source = model(&[(b"some", false), (b"some", true)]);
    let encoded = source.to_bytes();
    assert_eq!(Model::from_bytes(&encoded).unwrap().tokens, source.tokens);
    for end in 0..encoded.len() {
        assert!(Model::from_bytes(&encoded[..end]).is_err());
    }
    let mut invalid = encoded.clone();
    invalid.push(0);
    assert!(Model::from_bytes(&invalid).is_err());
    let mut tokens = basis();
    tokens.push(tokens[0].clone());
    assert!(Model::new(tokens).is_err());
    let mut tokens = basis();
    tokens[0].bytes[0] = 1;
    assert!(Model::new(tokens).is_err());
    let mut invalid = source.clone();
    invalid.tokens[0].bytes[0] = 1;
    assert!(Tokenizer::new(invalid).is_err());
    assert!(source.decode_field(&[b'a' as u32]).is_err());
    assert!(source.decode_field(&[EOF, EOF]).is_err());
    assert!(source.decode_field(&[u32::MAX]).is_err());
}

fn bpe(model: Model, merges: &[(u32, u32, u32)]) -> Bpe {
    let mut data = b"BRTBPE01".to_vec();
    data.extend(fingerprint(&model.to_bytes()).to_le_bytes());
    data.extend((merges.len() as u32).to_le_bytes());
    for &(a, b, out) in merges {
        for id in [a, b, out] {
            data.extend(id.to_le_bytes());
        }
    }
    Bpe::from_bytes(model, &data).unwrap()
}

#[test]
fn bpe_rank_parser_obeys_priority_leftmost_runs_and_exact_control() {
    let m = model(&[
        (b"ab", false),
        (b"bc", false),
        (b"abc", false),
        (b"aa", false),
        (b"aaaa", false),
    ]);
    let bpe = bpe(
        m.clone(),
        &[
            (98, 99, 258),
            (97, 98, 257),
            (97, 258, 259),
            (97, 97, 260),
            (260, 260, 261),
        ],
    );
    assert_eq!(bpe.encode(b"abc"), vec![259, EOF]);
    assert_eq!(bpe.encode(b"aaaaa"), vec![261, 97, EOF]);
    let exact = Tokenizer::new(m).unwrap();
    for n in 0..=7 {
        for mut pattern in 0..3usize.pow(n) {
            let mut bytes = Vec::new();
            for _ in 0..n {
                bytes.push(97 + (pattern % 3) as u8);
                pattern /= 3;
            }
            let ids = bpe.encode(&bytes);
            assert_eq!(bpe.model().decode_field(&ids).unwrap(), bytes);
            assert!(exact.encode(&bytes).len() <= ids.len());
        }
    }
}

#[test]
fn fixed_budget_selection_and_parallel_counts_match_with_heldout_boundaries() {
    let fields: Vec<_> = (0..32)
        .map(|i| format!("aaaaa abababc the quick queen {i}\n").into_bytes())
        .collect();
    let source = engine::compress(
        &fields,
        &Config {
            minimum_count: 2,
            ..Config::default()
        },
        |_| {},
    )
    .unwrap();
    let (pool, support) = Model::candidates(&source.compressed, 128).unwrap();
    let size = pool.tokens.len().min(275);
    assert!(size > BASE_SYMBOLS);
    let raw: Vec<_> = fields.iter().flatten().copied().collect();
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path =
        std::env::temp_dir().join(format!("bre-tokenizer-{}-{nonce}.raw", std::process::id()));
    fs::write(&path, &raw).unwrap();
    {
        let selected = vocabulary::select(
            pool,
            &support,
            &path,
            &SelectionOptions {
                vocabulary: size,
                workers: 2,
                method: SelectionMethod::Sampled,
                sample_bytes: u64::MAX,
                full_refine_at: 32768,
                support_weight: 0.02,
            },
            |_| {},
        )
        .unwrap();
        assert_eq!(selected.tokens.len(), size);
        let tokenizer = Tokenizer::new(selected).unwrap();
        assert!(
            vocabulary::scan(&path, &tokenizer, None, Some(&[]), 1, 1, 0, false, false).is_err()
        );
        let cutoff = fields[0].len() as u64 + 1;
        let a = vocabulary::scan(&path, &tokenizer, None, None, 1, 1, cutoff, true, true).unwrap();
        let b = vocabulary::scan(&path, &tokenizer, None, None, 4, 1, cutoff, true, true).unwrap();
        assert_eq!(a.all.tokens, b.all.tokens);
        assert_eq!(a.uses, b.uses);
        assert_eq!(a.all.bytes, raw.len() as u64);
        assert_eq!(a.all.fields, fields.len() as u64);
        assert_eq!(a.heldout.fields, fields.len() as u64 - 2);
        assert_eq!(a.uses.iter().sum::<u64>(), a.all.tokens);
    }
    fs::remove_file(path).unwrap();
}

#[test]
fn thirteen_bit_files_decode_and_seek_and_reject_wrong_models_and_truncation() {
    use bre_pair::token_stream::{self, TokenReader, HEADER_BYTES};
    let mut tokens = basis();
    for value in 0..8192 - BASE_SYMBOLS {
        tokens.push(Token {
            bytes: (value as u16).to_le_bytes().to_vec(),
            ends_field: false,
        });
    }
    let tokenizer = Tokenizer::new(Model::new(tokens).unwrap()).unwrap();
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root =
        std::env::temp_dir().join(format!("bre-token-stream-{}-{nonce}", std::process::id()));
    fs::create_dir(&root).unwrap();
    let original = root.join("source.raw");
    let packed = root.join("packed.bts");
    let decoded = root.join("decoded.raw");
    let raw: Vec<_> = (0..=255u8).cycle().take(10000).collect();
    fs::write(&original, &raw).unwrap();
    let header = token_stream::encode_file(&tokenizer, &original, &packed).unwrap();
    assert_eq!(header.width, 13);
    assert_eq!(
        fs::metadata(&packed).unwrap().len(),
        HEADER_BYTES + (header.tokens * 13).div_ceil(8)
    );
    token_stream::verify_file(tokenizer.model(), &packed, &original).unwrap();
    token_stream::decode_file(tokenizer.model(), &packed, &decoded).unwrap();
    assert_eq!(fs::read(&decoded).unwrap(), raw);
    let expected: Vec<_> = raw
        .split_inclusive(|&b| b == b'\n')
        .flat_map(|field| tokenizer.encode(field))
        .collect();
    let mut reader = TokenReader::open(&packed, tokenizer.model()).unwrap();
    for index in (0..expected.len()).rev() {
        assert_eq!(reader.token_at(index as u64).unwrap(), expected[index]);
    }
    assert!(reader.token_at(expected.len() as u64).is_err());
    drop(reader);
    let mut wrong = tokenizer.model().clone();
    wrong.tokens.last_mut().unwrap().bytes.push(91);
    assert!(TokenReader::open(&packed, &wrong).is_err());
    let data = fs::read(&packed).unwrap();
    let bad = root.join("bad.bts");
    fs::write(&bad, &data[..data.len() - 1]).unwrap();
    assert!(TokenReader::open(&bad, tokenizer.model()).is_err());
    let mut data = data.clone();
    data[32..40].copy_from_slice(&0u64.to_le_bytes());
    fs::write(&bad, &data).unwrap();
    let mut reader = TokenReader::open(&bad, tokenizer.model()).unwrap();
    assert!(reader
        .decode_to(tokenizer.model(), &mut Vec::new())
        .is_err());
    drop(reader);
    for path in [original, packed, decoded, bad] {
        fs::remove_file(path).unwrap();
    }
    fs::remove_dir(root).unwrap();
}
