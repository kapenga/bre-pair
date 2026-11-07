use bre_pair::grammar::{BASE_SYMBOLS, EOF};
use bre_pair::maximal::{self, NativeCompressed};
use bre_pair::packed::PackedSymbols;
use bre_pair::sequence_archive;
use std::collections::BTreeMap;

/// Deliberately slow oracle: rebuild every adjacency and match each chosen RHS
/// directly against a flat sequence. It does not use the linked occurrence index.
fn naive_rules(fields: &[Vec<u8>], minimum: usize, maximal: bool) -> (Vec<Vec<u32>>, Vec<u32>) {
    let mut text: Vec<_> = fields
        .iter()
        .flat_map(|f| f.iter().map(|&b| b as u32).chain([EOF]))
        .collect();
    let mut ends_field = vec![false; BASE_SYMBOLS];
    ends_field[EOF as usize] = true;
    let mut rules = Vec::new();
    loop {
        let mut counts = BTreeMap::new();
        for p in text.windows(2).filter(|p| !ends_field[p[0] as usize]) {
            *counts.entry((p[0], p[1])).or_insert(0usize) += 1;
        }
        let mut candidates: Vec<_> = counts
            .into_iter()
            .filter(|(_, count)| *count >= minimum)
            .collect();
        candidates.sort_by_key(|&(pair, count)| (std::cmp::Reverse(count), pair));
        let mut chosen = None;
        for (pair, _) in candidates {
            let mut starts: Vec<_> = text
                .windows(2)
                .enumerate()
                .filter(|(_, p)| *p == [pair.0, pair.1])
                .map(|(i, _)| i)
                .collect();
            let mut length = 2;
            if maximal {
                while starts.iter().all(|&at| {
                    at > 0
                        && !ends_field[text[at - 1] as usize]
                        && text[at - 1] == text[starts[0] - 1]
                }) {
                    for at in &mut starts {
                        *at -= 1;
                    }
                    length += 1;
                }
                while !ends_field[text[starts[0] + length - 1] as usize]
                    && starts.iter().all(|&at| {
                        at + length < text.len() && text[at + length] == text[starts[0] + length]
                    })
                {
                    length += 1;
                }
            }
            let mut rhs = text[starts[0]..starts[0] + length].to_vec();
            if maximal && rhs.len() > 2 && rhs.first() == rhs.last() {
                rhs.remove(0);
            }
            let mut count = 0;
            let mut at = 0;
            while at + rhs.len() <= text.len() {
                if text[at..].starts_with(&rhs) {
                    count += 1;
                    at += rhs.len();
                } else {
                    at += 1;
                }
            }
            if count >= minimum {
                chosen = Some(rhs);
                break;
            }
        }
        let Some(rhs) = chosen else {
            break;
        };
        let symbol = (BASE_SYMBOLS + rules.len()) as u32;
        let mut output = Vec::new();
        let mut at = 0;
        while at < text.len() {
            if text[at..].starts_with(&rhs) {
                output.push(symbol);
                at += rhs.len();
            } else {
                output.push(text[at]);
                at += 1;
            }
        }
        ends_field.push(ends_field[*rhs.last().unwrap() as usize]);
        rules.push(rhs);
        text = output;
    }
    (rules, text)
}

#[test]
fn paper_example_extends_and_trims_before_replacement() {
    let fields = vec![b"abracadabra".to_vec()];
    let result = maximal::compress(&fields, 2, true, |_| {}).unwrap();
    let first = &result.rounds[0];
    // abra is maximal; the equal endpoints cause the leading a to be trimmed.
    assert_eq!(first.rhs_length, 3);
    assert!(first.trimmed);
    assert_eq!(
        result.compressed.rules[0],
        [b'b' as u32, b'r' as u32, b'a' as u32]
    );
    assert_eq!(
        result.compressed.to_binary().unwrap().decode().unwrap(),
        fields
    );
}

#[test]
fn runs_obey_the_actual_replacement_threshold() {
    let fields = vec![b"aaaaaaa".to_vec()];
    let stopped = maximal::compress(&fields, 4, true, |_| {}).unwrap();
    assert!(stopped.compressed.rules.is_empty());
    let paired = maximal::compress(&fields, 3, true, |_| {}).unwrap();
    assert_eq!(paired.rounds[0].seed_frequency, 6);
    assert_eq!(paired.rounds[0].replacements, 3);
    assert_eq!(
        paired.compressed.to_binary().unwrap().decode().unwrap(),
        fields
    );
}

#[test]
fn repeats_may_absorb_eof_but_never_cross_fields() {
    let fields = vec![b"abcdef".to_vec(); 5];
    let result = maximal::compress(&fields, 4, true, |_| {}).unwrap();
    assert_eq!(result.compressed.rules.len(), 1);
    assert_eq!(result.compressed.rules[0], [97, 98, 99, 100, 101, 102, EOF]);
    assert_eq!(result.compressed.token_count(), 5);
    assert_eq!(
        result.compressed.to_binary().unwrap().decode().unwrap(),
        fields
    );
}

#[test]
fn native_archives_round_trip_empty_binary_and_repetitive_fields() {
    let cases = [
        vec![],
        vec![vec![]],
        vec![vec![], vec![], vec![]],
        vec![(0..=255).collect(), b"abcabcabcabcabcabca".to_vec(), vec![]],
        vec![b"abababababababababa".to_vec(); 4],
    ];
    for fields in cases {
        for maximal in [false, true] {
            let result = maximal::compress(&fields, 2, maximal, |_| {}).unwrap();
            let encoded = sequence_archive::encode(&result.compressed).unwrap();
            let restored = sequence_archive::decode(&encoded).unwrap();
            assert_eq!(restored.to_binary().unwrap().decode().unwrap(), fields);
            assert_eq!(encoded, sequence_archive::encode(&restored).unwrap());
        }
    }
}

#[test]
fn exhaustive_short_sequences_preserve_bytes_and_additive_counts() {
    for pattern in 0..3usize.pow(7) {
        let mut remaining = pattern;
        let field: Vec<_> = (0..7)
            .map(|_| {
                let byte = (remaining % 3) as u8;
                remaining /= 3;
                byte
            })
            .collect();
        let fields = vec![field];
        for maximal in [false, true] {
            let result = maximal::compress(&fields, 2, maximal, |_| {}).unwrap();
            for round in result.rounds {
                assert_eq!(
                    round.before - round.after,
                    (round.rhs_length - 1) * round.replacements
                );
                assert!(round.replacements >= 2);
            }
            assert_eq!(
                result.compressed.to_binary().unwrap().decode().unwrap(),
                fields
            );
        }
    }
}

#[test]
fn local_updates_and_maximal_extensions_match_a_flat_recounting_oracle() {
    let mut state = 8317u64;
    for case in 0..240 {
        let fields: Vec<Vec<u8>> = (0..1 + case % 4)
            .map(|_| {
                (0..20 + case % 61)
                    .map(|_| {
                        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                        ((state >> 32) % 4) as u8
                    })
                    .collect()
            })
            .collect();
        for minimum in [2, 4] {
            for use_maximal in [false, true] {
                let result = maximal::compress(&fields, minimum, use_maximal, |_| {}).unwrap();
                let (rules, text) = naive_rules(&fields, minimum, use_maximal);
                assert_eq!(
                    result.compressed.rules, rules,
                    "case {case}, minimum {minimum}, maximal {use_maximal}"
                );
                assert_eq!(
                    result
                        .compressed
                        .blocks
                        .iter()
                        .flat_map(PackedSymbols::iter)
                        .collect::<Vec<_>>(),
                    text
                );
            }
        }
    }
}

#[test]
fn native_rules_cover_large_arities_and_progressive_symbol_widths() {
    let mut rules = vec![vec![97; 513]];
    for i in 0..600 {
        rules.push(vec![BASE_SYMBOLS as u32 + i, 98]);
    }
    let last = (BASE_SYMBOLS + rules.len() - 1) as u32;
    let mut block = PackedSymbols::new(10);
    block.push(last);
    block.push(EOF);
    let native = NativeCompressed {
        rules,
        blocks: vec![block],
        input_bytes: 1113,
        field_count: 1,
    };
    let encoded = sequence_archive::encode(&native).unwrap();
    let decoded = sequence_archive::decode(&encoded)
        .unwrap()
        .to_binary()
        .unwrap()
        .decode()
        .unwrap();
    assert_eq!(
        decoded,
        [vec![97; 513]
            .into_iter()
            .chain(vec![98; 600])
            .collect::<Vec<_>>()]
    );
}

#[test]
fn native_reader_rejects_truncation_forward_references_and_boundaries() {
    let mut block = PackedSymbols::new(9);
    block.push(BASE_SYMBOLS as u32);
    let valid = NativeCompressed {
        rules: vec![vec![97, 98, EOF]],
        blocks: vec![block],
        input_bytes: 2,
        field_count: 1,
    };
    let encoded = sequence_archive::encode(&valid).unwrap();
    for length in 0..encoded.len() {
        assert!(sequence_archive::decode(&encoded[..length]).is_err());
    }
    let mut trailing = encoded.clone();
    trailing.push(0);
    assert!(sequence_archive::decode(&trailing).is_err());
    // First rule: gamma(2) occupies three bits, then its first 9-bit child.
    // Change that child to 511, which cannot precede rule 257.
    let mut forward = encoded.clone();
    for bit in 3..12 {
        forward[40 + bit / 8] |= 1 << (bit % 8);
    }
    assert!(sequence_archive::decode(&forward).is_err());
    let mut extra_rule_bits = encoded.clone();
    let dictionary_bits = u64::from_le_bytes(encoded[32..40].try_into().unwrap());
    extra_rule_bits[32..40].copy_from_slice(&(dictionary_bits + 1).to_le_bytes());
    assert!(sequence_archive::decode(&extra_rule_bits).is_err());
    let mut invalid_arity = encoded.clone();
    // An unterminated gamma prefix consumes the dictionary and must fail.
    let dictionary_bytes = dictionary_bits.div_ceil(8) as usize;
    invalid_arity[40..40 + dictionary_bytes].fill(0);
    assert!(sequence_archive::decode(&invalid_arity).is_err());
    let invalid = NativeCompressed {
        rules: vec![vec![EOF, 97]],
        ..valid
    };
    let bytes = sequence_archive::encode(&invalid).unwrap();
    assert!(sequence_archive::decode(&bytes).is_err());
    let invalid = NativeCompressed {
        rules: vec![vec![BASE_SYMBOLS as u32, 97]],
        ..invalid
    };
    assert!(sequence_archive::encode(&invalid).is_err());
}
