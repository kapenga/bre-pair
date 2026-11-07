use bre_pair::{
    codec::{
        recipe_compaction::{analyze, Compact},
        Frozen,
    },
    huffman::Codebook,
    huffman_archive::Totals,
    huffman_dictionary::{byte_model, Dictionary},
};

#[test]
fn nary_inline_removes_construction_helpers_and_round_trips_bytes_and_ai_ids() {
    let dictionary = Dictionary::new(
        byte_model(),
        vec![(97, 98), (99, 100), (257, 97), (259, 256)],
        true,
    )
    .unwrap();
    let mut weights = vec![0; 261];
    weights[..257].fill(1);
    weights[260] = 512;
    let model = Frozen {
        dictionary,
        book: Codebook::from_counts(&weights).unwrap(),
        training: Totals {
            bytes: 300,
            fields: 100,
            tokens: 400,
            symbols: 100,
        },
    };
    let result = analyze(&model, 128).unwrap();
    let inlined = result.variants.iter().find(|v| v.name == "inline").unwrap();
    assert_eq!(inlined.rules, 1);
    assert_eq!(inlined.nonbinary_rules, 1);
    assert_eq!(inlined.child_references, 4);
    let restored = Compact::from_bytes(&result.compact.to_bytes()).unwrap();
    assert_eq!(restored.model.to_bytes(), model.dictionary.model.to_bytes());
    for ids in [false, true] {
        let mut output = Vec::<u8>::new();
        restored
            .emit(result.mapping[260], ids, |b| {
                output.extend(b);
                Ok(())
            })
            .unwrap();
        let expected = if ids {
            [97u32, 98, 97, 256]
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect()
        } else {
            b"aba".to_vec()
        };
        assert_eq!(output, expected);
    }
    let mut damaged = result.compact.to_bytes();
    damaged[45] ^= 1;
    assert!(Compact::from_bytes(&damaged).is_err());
    assert!(Compact::from_bytes(b"BRMGP001").is_err());
    assert!(restored.emit(u32::MAX, false, |_| Ok(())).is_err());
}

#[test]
fn identical_bytes_with_different_ai_ids_stay_distinct() {
    let mut tokens = byte_model().tokens;
    tokens.push(bre_pair::tokenizer::Token {
        bytes: b"ab".to_vec(),
        ends_field: false,
    });
    let vocabulary = bre_pair::tokenizer::Model::new(tokens).unwrap();
    let dictionary =
        Dictionary::new(vocabulary, vec![(97, 98), (257, 99), (258, 99)], true).unwrap();
    let model = Frozen {
        dictionary,
        book: Codebook::from_counts(&[1; 261]).unwrap(),
        training: Totals::default(),
    };
    let result = analyze(&model, 128).unwrap();
    let mut outputs = Vec::new();
    for id in [259, 260] {
        let mut output = Vec::<u8>::new();
        result
            .compact
            .emit(result.mapping[id], true, |b| {
                output.extend(b);
                Ok(())
            })
            .unwrap();
        outputs.push(output);
    }
    assert_ne!(outputs[0], outputs[1]);
    assert_eq!(
        outputs[0],
        [257u32, 99]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        outputs[1],
        [97u32, 98, 99]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>()
    );
}
