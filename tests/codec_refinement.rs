use bre_pair::{
    codec::{
        refinement::{Calibration, Options},
        Frozen,
    },
    huffman::Codebook,
    huffman_archive::{Encoder, Reader, Totals},
    huffman_dictionary::{byte_model, Dictionary},
};
use std::io::Cursor;

fn fixture() -> (Frozen, Vec<u8>) {
    // The original boundaries are a | bc. ab is available but has no direct
    // uses; keeping a zero-use code can nevertheless improve the next parse.
    let dictionary = Dictionary::new(byte_model(), vec![(97, 98), (98, 99)], true).unwrap();
    let stream: Vec<_> = (0..64).flat_map(|_| [97, 258, 256]).collect();
    let mut weights = vec![1; 259];
    weights[257] = 512;
    let book = Codebook::from_counts(&weights).unwrap();
    let training = Totals {
        bytes: 192,
        fields: 64,
        tokens: 256,
        symbols: 192,
    };
    let model = Frozen {
        dictionary,
        book,
        training,
    };
    let mut archive = Cursor::new(Vec::new());
    let mut encoder =
        Encoder::with_codebook(&mut archive, &model.dictionary, 7, &model.book).unwrap();
    encoder.extend(&stream).unwrap();
    encoder.finish(192, 64).unwrap();
    (model, archive.into_inner())
}

#[test]
fn complete_reparse_crosses_old_roots_and_keeps_exact_ai_ids() {
    let (model, archive) = fixture();
    let calibration = Calibration::read(
        &model,
        Reader::new(Cursor::new(archive.clone())).unwrap(),
        4096,
    )
    .unwrap();
    let mut trials = Vec::new();
    let (result, report) = model
        .refine_codes(
            &calibration,
            Options {
                max_trials: 8,
                refits: 2,
            },
            |trial| {
                trials.push((trial.accepted, trial.boundary_gain_bits));
            },
        )
        .unwrap();
    assert!(report.after_payload_bits < calibration.original_payload_bits);
    assert!(
        report.after_payload_bits + 8 * report.after_metadata_bytes as u128
            <= report.before_payload_bits + 8 * report.before_metadata_bytes as u128
    );
    assert_eq!(
        result.dictionary.model.to_bytes(),
        model.dictionary.model.to_bytes()
    );
    assert_eq!(result.training.tokens, model.training.tokens);
    // The initial minimum-bit parse itself recovers ab | c from a | bc.
    assert!(report.before_payload_bits < calibration.original_payload_bits);
    let expected: Vec<_> = (0..64)
        .flat_map(|_| [97u32, 98, 99, 256])
        .flat_map(u32::to_le_bytes)
        .collect();
    let mut reader = Reader::new(Cursor::new(archive)).unwrap();
    let mut ids = Vec::new();
    reader.decode_to(true, &mut ids).unwrap();
    assert_eq!(ids, expected);
    assert_eq!(trials.len(), report.trials);
}

#[test]
fn calibration_rejects_wrong_model_incomplete_input_and_oversized_storage() {
    let (mut model, archive) = fixture();
    assert!(Calibration::read(
        &model,
        Reader::new(Cursor::new(archive.clone())).unwrap(),
        8
    )
    .is_err());
    let calibration = Calibration::read(
        &model,
        Reader::new(Cursor::new(archive.clone())).unwrap(),
        4096,
    )
    .unwrap();
    model.training.bytes += 1;
    assert!(Calibration::read(&model, Reader::new(Cursor::new(archive)).unwrap(), 4096).is_err());
    model.training.bytes -= 1;
    model.book = Codebook::from_counts(&vec![1; 259]).unwrap();
    assert!(model
        .refine_codes(&calibration, Options::default(), |_| {})
        .is_err());
}

#[test]
fn unchanged_zero_group_does_not_skip_positive_proposals() {
    let dictionary = Dictionary::new(byte_model(), vec![(97, 98)], true).unwrap();
    let mut counts = vec![1u64; 258];
    counts[257] = 16;
    let book = Codebook::from_counts(&counts).unwrap();
    let stream: Vec<_> = (0..16).flat_map(|_| [257, 256]).collect();
    let model = Frozen {
        dictionary,
        book,
        training: Totals {
            bytes: 32,
            fields: 16,
            tokens: 48,
            symbols: 32,
        },
    };
    let mut archive = Cursor::new(Vec::new());
    let mut encoder =
        Encoder::with_codebook(&mut archive, &model.dictionary, 1024, &model.book).unwrap();
    encoder.extend(&stream).unwrap();
    encoder.finish(32, 16).unwrap();
    let calibration = Calibration::read(
        &model,
        Reader::new(Cursor::new(archive.into_inner())).unwrap(),
        4096,
    )
    .unwrap();
    let mut saw_positive = false;
    model
        .refine_codes(
            &calibration,
            Options {
                max_trials: 4,
                refits: 0,
            },
            |t| {
                saw_positive |= t.removed_id == Some(257);
            },
        )
        .unwrap();
    assert!(saw_positive);
    let (same, report) = model
        .refine_codes(
            &calibration,
            Options {
                max_trials: 0,
                refits: 0,
            },
            |_| panic!(),
        )
        .unwrap();
    assert_eq!(same.to_bytes().unwrap(), model.to_bytes().unwrap());
    assert_eq!(report.trials, 0);
}

#[test]
fn removed_code_is_rescored_with_alternative_across_its_old_boundary() {
    let dictionary = Dictionary::new(byte_model(), vec![(97, 98), (98, 99)], true).unwrap();
    let stream: Vec<_> = (0..64).flat_map(|_| [97, 258, 256, 257, 256]).collect();
    let mut weights = vec![1; 259];
    for (id, n) in [(97, 64), (258, 64), (257, 64), (256, 128)] {
        weights[id] = n;
    }
    let model = Frozen {
        dictionary,
        book: Codebook::from_counts(&weights).unwrap(),
        training: Totals {
            bytes: 320,
            fields: 128,
            tokens: 448,
            symbols: 320,
        },
    };
    let mut archive = Cursor::new(Vec::new());
    let mut encoder =
        Encoder::with_codebook(&mut archive, &model.dictionary, 1024, &model.book).unwrap();
    encoder.extend(&stream).unwrap();
    encoder.finish(320, 128).unwrap();
    let calibration = Calibration::read(
        &model,
        Reader::new(Cursor::new(archive.into_inner())).unwrap(),
        4096,
    )
    .unwrap();
    let mut gain = None;
    model
        .refine_codes(
            &calibration,
            Options {
                max_trials: 4,
                refits: 0,
            },
            |trial| {
                if trial.removed_id == Some(258) {
                    gain = Some(trial.boundary_gain_bits);
                }
            },
        )
        .unwrap();
    assert!(
        gain.unwrap() > 0,
        "surviving ab should cover a | b from the expanded bc"
    );
}
