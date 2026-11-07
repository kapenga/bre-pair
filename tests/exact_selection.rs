use bre_pair::{
    grammar::BASE_SYMBOLS,
    tokenizer::{Model, Token, Tokenizer},
    vocabulary,
};
use std::{
    fs,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

struct Input(PathBuf);
impl Input {
    fn new(raw: &[u8]) -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "bre-exact-selection-{}-{nonce}",
            std::process::id()
        ));
        fs::write(&path, raw).unwrap();
        Self(path)
    }
}
impl Drop for Input {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}
fn model(extras: Vec<Token>) -> Model {
    let mut basis: Vec<_> = (0..=255)
        .map(|b| Token {
            bytes: vec![b],
            ends_field: false,
        })
        .chain([Token {
            bytes: vec![],
            ends_field: true,
        }])
        .collect();
    basis.extend(extras);
    Model::new(basis).unwrap()
}
// Independent shortest-path cost; no production trie, masks or usage counters.
fn minimum(model: &Model, field: &[u8]) -> u64 {
    let labels: Vec<_> = field.iter().map(|&b| b as u16).chain([256]).collect();
    let mut costs = vec![u64::MAX; labels.len() + 1];
    costs[0] = 0;
    for at in 0..labels.len() {
        for token in &model.tokens {
            let t: Vec<_> = token
                .bytes
                .iter()
                .map(|&b| b as u16)
                .chain(token.ends_field.then_some(256))
                .collect();
            let end = at + t.len();
            if end <= labels.len() && labels[at..end] == t && costs[at] != u64::MAX {
                costs[end] = costs[end].min(costs[at] + 1);
            }
        }
    }
    costs[labels.len()]
}

#[test]
fn exact_emission_selection_chooses_the_best_single_piece_and_reads_the_full_tail() {
    let ab = Token {
        bytes: b"ab".to_vec(),
        ends_field: false,
    };
    let xy = Token {
        bytes: b"xy".to_vec(),
        ends_field: false,
    };
    let mut raw = b"ab\n".repeat(2);
    raw.extend(b"xy\n".repeat(19));
    let input = Input::new(&raw);
    let pool = model(vec![ab.clone(), xy.clone()]);
    let mut output = None;
    for workers in [1, 4] {
        let mut rounds = 0;
        let selected = vocabulary::select_exact(
            pool.clone(),
            &vec![0; pool.tokens.len()],
            &input.0,
            258,
            workers,
            |r| {
                rounds += 1;
                assert_eq!(r.totals.bytes, raw.len() as u64);
                assert_eq!(r.totals.fields, 21);
                assert_eq!(r.totals.tokens, 63);
            },
        )
        .unwrap();
        assert_eq!(rounds, 1);
        assert_eq!(selected.tokens.last().unwrap(), &xy);
        let cost: u64 = raw
            .split_inclusive(|&b| b == b'\n')
            .map(|f| minimum(&selected, f))
            .sum();
        for candidate in [ab.clone(), xy.clone()] {
            let alternative = model(vec![candidate]);
            let other: u64 = raw
                .split_inclusive(|&b| b == b'\n')
                .map(|f| minimum(&alternative, f))
                .sum();
            assert!(cost <= other);
        }
        let bytes = selected.to_bytes();
        if let Some(expected) = &output {
            assert_eq!(&bytes, expected);
        } else {
            output = Some(bytes);
        }
    }
}

#[test]
fn exact_rounds_recount_after_pruning_and_compaction_preserves_counts() {
    let mut extras = vec![];
    let mut raw = vec![];
    for i in 0..100 {
        let pair = vec![b'a' + i / 10, b'0' + i % 10];
        extras.push(Token {
            bytes: pair.clone(),
            ends_field: false,
        });
        for _ in 0..=i {
            raw.extend(&pair);
            raw.push(b'\n');
        }
    }
    let input = Input::new(&raw);
    let pool = model(extras);
    let mut measured = vec![];
    let selected = vocabulary::select_exact(
        pool.clone(),
        &vec![0; pool.tokens.len()],
        &input.0,
        BASE_SYMBOLS + 1,
        4,
        |r| {
            assert_eq!(r.totals.bytes, raw.len() as u64);
            assert_eq!(r.totals.fields, 5050);
            measured.push(r.totals.tokens);
        },
    )
    .unwrap();
    assert_eq!(measured, vec![15150, 16635]);
    assert_eq!(selected.tokens.last().unwrap().bytes, b"j9");
    let scan = vocabulary::scan(
        &input.0,
        &Tokenizer::new(selected).unwrap(),
        None,
        None,
        2,
        1,
        0,
        true,
        true,
    )
    .unwrap();
    assert_eq!(scan.all.tokens, 20100);
    assert_eq!(scan.uses.iter().sum::<u64>(), 20100);
}

#[test]
fn exact_selection_rejects_invalid_parameters() {
    let input = Input::new(b"ab\n");
    let pool = model(vec![Token {
        bytes: b"ab".to_vec(),
        ends_field: false,
    }]);
    for (size, workers, n) in [(256, 1, 258), (258, 0, 258), (259, 1, 258), (258, 1, 257)] {
        assert!(vocabulary::select_exact(
            pool.clone(),
            &vec![0; n],
            &input.0,
            size,
            workers,
            |_| {}
        )
        .is_err());
    }
}

#[test]
fn compact_exact_selection_matches_uncompacted_full_parses_with_overlaps_and_eof() {
    let mut extras = vec![];
    for (bytes, ends_field) in [
        (b"ab".as_slice(), false),
        (b"bc", false),
        (b"abc", false),
        (b"abcabc", false),
        (b"a", true),
        (b"\n", true),
        (b"abc\n", true),
        (b"xy", false),
    ] {
        extras.push(Token {
            bytes: bytes.to_vec(),
            ends_field,
        });
    }
    for n in 0..120 {
        extras.push(Token {
            bytes: format!("unused{n}").into_bytes(),
            ends_field: false,
        });
    }
    let pool = model(extras);
    let support = vec![0; pool.tokens.len()];
    let input = Input::new(&b"abc\nbcabc\nabcabc\nxyxy\n\x00\xffabc\n".repeat(10));
    let mut exact_rounds = vec![];
    let exact = vocabulary::select_exact(pool.clone(), &support, &input.0, 265, 4, |r| {
        exact_rounds.push(r.totals.tokens)
    })
    .unwrap();
    let mut reference_rounds = vec![];
    let reference = vocabulary::select(
        pool,
        &support,
        &input.0,
        &vocabulary::SelectionOptions {
            vocabulary: 265,
            workers: 1,
            method: vocabulary::SelectionMethod::Sampled,
            sample_bytes: u64::MAX,
            ..vocabulary::SelectionOptions::default()
        },
        |r| reference_rounds.push(r.totals.unwrap().tokens),
    )
    .unwrap();
    assert!(exact_rounds.len() > 1);
    assert_eq!(exact_rounds, reference_rounds);
    assert_eq!(exact.to_bytes(), reference.to_bytes());
}

#[test]
fn tapered_schedule_recounts_more_often_and_rejects_nonterminating_rates() {
    let mut extras = vec![];
    let mut raw = vec![];
    for i in 0..100 {
        let pair = vec![b'a' + i / 10, b'0' + i % 10];
        extras.push(Token {
            bytes: pair.clone(),
            ends_field: false,
        });
        for _ in 0..=i {
            raw.extend(&pair);
            raw.push(b'\n');
        }
    }
    let pool = model(extras);
    let input = Input::new(&raw);
    let support = vec![0; pool.tokens.len()];
    let mut plan = vec![];
    let schedule = vocabulary::PruningSchedule {
        coarse_keep: 0.85,
        fine_keep: 0.95,
        fine_below: 300,
    };
    let selected = vocabulary::select_exact_with_schedule(
        pool.clone(),
        &support,
        &input.0,
        258,
        2,
        schedule,
        |r| plan.push((r.before, r.after)),
    )
    .unwrap();
    assert_eq!(
        plan,
        [(357, 303), (303, 300), (300, 285), (285, 270), (270, 258)]
    );
    assert_eq!(selected.tokens.last().unwrap().bytes, b"j9");
    for keep in [0.0, 1.0, f64::NAN] {
        let invalid = vocabulary::PruningSchedule {
            fine_keep: keep,
            ..schedule
        };
        assert!(vocabulary::select_exact_with_schedule(
            pool.clone(),
            &support,
            &input.0,
            258,
            2,
            invalid,
            |_| {}
        )
        .is_err());
    }
}
