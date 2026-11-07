use bre_pair::{
    disk_seed,
    engine::{Compressed, Config},
    field_engine::{self, Options},
    scoring::Model,
    search,
};
use std::io::{BufReader, Cursor};

fn config(score: Model, waves: u8, minimum: u64) -> Config {
    Config {
        buffer_bytes: 64_000_000,
        minimum_count: minimum,
        score,
        search: search::Options {
            fill_waves: waves,
            fill_min_count: minimum,
            certify: false,
            ..Default::default()
        },
        ..Default::default()
    }
}
fn fields(raw: &[u8]) -> Vec<Vec<u8>> {
    if raw.is_empty() {
        vec![]
    } else {
        raw.split_inclusive(|&b| b == b'\n')
            .map(<[u8]>::to_vec)
            .collect()
    }
}
fn symbols(source: &Compressed) -> Vec<u32> {
    source.blocks.iter().flat_map(|b| b.iter()).collect()
}
fn definitions(source: &Compressed) -> Vec<(u32, u32, usize, bool)> {
    source
        .grammar
        .rules
        .iter()
        .map(|r| (r.left, r.right, r.terminals, r.ends_field))
        .collect()
}
fn compare(raw: &[u8], config: &Config, chunk: usize) {
    let expected = field_engine::compress_with_options(
        &fields(raw),
        config,
        1,
        64,
        chunk,
        Options {
            max_rounds: Some(1),
            frequency_cutoff: true,
            memory_saving: true,
            ..Default::default()
        },
        |_| {},
    )
    .unwrap();
    for read_chunk in [1, 7, 65536] {
        let seed = disk_seed::prepare(
            BufReader::with_capacity(read_chunk, Cursor::new(raw)),
            BufReader::with_capacity(read_chunk, Cursor::new(raw)),
            config,
            chunk,
        )
        .unwrap();
        assert_eq!(
            definitions(&seed.compressed),
            definitions(&expected.training.compressed)
        );
        assert_eq!(
            symbols(&seed.compressed),
            symbols(&expected.training.compressed)
        );
        assert_eq!(seed.compressed.decode().unwrap(), fields(raw));
        assert_eq!(
            seed.report.symbols_before - seed.report.replacements,
            seed.report.symbols_after
        );
        assert_eq!(
            seed.report.symbols_after,
            symbols(&seed.compressed).len() as u64
        );
        assert_eq!(seed.report.input_bytes, raw.len() as u64);
        assert_eq!(seed.report.fields, fields(raw).len() as u64);
    }
}
#[test]
fn disk_batch_matches_normal_first_batch_for_bytes_runs_fields_and_score_modes() {
    let inputs = vec![
        vec![],
        b"\n".to_vec(),
        b"aaaaa\naabcdaafgcdaacgadaaee\nabcabc\n\nlast".repeat(7),
        (0..=255u8).cycle().take(4001).collect(),
        b"a".repeat(4001),
    ];
    for raw in inputs {
        for score in [Model::Frequency, Model::WeightedPmi] {
            for waves in [1, 2] {
                for minimum in [2, 4] {
                    compare(&raw, &config(score, waves, minimum), 17);
                }
            }
        }
    }
}
#[test]
fn exact_observed_conflicts_survive_a_large_first_batch_and_width_growth() {
    let mut raw = Vec::new();
    for a in (0..128u8).filter(|&a| a != b'\n') {
        for b in 128..=255u8 {
            for _ in 0..2 {
                raw.extend([a, b, b'\n']);
            }
        }
    }
    let config = config(Model::Frequency, 2, 2);
    let seed = disk_seed::prepare(Cursor::new(&raw), Cursor::new(&raw), &config, 65536).unwrap();
    assert!(seed.report.selected_pairs > 255);
    assert!(seed.report.width > 9);
    compare(&raw, &config, 65536);
}
#[test]
fn continued_training_matches_the_normal_complete_grammar() {
    let raw = b"aaaaa abcabc xyzxyz\nabcabc aaa\n\nlast line".repeat(53);
    for score in [Model::Frequency, Model::WeightedPmi] {
        let config = config(score, 2, 4);
        let seed = disk_seed::prepare(Cursor::new(&raw), Cursor::new(&raw), &config, 37).unwrap();
        let options = Options {
            frequency_cutoff: true,
            memory_saving: true,
            ..Default::default()
        };
        let normal =
            field_engine::compress_with_options(&fields(&raw), &config, 2, 64, 37, options, |_| {})
                .unwrap();
        let continued =
            field_engine::compress_prepacked(seed.compressed, &config, 2, 37, options, |_| {})
                .unwrap();
        assert_eq!(
            definitions(&continued.training.compressed),
            definitions(&normal.training.compressed)
        );
        assert_eq!(
            symbols(&continued.training.compressed),
            symbols(&normal.training.compressed)
        );
        assert_eq!(
            continued.training.compressed.decode().unwrap(),
            fields(&raw)
        );
    }
}
#[test]
fn source_changes_and_unsupported_configuration_fail_before_archive_output() {
    let config = config(Model::WeightedPmi, 2, 2);
    let first = b"aaaaa\nababab\n";
    for second in [
        b"aaaaa\nababac\n".as_slice(),
        b"aaaaa\nababab\na\n".as_slice(),
        b"".as_slice(),
    ] {
        assert!(disk_seed::prepare(Cursor::new(first), Cursor::new(second), &config, 17).is_err());
    }
    let mut unsupported = config.clone();
    unsupported.search.frequency_override_percent = 50;
    assert!(disk_seed::prepare(Cursor::new(first), Cursor::new(first), &unsupported, 17).is_err());
    assert!(disk_seed::prepare(Cursor::new(first), Cursor::new(first), &config, 0).is_err());
    let mut small = config;
    small.buffer_bytes = 1;
    assert!(disk_seed::prepare(Cursor::new(first), Cursor::new(first), &small, 17).is_err());
}
