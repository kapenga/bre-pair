use bre_pair::{
    archive, archive_io, batch::Counting, engine::Config, field_engine, registry::Registry,
    scoring::Model,
};
use std::{
    fs,
    path::PathBuf,
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        let id = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("bre-bounded-{}-{id}", std::process::id()));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn retired_pairs_stay_absent_but_zero_live_keys_survive_growth() {
    let mut registry = Registry::new(2).unwrap();
    let mut writer = registry.writer();
    writer.pair((1, 2), 8).unwrap();
    writer.pair((2, 3), 3).unwrap();
    writer.finish().unwrap();
    assert_eq!(registry.compact_frequent(8, 257).unwrap(), 1);
    let mut writer = registry.writer();
    writer.pair((2, 3), -3).unwrap();
    writer.pair((2, 3), 3).unwrap();
    writer.pair((1, 2), -8).unwrap();
    writer.finish().unwrap();
    let mut writer = registry.writer();
    for id in 257..3000 {
        writer.pair((id, id + 1), 1).unwrap();
    }
    writer.finish().unwrap();
    let mut writer = registry.writer();
    writer.pair((1, 2), 1).unwrap();
    writer.finish().unwrap();
    assert_eq!(registry.freeze().get((1, 2)), 1);
    assert_eq!(registry.freeze().get((2, 3)), 0);
    registry.clear();
    let mut writer = registry.writer();
    writer.pair((2, 3), 9).unwrap();
    writer.finish().unwrap();
    assert_eq!(registry.freeze().get((2, 3)), 9);
}

#[test]
fn streaming_memory_mode_matches_every_reference_batch_and_archive() {
    let dir = Fixture::new();
    let input = dir.0.join("input");
    let mut raw = b"aaaaa aabcdaafgcdaacgadaaee abcabc he the there\n\n".repeat(18);
    let mut seed = 571u64;
    for i in 0..3000 {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        raw.push(if i % 73 == 0 {
            b'\n'
        } else {
            (seed % 19) as u8
        });
    }
    raw.extend([0, 255, 0, 255, b'\n']);
    fs::write(&input, &raw).unwrap();
    let fields: Vec<_> = raw
        .split_inclusive(|&b| b == b'\n')
        .map(<[u8]>::to_vec)
        .collect();
    let defaults = bre_pair::search::Options::default();
    let mut searches = vec![
        defaults,
        bre_pair::search::Options {
            frequency_override_percent: 50,
            ..defaults
        },
    ];
    for batch_priority in [
        bre_pair::search::BatchPriority::Score,
        bre_pair::search::BatchPriority::Bands2,
        bre_pair::search::BatchPriority::Bands4,
        bre_pair::search::BatchPriority::Bounded2,
        bre_pair::search::BatchPriority::Bounded4,
        bre_pair::search::BatchPriority::Bounded8,
    ] {
        for fill_waves in [1, 2] {
            searches.push(bre_pair::search::Options {
                batch_priority,
                fill_waves,
                ..defaults
            });
        }
    }
    searches.push(bre_pair::search::Options {
        fill_waves: 4,
        fill_min_count: 16,
        ..defaults
    });
    searches.push(bre_pair::search::Options {
        fill_waves: 2,
        fill_min_count: 8,
        fill_after_rounds: 2,
        ..defaults
    });
    for search in searches {
        for score in [Model::Frequency, Model::WeightedPmi] {
            for counting in [Counting::Full, Counting::Incremental, Counting::Adaptive] {
                for workers in [1, 4] {
                    let config = Config {
                        buffer_bytes: 1 << 20,
                        minimum_count: 4,
                        score,
                        pair_counting: counting,
                        search,
                        ..Default::default()
                    };
                    let mut expected_rounds = Vec::new();
                    let mut actual_rounds = Vec::new();
                    let signature = |r: &bre_pair::engine::Round| {
                        (r.candidates, r.selected, r.total_rules, r.rewritten)
                    };
                    let expected = field_engine::compress_with_options(
                        &fields,
                        &config,
                        workers,
                        8,
                        61,
                        Default::default(),
                        |r| expected_rounds.push(signature(r)),
                    )
                    .unwrap();
                    let actual = field_engine::compress_file(
                        &input,
                        &config,
                        workers,
                        31,
                        field_engine::Options {
                            frequency_cutoff: true,
                            memory_saving: true,
                            ..Default::default()
                        },
                        |r| actual_rounds.push(signature(r)),
                    )
                    .unwrap();
                    assert_eq!(
                        actual_rounds, expected_rounds,
                        "{score:?} {counting:?} {workers}"
                    );
                    let mut a = Vec::new();
                    let mut b = Vec::new();
                    archive_io::write(&actual.training.compressed, &mut a).unwrap();
                    archive_io::write(&expected.training.compressed, &mut b).unwrap();
                    assert_eq!(a, b);
                    assert_eq!(archive::decode(&a).unwrap().decode().unwrap(), fields);
                    let mut decoded = Vec::new();
                    archive_io::decompress(&mut a.as_slice(), &mut decoded).unwrap();
                    assert_eq!(decoded, raw);
                }
            }
        }
    }
}

#[test]
fn delayed_fill_preserves_the_exact_standard_prefix() {
    let dir = Fixture::new();
    let input = dir.0.join("warmup");
    let raw = b"aaaaa abcdef abcabc the there their aabcdaafgcdaacgadaaee\n".repeat(80);
    fs::write(&input, &raw).unwrap();
    let config = Config {
        minimum_count: 4,
        score: Model::WeightedPmi,
        ..Default::default()
    };
    let options = field_engine::Options {
        max_rounds: Some(3),
        frequency_cutoff: true,
        memory_saving: true,
        ..Default::default()
    };
    let normal = field_engine::compress_file(&input, &config, 4, 61, options, |_| {}).unwrap();
    assert_eq!(normal.training.rounds.len(), 3);
    let delayed = Config {
        search: bre_pair::search::Options {
            fill_waves: 2,
            fill_min_count: 8,
            fill_after_rounds: 3,
            ..Default::default()
        },
        ..config
    };
    let prefix = field_engine::compress_file(&input, &delayed, 4, 61, options, |_| {}).unwrap();
    let mut a = Vec::new();
    let mut b = Vec::new();
    archive_io::write(&normal.training.compressed, &mut a).unwrap();
    archive_io::write(&prefix.training.compressed, &mut b).unwrap();
    assert_eq!(a, b);
    let full = field_engine::compress_file(
        &input,
        &delayed,
        4,
        61,
        field_engine::Options {
            max_rounds: None,
            ..options
        },
        |_| {},
    )
    .unwrap();
    assert!(full.training.rounds.len() > 3);
    let signature =
        |r: &bre_pair::engine::Round| (r.selected, r.new_rules, r.total_rules, r.rewritten);
    assert_eq!(
        full.training.rounds[..3]
            .iter()
            .map(signature)
            .collect::<Vec<_>>(),
        normal
            .training
            .rounds
            .iter()
            .map(signature)
            .collect::<Vec<_>>()
    );
    let mut encoded = Vec::new();
    archive_io::write(&full.training.compressed, &mut encoded).unwrap();
    let mut decoded = Vec::new();
    archive_io::decompress(&mut encoded.as_slice(), &mut decoded).unwrap();
    assert_eq!(decoded, raw);
    let fields: Vec<_> = raw
        .split_inclusive(|&b| b == b'\n')
        .map(<[u8]>::to_vec)
        .collect();
    let reference = bre_pair::engine::compress(&fields, &delayed, |_| {}).unwrap();
    let mut reference_bytes = Vec::new();
    archive_io::write(&reference.compressed, &mut reference_bytes).unwrap();
    assert_eq!(reference_bytes, encoded);
}

#[test]
fn empty_and_binary_files_stream_and_malformed_archives_fail() {
    let dir = Fixture::new();
    let input = dir.0.join("input");
    for raw in [
        vec![],
        vec![0, 255],
        vec![b'\n', b'\n'],
        b"last field without newline".to_vec(),
    ] {
        fs::write(&input, &raw).unwrap();
        let result = field_engine::compress_file(
            &input,
            &Config {
                buffer_bytes: 1 << 20,
                minimum_count: 2,
                ..Default::default()
            },
            2,
            8,
            field_engine::Options {
                frequency_cutoff: true,
                memory_saving: true,
                ..Default::default()
            },
            |_| {},
        )
        .unwrap();
        let mut bytes = Vec::new();
        archive_io::write(&result.training.compressed, &mut bytes).unwrap();
        let mut decoded = Vec::new();
        archive_io::decompress(&mut bytes.as_slice(), &mut decoded).unwrap();
        assert_eq!(decoded, raw);
        for cut in [0, bytes.len() - 1] {
            assert!(archive_io::decompress(&mut &bytes[..cut], &mut Vec::new()).is_err());
        }
        let mut extra = bytes.clone();
        extra.push(0);
        assert!(archive_io::decompress(&mut extra.as_slice(), &mut Vec::new()).is_err());
    }
}

#[test]
fn public_cli_round_trips_enforces_memory_and_preserves_existing_outputs() {
    let dir = Fixture::new();
    let input = dir.0.join("input");
    let archive = dir.0.join("data.brp");
    let decoded = dir.0.join("decoded");
    let raw = b"aaaaa abcabc abcabc\n".repeat(25);
    fs::write(&input, &raw).unwrap();
    let run = |args: &[&str]| {
        Command::new(env!("CARGO_BIN_EXE_brepair"))
            .args(args)
            .output()
            .unwrap()
    };
    let result = run(&[
        "train",
        input.to_str().unwrap(),
        archive.to_str().unwrap(),
        "--min-count",
        "2",
        "--memory",
        "64MB",
        "--verify",
        "--frequency-override-percent",
        "50",
    ]);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(
        String::from_utf8_lossy(&result.stdout).contains("\"source_and_archive_verified\":true")
    );
    assert!(String::from_utf8_lossy(&result.stdout).contains("\"frequency_override_percent\":50"));
    let invalid = dir.0.join("invalid.brp");
    let result = run(&[
        "train",
        input.to_str().unwrap(),
        invalid.to_str().unwrap(),
        "--frequency-override-percent",
        "101",
    ]);
    assert!(!result.status.success());
    assert!(!invalid.exists());
    let filled = dir.0.join("filled.brp");
    let rounds = dir.0.join("rounds.csv");
    let result = run(&[
        "train",
        input.to_str().unwrap(),
        filled.to_str().unwrap(),
        "--min-count",
        "2",
        "--fill-waves",
        "2",
        "--fill-min-count",
        "8",
        "--verbose",
        "--rounds",
        rounds.to_str().unwrap(),
        "--verify",
    ]);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(String::from_utf8_lossy(&result.stdout).contains("\"fill_min_count\":8"));
    assert!(String::from_utf8_lossy(&result.stderr).contains("batch 1:"));
    assert!(fs::read_to_string(&rounds)
        .unwrap()
        .starts_with("batch,before,after,candidates"));
    let sidecar = dir.0.join("collision.json");
    let result = run(&[
        "train",
        input.to_str().unwrap(),
        invalid.to_str().unwrap(),
        "--report",
        sidecar.to_str().unwrap(),
        "--rounds",
        sidecar.to_str().unwrap(),
    ]);
    assert!(!result.status.success());
    assert!(!invalid.exists());
    assert!(!sidecar.exists());
    let strict = dir.0.join("strict.brp");
    let scaled = dir.0.join("scaled.brp");
    assert!(run(&[
        "train",
        input.to_str().unwrap(),
        strict.to_str().unwrap(),
        "--min-count",
        "2"
    ])
    .status
    .success());
    let result = run(&[
        "train",
        input.to_str().unwrap(),
        scaled.to_str().unwrap(),
        "--min-count",
        "2",
        "--fill-waves",
        "2",
        "--fill-min-count-per-gb",
        "1000000000",
    ]);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(String::from_utf8_lossy(&result.stdout)
        .contains(&format!("\"fill_min_count\":{}", raw.len())));
    assert_eq!(fs::read(scaled).unwrap(), fs::read(strict).unwrap());
    assert!(run(&[
        "decompress",
        archive.to_str().unwrap(),
        decoded.to_str().unwrap(),
        "--memory",
        "64MB"
    ])
    .status
    .success());
    assert_eq!(fs::read(decoded).unwrap(), raw);
    let saved = fs::read(&archive).unwrap();
    assert!(
        !run(&["train", input.to_str().unwrap(), archive.to_str().unwrap()])
            .status
            .success()
    );
    assert_eq!(fs::read(&archive).unwrap(), saved);
    assert!(run(&["train", "--help"]).status.success());
    let alias = dir.0.join("alias.brt");
    let alias_report = dir.0.join(".").join("alias.brt");
    let rejected = run(&[
        "vocab",
        archive.to_str().unwrap(),
        input.to_str().unwrap(),
        alias.to_str().unwrap(),
        "--report",
        alias_report.to_str().unwrap(),
    ]);
    assert!(!rejected.status.success());
    assert!(String::from_utf8_lossy(&rejected.stderr).contains("must be different"));
    assert!(!alias.exists());
    let too_small = dir.0.join("small");
    assert!(!run(&[
        "train",
        input.to_str().unwrap(),
        too_small.to_str().unwrap(),
        "--memory",
        "1MB"
    ])
    .status
    .success());
    assert!(!too_small.exists());
    let large = dir.0.join("large");
    fs::write(&large, vec![b'\n'; 20000000]).unwrap();
    let output = dir.0.join("exhausted.brp");
    let exhausted = run(&[
        "train",
        large.to_str().unwrap(),
        output.to_str().unwrap(),
        "--memory",
        "64MB",
    ]);
    assert!(!exhausted.status.success());
    assert!(String::from_utf8_lossy(&exhausted.stderr).contains("memory limit"));
    assert!(!output.exists());
}
