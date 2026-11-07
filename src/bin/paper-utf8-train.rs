//! Isolated UTF-8 scoring experiment; production CLI/defaults stay unchanged.
use bre_pair::{archive_io, engine::Config, field_engine, memory, scoring::Model};
use std::{
    env,
    fs::File,
    io::{BufWriter, Write},
    path::Path,
    process::ExitCode,
    time::Instant,
};

#[global_allocator]
static ALLOCATOR: memory::Allocator = memory::Allocator;

fn run() -> Result<(), String> {
    let args: Vec<_> = env::args().skip(1).collect();
    if args.len() != 7 {
        return Err(
            "paper-utf8-train RAW ARCHIVE REPORT BONUS WORKERS MINIMUM MEMORY_BYTES".into(),
        );
    }
    let bonus: f64 = args[3].parse().map_err(|_| "invalid UTF-8 bonus")?;
    let workers: usize = args[4].parse().map_err(|_| "invalid workers")?;
    let minimum: u64 = args[5].parse().map_err(|_| "invalid minimum")?;
    let total = args[6].parse().map_err(|_| "invalid memory budget")?;
    let stop = minimum.checked_mul(2).ok_or("stop threshold overflow")?;
    let limit = memory::Limit::set(total, workers)?;
    for path in [&args[1], &args[2]] {
        if Path::new(path).exists() || path == &args[0] {
            return Err("output exists or equals input".into());
        }
    }
    if args[1] == args[2] {
        return Err("archive and report must differ".into());
    }
    let config = Config {
        buffer_bytes: limit.heap,
        minimum_count: minimum,
        score: Model::WeightedPmi,
        search: bre_pair::search::Options {
            certify: false,
            fill_waves: 2,
            fill_min_count: minimum,
            ..Default::default()
        },
        ..Default::default()
    };
    let started = Instant::now();
    let _progress = bre_pair::training_progress::enable();
    let mut created = 0;
    let mut selection = 0.0;
    let mut rewrite = 0.0;
    let result = field_engine::compress_file_utf8_prior(
        Path::new(&args[0]),
        &config,
        workers,
        1_048_576,
        field_engine::Options {
            stop_max_pair_count: Some(stop),
            frequency_cutoff: true,
            memory_saving: true,
            ..Default::default()
        },
        bonus,
        |round| {
            created += round.new_rules;
            selection += round.selection_seconds;
            rewrite += round.rewrite_seconds;
            bre_pair::training_progress::completed(round, created);
            eprintln!(
                "round={} rules={} selected={} seconds={:.3}",
                round.index, round.total_rules, round.selected, round.seconds
            );
        },
    )?;
    let training_seconds = started.elapsed().as_secs_f64();
    bre_pair::vocabulary::verify_training_source(&result.training.compressed, Path::new(&args[0]))?;
    let mut writer = BufWriter::new(File::create_new(&args[1]).map_err(|e| e.to_string())?);
    archive_io::write(&result.training.compressed, &mut writer)?;
    writer.flush().map_err(|e| e.to_string())?;
    let report = format!(concat!("{{\"utf8_bonus\":{},\"minimum_count\":{},\"stop_max_pair_count\":{},",
        "\"workers\":{},\"fill_waves\":2,\"input_bytes\":{},\"fields\":{},\"rules\":{},\"symbols\":{},",
        "\"rounds\":{},\"training_seconds\":{:.6},\"total_seconds\":{:.6},",
        "\"selection_seconds\":{:.6},\"rewrite_seconds\":{:.6},\"peak_heap_bytes\":{},",
        "\"stopped_at_pair_count\":{},\"valid_utf8_source\":true,\"round_trip_verified\":true}}\n"),
        bonus, minimum, stop, workers, result.training.compressed.input_bytes,
        result.training.compressed.field_count, result.training.compressed.grammar.rules.len(),
        result.training.compressed.token_count(), result.training.rounds.len(), training_seconds,
        started.elapsed().as_secs_f64(), selection, rewrite, memory::peak_heap_bytes(),
        result.stopped_at_pair_count);
    File::create_new(&args[2])
        .and_then(|mut f| f.write_all(report.as_bytes()))
        .map_err(|e| e.to_string())?;
    print!("{report}");
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bre_pair::archive;

    #[test]
    fn zero_bonus_is_identical_and_positive_bonus_remains_byte_exact() {
        let path = env::temp_dir().join(format!("bre-utf8-prior-test-{}.txt", std::process::id()));
        let bytes = "中国中文 İstanbullu café 😀\nEnglish words repeated.\n"
            .repeat(100)
            .into_bytes();
        std::fs::write(&path, &bytes).unwrap();
        let config = Config {
            minimum_count: 4,
            score: Model::WeightedPmi,
            buffer_bytes: 1 << 22,
            ..Default::default()
        };
        let options = field_engine::Options {
            frequency_cutoff: true,
            memory_saving: true,
            ..Default::default()
        };
        let ordinary =
            field_engine::compress_file(&path, &config, 2, 1024, options, |_| {}).unwrap();
        let zero =
            field_engine::compress_file_utf8_prior(&path, &config, 2, 1024, options, 0.0, |_| {})
                .unwrap();
        assert_eq!(
            archive::encode(&ordinary.training.compressed).unwrap(),
            archive::encode(&zero.training.compressed).unwrap()
        );
        for bonus in [0.25, 0.5, 1.0] {
            let trained = field_engine::compress_file_utf8_prior(
                &path,
                &config,
                2,
                1024,
                options,
                bonus,
                |_| {},
            )
            .unwrap();
            assert_eq!(
                trained.training.compressed.decode().unwrap().concat(),
                bytes
            );
        }
        std::fs::write(&path, [0xe4, b'\n', 0xad]).unwrap();
        assert!(field_engine::compress_file_utf8_prior(
            &path,
            &config,
            2,
            1024,
            options,
            0.0,
            |_| {}
        )
        .is_err());
        std::fs::remove_file(path).unwrap();
    }
}
