//! Bounded training-only research harness; production codec defaults are unchanged.
use bre_pair::{
    codec::{
        refinement::{read_calibration, Options},
        Frozen,
    },
    memory,
};
use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    path::Path,
    time::Instant,
};

#[global_allocator]
static ALLOCATOR: memory::Allocator = memory::Allocator;

fn run() -> Result<(), String> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() < 3 {
        return Err("codec-refine MODEL.brc TRAINING.brh NEW_DIRECTORY [--trials N] [--refits N] [--calibration-mb N]".into());
    }
    let mut options = Options::default();
    let mut budget = 128_000_000u64;
    let mut at = 3;
    while at < args.len() {
        let value = args.get(at + 1).ok_or("missing option value")?;
        match args[at].as_str() {
            "--trials" => options.max_trials = value.parse().map_err(|_| "invalid trial count")?,
            "--refits" => options.refits = value.parse().map_err(|_| "invalid refit count")?,
            "--calibration-mb" => {
                budget = value
                    .parse::<u64>()
                    .map_err(|_| "invalid calibration size")?
                    .checked_mul(1_000_000)
                    .ok_or("calibration size overflow")?
            }
            _ => return Err("unknown refinement option".into()),
        }
        at += 2;
    }
    let _limit = memory::Limit::set(2_000_000_000, 1)?;
    let start = Instant::now();
    let model = Frozen::from_bytes(&fs::read(&args[0]).map_err(|e| e.to_string())?)?;
    let calibration = read_calibration(&model, Path::new(&args[1]), budget)?;
    let output = Path::new(&args[2]);
    fs::create_dir(output).map_err(|e| e.to_string())?;
    let mut trials =
        BufWriter::new(File::create_new(output.join("trials.csv")).map_err(|e| e.to_string())?);
    writeln!(trials, "trial,removed_id,zero_group,accepted,parses,payload_bits,metadata_bytes,boundary_gain_bits").map_err(|e| e.to_string())?;
    let mut trial_rows = Vec::new();
    let fit_start = Instant::now();
    let (optimized, report) = model.refine_codes(&calibration, options, |trial| {
        eprintln!(
            "trial {}: accepted={}, {} bits, metadata {}, boundary gain {} bits",
            trial.number,
            trial.accepted,
            trial.payload_bits,
            trial.metadata_bytes,
            trial.boundary_gain_bits
        );
        trial_rows.push(format!(
            "{},{},{},{},{},{},{},{}\n",
            trial.number,
            trial.removed_id.map_or_else(String::new, |n| n.to_string()),
            trial.zero_group,
            trial.accepted,
            trial.parses,
            trial.payload_bits,
            trial.metadata_bytes,
            trial.boundary_gain_bits
        ));
    })?;
    let fitting_seconds = fit_start.elapsed().as_secs_f64();
    for row in trial_rows {
        trials
            .write_all(row.as_bytes())
            .map_err(|e| e.to_string())?;
    }
    trials.flush().map_err(|e| e.to_string())?;
    fs::write(output.join("model.brc"), optimized.to_bytes()?).map_err(|e| e.to_string())?;
    let totals = calibration.write_archive(&optimized, &output.join("training.brh"))?;
    let json = format!(concat!("{{\n",
        "  \"training_only\":true,\n  \"complete_training_calibration\":true,\n  \"AI_IDs_preserved\":true,\n",
        "  \"training_bytes\":{},\n  \"training_fields\":{},\n  \"training_AI_tokens\":{},\n",
        "  \"calibration_storage_bytes\":{},\n  \"max_trials\":{},\n  \"refits\":{},\n",
        "  \"trials\":{},\n  \"accepted\":{},\n  \"parses\":{},\n",
        "  \"before_payload_bits\":{},\n  \"after_payload_bits\":{},\n",
        "  \"before_metadata_bytes\":{},\n  \"after_metadata_bytes\":{},\n",
        "  \"before_codes\":{},\n  \"after_codes\":{},\n  \"before_rules\":{},\n  \"after_rules\":{},\n",
        "  \"removed_zero_codes\":{},\n  \"removed_positive_codes\":{},\n",
        "  \"training_archive_bytes\":{},\n  \"training_codec_symbols\":{},\n",
        "  \"fitting_seconds\":{:.6},\n  \"seconds\":{:.6},\n  \"peak_heap_bytes\":{},\n  \"peak_private_bytes\":{}\n}}\n"),
        calibration.totals.bytes, calibration.totals.fields, calibration.totals.tokens,
        calibration.storage_bytes(), options.max_trials, options.refits, report.trials,
        report.accepted, report.parses, report.before_payload_bits, report.after_payload_bits,
        report.before_metadata_bytes, report.after_metadata_bytes, model.book.symbols.len(),
        optimized.book.symbols.len(), model.dictionary.rules.len(), optimized.dictionary.rules.len(),
        report.removed_zero_codes, report.removed_positive_codes,
        fs::metadata(output.join("training.brh")).map_err(|e| e.to_string())?.len(), totals.symbols,
        fitting_seconds, start.elapsed().as_secs_f64(), memory::peak_heap_bytes(), memory::process().peak_private);
    fs::write(output.join("report.json"), &json).map_err(|e| e.to_string())?;
    print!("{json}");
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("codec-refine: {e}");
        std::process::exit(1);
    }
}
