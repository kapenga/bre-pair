//! Export bounded training-only calibration data for joint vocabulary search.
use bre_pair::{archive, joint, tokenizer::Model, vocabulary};
use std::{env, fs, path::Path, process::ExitCode};

fn run() -> Result<(), String> {
    let args: Vec<_> = env::args().skip(1).collect();
    let help = "prepare-joint ARCHIVE TRAIN INITIAL.brt DIRECTORY [--sample-bytes 262144] [--max-edges 5000000] [--threads 4] [--min-token-uses 8] [--max-token-bytes 128]";
    if args.first().is_some_and(|x| x == "--help") {
        println!("{help}");
        return Ok(());
    }
    if args.len() < 4 || (args.len() - 4) % 2 != 0 {
        return Err(help.into());
    }
    let mut options = joint::Options {
        sample_bytes: 262144,
        maximum_edges: 5_000_000,
        workers: 4,
        minimum_uses: 8,
    };
    let mut max_bytes = 128;
    for pair in args[4..].chunks_exact(2) {
        match pair[0].as_str() {
            "--sample-bytes" => {
                options.sample_bytes = pair[1].parse().map_err(|_| "invalid sample bytes")?
            }
            "--max-edges" => {
                options.maximum_edges = pair[1].parse().map_err(|_| "invalid edge budget")?
            }
            "--threads" => options.workers = pair[1].parse().map_err(|_| "invalid workers")?,
            "--min-token-uses" => {
                options.minimum_uses = pair[1].parse().map_err(|_| "invalid minimum usage")?
            }
            "--max-token-bytes" => {
                max_bytes = pair[1].parse().map_err(|_| "invalid max token bytes")?
            }
            _ => return Err(format!("unknown option {}", pair[0])),
        }
    }
    if Path::new(&args[3]).exists() {
        return Err("output directory exists".into());
    }
    let source = archive::decode(&fs::read(&args[0]).map_err(|e| e.to_string())?)?;
    vocabulary::verify_training_source(&source, Path::new(&args[1]))?;
    let (pool, _) = Model::candidates(&source, max_bytes)?;
    drop(source);
    let initial = Model::from_bytes(&fs::read(&args[2]).map_err(|e| e.to_string())?)?;
    joint::prepare(
        pool,
        &initial,
        Path::new(&args[1]),
        Path::new(&args[3]),
        &options,
    )
}
fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
