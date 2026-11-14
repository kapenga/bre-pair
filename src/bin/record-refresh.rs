//! Training-only adapters for a refreshed, fixed-build record benchmark.
//! Direct decoding uses record-decode from the same frozen library build.
use bre_pair::{
    codec::{Frozen, MatchScratch},
    huffman_dictionary::byte_model,
    record_frame::Format,
    record_refit,
    shared_huffman::{RuleLengths, StaticModel},
};
use std::{
    fs,
    io::{BufWriter, Write},
    path::Path,
    time::Instant,
};

fn encode(model_path: &Path, input: &Path, folder: &Path) -> Result<(), String> {
    fs::create_dir(folder).map_err(|e| e.to_string())?;
    let stored = fs::read(model_path).map_err(|e| e.to_string())?;
    let model = StaticModel::from_bytes(&stored)?;
    let matcher = model.prepare()?;
    let encoder = model.codebook().encoding();
    fs::write(folder.join("model.bin"), &stored).map_err(|e| e.to_string())?;
    let raw = fs::read(input).map_err(|e| e.to_string())?;
    if raw.is_empty() || raw.last() != Some(&b'\n') {
        return Err("benchmark requires LF-retaining fields".into());
    }
    let mut payload =
        BufWriter::new(fs::File::create(folder.join("payload.bin")).map_err(|e| e.to_string())?);
    let header = Format::EofPlain.file_header();
    payload.write_all(&header).map_err(|e| e.to_string())?;
    let mut index = vec![header.len() as u64];
    let mut terminals = Vec::new();
    let mut scratch = MatchScratch::default();
    for field in raw.split_inclusive(|&byte| byte == b'\n') {
        if field.len() > 1 << 20 {
            return Err("record exceeds declared 1 MiB bound".into());
        }
        terminals.clear();
        terminals.extend(field.iter().map(|&byte| u32::from(byte)));
        terminals.push(256);
        let symbols = matcher.segment_with(&terminals, &mut scratch)?;
        let (bits, _) = encoder.encode(symbols)?;
        payload.write_all(&bits).map_err(|e| e.to_string())?;
        index.push(index.last().unwrap() + bits.len() as u64);
    }
    payload.flush().map_err(|e| e.to_string())?;
    drop(payload);
    let mut offsets =
        BufWriter::new(fs::File::create(folder.join("offsets.u64")).map_err(|e| e.to_string())?);
    for value in &index {
        offsets
            .write_all(&value.to_le_bytes())
            .map_err(|e| e.to_string())?;
    }
    offsets.flush().map_err(|e| e.to_string())?;
    let payload_bytes = fs::metadata(folder.join("payload.bin"))
        .map_err(|e| e.to_string())?
        .len();
    if payload_bytes != *index.last().unwrap() {
        return Err("stored payload size differs".into());
    }
    println!("{{\"rows\":{},\"raw_bytes\":{},\"model_bytes\":{},\"payload_bytes\":{payload_bytes},\"index_bytes\":{},\"complete_bytes\":{}}}",
        index.len()-1, raw.len(), stored.len(), index.len()*8, stored.len() as u64+payload_bytes+(index.len()*8) as u64);
    Ok(())
}

fn run() -> Result<(), String> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("byte-model") if args.len() == 2 => {
            if Path::new(&args[1]).exists() {
                return Err("output already exists".into());
            }
            fs::write(&args[1], byte_model().to_bytes()).map_err(|e| e.to_string())?;
        }
        Some("freeze") if args.len() == 5 => {
            let output = Path::new(&args[4]);
            if output.exists() {
                return Err("output already exists".into());
            }
            let source = Frozen::from_bytes(&fs::read(&args[1]).map_err(|e| e.to_string())?)?;
            let refits: usize = args[3].parse().map_err(|_| "invalid refit count")?;
            let start = Instant::now();
            let fitted = record_refit::training(&source, Path::new(&args[2]), refits, true)?;
            let bytes = StaticModel::encode_frozen(&fitted.model, RuleLengths::Auto)?;
            let restored = StaticModel::from_bytes(&bytes)?;
            if restored.training.bytes != fitted.model.training.bytes {
                return Err("serialized training identity differs".into());
            }
            fs::write(output, &bytes).map_err(|e| e.to_string())?;
            println!("{{\"training_only\":true,\"refits\":{refits},\"model_bytes\":{},\"training_bytes\":{},\"fit_points\":{},\"seconds\":{:.6}}}",
                bytes.len(), fitted.model.training.bytes, fitted.points.len(), start.elapsed().as_secs_f64());
        }
        Some("encode") if args.len() == 4 => encode(Path::new(&args[1]),Path::new(&args[2]),Path::new(&args[3]))?,
        _ => {
            return Err(
                "record-refresh byte-model NEW.brt | freeze MODEL.brc TRAIN REFITS NEW.brhm | encode MODEL.brhm RAW NEW_DIRECTORY".into(),
            )
        }
    }
    Ok(())
}

fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            std::process::ExitCode::FAILURE
        }
    }
}
