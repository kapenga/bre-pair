//! Independent sequential reference; the default bre-pair command stays batched.
use std::env;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use bre_pair::{archive, maximal, sequence_archive};

const HELP: &str = "Whole-file MR-RePair reference experiment (not memory bounded)\n\n\
Usage:\n  mr-repair compress INPUT OUTPUT.mrp [--mode maximal|pair] [--min-count N]\n\
                     [--report FILE.json] [--rounds FILE.csv] [--verbose]\n\
  mr-repair decompress INPUT.mrp OUTPUT\n\n\
Defaults: maximal repeats, minimum 4 actual replacements.\n\
Priority uses overlapping occurrence frequency, following the paper.\n\
One input field; EOF cannot occur inside a rule. No batching, refill or Huffman.\n";

fn write_new(path: &Path, bytes: &[u8]) -> Result<(), String> {
    fs::File::create_new(path)
        .and_then(|mut file| file.write_all(bytes))
        .map_err(|e| format!("write {}: {e}", path.display()))
}

fn run() -> Result<(), String> {
    let args: Vec<_> = env::args_os().skip(1).collect();
    if args.is_empty() || args[0] == "--help" || args[0] == "-h" {
        print!("{HELP}");
        return Ok(());
    }
    if args.len() < 3 {
        return Err(HELP.into());
    }
    let input = PathBuf::from(&args[1]);
    let output = PathBuf::from(&args[2]);
    if input == output || output.exists() {
        return Err("output must be a new file, different from input".into());
    }
    if args[0] == "decompress" {
        if args.len() != 3 {
            return Err("decompress expects two paths".into());
        }
        let bytes = fs::read(&input).map_err(|e| e.to_string())?;
        let native = sequence_archive::decode(&bytes)?;
        let fields = native.to_binary()?.decode()?;
        let mut writer = fs::File::create_new(&output).map_err(|e| e.to_string())?;
        for field in fields {
            writer.write_all(&field).map_err(|e| e.to_string())?;
        }
        println!(
            "Decoded {} bytes in {} field(s).",
            native.input_bytes, native.field_count
        );
        return Ok(());
    }
    if args[0] != "compress" {
        return Err(HELP.into());
    }
    let mut minimum = 4;
    let mut use_maximal = true;
    let mut report_path = None;
    let mut rounds_path = None;
    let mut verbose = false;
    let mut at = 3;
    while at < args.len() {
        let flag = args[at].to_str().ok_or("option must be UTF-8")?;
        if flag == "--verbose" {
            verbose = true;
            at += 1;
            continue;
        }
        let value = args
            .get(at + 1)
            .ok_or_else(|| format!("missing value for {flag}"))?;
        match flag {
            "--min-count" => {
                minimum = value
                    .to_str()
                    .ok_or("invalid count")?
                    .parse()
                    .map_err(|_| "invalid count")?
            }
            "--mode" => match value.to_str() {
                Some("maximal") => use_maximal = true,
                Some("pair") => use_maximal = false,
                _ => return Err("mode must be maximal or pair".into()),
            },
            "--report" => report_path = Some(PathBuf::from(value)),
            "--rounds" => rounds_path = Some(PathBuf::from(value)),
            _ => return Err(format!("unknown option: {flag}")),
        }
        at += 2;
    }
    let mut destinations = vec![output.clone()];
    destinations.extend(report_path.iter().cloned());
    destinations.extend(rounds_path.iter().cloned());
    for (i, path) in destinations.iter().enumerate() {
        if path.exists() || path == &input || destinations[..i].contains(path) {
            return Err(format!(
                "destination exists or is duplicated: {}",
                path.display()
            ));
        }
        let parent = path.parent().unwrap_or(Path::new("."));
        if !parent.as_os_str().is_empty() && !parent.is_dir() {
            return Err(format!(
                "destination directory does not exist: {}",
                parent.display()
            ));
        }
    }
    let fields = vec![fs::read(&input).map_err(|e| format!("read {}: {e}", input.display()))?];
    eprintln!(
        "Sequential {} reference on {} bytes; minimum {} replacements.",
        if use_maximal {
            "maximal-repeat"
        } else {
            "pair"
        },
        fields[0].len(),
        minimum
    );
    let training = maximal::compress(&fields, minimum, use_maximal, |round| {
        if verbose || round.index % 1000 == 0 {
            eprintln!(
                "rule {:5}: length {}, {} replacements, {} tokens",
                round.index, round.rhs_length, round.replacements, round.after
            );
        }
    })?;
    let compressed = &training.compressed;
    let bytes = sequence_archive::encode(compressed)?;
    if sequence_archive::decode(&bytes)?.to_binary()?.decode()? != fields {
        return Err("native archive round-trip verification failed".into());
    }
    let binary = compressed.to_binary()?;
    let binary_archive_bytes = archive::encode(&binary)?.len();
    let dictionary_bits = sequence_archive::dictionary_bits(compressed)?;
    let rhs_symbols: usize = compressed.rules.iter().map(Vec::len).sum();
    let extended_rules = compressed.rules.iter().filter(|r| r.len() > 2).count();
    let trimmed_rules = training.rounds.iter().filter(|r| r.trimmed).count();
    let mode = if use_maximal { "maximal" } else { "pair" };
    write_new(&output, &bytes)?;
    if let Some(path) = report_path {
        let report = format!(
            concat!(
                "{{\n  \"format\": \"BRPNAR01\",\n  \"mode\": \"{}\",\n",
                "  \"frequency\": \"overlapping-occurrences\",\n",
                "  \"minimum_actual_replacements\": {},\n",
                "  \"input_bytes\": {},\n  \"fields\": {},\n",
                "  \"rules\": {},\n  \"extended_rules\": {},\n  \"trimmed_rules\": {},\n",
                "  \"rule_rhs_symbols\": {},\n  \"grammar_symbols_including_start\": {},\n",
                "  \"maximum_rhs_length\": {},\n  \"tokens\": {},\n",
                "  \"dictionary_bits\": {},\n  \"dictionary_bytes\": {},\n",
                "  \"fixed_width_data_bytes\": {},\n  \"archive_bytes\": {},\n",
                "  \"binary_equivalent_rules\": {},\n  \"binary_equivalent_archive_bytes\": {},\n",
                "  \"training_seconds\": {:.6},\n",
                "  \"round_trip_verified\": true,\n  \"huffman_coded\": false,\n",
                "  \"batched\": false,\n  \"memory_bounded\": false\n}}\n"
            ),
            mode,
            minimum,
            compressed.input_bytes,
            compressed.field_count,
            compressed.rules.len(),
            extended_rules,
            trimmed_rules,
            rhs_symbols,
            rhs_symbols + compressed.token_count(),
            compressed.rules.iter().map(Vec::len).max().unwrap_or(0),
            compressed.token_count(),
            dictionary_bits,
            dictionary_bits.div_ceil(8),
            compressed.data_bytes(),
            bytes.len(),
            binary.grammar.rules.len(),
            binary_archive_bytes,
            training.seconds
        );
        write_new(&path, report.as_bytes())?;
    }
    if let Some(path) = rounds_path {
        let mut csv = String::from("round,seed_left,seed_right,seed_frequency,rhs_length,replacements,trimmed,before_tokens,after_tokens\n");
        for r in &training.rounds {
            csv.push_str(&format!(
                "{},{},{},{},{},{},{},{},{}\n",
                r.index,
                r.seed.0,
                r.seed.1,
                r.seed_frequency,
                r.rhs_length,
                r.replacements,
                r.trimmed,
                r.before,
                r.after
            ));
        }
        write_new(&path, csv.as_bytes())?;
    }
    println!(
        "Mode: {mode}; native rules: {}; extended: {extended_rules}; tokens: {}",
        compressed.rules.len(),
        compressed.token_count()
    );
    println!(
        "Grammar RHS symbols including start: {}",
        rhs_symbols + compressed.token_count()
    );
    println!(
        "Native archive: {} bytes; binary equivalent: {binary_archive_bytes} bytes",
        bytes.len()
    );
    println!(
        "Training: {:.3}s; serialized round trip verified byte-for-byte",
        training.seconds
    );
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}
