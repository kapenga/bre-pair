//! Offline context-entropy signal experiment; standard BRPAIR01 output.
use bre_pair::{
    archive,
    engine::Compressed,
    entropy_engine,
    packed::{symbol_width, PackedSymbols},
};
use std::{
    env, fs,
    io::Write,
    path::{Path, PathBuf},
    process::ExitCode,
};

const HELP: &str = "Offline boundary-entropy Re-Pair experiment\n\
entropy-repair INPUT OUTPUT.brp [--order 3] [--min-context-count 32]\n\
  [--max-ngrams 20000000] [--min-count 8] [--threads 4] [--fields lines|single]\n\
  [--buffer-bytes 134217728] [--chunk-symbols 65536] [--max-rules N]\n\
  [--max-rounds N] [--report FILE.json] [--rounds FILE.csv] [--fast] [--verbose]\n\
Statistics use the entire training file before any float32 boundary score is\n\
assigned. Contexts reset at EOF. High-order contexts back off below minimum\n\
support. The predictor is discarded after annotation and is not a decoder.\n";

fn write(path: &Path, bytes: &[u8]) -> Result<(), String> {
    fs::File::create_new(path)
        .and_then(|mut file| file.write_all(bytes))
        .map_err(|e| format!("{}: {e}", path.display()))
}

fn run() -> Result<(), String> {
    let args: Vec<_> = env::args_os().skip(1).collect();
    if args.is_empty() || args[0] == "--help" {
        print!("{HELP}");
        return Ok(());
    }
    if args.len() < 2 {
        return Err(HELP.into());
    }
    let input = PathBuf::from(&args[0]);
    let output = PathBuf::from(&args[1]);
    let mut options = entropy_engine::Options::default();
    let mut policy = "lines";
    let (mut fast, mut verbose) = (false, false);
    let (mut report, mut rounds) = (None, None);
    let mut at = 2;
    while at < args.len() {
        let flag = args[at].to_str().ok_or("invalid option")?;
        if flag == "--fast" {
            fast = true;
            at += 1;
            continue;
        }
        if flag == "--verbose" {
            verbose = true;
            at += 1;
            continue;
        }
        let value = args
            .get(at + 1)
            .and_then(|v| v.to_str())
            .ok_or("missing option value")?;
        match flag {
            "--fields" => policy = value,
            "--order" => {
                options.context.order = value.parse().map_err(|_| "invalid context order")?
            }
            "--min-context-count" => {
                options.context.minimum_context_count =
                    value.parse().map_err(|_| "invalid context minimum")?
            }
            "--max-ngrams" => {
                options.context.maximum_ngrams =
                    value.parse().map_err(|_| "invalid n-gram limit")?
            }
            "--min-count" => {
                options.minimum_count = value.parse().map_err(|_| "invalid minimum")?
            }
            "--threads" => options.workers = value.parse().map_err(|_| "invalid workers")?,
            "--buffer-bytes" => {
                options.buffer_bytes = value.parse().map_err(|_| "invalid packed budget")?
            }
            "--chunk-symbols" => {
                options.chunk_symbols = value.parse().map_err(|_| "invalid chunk size")?
            }
            "--max-rules" => {
                options.max_rules = Some(value.parse().map_err(|_| "invalid rule limit")?)
            }
            "--max-rounds" => {
                options.max_rounds = Some(value.parse().map_err(|_| "invalid round limit")?)
            }
            "--report" => report = Some(PathBuf::from(value)),
            "--rounds" => rounds = Some(PathBuf::from(value)),
            _ => return Err(format!("unknown option {flag}")),
        }
        at += 2;
    }
    if !["lines", "single"].contains(&policy) {
        return Err("fields must be lines or single".into());
    }
    let mut destinations = vec![output.clone()];
    destinations.extend(report.iter().cloned());
    destinations.extend(rounds.iter().cloned());
    for (i, path) in destinations.iter().enumerate() {
        if path.exists() || path == &input || destinations[..i].contains(path) {
            return Err(format!(
                "destination exists or duplicated: {}",
                path.display()
            ));
        }
    }
    let raw = fs::read(&input).map_err(|e| e.to_string())?;
    let fields: Vec<_> = if policy == "single" || raw.is_empty() {
        vec![raw]
    } else {
        let fields = raw
            .split_inclusive(|&b| b == b'\n')
            .map(<[u8]>::to_vec)
            .collect();
        drop(raw);
        fields
    };
    let mut trace = String::from("round,before_tokens,rewritten_tokens,selected,new_rules,candidates,largest_component,score_statistics_seconds,selection_seconds,rewrite_and_count_seconds,search_seconds\n");
    let result = entropy_engine::compress(&fields, &options, |r| {
        trace.push_str(&format!(
            "{},{},{},{},{},{},{},{:.6},{:.6},{:.6},{:.6}\n",
            r.index,
            r.before,
            r.rewritten,
            r.selected,
            r.new_rules,
            r.candidates,
            r.largest_component,
            r.score_seconds,
            r.selection_seconds,
            r.rewrite_seconds,
            r.search_seconds
        ));
        if verbose {
            println!(
                "{{\"event\":\"batch\",\"round\":{},\"rules\":{},\"tokens\":{},\"seconds\":{:.6}}}",
                r.index, r.total_rules, r.rewritten, r.seconds
            );
        }
    })?;
    let training = &result.training;
    let source = &training.compressed;
    let mut block = PackedSymbols::with_capacity(
        symbol_width(source.grammar.alphabet_size()),
        source.token_count(),
    );
    for part in &source.blocks {
        for symbol in part.iter() {
            block.push(symbol);
        }
    }
    let canonical = Compressed {
        grammar: source.grammar.clone(),
        blocks: vec![block],
        input_bytes: source.input_bytes,
        field_count: source.field_count,
    };
    let encoded = archive::encode(&canonical)?;
    if !fast && archive::decode(&encoded)?.decode()? != fields {
        return Err("entropy archive round trip failed".into());
    }
    write(&output, &encoded)?;
    if let Some(path) = rounds {
        write(&path, trace.as_bytes())?;
    }
    let selection_seconds = training
        .rounds
        .iter()
        .map(|r| r.selection_seconds)
        .sum::<f64>();
    let rewrite_seconds = training
        .rounds
        .iter()
        .map(|r| r.rewrite_seconds)
        .sum::<f64>();
    let json = format!(concat!("{{\n",
        "  \"score_model\": \"context-entropy\", \"statistics_scope\": \"full-training-file-frozen\",\n",
        "  \"weight_type\": \"float32\", \"workers\": {}, \"fields_policy\": \"{}\",\n",
        "  \"input_bytes\": {}, \"fields\": {}, \"minimum_count\": {},\n",
        "  \"context_order\": {}, \"minimum_context_count\": {}, \"observed_ngrams\": {}, \"retained_contexts\": {},\n",
        "  \"contexts_by_order\": {:?}, \"backoff_by_order\": {:?},\n",
        "  \"statistics_seconds\": {:.6}, \"annotation_seconds\": {:.6}, \"discovery_seconds\": {:.6}, \"training_seconds\": {:.6},\n",
        "  \"selection_seconds\": {:.6}, \"rewrite_and_count_seconds\": {:.6},\n",
        "  \"initial_weight_bytes\": {}, \"final_weight_bytes\": {}, \"initial_weight_capacity_bytes\": {}, \"final_weight_capacity_bytes\": {},\n",
        "  \"mean_initial_weight\": {:.9}, \"zero_weights\": {}, \"peak_pair_entries\": {},\n",
        "  \"rules\": {}, \"tokens\": {}, \"rounds\": {}, \"archive_bytes\": {}, \"bits_per_symbol\": {},\n",
        "  \"round_trip_verified\": {}, \"stopped_at_rule_limit\": {}, \"stopped_at_round_limit\": {}\n}}\n"),
        options.workers,policy,source.input_bytes,source.field_count,options.minimum_count,
        options.context.order,options.context.minimum_context_count,result.context_statistics.observed_ngrams,result.context_statistics.retained_contexts,
        result.context_statistics.contexts_by_order,result.backoff_by_order,
        result.statistics_seconds,result.annotation_seconds,result.discovery_seconds,training.seconds,
        selection_seconds,rewrite_seconds,result.initial_weight_bytes,result.final_weight_bytes,result.initial_weight_capacity_bytes,result.final_weight_capacity_bytes,
        result.mean_initial_weight,result.zero_weights,result.peak_pair_entries,
        source.grammar.rules.len(),source.token_count(),training.rounds.len(),encoded.len(),symbol_width(source.grammar.alphabet_size()),
        !fast,result.stopped_at_rule_limit,result.stopped_at_round_limit);
    if let Some(path) = report {
        write(&path, json.as_bytes())?;
    }
    print!("{json}");
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
