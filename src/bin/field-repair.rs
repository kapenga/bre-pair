//! Controlled EOF-field concurrency experiments; no per-worker pair tables.
use bre_pair::{
    archive,
    engine::{self, Config},
    field_engine,
    packed::{symbol_width, PackedSymbols},
    registry::Registry,
    scoring::Model,
};
use std::{env, fs, io::Write, path::PathBuf, process::ExitCode, time::Instant};

const HELP: &str = "Shared-registry field experiment\n\
field-repair INPUT OUTPUT.brp [--threads N] [--fields lines|paragraphs|single]\n\
  [--reference] [--shards N] [--chunk-symbols N] [--min-count N]\n\
  [--buffer-bytes N] [--score MODEL] [--pair-counting MODE] [--selector MODE]\n\
  [--fast] [--report FILE.json] [--rounds FILE.csv]\n\
  [--max-rounds N] [--max-rules N] [--train-only] [--verbose] [--frequency-cutoff]\n\
Defaults: 1 thread, lines, 64 shared shards, 65536-symbol chunks, minimum 8,\n\
  weighted PMI alpha 1, adaptive counting, greedy selection, 16 MiB packed budget.\n\
Fields retain their original delimiter bytes and each gains an EOF. Source and\n\
active chunks still reside in memory; this is not the TB streaming trainer.\n\
--reference uses the existing single-thread engine on identical fields.\n\
--frequency-cutoff enables experimental rare-endpoint filtering; default off.\n\
--max-rounds stops at a complete batch without certifying convergence.\n\
--max-rules stops at a hard learned-rule budget, using a partial final batch.\n\
--train-only skips archive export and verification; OUTPUT.brp is unused.\n\
--verbose emits JSON progress events suitable for memory/CPU sampling.\n";

fn fields(input: &[u8], policy: &str) -> Vec<Vec<u8>> {
    if policy == "single" || input.is_empty() {
        return vec![input.to_vec()];
    }
    if policy == "lines" {
        return input
            .split_inclusive(|&b| b == b'\n')
            .map(<[u8]>::to_vec)
            .collect();
    }
    let mut fields = Vec::new();
    let mut start = 0;
    let mut at = 0;
    while at < input.len() {
        let size = if input[at..].starts_with(b"\r\n\r\n") {
            4
        } else if input[at..].starts_with(b"\n\n") {
            2
        } else {
            0
        };
        if size > 0 {
            at += size;
            fields.push(input[start..at].to_vec());
            start = at;
        } else {
            at += 1;
        }
    }
    if start < input.len() {
        fields.push(input[start..].to_vec());
    }
    fields
}

fn write(path: &PathBuf, bytes: &[u8]) -> Result<(), String> {
    fs::File::create_new(path)
        .and_then(|mut f| f.write_all(bytes))
        .map_err(|e| format!("{}: {e}", path.display()))
}

fn run() -> Result<(), String> {
    let process_started = Instant::now();
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
    let mut config = Config {
        score: Model::WeightedPmi,
        minimum_count: 8,
        buffer_bytes: 16 * 1024 * 1024,
        ..Config::default()
    };
    let (mut workers, mut shards, mut chunk) = (1usize, 64usize, 65536usize);
    let mut policy = "lines";
    let (mut reference, mut fast) = (false, false);
    let mut frequency_cutoff = false;
    let (mut train_only, mut verbose, mut max_rounds) = (false, false, None);
    let mut max_rules = None;
    let (mut report, mut trace) = (None, None);
    let mut at = 2;
    while at < args.len() {
        let flag = args[at].to_str().ok_or("invalid option")?;
        match flag {
            "--frequency-cutoff" => {
                frequency_cutoff = true;
                at += 1;
                continue;
            }
            "--reference" => {
                reference = true;
                at += 1;
                continue;
            }
            "--fast" => {
                fast = true;
                at += 1;
                continue;
            }
            "--train-only" => {
                train_only = true;
                at += 1;
                continue;
            }
            "--verbose" => {
                verbose = true;
                at += 1;
                continue;
            }
            _ => (),
        }
        let value = args
            .get(at + 1)
            .and_then(|v| v.to_str())
            .ok_or("missing or invalid option value")?;
        match flag {
            "--threads" => workers = value.parse().map_err(|_| "invalid worker count")?,
            "--shards" => shards = value.parse().map_err(|_| "invalid shard count")?,
            "--chunk-symbols" => chunk = value.parse().map_err(|_| "invalid chunk size")?,
            "--fields" => policy = value,
            "--min-count" => config.minimum_count = value.parse().map_err(|_| "invalid minimum")?,
            "--buffer-bytes" => {
                config.buffer_bytes = value.parse().map_err(|_| "invalid budget")?
            }
            "--score" => config.score = value.parse()?,
            "--pair-counting" => config.pair_counting = value.parse()?,
            "--selector" => config.search.strategy = value.parse()?,
            "--report" => report = Some(PathBuf::from(value)),
            "--rounds" => trace = Some(PathBuf::from(value)),
            "--max-rounds" => max_rounds = Some(value.parse().map_err(|_| "invalid round limit")?),
            "--max-rules" => max_rules = Some(value.parse().map_err(|_| "invalid rule limit")?),
            _ => return Err(format!("unknown option {flag}")),
        }
        at += 2;
    }
    if !["lines", "paragraphs", "single"].contains(&policy) || (reference && workers != 1) {
        return Err("invalid fields policy or reference worker count".into());
    }
    if max_rounds == Some(0)
        || max_rules == Some(0)
        || (reference && (max_rounds.is_some() || max_rules.is_some()))
    {
        return Err("round/rule limits must be positive and require the shared backend".into());
    }
    if reference && frequency_cutoff {
        return Err("frequency cutoff requires the shared backend".into());
    }
    config.search.certify = !fast;
    let mut destinations = if train_only {
        vec![]
    } else {
        vec![output.clone()]
    };
    #[cfg(feature = "profile")]
    let profile_output = env::var_os("BRE_PAIR_PROFILE").map(PathBuf::from);
    #[cfg(feature = "profile")]
    destinations.extend(profile_output.iter().cloned());
    destinations.extend(report.iter().cloned());
    destinations.extend(trace.iter().cloned());
    for (i, p) in destinations.iter().enumerate() {
        if p.exists() || p == &input || destinations[..i].contains(p) {
            return Err(format!("destination exists or duplicated: {}", p.display()));
        }
    }
    let source = fs::read(&input).map_err(|e| e.to_string())?;
    if verbose {
        println!(
            "{{\"event\":\"input_loaded\",\"elapsed_seconds\":{:.6},\"bytes\":{}}}",
            process_started.elapsed().as_secs_f64(),
            source.len()
        );
    }
    #[cfg(feature = "profile")]
    let split_profile = bre_pair::profile::Span::new("io.split_fields");
    let fields = fields(&source, policy);
    drop(source);
    if verbose {
        println!(
            "{{\"event\":\"training_start\",\"elapsed_seconds\":{:.6},\"fields\":{}}}",
            process_started.elapsed().as_secs_f64(),
            fields.len()
        );
    }
    #[cfg(feature = "profile")]
    drop(split_profile);
    let mut trace_csv=String::from("round,before_tokens,rewritten_tokens,selected,new_rules,candidates,largest_component,score_statistics_seconds,selection_seconds,rewrite_and_count_seconds,search_seconds\n");
    let mut callback = |r: &engine::Round| {
        if verbose {
            println!(concat!("{{\"event\":\"round\",\"elapsed_seconds\":{:.6},\"round\":{},\"seconds\":{:.6},",
                "\"before_tokens\":{},\"rewritten_tokens\":{},\"candidates\":{},\"selected\":{},\"rules\":{},\"width\":{},",
                "\"statistics_seconds\":{:.6},\"selection_seconds\":{:.6},\"rewrite_seconds\":{:.6}}}"),
                process_started.elapsed().as_secs_f64(), r.index, r.seconds, r.before, r.rewritten, r.candidates, r.selected, r.total_rules, r.width,
                r.score_seconds, r.selection_seconds, r.rewrite_seconds);
        }
        trace_csv.push_str(&format!(
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
    };
    let (
        mut training,
        slots,
        peak,
        growths,
        retired,
        compactions,
        symbol_bytes,
        initial,
        unchanged,
        chunks,
        limited,
        rule_limited,
    ) = if reference {
        (
            engine::compress(&fields, &config, &mut callback)?,
            0,
            0,
            0,
            0,
            0,
            0,
            0.0,
            0,
            1,
            false,
            false,
        )
    } else {
        let result = field_engine::compress_with_options(
            &fields,
            &config,
            workers,
            shards,
            chunk,
            field_engine::Options {
                max_rounds,
                max_rules,
                stop_max_pair_count: None,
                frequency_cutoff,
                memory_saving: false,
                last_seen_documents: false,
                shake: None,
            },
            &mut callback,
        )?;
        (
            result.training,
            result.registry_slot_bytes,
            result.registry_peak_slot_bytes,
            result.registry_growths,
            result.registry_retired_pairs,
            result.registry_compactions,
            result.symbol_statistics_bytes,
            result.initial_count_seconds,
            result.unchanged_chunks,
            result.chunks,
            result.stopped_at_round_limit,
            result.stopped_at_rule_limit,
        )
    };
    if verbose {
        println!("{{\"event\":\"training_end\",\"elapsed_seconds\":{:.6},\"training_seconds\":{:.6},\"stopped_at_round_limit\":{},\"stopped_at_rule_limit\":{}}}", process_started.elapsed().as_secs_f64(), training.seconds, limited, rule_limited);
    }
    // Canonicalize only for output comparison, outside timed training. Chunk
    // boundaries are physical layout choices, not vocabulary/field boundaries.
    let tokens = training.compressed.token_count();
    let encoded = if train_only {
        None
    } else {
        #[cfg(feature = "profile")]
        let canonical_profile = bre_pair::profile::Span::new("io.canonicalize");
        let width = symbol_width(training.compressed.grammar.alphabet_size());
        let mut packed = PackedSymbols::with_capacity(width, tokens);
        for block in &training.compressed.blocks {
            for symbol in block.iter() {
                packed.push(symbol);
            }
        }
        training.compressed.blocks = if tokens > 0 { vec![packed] } else { vec![] };
        #[cfg(feature = "profile")]
        drop(canonical_profile);
        #[cfg(feature = "profile")]
        let encode_profile = bre_pair::profile::Span::new("io.encode_archive");
        let encoded = archive::encode(&training.compressed)?;
        #[cfg(feature = "profile")]
        drop(encode_profile);
        if !fast && archive::decode(&encoded)?.decode()? != fields {
            return Err("field archive verification failed".into());
        }
        write(&output, &encoded)?;
        Some(encoded)
    };
    if let Some(path) = trace {
        write(&path, trace_csv.as_bytes())?;
    }
    if let Some(path) = report {
        let metrics=format!(concat!("{{\n\"backend\":\"{}\",\"workers\":{},\"fields_policy\":\"{}\",\"fields\":{},\n",
            "\"input_bytes\":{},\"minimum_count\":{},\"score_model\":\"{}\",\"pair_counting\":\"{}\",\"selector\":\"{}\",\n",
            "\"training_seconds\":{:.6},\"initial_count_seconds\":{:.6},\"score_statistics_seconds\":{:.6},\"selection_seconds\":{:.6},\"rewrite_and_count_seconds\":{:.6},\"search_seconds\":{:.6},\n",
            "\"registry_slot_bytes\":{},\"registry_peak_slot_bytes\":{},\"registry_growths\":{},\"worker_log_bytes_each\":{},\n",
            "\"registry_retired_pairs\":{},\"registry_compactions\":{},\"symbol_statistics_bytes\":{},\"frequency_cutoff\":{},\n",
            "\"chunks\":{},\"unchanged_chunk_events\":{},\"rounds\":{},\"largest_component\":{},\"rules\":{},\"tokens\":{},\"archive_bytes\":{},\"round_trip_verified\":{},\"archive_written\":{},\"stopped_at_round_limit\":{},\"stopped_at_rule_limit\":{},\"max_rules\":{}\n}}\n"),
            if reference{"reference"}else{"shared"},workers,policy,fields.len(),training.compressed.input_bytes,config.minimum_count,config.score.name(),config.pair_counting.name(),config.search.strategy.name(),
            training.seconds,initial,training.rounds.iter().map(|r|r.score_seconds).sum::<f64>(),training.rounds.iter().map(|r|r.selection_seconds).sum::<f64>(),training.rounds.iter().map(|r|r.rewrite_seconds).sum::<f64>(),training.rounds.iter().map(|r|r.search_seconds).sum::<f64>(),
            slots,peak,growths,if reference{0}else{Registry::worker_log_bytes()},retired,compactions,symbol_bytes,frequency_cutoff,chunks,unchanged,training.rounds.len(),training.rounds.iter().map(|r|r.largest_component).max().unwrap_or(0),training.compressed.grammar.rules.len(),tokens,encoded.as_ref().map_or(0,Vec::len),!fast&&!train_only,!train_only,limited,rule_limited,max_rules.map_or("null".into(), |n: usize| n.to_string()));
        write(&path, metrics.as_bytes())?;
    }
    println!(
        "{} workers, {} fields: {:.6}s, {} rules, {tokens} tokens, {} rounds",
        workers,
        fields.len(),
        training.seconds,
        training.compressed.grammar.rules.len(),
        training.rounds.len()
    );
    #[cfg(feature = "profile")]
    if let Some(path) = profile_output {
        bre_pair::profile::write(&path)?;
    }
    Ok(())
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
