use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use bre_pair::archive;
use bre_pair::engine::{self, Config, Round, Training};

const HELP: &str = "Batched Re-Pair prototype (no Huffman coding)\n\n\
Usage:\n  bre-pair compress INPUT OUTPUT [--min-count N] [--buffer-bytes N]\n\
                    [--report FILE.json] [--rounds FILE.csv] [--verbose]\n\
                    [--selector greedy|beam|hybrid] [--beam-width N] [--search-nodes N]\n\
                    [--tie-break lex|reverse|seeded] [--seed N]\n\
                    [--score frequency|conditional|pmi|weighted-pmi|likelihood]\n\
                    [--pmi-alpha N] [--pair-counting full|reuse|incremental|adaptive]\n\
                    [--fast]\n\
  bre-pair decompress INPUT.brp OUTPUT\n\n\
Defaults: --min-count 4 --buffer-bytes 1048576 --pair-counting adaptive\n\
The input file is one field. Every byte is preserved; one EOF is appended.\n\
The buffer setting limits the active packed sequence, not total process RAM.\n\
--fast skips archive verification and greedy optimality certificates.\n";

fn round_csv(rounds: &[Round]) -> String {
    let mut csv = String::from("round,candidates,eligible,tie_edges,selected,new_rules,before_tokens,rewritten_tokens,after_refill_tokens,removed_tokens,refill_bytes,total_rules,bits_per_symbol,seconds,components,largest_component,greedy_score,score_upper_bound,proven_optimal,search_nodes,search_seconds,improved_components,score_statistics_seconds,selection_seconds,rewrite_and_count_seconds,refill_seconds\n");
    for r in rounds {
        csv.push_str(&format!(
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{:.6},{},{},{},{},{},{},{:.6},{},{:.6},{:.6},{:.6},{:.6}\n",
            r.index,
            r.candidates,
            r.eligible,
            r.tie_edges,
            r.selected,
            r.new_rules,
            r.before,
            r.rewritten,
            r.after_refill,
            r.removed,
            r.refill_bytes,
            r.total_rules,
            r.width,
            r.seconds,
            r.components,
            r.largest_component,
            r.greedy_score,
            r.score_upper_bound,
            r.proven_optimal,
            r.search_nodes,
            r.search_seconds,
            r.improved_components, r.score_seconds, r.selection_seconds, r.rewrite_seconds, r.refill_seconds
        ));
    }
    csv
}

fn per_item(value: u128, count: usize) -> String {
    if count == 0 {
        "null".into()
    } else {
        format!("{:.6}", value as f64 / count as f64)
    }
}

fn report(training: &Training, config: &Config, archive_bytes: usize, fast: bool) -> String {
    let compressed = &training.compressed;
    let data = compressed.data_bytes();
    let dictionary = compressed.dictionary_bits().div_ceil(8);
    let rules = compressed.grammar.rules.len();
    let symbols = compressed.token_count();
    // Rule metadata already stores expansion lengths. No input or payload scan.
    let (expansion_sum, expansion_max) =
        compressed
            .grammar
            .rules
            .iter()
            .fold((0u128, 0usize), |(sum, maximum), rule| {
                let bytes = rule.terminals - usize::from(rule.ends_field);
                (sum + bytes as u128, maximum.max(bytes))
            });
    let ratio = if compressed.input_bytes == 0 {
        0.0
    } else {
        archive_bytes as f64 / compressed.input_bytes as f64
    };
    format!(
        concat!(
            "{{\n",
            "  \"format\": \"BRPAIR01\",\n",
            "  \"input_bytes\": {},\n",
            "  \"fields\": {},\n",
            "  \"minimum_count\": {},\n",
            "  \"fast_mode\": {},\n",
            "  \"optimality_certificates_enabled\": {},\n",
            "  \"score_model\": \"{}\",\n",
            "  \"pmi_alpha\": {},\n",
            "  \"pair_counting\": \"{}\",\n",
            "  \"selector\": \"{}\",\n",
            "  \"tie_break\": \"{}\",\n",
            "  \"seed\": {},\n",
            "  \"beam_width\": {},\n",
            "  \"search_node_budget_per_component\": {},\n",
            "  \"buffer_budget_bytes\": {},\n",
            "  \"peak_active_packed_bytes\": {},\n",
            "  \"rules\": {},\n",
            "  \"alphabet_size\": {},\n",
            "  \"rounds\": {},\n",
            "  \"blocks\": {},\n",
            "  \"tokens\": {},\n",
            "  \"input_bytes_per_symbol\": {},\n",
            "  \"data_bytes_per_symbol\": {},\n",
            "  \"dictionary_bytes_per_symbol\": {},\n",
            "  \"archive_bytes_per_symbol\": {},\n",
            "  \"final_bits_per_symbol\": {},\n",
            "  \"dictionary_bits\": {},\n",
            "  \"dictionary_bytes\": {},\n",
            "  \"dictionary_bytes_per_rule\": {},\n",
            "  \"input_bytes_per_rule\": {},\n",
            "  \"archive_bytes_per_rule\": {},\n",
            "  \"mean_rule_expansion_bytes\": {},\n",
            "  \"max_rule_expansion_bytes\": {},\n",
            "  \"fixed_width_data_bytes\": {},\n",
            "  \"container_overhead_bytes\": {},\n",
            "  \"archive_bytes\": {},\n",
            "  \"archive_to_input_ratio\": {:.6},\n",
            "  \"refill_calls\": {},\n",
            "  \"stalled_buffer_flushes\": {},\n",
            "  \"width_pressure_flushes\": {},\n",
            "  \"training_seconds\": {:.6},\n",
            "  \"score_statistics_seconds\": {:.6},\n",
            "  \"selection_seconds\": {:.6},\n",
            "  \"rewrite_and_count_seconds\": {:.6},\n",
            "  \"refill_seconds\": {:.6},\n",
            "  \"all_batches_proven_optimal\": {},\n",
            "  \"unproven_batches\": {},\n",
            "  \"search_seconds\": {:.6},\n",
            "  \"improved_components\": {},\n",
            "  \"same_snapshot_gain_over_greedy\": {},\n",
            "  \"round_trip_verified\": {},\n",
            "  \"huffman_coded\": false\n",
            "}}\n"
        ),
        compressed.input_bytes,
        compressed.field_count,
        config.minimum_count,
        fast,
        config.search.certify,
        config.score.name(),
        config.pmi_alpha,
        config.pair_counting.name(),
        config.search.strategy.name(),
        config.search.tie_break.name(),
        config.search.seed,
        config.search.beam_width,
        config.search.node_limit,
        config.buffer_bytes,
        training.max_buffer_bytes,
        compressed.grammar.rules.len(),
        compressed.grammar.alphabet_size(),
        training.rounds.len(),
        compressed.blocks.len(),
        symbols,
        per_item(compressed.input_bytes as u128, symbols),
        per_item(data as u128, symbols),
        per_item(dictionary as u128, symbols),
        per_item(archive_bytes as u128, symbols),
        bre_pair::packed::symbol_width(compressed.grammar.alphabet_size()),
        compressed.dictionary_bits(),
        dictionary,
        per_item(dictionary as u128, rules),
        per_item(compressed.input_bytes as u128, rules),
        per_item(archive_bytes as u128, rules),
        per_item(expansion_sum, rules),
        expansion_max,
        data,
        archive_bytes - dictionary - data,
        archive_bytes,
        ratio,
        training.refill_calls,
        training.stalled_flushes,
        training.width_flushes,
        training.seconds,
        training.rounds.iter().map(|r| r.score_seconds).sum::<f64>(),
        training
            .rounds
            .iter()
            .map(|r| r.selection_seconds)
            .sum::<f64>(),
        training
            .rounds
            .iter()
            .map(|r| r.rewrite_seconds)
            .sum::<f64>(),
        training
            .rounds
            .iter()
            .map(|r| r.refill_seconds)
            .sum::<f64>(),
        config.search.certify && training.rounds.iter().all(|r| r.proven_optimal),
        training.rounds.iter().filter(|r| !r.proven_optimal).count(),
        training
            .rounds
            .iter()
            .map(|r| r.search_seconds)
            .sum::<f64>(),
        training
            .rounds
            .iter()
            .map(|r| r.improved_components)
            .sum::<usize>(),
        training
            .rounds
            .iter()
            .map(|r| r.removed as u64 - r.greedy_score)
            .sum::<u64>(),
        !fast
    )
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
    if input == output {
        return Err("input and output must be different files".into());
    }
    // Create-new writes also protect source files reached through aliases or links.
    if output.exists() {
        return Err(format!("output already exists: {}", output.display()));
    }
    if args[0] == "decompress" {
        if args.len() != 3 {
            return Err("decompress expects exactly two paths".into());
        }
        let bytes = fs::read(&input).map_err(|e| format!("read {}: {e}", input.display()))?;
        let compressed = archive::decode(&bytes)?;
        let fields = compressed.decode()?;
        let mut writer = fs::File::create_new(&output).map_err(|e| e.to_string())?;
        use std::io::Write;
        for field in fields {
            writer.write_all(&field).map_err(|e| e.to_string())?;
        }
        println!(
            "Decoded {} bytes in {} field(s).",
            compressed.input_bytes, compressed.field_count
        );
        return Ok(());
    }
    if args[0] != "compress" {
        return Err(HELP.into());
    }
    let mut config = Config::default();
    let mut report_path = None;
    let mut rounds_path = None;
    let mut verbose = false;
    let mut fast = false;
    let mut at = 3;
    while at < args.len() {
        let flag = args[at].to_str().ok_or("option must be UTF-8")?;
        if flag == "--verbose" {
            verbose = true;
            at += 1;
            continue;
        }
        if flag == "--fast" {
            fast = true;
            config.search.certify = false;
            at += 1;
            continue;
        }
        let value = args
            .get(at + 1)
            .ok_or_else(|| format!("missing value for {flag}"))?;
        match flag {
            "--min-count" => {
                config.minimum_count = value
                    .to_str()
                    .ok_or("invalid count")?
                    .parse()
                    .map_err(|_| "invalid minimum count")?
            }
            "--buffer-bytes" => {
                config.buffer_bytes = value
                    .to_str()
                    .ok_or("invalid buffer size")?
                    .parse()
                    .map_err(|_| "invalid buffer size")?
            }
            "--selector" => {
                config.search.strategy = value.to_str().ok_or("invalid selector")?.parse()?
            }
            "--score" => config.score = value.to_str().ok_or("invalid score model")?.parse()?,
            "--pair-counting" => {
                config.pair_counting = value.to_str().ok_or("invalid counting mode")?.parse()?
            }
            "--pmi-alpha" => {
                config.pmi_alpha = value
                    .to_str()
                    .ok_or("invalid PMI alpha")?
                    .parse()
                    .map_err(|_| "invalid PMI alpha")?
            }
            "--tie-break" => {
                config.search.tie_break = value.to_str().ok_or("invalid tie break")?.parse()?
            }
            "--seed" => {
                config.search.seed = value
                    .to_str()
                    .ok_or("invalid seed")?
                    .parse()
                    .map_err(|_| "invalid seed")?
            }
            "--beam-width" => {
                config.search.beam_width = value
                    .to_str()
                    .ok_or("invalid beam width")?
                    .parse()
                    .map_err(|_| "invalid beam width")?
            }
            "--search-nodes" => {
                config.search.node_limit = value
                    .to_str()
                    .ok_or("invalid search budget")?
                    .parse()
                    .map_err(|_| "invalid search budget")?
            }
            "--report" => report_path = Some(PathBuf::from(value)),
            "--rounds" => rounds_path = Some(PathBuf::from(value)),
            _ => return Err(format!("unknown option: {flag}")),
        }
        at += 2;
    }
    if fast && config.search.strategy != bre_pair::search::Strategy::Greedy {
        return Err("--fast requires --selector greedy".into());
    }
    // Validate every destination before doing the training work.
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
        if !path
            .parent()
            .unwrap_or(std::path::Path::new("."))
            .as_os_str()
            .is_empty()
            && !path.parent().unwrap().is_dir()
        {
            return Err(format!(
                "destination directory does not exist: {}",
                path.display()
            ));
        }
    }
    let input_bytes = fs::read(&input).map_err(|e| format!("read {}: {e}", input.display()))?;
    let fields = vec![input_bytes];
    eprintln!(
        "Training on {} bytes; min count {}; active packed buffer {} bytes.",
        fields[0].len(),
        config.minimum_count,
        config.buffer_bytes
    );
    let training = engine::compress(&fields, &config, |r| {
        if verbose || r.index % 100 == 0 {
            eprintln!("round {:4}: {:4} pairs, {:6} rules, {:7} tokens -> {:7}, +{:6} input bytes, {} bits",
                r.index, r.selected, r.total_rules, r.before, r.rewritten, r.refill_bytes, r.width);
            if verbose && !fast {
                eprintln!("  candidates {}, eligible {}, max component {}, gain {}, proof {}, search {:.6}s, round {:.3}s",
                    r.candidates, r.eligible, r.largest_component,
                    r.removed as u64 - r.greedy_score, r.proven_optimal, r.search_seconds, r.seconds);
            }
        }
    })?;
    let bytes = archive::encode(&training.compressed)?;
    if !fast {
        // Validate the serialized artifact, not just the in-memory grammar.
        let restored = archive::decode(&bytes)?.decode()?;
        if restored != fields {
            return Err("round-trip verification failed".into());
        }
    }
    use std::io::Write;
    fs::File::create_new(&output)
        .and_then(|mut file| file.write_all(&bytes))
        .map_err(|e| format!("write {}: {e}", output.display()))?;
    if let Some(path) = report_path {
        fs::File::create_new(path)
            .and_then(|mut file| {
                file.write_all(report(&training, &config, bytes.len(), fast).as_bytes())
            })
            .map_err(|e| e.to_string())?;
    }
    if let Some(path) = rounds_path {
        fs::File::create_new(path)
            .and_then(|mut file| file.write_all(round_csv(&training.rounds).as_bytes()))
            .map_err(|e| e.to_string())?;
    }
    let compressed = &training.compressed;
    println!("Input:               {} bytes", compressed.input_bytes);
    println!("Rules:               {}", compressed.grammar.rules.len());
    println!(
        "Rounds / blocks:     {} / {}",
        training.rounds.len(),
        compressed.blocks.len()
    );
    println!("Tokens:              {}", compressed.token_count());
    println!(
        "Final symbol width:  {} bits",
        bre_pair::packed::symbol_width(compressed.grammar.alphabet_size())
    );
    println!(
        "Packed dictionary:   {} bytes",
        compressed.dictionary_bits().div_ceil(8)
    );
    if !compressed.grammar.rules.is_empty() {
        println!(
            "Dictionary / rule:   {:.3} bytes",
            compressed.dictionary_bits().div_ceil(8) as f64 / compressed.grammar.rules.len() as f64
        );
    }
    println!("Fixed-width data:    {} bytes", compressed.data_bytes());
    println!("Archive:             {} bytes", bytes.len());
    println!("Training:            {:.3} s", training.seconds);
    println!("Selector:            {}", config.search.strategy.name());
    println!("Merge priority:      {}", config.score.name());
    if config.score == bre_pair::scoring::Model::WeightedPmi {
        println!("PMI count exponent:  {}", config.pmi_alpha);
    }
    if fast {
        println!("Fast mode:           archive verification and clique certificates skipped");
    } else {
        println!(
            "Proven best batches: {} / {}",
            training.rounds.iter().filter(|r| r.proven_optimal).count(),
            training.rounds.len()
        );
        println!("Round trip:          verified byte-for-byte");
    }
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
