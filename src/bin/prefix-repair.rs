//! Early beam discovery and ordinary-trainer continuations, kept opt-in.
use bre_pair::{
    archive,
    engine::Config,
    field_engine,
    grammar::BASE_SYMBOLS,
    prefix_beam::{self, Options, Prefix},
    scoring::Model,
};
use std::{env, fs, io::Write, path::Path, process::ExitCode, time::Instant};

const HELP: &str = "prefix-repair discover INPUT DIRECTORY [--rules 16] [--width 8] [--branch 4] [--minimum 8] [--threads 4]\n\
prefix-repair train INPUT DIRECTORY [--prefix FILE.rules] [--learned-budget 130815] [--minimum 8] [--threads 4]\n\
Both accept --max-input-bytes 134217728. Input is raw lines, retaining newlines and appending EOF.\n\
Discovery retains recipes and shared dense counters, not corpus copies per state.\n\
Train resumes the exact prefix segmentation with ordinary weighted PMI (alpha=1).\n\
All output directories/files must be new. This file-input prototype has an explicit size cap.\n";

fn write(path: &Path, contents: &[u8]) -> Result<(), String> {
    fs::File::create_new(path)
        .and_then(|mut f| f.write_all(contents))
        .map_err(|e| format!("{}: {e}", path.display()))
}

fn save_prefix(path: &Path, prefix: &Prefix) -> Result<(), String> {
    let mut text = format!("BREPAIR-PREFIX 1 {}\n", prefix.symbols);
    for &(left, right) in &prefix.pairs {
        text.push_str(&format!("{left} {right}\n"));
    }
    write(path, text.as_bytes())
}

fn read_prefix(path: &Path) -> Result<Prefix, String> {
    let text = fs::read_to_string(path).map_err(|e| e.to_string())?;
    let mut words = text.split_whitespace();
    if words.next() != Some("BREPAIR-PREFIX") || words.next() != Some("1") {
        return Err("invalid prefix file header".into());
    }
    let symbols = words
        .next()
        .ok_or("missing prefix size")?
        .parse()
        .map_err(|_| "invalid prefix size")?;
    let mut pairs = Vec::new();
    while let Some(left) = words.next() {
        let right = words.next().ok_or("incomplete prefix rule")?;
        pairs.push((
            left.parse().map_err(|_| "invalid left symbol")?,
            right.parse().map_err(|_| "invalid right symbol")?,
        ));
    }
    let prefix = Prefix { pairs, symbols };
    prefix.grammar()?;
    Ok(prefix)
}

fn run() -> Result<(), String> {
    let args: Vec<_> = env::args().skip(1).collect();
    if args.first().is_some_and(|s| s == "--help") {
        print!("{HELP}");
        return Ok(());
    }
    if args.len() < 3 {
        return Err(HELP.into());
    }
    let mode = &args[0];
    if mode != "discover" && mode != "train" {
        return Err(HELP.into());
    }
    let mut options = Options::default();
    let mut budget = 130815usize;
    let mut maximum = 134217728u64;
    let mut prefix_path = None;
    for chunk in args[3..].chunks(2) {
        if chunk.len() != 2 {
            return Err("missing option value".into());
        }
        match chunk[0].as_str() {
            "--rules" if mode == "discover" => {
                options.rules = chunk[1].parse().map_err(|_| "invalid rules")?
            }
            "--width" if mode == "discover" => {
                options.width = chunk[1].parse().map_err(|_| "invalid width")?
            }
            "--branch" if mode == "discover" => {
                options.branch = chunk[1].parse().map_err(|_| "invalid branch")?
            }
            "--prefix" if mode == "train" => prefix_path = Some(Path::new(&chunk[1])),
            "--learned-budget" if mode == "train" => {
                budget = chunk[1].parse().map_err(|_| "invalid budget")?
            }
            "--minimum" => {
                options.minimum_count = chunk[1].parse().map_err(|_| "invalid minimum")?
            }
            "--threads" => options.workers = chunk[1].parse().map_err(|_| "invalid threads")?,
            "--max-input-bytes" => maximum = chunk[1].parse().map_err(|_| "invalid input cap")?,
            _ => return Err(format!("unknown option {}", chunk[0])),
        }
    }
    if budget == 0 || options.minimum_count < 2 || options.workers == 0 || options.workers > 256 {
        return Err("invalid training options".into());
    }
    if fs::metadata(&args[1]).map_err(|e| e.to_string())?.len() > maximum {
        return Err("input exceeds explicit prototype cap".into());
    }
    let raw = fs::read(&args[1]).map_err(|e| e.to_string())?;
    if raw.len() as u64 > maximum {
        return Err("input exceeds explicit prototype cap".into());
    }
    let input_bytes = raw.len();
    let fields: Vec<_> = if raw.is_empty() {
        vec![vec![]]
    } else {
        raw.split_inclusive(|&b| b == b'\n')
            .map(<[u8]>::to_vec)
            .collect()
    };
    drop(raw);
    let root = Path::new(&args[2]);
    fs::create_dir(root).map_err(|e| e.to_string())?;
    let mut trace = fs::File::create_new(root.join("trace.jsonl")).map_err(|e| e.to_string())?;
    let mut trace_error = None;
    let started = Instant::now();
    if mode == "discover" {
        let result = prefix_beam::discover(&fields, options, |level| {
            let row=format!("{{\"depth\":{},\"states\":{},\"proposed\":{},\"distinct\":{},\"best_symbols\":{},\"seconds\":{:.6}}}",level.depth,level.states,level.proposed,level.distinct,level.best_symbols,level.seconds);
            println!("{row}");
            if let Err(e) = writeln!(trace, "{row}") {
                trace_error = Some(e.to_string());
            }
        })?;
        let mut beam = Vec::new();
        for (index, prefix) in result.beam.iter().enumerate() {
            let name = format!("beam-{index:02}");
            save_prefix(&root.join(format!("{name}.rules")), prefix)?;
            beam.push(format!(
                "{{\"name\":\"{name}\",\"rules\":{},\"symbols\":{}}}",
                prefix.pairs.len(),
                prefix.symbols
            ));
        }
        save_prefix(&root.join("greedy.rules"), &result.greedy)?;
        let report=format!("{{\"rules\":{},\"width\":{},\"branch_per_ranking\":{},\"minimum_count\":{},\"workers\":{},\"input_bytes\":{},\"fields\":{},\"counter_arena_bytes\":{},\"seconds\":{:.6},\"greedy_symbols\":{},\"beam\":[{}]}}\n",options.rules,options.width,options.branch,options.minimum_count,options.workers,input_bytes,fields.len(),result.counter_arena_bytes,started.elapsed().as_secs_f64(),result.greedy.symbols,beam.join(","));
        write(&root.join("discovery.json"), report.as_bytes())?;
        print!("{report}");
    } else {
        let config = Config {
            buffer_bytes: usize::MAX,
            minimum_count: options.minimum_count,
            score: Model::WeightedPmi,
            ..Config::default()
        };
        let prefix = prefix_path.map(read_prefix).transpose()?;
        let on_round = |r: &bre_pair::engine::Round| {
            let row = format!(
                "{{\"round\":{},\"rules\":{},\"symbols\":{},\"selected\":{},\"seconds\":{:.6}}}",
                r.index, r.total_rules, r.rewritten, r.selected, r.seconds
            );
            if r.index % 8 == 0 {
                println!("{row}");
            }
            if let Err(e) = writeln!(trace, "{row}") {
                trace_error = Some(e.to_string());
            }
        };
        let result = if let Some(prefix) = &prefix {
            prefix_beam::continue_training(
                prefix,
                &fields,
                &config,
                options.workers,
                budget,
                on_round,
            )?
        } else {
            field_engine::compress_with_options(
                &fields,
                &config,
                options.workers,
                64,
                65536,
                field_engine::Options {
                    max_rules: Some(budget),
                    ..Default::default()
                },
                on_round,
            )?
        };
        let training_seconds = started.elapsed().as_secs_f64();
        let compressed = result.training.compressed;
        let bytes = archive::encode(&compressed)?;
        if archive::decode(&bytes)?.decode()? != fields {
            return Err("serialized archive differs from training fields".into());
        }
        write(&root.join("candidates.brp"), &bytes)?;
        let report=format!("{{\"input_bytes\":{},\"fields\":{},\"minimum_count\":{},\"score\":\"weighted-pmi\",\"pmi_alpha\":1,\"workers\":{},\"learned_budget\":{},\"prefix_rules\":{},\"rules\":{},\"symbols\":{},\"archive_bytes\":{},\"training_seconds\":{:.6},\"total_seconds\":{:.6},\"stopped_at_rule_limit\":{},\"registry_peak_slot_bytes\":{},\"round_trip_verified\":true}}\n",input_bytes,fields.len(),options.minimum_count,options.workers,budget,prefix.as_ref().map_or(0,|p|p.pairs.len()),compressed.grammar.alphabet_size()-BASE_SYMBOLS,compressed.token_count(),bytes.len(),training_seconds,started.elapsed().as_secs_f64(),result.stopped_at_rule_limit,result.registry_peak_slot_bytes);
        write(&root.join("training.json"), report.as_bytes())?;
        print!("{report}");
    }
    if let Some(e) = trace_error {
        return Err(e);
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
