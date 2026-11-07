//! Probe complementary pair-type policies without changing the trainer.
use bre_pair::{
    archive,
    lookahead::{self, Options, Stage, State},
    scoring::Model,
};
use std::{env, fs, io::Write, path::PathBuf, process::ExitCode, time::Instant};

fn stage(stage: &Stage) -> String {
    let pairs = stage
        .pairs
        .iter()
        .map(|(p, n)| format!("{{\"left\":{},\"right\":{},\"count\":{}}}", p.0, p.1, n))
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "{{\"before_symbols\":{},\"after_symbols\":{},\"new_rules\":{},\"selected_pairs\":[{}]}}",
        stage.before_symbols, stage.after_symbols, stage.new_rules, pairs
    )
}

fn run() -> Result<(), String> {
    let args: Vec<_> = env::args().skip(1).collect();
    let help = "lookahead-worlds INPUT DIRECTORY [--minimum 4] [--score weighted-pmi] [--layer-rules 32] [--checkpoints 0,8,16,24,32] [--max-input-bytes 16777216] [--archives]";
    if args.first().is_some_and(|s| s == "--help") {
        println!("{help}");
        return Ok(());
    }
    if args.len() < 2 {
        return Err(help.into());
    }
    let mut options = Options {
        minimum_count: 4,
        score: Model::WeightedPmi,
        layer_rules: 32,
    };
    let mut checkpoints = vec![0, 8, 16, 24, 32];
    let mut maximum = 16 * 1024 * 1024u64;
    let mut archives = false;
    let mut at = 2;
    while at < args.len() {
        if args[at] == "--archives" {
            archives = true;
            at += 1;
            continue;
        }
        let value = args.get(at + 1).ok_or("missing option value")?;
        match args[at].as_str() {
            "--minimum" => options.minimum_count = value.parse().map_err(|_| "invalid minimum")?,
            "--score" => options.score = value.parse()?,
            "--layer-rules" => {
                options.layer_rules = value.parse().map_err(|_| "invalid layer budget")?
            }
            "--checkpoints" => {
                checkpoints = value
                    .split(',')
                    .map(|s| s.parse().map_err(|_| "invalid checkpoint".to_string()))
                    .collect::<Result<Vec<usize>, _>>()?
            }
            "--max-input-bytes" => maximum = value.parse().map_err(|_| "invalid input cap")?,
            _ => return Err(format!("unknown option {}", args[at])),
        }
        at += 2;
    }
    checkpoints.sort_unstable();
    checkpoints.dedup();
    if checkpoints.is_empty() {
        return Err("at least one checkpoint is required".into());
    }
    if fs::metadata(&args[0]).map_err(|e| e.to_string())?.len() > maximum {
        return Err("input exceeds explicit prototype size cap".into());
    }
    let raw = fs::read(&args[0]).map_err(|e| e.to_string())?;
    if raw.len() as u64 > maximum {
        return Err("input exceeds explicit prototype size cap".into());
    }
    let fields: Vec<_> = if raw.is_empty() {
        vec![vec![]]
    } else {
        raw.split_inclusive(|&b| b == b'\n')
            .map(<[u8]>::to_vec)
            .collect()
    };
    let directory = PathBuf::from(&args[1]);
    fs::create_dir(&directory).map_err(|e| e.to_string())?;
    let mut state = State::from_fields(&fields)?;
    let mut batch = 0;
    let started = Instant::now();
    let mut rows = Vec::new();
    for checkpoint in checkpoints {
        while batch < checkpoint {
            let Some(next) = lookahead::advance(&state, options)? else {
                break;
            };
            state = next;
            batch += 1;
        }
        if batch < checkpoint {
            break;
        }
        let probe_start = Instant::now();
        let probe = lookahead::probe(&state, options)?;
        let mut worlds = Vec::new();
        for world in &probe.worlds {
            let encoded = archive::encode(&world.state.compressed())?;
            if archive::decode(&encoded)?.decode()? != fields {
                return Err("world archive differs from original fields".into());
            }
            if archives {
                let path = directory.join(format!("batch{batch:03}-{}.brp", world.name));
                fs::File::create_new(path)
                    .and_then(|mut f| f.write_all(&encoded))
                    .map_err(|e| e.to_string())?;
            }
            worlds.push(format!("{{\"name\":\"{}\",\"first\":{},\"second\":{},\"rules\":{},\"symbols\":{},\"archive_bytes\":{},\"round_trip_verified\":true}}",world.name,stage(&world.first),stage(&world.second),world.state.grammar.rules.len(),world.state.symbols.len(),encoded.len()));
        }
        let baseline = probe.worlds[0].state.symbols.len();
        let best = probe.worlds[probe.winner].state.symbols.len();
        let row=format!("{{\"batch\":{batch},\"source_rules\":{},\"source_symbols\":{},\"eligible_pair_types\":{},\"enabled_types_A\":{},\"enabled_types_B\":{},\"additional_rules\":{},\"effective_first_cap\":{},\"budget_retries\":{},\"winner\":\"{}\",\"saved_vs_control\":{},\"seconds\":{:.6},\"worlds\":[{}]}}",state.grammar.rules.len(),state.symbols.len(),probe.eligible_types,probe.enabled_types[0],probe.enabled_types[1],probe.additional_rules,probe.effective_first_cap,probe.budget_retries,probe.worlds[probe.winner].name,baseline-best,probe_start.elapsed().as_secs_f64(),worlds.join(","));
        println!(
            "{{\"batch\":{batch},\"winner\":\"{}\",\"saved_vs_control\":{},\"rules\":{}}}",
            probe.worlds[probe.winner].name,
            baseline - best,
            state.grammar.rules.len() + probe.additional_rules
        );
        rows.push(row);
    }
    if rows.is_empty() {
        return Err("training converged before any requested probe checkpoint".into());
    }
    let result=format!("{{\"input_bytes\":{},\"fields\":{},\"minimum_count\":{},\"score\":\"{}\",\"layer_rules\":{},\"mask\":\"alternating original eligible types in descending score/count, lex tie order; complement in world B\",\"second_layer\":\"all eligible types, normal conflict selection\",\"rule_budget_matched\":true,\"seconds\":{:.6},\"probes\":[{}]}}\n",raw.len(),fields.len(),options.minimum_count,options.score.name(),options.layer_rules,started.elapsed().as_secs_f64(),rows.join(","));
    fs::File::create_new(directory.join("report.json"))
        .and_then(|mut f| f.write_all(result.as_bytes()))
        .map_err(|e| e.to_string())?;
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
