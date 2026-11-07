//! Complete rolling lookahead, with an exact candidate-budget checkpoint.
use bre_pair::{
    archive,
    lookahead::{self, Decision, Options, Policy, State},
    scoring::Model,
};
use std::{
    env, fs,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    process::ExitCode,
    time::Instant,
};

fn export(state: &State, fields: &[Vec<u8>], path: &Path) -> Result<usize, String> {
    let bytes = archive::encode(&state.compressed())?;
    if archive::decode(&bytes)?.decode()? != fields {
        return Err("serialized training archive differs from source".into());
    }
    fs::File::create_new(path)
        .and_then(|mut f| f.write_all(&bytes))
        .map_err(|e| e.to_string())?;
    Ok(bytes.len())
}
fn decision(d: &Decision, phase: &str) -> String {
    format!("{{\"phase\":\"{phase}\",\"index\":{},\"before_symbols\":{},\"after_symbols\":{},\"rules\":{},\"new_rules\":{},\"winner\":{},\"world_symbols\":[{}],\"first_rules\":{},\"second_rules\":{},\"budget_retries\":{}}}",d.index,d.before_symbols,d.after_symbols,d.total_rules,d.new_rules,d.winner,d.world_symbols.iter().map(u64::to_string).collect::<Vec<_>>().join(","),d.first_rules,d.second_rules,d.budget_retries)
}
fn run() -> Result<(), String> {
    let args: Vec<_> = env::args().skip(1).collect();
    let help="train-worlds INPUT DIRECTORY [--policy standard|control|lookahead] [--minimum 8] [--score weighted-pmi] [--layer-rules 512] [--candidate-budget 130815] [--max-input-bytes 134217728] [--candidate-only]";
    if args.first().is_some_and(|s| s == "--help") {
        println!("{help}");
        return Ok(());
    }
    if args.len() < 2 {
        return Err(help.into());
    }
    let mut options = Options {
        minimum_count: 8,
        score: Model::WeightedPmi,
        layer_rules: 512,
    };
    let mut policy = Policy::Lookahead;
    let mut budget = None;
    let mut maximum = 134217728u64;
    let mut candidate_only = false;
    let mut at = 2;
    while at < args.len() {
        if args[at] == "--candidate-only" {
            candidate_only = true;
            at += 1;
            continue;
        }
        let value = args.get(at + 1).ok_or("missing option value")?;
        match args[at].as_str() {
            "--policy" => {
                policy = match value.as_str() {
                    "standard" => Policy::Standard,
                    "control" => Policy::Control,
                    "lookahead" => Policy::Lookahead,
                    _ => return Err("invalid training policy".into()),
                }
            }
            "--minimum" => options.minimum_count = value.parse().map_err(|_| "invalid minimum")?,
            "--score" => options.score = value.parse()?,
            "--layer-rules" => {
                options.layer_rules = value.parse().map_err(|_| "invalid layer budget")?
            }
            "--candidate-budget" => {
                budget = Some(value.parse().map_err(|_| "invalid candidate budget")?)
            }
            "--max-input-bytes" => maximum = value.parse().map_err(|_| "invalid input cap")?,
            _ => return Err(format!("unknown option {}", args[at])),
        }
        at += 2;
    }
    if candidate_only && budget.is_none() {
        return Err("candidate-only requires a candidate budget".into());
    }
    if fs::metadata(&args[0]).map_err(|e| e.to_string())?.len() > maximum {
        return Err("input exceeds prototype size cap".into());
    }
    let raw = fs::read(&args[0]).map_err(|e| e.to_string())?;
    if raw.len() as u64 > maximum {
        return Err("input exceeds prototype size cap".into());
    }
    let fields: Vec<_> = if raw.is_empty() {
        vec![vec![]]
    } else {
        raw.split_inclusive(|&b| b == b'\n')
            .map(<[u8]>::to_vec)
            .collect()
    };
    let root = PathBuf::from(&args[1]);
    fs::create_dir(&root).map_err(|e| e.to_string())?;
    let mut trace = BufWriter::new(
        fs::File::create_new(root.join("decisions.jsonl")).map_err(|e| e.to_string())?,
    );
    let mut trace_error = None;
    let mut committed = 0usize;
    let started = Instant::now();
    let initial = State::from_fields(&fields)?;
    let candidate = lookahead::train(initial, options, policy, budget, |d, _| {
        let row = decision(d, "candidates");
        if let Err(e) = writeln!(trace, "{row}") {
            trace_error = Some(e.to_string());
        }
        committed += 1;
        if committed % 8 == 0 {
            println!("{row}");
        }
    })?;
    let candidate_seconds = started.elapsed().as_secs_f64();
    let candidate_rules = candidate.state.grammar.rules.len();
    let candidate_symbols = candidate.state.symbols.len();
    let candidate_converged = candidate.converged;
    let candidate_decisions = candidate.decisions;
    let candidate_worlds = candidate.chosen_worlds;
    let candidate_bytes = export(&candidate.state, &fields, &root.join("candidates.brp"))?;
    let candidate_limit = candidate.stopped_at_rule_limit;
    let continuing = Instant::now();
    let final_run = if candidate_only || candidate_converged {
        candidate
    } else {
        lookahead::train(candidate.state, options, policy, None, |d, _| {
            let row = decision(d, "convergence");
            if let Err(e) = writeln!(trace, "{row}") {
                trace_error = Some(e.to_string());
            }
            committed += 1;
            if committed % 8 == 0 {
                println!("{row}");
            }
        })?
    };
    let continuation_seconds = if candidate_only || candidate_converged {
        0.0
    } else {
        continuing.elapsed().as_secs_f64()
    };
    if let Some(e) = trace_error {
        return Err(e);
    }
    trace.flush().map_err(|e| e.to_string())?;
    let final_bytes = export(&final_run.state, &fields, &root.join("final.brp"))?;
    let final_worlds = if candidate_only || candidate_converged {
        candidate_worlds
    } else {
        std::array::from_fn(|i| candidate_worlds[i] + final_run.chosen_worlds[i])
    };
    let final_decisions = if candidate_only || candidate_converged {
        candidate_decisions
    } else {
        candidate_decisions + final_run.decisions
    };
    let limit = budget.map_or("null".into(), |n: usize| n.to_string());
    let report=format!(concat!("{{\"policy\":\"{}\",\"score\":\"{}\",\"minimum_count\":{},\"layer_rules\":{},\"input_bytes\":{},\"fields\":{},\"candidate_budget\":{},",
        "\"candidate_rules\":{},\"candidate_symbols\":{},\"candidate_archive_bytes\":{},\"candidate_seconds\":{:.6},\"candidate_converged\":{},\"candidate_stopped_at_rule_limit\":{},\"candidate_decisions\":{},\"candidate_chosen_worlds\":[{},{},{}],",
        "\"final_rules\":{},\"final_symbols\":{},\"final_archive_bytes\":{},\"final_converged\":{},\"continuation_seconds\":{:.6},\"decisions\":{},\"chosen_worlds\":[{},{},{}],\"total_seconds\":{:.6},\"round_trip_verified\":true}}\n"),
        policy.name(),options.score.name(),options.minimum_count,options.layer_rules,raw.len(),fields.len(),limit,
        candidate_rules,candidate_symbols,candidate_bytes,candidate_seconds,candidate_converged,candidate_limit,candidate_decisions,candidate_worlds[0],candidate_worlds[1],candidate_worlds[2],
        final_run.state.grammar.rules.len(),final_run.state.symbols.len(),final_bytes,final_run.converged,continuation_seconds,final_decisions,final_worlds[0],final_worlds[1],final_worlds[2],started.elapsed().as_secs_f64());
    fs::File::create_new(root.join("training.json"))
        .and_then(|mut f| f.write_all(report.as_bytes()))
        .map_err(|e| e.to_string())?;
    print!("{report}");
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
