//! Single-thread pruning experiment, exporting native variable-arity rules.
use std::env;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use bre_pair::engine::{self, Config};
use bre_pair::prune::{self, Checkpoint};
use bre_pair::scoring::Model;
use bre_pair::{archive, sequence_archive};

const HELP: &str = "Rare-symbol pruning experiment\n\n\
Usage:\n  prune-repair control INPUT.brp OUTPUT.mrp [--report FILE.json] [--verify]\n\
  prune-repair post INPUT.brp OUTPUT.mrp [--min-count N] [--max-uses N]\n\
               [--strategy frequency|cost] [--report FILE.json] [--verify]\n\
  prune-repair during INPUT OUTPUT.mrp --after-round N [--max-uses N]\n\
               [--pmi-alpha N] [--min-count N] [--buffer-bytes N] [--report FILE.json] [--verify]\n\n\
Defaults: prune direct uses below --min-count (4), weighted PMI alpha 1, 1 MiB buffer.\n\
--max-uses explicitly overrides the pruning cutoff for experiments.\n\
--strategy cost chooses post-pruning by native packed storage cost.\n\
Measure dictionary-compressed packages separately before choosing an export.\n\
Post mode needs the original training minimum supplied if it was not 4;\n\
BRPAIR01 archives do not store that setting.\n\
During mode prunes once, requires the complete input in memory and retains\n\
historical construction nodes. Decompress .mrp with mr-repair.\n";

fn write_new(path: &Path, bytes: &[u8]) -> Result<(), String> {
    fs::File::create_new(path)
        .and_then(|mut f| f.write_all(bytes))
        .map_err(|e| format!("write {}: {e}", path.display()))
}

fn ratio(value: u128, count: usize) -> String {
    if count == 0 {
        "null".into()
    } else {
        format!("{:.6}", value as f64 / count as f64)
    }
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
    let mode = args[0].to_str().ok_or("invalid mode")?;
    if !["control", "post", "during"].contains(&mode) {
        return Err(HELP.into());
    }
    let input = PathBuf::from(&args[1]);
    let output = PathBuf::from(&args[2]);
    let mut config = Config {
        score: Model::WeightedPmi,
        ..Config::default()
    };
    config.search.certify = false;
    let mut max_uses = None;
    let mut after_round = None;
    let mut report_path = None;
    let mut verify = false;
    let mut strategy = "frequency";
    let mut at = 3;
    while at < args.len() {
        let flag = args[at].to_str().ok_or("invalid flag")?;
        if flag == "--verify" {
            verify = true;
            at += 1;
            continue;
        }
        let value = args
            .get(at + 1)
            .ok_or_else(|| format!("missing value for {flag}"))?;
        match flag {
            "--strategy" => {
                strategy = value.to_str().ok_or("invalid pruning strategy")?;
                if !["frequency", "cost"].contains(&strategy) {
                    return Err("strategy must be frequency or cost".into());
                }
            }
            "--max-uses" => {
                max_uses = Some(
                    value
                        .to_str()
                        .ok_or("invalid max uses")?
                        .parse()
                        .map_err(|_| "invalid max uses")?,
                )
            }
            "--after-round" => {
                after_round = Some(
                    value
                        .to_str()
                        .ok_or("invalid checkpoint")?
                        .parse()
                        .map_err(|_| "invalid checkpoint")?,
                )
            }
            "--pmi-alpha" => {
                config.pmi_alpha = value
                    .to_str()
                    .ok_or("invalid alpha")?
                    .parse()
                    .map_err(|_| "invalid alpha")?
            }
            "--min-count" => {
                config.minimum_count = value
                    .to_str()
                    .ok_or("invalid minimum")?
                    .parse()
                    .map_err(|_| "invalid minimum")?
            }
            "--buffer-bytes" => {
                config.buffer_bytes = value
                    .to_str()
                    .ok_or("invalid buffer")?
                    .parse()
                    .map_err(|_| "invalid buffer")?
            }
            "--report" => report_path = Some(PathBuf::from(value)),
            _ => return Err(format!("unknown flag: {flag}")),
        }
        at += 2;
    }
    if mode == "during" && after_round.is_none() {
        return Err("during mode needs --after-round".into());
    }
    if mode != "during" && after_round.is_some() {
        return Err("--after-round applies only to during mode".into());
    }
    if strategy == "cost" && mode != "post" {
        return Err("cost strategy applies only to post mode".into());
    }
    if config.minimum_count < 2 {
        return Err("minimum count must be at least 2".into());
    }
    let max_uses = max_uses.unwrap_or(if mode == "control" {
        0
    } else {
        config.minimum_count - 1
    });
    let mut destinations = vec![output.clone()];
    destinations.extend(report_path.iter().cloned());
    for (i, path) in destinations.iter().enumerate() {
        if path == &input || path.exists() || destinations[..i].contains(path) {
            return Err(format!(
                "destination exists or is duplicated: {}",
                path.display()
            ));
        }
        let parent = path.parent().unwrap_or(Path::new("."));
        if !parent.as_os_str().is_empty() && !parent.is_dir() {
            return Err(format!(
                "missing destination directory: {}",
                parent.display()
            ));
        }
    }
    let training_seconds;
    let rounds;
    let preparation_seconds;
    let selected_rules;
    let reactivated;
    let event_before;
    let event_after;
    let event_seconds;
    let source;
    let retired;
    let expected;
    let mut cost_estimate = None;
    if mode == "during" {
        config.pruning = Some(Checkpoint {
            after_round: after_round.unwrap(),
            max_uses,
        });
        let fields = vec![fs::read(&input).map_err(|e| e.to_string())?];
        let training = engine::compress(&fields, &config, |_| {})?;
        let state = training
            .pruning
            .ok_or("pruning checkpoint was not reached")?;
        training_seconds = training.seconds;
        rounds = training.rounds.len();
        preparation_seconds = 0.0;
        selected_rules = state.event.selected_rules;
        reactivated = state.reactivated;
        event_before = state.event.before_tokens;
        event_after = state.event.after_tokens;
        event_seconds = state.event.seconds;
        source = training.compressed;
        retired = state.retired;
        expected = if verify { Some(fields) } else { None };
    } else {
        source = archive::decode(&fs::read(&input).map_err(|e| e.to_string())?)?;
        training_seconds = 0.0;
        rounds = 0;
        reactivated = 0;
        event_before = source.token_count();
        let started = Instant::now();
        if mode == "post" {
            let plan = if strategy == "cost" {
                let cost = prune::cost_plan(&source, max_uses)?;
                cost_estimate = Some((cost.baseline_bytes, cost.predicted_bytes, cost.choice));
                cost.plan
            } else {
                prune::plan(&source.grammar, &source.blocks, max_uses)?
            };
            selected_rules = plan.selected_rules;
            event_after = plan
                .before_tokens
                .checked_add(plan.added_tokens)
                .ok_or("pruning length overflow")?;
            retired = plan.retired;
        } else {
            selected_rules = 0;
            event_after = event_before;
            retired = vec![];
        }
        preparation_seconds = started.elapsed().as_secs_f64();
        event_seconds = preparation_seconds;
        expected = if verify { Some(source.decode()?) } else { None };
    }
    let export_started = Instant::now();
    let native = prune::export(&source, &retired)?;
    let export_seconds = export_started.elapsed().as_secs_f64();
    if mode != "during" && native.token_count() != event_after {
        return Err("post-pruning prediction disagrees with output".into());
    }
    let encoded = sequence_archive::encode(&native)?;
    if let Some((_, predicted, _)) = cost_estimate {
        if encoded.len() != predicted {
            return Err("cost-aware size prediction disagrees with export".into());
        }
    }
    let dictionary_bits = u64::from_le_bytes(encoded[32..40].try_into().unwrap()) as usize;
    let dictionary_bytes = dictionary_bits.div_ceil(8);
    if let Some(expected) = expected {
        if sequence_archive::decode(&encoded)?.to_binary()?.decode()? != expected {
            return Err("pruned archive verification failed".into());
        }
    }
    let tokens = native.token_count();
    let rules = native.rules.len();
    let rhs: usize = native.rules.iter().map(Vec::len).sum();
    write_new(&output, &encoded)?;
    if let Some(path) = report_path {
        let report=format!(concat!(
            "{{\n  \"format\": \"BRPNAR01\",\n  \"mode\": \"{}\",\n",
            "  \"minimum_count\": {},\n",
            "  \"pruning_strategy\": \"{}\",\n",
            "  \"native_control_bytes\": {},\n  \"predicted_archive_bytes\": {},\n",
            "  \"cost_choice\": {},\n",
            "  \"max_direct_uses\": {},\n  \"after_round\": {},\n",
            "  \"input_bytes\": {},\n  \"fields\": {},\n",
            "  \"historical_binary_rules\": {},\n  \"rules\": {},\n",
            "  \"selected_for_retirement\": {},\n  \"reactivated_rules\": {},\n",
            "  \"checkpoint_before_tokens\": {},\n  \"checkpoint_after_tokens\": {},\n",
            "  \"rounds\": {},\n  \"tokens\": {},\n",
            "  \"rule_rhs_symbols\": {},\n  \"maximum_rhs_length\": {},\n",
            "  \"dictionary_bytes\": {},\n  \"fixed_width_data_bytes\": {},\n  \"archive_bytes\": {},\n",
            "  \"input_bytes_per_symbol\": {},\n  \"dictionary_bytes_per_rule\": {},\n",
            "  \"training_seconds\": {:.6},\n  \"pruning_event_seconds\": {:.6},\n",
            "  \"post_preparation_seconds\": {:.6},\n  \"export_seconds\": {:.6},\n",
            "  \"training_and_pruning_seconds\": {:.6},\n",
            "  \"round_trip_verified\": {},\n  \"huffman_coded\": false,\n",
            "  \"training_graph_reclaimed\": false\n}}\n"
        ),mode,config.minimum_count,strategy,
            cost_estimate.map(|c| c.0.to_string()).unwrap_or("null".into()),
            cost_estimate.map(|c| c.1.to_string()).unwrap_or("null".into()),
            cost_estimate.map(|c| format!("\"{}\"",c.2)).unwrap_or("null".into()),
            max_uses,after_round.map(|n|n.to_string()).unwrap_or("null".into()),
            native.input_bytes,native.field_count,source.grammar.rules.len(),rules,
            selected_rules,reactivated,event_before,event_after,rounds,tokens,rhs,
            native.rules.iter().map(Vec::len).max().unwrap_or(0),dictionary_bytes,native.data_bytes(),encoded.len(),
            ratio(native.input_bytes as u128,tokens),ratio(dictionary_bytes as u128,rules),
            training_seconds,event_seconds,preparation_seconds,export_seconds,
            training_seconds+preparation_seconds+export_seconds,verify);
        write_new(&path, report.as_bytes())?;
    }
    println!("{mode}: {rules} exported rules, {tokens} symbols, {} original bytes/symbol, {} archive bytes",ratio(native.input_bytes as u128,tokens),encoded.len());
    println!("Historical binary rules: {}; retired at checkpoint: {selected_rules}; reactivated: {reactivated}",source.grammar.rules.len());
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
