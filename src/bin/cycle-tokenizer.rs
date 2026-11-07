//! Experimental train-only collapse/discover/select cycle; evaluation is separate.
use bre_pair::{
    archive,
    engine::Config,
    repeated, scoring,
    tokenizer::{fingerprint, Model, Tokenizer},
    vocabulary::{self, RefinementOptions, SelectionOptions},
};
use std::{env, fs, io::Write, path::Path, process::ExitCode, time::Instant};

const HELP: &str = "cycle-tokenizer TRAIN SEED.brt OUTPUT_DIRECTORY [--pool-multiple 8]\n\
  [--threads 4] [--min-count 8] [--buffer-bytes 134217728]\n\
  [--collapse-ranking FILE --collapse-count N] [--block-rounds 2]\n\
  [--shake-seed N] [--shake-percent 10] [--shake-pulses 2]\n\
One training-only cycle: exact seed parse, BRe-Pair, selection back to seed size.\n\
Output directory must be new. Preserves the seed if training token count regresses.\n\
Writes proposal.brt as well as model.brt so rejected proposals remain reviewable.\n";

fn write(path: &Path, bytes: &[u8]) -> Result<(), String> {
    fs::File::create_new(path)
        .and_then(|mut f| f.write_all(bytes))
        .map_err(|e| format!("{}: {e}", path.display()))
}

fn run() -> Result<(), String> {
    let args: Vec<_> = env::args().skip(1).collect();
    if args.is_empty() || args[0] == "--help" {
        print!("{HELP}");
        return Ok(());
    }
    if args.len() < 3 {
        return Err(HELP.into());
    }
    let (mut multiple, mut workers, mut minimum, mut buffer) =
        (8usize, 4usize, 8u64, 134217728usize);
    let mut at = 3;
    let (mut collapse_ranking, mut collapse_count, mut block_rounds) = (None, 0usize, 2usize);
    let (mut shake_seed, mut shake_percent, mut shake_pulses) = (None, 10.0f64, 2usize);
    while at < args.len() {
        let value = args.get(at + 1).ok_or("missing option value")?;
        match args[at].as_str() {
            "--pool-multiple" => multiple = value.parse().map_err(|_| "invalid multiple")?,
            "--threads" => workers = value.parse().map_err(|_| "invalid workers")?,
            "--min-count" => minimum = value.parse().map_err(|_| "invalid minimum")?,
            "--buffer-bytes" => buffer = value.parse().map_err(|_| "invalid buffer")?,
            "--collapse-ranking" => collapse_ranking = Some(value.clone()),
            "--collapse-count" => {
                collapse_count = value.parse().map_err(|_| "invalid collapse count")?
            }
            "--block-rounds" => block_rounds = value.parse().map_err(|_| "invalid block rounds")?,
            "--shake-seed" => {
                shake_seed = Some(value.parse::<u64>().map_err(|_| "invalid shake seed")?)
            }
            "--shake-percent" => {
                shake_percent = value.parse().map_err(|_| "invalid shake percentage")?
            }
            "--shake-pulses" => {
                shake_pulses = value.parse().map_err(|_| "invalid shake pulse count")?
            }
            _ => return Err(format!("unknown option {}", args[at])),
        }
        at += 2;
    }
    if multiple < 2 || minimum < 2 || workers == 0 || workers > 256 {
        return Err("pool multiple >=2, minimum >=2, and workers 1..=256 required".into());
    }
    if shake_seed.is_none()
        && args
            .iter()
            .any(|s| s == "--shake-percent" || s == "--shake-pulses")
    {
        return Err("shake settings require --shake-seed".into());
    }
    let started = Instant::now();
    let path = Path::new(&args[0]);
    let root = Path::new(&args[2]);
    let seed = Model::from_bytes(&fs::read(&args[1]).map_err(|e| e.to_string())?)?;
    let size = seed.tokens.len();
    let limit = size.checked_mul(multiple).ok_or("pool limit overflow")?;
    let seed_fingerprint = fingerprint(&seed.to_bytes());
    let mut collapse = Vec::new();
    if let Some(ranking) = collapse_ranking {
        if collapse_count == 0 || block_rounds == 0 {
            return Err("collapse count and blocked rounds must be positive".into());
        }
        let text = fs::read_to_string(ranking).map_err(|e| e.to_string())?;
        let mut parts = text.split_whitespace();
        if parts.next() != Some("BRCOLLAPSE01")
            || parts.next().and_then(|s| s.parse::<u64>().ok()) != Some(seed_fingerprint)
        {
            return Err("collapse ranking belongs to a different model".into());
        }
        collapse = parts
            .take(collapse_count)
            .map(|s| {
                s.parse::<u32>()
                    .map_err(|_| "invalid collapse ID".to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;
        if collapse.len() != collapse_count {
            return Err("collapse ranking is too short".into());
        }
    } else if collapse_count != 0 {
        return Err("collapse count requires a ranking file".into());
    }
    let baseline = vocabulary::scan(
        path,
        &Tokenizer::new(seed.clone())?,
        None,
        None,
        workers,
        1,
        0,
        true,
        true,
    )?;
    if baseline.uses[257..].iter().any(|&n| n < minimum) {
        return Err("seed contains below-minimum learned tokens on this training corpus".into());
    }
    fs::create_dir(root).map_err(|e| format!("{}: {e}", root.display()))?;
    let mut config = Config {
        score: scoring::Model::WeightedPmi,
        minimum_count: minimum,
        buffer_bytes: buffer,
        ..Config::default()
    };
    config.search.certify = false;
    let shake = shake_seed.map(|seed| bre_pair::shake::Options {
        seed,
        fraction: shake_percent / 100.0,
        pulses: shake_pulses,
        ..bre_pair::shake::Options::default()
    });
    let discovery = repeated::discover_with_options(
        path,
        &seed,
        &config,
        &repeated::DiscoveryOptions {
            workers,
            candidate_limit: limit,
            max_token_bytes: 128,
            collapse: collapse.clone(),
            block_rounds,
            shake,
        },
        |r| {
            println!(
            "{{\"event\":\"batch\",\"index\":{},\"selected\":{},\"tokens\":{},\"seconds\":{:.6}}}",
            r.index, r.selected, r.rewritten, r.seconds
        );
        },
    )?;
    if discovery.initial_tokens < baseline.all.tokens
        || (collapse.is_empty() && discovery.initial_tokens != baseline.all.tokens)
    {
        return Err("seed segmentation count differs".into());
    }
    let candidates = discovery.pool.tokens.len();
    let discovery_seconds = discovery.training.training.seconds;
    let discovery_batches = discovery.training.training.rounds.len();
    let registry_peak_bytes = discovery.training.registry_peak_slot_bytes;
    let stopped_at_limit = discovery.training.stopped_at_rule_limit;
    let blocked_candidates = discovery.blocked_candidates;
    let block_lifted_early = discovery.block_lifted_early;
    let initial_tokens = discovery.initial_tokens;
    let source_verified = Instant::now();
    vocabulary::verify_training_source(&discovery.training.training.compressed, path)?;
    write(
        &root.join("candidates.brp"),
        &archive::encode(&discovery.training.training.compressed)?,
    )?;
    // Preserve the exact filtered pool; the diagnostic archive includes helpers.
    write(&root.join("candidate-pool.brt"), &discovery.pool.to_bytes())?;
    let support: Vec<_> = discovery
        .support
        .iter()
        .flat_map(|n| n.to_le_bytes())
        .collect();
    write(&root.join("candidate-support.u64"), &support)?;
    let verification_export_seconds = source_verified.elapsed().as_secs_f64();
    let segmentation_seconds = discovery.segmentation_seconds;
    let helper_rules = discovery.helper_rules;
    let shakes = discovery.training.shakes.clone();
    drop(discovery.training);
    println!("{{\"event\":\"candidates\",\"vocabulary\":{candidates}}}");
    let select_started = Instant::now();
    let selected = vocabulary::select_refined(
        discovery.pool,
        &discovery.support,
        path,
        &SelectionOptions {
            vocabulary: size,
            workers,
            ..SelectionOptions::default()
        },
        &RefinementOptions {
            workers,
            minimum_uses: minimum,
            ..RefinementOptions::default()
        },
        |r| {
            println!(
                "{{\"event\":\"prune\",\"before\":{},\"after\":{},\"seconds\":{:.6}}}",
                r.before, r.after, r.seconds
            )
        },
        |r| {
            println!(
                "{{\"event\":\"refine\",\"index\":{},\"removed\":{},\"added\":{},\"tokens\":{}}}",
                r.index, r.removed, r.added, r.totals.tokens
            )
        },
    );
    let selection_seconds = select_started.elapsed().as_secs_f64();
    let (model, accepted, proposal_tokens, selection_failed) = match selected {
        Ok(proposal) => {
            let bytes = proposal.model.to_bytes();
            write(&root.join("proposal.brt"), &bytes)?;
            let checked = vocabulary::scan(
                path,
                &Tokenizer::new(proposal.model.clone())?,
                None,
                None,
                workers,
                1,
                0,
                true,
                true,
            )?;
            if checked.all.tokens != proposal.after.tokens
                || checked.uses[257..].iter().any(|&n| n < minimum)
                || proposal.model.tokens.len() != size
            {
                return Err("proposal usage/round-trip/budget verification failed".into());
            }
            let accepted = proposal.after.tokens < baseline.all.tokens;
            (
                if accepted { proposal.model } else { seed },
                accepted,
                Some(proposal.after.tokens),
                false,
            )
        }
        Err(error) => {
            write(&root.join("selection-error.txt"), error.as_bytes())?;
            (seed, false, None, true)
        }
    };
    let bytes = model.to_bytes();
    write(&root.join("model.brt"), &bytes)?;
    let proposal_json = proposal_tokens.map_or("null".into(), |n| n.to_string());
    let collapse_ids = collapse
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(",");
    let output_tokens = if accepted {
        proposal_tokens.unwrap()
    } else {
        baseline.all.tokens
    };
    let report = format!("{{\"vocabulary\":{size},\"collapse_ids\":[{collapse_ids}],\"block_rounds\":{block_rounds},\"blocked_candidates\":{blocked_candidates},\"block_lifted_early\":{block_lifted_early},\"initial_discovery_tokens\":{initial_tokens},\"candidate_limit\":{limit},\"candidates\":{candidates},\"initial_helper_rules\":{helper_rules},\"minimum_count\":{minimum},\"workers\":{workers},\"seed_fingerprint\":{seed_fingerprint},\"model_fingerprint\":{},\"training_bytes\":{},\"seed_training_tokens\":{},\"proposal_training_tokens\":{proposal_json},\"training_tokens\":{output_tokens},\"accepted\":{accepted},\"selection_failed\":{selection_failed},\"discovery_batches\":{discovery_batches},\"stopped_at_candidate_limit\":{stopped_at_limit},\"segmentation_seconds\":{segmentation_seconds:.6},\"discovery_seconds\":{discovery_seconds:.6},\"selection_seconds\":{selection_seconds:.6},\"verification_export_seconds\":{verification_export_seconds:.6},\"registry_peak_slot_bytes\":{registry_peak_bytes},\"source_verified\":true,\"round_trip_verified\":true,\"seconds\":{:.6}}}\n",
        fingerprint(&bytes), baseline.all.bytes, baseline.all.tokens, started.elapsed().as_secs_f64());
    let report = if let Some(options) = shake {
        let events = shakes.iter().map(|e| format!("{{\"after_batch\":{},\"selected_symbols\":[{}],\"before\":{},\"after\":{},\"seconds\":{:.6}}}",
            e.after_batch,e.selected_symbols.iter().map(u32::to_string).collect::<Vec<_>>().join(","),e.before,e.after,e.seconds)).collect::<Vec<_>>().join(",");
        format!("{},\"shake_seed\":{},\"shake_fraction\":{},\"shake_pulses\":{},\"shake_first_batch\":{},\"shake_interval\":{},\"shake_block_rounds\":{},\"shakes\":[{}]}}\n",
            report.trim_end().strip_suffix('}').unwrap(), options.seed,options.fraction,options.pulses,options.first_batch,options.interval,options.block_rounds,events)
    } else {
        report
    };
    write(&root.join("report.json"), report.as_bytes())?;
    if shake.is_some() {
        println!("{{\"event\":\"complete\",\"accepted\":{accepted},\"training_tokens\":{output_tokens},\"shakes\":{}}}",shakes.len());
    } else {
        println!("{report}");
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
