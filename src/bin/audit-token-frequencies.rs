//! Compare grammar-derived counts with one exact, fixed-vocabulary training scan.
use bre_pair::{
    archive, document_support,
    tokenizer::{fingerprint, Model, Tokenizer},
    training_progress,
    vocabulary::{self, frequency_estimates, SelectionOptions, Totals},
};
use std::{
    env, fs,
    io::Write,
    path::{Path, PathBuf},
    process::ExitCode,
    time::Instant,
};

fn write(path: &Path, bytes: &[u8]) -> Result<(), String> {
    fs::File::create_new(path)
        .and_then(|mut f| f.write_all(bytes))
        .map_err(|e| format!("{}: {e}", path.display()))
}
fn array(values: &[u64]) -> String {
    values
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join(",")
}
fn totals(t: Totals) -> String {
    format!(
        "{{\"bytes\":{},\"tokens\":{},\"fields\":{},\"byte_tokens\":{},\"bytes_per_token\":{:.9}}}",
        t.bytes,
        t.tokens,
        t.fields,
        t.byte_tokens,
        t.bytes_per_token()
    )
}
fn run() -> Result<(), String> {
    let args: Vec<_> = env::args().skip(1).collect();
    let help = "audit-token-frequencies ARCHIVE TRAIN OUTPUT_DIRECTORY (--model MODEL.brt | --select-vocabulary N) [--selection-method statistics|estimated|exact|reverse|reverse-exact] [--measure true|false] [--threads 16] [--document-index TRAIN.docs --min-token-documents 16] [--coarse-drop-percent 40 --fine-drop-percent 15 --fine-below 32768] (exact/reverse selection only) [--coverage-discount-ppm 0 --coverage-types all|starter|container|ender|boundaries|none --coverage-internal false --coverage-overlap-weight presence|linear|square --coverage-frequency count|presence|log|availability --coverage-min-overlap 1 --coverage-normalization local|global --coverage-signal overlap|shuffle|stratified|uniform --coverage-seed 0] (reverse selection only; shuffle/stratified/uniform are research controls) [--reverse-search-trials 8 --reverse-search-seed 1 --reverse-search-swaps 4] (reverse-exact only; training-only batch comparison)";
    if args.first().is_some_and(|x| x == "--help") {
        println!("{help}");
        return Ok(());
    }
    if args.len() < 3 || !(args.len() - 3).is_multiple_of(2) {
        return Err(help.into());
    }
    let (mut model_path, mut size, mut index, mut minimum, mut workers) =
        (None, None, None, 16usize, 16usize);
    let mut selection_method = "statistics";
    let mut measure = true;
    let mut maximum_token_bytes = 128usize;
    let mut schedule = vocabulary::PruningSchedule::default();
    let mut custom_schedule = false;
    let mut coverage_discount_ppm = 0u32;
    let mut coverage_option = false;
    let mut coverage_settings = vocabulary::reverse::CoverageSettings::default();
    let mut reverse_search_trials = 0usize;
    let mut reverse_search_seed = 1u64;
    let mut reverse_search_swaps = 4usize;
    let mut reverse_search_option = false;
    for pair in args[3..].chunks_exact(2) {
        match pair[0].as_str() {
            "--reverse-search-trials" => {
                reverse_search_trials = pair[1]
                    .parse()
                    .map_err(|_| "invalid reverse search trials")?;
                if !(1..=64).contains(&reverse_search_trials) {
                    return Err("reverse search trials must be1..64".into());
                }
                reverse_search_option = true;
            }
            "--reverse-search-seed" => {
                reverse_search_seed = pair[1].parse().map_err(|_| "invalid reverse search seed")?;
                reverse_search_option = true;
            }
            "--reverse-search-swaps" => {
                reverse_search_swaps = pair[1]
                    .parse()
                    .map_err(|_| "invalid reverse search swaps")?;
                if !(1..=64).contains(&reverse_search_swaps) {
                    return Err("reverse search swaps must be1..64".into());
                }
                reverse_search_option = true;
            }
            "--coverage-signal" => {
                coverage_settings.signal = match pair[1].as_str() {
                    "overlap" => vocabulary::reverse::CoverageSignal::Overlap,
                    "shuffle" => vocabulary::reverse::CoverageSignal::Shuffle,
                    "stratified" => vocabulary::reverse::CoverageSignal::Stratified,
                    "uniform" => vocabulary::reverse::CoverageSignal::Uniform,
                    _ => return Err("invalid coverage signal".into()),
                };
                coverage_option = true;
            }
            "--coverage-seed" => {
                coverage_settings.seed = pair[1].parse().map_err(|_| "invalid coverage seed")?;
                coverage_option = true;
            }
            "--coverage-types" => {
                coverage_settings.types = match pair[1].as_str() {
                    "all" => 7,
                    "starter" => 1,
                    "container" => 2,
                    "none" => 0,
                    "ender" => 4,
                    "boundaries" => 5,
                    _ => return Err("invalid coverage types".into()),
                };
                coverage_option = true;
            }
            "--coverage-overlap-weight" => {
                coverage_settings.overlap_weight = match pair[1].as_str() {
                    "presence" => vocabulary::reverse::OverlapWeight::Presence,
                    "linear" => vocabulary::reverse::OverlapWeight::Linear,
                    "square" => vocabulary::reverse::OverlapWeight::Square,
                    _ => return Err("invalid overlap weight".into()),
                };
                coverage_option = true;
            }
            "--coverage-frequency" => {
                coverage_settings.frequency_weight = match pair[1].as_str() {
                    "count" => vocabulary::reverse::FrequencyWeight::Count,
                    "presence" => vocabulary::reverse::FrequencyWeight::Presence,
                    "log" => vocabulary::reverse::FrequencyWeight::Log,
                    "availability" => vocabulary::reverse::FrequencyWeight::Availability,
                    _ => return Err("invalid frequency weight".into()),
                };
                coverage_option = true;
            }
            "--coverage-min-overlap" => {
                coverage_settings.minimum_overlap =
                    pair[1].parse().map_err(|_| "invalid minimum overlap")?;
                if coverage_settings.minimum_overlap == 0 {
                    return Err("minimum overlap must be positive".into());
                }
                coverage_option = true;
            }
            "--coverage-internal" => {
                coverage_settings.internal = pair[1]
                    .parse()
                    .map_err(|_| "coverage-internal must be true or false")?;
                coverage_option = true;
            }
            "--coverage-normalization" => {
                coverage_settings.normalization = match pair[1].as_str() {
                    "local" => vocabulary::reverse::CoverageNormalization::Local,
                    "global" => vocabulary::reverse::CoverageNormalization::Global,
                    _ => return Err("invalid normalization".into()),
                };
                coverage_option = true;
            }
            "--coverage-discount-ppm" => {
                coverage_discount_ppm = pair[1].parse().map_err(|_| "invalid coverage discount")?;
                if coverage_discount_ppm > 1_000_000 {
                    return Err("coverage discount must be0..1000000".into());
                }
                coverage_option = true;
            }
            "--coarse-drop-percent" | "--fine-drop-percent" => {
                let drop = pair[1]
                    .parse::<f64>()
                    .map_err(|_| "invalid pruning percentage")?;
                if !(0.0..100.0).contains(&drop) || drop == 0.0 {
                    return Err("pruning percentage must be greater than0 and less than100".into());
                }
                if pair[0] == "--coarse-drop-percent" {
                    schedule.coarse_keep = 1.0 - drop / 100.0;
                } else {
                    schedule.fine_keep = 1.0 - drop / 100.0;
                }
                custom_schedule = true;
            }
            "--fine-below" => {
                schedule.fine_below = pair[1]
                    .parse()
                    .map_err(|_| "invalid fine pruning threshold")?;
                custom_schedule = true;
            }
            "--selection-method" => {
                selection_method = &pair[1];
                if !matches!(
                    selection_method,
                    "statistics" | "estimated" | "exact" | "reverse" | "reverse-exact"
                ) {
                    return Err("invalid selection method".into());
                }
            }
            "--measure" => {
                measure = pair[1]
                    .parse()
                    .map_err(|_| "measure must be true or false")?
            }
            "--model" => model_path = Some(PathBuf::from(&pair[1])),
            "--select-vocabulary" => {
                size = Some(pair[1].parse::<usize>().map_err(|_| "invalid vocabulary")?)
            }
            "--threads" => workers = pair[1].parse().map_err(|_| "invalid workers")?,
            "--max-token-bytes" => {
                maximum_token_bytes = pair[1].parse().map_err(|_| "invalid maximum token bytes")?
            }
            "--document-index" => index = Some(PathBuf::from(&pair[1])),
            "--min-token-documents" => {
                minimum = pair[1].parse().map_err(|_| "invalid document minimum")?
            }
            _ => return Err(format!("unknown option {}", pair[0])),
        }
    }
    if model_path.is_some() == size.is_some()
        || maximum_token_bytes == 0
        || !(1..=256).contains(&workers)
        || !(1..=64).contains(&minimum)
        || (model_path.is_some() && selection_method != "statistics")
        || (custom_schedule && !matches!(selection_method, "exact" | "reverse" | "reverse-exact"))
        || (coverage_option && !matches!(selection_method, "reverse" | "reverse-exact"))
        || !coverage_settings.valid()
        || (reverse_search_option
            && (selection_method != "reverse-exact"
                || reverse_search_trials == 0
                || coverage_discount_ppm != 0))
    {
        return Err(help.into());
    }
    let root = Path::new(&args[2]);
    fs::create_dir(root).map_err(|e| format!("{}: {e}", root.display()))?;
    let _progress = training_progress::enable();
    let started = Instant::now();
    let source = archive::decode(&fs::read(&args[0]).map_err(|e| e.to_string())?)?;
    if fs::metadata(&args[1]).map_err(|e| e.to_string())?.len() != source.input_bytes {
        return Err("source length mismatch".into());
    }
    let policy = index.map(|index| document_support::Policy {
        index,
        minimum_groups: minimum,
    });
    let mut rounds = Vec::new();
    let mut reverse_counts = None;
    let model = if let Some(path) = model_path {
        Model::from_bytes(&fs::read(path).map_err(|e| e.to_string())?)?
    } else {
        let cleanup = policy
            .as_ref()
            .map(|policy| {
                let mut protected = vec![false; source.grammar.alphabet_size()];
                protected[..257].fill(true);
                document_support::Cleanup::plan(&source, policy, &protected)
            })
            .transpose()?;
        let (pool, support) = Model::candidates_filtered(
            &source,
            maximum_token_bytes,
            cleanup.as_ref().map(|p| p.active.as_slice()),
        )?;
        drop(cleanup);
        if selection_method == "exact" {
            vocabulary::select_exact_with_schedule(
                pool,
                &support,
                Path::new(&args[1]),
                size.unwrap(),
                workers,
                schedule,
                |r| {
                    training_progress::phase(
                        "vocabulary_selection",
                        rounds.len() + 1,
                        r.after as u64,
                        r.before as u64,
                        "candidates_remaining",
                    );
                    let row=format!("{{\"round\":{},\"before\":{},\"after\":{},\"measured\":{},\"scan_stride\":1,\"seconds\":{:.6}}}",rounds.len()+1,r.before,r.after,totals(r.totals),r.seconds);
                    eprintln!("vocabulary selection {row}");
                    rounds.push(row);
                },
            )?
        } else if matches!(selection_method, "reverse" | "reverse-exact") {
            let selected = vocabulary::reverse::select_with_options(
                pool,
                &support,
                Path::new(&args[1]),
                vocabulary::reverse::Options {
                    batch_search: (reverse_search_trials != 0).then_some(
                        vocabulary::reverse::SearchOptions {
                            trials: reverse_search_trials,
                            seed: reverse_search_seed,
                            swaps_per_token: reverse_search_swaps,
                        },
                    ),
                    coverage_discount_ppm,
                    coverage_settings,
                    vocabulary: size.unwrap(),
                    workers,
                    schedule,
                    exact_refresh: selection_method == "reverse-exact",
                },
                |r| {
                    training_progress::phase(
                        "vocabulary_selection",
                        rounds.len() + 1,
                        r.after as u64,
                        r.before as u64,
                        "candidates_remaining",
                    );
                    let row = format!(
                        "{{\"round\":{},\"before\":{},\"after\":{},\"maintained_before\":{},\"maintained_after\":{},\"added_tokens\":{},\"deferred\":{},\"replacement_edges\":{},\"workspace_bytes\":{},\"replacement_reused\":{},\"replacement_rebuilt\":{},\"planning_seconds\":{:.6},\"ranking_seconds\":{:.6},\"conflict_seconds\":{:.6},\"update_seconds\":{:.6},\"coverage_seconds\":{:.6},\"coverage_targets\":{},\"coverage_pair_checks\":{},\"coverage_index_bytes\":{},\"coverage_mean_ratio_ppm\":{:.3},\"coverage_max_ratio_ppm\":{},\"coverage_saturated_targets\":{},\"coverage_zero_support_targets\":{},\"coverage_mean_one_overlap_share_ppm\":{:.3},\"coverage_mean_internal_share_ppm\":{:.3},\"reparse_saved\":{},\"refresh_seconds\":{:.6},\"seconds\":{:.6},\"batch_search\":{{\"requested\":{},\"evaluated\":{},\"duplicates\":{},\"baseline_tokens\":{},\"chosen_tokens\":{},\"chosen_trial\":{},\"chosen_exchanges\":{},\"workspace_bytes\":{},\"seconds\":{:.6}}}}}",
                        rounds.len()+1,r.before,r.after,totals(r.before_totals),totals(r.after_totals),
                        r.added_tokens,r.deferred,r.replacement_edges,r.workspace_bytes,
                        r.replacement_reused,r.replacement_rebuilt,r.planning_seconds,r.ranking_seconds,r.conflict_seconds,r.update_seconds,
                        r.coverage_seconds,r.coverage_targets,r.coverage_pair_checks,r.coverage_index_bytes,
                        r.coverage_mean_ratio_ppm,r.coverage_max_ratio_ppm,r.coverage_saturated_targets,r.coverage_zero_support_targets,r.coverage_mean_one_overlap_share_ppm,
                        r.coverage_mean_internal_share_ppm,
                        r.reparse_saved,r.refresh_seconds,r.seconds,
                        r.batch_search.requested,r.batch_search.evaluated,r.batch_search.duplicates,
                        r.batch_search.baseline_tokens,r.batch_search.chosen_tokens,r.batch_search.chosen_trial,
                        r.batch_search.chosen_exchanges,r.batch_search.workspace_bytes,r.batch_search.seconds
                    );
                    eprintln!("vocabulary selection {row}");
                    rounds.push(row);
                },
            )?;
            reverse_counts = Some((
                selected.initial,
                selected.maintained,
                selected.maintained_uses,
                selected.initial_scan_seconds,
                selected.refresh_scans,
                selected.refresh_seconds,
            ));
            selected.model
        } else if selection_method == "estimated" {
            vocabulary::select_estimated(pool, &support, &source, size.unwrap(), |r| {
                training_progress::phase(
                    "vocabulary_selection",
                    0,
                    r.after as u64,
                    r.before as u64,
                    "candidates_remaining",
                );
                rounds.push(format!(
                    "{{\"before\":{},\"after\":{},\"estimated\":{},\"seconds\":{:.6}}}",
                    r.before,
                    r.after,
                    totals(r.estimated),
                    r.seconds
                ));
            })?
        } else {
            vocabulary::select(
                pool,
                &support,
                Path::new(&args[1]),
                &SelectionOptions {
                    vocabulary: size.unwrap(),
                    workers,
                    ..SelectionOptions::default()
                },
                |r| {
                    training_progress::phase(
                        "vocabulary_selection",
                        0,
                        r.after as u64,
                        r.before as u64,
                        "candidates_remaining",
                    )
                },
            )?
        }
    };
    let preparation_seconds = started.elapsed().as_secs_f64();
    let bytes = model.to_bytes();
    let model_fingerprint = fingerprint(&bytes);
    write(&root.join("model.brt"), &bytes)?;
    if let Some((initial, maintained, uses, seconds, refresh_scans, refresh_seconds)) =
        &reverse_counts
    {
        let report = format!(
            "{{\"model_fingerprint\":{model_fingerprint},\"semantics\":\"{}\",\"initial\":{},\"maintained\":{},\"uses\":[{}],\"initial_scan_seconds\":{seconds:.6},\"refresh_scans\":{refresh_scans},\"refresh_seconds\":{refresh_seconds:.6}}}",
            if selection_method == "reverse-exact" { "literal_rewrite_after_each_exact_reparse" } else { "literal_rewrite_of_initial_exact_parse" },
            totals(*initial),totals(*maintained),array(uses)
        );
        write(&root.join("reverse-maintained.json"), report.as_bytes())?;
    }
    let timer = Instant::now();
    let estimates = frequency_estimates::from_grammar(&source, &model)?;
    let estimate_seconds = timer.elapsed().as_secs_f64();
    drop(source);
    let estimated=format!("{{\"model_fingerprint\":{model_fingerprint},\"collapsed\":{},\"inclusive_uses\":[{}],\"collapsed_uses\":[{}],\"matched_rules\":{},\"absent_model_tokens\":{},\"seconds\":{estimate_seconds:.6}}}",
        totals(estimates.collapsed),array(&estimates.inclusive_uses),array(&estimates.collapsed_uses),estimates.matched_rules,estimates.absent_model_tokens);
    write(&root.join("estimates.json"), estimated.as_bytes())?;
    let schedule_json = format!(
        "{{\"coarse_keep\":{},\"fine_keep\":{},\"fine_below\":{}}}",
        schedule.coarse_keep, schedule.fine_keep, schedule.fine_below
    );
    let selection = format!(
        "{{\"method\":\"{selection_method}\",\"support_prior\":{},\"pruning_schedule\":{},\"coverage_discount_ppm\":{coverage_discount_ppm},\"batch_search\":{{\"trials\":{reverse_search_trials},\"seed\":{reverse_search_seed},\"swaps_per_token\":{reverse_search_swaps}}},\"coverage_settings\":{{\"types\":{},\"overlap_weight\":\"{}\",\"frequency\":\"{}\",\"min_overlap\":{},\"normalization\":\"{}\",\"internal\":{},\"signal\":\"{}\",\"seed\":{}}},\"rounds\":[{}]}}",
        if matches!(selection_method, "estimated" | "exact" | "reverse" | "reverse-exact") {
            "0.02"
        } else {
            "null"
        },
        schedule_json,coverage_settings.types,coverage_settings.overlap_weight.name(),coverage_settings.frequency_weight.name(),coverage_settings.minimum_overlap,coverage_settings.normalization.name(),coverage_settings.internal,coverage_settings.signal.name(),coverage_settings.seed,rounds.join(",")
    );
    write(&root.join("selection.json"), selection.as_bytes())?;
    // Freeze predictions before measuring. The vocabulary never changes below.
    println!("{{\"event\":\"estimates_frozen\",\"model_fingerprint\":{model_fingerprint},\"estimated_tokens\":{},\"estimate_seconds\":{estimate_seconds:.6}}}",estimates.collapsed.tokens);
    if !measure {
        let report=format!("{{\"model_fingerprint\":{model_fingerprint},\"vocabulary\":{},\"selection_method\":\"{selection_method}\",\"selection_refitted\":false,\"measured\":false,\"actual\":null,\"estimated\":{},\"preparation_seconds\":{preparation_seconds:.6},\"estimate_seconds\":{estimate_seconds:.6},\"total_seconds\":{:.6}}}",model.tokens.len(),totals(estimates.collapsed),started.elapsed().as_secs_f64());
        write(&root.join("report.json"), report.as_bytes())?;
        println!("{report}");
        return Ok(());
    }
    let tokenizer = Tokenizer::new(model)?;
    let (measured, groups) = if let Some(policy) = &policy {
        let found = document_support::scan(Path::new(&args[1]), &tokenizer, None, workers, policy)?;
        (found.scan, Some(found.groups))
    } else {
        (
            vocabulary::scan(
                Path::new(&args[1]),
                &tokenizer,
                None,
                None,
                workers,
                1,
                0,
                true,
                true,
            )?,
            None,
        )
    };
    if estimates.collapsed.bytes != measured.all.bytes
        || estimates.collapsed.fields != measured.all.fields
        || estimates.collapsed.tokens < measured.all.tokens
    {
        return Err("estimated partition violates exact-parse conservation/minimum bound".into());
    }
    if reverse_counts
        .as_ref()
        .is_some_and(|(_, maintained, _, _, _, _)| {
            maintained.bytes != measured.all.bytes
                || maintained.fields != measured.all.fields
                || maintained.tokens < measured.all.tokens
        })
    {
        return Err("reverse partition violates exact-parse conservation/minimum bound".into());
    }
    let groups = groups.map_or("null".into(), |g| {
        format!(
            "[{}]",
            g.iter().map(u32::to_string).collect::<Vec<_>>().join(",")
        )
    });
    let actual=format!("{{\"model_fingerprint\":{model_fingerprint},\"all\":{},\"uses\":[{}],\"document_groups\":{groups},\"minimum_document_groups\":{},\"seconds\":{:.6},\"round_trip_verified\":true}}",totals(measured.all),array(&measured.uses),policy.as_ref().map_or("null".into(),|p|p.minimum_groups.to_string()),measured.seconds);
    write(&root.join("actual.json"), actual.as_bytes())?;
    let report=format!("{{\"model_fingerprint\":{model_fingerprint},\"vocabulary\":{},\"selection_method\":\"{selection_method}\",\"measured\":true,\"selection_refitted\":false,\"actual\":{},\"estimated\":{},\"preparation_seconds\":{preparation_seconds:.6},\"estimate_seconds\":{estimate_seconds:.6},\"scan_seconds\":{:.6},\"total_seconds\":{:.6},\"round_trip_verified\":true}}",tokenizer.model().tokens.len(),totals(measured.all),totals(estimates.collapsed),measured.seconds,started.elapsed().as_secs_f64());
    write(&root.join("report.json"), report.as_bytes())?;
    println!("{report}");
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
