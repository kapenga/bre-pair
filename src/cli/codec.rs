use bre_pair::{
    atomic_output::Output,
    codec::{DocumentCounting, EncodeOptions, Frozen, OptimizeOptions, TrainOptions},
    engine::Config,
    huffman_archive::DEFAULT_BLOCK_SYMBOLS,
    memory,
    scoring::Model as Score,
    tokenizer::{fingerprint, Model},
};
use std::{fs, io::Write, path::Path, time::Instant};

pub fn run(args: &[String], limit: &memory::Limit, verbose: bool) -> Result<(), String> {
    let start = Instant::now();
    if args[0] == "codec-inspect" {
        if args.len() != 2 {
            return Err("codec-inspect MODEL.brc".into());
        }
        let bytes = fs::read(&args[1]).map_err(|e| e.to_string())?;
        let model = Frozen::from_bytes(&bytes)?;
        println!("{{\"format\":\"BRCODE01\",\"model_fingerprint\":{},\"ai_vocabulary\":{},\"codec_rules\":{},\"huffman_codes\":{},\"max_code_bits\":{},\"training_bytes\":{},\"training_fields\":{},\"training_ai_tokens\":{},\"training_codec_symbols\":{}}}",fingerprint(&bytes),model.dictionary.model.tokens.len(),model.dictionary.rules.len(),model.book.symbols.len(),model.book.max_bits,model.training.bytes,model.training.fields,model.training.tokens,model.training.symbols);
        return Ok(());
    }
    if args.len() < 4 {
        return Err(
            "codec-train MODEL.brt TRAIN OUTPUT.brc | codec-encode MODEL.brc INPUT OUTPUT.brh"
                .into(),
        );
    }
    let output = Output::new(Path::new(&args[3]))?;
    if args[0] == "codec-optimize" {
        let mut options = OptimizeOptions::default();
        let (mut report, mut recode) = (None, None);
        for pair in args[4..].chunks(2) {
            let value = pair.get(1).ok_or("missing optimizer option value")?;
            match pair[0].as_str() {
                "--max-trials" => {
                    options.max_trials = value.parse().map_err(|_| "invalid trial limit")?
                }
                "--report" => report = Some(Output::new(Path::new(value))?),
                "--recode" => recode = Some(Output::new(Path::new(value))?),
                _ => return Err("unknown codec-optimize option".into()),
            }
        }
        let paths: Vec<_> = std::iter::once(output.destination())
            .chain(report.iter().map(Output::destination))
            .chain(recode.iter().map(Output::destination))
            .collect();
        if paths
            .iter()
            .enumerate()
            .any(|(i, p)| paths[..i].contains(p))
        {
            return Err("optimizer outputs must differ".into());
        }
        let model = Frozen::from_bytes(&fs::read(&args[1]).map_err(|e| e.to_string())?)?;
        let optimized = model.optimize_archive(Path::new(&args[2]), options)?;
        let archive_bytes = if let Some(recode) = &recode {
            optimized.recode_archive(&model, Path::new(&args[2]), recode.path())?;
            Some(
                fs::metadata(recode.path())
                    .map_err(|e| e.to_string())?
                    .len(),
            )
        } else {
            None
        };
        fs::File::create_new(output.path())
            .and_then(|mut f| {
                f.write_all(&optimized.model.to_bytes().map_err(std::io::Error::other)?)
            })
            .map_err(|e| e.to_string())?;
        let r = &optimized.report;
        let text = format!("{{\"training_only\":true,\"ai_vocabulary_unchanged\":true,\"trials\":{},\"accepted\":{},\"before_codes\":{},\"after_codes\":{},\"before_rules\":{},\"after_rules\":{},\"removed_zero_use_codes\":{},\"removed_positive_use_codes\":{},\"before_payload_bits\":{},\"after_payload_bits\":{},\"before_metadata_bytes\":{},\"after_metadata_bytes\":{},\"source_training_archive_bytes\":{},\"recoded_training_archive_bytes\":{},\"seconds\":{:.6}}}\n",r.trials,r.accepted,r.before_codes,r.after_codes,r.before_rules,r.after_rules,r.removed_zero_use_codes,r.removed_positive_use_codes,r.before_payload_bits,r.after_payload_bits,r.before_metadata_bytes,r.after_metadata_bytes,fs::metadata(&args[2]).map_err(|e|e.to_string())?.len(),archive_bytes.map_or_else(||"null".into(),|n|n.to_string()),start.elapsed().as_secs_f64());
        if let Some(report) = &report {
            fs::File::create_new(report.path())
                .and_then(|mut f| f.write_all(text.as_bytes()))
                .map_err(|e| e.to_string())?;
        }
        if let Some(recode) = recode {
            recode.commit()?;
        }
        output.commit()?;
        if let Some(report) = report {
            report.commit()?;
        }
        print!("{text}");
        return Ok(());
    }
    if args[0] == "codec-encode" {
        let mut size = DEFAULT_BLOCK_SYMBOLS;
        let mut options = EncodeOptions::default();
        for pair in args[4..].chunks(2) {
            let value = pair.get(1).ok_or("missing codec encoding option value")?;
            match pair[0].as_str() {
                "--block-symbols" => size = value.parse().map_err(|_| "invalid block size")?,
                "--threads" => {
                    options.workers = value.parse().map_err(|_| "invalid worker count")?
                }
                "--mode" => options.mode = value.parse()?,
                _ => return Err("unknown codec-encode option".into()),
            }
        }
        let model = Frozen::from_bytes(&fs::read(&args[1]).map_err(|e| e.to_string())?)?;
        let totals =
            model.encode_file_with_options(Path::new(&args[2]), output.path(), size, options)?;
        output.commit()?;
        let seconds = start.elapsed().as_secs_f64();
        println!("{{\"original_bytes\":{},\"fields\":{},\"terminal_tokens\":{},\"codec_symbols\":{},\"seconds\":{seconds:.6},\"original_MB_per_second\":{:.3},\"input_passes\":1,\"huffman_retrained\":false,\"threads\":{},\"mode\":\"{}\",\"peak_heap_bytes\":{},\"peak_private_bytes\":{}}}",totals.bytes,totals.fields,totals.tokens,totals.symbols,totals.bytes as f64/seconds/1e6,options.workers,options.mode.as_str(),memory::peak_heap_bytes(),memory::process().peak_private);
        return Ok(());
    }
    let (mut workers, mut minimum, mut maximum, mut score, mut report) =
        (4, 8, None, Score::WeightedPmi, None);
    let mut max_rounds = None;
    let mut minimum_count_per_gb = 0u64;
    let mut minimum_documents = 1usize;
    let mut document_index = None;
    let mut document_counting = DocumentCounting::Exact;
    let mut fill_waves = 1u8;
    let mut fill_min_count = 0u64;
    let mut fill_min_count_per_gb = 0u64;
    for pair in args[4..].chunks(2) {
        let value = pair.get(1).ok_or("missing codec training option value")?;
        match pair[0].as_str() {
            "--threads" => workers = value.parse().map_err(|_| "invalid workers")?,
            "--min-count" => minimum = value.parse().map_err(|_| "invalid minimum count")?,
            "--max-rules" => {
                maximum = Some(value.parse::<usize>().map_err(|_| "invalid rule limit")?)
            }
            "--max-rounds" => {
                max_rounds = Some(value.parse::<usize>().map_err(|_| "invalid round limit")?)
            }
            "--min-count-per-gb" => {
                minimum_count_per_gb = value.parse().map_err(|_| "invalid count per GB")?
            }
            "--min-rule-documents" => {
                minimum_documents = value.parse().map_err(|_| "invalid document minimum")?
            }
            "--document-index" => document_index = Some(Path::new(value).to_owned()),
            "--fill-waves" => fill_waves = value.parse().map_err(|_| "invalid fill-wave count")?,
            "--fill-min-count" => {
                fill_min_count = value.parse().map_err(|_| "invalid fill count floor")?
            }
            "--fill-min-count-per-gb" => {
                fill_min_count_per_gb =
                    value.parse().map_err(|_| "invalid normalized fill count")?
            }
            "--document-counting" => {
                document_counting = value.parse()?;
            }
            "--score" => score = value.parse()?,
            "--report" => report = Some(Output::new(Path::new(value))?),
            _ => return Err("unknown codec-train option".into()),
        }
    }
    if minimum < 2 || !matches!(score, Score::Frequency | Score::WeightedPmi) {
        return Err("codec training requires min-count >=2 and frequency or weighted-pmi".into());
    }
    if max_rounds == Some(0) {
        return Err("codec round limit must be positive".into());
    }
    if !(1..=8).contains(&fill_waves) {
        return Err("fill waves must be 1..=8".into());
    }
    let fill_min_count_base = fill_min_count;
    fill_min_count = bre_pair::search::scaled_fill_minimum(
        fill_min_count,
        fill_min_count_per_gb,
        fs::metadata(&args[2]).map_err(|e| e.to_string())?.len(),
    )?;
    if !(1..=64).contains(&minimum_documents) || (minimum_documents > 1 && document_index.is_none())
    {
        return Err(
            "rule support requires 1..64 documents and --document-index when greater than one"
                .into(),
        );
    }
    if report
        .as_ref()
        .is_some_and(|r| r.destination() == output.destination())
    {
        return Err("codec model and report must differ".into());
    }
    let vocabulary = Model::from_bytes(&fs::read(&args[1]).map_err(|e| e.to_string())?)?;
    let config = Config {
        buffer_bytes: limit.heap,
        minimum_count: minimum,
        score,
        search: bre_pair::search::Options {
            certify: false,
            fill_waves,
            fill_min_count,
            ..Default::default()
        },
        ..Default::default()
    };
    let mut rules_created = 0usize;
    let _progress = bre_pair::training_progress::enable();
    let (model, training) = Frozen::train_with_options(
        &vocabulary,
        Path::new(&args[2]),
        &config,
        workers,
        TrainOptions {
            document_counting,
            max_rules: maximum,
            max_rounds,
            minimum_count_per_gb,
            documents: document_index.map(|index| bre_pair::document_support::Policy {
                index,
                minimum_groups: minimum_documents,
            }),
        },
        |r| {
            rules_created += r.new_rules;
            bre_pair::training_progress::completed(r, rules_created);
            // One small record per completed round; no corpus scan or model
            // snapshot. Keep the periodic human-readable lines compatible.
            eprintln!(
                "codec progress {{\"event\":\"round\",\"round\":{},\"symbols\":{},\"grammar_rules\":{},\"new_rules\":{},\"rules_created\":{},\"packed_bytes\":{},\"symbol_bits\":{},\"selected_pairs\":{},\"minimum_selected_count\":{},\"maximum_selected_count\":{},\"removed_symbols\":{},\"removed_percent\":{:.6},\"seconds\":{:.3}}}",
                r.index, r.after_refill, r.total_rules, r.new_rules, rules_created,
                r.packed_bytes, r.width, r.selected, r.minimum_selected_count,
                r.maximum_selected_count, r.removed,
                100.0 * r.removed as f64 / r.before.max(1) as f64,
                start.elapsed().as_secs_f64(),
            );
            if verbose || r.index % 10 == 0 {
                eprintln!(
                    "codec batch {}: {} symbols; selected {} pairs; {:.3}s; {} grammar rules (including helpers); {:.6}% symbols removed",
                    r.index,
                    r.rewritten,
                    r.selected,
                    start.elapsed().as_secs_f64(),
                    r.total_rules,
                    100.0 * r.removed as f64 / r.before.max(1) as f64,
                );
            }
        },
    )?;
    bre_pair::training_progress::phase("writing_model", training.rounds, 0, 0, "bytes");
    fs::File::create_new(output.path())
        .and_then(|mut f| f.write_all(&model.to_bytes().map_err(std::io::Error::other)?))
        .map_err(|e| e.to_string())?;
    let minimum_base = minimum;
    let minimum = training.minimum_count;
    let text = format!("{{\"format\":\"BRCODE01\",\"ai_vocabulary\":{},\"codec_rules\":{},\"huffman_codes\":{},\"training_bytes\":{},\"training_fields\":{},\"training_ai_tokens\":{},\"training_codec_symbols\":{},\"minimum_count\":{minimum},\"score\":\"{}\",\"threads\":{workers},\"rounds\":{},\"segmentation_seconds\":{:.6},\"merging_seconds\":{:.6},\"huffman_seconds\":{:.6},\"zero_usage_codes_with_weight_one\":{},\"max_packed_bytes\":{},\"registry_peak_slot_bytes\":{},\"stopped_at_rule_limit\":{},\"stopped_at_round_limit\":{},\"max_rules\":{},\"max_rounds\":{},\"seconds\":{:.6},\"peak_heap_bytes\":{},\"peak_private_bytes\":{}}}\n",model.dictionary.model.tokens.len(),model.dictionary.rules.len(),model.book.symbols.len(),model.training.bytes,model.training.fields,model.training.tokens,model.training.symbols,score.name(),training.rounds,training.segmentation_seconds,training.merging_seconds,training.huffman_seconds,training.zero_usage_codes,training.max_packed_bytes,training.registry_peak_slot_bytes,training.stopped_at_limit,training.stopped_at_round_limit,maximum.map_or_else(||"null".into(),|n|n.to_string()),max_rounds.map_or_else(||"null".into(),|n|n.to_string()),start.elapsed().as_secs_f64(),memory::peak_heap_bytes(),memory::process().peak_private);
    let document_counting = document_counting.as_str();
    let text = {
        let at = text.rfind('}').unwrap();
        format!("{},\"fill_waves\":{fill_waves},\"fill_min_count\":{fill_min_count},\"fill_min_count_base\":{fill_min_count_base},\"fill_min_count_per_gb\":{fill_min_count_per_gb}{}",&text[..at],&text[at..])
    };
    let at = text.rfind('}').unwrap();
    let extras=format!(",\"seed_pair_lookup\":\"fresh\",\"minimum_count_base\":{minimum_base},\"minimum_count_per_gb\":{minimum_count_per_gb},\"minimum_rule_documents\":{minimum_documents},\"document_counting\":\"{document_counting}\",\"document_filtered_candidates\":{},\"document_support_seconds\":{:.6},\"rules_created\":{rules_created},\"initial_helper_rules\":{}",training.document_filtered_candidates,training.document_support_seconds,training.helper_rules);
    let text = format!("{}{}{}", &text[..at], extras, &text[at..]);
    let at = text.rfind('}').unwrap();
    let cleanup = format!(",\"document_cleanup_seconds\":{:.6},\"document_removed_codes\":{},\"document_removed_used_codes\":{},\"document_cleanup_before_symbols\":{},\"document_cleanup_after_symbols\":{}",training.document_cleanup_seconds,training.document_removed_codes,training.document_removed_used_codes,training.document_cleanup_before_symbols,training.document_cleanup_after_symbols);
    let text = format!("{}{}{}", &text[..at], cleanup, &text[at..]);
    if let Some(report) = report {
        fs::File::create_new(report.path())
            .and_then(|mut f| f.write_all(text.as_bytes()))
            .map_err(|e| e.to_string())?;
        output.commit()?;
        report.commit()?;
    } else {
        output.commit()?;
    }
    print!("{text}");
    Ok(())
}
