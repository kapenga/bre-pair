//! Frozen 8K byte tokenizer: select on training data, then evaluate unchanged.
use bre_pair::{
    archive, token_stream,
    tokenizer::{fingerprint, Bpe, Model, Tokenizer},
    vocabulary::{
        self, Refinement, RefinementOptions, RefinementRound, SelectionMethod, SelectionOptions,
        Totals,
    },
};
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
    time::Instant,
};

const HELP: &str = "Frozen byte tokenizer\n\
tokenize-repair build ARCHIVE.brp TRAIN MODEL.brt [--vocab-size 8192] [--threads 4]\n\
  [--max-token-bytes 128] [--selection statistics|sampled]\n\
  [--sample-bytes 10000000 --support-weight 0.02] (sampled selection only)\n\
  [--min-token-uses 8] [--refine-rounds 64] [--augment-byte-pairs]\n\
  [--raw-byte-pair-support] [--report FILE]\n\
  [--document-index TRAIN.docs --min-token-documents 64]\n\
tokenize-repair refine ARCHIVE.brp TRAIN INITIAL.brt MODEL.brt [--threads 4]\n\
  [--max-token-bytes 128] [--min-token-uses 8] [--refine-rounds 64]\n\
  [--token-ceiling N] [--counts FILE] [--report FILE]\n\
  [--document-index TRAIN.docs --min-token-documents 8]\n\
tokenize-repair evaluate MODEL.brt INPUT [--threads 4] [--overlap-bytes N]\n\
  [--bpe-merges RANKS.brm] [--verify] [--report FILE] [--counts FILE]\n\
tokenize-repair ids MODEL.brt INPUT OUTPUT.ids [--bpe-merges RANKS.brm]\n\
tokenize-repair encode MODEL.brt INPUT OUTPUT.bts [--report FILE]\n\
tokenize-repair decode MODEL.brt INPUT.bts OUTPUT [--report FILE]\n\
tokenize-repair verify MODEL.brt INPUT.bts ORIGINAL [--report FILE]\n\
Fields are raw lines retaining delimiter bytes, followed by the same mergeable EOF.\n\
IDs 0..255 are bytes; 256 is EOF. No normalization or Unicode replacement.\n\
All output files must be new. Exact parsing minimizes tokens for a fixed model;\n\
vocabulary selection is a heuristic, trained only on TRAIN.\n";

fn write(path: &Path, bytes: &[u8]) -> Result<(), String> {
    fs::File::create_new(path)
        .and_then(|mut f| f.write_all(bytes))
        .map_err(|e| format!("{}: {e}", path.display()))
}
fn totals(t: Totals) -> String {
    format!(
        "{{\"bytes\":{},\"tokens\":{},\"fields\":{},\"bytes_per_token\":{:.9},\"byte_tokens\":{}}}",
        t.bytes,
        t.tokens,
        t.fields,
        t.bytes_per_token(),
        t.byte_tokens
    )
}
fn refinement_round(r: &RefinementRound) -> String {
    format!("{{\"event\":\"refine\",\"index\":{},\"removed\":{},\"added\":{},\"active\":{},\"corpus\":{},\"seconds\":{:.6}}}",
        r.index, r.removed, r.added, r.active, totals(r.totals), r.seconds)
}
fn refinement_report(r: &Refinement, options: &RefinementOptions) -> String {
    let mut result=format!("{{\"minimum_uses\":{},\"minimum_learned_uses\":{},\"before\":{},\"after\":{},\"rounds\":[{}]}}",
        options.minimum_uses, r.minimum_learned_uses.map_or("null".into(), |n| n.to_string()),
        totals(r.before), totals(r.after), r.rounds.iter().map(refinement_round).collect::<Vec<_>>().join(","));
    if let Some(minimum) = r.minimum_document_groups {
        result.pop();
        result.push_str(&format!(",\"minimum_training_document_groups\":{minimum},\"minimum_learned_document_groups_lower_bound\":{}}}",r.minimum_learned_document_groups.map_or("null".into(),|n|n.to_string())));
    }
    result
}
fn value<'a>(args: &'a [String], at: &mut usize) -> Result<&'a str, String> {
    *at += 1;
    args.get(*at)
        .map(String::as_str)
        .ok_or("missing option value".into())
}
pub fn run(args: Vec<String>, progress: bool) -> Result<(), String> {
    if args.is_empty() || args[0] == "--help" {
        print!("{HELP}");
        return Ok(());
    }
    match args[0].as_str() {
        "build" => {
            if args.len() < 4 {
                return Err(HELP.into());
            }
            let mut options = SelectionOptions::default();
            let mut sampling_options = false;
            let _progress = bre_pair::training_progress::enable();
            let mut refinement = RefinementOptions::default();
            let mut max_bytes = 128;
            let mut augment_byte_pairs = false;
            let mut raw_byte_pair_support = false;
            let mut document_index = None;
            let mut minimum_documents = None;
            let mut candidate_document_support = "stream";
            let mut report = None;
            let mut at = 4;
            while at < args.len() {
                match args[at].as_str() {
                    "--selection" => {
                        options.method = match value(&args, &mut at)? {
                            "statistics" => SelectionMethod::Statistics,
                            "sampled" => SelectionMethod::Sampled,
                            _ => return Err("selection must be statistics or sampled".into()),
                        };
                    }
                    "--candidate-document-support" => {
                        candidate_document_support = value(&args, &mut at)?;
                        if !matches!(candidate_document_support, "stream" | "grammar") {
                            return Err(
                                "candidate document support must be stream or grammar".into()
                            );
                        }
                    }
                    "--augment-byte-pairs" => augment_byte_pairs = true,
                    "--raw-byte-pair-support" => raw_byte_pair_support = true,
                    "--document-index" => {
                        document_index = Some(PathBuf::from(value(&args, &mut at)?))
                    }
                    "--min-token-documents" => {
                        minimum_documents = Some(
                            value(&args, &mut at)?
                                .parse::<usize>()
                                .map_err(|_| "invalid document minimum")?,
                        )
                    }
                    "--vocab-size" => {
                        options.vocabulary = value(&args, &mut at)?
                            .parse()
                            .map_err(|_| "invalid vocabulary")?
                    }
                    "--threads" => {
                        options.workers = value(&args, &mut at)?
                            .parse()
                            .map_err(|_| "invalid workers")?
                    }
                    "--max-token-bytes" => {
                        max_bytes = value(&args, &mut at)?
                            .parse()
                            .map_err(|_| "invalid max token bytes")?
                    }
                    "--sample-bytes" => {
                        sampling_options = true;
                        options.sample_bytes = value(&args, &mut at)?
                            .parse()
                            .map_err(|_| "invalid sample bytes")?
                    }
                    "--support-weight" => {
                        sampling_options = true;
                        options.support_weight = value(&args, &mut at)?
                            .parse()
                            .map_err(|_| "invalid support weight")?
                    }
                    "--min-token-uses" => {
                        refinement.minimum_uses = value(&args, &mut at)?
                            .parse()
                            .map_err(|_| "invalid minimum token uses")?
                    }
                    "--refine-rounds" => {
                        refinement.max_rounds = value(&args, &mut at)?
                            .parse()
                            .map_err(|_| "invalid refinement round limit")?
                    }
                    "--report" => report = Some(PathBuf::from(value(&args, &mut at)?)),
                    _ => return Err(format!("unknown option {}", args[at])),
                }
                at += 1;
            }
            if sampling_options && options.method != SelectionMethod::Sampled {
                return Err(
                    "--sample-bytes and --support-weight require --selection sampled".into(),
                );
            }
            if augment_byte_pairs && max_bytes < 2 {
                return Err("byte-pair augmentation requires max-token-bytes >=2".into());
            }
            if raw_byte_pair_support && !augment_byte_pairs {
                return Err("raw-byte-pair-support requires augment-byte-pairs".into());
            }
            if Path::new(&args[3]).exists() || report.as_ref().is_some_and(|p| p.exists()) {
                return Err("output exists".into());
            }
            let started = Instant::now();
            let document_policy = match (document_index, minimum_documents) {
                (Some(index), minimum) => Some(bre_pair::document_support::Policy {
                    index,
                    minimum_groups: minimum.unwrap_or(8),
                }),
                (None, Some(_)) => {
                    return Err("--min-token-documents requires a training --document-index".into())
                }
                (None, None) => None,
            };
            if document_policy
                .as_ref()
                .is_some_and(|p| !(1..=64).contains(&p.minimum_groups))
            {
                return Err("minimum token documents must be between 1 and 64".into());
            }
            if document_policy.is_some() && augment_byte_pairs {
                return Err("document-boundary vocabulary selection requires grammar candidates; byte-pair augmentation is not supported".into());
            }
            if candidate_document_support == "grammar" && document_policy.is_none() {
                return Err(
                    "grammar candidate document support requires a training --document-index"
                        .into(),
                );
            }
            let source = archive::decode(&fs::read(&args[1]).map_err(|e| e.to_string())?)?;
            let source_bytes = source.input_bytes;
            if fs::metadata(&args[2]).map_err(|e| e.to_string())?.len() != source_bytes {
                return Err("training size differs from candidate archive source".into());
            }
            vocabulary::verify_training_source(&source, Path::new(&args[2]))?;
            let cleanup = document_policy
                .as_ref()
                .filter(|_| candidate_document_support == "stream")
                .map(|policy| {
                    let mut protected = vec![false; source.grammar.alphabet_size()];
                    protected[..257].fill(true);
                    bre_pair::document_support::Cleanup::plan(&source, policy, &protected)
                })
                .transpose()?;
            let candidate_support = document_policy
                .as_ref()
                .filter(|_| candidate_document_support == "grammar")
                .map(|policy| bre_pair::document_support::CandidateSupport::plan(&source, policy))
                .transpose()?;
            let cleanup_json=cleanup.as_ref().map_or("null".into(),|p|format!("{{\"removed_rules\":{},\"removed_used_rules\":{},\"before_symbols\":{},\"after_symbols\":{},\"witness_slots\":{},\"seconds\":{:.6}}}",p.removed_rules,p.removed_used_rules,p.before_symbols,p.after_symbols,p.witness_slots,p.seconds));
            let candidate_support_json = candidate_support.as_ref().map_or("null".into(), |p| format!(
                "{{\"included_rules\":{},\"excluded_rules\":{},\"witness_slots\":{},\"seconds\":{:.6}}}",
                p.included[257..].iter().filter(|&&b| b).count(),
                p.included[257..].iter().filter(|&&b| !b).count(), p.witness_slots, p.seconds));
            let included = cleanup
                .as_ref()
                .map(|p| p.active.as_slice())
                .or_else(|| candidate_support.as_ref().map(|p| p.included.as_slice()));
            let (mut pool, mut support) = Model::candidates_filtered(&source, max_bytes, included)?;
            drop(cleanup);
            drop(candidate_support);
            drop(source);
            let augmented = if augment_byte_pairs {
                vocabulary::augment_byte_pairs_with_prior(
                    &mut pool,
                    &mut support,
                    Path::new(&args[2]),
                    refinement.minimum_uses,
                    raw_byte_pair_support,
                )?
            } else {
                0
            };
            let candidates = pool.tokens.len();
            bre_pair::training_progress::phase(
                "vocabulary_selection",
                0,
                candidates as u64,
                candidates as u64,
                "candidates_remaining",
            );
            if progress {
                println!("{{\"event\":\"candidates\",\"vocabulary\":{candidates}}}");
            }
            let mut phases = Vec::new();
            refinement.workers = options.workers;
            let selected = vocabulary::select_refined_with_documents(
                pool,
                &support,
                Path::new(&args[2]),
                &options,
                &refinement,
                document_policy.as_ref(),
                (
                    |r| {
                        bre_pair::training_progress::phase(
                            "vocabulary_selection",
                            0,
                            r.after as u64,
                            candidates as u64,
                            "candidates_remaining",
                        );
                        let row=format!("{{\"event\":\"prune\",\"selection_method\":\"{}\",\"before\":{},\"after\":{},\"stride\":{},\"corpus\":{},\"seconds\":{:.6}}}",r.method.name(),r.before,r.after,r.stride.map_or("null".into(),|n|n.to_string()),r.totals.map_or("null".into(),totals),r.seconds);
                        if progress {
                            println!("{row}");
                        }
                        phases.push(row);
                    },
                    |r| {
                        bre_pair::training_progress::phase(
                            "vocabulary_refinement",
                            0,
                            r.active as u64,
                            options.vocabulary as u64,
                            "tokens",
                        );
                        if progress {
                            println!("{}", refinement_round(r));
                        }
                    },
                ),
            )?;
            let refined = refinement_report(&selected, &refinement);
            let bytes = selected.model.to_bytes();
            let fingerprint = fingerprint(&bytes);
            let tokenizer = Tokenizer::new(selected.model)?;
            let document_score = document_policy
                .as_ref()
                .map(|policy| {
                    bre_pair::document_support::scan(
                        Path::new(&args[2]),
                        &tokenizer,
                        None,
                        options.workers,
                        policy,
                    )
                })
                .transpose()?;
            let final_score = if let Some(found) = document_score {
                if found.groups[257..]
                    .iter()
                    .any(|&n| n < document_policy.as_ref().unwrap().minimum_groups as u32)
                {
                    return Err("exported vocabulary violates document support".into());
                }
                found.scan
            } else {
                vocabulary::scan(
                    Path::new(&args[2]),
                    &tokenizer,
                    None,
                    None,
                    options.workers,
                    1,
                    0,
                    true,
                    true,
                )?
            };
            if final_score.uses[257..]
                .iter()
                .any(|&n| n < refinement.minimum_uses)
                || final_score.all.tokens != selected.after.tokens
            {
                return Err("exported model differs from refined counts".into());
            }
            write(Path::new(&args[3]), &bytes)?;
            let sampled = options.method == SelectionMethod::Sampled;
            let sample_bytes = if sampled {
                options.sample_bytes.to_string()
            } else {
                "null".into()
            };
            let support_weight = if sampled {
                options.support_weight.to_string()
            } else {
                "null".into()
            };
            let result=format!("{{\"vocabulary\":{},\"candidates\":{},\"selection_method\":\"{}\",\"augmented_byte_pairs\":{},\"raw_byte_pair_support\":{},\"support_weight\":{},\"max_token_bytes\":{},\"sample_bytes\":{},\"model_fingerprint\":{},\"training\":{},\"seconds\":{:.6},\"round_trip_verified\":true,\"source_verified\":true,\"refinement\":{},\"phases\":[{}]}}\n",
                tokenizer.model().tokens.len(),candidates,options.method.name(),augmented,raw_byte_pair_support,support_weight,max_bytes,sample_bytes,fingerprint,totals(final_score.all),started.elapsed().as_secs_f64(),refined,phases.join(","));
            let at = result.rfind('}').unwrap();
            let result = format!(
                "{},\"document_cleanup\":{},\"candidate_document_support\":\"{}\",\"candidate_document_qualification\":{}{}",
                &result[..at],
                cleanup_json,
                if document_policy.is_some() { candidate_document_support } else { "none" },
                candidate_support_json,
                &result[at..]
            );
            if let Some(path) = report {
                write(&path, result.as_bytes())?;
            }
            println!("{result}");
        }
        "refine" => {
            if args.len() < 5 {
                return Err(HELP.into());
            }
            let mut options = RefinementOptions::default();
            let mut max_bytes = 128;
            let mut report = None;
            let mut counts = None;
            let mut token_ceiling = None;
            let mut document_index = None;
            let mut minimum_documents = None;
            let mut at = 5;
            while at < args.len() {
                match args[at].as_str() {
                    "--threads" => {
                        options.workers = value(&args, &mut at)?
                            .parse()
                            .map_err(|_| "invalid workers")?
                    }
                    "--min-token-uses" => {
                        options.minimum_uses = value(&args, &mut at)?
                            .parse()
                            .map_err(|_| "invalid minimum token uses")?
                    }
                    "--refine-rounds" => {
                        options.max_rounds = value(&args, &mut at)?
                            .parse()
                            .map_err(|_| "invalid refinement round limit")?
                    }
                    "--max-token-bytes" => {
                        max_bytes = value(&args, &mut at)?
                            .parse()
                            .map_err(|_| "invalid maximum token length")?
                    }
                    "--token-ceiling" => {
                        token_ceiling = Some(
                            value(&args, &mut at)?
                                .parse()
                                .map_err(|_| "invalid token ceiling")?,
                        )
                    }
                    "--document-index" => {
                        document_index = Some(PathBuf::from(value(&args, &mut at)?))
                    }
                    "--min-token-documents" => {
                        minimum_documents = Some(
                            value(&args, &mut at)?
                                .parse::<usize>()
                                .map_err(|_| "invalid minimum token documents")?,
                        )
                    }
                    "--counts" => counts = Some(PathBuf::from(value(&args, &mut at)?)),
                    "--report" => report = Some(PathBuf::from(value(&args, &mut at)?)),
                    _ => return Err(format!("unknown option {}", args[at])),
                }
                at += 1;
            }
            if Path::new(&args[4]).exists()
                || report
                    .as_ref()
                    .is_some_and(|p| p.exists() || p == Path::new(&args[4]))
                || counts.as_ref().is_some_and(|p| {
                    p.exists() || p == Path::new(&args[4]) || report.as_ref() == Some(p)
                })
            {
                return Err("output exists or output paths coincide".into());
            }
            let document_policy = match (document_index, minimum_documents) {
                (Some(index), minimum) => Some(bre_pair::document_support::Policy {
                    index,
                    minimum_groups: minimum.unwrap_or(8),
                }),
                (None, Some(_)) => {
                    return Err("--min-token-documents requires a training --document-index".into())
                }
                (None, None) => None,
            };
            if document_policy
                .as_ref()
                .is_some_and(|p| !(1..=64).contains(&p.minimum_groups))
            {
                return Err("minimum token documents must be between 1 and 64".into());
            }
            let started = Instant::now();
            let source = archive::decode(&fs::read(&args[1]).map_err(|e| e.to_string())?)?;
            vocabulary::verify_training_source(&source, Path::new(&args[2]))?;
            let (pool, _) = Model::candidates(&source, max_bytes)?;
            drop(source);
            let initial = Model::from_bytes(&fs::read(&args[3]).map_err(|e| e.to_string())?)?;
            let selected = vocabulary::refine_with_documents(
                pool,
                &initial,
                Path::new(&args[2]),
                &options,
                token_ceiling,
                document_policy.as_ref(),
                |r| {
                    if progress {
                        println!("{}", refinement_round(r));
                    }
                },
            )?;
            let refined = refinement_report(&selected, &options);
            let tokenizer = Tokenizer::new(selected.model)?;
            let document_score = document_policy
                .as_ref()
                .map(|policy| {
                    bre_pair::document_support::scan(
                        Path::new(&args[2]),
                        &tokenizer,
                        None,
                        options.workers,
                        policy,
                    )
                })
                .transpose()?;
            let (scored, final_document_groups) = if let Some(found) = document_score {
                if found.groups[257..]
                    .iter()
                    .any(|&n| n < document_policy.as_ref().unwrap().minimum_groups as u32)
                {
                    return Err("exported model violates training-document support".into());
                }
                (found.scan, Some(found.groups))
            } else {
                (
                    vocabulary::scan(
                        Path::new(&args[2]),
                        &tokenizer,
                        None,
                        None,
                        options.workers,
                        1,
                        0,
                        true,
                        true,
                    )?,
                    None,
                )
            };
            if scored.uses[257..].iter().any(|&n| n < options.minimum_uses)
                || scored.all.tokens != selected.after.tokens
            {
                return Err("exported model differs from refined counts".into());
            }
            write(Path::new(&args[4]), &tokenizer.model().to_bytes())?;
            if let Some(path) = counts {
                let uses = scored
                    .uses
                    .iter()
                    .map(u64::to_string)
                    .collect::<Vec<_>>()
                    .join(",");
                let document_fields=final_document_groups.as_ref().map_or(String::new(),|groups|format!(",\"document_group_saturation\":{},\"document_groups_lower_bound\":[{}]",document_policy.as_ref().unwrap().minimum_groups,groups.iter().map(u32::to_string).collect::<Vec<_>>().join(",")));
                write(
                    &path,
                    format!(
                        "{{\"model_fingerprint\":{},\"all\":{},\"uses\":[{}]{}}}\n",
                        fingerprint(&tokenizer.model().to_bytes()),
                        totals(scored.all),
                        uses,
                        document_fields
                    )
                    .as_bytes(),
                )?;
            }
            let result = format!("{{\"vocabulary\":{},\"training\":{},\"refinement\":{},\"source_verified\":true,\"round_trip_verified\":true,\"seconds\":{:.6}}}\n",
                tokenizer.model().tokens.len(), totals(scored.all), refined, started.elapsed().as_secs_f64());
            if let Some(path) = report {
                write(&path, result.as_bytes())?;
            }
            println!("{result}");
        }
        "encode" | "decode" | "verify" => {
            if args.len() < 4 {
                return Err(HELP.into());
            }
            let mut report = None;
            let mut at = 4;
            while at < args.len() {
                if args[at] != "--report" {
                    return Err(format!("unknown option {}", args[at]));
                }
                report = Some(PathBuf::from(value(&args, &mut at)?));
                at += 1;
            }
            if report.as_ref().is_some_and(|p| p.exists()) {
                return Err("report exists".into());
            }
            let model = Model::from_bytes(&fs::read(&args[1]).map_err(|e| e.to_string())?)?;
            let started = Instant::now();
            let header = match args[0].as_str() {
                "encode" => token_stream::encode_file(
                    &Tokenizer::new(model)?,
                    Path::new(&args[2]),
                    Path::new(&args[3]),
                )?,
                "decode" => {
                    token_stream::decode_file(&model, Path::new(&args[2]), Path::new(&args[3]))?
                }
                _ => token_stream::verify_file(&model, Path::new(&args[2]), Path::new(&args[3]))?,
            };
            let result=format!("{{\"operation\":\"{}\",\"bytes\":{},\"fields\":{},\"tokens\":{},\"vocabulary\":{},\"bits_per_token\":{},\"payload_bytes\":{},\"stream_bytes\":{},\"seconds\":{:.6}}}\n",
                args[0],header.bytes,header.fields,header.tokens,header.vocabulary,header.width,header.payload_bytes()?,token_stream::HEADER_BYTES+header.payload_bytes()?,started.elapsed().as_secs_f64());
            if let Some(path) = report {
                write(&path, result.as_bytes())?;
            }
            println!("{result}");
        }
        "evaluate" | "ids" => {
            let ids_mode = args[0] == "ids";
            if args.len() < if ids_mode { 4 } else { 3 } {
                return Err(HELP.into());
            }
            let model = Model::from_bytes(&fs::read(&args[1]).map_err(|e| e.to_string())?)?;
            let mut workers = 4;
            let mut overlap = 0;
            let mut ranks = None;
            let mut report = None;
            let mut verify = false;
            let mut counts = None;
            let mut at = if ids_mode { 4 } else { 3 };
            while at < args.len() {
                match args[at].as_str() {
                    "--threads" => {
                        workers = value(&args, &mut at)?
                            .parse()
                            .map_err(|_| "invalid workers")?
                    }
                    "--overlap-bytes" => {
                        overlap = value(&args, &mut at)?
                            .parse()
                            .map_err(|_| "invalid overlap")?
                    }
                    "--bpe-merges" => ranks = Some(PathBuf::from(value(&args, &mut at)?)),
                    "--report" => report = Some(PathBuf::from(value(&args, &mut at)?)),
                    "--counts" => counts = Some(PathBuf::from(value(&args, &mut at)?)),
                    "--verify" => verify = true,
                    _ => return Err(format!("unknown option {}", args[at])),
                }
                at += 1;
            }
            let bpe = ranks
                .map(|path| {
                    fs::read(path)
                        .map_err(|e| e.to_string())
                        .and_then(|bytes| Bpe::from_bytes(model.clone(), &bytes))
                })
                .transpose()?;
            let tokenizer = Tokenizer::new(model)?;
            if ids_mode {
                if counts.is_some() {
                    return Err("--counts is available with evaluate".into());
                }
                use std::io::{BufRead, BufReader, BufWriter};
                let mut input =
                    BufReader::new(fs::File::open(&args[2]).map_err(|e| e.to_string())?);
                let mut output =
                    BufWriter::new(fs::File::create_new(&args[3]).map_err(|e| e.to_string())?);
                let mut line = Vec::new();
                while input
                    .read_until(b'\n', &mut line)
                    .map_err(|e| e.to_string())?
                    > 0
                {
                    let ids = bpe
                        .as_ref()
                        .map_or_else(|| tokenizer.encode(&line), |b| b.encode(&line));
                    for (i, id) in ids.iter().enumerate() {
                        if i > 0 {
                            write!(output, " ").map_err(|e| e.to_string())?;
                        }
                        write!(output, "{id}").map_err(|e| e.to_string())?;
                    }
                    writeln!(output).map_err(|e| e.to_string())?;
                    line.clear();
                }
                output.flush().map_err(|e| e.to_string())?;
            } else {
                if report.as_ref().is_some_and(|p| p.exists())
                    || counts.as_ref().is_some_and(|p| p.exists())
                    || report.as_ref().is_some_and(|p| Some(p) == counts.as_ref())
                {
                    return Err("output exists or output paths coincide".into());
                }
                let scored = vocabulary::scan(
                    Path::new(&args[2]),
                    &tokenizer,
                    bpe.as_ref(),
                    None,
                    workers,
                    1,
                    overlap,
                    counts.is_some(),
                    verify,
                )?;
                if let Some(path) = counts {
                    let uses = scored
                        .uses
                        .iter()
                        .map(u64::to_string)
                        .collect::<Vec<_>>()
                        .join(",");
                    let result = format!("{{\"vocabulary\":{},\"model_fingerprint\":{},\"scope\":\"all input fields\",\"all\":{},\"uses\":[{}]}}\n",
                        tokenizer.model().tokens.len(), fingerprint(&tokenizer.model().to_bytes()), totals(scored.all), uses);
                    write(&path, result.as_bytes())?;
                }
                let result=format!("{{\"vocabulary\":{},\"algorithm\":\"{}\",\"all\":{},\"heldout\":{},\"heldout_start\":{},\"seconds\":{:.6},\"round_trip_verified\":{verify}}}\n",
                    tokenizer.model().tokens.len(),if bpe.is_some(){"bpe-rank"}else{"exact"},totals(scored.all),totals(scored.heldout),scored.all.bytes-scored.heldout.bytes,scored.seconds);
                if let Some(path) = report {
                    write(&path, result.as_bytes())?;
                }
                println!("{result}");
            }
        }
        _ => return Err(HELP.into()),
    }
    Ok(())
}
