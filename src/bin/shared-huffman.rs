//! Variable-arity shared dictionary/data canonical Huffman experiment.
use bre_pair::{
    memory,
    shared_huffman::{self, Options, Stats},
};
use std::{fs, io::BufReader, path::Path, time::Instant};

#[global_allocator]
static ALLOCATOR: memory::Allocator = memory::Allocator;

fn stats(s: &Stats) -> String {
    format!(concat!("{{\"format\":\"{}\",\"archive_bytes\":{},\"dictionary_bytes_including_table\":{},",
        "\"codebook_bytes\":{},\"descriptor_bytes\":{},\"huffman_rule_lengths\":{},\"reference_bits\":{},\"references\":{},",
        "\"payload_bits\":{},\"leaves\":{},\"rules\":{},\"nonbinary_rules\":{},",
        "\"original_bytes\":{},\"fields\":{},\"terminal_tokens\":{},\"codec_symbols\":{},\"blocks\":{}}}"),
        s.format(),s.archive_bytes,s.dictionary_bytes,s.codebook_bytes,s.descriptor_bytes,s.huffman_rule_lengths,s.reference_bits,s.references,
        s.payload_bits,s.leaves,s.rules,s.nonbinary_rules,s.totals.bytes,s.totals.fields,s.totals.tokens,s.totals.symbols,s.blocks)
}
fn run() -> Result<(), String> {
    let mut args: Vec<_> = std::env::args().skip(1).collect();
    let mut budget = 8_000_000_000usize;
    if let Some(at) = args.iter().position(|a| a == "--memory") {
        budget = memory::parse_bytes(args.get(at + 1).ok_or("missing memory limit")?)?;
        args.drain(at..at + 2);
    }
    let usage = "shared-huffman optimize SOURCE.brh NEW_DIRECTORY [--rounds 2 --max-children 256 --max-parent-visits 4096 --min-gain-bits 1 --block-symbols 1048576 --rule-lengths auto|varint]\n  [--cost-model shannon|huffman-local|huffman-corrected --shannon-warmup 2 --refit-slices 8 --audit-pruning true --huffman-arity-cost true] (research policies)\nshared-huffman repack-lengths INPUT.brhv NEW_OUTPUT [--rule-lengths auto|varint]\nshared-huffman project-tables INPUT.brhv (read-only size projection; no new archive)\nshared-huffman project-arities INPUT.brhv (read-only run-descriptor projection)\nshared-huffman decode INPUT.brhv NEW_OUTPUT [--ids]\nshared-huffman inspect INPUT.brhv";
    let command = args.first().map(String::as_str).unwrap_or("");
    let started = Instant::now();
    let _limit = memory::Limit::set(budget, 1)?;
    match command {
        "repack-runs" if args.len() == 3 => {
            let report =
                shared_huffman::repack_arity_runs(Path::new(&args[1]), Path::new(&args[2]))?;
            println!("{{\"format\":\"{}\",\"before_bytes\":{},\"archive_bytes\":{},\"before_descriptor_bytes\":{},\"descriptor_bytes\":{},\"payload_unchanged\":true}}",report.after.format(),report.before.archive_bytes,report.after.archive_bytes,report.before.descriptor_bytes,report.after.descriptor_bytes);
        }
        "project-arities" if args.len() == 2 => {
            println!("{}", shared_huffman::project_arities(Path::new(&args[1]))?);
        }
        "project-tables" if args.len() == 2 => {
            let projection = shared_huffman::project_tables(Path::new(&args[1]))?;
            println!("{}", projection.json());
        }
        "optimize" if args.len() >= 3 => {
            let mut options = Options::default();
            let mut audit_pruning = false;
            for pair in args[3..].chunks(2) {
                let value = pair.get(1).ok_or("missing shared Huffman option value")?;
                match pair[0].as_str() {
                    "--rounds" => options.rounds = value.parse().map_err(|_| "invalid rounds")?,
                    "--rule-lengths" => options.rule_lengths = value.parse()?,
                    "--huffman-arity-cost" => {
                        options.huffman_arity_cost = value
                            .parse()
                            .map_err(|_| "invalid Huffman arity cost flag")?
                    }
                    "--audit-pruning" => {
                        audit_pruning = value.parse().map_err(|_| "invalid pruning audit flag")?
                    }
                    "--cost-model" => options.cost_model = value.parse()?,
                    "--shannon-warmup" => {
                        options.shannon_warmup = value.parse().map_err(|_| "invalid warmup")?
                    }
                    "--refit-slices" => {
                        options.refit_slices = value.parse().map_err(|_| "invalid refit slices")?
                    }
                    "--max-children" => {
                        options.max_children =
                            value.parse().map_err(|_| "invalid maximum children")?
                    }
                    "--max-parent-visits" => {
                        options.max_parent_visits =
                            value.parse().map_err(|_| "invalid parent visit limit")?
                    }
                    "--min-gain-bits" => {
                        options.minimum_gain_bits =
                            value.parse().map_err(|_| "invalid minimum gain")?
                    }
                    "--block-symbols" => {
                        options.block_symbols = value.parse().map_err(|_| "invalid block size")?
                    }
                    _ => return Err(usage.into()),
                }
            }
            let (report, audit) = if audit_pruning {
                shared_huffman::optimize_audited(Path::new(&args[1]), Path::new(&args[2]), options)?
            } else {
                (
                    shared_huffman::optimize(Path::new(&args[1]), Path::new(&args[2]), options)?,
                    None,
                )
            };
            let audit_json = audit.map_or_else(|| "null".to_owned(), |a| format!(concat!(
                "{{\"live_rules\":{},\"data_once_unreferenced_rules\":{},",
                "\"data_once_unreferenced_over_child_limit\":{},\"max_arity_data_once_unreferenced\":{},",
                "\"single_use_helpers\":{},\"single_use_helpers_over_parent_walk_limit\":{},",
                "\"over_child_limit_rules\":{},\"over_reference_limit_rules\":{}}}"),
                a.live_rules,a.data_once_unreferenced_rules,a.data_once_unreferenced_over_child_limit,
                a.max_arity_data_once_unreferenced,a.single_use_helpers,
                a.single_use_helpers_over_parent_walk_limit,a.over_child_limit_rules,a.over_reference_limit_rules));
            let variants: Vec<_> = report.variants.iter().map(|v|format!("{{\"name\":\"{}\",\"candidates\":{},\"removed_this_step\":{},\"seconds\":{:.6},\"stats\":{}}}",v.name,v.candidates,v.removed,v.seconds,stats(&v.stats))).collect();
            let p = memory::process();
            let json = format!(concat!("{{\n  \"format\":\"shared-canonical-huffman\",\n  \"shared_dictionary_and_data_table\":true,\n",
                "  \"input_format_unchanged\":true,\n  \"original_archive_bytes\":{},\n  \"selected\":\"{}\",\n",
                "  \"pruning_objective\":\"estimated coding cost plus {}; actual complete checkpoints select winner\",\n",
                "  \"options\":{{\"rounds\":{},\"max_children\":{},\"max_parent_visits\":{},\"minimum_gain_bits\":{},\"block_symbols\":{},\"cost_model\":\"{}\",\"shannon_warmup\":{},\"refit_slices\":{},\"rule_lengths\":\"{}\",\"huffman_arity_cost\":{}}},\n",
                "  \"pruning_audit\":{},\n  \"variants\":[{}],\n  \"seconds\":{:.6},\n  \"peak_heap_bytes\":{},\n  \"peak_private_bytes\":{}\n}}\n"),
                report.original_bytes,report.selected,if options.huffman_arity_cost { "periodically fitted canonical arity costs with table/padding deltas and auto fallback" } else { "unchanged varint arity cost" },options.rounds,options.max_children,options.max_parent_visits,options.minimum_gain_bits,options.block_symbols,
                options.cost_model.name(),options.shannon_warmup,options.refit_slices,options.rule_lengths.name(),options.huffman_arity_cost,
                audit_json,variants.join(","),started.elapsed().as_secs_f64(),memory::peak_heap_bytes(),p.peak_private);
            fs::write(Path::new(&args[2]).join("report.json"), &json).map_err(|e| e.to_string())?;
            print!("{json}");
        }
        "repack-lengths" if args.len() == 3 || (args.len() == 5 && args[3] == "--rule-lengths") => {
            let coding = if args.len() == 5 {
                args[4].parse()?
            } else {
                shared_huffman::RuleLengths::Auto
            };
            let report = shared_huffman::repack_rule_lengths(
                Path::new(&args[1]),
                Path::new(&args[2]),
                coding,
            )?;
            println!("{{\"format\":\"{}\",\"before_bytes\":{},\"archive_bytes\":{},\"before_descriptor_bytes\":{},\"descriptor_bytes\":{},\"payload_unchanged\":true,\"seconds\":{:.6}}}",report.after.format(),report.before.archive_bytes,report.after.archive_bytes,report.before.descriptor_bytes,report.after.descriptor_bytes,started.elapsed().as_secs_f64());
        }
        "decode" if args.len() == 3 || (args.len() == 4 && args[3] == "--ids") => {
            let totals = shared_huffman::decode_file(
                Path::new(&args[1]),
                Path::new(&args[2]),
                args.len() == 4,
            )?;
            println!("{{\"original_bytes\":{},\"fields\":{},\"terminal_tokens\":{},\"codec_symbols\":{},\"seconds\":{:.6}}}",totals.bytes,totals.fields,totals.tokens,totals.symbols,started.elapsed().as_secs_f64());
        }
        "inspect" if args.len() == 2 => {
            let reader = shared_huffman::Reader::new(BufReader::new(
                fs::File::open(&args[1]).map_err(|e| e.to_string())?,
            ))?;
            println!(
                "{{\"format\":\"{}\",\"ai_tokens\":{},\"stats\":{},\"initialization_seconds\":{:.6}}}",
                reader.stats.format(),
                reader.ai_tokens,
                stats(&reader.stats),
                started.elapsed().as_secs_f64()
            );
        }
        _ => return Err(usage.into()),
    }
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("shared-huffman: {e}");
        std::process::exit(1);
    }
}
