//! Finite research comparison of reparsing during BRe-Pair training.
use bre_pair::{
    codec::{periodic, TrainOptions},
    engine::Config,
    memory,
    scoring::Model as Score,
    tokenizer::Model,
};
use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    path::Path,
    time::Instant,
};
#[global_allocator]
static ALLOCATOR: memory::Allocator = memory::Allocator;

fn run() -> Result<(), String> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() < 3 {
        return Err("periodic-codec VOCAB.brt TRAINING NEW_DIRECTORY [--every 64] [--max-rounds 192] [--threads 4] [--min-count 8] [--mode reparse|recount] [--memory 2GB]".into());
    }
    let (mut every, mut cap, mut workers, mut minimum, mut reparse) = (64, 192, 4, 8, true);
    let mut budget = 2_000_000_000;
    for pair in args[3..].chunks(2) {
        let value = pair.get(1).ok_or("missing periodic option")?;
        match pair[0].as_str() {
            "--every" => every = value.parse().map_err(|_| "invalid interval")?,
            "--max-rounds" => cap = value.parse().map_err(|_| "invalid round cap")?,
            "--threads" => workers = value.parse().map_err(|_| "invalid workers")?,
            "--min-count" => minimum = value.parse().map_err(|_| "invalid minimum")?,
            "--memory" => budget = memory::parse_bytes(value)?,
            "--mode" => {
                reparse = match value.as_str() {
                    "reparse" => true,
                    "recount" => false,
                    _ => return Err("mode must be reparse or recount".into()),
                }
            }
            _ => return Err("unknown periodic option".into()),
        }
    }
    if cap == 0 || !(1..=256).contains(&workers) || minimum < 2 {
        return Err("invalid periodic limits".into());
    }
    let limit = memory::Limit::set(budget, workers)?;
    let start = Instant::now();
    let vocabulary = Model::from_bytes(&fs::read(&args[0]).map_err(|e| e.to_string())?)?;
    let output = Path::new(&args[2]);
    fs::create_dir(output).map_err(|e| e.to_string())?;
    let mut rounds =
        BufWriter::new(File::create_new(output.join("rounds.csv")).map_err(|e| e.to_string())?);
    writeln!(rounds, "pass,symbols,selected,new_rules,rules,seconds").map_err(|e| e.to_string())?;
    let mut rows = Vec::new();
    let config = Config {
        buffer_bytes: limit.heap,
        minimum_count: minimum,
        score: Score::WeightedPmi,
        ..Default::default()
    };
    let result = periodic::train(
        &vocabulary,
        Path::new(&args[1]),
        &config,
        workers,
        &TrainOptions {
            max_rounds: Some(cap),
            ..Default::default()
        },
        every,
        reparse,
        |index, r| {
            if index % 10 == 0 {
                eprintln!(
                    "pass {index}: {} roots; {} rules; {:.2}s",
                    r.rewritten,
                    r.total_rules,
                    start.elapsed().as_secs_f64()
                );
            }
            rows.push(format!(
                "{index},{},{},{},{},{:.6}\n",
                r.rewritten, r.selected, r.new_rules, r.total_rules, r.seconds
            ));
        },
        |e| {
            eprintln!(
                "reparse after {}: {} -> {} roots; {} -> {} temporary Huffman bits; {:.3}s",
                e.after_pass,
                e.before_symbols,
                e.after_symbols,
                e.before_bits,
                e.after_bits,
                e.seconds
            )
        },
    )?;
    for row in rows {
        rounds
            .write_all(row.as_bytes())
            .map_err(|e| e.to_string())?;
    }
    rounds.flush().map_err(|e| e.to_string())?;
    fs::write(output.join("model.brc"), result.model.to_bytes()?).map_err(|e| e.to_string())?;
    result.write_archive(&output.join("training.brh"))?;
    let events:Vec<_>=result.events.iter().map(|e|format!("{{\"after_pass\":{},\"before_symbols\":{},\"after_symbols\":{},\"before_bits\":{},\"after_bits\":{},\"changed_fields\":{},\"maximum_field_tokens\":{},\"seconds\":{:.6}}}",e.after_pass,e.before_symbols,e.after_symbols,e.before_bits,e.after_bits,e.changed_fields,e.maximum_field_tokens,e.seconds)).collect();
    let json=format!(concat!("{{\n  \"training_only\":true,\n  \"interval\":{},\n  \"reparse_enabled\":{},\n  \"maximum_rounds\":{},\n",
        "  \"rounds\":{},\n  \"stopped_at_round_limit\":{},\n  \"training_bytes\":{},\n  \"training_AI_tokens\":{},\n  \"training_symbols\":{},\n  \"rules\":{},\n",
        "  \"segmentation_seconds\":{:.6},\n  \"merging_seconds\":{:.6},\n  \"training_seconds\":{:.6},\n  \"seconds\":{:.6},\n  \"registry_peak_bytes\":{},\n  \"peak_heap_bytes\":{},\n  \"peak_private_bytes\":{},\n  \"events\":[{}]\n}}\n"),
        every,reparse,cap,result.rounds,result.stopped_at_round_limit,result.model.training.bytes,result.model.training.tokens,result.model.training.symbols,result.model.dictionary.rules.len(),result.segmentation_seconds,result.merging_seconds,result.seconds,start.elapsed().as_secs_f64(),result.registry_peak_bytes,memory::peak_heap_bytes(),memory::process().peak_private,events.join(","));
    fs::write(output.join("report.json"), &json).map_err(|e| e.to_string())?;
    print!("{json}");
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("periodic-codec: {e}");
        std::process::exit(1);
    }
}
