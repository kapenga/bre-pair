//! Experimental fresh discovery with a training-only language allocation prior.
use bre_pair::{
    archive_io, engine::Config, field_engine, language_prior::Policy, memory, scoring::Model,
};
use std::{
    fs::File,
    io::{BufWriter, Write},
    path::Path,
    time::Instant,
};

#[global_allocator]
static ALLOCATOR: memory::Allocator = memory::Allocator;

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 11 {
        return Err("usage: language-train TEXT INDEX OUTPUT.brp REPORT.json ROUNDS.csv MULTIPLIER MINIMUM WORKERS MEMORY_BYTES STOP_MAX_COUNT".into());
    }
    for p in &args[3..6] {
        if Path::new(p).exists() {
            return Err(format!("destination exists: {p}"));
        }
    }
    let multiplier: f64 = args[6].parse().map_err(|_| "invalid multiplier")?;
    let minimum: u64 = args[7].parse().map_err(|_| "invalid minimum")?;
    let workers: usize = args[8].parse().map_err(|_| "invalid workers")?;
    let budget: usize = args[9].parse().map_err(|_| "invalid memory")?;
    let stop: u64 = args[10].parse().map_err(|_| "invalid stop count")?;
    let _limit = memory::Limit::set(budget, workers)?;
    let _progress = bre_pair::training_progress::enable();
    let started = Instant::now();
    let policy = Policy::read(Path::new(&args[2]), 4, multiplier)?;
    let mut config = Config {
        minimum_count: minimum,
        buffer_bytes: _limit.heap,
        score: Model::WeightedPmi,
        ..Config::default()
    };
    config.search.certify = false;
    config.search.fill_waves = 2;
    config.search.fill_min_count = minimum;
    let mut trace = File::create_new(&args[5]).map_err(|e| e.to_string())?;
    writeln!(
        trace,
        "round,rules,symbols,packed_bytes,selected,min_count,max_count,seconds"
    )
    .map_err(|e| e.to_string())?;
    let training = field_engine::compress_file_language_weighted(
        Path::new(&args[1]),
        &policy,
        &config,
        workers,
        1048576,
        field_engine::Options {
            frequency_cutoff: true,
            memory_saving: true,
            stop_max_pair_count: Some(stop),
            ..Default::default()
        },
        |r| {
            bre_pair::training_progress::completed(r, r.total_rules);
            let _ = writeln!(
                trace,
                "{},{},{},{},{},{},{},{:.6}",
                r.index,
                r.total_rules,
                r.rewritten,
                r.packed_bytes,
                r.selected,
                r.minimum_selected_count,
                r.maximum_selected_count,
                r.seconds
            );
            let _ = trace.flush();
            eprintln!("{{\"event\":\"round\",\"round\":{},\"rules\":{},\"symbols\":{},\"packed_bytes\":{},\"selected\":{},\"minimum_count\":{},\"maximum_count\":{},\"seconds\":{:.3}}}",r.index,r.total_rules,r.rewritten,r.packed_bytes,r.selected,r.minimum_selected_count,r.maximum_selected_count,r.seconds);
        },
    )?;
    let result = &training.training;
    let mut out = BufWriter::new(File::create_new(&args[3]).map_err(|e| e.to_string())?);
    archive_io::write(&result.compressed, &mut out)?;
    out.flush().map_err(|e| e.to_string())?;
    let report = format!("{{\"input_bytes\":{},\"fields\":{},\"rules\":{},\"symbols\":{},\"batches\":{},\"minimum_count\":{},\"fill_waves\":2,\"fill_min_count\":{},\"mandarin_multiplier\":{},\"weighted_marginals\":true,\"eligibility\":\"raw counts\",\"discovery_seconds\":{:.6},\"total_seconds\":{:.6},\"stop_max_pair_count\":{},\"stopped_at_pair_count\":{},\"maximum_pair_count_at_stop\":{},\"peak_heap_bytes\":{},\"memory_limit_bytes\":{}}}\n",
        result.compressed.input_bytes,result.compressed.field_count,result.compressed.grammar.rules.len(),result.compressed.token_count(),result.rounds.len(),minimum,minimum,multiplier,result.seconds,started.elapsed().as_secs_f64(),stop,training.stopped_at_pair_count,training.maximum_pair_count_at_stop.map_or("null".into(),|n|n.to_string()),memory::peak_heap_bytes(),budget);
    std::fs::write(&args[4], report).map_err(|e| e.to_string())?;
    Ok(())
}
fn main() {
    if let Err(error) = run() {
        eprintln!("language-train: {error}");
        std::process::exit(2);
    }
}
