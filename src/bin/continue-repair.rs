//! Continue an existing packed grammar at a lower frequency floor. Research
//! only: no reparsing, document regrouping or grammar-ID reset is performed.
use bre_pair::{
    archive_io, atomic_output::Output, engine::Config, field_engine, memory, scoring::Model,
};
use std::{
    fs,
    io::{BufReader, BufWriter, Write},
    path::Path,
    process::ExitCode,
    time::Instant,
};

#[global_allocator]
static ALLOCATOR: memory::Allocator = memory::Allocator;

fn run() -> Result<(), String> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let usage="continue-repair INPUT.brp OUTPUT.brp [--min-count 2 --score weighted-pmi --threads 4 --memory 8GB --report FILE.json]";
    if args.len() < 2 {
        return Err(usage.into());
    }
    let (mut minimum, mut workers, mut budget) = (2u64, 4usize, 8_000_000_000usize);
    let mut score = Model::WeightedPmi;
    let mut report = None;
    for pair in args[2..].chunks(2) {
        let value = pair.get(1).ok_or("missing continuation option value")?;
        match pair[0].as_str() {
            "--min-count" => minimum = value.parse().map_err(|_| "invalid minimum")?,
            "--threads" => workers = value.parse().map_err(|_| "invalid worker count")?,
            "--memory" => budget = memory::parse_bytes(value)?,
            "--score" => score = value.parse()?,
            "--report" => report = Some(value),
            _ => return Err(usage.into()),
        }
    }
    if minimum < 2
        || !(1..=256).contains(&workers)
        || !matches!(score, Model::Frequency | Model::WeightedPmi)
    {
        return Err("invalid continuation settings".into());
    }
    let limit = memory::Limit::set(budget, workers)?;
    let output = Output::new(Path::new(&args[1]))?;
    let report_output = report.map(|p| Output::new(Path::new(p))).transpose()?;
    if report_output
        .as_ref()
        .is_some_and(|r| r.destination() == output.destination())
    {
        return Err("report must differ from archive".into());
    }
    let started = Instant::now();
    let source = archive_io::read_packed(
        &mut BufReader::new(fs::File::open(&args[0]).map_err(|e| e.to_string())?),
        1 << 20,
    )?;
    let old_rules = source.grammar.rules.len();
    let old_symbols = source.token_count();
    let result = field_engine::compress_prepacked(
        source,
        &Config {
            buffer_bytes: limit.heap,
            minimum_count: minimum,
            score,
            search: bre_pair::search::Options {
                certify: false,
                fill_waves: 2,
                fill_min_count: minimum,
                ..Default::default()
            },
            ..Default::default()
        },
        workers,
        1 << 20,
        field_engine::Options {
            frequency_cutoff: true,
            memory_saving: true,
            ..Default::default()
        },
        |r| {
            eprintln!(
                "continuation round {}: {} rules, {} symbols, selected {} ({}..{})",
                r.index,
                r.total_rules,
                r.rewritten,
                r.selected,
                r.minimum_selected_count,
                r.maximum_selected_count
            );
        },
    )?;
    let compressed = &result.training.compressed;
    let mut writer =
        BufWriter::new(fs::File::create_new(output.path()).map_err(|e| e.to_string())?);
    archive_io::write(compressed, &mut writer)?;
    writer.flush().map_err(|e| e.to_string())?;
    drop(writer);
    let json=format!(concat!("{{\"input_bytes\":{},\"fields\":{},\"minimum_count\":{},\"score\":\"{}\",",
        "\"initial_rules\":{},\"rules\":{},\"initial_symbols\":{},\"symbols\":{},\"batches\":{},\"seconds\":{:.6}}}\n"),
        compressed.input_bytes,compressed.field_count,minimum,score.name(),old_rules,compressed.grammar.rules.len(),old_symbols,compressed.token_count(),result.training.rounds.len(),started.elapsed().as_secs_f64());
    if let Some(report) = report_output {
        fs::write(report.path(), &json).map_err(|e| e.to_string())?;
        report.commit()?;
    }
    output.commit()?;
    print!("{json}");
    Ok(())
}
fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}
