//! Opt-in comparison of normal loading and two-pass disk initialization.
use bre_pair::{
    archive_io, atomic_output::Output, disk_seed, engine::Config, field_engine, memory,
    scoring::Model, search,
};
use std::{
    fs::File,
    io::{BufReader, BufWriter, Write},
    path::Path,
    time::Instant,
};

#[global_allocator]
static ALLOCATOR: memory::Allocator = memory::Allocator;

fn progress(round: &bre_pair::engine::Round, offset: usize, started: Instant) {
    eprintln!(
        concat!(
            "codec progress {{\"event\":\"round\",\"round\":{},\"symbols\":{},",
            "\"grammar_rules\":{},\"new_rules\":{},\"rules_created\":{},\"packed_bytes\":{},",
            "\"symbol_bits\":{},\"selected_pairs\":{},\"minimum_selected_count\":{},",
            "\"maximum_selected_count\":{},\"seconds\":{:.3}}}"
        ),
        round.index + offset,
        round.after_refill,
        round.total_rules,
        round.new_rules,
        round.total_rules,
        round.packed_bytes,
        round.width,
        round.selected,
        round.minimum_selected_count,
        round.maximum_selected_count,
        started.elapsed().as_secs_f64()
    );
}

fn run() -> Result<(), String> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("--compare") {
        if args.len() != 3 {
            return Err("disk-seed-bench --compare FIRST.brp SECOND.brp".into());
        }
        let open = |name: &str| {
            archive_io::SymbolReader::new(BufReader::with_capacity(
                1 << 20,
                File::open(name).map_err(|e| e.to_string())?,
            ))
        };
        let mut first = open(&args[1])?;
        let mut second = open(&args[2])?;
        if first.expected_bytes != second.expected_bytes
            || first.expected_fields != second.expected_fields
            || !first
                .grammar
                .rules
                .iter()
                .map(|r| (r.left, r.right))
                .eq(second.grammar.rules.iter().map(|r| (r.left, r.right)))
        {
            return Err("archive metadata or grammar differs".into());
        }
        let mut symbols = 0u64;
        loop {
            let a = first.next_symbol()?;
            let b = second.next_symbol()?;
            if a != b {
                return Err(format!("archive symbols differ at index {symbols}"));
            }
            if a.is_none() {
                break;
            }
            symbols += 1;
        }
        println!("{{\"grammar_and_stream_identical\":true,\"symbols\":{symbols}}}");
        return Ok(());
    }
    if args.len() < 2 {
        return Err("disk-seed-bench INPUT NEW_OUTPUT.brp [--initialization disk|memory --mode seed|full --score frequency|weighted-pmi --min-count 8 --threads 4 --memory 2GB --chunk-symbols 1048576]".into());
    }
    let (mut disk, mut full, mut score, mut minimum, mut workers, mut budget, mut chunk) = (
        true,
        false,
        Model::Frequency,
        8u64,
        4usize,
        2_000_000_000usize,
        1048576usize,
    );
    for pair in args[2..].chunks(2) {
        let value = pair.get(1).ok_or("missing disk-seed option value")?;
        match pair[0].as_str() {
            "--initialization" => {
                disk = match value.as_str() {
                    "disk" => true,
                    "memory" => false,
                    _ => return Err("invalid initialization".into()),
                }
            }
            "--mode" => {
                full = match value.as_str() {
                    "full" => true,
                    "seed" => false,
                    _ => return Err("invalid mode".into()),
                }
            }
            "--score" => score = value.parse()?,
            "--min-count" => minimum = value.parse().map_err(|_| "invalid minimum")?,
            "--threads" => workers = value.parse().map_err(|_| "invalid workers")?,
            "--memory" => budget = memory::parse_bytes(value)?,
            "--chunk-symbols" => chunk = value.parse().map_err(|_| "invalid chunk size")?,
            _ => return Err("unknown disk-seed option".into()),
        }
    }
    if workers == 0
        || workers > 256
        || chunk == 0
        || minimum < 2
        || !matches!(score, Model::Frequency | Model::WeightedPmi)
    {
        return Err("invalid disk-seed comparison options".into());
    }
    let started = Instant::now();
    let limit = memory::Limit::set(budget, workers)?;
    let config = Config {
        buffer_bytes: limit.heap,
        minimum_count: minimum,
        score,
        search: search::Options {
            certify: false,
            fill_waves: 2,
            fill_min_count: minimum,
            ..Default::default()
        },
        ..Default::default()
    };
    let options = field_engine::Options {
        frequency_cutoff: true,
        memory_saving: true,
        max_rounds: (!full).then_some(1),
        ..Default::default()
    };
    let mut seed_json = String::from("null");
    let (source, rounds) = if disk {
        eprintln!("Two disk passes: global byte statistics/selection, then batch rewrite into packed memory.");
        let seed = disk_seed::prepare_file(Path::new(&args[0]), &config, chunk)?;
        let r = &seed.report;
        seed_json=format!(concat!("{{\"symbols_before\":{},\"symbols_after\":{},\"selected_pairs\":{},",
            "\"replacements\":{},\"width\":{},\"dense_statistics_bytes\":{},\"packed_payload_bytes\":{},",
            "\"packed_allocation_bytes\":{},\"first_scan_seconds\":{:.6},\"selection_seconds\":{:.6},\"second_scan_seconds\":{:.6}}}"),
            r.symbols_before,r.symbols_after,r.selected_pairs,r.replacements,r.width,r.dense_statistics_bytes,
            r.packed_payload_bytes,r.packed_allocation_bytes,r.first_scan_seconds,r.selection_seconds,r.second_scan_seconds);
        let initial = usize::from(!seed.selected.is_empty());
        drop(seed.selected);
        if full {
            let result = field_engine::compress_prepacked(
                seed.compressed,
                &config,
                workers,
                chunk,
                options,
                |r| progress(r, initial, started),
            )?;
            let rounds = initial + result.training.rounds.len();
            (result.training.compressed, rounds)
        } else {
            (seed.compressed, initial)
        }
    } else {
        let result = field_engine::compress_file(
            Path::new(&args[0]),
            &config,
            workers,
            chunk,
            options,
            |r| progress(r, 0, started),
        )?;
        let rounds = result.training.rounds.len();
        (result.training.compressed, rounds)
    };
    let training_seconds = started.elapsed().as_secs_f64();
    let output = Output::new(Path::new(&args[1]))?;
    let mut writer = BufWriter::with_capacity(
        1 << 20,
        File::create_new(output.path()).map_err(|e| e.to_string())?,
    );
    archive_io::write(&source, &mut writer)?;
    writer.flush().map_err(|e| e.to_string())?;
    drop(writer);
    output.commit()?;
    let process = memory::process();
    println!(concat!("{{\"initialization\":\"{}\",\"mode\":\"{}\",\"score\":\"{}\",\"minimum_count\":{},",
        "\"workers\":{},\"input_bytes\":{},\"fields\":{},\"rules\":{},\"rounds\":{},\"symbols\":{},",
        "\"training_seconds\":{:.6},\"seconds\":{:.6},\"peak_heap_bytes\":{},\"peak_private_bytes\":{},",
        "\"peak_resident_bytes\":{},\"seed\":{}}}"),
        if disk{"disk"}else{"memory"},if full{"full"}else{"seed"},score.name(),minimum,workers,source.input_bytes,
        source.field_count,source.grammar.rules.len(),rounds,source.token_count(),training_seconds,started.elapsed().as_secs_f64(),
        memory::peak_heap_bytes(),process.peak_private,process.peak_resident,seed_json);
    Ok(())
}
fn main() {
    if let Err(error) = run() {
        eprintln!("disk-seed-bench: {error}");
        std::process::exit(1);
    }
}
