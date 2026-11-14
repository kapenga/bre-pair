//! Matched tokenizer-discovery screen; does not change production CLI defaults.
use bre_pair::{
    archive, batch::Counting, engine::Config, entropy_engine, field_engine, memory, scalar_bonds,
    scoring, search,
};
use std::{
    env, fs,
    io::{BufRead, BufReader, BufWriter, Write},
    path::Path,
    process::ExitCode,
    time::Instant,
};

#[global_allocator]
static ALLOCATOR: memory::Allocator = memory::Allocator;

fn status(root: &Path, value: &str) -> Result<(), String> {
    let temporary = root.join("progress.tmp");
    fs::write(&temporary, value).map_err(|e| e.to_string())?;
    fs::rename(temporary, root.join("progress.json")).map_err(|e| e.to_string())
}
fn write(path: &Path, bytes: &[u8]) -> Result<(), String> {
    fs::File::create_new(path)
        .and_then(|mut f| f.write_all(bytes))
        .map_err(|e| e.to_string())
}
fn run() -> Result<(), String> {
    let args: Vec<_> = env::args().skip(1).collect();
    if args.len() != 6 || !["gain", "weighted-pmi"].contains(&args[2].as_str()) {
        return Err(
            "boundary-discovery INPUT NEW_RUN_DIR gain|weighted-pmi MIN_COUNT MAX_RULES WORKERS"
                .into(),
        );
    }
    let root = Path::new(&args[1]);
    let minimum = args[3].parse::<u64>().map_err(|_| "invalid minimum")?;
    let max_rules = args[4].parse::<usize>().map_err(|_| "invalid rule cap")?;
    let workers = args[5]
        .parse::<usize>()
        .map_err(|_| "invalid worker count")?;
    if minimum < 2 || max_rules == 0 || !(1..=256).contains(&workers) {
        return Err("invalid discovery settings".into());
    }
    fs::create_dir(root).map_err(|e| e.to_string())?;
    let started = Instant::now();
    status(root, "{\"stage\":\"reading training fields\"}")?;
    let mut input = BufReader::with_capacity(
        1 << 20,
        fs::File::open(&args[0]).map_err(|e| e.to_string())?,
    );
    let mut fields = Vec::new();
    loop {
        let mut field = Vec::new();
        if input
            .read_until(b'\n', &mut field)
            .map_err(|e| e.to_string())?
            == 0
        {
            break;
        }
        fields.push(field);
    }
    let read_seconds = started.elapsed().as_secs_f64();
    let input_bytes: usize = fields.iter().map(Vec::len).sum();
    let search = search::Options {
        certify: false,
        fill_waves: 2,
        fill_min_count: minimum,
        ..Default::default()
    };
    let mut trace =
        BufWriter::new(fs::File::create_new(root.join("rounds.csv")).map_err(|e| e.to_string())?);
    writeln!(trace,"round,rules,tokens,selected,min_count,max_count,seconds,removed_boundary_units,remaining_boundary_units").map_err(|e|e.to_string())?;
    let mut progress_error = None;
    let mut progress = |r: &bre_pair::engine::Round,
                        loss: Option<&entropy_engine::BoundaryRound>| {
        let removed = loss.map_or("null".into(), |v| v.removed_units.to_string());
        let after = loss.map_or("null".into(), |v| v.after_units.to_string());
        let row = format!(
            "{},{},{},{},{},{},{:.6},{},{}\n",
            r.index,
            r.total_rules,
            r.rewritten,
            r.selected,
            r.minimum_selected_count,
            r.maximum_selected_count,
            r.seconds,
            removed,
            after
        );
        let value=format!("{{\"stage\":\"discovery\",\"round\":{},\"rules\":{},\"tokens\":{},\"selected\":{},\"minimumSelectedCount\":{},\"maximumSelectedCount\":{},\"roundSeconds\":{:.6},\"boundaryLossUnits\":{}}}",
            r.index,r.total_rules,r.rewritten,r.selected,r.minimum_selected_count,r.maximum_selected_count,r.seconds,after);
        if let Err(error) = trace
            .write_all(row.as_bytes())
            .and_then(|_| trace.flush())
            .map_err(|e| e.to_string())
            .and_then(|_| status(root, &value))
        {
            progress_error = Some(error);
        }
        println!("{value}");
    };
    let (
        training,
        statistics_seconds,
        initial_loss,
        final_loss,
        weights_bytes,
        peak_pairs,
        rule_limited,
    ) = if args[2] == "gain" {
        status(root, "{\"stage\":\"counting training scalar bonds\"}")?;
        let statistics_started = Instant::now();
        let counts = scalar_bonds::Statistics::count(&fields)?;
        let mut bonds = BufWriter::new(
            fs::File::create_new(root.join("bonds.bin")).map_err(|e| e.to_string())?,
        );
        counts.write_counts(&mut bonds)?;
        bonds.flush().map_err(|e| e.to_string())?;
        let model = counts.freeze();
        let statistics_seconds = statistics_started.elapsed().as_secs_f64();
        status(root, "{\"stage\":\"annotating frozen training bonds\"}")?;
        let result = entropy_engine::compress_boundaries(
            &fields,
            &entropy_engine::Options {
                minimum_count: minimum,
                workers,
                max_rules: Some(max_rules),
                buffer_bytes: 4usize << 30,
                search,
                ..Default::default()
            },
            move |field| model.field_weights(field),
            |r, loss| progress(r, Some(loss)),
        )?;
        (
            result.training,
            statistics_seconds,
            result.initial_loss_units.to_string(),
            result.final_loss_units.to_string(),
            result.initial_weight_bytes,
            result.peak_pair_entries,
            result.stopped_at_rule_limit,
        )
    } else {
        let result = field_engine::compress_with_options(
            &fields,
            &Config {
                minimum_count: minimum,
                score: scoring::Model::WeightedPmi,
                buffer_bytes: 4usize << 30,
                pair_counting: Counting::Adaptive,
                search,
                ..Default::default()
            },
            workers,
            64,
            65536,
            field_engine::Options {
                max_rules: Some(max_rules),
                frequency_cutoff: true,
                memory_saving: true,
                ..Default::default()
            },
            |r| progress(r, None),
        )?;
        let limited = result.stopped_at_rule_limit;
        (
            result.training,
            0.0,
            "null".into(),
            "null".into(),
            0,
            0,
            limited,
        )
    };
    if let Some(error) = progress_error {
        return Err(error);
    }
    status(root, "{\"stage\":\"serializing and verifying grammar\"}")?;
    let encoded = archive::encode(&training.compressed)?;
    if archive::decode(&encoded)?.decode()? != fields {
        return Err("discovery archive failed exact reconstruction".into());
    }
    write(&root.join("grammar.brp"), &encoded)?;
    let report=format!(concat!("{{\n\"stage\":\"complete\",\"mode\":\"{}\",\"inputBytes\":{},\"fields\":{},\"minimumCount\":{},",
        "\"workers\":{},\"fillWaves\":2,\"maxRules\":{},\"ruleLimited\":{},\"rounds\":{},\"rules\":{},\"tokens\":{},",
        "\"readSeconds\":{:.6},\"statisticsSeconds\":{:.6},\"trainingSeconds\":{:.6},\"totalSeconds\":{:.6},",
        "\"initialLossUnits\":{},\"finalLossUnits\":{},\"initialBoundaryBufferBytes\":{},\"peakPairEntries\":{},",
        "\"peakHeapBytes\":{},\"archiveBytes\":{},\"roundTripVerified\":true}}\n"),
        args[2],input_bytes,fields.len(),minimum,workers,max_rules,rule_limited,training.rounds.len(),training.compressed.grammar.rules.len(),
        training.compressed.token_count(),read_seconds,statistics_seconds,training.seconds,started.elapsed().as_secs_f64(),initial_loss,final_loss,
        weights_bytes,peak_pairs,memory::peak_heap_bytes(),encoded.len());
    write(&root.join("report.json"), report.as_bytes())?;
    status(root, &report)?;
    print!("{report}");
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
