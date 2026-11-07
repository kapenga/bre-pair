//! Research harness: apply one frozen vocabulary to a bounded in-memory corpus.
//! Model setup, input loading, verification, and optional ID export are untimed.
use bre_pair::{
    atomic_output::Output,
    memory,
    tokenizer::{fingerprint, Bpe, Model, Tokenizer},
};
use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    path::Path,
    sync::atomic::{AtomicUsize, Ordering},
    time::Instant,
};

#[global_allocator]
static ALLOCATOR: memory::Allocator = memory::Allocator;

fn run() -> Result<(), String> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        return Err("static-apply-bench MODEL.brt INPUT [--threads N] [--iterations N] [--bpe-merges RANKS.brm] [--ids OUTPUT]".into());
    }
    let (mut workers, mut iterations, mut ranks, mut id_path) = (1, 3, None, None);
    let mut at = 2;
    while at < args.len() {
        let value = args.get(at + 1).ok_or("missing option value")?;
        match args[at].as_str() {
            "--threads" => workers = value.parse::<usize>().map_err(|_| "invalid workers")?,
            "--iterations" => {
                iterations = value.parse::<usize>().map_err(|_| "invalid iterations")?
            }
            "--bpe-merges" => ranks = Some(value),
            "--ids" => id_path = Some(value),
            _ => return Err("unknown benchmark option".into()),
        }
        at += 2;
    }
    if !(1..=256).contains(&workers) || !(1..=10).contains(&iterations) {
        return Err("workers must be 1..256 and iterations 1..10".into());
    }
    let _limit = memory::Limit::set(2_000_000_000, workers)?;
    let start = Instant::now();
    let model = Model::from_bytes(&fs::read(&args[0]).map_err(|e| e.to_string())?)?;
    let model_fingerprint = fingerprint(&model.to_bytes());
    let bpe = ranks
        .map(|path| Bpe::from_bytes(model.clone(), &fs::read(path).map_err(|e| e.to_string())?))
        .transpose()?;
    let tokenizer = Tokenizer::new(model)?;
    let setup = start.elapsed().as_secs_f64();
    let start = Instant::now();
    if fs::metadata(&args[1]).map_err(|e| e.to_string())?.len() > 256 << 20 {
        return Err("research fixture must be at most 256 MiB; use streaming public commands for large inputs".into());
    }
    let data = fs::read(&args[1]).map_err(|e| e.to_string())?;
    let mut begin = 0;
    let fields: Vec<_> = data
        .split_inclusive(|&b| b == b'\n')
        .map(|field| {
            let end = begin + field.len();
            let range = begin..end;
            begin = end;
            range
        })
        .collect();
    let loading = start.elapsed().as_secs_f64();
    let encode = |field: &[u8]| {
        bpe.as_ref()
            .map_or_else(|| tokenizer.encode(field), |b| b.encode(field))
    };
    let id_output = id_path.map(|p| Output::new(Path::new(p))).transpose()?;
    let mut writer = id_output
        .as_ref()
        .map(|o| File::create_new(o.path()).map(BufWriter::new))
        .transpose()
        .map_err(|e| e.to_string())?;
    let mut expected = 0u64;
    let start = Instant::now();
    for range in &fields {
        let field = &data[range.clone()];
        let ids = encode(field);
        if tokenizer.model().decode_field(&ids)? != field {
            return Err("tokenizer changed field bytes or EOF".into());
        }
        expected += ids.len() as u64;
        if let Some(writer) = &mut writer {
            for id in ids {
                writer
                    .write_all(&id.to_le_bytes())
                    .map_err(|e| e.to_string())?;
            }
        }
    }
    if let Some(writer) = &mut writer {
        writer.flush().map_err(|e| e.to_string())?;
    }
    drop(writer);
    if let Some(output) = id_output {
        output.commit()?;
    }
    let verification = start.elapsed().as_secs_f64();
    let mut seconds = Vec::new();
    for _ in 0..iterations {
        let next = AtomicUsize::new(0);
        let start = Instant::now();
        let actual = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..workers)
                .map(|_| {
                    scope.spawn(|| {
                        let mut tokens = 0u64;
                        loop {
                            let from = next.fetch_add(32, Ordering::Relaxed);
                            if from >= fields.len() {
                                break;
                            }
                            for range in &fields[from..(from + 32).min(fields.len())] {
                                let ids = encode(&data[range.clone()]);
                                std::hint::black_box(&ids);
                                tokens += ids.len() as u64;
                            }
                        }
                        tokens
                    })
                })
                .collect();
            handles.into_iter().try_fold(0u64, |n, h| {
                h.join()
                    .map(|found| n + found)
                    .map_err(|_| "worker panicked")
            })
        })?;
        let elapsed = start.elapsed().as_secs_f64();
        if actual != expected {
            return Err("parallel token count differs from verified serial parse".into());
        }
        seconds.push(elapsed);
    }
    let durations = seconds
        .iter()
        .map(|n| format!("{n:.9}"))
        .collect::<Vec<_>>()
        .join(",");
    seconds.sort_by(f64::total_cmp);
    let median = (seconds[(iterations - 1) / 2] + seconds[iterations / 2]) / 2.0;
    println!("{{\"scope\":\"warm in-memory parsing to u32 token vectors; worker startup included; setup, IO, verification and serialization excluded\",\"algorithm\":\"{}\",\"model_fingerprint\":{model_fingerprint},\"vocabulary\":{},\"input_bytes\":{},\"fields\":{},\"maximum_field_bytes\":{},\"tokens\":{expected},\"workers\":{workers},\"round_trip_verified\":true,\"iterations\":{iterations},\"iteration_seconds\":[{durations}],\"median_seconds\":{median:.9},\"original_MB_per_second\":{:.6},\"million_tokens_per_second\":{:.6},\"model_setup_seconds\":{setup:.6},\"input_loading_seconds\":{loading:.6},\"verification_export_seconds\":{verification:.6},\"peak_heap_bytes\":{},\"peak_private_bytes\":{}}}",if bpe.is_some() {"native-reference-ranked-bpe"} else {"exact-minimum-token"},tokenizer.model().tokens.len(),data.len(),fields.len(),fields.iter().map(|r|r.len()).max().unwrap_or(0),data.len() as f64/median/1e6,expected as f64/median/1e6,memory::peak_heap_bytes(),memory::process().peak_private);
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("static-apply-bench: {error}");
        std::process::exit(1);
    }
}
