//! Self-contained byte-rank transport benchmark; no raw-text parser is replaced.
#[cfg(feature = "bytecode-experiments")]
#[global_allocator]
static ALLOC: bre_pair::memory::Allocator = bre_pair::memory::Allocator;

#[cfg(feature = "bytecode-experiments")]
fn run() -> Result<(), String> {
    use bre_pair::{memory, shared_huffman::byte_archive};
    use std::{path::Path, time::Instant};
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() < 3 || !matches!(args[0].as_str(), "pack" | "decode") {
        return Err("byte-archive pack INPUT.brh OUTPUT.brb | decode INPUT.brb OUTPUT [--threads 1|4] [--memory SIZE]".into());
    }
    let (mut workers, mut budget) = (1, 2_000_000_000);
    for pair in args[3..].chunks(2) {
        let value = pair.get(1).ok_or("missing byte archive option value")?;
        match pair[0].as_str() {
            "--threads" => workers = value.parse().map_err(|_| "invalid byte worker count")?,
            "--memory" => budget = memory::parse_bytes(value)?,
            _ => return Err("unknown byte archive option".into()),
        }
    }
    let _limit = memory::Limit::set(budget, workers)?;
    let start = Instant::now();
    let totals = if args[0] == "pack" {
        byte_archive::pack(Path::new(&args[1]), Path::new(&args[2]))?
    } else {
        byte_archive::decode(Path::new(&args[1]), Path::new(&args[2]), workers)?
    };
    let seconds = start.elapsed().as_secs_f64();
    println!("{{\"format\":\"BRBYTEA1\",\"original_bytes\":{},\"fields\":{},\"codec_symbols\":{},\"threads\":{},\"seconds\":{:.6},\"original_MB_per_second\":{:.3},\"peak_heap_bytes\":{},\"saved_rank_transcode\":{}}}",
        totals.bytes,totals.fields,totals.symbols,workers,seconds,totals.bytes as f64/seconds/1e6,memory::peak_heap_bytes(),args[0]=="pack");
    Ok(())
}

fn main() {
    #[cfg(feature = "bytecode-experiments")]
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(2);
    }
    #[cfg(not(feature = "bytecode-experiments"))]
    {
        eprintln!("enable --features bytecode-experiments");
        std::process::exit(2);
    }
}
