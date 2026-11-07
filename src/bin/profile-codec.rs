//! Opt-in phase profiling of the complete frozen encoder.
#[cfg(not(feature = "profile"))]
fn main() {
    eprintln!("build with --features profile");
    std::process::exit(1);
}

#[cfg(feature = "profile")]
#[global_allocator]
static ALLOCATOR: bre_pair::memory::Allocator = bre_pair::memory::Allocator;

#[cfg(feature = "profile")]
fn run() -> Result<(), String> {
    use bre_pair::{
        atomic_output::Output,
        codec::{EncodeOptions, Frozen},
        huffman_archive::DEFAULT_BLOCK_SYMBOLS,
        memory, profile,
    };
    use std::{fs, path::Path, time::Instant};
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() < 4 || !(args.len() - 4).is_multiple_of(2) {
        return Err("profile-codec MODEL.brc INPUT OUTPUT.brh PROFILE.csv [--threads N] [--mode advanced|greedy]".into());
    }
    let mut options = EncodeOptions::default();
    for pair in args[4..].chunks(2) {
        match pair[0].as_str() {
            "--threads" => options.workers = pair[1].parse().map_err(|_| "invalid workers")?,
            "--mode" => options.mode = pair[1].parse()?,
            _ => return Err("unknown profile-codec option".into()),
        }
    }
    let _limit = memory::Limit::set(2_000_000_000, options.workers)?;
    let output = Output::new(Path::new(&args[2]))?;
    let trace = Output::new(Path::new(&args[3]))?;
    if output.destination() == trace.destination() {
        return Err("output and profile must differ".into());
    }
    let start = Instant::now();
    let model = {
        let _span = profile::Span::new("codec.startup.model");
        Frozen::from_bytes(&fs::read(&args[0]).map_err(|e| e.to_string())?)?
    };
    let totals = model.encode_file_with_options(
        Path::new(&args[1]),
        output.path(),
        DEFAULT_BLOCK_SYMBOLS,
        options,
    )?;
    output.commit()?;
    let seconds = start.elapsed().as_secs_f64();
    profile::write(trace.path())?;
    trace.commit()?;
    println!("{{\"bytes\":{},\"tokens\":{},\"codec_symbols\":{},\"seconds\":{seconds:.6},\"original_MB_per_second\":{:.3}}}",totals.bytes,totals.tokens,totals.symbols,totals.bytes as f64/seconds/1e6);
    Ok(())
}
#[cfg(feature = "profile")]
fn main() {
    if let Err(e) = run() {
        eprintln!("profile-codec: {e}");
        std::process::exit(1);
    }
}
