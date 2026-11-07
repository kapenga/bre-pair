//! Research-only coarse profiling; normal decoder builds contain no spans.
#[cfg(not(feature = "profile"))]
fn main() {
    eprintln!("build with cargo build --release --features profile --bin profile-huffman");
    std::process::exit(1);
}

#[cfg(feature = "profile")]
#[global_allocator]
static ALLOCATOR: bre_pair::memory::Allocator = bre_pair::memory::Allocator;

#[cfg(feature = "profile")]
fn run() -> Result<(), String> {
    use bre_pair::{
        atomic_output::Output,
        huffman_archive::Reader,
        huffman_pipeline::{self, Options},
        memory, profile,
    };
    use std::{
        fs::File,
        io::{self, BufReader},
        path::Path,
        time::Instant,
    };
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() < 3 {
        return Err("profile-huffman INPUT.brh OUTPUT PROFILE.csv [--sink] [--buffers N] [--references N] [--output-bytes N] [--memory SIZE]".into());
    }
    let mut options = Options::default();
    let mut sink = false;
    let mut budget = 1_000_000_000usize;
    let mut at = 3;
    while at < args.len() {
        if args[at] == "--sink" {
            sink = true;
            at += 1;
            continue;
        }
        let value = args.get(at + 1).ok_or("missing profile option value")?;
        match args[at].as_str() {
            "--buffers" => {
                options.buffers_per_stage = value.parse().map_err(|_| "invalid buffer count")?
            }
            "--references" => {
                options.references_per_buffer =
                    value.parse().map_err(|_| "invalid references count")?
            }
            "--output-bytes" => {
                options.output_buffer_bytes =
                    value.parse().map_err(|_| "invalid output buffer size")?
            }
            "--memory" => budget = memory::parse_bytes(value)?,
            _ => return Err("unknown profile option".into()),
        }
        at += 2;
    }
    let _limit = memory::Limit::set(budget, 4)?;
    let trace = Output::new(Path::new(&args[2]))?;
    let output = if sink {
        None
    } else {
        Some(Output::new(Path::new(&args[1]))?)
    };
    if output
        .as_ref()
        .is_some_and(|o| o.destination() == trace.destination())
    {
        return Err("output and profile must differ".into());
    }
    let start = Instant::now();
    let reader = {
        let _setup = profile::Span::new("huffman.startup.dictionary_and_index");
        Reader::new(BufReader::with_capacity(
            1 << 20,
            File::open(&args[0]).map_err(|e| e.to_string())?,
        ))?
    };
    let report = if let Some(output) = &output {
        let mut writer = File::create_new(output.path()).map_err(|e| e.to_string())?;
        huffman_pipeline::decode(reader, false, &mut writer, options)?
    } else {
        huffman_pipeline::decode(reader, false, &mut io::sink(), options)?
    };
    let seconds = start.elapsed().as_secs_f64();
    profile::write(trace.path())?;
    if let Some(output) = output {
        output.commit()?;
    }
    trace.commit()?;
    let process = memory::process();
    println!("{{\"original_bytes\":{},\"seconds\":{seconds:.6},\"original_MB_per_second\":{:.3},\"pipeline_seconds\":{:.6},\"active_seconds\":[{:.6},{:.6},{:.6},{:.6}],\"wait_seconds\":[{:.6},{:.6},{:.6},{:.6}],\"buffers\":{},\"references\":{},\"output_buffer_bytes\":{},\"sink\":{},\"peak_heap_bytes\":{},\"peak_private_bytes\":{}}}",report.totals.bytes,report.totals.bytes as f64/seconds/1e6,report.seconds,report.stages[0].active_seconds,report.stages[1].active_seconds,report.stages[2].active_seconds,report.stages[3].active_seconds,report.stages[0].wait_seconds,report.stages[1].wait_seconds,report.stages[2].wait_seconds,report.stages[3].wait_seconds,options.buffers_per_stage,options.references_per_buffer,options.output_buffer_bytes,sink,memory::peak_heap_bytes(),process.peak_private);
    Ok(())
}

#[cfg(feature = "profile")]
fn main() {
    if let Err(e) = run() {
        eprintln!("profile-huffman: {e}");
        std::process::exit(1);
    }
}
