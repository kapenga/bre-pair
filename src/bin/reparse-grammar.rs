//! Research driver for fixed-grammar minimum-Huffman-bit reparsing.
use bre_pair::{archive_io, grammar_reparse, memory};
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
    if args.len() < 3 {
        return Err("reparse-grammar SOURCE.brp RAW NEW_DIRECTORY [--fields lines|single --weights shared|data --refits 2 --storage bounded|memory --max-field-bytes 134217728 --memory 8GB]".into());
    }
    let (mut single, mut shared, mut refits, mut maximum_field, mut budget) =
        (false, true, 2usize, 134217728usize, 8_000_000_000usize);
    let mut bounded = true;
    for pair in args[3..].chunks(2) {
        let value = pair.get(1).ok_or("missing reparse option value")?;
        match pair[0].as_str() {
            "--fields" => {
                single = match value.as_str() {
                    "single" => true,
                    "lines" => false,
                    _ => return Err("invalid fields".into()),
                }
            }
            "--weights" => {
                shared = match value.as_str() {
                    "shared" => true,
                    "data" => false,
                    _ => return Err("invalid weights".into()),
                }
            }
            "--refits" => refits = value.parse().map_err(|_| "invalid refits")?,
            "--max-field-bytes" => {
                maximum_field = value.parse().map_err(|_| "invalid field bound")?
            }
            "--memory" => budget = memory::parse_bytes(value)?,
            "--storage" => {
                bounded = match value.as_str() {
                    "bounded" => true,
                    "memory" => false,
                    _ => return Err("storage must be bounded or memory".into()),
                }
            }
            _ => return Err("unknown reparse option".into()),
        }
    }
    if !(1..=16).contains(&refits) || maximum_field == 0 || maximum_field == usize::MAX {
        return Err("invalid reparse bounds".into());
    }
    let _limit = memory::Limit::set(budget, 1)?;
    let _progress = bre_pair::training_progress::enable();
    let mut source = archive_io::read_packed(
        &mut BufReader::new(fs::File::open(&args[0]).map_err(|e| e.to_string())?),
        1 << 20,
    )?;
    let directory = Path::new(&args[2]);
    fs::create_dir(directory).map_err(|e| e.to_string())?;
    let started = Instant::now();
    for index in 1..=refits {
        let (next, before, after) = if bounded {
            grammar_reparse::reparse_bounded(
                source,
                Path::new(&args[1]),
                single,
                shared,
                maximum_field,
                directory,
            )?
        } else {
            grammar_reparse::reparse(source, Path::new(&args[1]), single, shared, maximum_field)?
        };
        let name = directory.join(format!("reparse-{index}.brp"));
        let mut writer = BufWriter::new(fs::File::create_new(name).map_err(|e| e.to_string())?);
        archive_io::write(&next, &mut writer)?;
        writer.flush().map_err(|e| e.to_string())?;
        eprintln!(
            "reparse {index}: {before} -> {after} payload bits under current costs; {} roots",
            next.token_count()
        );
        let json = format!("{{\"refit\":{index},\"weights\":\"{}\",\"fields_policy\":\"{}\",\"storage\":\"{}\",\"before_payload_bits\":{before},\"after_payload_bits\":{after},\"symbols\":{},\"seconds\":{:.6},\"peak_heap_bytes\":{}}}\n",
            if shared {"shared"} else {"data"}, if single {"single"} else {"lines"}, if bounded {"bounded"} else {"memory"}, next.token_count(), started.elapsed().as_secs_f64(), memory::peak_heap_bytes());
        fs::write(directory.join(format!("reparse-{index}.json")), &json)
            .map_err(|e| e.to_string())?;
        print!("{json}");
        source = next;
    }
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
