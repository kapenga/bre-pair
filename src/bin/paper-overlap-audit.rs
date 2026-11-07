//! Target-only raw-shingle overlap screening; no full training shingle table.
use bre_pair::{
    grammar::BASE_SYMBOLS,
    memory,
    tokenizer::{ByteParser, Model},
};
use std::{
    env,
    fs::{self, File},
    io::{BufRead, BufReader, Write},
    process::ExitCode,
};
#[global_allocator]
static ALLOCATOR: memory::Allocator = memory::Allocator;
fn run() -> Result<(), String> {
    let args: Vec<_> = env::args().skip(1).collect();
    if args.len() != 4 {
        return Err(
            "paper-overlap-audit TARGET_SHINGLES.brt TRAINING.txt OUTPUT.json MEMORY_BYTES".into(),
        );
    }
    let _limit = memory::Limit::set(args[3].parse().map_err(|_| "invalid memory")?, 1)?;
    let model = Model::from_bytes(&fs::read(&args[0]).map_err(|e| e.to_string())?)?;
    let mut counts = vec![0u64; model.tokens.len()];
    let parser = ByteParser::new(model)?;
    let mut input = BufReader::new(File::open(&args[1]).map_err(|e| e.to_string())?);
    let mut field = Vec::new();
    let mut bytes = 0u64;
    let mut fields = 0u64;
    loop {
        field.clear();
        if input
            .read_until(b'\n', &mut field)
            .map_err(|e| e.to_string())?
            == 0
        {
            break;
        }
        parser.visit_fragment_matches(&field, |id| counts[id as usize] += 1);
        bytes += field.len() as u64;
        fields += 1;
    }
    let mut output = File::create_new(&args[2]).map_err(|e| e.to_string())?;
    write!(
        output,
        "{{\"bytes\":{bytes},\"fields\":{fields},\"pattern_counts\":[{}
]}}",
        counts[BASE_SYMBOLS..]
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .join(",")
    )
    .map_err(|e| e.to_string())?;
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
