//! Streaming byte-for-byte verification of decoded AI IDs against the frozen
//! tokenizer. No whole-corpus/reference-ID copy; verification is not encoding.
use bre_pair::{
    memory,
    tokenizer::{Model, Tokenizer},
};
use std::{
    fs::{self, File},
    io::{BufRead, BufReader, Read},
    time::Instant,
};

#[global_allocator]
static ALLOCATOR: memory::Allocator = memory::Allocator;

fn run() -> Result<(), String> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 3 {
        return Err("verify-token-ids MODEL.brt ORIGINAL IDS.u32le".into());
    }
    let _limit = memory::Limit::set(2_000_000_000, 1)?;
    let tokenizer = Tokenizer::new(Model::from_bytes(
        &fs::read(&args[0]).map_err(|e| e.to_string())?,
    )?)?;
    let mut input =
        BufReader::with_capacity(1 << 20, File::open(&args[1]).map_err(|e| e.to_string())?);
    let mut ids =
        BufReader::with_capacity(1 << 20, File::open(&args[2]).map_err(|e| e.to_string())?);
    let mut field = Vec::new();
    let (mut bytes, mut fields, mut tokens) = (0u64, 0u64, 0u64);
    let start = Instant::now();
    loop {
        field.clear();
        if input
            .read_until(b'\n', &mut field)
            .map_err(|e| e.to_string())?
            == 0
        {
            break;
        }
        bytes = bytes
            .checked_add(field.len() as u64)
            .ok_or("input size overflow")?;
        fields += 1;
        for expected in tokenizer.encode(&field) {
            let mut word = [0; 4];
            ids.read_exact(&mut word)
                .map_err(|e| format!("AI ID stream at token {tokens}: {e}"))?;
            if u32::from_le_bytes(word) != expected {
                return Err(format!("AI ID differs at token {tokens}"));
            }
            tokens += 1;
        }
    }
    if ids.read(&mut [0]).map_err(|e| e.to_string())? != 0 {
        return Err("AI ID stream has trailing data".into());
    }
    println!("{{\"bytes\":{bytes},\"fields\":{fields},\"tokens\":{tokens},\"exact_AI_IDs_verified\":true,\"seconds\":{:.6}}}",start.elapsed().as_secs_f64());
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("verify-token-ids: {e}");
        std::process::exit(1);
    }
}
