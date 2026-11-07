//! Steady-state decoding of the first independent block. Container checksums
//! and codebook construction are checked once before timing, not in the loop.
use bre_pair::{huffman_archive::Reader, tokenizer::fingerprint};
use std::{fs::File, io::BufReader, time::Instant};
fn run() -> Result<(), String> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.is_empty() || args.len() > 2 {
        return Err("huffman-bench INPUT.brh [ITERATIONS]".into());
    }
    let start = Instant::now();
    let mut reader = Reader::new(BufReader::new(
        File::open(&args[0]).map_err(|e| e.to_string())?,
    ))?;
    let block = reader.block(0)?;
    let decoder = reader.prepare_decoder(false, 64 << 20)?;
    let startup = start.elapsed().as_secs_f64();
    let n: usize = if let Some(n) = args.get(1) {
        n.parse().map_err(|_| "invalid iterations")?
    } else {
        (2_000_000_000u64 / block.totals.bytes.max(1)).clamp(3, 4096) as usize
    };
    if n == 0 {
        return Err("iterations must be positive".into());
    }
    let mut output = Vec::new();
    decoder.decode_block(&reader.dictionary, &block, &mut output)?;
    let expected = fingerprint(&output);
    let start = Instant::now();
    for _ in 0..n {
        output.clear();
        decoder.decode_block(&reader.dictionary, &block, &mut output)?;
        std::hint::black_box(&output);
    }
    let elapsed = start.elapsed().as_secs_f64();
    if fingerprint(&output) != expected {
        return Err("benchmark output changed".into());
    }
    println!("{{\"scope\":\"first block; warm arena; no file IO or block checksum in timed loop\",\"iterations\":{n},\"block_original_bytes\":{},\"block_payload_bytes\":{},\"block_symbols\":{},\"used_codes\":{},\"max_code_bits\":{},\"dictionary_symbols\":{},\"arena_bytes_including_references\":{},\"startup_seconds\":{startup:.6},\"byte_decode_seconds\":{elapsed:.6},\"byte_decode_original_MB_per_second\":{:.3},\"payload_MB_per_second\":{:.3},\"codec_Msymbols_per_second\":{:.3},\"decoded_fnv64\":\"{expected:016x}\"}}",block.totals.bytes,block.payload.len(),block.totals.symbols,block.book.symbols.len(),block.book.max_bits,reader.dictionary.expansions.len(),decoder.arena_bytes(),block.totals.bytes as f64*n as f64/elapsed/1e6,block.payload.len() as f64*n as f64/elapsed/1e6,block.totals.symbols as f64*n as f64/elapsed/1e6);
    if reader.dictionary.ai_tokens {
        let decoder = reader.prepare_decoder(true, 64 << 20)?;
        output.clear();
        decoder.decode_block(&reader.dictionary, &block, &mut output)?;
        let expected = fingerprint(&output);
        let start = Instant::now();
        for _ in 0..n {
            output.clear();
            decoder.decode_block(&reader.dictionary, &block, &mut output)?;
            std::hint::black_box(&output);
        }
        let elapsed = start.elapsed().as_secs_f64();
        if fingerprint(&output) != expected {
            return Err("benchmark IDs changed".into());
        }
        println!("{{\"mode\":\"stable AI IDs, u32 little endian\",\"iterations\":{n},\"seconds\":{elapsed:.6},\"million_AI_tokens_per_second\":{:.3},\"original_MB_per_second\":{:.3},\"ID_output_MB_per_second\":{:.3},\"decoded_fnv64\":\"{expected:016x}\"}}",block.totals.tokens as f64*n as f64/elapsed/1e6,block.totals.bytes as f64*n as f64/elapsed/1e6,block.totals.tokens as f64*4.0*n as f64/elapsed/1e6);
    }
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("huffman-bench: {e}");
        std::process::exit(1);
    }
}
