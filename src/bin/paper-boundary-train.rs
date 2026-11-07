//! Research-only training-boundary control. Segment records never invent EOF.
use bre_pair::{
    archive,
    batch::Counting,
    engine::{Compressed, Config},
    field_engine,
    grammar::{Grammar, EOF},
    memory,
    packed::PackedSymbols,
    scoring::Model,
};
use std::{
    env,
    fs::File,
    io::{BufReader, Read, Write},
    path::Path,
    process::ExitCode,
};

#[global_allocator]
static ALLOCATOR: memory::Allocator = memory::Allocator;

fn read_segments(mut input: impl Read, bounded: bool) -> Result<Compressed, String> {
    let mut magic = [0; 8];
    input.read_exact(&mut magic).map_err(|e| e.to_string())?;
    if &magic != b"BRSEGM01" {
        return Err("invalid segmented corpus header".into());
    }
    let mut blocks = Vec::new();
    let mut block = PackedSymbols::new(9);
    let (mut input_bytes, mut field_count, mut open) = (0u64, 0u64, false);
    loop {
        let mut header = [0; 5];
        match input.read(&mut header[..1]).map_err(|e| e.to_string())? {
            0 => break,
            _ => input
                .read_exact(&mut header[1..])
                .map_err(|e| e.to_string())?,
        }
        let length = u32::from_le_bytes(header[..4].try_into().unwrap()) as usize;
        if length > 64 * 1024 * 1024 || header[4] > 1 || (length == 0 && header[4] == 0) {
            return Err("invalid segmented corpus record".into());
        }
        let mut bytes = vec![0; length];
        input.read_exact(&mut bytes).map_err(|e| e.to_string())?;
        if bytes
            .iter()
            .enumerate()
            .any(|(i, &b)| b == b'\n' && (i + 1 != length || header[4] != 1))
        {
            return Err("segment crosses a natural line field ending".into());
        }
        for byte in bytes {
            block.push(byte as u32);
        }
        input_bytes += length as u64;
        open = true;
        if header[4] == 1 {
            block.push(EOF);
            field_count += 1;
            open = false;
        }
        if bounded || (header[4] == 1 && block.len() >= 1_048_576) {
            block.shrink_to_fit();
            blocks.push(std::mem::replace(&mut block, PackedSymbols::new(9)));
        }
    }
    if open {
        return Err("last segment has no natural field ending".into());
    }
    if !block.is_empty() {
        block.shrink_to_fit();
        blocks.push(block);
    }
    Ok(Compressed {
        grammar: Grammar::default(),
        blocks,
        input_bytes,
        field_count,
    })
}

fn run() -> Result<(), String> {
    let args: Vec<_> = env::args().skip(1).collect();
    if args.len() != 7 {
        return Err("paper-boundary-train SEGMENTS RAW ARCHIVE bounded|unbounded MINIMUM RULE_BUDGET MEMORY_BYTES".into());
    }
    let bounded = match args[3].as_str() {
        "bounded" => true,
        "unbounded" => false,
        _ => return Err("invalid boundary mode".into()),
    };
    let minimum = args[4].parse().map_err(|_| "invalid minimum")?;
    let budget = args[5].parse().map_err(|_| "invalid rule budget")?;
    let total = args[6].parse().map_err(|_| "invalid memory budget")?;
    let limit = memory::Limit::set(total, 4)?;
    let source = read_segments(
        BufReader::new(File::open(&args[0]).map_err(|e| e.to_string())?),
        bounded,
    )?;
    bre_pair::vocabulary::verify_training_source(&source, Path::new(&args[1]))?;
    let config = Config {
        buffer_bytes: limit.heap,
        minimum_count: minimum,
        score: Model::WeightedPmi,
        pair_counting: Counting::Adaptive,
        search: bre_pair::search::Options {
            certify: false,
            fill_waves: 2,
            fill_min_count: minimum,
            ..Default::default()
        },
        ..Default::default()
    };
    let result = field_engine::compress_segments(
        source,
        &config,
        4,
        field_engine::Options {
            max_rules: (budget != 0).then_some(budget),
            stop_max_pair_count: Some(2 * minimum),
            ..Default::default()
        },
        |r| {
            eprintln!(
                "round={} rules={} selected={} seconds={:.3}",
                r.index, r.total_rules, r.selected, r.seconds
            )
        },
    )?;
    bre_pair::vocabulary::verify_training_source(&result.training.compressed, Path::new(&args[1]))?;
    let bytes = archive::encode(&result.training.compressed)?;
    File::create_new(&args[2])
        .and_then(|mut f| f.write_all(&bytes))
        .map_err(|e| e.to_string())?;
    println!("{{\"rounds\":{},\"rules\":{},\"bounded\":{},\"stopped_at_pair_count\":{},\"stopped_at_rule_limit\":{},\"heap_peak_bytes\":{}}}",
        result.training.rounds.len(),result.training.compressed.grammar.rules.len(),bounded,result.stopped_at_pair_count,result.stopped_at_rule_limit,memory::peak_heap_bytes());
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

#[cfg(test)]
mod tests {
    use super::*;
    fn record(out: &mut Vec<u8>, bytes: &[u8], final_segment: bool) {
        out.extend((bytes.len() as u32).to_le_bytes());
        out.push(u8::from(final_segment));
        out.extend(bytes);
    }
    #[test]
    fn internal_boundaries_have_no_synthetic_eof_and_survive_training() {
        let mut bytes = b"BRSEGM01".to_vec();
        for _ in 0..20 {
            record(&mut bytes, b"a", false);
            record(&mut bytes, b"b\n", true);
        }
        let source = read_segments(bytes.as_slice(), true).unwrap();
        assert_eq!(source.field_count, 20);
        let config = Config {
            minimum_count: 2,
            score: Model::WeightedPmi,
            buffer_bytes: 1 << 20,
            ..Default::default()
        };
        let trained = field_engine::compress_segments(
            source,
            &config,
            2,
            field_engine::Options::default(),
            |_| {},
        )
        .unwrap();
        assert!(trained.training.compressed.grammar.find((97, 98)).is_none());
        assert_eq!(
            trained.training.compressed.decode().unwrap(),
            vec![b"ab\n".to_vec(); 20]
        );
        let flat = read_segments(bytes.as_slice(), false).unwrap();
        let trained = field_engine::compress_segments(
            flat,
            &config,
            2,
            field_engine::Options::default(),
            |_| {},
        )
        .unwrap();
        assert_eq!(
            trained.training.compressed.decode().unwrap(),
            vec![b"ab\n".to_vec(); 20]
        );
        assert!(trained
            .training
            .compressed
            .grammar
            .rules
            .iter()
            .any(|r| r.terminals >= 4));
    }
    #[test]
    fn malformed_or_unclosed_records_are_rejected() {
        let mut bytes = b"BRSEGM01".to_vec();
        record(&mut bytes, b"a", false);
        assert!(read_segments(bytes.as_slice(), true).is_err());
        let mut bytes = b"BRSEGM01".to_vec();
        record(&mut bytes, b"a\nb", true);
        assert!(read_segments(bytes.as_slice(), true).is_err());
    }
}
