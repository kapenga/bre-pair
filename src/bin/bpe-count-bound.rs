//! Conservative bound for the signed 32-bit pair counters in tokenizers 0.22.2.
//! Any merged pair has a fixed original-byte boundary; its count cannot exceed
//! that boundary's original adjacency count. Include the ordinary field EOF.
use std::fs::File;
use std::io::{self, Read};

struct Bound {
    bytes: u64,
    fields: u64,
    maximum: u64,
    pair: (usize, usize),
}

fn count(mut input: impl Read, chunk: usize) -> io::Result<Bound> {
    let mut counts = vec![0u64; 257 * 257];
    let mut buffer = vec![0u8; chunk];
    let (mut bytes, mut fields) = (0u64, 0u64);
    let mut previous = None;
    loop {
        let length = input.read(&mut buffer)?;
        if length == 0 {
            break;
        }
        bytes += length as u64;
        for &byte in &buffer[..length] {
            let symbol = byte as usize;
            if let Some(left) = previous {
                counts[left * 257 + symbol] += 1;
            }
            if byte == b'\n' {
                counts[symbol * 257 + 256] += 1;
                fields += 1;
                previous = None;
            } else {
                previous = Some(symbol);
            }
        }
    }
    if let Some(left) = previous {
        counts[left * 257 + 256] += 1;
        fields += 1;
    }
    let (index, maximum) = counts.iter().enumerate().max_by_key(|(_, n)| *n).unwrap();
    Ok(Bound {
        bytes,
        fields,
        maximum: *maximum,
        pair: (index / 257, index % 257),
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args_os().collect();
    if args.len() != 2 {
        return Err("bpe-count-bound TRAINING.txt".into());
    }
    let bound = count(File::open(&args[1])?, 1024 * 1024)?;
    println!(
        "{{\"input_bytes\":{},\"fields\":{},\"max_adjacency_count\":{},\"pair\":[{},{}],\"hf_i32_safe_with_2x_margin\":{}}}",
        bound.bytes,
        bound.fields,
        bound.maximum,
        bound.pair.0,
        bound.pair.1,
        bound.maximum <= i32::MAX as u64 / 2
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_overlap_and_eof_without_crossing_fields_or_chunks() {
        for chunk in 1..9 {
            let result = count(&b"aaaaa\na\na\na\na\nx"[..], chunk).unwrap();
            assert_eq!(result.bytes, 15);
            assert_eq!(result.fields, 6);
            assert_eq!(result.maximum, 5); // a, LF and LF, EOF tie.
            assert!([(10, 256), (97, 10)].contains(&result.pair));
            let run = count(&b"aaaaa"[..], chunk).unwrap();
            assert_eq!(run.maximum, 4); // HF BPE counts overlapping initial aa.
            assert_eq!(run.pair, (97, 97));
            assert_eq!(run.fields, 1);
        }
    }

    #[test]
    fn empty_input_has_no_fields_or_pairs() {
        let result = count(&b""[..], 1).unwrap();
        assert_eq!((result.bytes, result.fields, result.maximum), (0, 0, 0));
    }
}
