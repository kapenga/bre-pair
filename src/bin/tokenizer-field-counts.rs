//! Exact intrinsic counts for length-framed fields, retaining internal newlines.
use bre_pair::{
    atomic_output::Output,
    tokenizer::{Model, Tokenizer},
};
use std::{
    fs::{self, File},
    io::{BufReader, BufWriter, Read, Write},
    path::Path,
};

fn count(tokenizer: &Tokenizer, raw: &[u8]) -> Result<[u64; 3], String> {
    let ids = tokenizer.encode(raw);
    if tokenizer.model().decode_field(&ids)? != raw {
        return Err("intrinsic field reconstruction differs".into());
    }
    Ok([
        ids.len() as u64,
        ids.iter().filter(|&&id| id != 256).count() as u64,
        ids.iter().filter(|&&id| id < 256).count() as u64,
    ])
}

fn u64_read(input: &mut impl Read) -> Result<u64, String> {
    let mut bytes = [0; 8];
    input.read_exact(&mut bytes).map_err(|e| e.to_string())?;
    Ok(u64::from_le_bytes(bytes))
}

fn run() -> Result<(), String> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 3 {
        return Err("tokenizer-field-counts MODEL.brt INPUT.fields NEW_OUTPUT.counts".into());
    }
    let tokenizer = Tokenizer::new(Model::from_bytes(
        &fs::read(&args[0]).map_err(|e| e.to_string())?,
    )?)?;
    let file = File::open(&args[1]).map_err(|e| e.to_string())?;
    let size = file.metadata().map_err(|e| e.to_string())?.len();
    let mut input = BufReader::new(file);
    let mut magic = [0; 8];
    input.read_exact(&mut magic).map_err(|e| e.to_string())?;
    if &magic != b"BRTFLD01" {
        return Err("invalid intrinsic field framing".into());
    }
    let fields = u64_read(&mut input)?;
    if size < 16 || fields > (size - 16) / 8 {
        return Err("invalid intrinsic field count".into());
    }
    let output = Output::new(Path::new(&args[2]))?;
    let mut out = BufWriter::new(File::create_new(output.path()).map_err(|e| e.to_string())?);
    out.write_all(b"BRTCNT01")
        .and_then(|_| out.write_all(&fields.to_le_bytes()))
        .map_err(|e| e.to_string())?;
    let mut consumed = 16u64;
    let mut raw = Vec::new();
    for _ in 0..fields {
        if size - consumed < 8 {
            return Err("truncated intrinsic field length".into());
        }
        let length = u64_read(&mut input)?;
        consumed += 8;
        if length > size - consumed {
            return Err("truncated intrinsic field".into());
        }
        raw.resize(
            usize::try_from(length).map_err(|_| "intrinsic field too large")?,
            0,
        );
        input.read_exact(&mut raw).map_err(|e| e.to_string())?;
        consumed += length;
        for n in count(&tokenizer, &raw)? {
            out.write_all(&n.to_le_bytes()).map_err(|e| e.to_string())?;
        }
    }
    if consumed != size {
        return Err("trailing intrinsic field bytes".into());
    }
    out.flush().map_err(|e| e.to_string())?;
    drop(out);
    output.commit()
}

fn main() {
    if let Err(error) = run() {
        eprintln!("tokenizer-field-counts: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn intrinsic_counts_preserve_raw_fields_and_distinguish_merged_endings() {
        let mut tokens = bre_pair::huffman_dictionary::byte_model().tokens;
        tokens.push(bre_pair::tokenizer::Token {
            bytes: b"ab".to_vec(),
            ends_field: true,
        });
        let parser = Tokenizer::new(Model::new(tokens).unwrap()).unwrap();
        assert_eq!(count(&parser, b"ab").unwrap(), [1, 1, 0]);
        assert_eq!(count(&parser, b"").unwrap(), [1, 0, 0]);
        assert_eq!(count(&parser, b"a\nb").unwrap(), [4, 3, 3]);
        assert_eq!(count(&parser, &[0, 255]).unwrap(), [3, 2, 2]);
    }
}
