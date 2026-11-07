//! Framed, byte-exact research adapter. Production tokenizer behavior is reused.
use std::borrow::Cow;
use std::fs;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::time::Instant;

use bre_pair::tokenizer::{GreedyParser, Model, ParseScratch, Tokenizer};

const MAX_FIELD: u64 = 64 * 1024 * 1024;

enum Parser {
    Exact(Tokenizer),
    Fragment(Tokenizer),
    Greedy(GreedyParser),
}

impl Parser {
    fn model(&self) -> &Model {
        match self {
            Self::Exact(p) | Self::Fragment(p) => p.model(),
            Self::Greedy(p) => p.model(),
        }
    }

    fn encode<'a>(&self, raw: &[u8], scratch: &'a mut ParseScratch) -> Cow<'a, [u32]> {
        match self {
            Self::Exact(p) => Cow::Owned(p.encode(raw)),
            Self::Fragment(p) => Cow::Owned(p.encode_fragment(raw)),
            Self::Greedy(p) => Cow::Borrowed(p.encode(raw, scratch)),
        }
    }
}

fn serve(parser: Parser, input: &mut impl Read, output: &mut impl Write) -> Result<(), String> {
    output.write_all(b"BRTPRB01").map_err(|e| e.to_string())?;
    output.flush().map_err(|e| e.to_string())?;
    let mut raw = Vec::new();
    let mut scratch = ParseScratch::default();
    loop {
        let mut length = [0; 8];
        match input.read(&mut length[..1]).map_err(|e| e.to_string())? {
            0 => break,
            1 => {}
            _ => unreachable!(),
        }
        input
            .read_exact(&mut length[1..])
            .map_err(|e| e.to_string())?;
        let size = u64::from_le_bytes(length);
        if size == u64::MAX {
            break;
        }
        if size > MAX_FIELD {
            return Err("research field exceeds 64 MiB limit".into());
        }
        raw.resize(
            usize::try_from(size).map_err(|_| "field exceeds address space")?,
            0,
        );
        input.read_exact(&mut raw).map_err(|e| e.to_string())?;
        let start = Instant::now();
        let ids = parser.encode(&raw, &mut scratch);
        let encode_ns = u64::try_from(start.elapsed().as_nanos()).map_err(|_| "timer overflow")?;
        let start = Instant::now();
        let decoded = if matches!(parser, Parser::Fragment(_)) {
            let mut result = Vec::new();
            for &id in ids.iter() {
                let token = &parser.model().tokens[id as usize];
                if token.ends_field {
                    return Err("fragment parser emitted a field ending".into());
                }
                result.extend_from_slice(&token.bytes);
            }
            result
        } else {
            parser.model().decode_field(&ids)?
        };
        let decode_ns = u64::try_from(start.elapsed().as_nanos()).map_err(|_| "timer overflow")?;
        if decoded != raw {
            return Err("research round trip differs".into());
        }
        for value in [ids.len() as u64, encode_ns, decode_ns] {
            output
                .write_all(&value.to_le_bytes())
                .map_err(|e| e.to_string())?;
        }
        for id in ids.iter() {
            output
                .write_all(&id.to_le_bytes())
                .map_err(|e| e.to_string())?;
        }
        output.flush().map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn run() -> Result<(), String> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 2 {
        return Err("tokenizer-paper-probe MODEL.brt exact|greedy|fragment".into());
    }
    let model = Model::from_bytes(&fs::read(&args[0]).map_err(|e| e.to_string())?)?;
    let parser = match args[1].as_str() {
        "exact" => Parser::Exact(Tokenizer::new(model)?),
        "fragment" => Parser::Fragment(Tokenizer::new(model)?),
        "greedy" => Parser::Greedy(GreedyParser::new(model)?),
        _ => return Err("parser must be exact, greedy or fragment".into()),
    };
    serve(
        parser,
        &mut BufReader::new(io::stdin().lock()),
        &mut BufWriter::new(io::stdout().lock()),
    )
}

fn main() {
    if let Err(error) = run() {
        eprintln!("tokenizer-paper-probe: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bre_pair::huffman_dictionary::byte_model;
    use bre_pair::tokenizer::Token;

    fn response(raw: &[u8], greedy: bool) -> Vec<u32> {
        let mut model = byte_model();
        for s in [b"ab".as_slice(), b"bc", b"bcd"] {
            model.tokens.push(Token {
                bytes: s.to_vec(),
                ends_field: false,
            });
        }
        let parser = if greedy {
            Parser::Greedy(GreedyParser::new(model).unwrap())
        } else {
            Parser::Exact(Tokenizer::new(model).unwrap())
        };
        let mut request = (raw.len() as u64).to_le_bytes().to_vec();
        request.extend(raw);
        let mut result = Vec::new();
        serve(parser, &mut request.as_slice(), &mut result).unwrap();
        assert_eq!(&result[..8], b"BRTPRB01");
        let count = u64::from_le_bytes(result[8..16].try_into().unwrap()) as usize;
        assert_eq!(result.len(), 32 + count * 4);
        result[32..]
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn binary_empty_newlines_and_modes_are_exact() {
        for raw in [Vec::new(), (0..=255).collect(), b"a\0\n\xff\r\nb".to_vec()] {
            for mode in [false, true] {
                let ids = response(&raw, mode);
                assert_eq!(ids.last(), Some(&256));
                assert_eq!(ids.iter().filter(|&&id| id == 256).count(), 1);
                assert_eq!(response(&raw, mode), ids);
            }
        }
        assert_eq!(response(b"abcd", false).len(), 3);
        assert_eq!(response(b"abcd", true).len(), 4);
    }

    #[test]
    fn truncated_or_oversized_frames_fail() {
        for request in [
            (MAX_FIELD + 1).to_le_bytes().to_vec(),
            vec![1],
            3u64.to_le_bytes().to_vec(),
        ] {
            let parser = Parser::Exact(Tokenizer::new(byte_model()).unwrap());
            assert!(serve(parser, &mut request.as_slice(), &mut Vec::new()).is_err());
        }
    }

    #[test]
    fn fragments_cannot_use_field_final_tokens() {
        let mut model = byte_model();
        model.tokens.push(Token {
            bytes: b"ab".to_vec(),
            ends_field: true,
        });
        let parser = Tokenizer::new(model).unwrap();
        assert_eq!(parser.encode(b"ab"), vec![257]);
        assert_eq!(parser.encode_fragment(b"ab"), vec![97, 98]);
        assert!(parser.encode_fragment(b"").is_empty());
    }
}
