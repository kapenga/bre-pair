//! Standalone fixed-width token streams. Header plus N-bit IDs, with direct
//! token-index seeking. The fingerprint detects accidental model mismatches;
//! it is deliberately not a cryptographic integrity check.
use std::fs::{File, OpenOptions};
use std::io::{self, BufRead, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

use crate::archive::Reader;
use crate::packed::symbol_width;
use crate::tokenizer::{fingerprint, Model, Tokenizer};

const MAGIC: &[u8; 8] = b"BRTOKS01";
pub const HEADER_BYTES: u64 = 48;

#[derive(Clone, Copy, Debug)]
pub struct Header {
    pub model_fingerprint: u64,
    pub vocabulary: u32,
    pub width: u8,
    pub bytes: u64,
    pub fields: u64,
    pub tokens: u64,
}
impl Header {
    fn to_bytes(self) -> Vec<u8> {
        let mut data = MAGIC.to_vec();
        data.extend(self.model_fingerprint.to_le_bytes());
        data.extend(self.vocabulary.to_le_bytes());
        data.push(self.width);
        data.extend([0; 3]);
        data.extend(self.bytes.to_le_bytes());
        data.extend(self.fields.to_le_bytes());
        data.extend(self.tokens.to_le_bytes());
        data
    }
    pub fn payload_bytes(self) -> Result<u64, String> {
        u64::try_from((self.tokens as u128 * self.width as u128).div_ceil(8))
            .map_err(|_| "token payload size overflow".into())
    }
    fn read(source: &mut impl Read) -> Result<Self, String> {
        let mut data = [0u8; HEADER_BYTES as usize];
        source.read_exact(&mut data).map_err(|e| e.to_string())?;
        let mut r = Reader {
            bytes: &data,
            at: 0,
        };
        if r.take(8)? != MAGIC {
            return Err("not a BRTOKS01 stream".into());
        }
        let model_fingerprint = r.u64()?;
        let vocabulary = r.u32()?;
        let width = r.take(1)?[0];
        if r.take(3)? != [0, 0, 0]
            || vocabulary < 257
            || !(9..=32).contains(&width)
            || width != symbol_width(vocabulary as usize)
        {
            return Err("invalid token stream header".into());
        }
        Ok(Self {
            model_fingerprint,
            vocabulary,
            width,
            bytes: r.u64()?,
            fields: r.u64()?,
            tokens: r.u64()?,
        })
    }
}

struct BitsOut<W: Write> {
    target: W,
    word: u64,
    bits: u8,
    width: u8,
}
impl<W: Write> BitsOut<W> {
    fn new(target: W, width: u8) -> Self {
        Self {
            target,
            word: 0,
            bits: 0,
            width,
        }
    }
    fn push(&mut self, id: u32) -> io::Result<()> {
        let room = 64 - self.bits;
        self.word |= (id as u64) << self.bits;
        if self.width < room {
            self.bits += self.width;
        } else {
            self.target.write_all(&self.word.to_le_bytes())?;
            self.word = if self.width == room {
                0
            } else {
                (id as u64) >> room
            };
            self.bits = self.width - room;
        }
        Ok(())
    }
    fn finish(mut self) -> io::Result<W> {
        if self.bits > 0 {
            self.target
                .write_all(&self.word.to_le_bytes()[..self.bits.div_ceil(8) as usize])?;
        }
        self.target.flush()?;
        Ok(self.target)
    }
}

struct BitsIn<R: Read> {
    source: R,
    word: u64,
    bits: u8,
    width: u8,
    remaining: u64,
}
impl<R: Read> BitsIn<R> {
    fn next(&mut self) -> Result<u32, String> {
        let mask = (1u64 << self.width) - 1;
        if self.bits >= self.width {
            let value = self.word & mask;
            self.word >>= self.width;
            self.bits -= self.width;
            return Ok(value as u32);
        }
        let mut data = [0u8; 8];
        let n = self.remaining.min(8) as usize;
        self.source
            .read_exact(&mut data[..n])
            .map_err(|e| e.to_string())?;
        self.remaining -= n as u64;
        let next = u64::from_le_bytes(data);
        let needed = self.width - self.bits;
        if n * 8 < needed as usize {
            return Err("truncated token payload".into());
        }
        let value = self.word | ((next & ((1u64 << needed) - 1)) << self.bits);
        self.word = next >> needed;
        self.bits = (n * 8) as u8 - needed;
        Ok(value as u32)
    }
    fn finish(self) -> Result<(), String> {
        if self.remaining != 0 || self.bits >= 8 || self.word != 0 {
            return Err("nonzero or excessive token padding".into());
        }
        Ok(())
    }
}

pub fn encode_file(tokenizer: &Tokenizer, input: &Path, output: &Path) -> Result<Header, String> {
    let mut source = BufReader::new(File::open(input).map_err(|e| e.to_string())?);
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)
        .map_err(|e| e.to_string())?;
    let mut header = Header {
        model_fingerprint: fingerprint(&tokenizer.model.to_bytes()),
        vocabulary: tokenizer.model.tokens.len() as u32,
        width: symbol_width(tokenizer.model.tokens.len()),
        bytes: 0,
        fields: 0,
        tokens: 0,
    };
    file.write_all(&header.to_bytes())
        .map_err(|e| e.to_string())?;
    let mut bits = BitsOut::new(BufWriter::new(&mut file), header.width);
    let mut field = Vec::new();
    while source
        .read_until(b'\n', &mut field)
        .map_err(|e| e.to_string())?
        > 0
    {
        let ids = tokenizer.encode(&field);
        for id in &ids {
            bits.push(*id).map_err(|e| e.to_string())?;
        }
        header.bytes += field.len() as u64;
        header.fields += 1;
        header.tokens += ids.len() as u64;
        field.clear();
    }
    drop(bits.finish().map_err(|e| e.to_string())?);
    file.seek(SeekFrom::Start(0)).map_err(|e| e.to_string())?;
    file.write_all(&header.to_bytes())
        .map_err(|e| e.to_string())?;
    Ok(header)
}

pub struct TokenReader {
    file: File,
    pub header: Header,
}
impl TokenReader {
    pub fn open(path: &Path, model: &Model) -> Result<Self, String> {
        let mut file = File::open(path).map_err(|e| e.to_string())?;
        let header = Header::read(&mut file)?;
        if header.model_fingerprint != fingerprint(&model.to_bytes())
            || header.vocabulary as usize != model.tokens.len()
        {
            return Err("token stream model mismatch".into());
        }
        let expected = HEADER_BYTES
            .checked_add(header.payload_bytes()?)
            .ok_or("stream size overflow")?;
        if file.metadata().map_err(|e| e.to_string())?.len() != expected {
            return Err("token stream length mismatch".into());
        }
        Ok(Self { file, header })
    }

    /// O(1) seek to an ID; the model and header are checked once on open.
    pub fn token_at(&mut self, index: u64) -> Result<u32, String> {
        if index >= self.header.tokens {
            return Err("token index out of range".into());
        }
        let bit = index as u128 * self.header.width as u128;
        let offset = u64::try_from(bit / 8).map_err(|_| "token offset overflow")?;
        let shift = (bit % 8) as u8;
        let take = (shift + self.header.width).div_ceil(8) as usize;
        self.file
            .seek(SeekFrom::Start(HEADER_BYTES + offset))
            .map_err(|e| e.to_string())?;
        let mut bytes = [0u8; 8];
        self.file
            .read_exact(&mut bytes[..take])
            .map_err(|e| e.to_string())?;
        let id = ((u64::from_le_bytes(bytes) >> shift) & ((1u64 << self.header.width) - 1)) as u32;
        if id >= self.header.vocabulary {
            return Err("undefined token ID".into());
        }
        Ok(id)
    }

    pub fn decode_to(&mut self, model: &Model, target: &mut impl Write) -> Result<(), String> {
        if self.header.model_fingerprint != fingerprint(&model.to_bytes()) {
            return Err("token stream model mismatch".into());
        }
        self.file
            .seek(SeekFrom::Start(HEADER_BYTES))
            .map_err(|e| e.to_string())?;
        let mut bits = BitsIn {
            source: BufReader::new(&mut self.file),
            word: 0,
            bits: 0,
            width: self.header.width,
            remaining: self.header.payload_bytes()?,
        };
        let (mut bytes, mut fields, mut ended) = (0u64, 0u64, true);
        for _ in 0..self.header.tokens {
            let id = bits.next()?;
            let token = model.tokens.get(id as usize).ok_or("undefined token ID")?;
            bytes = bytes
                .checked_add(token.bytes.len() as u64)
                .ok_or("decoded length overflow")?;
            if bytes > self.header.bytes {
                return Err("decoded length exceeds header".into());
            }
            target.write_all(&token.bytes).map_err(|e| e.to_string())?;
            ended = token.ends_field;
            if ended {
                fields += 1;
            }
            if fields > self.header.fields {
                return Err("too many field endings".into());
            }
        }
        bits.finish()?;
        if bytes != self.header.bytes || fields != self.header.fields || !ended {
            return Err("decoded lengths or field endings differ from header".into());
        }
        target.flush().map_err(|e| e.to_string())
    }
}

pub fn decode_file(model: &Model, input: &Path, output: &Path) -> Result<Header, String> {
    let mut reader = TokenReader::open(input, model)?;
    let mut target = BufWriter::new(File::create_new(output).map_err(|e| e.to_string())?);
    reader.decode_to(model, &mut target)?;
    Ok(reader.header)
}

pub fn verify_file(model: &Model, input: &Path, original: &Path) -> Result<Header, String> {
    struct Compare {
        source: BufReader<File>,
        scratch: Vec<u8>,
    }
    impl Write for Compare {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.scratch.resize(bytes.len(), 0);
            self.source.read_exact(&mut self.scratch)?;
            if self.scratch != bytes {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "decoded bytes differ from source",
                ));
            }
            Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let mut reader = TokenReader::open(input, model)?;
    let source = File::open(original).map_err(|e| e.to_string())?;
    if source.metadata().map_err(|e| e.to_string())?.len() != reader.header.bytes {
        return Err("source length differs from token stream".into());
    }
    let mut compare = Compare {
        source: BufReader::new(source),
        scratch: Vec::new(),
    };
    reader.decode_to(model, &mut compare)?;
    Ok(reader.header)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn token_stream_sizes_preserve_counts_above_u32() {
        let header = Header {
            model_fingerprint: 0,
            vocabulary: 8192,
            width: 13,
            bytes: 1u64 << 40,
            fields: 1u64 << 33,
            tokens: (1u64 << 35) + 1,
        };
        assert_eq!(
            header.payload_bytes().unwrap(),
            ((1u64 << 35) + 1) * 13 / 8 + 1
        );
        let decoded = Header::read(&mut header.to_bytes().as_slice()).unwrap();
        assert_eq!(decoded.tokens, header.tokens);
        assert_eq!(decoded.fields, header.fields);
        assert_eq!(decoded.bytes, header.bytes);
    }
    #[test]
    fn bit_transport_round_trips_every_width_and_detects_padding() {
        for width in 1..=32 {
            let mask = (1u64 << width) - 1;
            for count in 0..=129 {
                let values: Vec<_> = (0..count)
                    .map(|i| ((i as u64 * 1234567 + 891) & mask) as u32)
                    .collect();
                let mut out = BitsOut::new(Vec::new(), width);
                for &value in &values {
                    out.push(value).unwrap();
                }
                let bytes = out.finish().unwrap();
                assert_eq!(bytes.len(), (count * width as usize).div_ceil(8));
                let mut input = BitsIn {
                    source: bytes.as_slice(),
                    word: 0,
                    bits: 0,
                    width,
                    remaining: bytes.len() as u64,
                };
                for &expected in &values {
                    assert_eq!(input.next().unwrap(), expected);
                }
                input.finish().unwrap();
                if count * width as usize % 8 != 0 {
                    let mut bad = bytes.clone();
                    *bad.last_mut().unwrap() |= 128;
                    let mut input = BitsIn {
                        source: bad.as_slice(),
                        word: 0,
                        bits: 0,
                        width,
                        remaining: bad.len() as u64,
                    };
                    for _ in &values {
                        input.next().unwrap();
                    }
                    assert!(input.finish().is_err());
                }
            }
        }
    }
}
