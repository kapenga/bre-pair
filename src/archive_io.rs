//! Streaming BRPAIR01 serialization and byte-exact decoding. No flattened
//! corpus copy, complete serialized archive, or complete decoded output in RAM.
use crate::{
    engine::Compressed,
    grammar::{Grammar, BASE_SYMBOLS, EOF},
    packed::symbol_width,
};
use std::io::{Read, Seek, SeekFrom, Write};

struct BitsOut<'a, W: Write> {
    out: &'a mut W,
    word: u64,
    bits: u8,
}
impl<'a, W: Write> BitsOut<'a, W> {
    fn new(out: &'a mut W) -> Self {
        Self {
            out,
            word: 0,
            bits: 0,
        }
    }
    fn push(&mut self, value: u32, width: u8) -> Result<(), String> {
        self.word |= (value as u64) << self.bits;
        let total = self.bits + width;
        if total >= 64 {
            self.out
                .write_all(&self.word.to_le_bytes())
                .map_err(|e| e.to_string())?;
            self.word = if self.bits == 0 {
                0
            } else {
                (value as u64) >> (64 - self.bits)
            };
            self.bits = total - 64;
        } else {
            self.bits = total;
        }
        Ok(())
    }
    fn finish(self) -> Result<(), String> {
        self.out
            .write_all(&self.word.to_le_bytes()[..(self.bits as usize).div_ceil(8)])
            .map_err(|e| e.to_string())
    }
}

struct BitsIn<'a, R: Read> {
    input: &'a mut R,
    word: u64,
    bits: u8,
}
impl<'a, R: Read> BitsIn<'a, R> {
    fn new(input: &'a mut R) -> Self {
        Self {
            input,
            word: 0,
            bits: 0,
        }
    }
    fn get(&mut self, width: u8) -> Result<u32, String> {
        while self.bits < width {
            let mut byte = [0];
            self.input
                .read_exact(&mut byte)
                .map_err(|e| e.to_string())?;
            self.word |= (byte[0] as u64) << self.bits;
            self.bits += 8;
        }
        let value = (self.word & ((1u64 << width) - 1)) as u32;
        self.word >>= width;
        self.bits -= width;
        Ok(value)
    }
    fn finish(self) -> Result<(), String> {
        if self.word != 0 {
            return Err("nonzero archive padding".into());
        }
        Ok(())
    }
}

fn dictionary_bits(rules: u32) -> u64 {
    let start = BASE_SYMBOLS as u64;
    let end = start + rules as u64;
    let mut bits = 0;
    for width in 1..=32 {
        let lower = ((1u64 << (width - 1)) + 1).max(start);
        let upper = ((1u64 << width) + 1).min(end);
        bits += upper.saturating_sub(lower) * 2 * width;
    }
    bits
}

/// Reload a saved grammar into EOF-aligned packed chunks without retaining
/// its serialized bytes or expanding it to the original byte corpus.
pub fn read_packed(
    input: &mut (impl Read + Seek),
    chunk_symbols: usize,
) -> Result<Compressed, String> {
    if chunk_symbols == 0 {
        return Err("chunk size must be positive".into());
    }
    let length = input.seek(SeekFrom::End(0)).map_err(|e| e.to_string())?;
    input.seek(SeekFrom::Start(0)).map_err(|e| e.to_string())?;
    let mut header = [0u8; 40];
    input.read_exact(&mut header).map_err(|e| e.to_string())?;
    if &header[..8] != b"BRPAIR01" {
        return Err("not a BRPAIR01 archive".into());
    }
    let input_bytes = u64::from_le_bytes(header[8..16].try_into().unwrap());
    let field_count = u64::from_le_bytes(header[16..24].try_into().unwrap());
    let rules = u32::from_le_bytes(header[24..28].try_into().unwrap());
    let blocks = u32::from_le_bytes(header[28..32].try_into().unwrap());
    let bits = u64::from_le_bytes(header[32..40].try_into().unwrap());
    if u64::from(rules) > length / 2
        || u64::from(blocks) > length / 9
        || rules as usize > u32::MAX as usize - BASE_SYMBOLS
        || bits != dictionary_bits(rules)
    {
        return Err("archive counts or dictionary bits exceed the file".into());
    }
    let mut grammar = Grammar::default();
    let mut reader = BitsIn::new(input);
    for i in 0..rules as usize {
        let width = symbol_width(BASE_SYMBOLS + i);
        let a = reader.get(width)?;
        let b = reader.get(width)?;
        if grammar.add((a, b))? as usize != BASE_SYMBOLS + i {
            return Err("duplicate dictionary rule".into());
        }
    }
    // Release the reader's borrow before parsing byte-aligned block headers.
    reader.finish()?;
    let width = symbol_width(grammar.alphabet_size());
    let mut packed = crate::packed::PackedSymbols::with_capacity(width, chunk_symbols);
    let mut output = Vec::new();
    let (mut bytes, mut fields, mut ended) = (0u64, 0u64, true);
    for _ in 0..blocks {
        let mut h = [0; 9];
        input.read_exact(&mut h).map_err(|e| e.to_string())?;
        let source_width = h[0];
        let count = u64::from_le_bytes(h[1..].try_into().unwrap());
        if !(9..=32).contains(&source_width)
            || count > length.saturating_mul(8) / u64::from(source_width)
        {
            return Err("invalid packed block width or length".into());
        }
        let mut reader = BitsIn::new(input);
        for _ in 0..count {
            let symbol = reader.get(source_width)?;
            if symbol as usize >= grammar.alphabet_size() {
                return Err("packed block refers to an undefined symbol".into());
            }
            ended = grammar.ends_field(symbol);
            let terminals = grammar.terminals(symbol) as u64;
            bytes = bytes
                .checked_add(terminals - u64::from(ended))
                .ok_or("decoded byte count overflow")?;
            fields = fields
                .checked_add(u64::from(ended))
                .ok_or("field count overflow")?;
            packed.push(symbol);
            if ended && packed.len() >= chunk_symbols {
                packed.shrink_to_fit();
                output.push(std::mem::replace(
                    &mut packed,
                    crate::packed::PackedSymbols::with_capacity(width, chunk_symbols),
                ));
            }
        }
        reader.finish()?;
    }
    if !ended || (bytes, fields) != (input_bytes, field_count) {
        return Err("saved grammar stream totals differ".into());
    }
    if input.stream_position().map_err(|e| e.to_string())? != length {
        return Err("trailing bytes after archive".into());
    }
    if !packed.is_empty() {
        packed.shrink_to_fit();
        output.push(packed);
    }
    Ok(Compressed {
        grammar,
        blocks: output,
        input_bytes,
        field_count,
    })
}

pub fn write(compressed: &Compressed, out: &mut impl Write) -> Result<(), String> {
    let rules =
        u32::try_from(compressed.grammar.rules.len()).map_err(|_| "too many archive rules")?;
    let symbols = compressed.token_count() as u64;
    out.write_all(b"BRPAIR01")
        .and_then(|_| out.write_all(&compressed.input_bytes.to_le_bytes()))
        .and_then(|_| out.write_all(&compressed.field_count.to_le_bytes()))
        .and_then(|_| out.write_all(&rules.to_le_bytes()))
        .and_then(|_| out.write_all(&(u32::from(symbols != 0)).to_le_bytes()))
        .and_then(|_| out.write_all(&dictionary_bits(rules).to_le_bytes()))
        .map_err(|e| e.to_string())?;
    let mut bits = BitsOut::new(out);
    for (index, rule) in compressed.grammar.rules.iter().enumerate() {
        let width = symbol_width(BASE_SYMBOLS + index);
        bits.push(rule.left, width)?;
        bits.push(rule.right, width)?;
    }
    bits.finish()?;
    if symbols != 0 {
        let width = symbol_width(compressed.grammar.alphabet_size());
        out.write_all(&[width])
            .and_then(|_| out.write_all(&symbols.to_le_bytes()))
            .map_err(|e| e.to_string())?;
        let mut bits = BitsOut::new(out);
        for block in &compressed.blocks {
            for symbol in block.iter() {
                bits.push(symbol, width)?;
            }
        }
        bits.finish()?;
    }
    out.flush().map_err(|e| e.to_string())
}

fn u64_from(input: &mut impl Read) -> Result<u64, String> {
    let mut bytes = [0; 8];
    input.read_exact(&mut bytes).map_err(|e| e.to_string())?;
    Ok(u64::from_le_bytes(bytes))
}
fn u32_from(input: &mut impl Read) -> Result<u32, String> {
    let mut bytes = [0; 4];
    input.read_exact(&mut bytes).map_err(|e| e.to_string())?;
    Ok(u32::from_le_bytes(bytes))
}

#[derive(Debug)]
pub struct Decoded {
    pub bytes: u64,
    pub fields: u64,
    pub rules: u32,
}

/// Sequential grammar-symbol reader for entropy-coding an archive without
/// materializing its packed corpus or decoded bytes. Counters stay u64.
pub struct SymbolReader<R> {
    input: R,
    pub grammar: Grammar,
    pub expected_bytes: u64,
    pub expected_fields: u64,
    alphabet_size: usize,
    blocks: u32,
    initial_blocks: u32,
    data_offset: u64,
    remaining: u64,
    width: u8,
    word: u64,
    bits: u8,
    finished: bool,
}
impl<R: Read> SymbolReader<R> {
    pub fn new(mut input: R) -> Result<Self, String> {
        let mut magic = [0; 8];
        input.read_exact(&mut magic).map_err(|e| e.to_string())?;
        if &magic != b"BRPAIR01" {
            return Err("not a BRPAIR01 archive".into());
        }
        let expected_bytes = u64_from(&mut input)?;
        let expected_fields = u64_from(&mut input)?;
        let rules = u32_from(&mut input)?;
        let blocks = u32_from(&mut input)?;
        let rule_bits = u64_from(&mut input)?;
        if rules as u64 + BASE_SYMBOLS as u64 > u32::MAX as u64
            || rule_bits != dictionary_bits(rules)
        {
            return Err("invalid archive dictionary length".into());
        }
        let mut grammar = Grammar::default();
        let mut bits = BitsIn::new(&mut input);
        for i in 0..rules as usize {
            let width = symbol_width(BASE_SYMBOLS + i);
            let pair = (bits.get(width)?, bits.get(width)?);
            if grammar.add(pair)? as usize != BASE_SYMBOLS + i {
                return Err("duplicate archive rule".into());
            }
        }
        bits.finish()?;
        let alphabet_size = grammar.alphabet_size();
        Ok(Self {
            input,
            grammar,
            expected_bytes,
            expected_fields,
            alphabet_size,
            blocks,
            initial_blocks: blocks,
            data_offset: 40 + rule_bits.div_ceil(8),
            remaining: 0,
            width: 0,
            word: 0,
            bits: 0,
            finished: false,
        })
    }
    pub fn next_symbol(&mut self) -> Result<Option<u32>, String> {
        if self.finished {
            return Ok(None);
        }
        while self.remaining == 0 {
            if self.word != 0 {
                return Err("nonzero archive padding".into());
            }
            self.word = 0;
            self.bits = 0;
            if self.blocks == 0 {
                let mut tail = [0];
                if self.input.read(&mut tail).map_err(|e| e.to_string())? != 0 {
                    return Err("trailing archive bytes".into());
                }
                self.finished = true;
                return Ok(None);
            }
            let mut width = [0];
            self.input
                .read_exact(&mut width)
                .map_err(|e| e.to_string())?;
            self.width = width[0];
            if !(1..=32).contains(&self.width) {
                return Err("invalid symbol width".into());
            }
            self.remaining = u64_from(&mut self.input)?;
            self.blocks -= 1;
        }
        while self.bits < self.width {
            let mut byte = [0];
            self.input
                .read_exact(&mut byte)
                .map_err(|e| e.to_string())?;
            self.word |= u64::from(byte[0]) << self.bits;
            self.bits += 8;
        }
        let id = (self.word & ((1u64 << self.width) - 1)) as u32;
        self.word >>= self.width;
        self.bits -= self.width;
        self.remaining -= 1;
        if id as usize >= self.alphabet_size {
            return Err("undefined archive symbol".into());
        }
        Ok(Some(id))
    }
}

impl<R: Read + std::io::Seek> SymbolReader<R> {
    /// Reuse the parsed grammar for a second streaming pass over symbols.
    pub fn rewind_symbols(&mut self) -> Result<(), String> {
        self.input
            .seek(std::io::SeekFrom::Start(self.data_offset))
            .map_err(|e| e.to_string())?;
        self.blocks = self.initial_blocks;
        self.remaining = 0;
        self.width = 0;
        self.word = 0;
        self.bits = 0;
        self.finished = false;
        Ok(())
    }
}

pub fn decompress(input: &mut impl Read, out: &mut impl Write) -> Result<Decoded, String> {
    let mut magic = [0; 8];
    input.read_exact(&mut magic).map_err(|e| e.to_string())?;
    if &magic != b"BRPAIR01" {
        return Err("not a BRPAIR01 archive".into());
    }
    let expected_bytes = u64_from(input)?;
    let expected_fields = u64_from(input)?;
    let rule_count = u32_from(input)?;
    let blocks = u32_from(input)?;
    let rule_bits = u64_from(input)?;
    if rule_count as u64 + BASE_SYMBOLS as u64 > u32::MAX as u64
        || rule_bits != dictionary_bits(rule_count)
    {
        return Err("invalid archive dictionary length".into());
    }
    let mut grammar = Grammar::default();
    let mut bits = BitsIn::new(input);
    for index in 0..rule_count as usize {
        let width = symbol_width(BASE_SYMBOLS + index);
        let pair = (bits.get(width)?, bits.get(width)?);
        if grammar.add(pair)? as usize != BASE_SYMBOLS + index {
            return Err("duplicate archive rule".into());
        }
    }
    bits.finish()?;
    let mut bytes = 0u64;
    let mut fields = 0u64;
    let mut ended = true;
    let mut buffer = Vec::with_capacity(65536);
    let mut stack = Vec::new();
    for _ in 0..blocks {
        let mut width = [0];
        input.read_exact(&mut width).map_err(|e| e.to_string())?;
        let width = width[0];
        if !(1..=32).contains(&width) {
            return Err("invalid symbol width".into());
        }
        let count = u64_from(input)?;
        let mut bits = BitsIn::new(input);
        for _ in 0..count {
            let symbol = bits.get(width)?;
            if symbol as usize >= grammar.alphabet_size() {
                return Err("undefined archive symbol".into());
            }
            stack.push(symbol);
            while let Some(symbol) = stack.pop() {
                if symbol == EOF {
                    fields = fields.checked_add(1).ok_or("field count overflow")?;
                    ended = true;
                    if fields > expected_fields {
                        return Err("archive exceeds declared field count".into());
                    }
                } else if symbol < EOF {
                    bytes = bytes.checked_add(1).ok_or("decoded byte count overflow")?;
                    ended = false;
                    if bytes > expected_bytes {
                        return Err("archive exceeds declared byte count".into());
                    }
                    buffer.push(symbol as u8);
                    if buffer.len() == 65536 {
                        out.write_all(&buffer).map_err(|e| e.to_string())?;
                        buffer.clear();
                    }
                } else {
                    let rule = &grammar.rules[symbol as usize - BASE_SYMBOLS];
                    stack.push(rule.right);
                    stack.push(rule.left);
                }
            }
        }
        bits.finish()?;
    }
    let mut tail = [0];
    if input.read(&mut tail).map_err(|e| e.to_string())? != 0 {
        return Err("trailing archive bytes".into());
    }
    if bytes != expected_bytes || fields != expected_fields || !ended {
        return Err("archive totals or final EOF differ".into());
    }
    out.write_all(&buffer)
        .and_then(|_| out.flush())
        .map_err(|e| e.to_string())?;
    Ok(Decoded {
        bytes,
        fields,
        rules: rule_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn packed_reload_preserves_grammar_roots_and_eof_boundaries() {
        let fields = vec![
            Vec::new(),
            (0..=255).collect(),
            b"abcabcabc\naaaaa".repeat(20),
        ];
        let trained =
            crate::engine::compress(&fields, &crate::engine::Config::default(), |_| {}).unwrap();
        let mut serialized = Vec::new();
        write(&trained.compressed, &mut serialized).unwrap();
        let original: Vec<_> = trained
            .compressed
            .blocks
            .iter()
            .flat_map(|b| b.iter())
            .collect();
        for chunk in [1, 17, 1 << 20] {
            let loaded = read_packed(&mut Cursor::new(&serialized), chunk).unwrap();
            assert_eq!(loaded.decode().unwrap(), fields);
            assert_eq!(
                loaded
                    .blocks
                    .iter()
                    .flat_map(|b| b.iter())
                    .collect::<Vec<_>>(),
                original
            );
            assert_eq!(
                loaded.grammar.rules.len(),
                trained.compressed.grammar.rules.len()
            );
            assert!(loaded
                .blocks
                .iter()
                .all(|b| loaded.grammar.ends_field(b.get(b.len() - 1))));
            let continued = crate::field_engine::compress_prepacked(
                loaded,
                &crate::engine::Config {
                    minimum_count: 2,
                    search: crate::search::Options {
                        certify: false,
                        ..Default::default()
                    },
                    ..Default::default()
                },
                2,
                chunk,
                crate::field_engine::Options {
                    frequency_cutoff: true,
                    memory_saving: true,
                    ..Default::default()
                },
                |_| {},
            )
            .unwrap();
            assert_eq!(continued.training.compressed.decode().unwrap(), fields);
            for (old, new) in trained
                .compressed
                .grammar
                .rules
                .iter()
                .zip(&continued.training.compressed.grammar.rules)
            {
                assert_eq!((old.left, old.right), (new.left, new.right));
            }
        }
        let mut trailing = serialized.clone();
        trailing.push(0);
        let mut wrong_total = serialized.clone();
        wrong_total[8] ^= 1;
        let mut bad_counts = serialized.clone();
        bad_counts[24..28].copy_from_slice(&u32::MAX.to_le_bytes());
        for malformed in [
            trailing,
            wrong_total,
            bad_counts,
            serialized[..serialized.len() - 1].to_vec(),
        ] {
            assert!(read_packed(&mut Cursor::new(malformed), 17).is_err());
        }
        assert!(read_packed(&mut Cursor::new(serialized), 0).is_err());
    }
    #[test]
    fn progressive_dictionary_lengths_match_original() {
        for count in [0, 1, 255, 256, 257, 1024, 32768, 130815, 524031] {
            let expected: u64 = (0..count as usize)
                .map(|i| 2 * symbol_width(BASE_SYMBOLS + i) as u64)
                .sum();
            assert_eq!(dictionary_bits(count), expected);
        }
    }
}
