//! Prototype format: fixed-width data and progressively packed binary rules.
//! It deliberately contains no Huffman model or entropy-coded payload.
use crate::engine::Compressed;
use crate::grammar::{Grammar, BASE_SYMBOLS};
use crate::packed::{symbol_width, PackedSymbols};

const MAGIC: &[u8; 8] = b"BRPAIR01";

#[derive(Default)]
pub(crate) struct BitWriter {
    pub(crate) bytes: Vec<u8>,
    pub(crate) bits: usize,
}
impl BitWriter {
    pub(crate) fn push(&mut self, value: u32, width: u8) {
        for bit in 0..width {
            if self.bits % 8 == 0 {
                self.bytes.push(0);
            }
            self.bytes[self.bits / 8] |= (((value >> bit) & 1) as u8) << (self.bits % 8);
            self.bits += 1;
        }
    }
}

pub(crate) struct Reader<'a> {
    pub(crate) bytes: &'a [u8],
    pub(crate) at: usize,
}
impl<'a> Reader<'a> {
    pub(crate) fn take(&mut self, len: usize) -> Result<&'a [u8], String> {
        let end = self.at.checked_add(len).ok_or("archive length overflow")?;
        let bytes = self.bytes.get(self.at..end).ok_or("truncated archive")?;
        self.at = end;
        Ok(bytes)
    }
    pub(crate) fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    pub(crate) fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
}

pub fn encode(compressed: &Compressed) -> Result<Vec<u8>, String> {
    let rule_count = u32::try_from(compressed.grammar.rules.len()).map_err(|_| "too many rules")?;
    let block_count = u32::try_from(compressed.blocks.len()).map_err(|_| "too many blocks")?;
    let mut rules = BitWriter::default();
    for (i, rule) in compressed.grammar.rules.iter().enumerate() {
        let width = symbol_width(BASE_SYMBOLS + i);
        rules.push(rule.left, width);
        rules.push(rule.right, width);
    }
    let mut bytes = Vec::new();
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(&compressed.input_bytes.to_le_bytes());
    bytes.extend_from_slice(&compressed.field_count.to_le_bytes());
    bytes.extend_from_slice(&rule_count.to_le_bytes());
    bytes.extend_from_slice(&block_count.to_le_bytes());
    bytes.extend_from_slice(&(rules.bits as u64).to_le_bytes());
    bytes.extend(rules.bytes);
    for block in &compressed.blocks {
        bytes.push(block.width());
        bytes.extend_from_slice(&(block.len() as u64).to_le_bytes());
        bytes.extend(block.to_bytes());
    }
    Ok(bytes)
}

pub fn decode(bytes: &[u8]) -> Result<Compressed, String> {
    let mut reader = Reader { bytes, at: 0 };
    if reader.take(8)? != MAGIC {
        return Err("not a BRPAIR01 archive".into());
    }
    let input_bytes = reader.u64()?;
    let field_count = reader.u64()?;
    let rule_count = reader.u32()? as usize;
    let block_count = reader.u32()? as usize;
    if rule_count > u32::MAX as usize - BASE_SYMBOLS {
        return Err("too many rules".into());
    }
    let rule_bits = usize::try_from(reader.u64()?).map_err(|_| "rule size overflow")?;
    // Validate counts against the file before allocating the grammar.
    if rule_count > bytes.len() / 2 || block_count > bytes.len() / 9 {
        return Err("archive counts exceed file size".into());
    }
    let expected_bits: usize = (0..rule_count)
        .map(|i| 2 * symbol_width(BASE_SYMBOLS + i) as usize)
        .sum();
    if rule_bits != expected_bits {
        return Err("incorrect dictionary bit length".into());
    }
    let rule_bytes = reader.take(rule_bits.div_ceil(8))?;
    let mut bit_at = 0;
    let mut read_symbol = |width: u8| {
        let mut value = 0;
        for bit in 0..width {
            value |= (((rule_bytes[bit_at / 8] >> (bit_at % 8)) & 1) as u32) << bit;
            bit_at += 1;
        }
        value
    };
    let mut grammar = Grammar::default();
    for i in 0..rule_count {
        let width = symbol_width(BASE_SYMBOLS + i);
        let left = read_symbol(width);
        let right = read_symbol(width);
        let symbol = grammar.add((left, right))?;
        if symbol as usize != BASE_SYMBOLS + i {
            return Err("duplicate dictionary rule".into());
        }
    }
    let mut blocks = Vec::with_capacity(block_count);
    for _ in 0..block_count {
        let width = reader.take(1)?[0];
        if !(9..=32).contains(&width) {
            return Err("invalid block symbol width".into());
        }
        let len = usize::try_from(reader.u64()?).map_err(|_| "block length overflow")?;
        let bits = len
            .checked_mul(width as usize)
            .ok_or("block size overflow")?;
        let payload = reader.take(bits.div_ceil(8))?;
        let block = PackedSymbols::from_bytes(width, len, payload)?;
        if block.iter().any(|s| s as usize >= grammar.alphabet_size()) {
            return Err("block refers to an undefined symbol".into());
        }
        blocks.push(block);
    }
    if reader.at != bytes.len() {
        return Err("trailing bytes after archive".into());
    }
    Ok(Compressed {
        grammar,
        blocks,
        input_bytes,
        field_count,
    })
}
