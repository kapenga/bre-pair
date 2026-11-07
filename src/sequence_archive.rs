//! Experimental native, variable-arity grammar format. No entropy coding.
//! Same 40-byte header and block framing as BRPAIR01, but magic BRPNAR01.
//! Each rule stores gamma(rhs_length - 1), then children at the width for all
//! preceding symbols. Gamma uses unary leading zeros followed by MSB-first
//! integer bits; child integers use the original LSB-first bit packing.
use crate::archive::{BitWriter, Reader};
use crate::grammar::BASE_SYMBOLS;
use crate::maximal::NativeCompressed;
use crate::packed::{symbol_width, PackedSymbols};

const MAGIC: &[u8; 8] = b"BRPNAR01";

fn dictionary(compressed: &NativeCompressed) -> Result<BitWriter, String> {
    let mut writer = BitWriter::default();
    for (i, rule) in compressed.rules.iter().enumerate() {
        let length = u32::try_from(rule.len()).map_err(|_| "native rule too long")?;
        if length < 2 {
            return Err("native rule must have at least two children".into());
        }
        let value = length - 1;
        let width = 32 - value.leading_zeros();
        for _ in 1..width {
            writer.push(0, 1);
        }
        for bit in (0..width).rev() {
            writer.push((value >> bit) & 1, 1);
        }
        let width = symbol_width(BASE_SYMBOLS + i);
        for &symbol in rule {
            if symbol as usize >= BASE_SYMBOLS + i {
                return Err("native rule refers to an undefined child".into());
            }
            writer.push(symbol, width);
        }
    }
    Ok(writer)
}

pub fn dictionary_bits(compressed: &NativeCompressed) -> Result<usize, String> {
    Ok(dictionary(compressed)?.bits)
}

pub fn encode(compressed: &NativeCompressed) -> Result<Vec<u8>, String> {
    let rule_count = u32::try_from(compressed.rules.len()).map_err(|_| "too many native rules")?;
    let block_count = u32::try_from(compressed.blocks.len()).map_err(|_| "too many blocks")?;
    let rules = dictionary(compressed)?;
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

struct Bits<'a> {
    bytes: &'a [u8],
    at: usize,
    length: usize,
}

impl Bits<'_> {
    fn read(&mut self, width: u8) -> Result<u32, String> {
        if self.length - self.at < width as usize {
            return Err("truncated native rule".into());
        }
        let mut value = 0;
        for bit in 0..width {
            value |= (((self.bytes[self.at / 8] >> (self.at % 8)) & 1) as u32) << bit;
            self.at += 1;
        }
        Ok(value)
    }

    fn arity(&mut self) -> Result<usize, String> {
        let mut zeros = 0;
        while self.read(1)? == 0 {
            zeros += 1;
            if zeros >= 32 {
                return Err("native rule length overflow".into());
            }
        }
        let mut value = 1u32;
        for _ in 0..zeros {
            value = (value << 1) | self.read(1)?;
        }
        (value as usize)
            .checked_add(1)
            .ok_or("native rule length overflow".into())
    }
}

pub fn decode(bytes: &[u8]) -> Result<NativeCompressed, String> {
    let mut reader = Reader { bytes, at: 0 };
    if reader.take(8)? != MAGIC {
        return Err("not a BRPNAR01 archive".into());
    }
    let input_bytes = reader.u64()?;
    let field_count = reader.u64()?;
    let rule_count = reader.u32()? as usize;
    let block_count = reader.u32()? as usize;
    let rule_bits =
        usize::try_from(reader.u64()?).map_err(|_| "native dictionary size overflow")?;
    if rule_count > u32::MAX as usize - BASE_SYMBOLS
        || rule_count > bytes.len() / 2
        || block_count > bytes.len() / 9
    {
        return Err("native archive counts exceed file size".into());
    }
    let mut bits = Bits {
        bytes: reader.take(rule_bits.div_ceil(8))?,
        at: 0,
        length: rule_bits,
    };
    let mut rules = Vec::with_capacity(rule_count);
    let mut ends_field = vec![false; BASE_SYMBOLS];
    ends_field[crate::grammar::EOF as usize] = true;
    let mut lengths = vec![1usize; BASE_SYMBOLS];
    for i in 0..rule_count {
        let length = bits.arity()?;
        let width = symbol_width(BASE_SYMBOLS + i);
        if length > (bits.length - bits.at) / width as usize {
            return Err("native rule length exceeds dictionary".into());
        }
        let mut rule = Vec::with_capacity(length);
        let mut terminals = 0usize;
        for at in 0..length {
            let child = bits.read(width)?;
            if child as usize >= BASE_SYMBOLS + i {
                return Err("native rule refers to an undefined child".into());
            }
            if ends_field[child as usize] && at + 1 != length {
                return Err("native rule crosses a field boundary".into());
            }
            terminals = terminals
                .checked_add(lengths[child as usize])
                .ok_or("native expansion length overflow")?;
            rule.push(child);
        }
        ends_field.push(ends_field[*rule.last().unwrap() as usize]);
        lengths.push(terminals);
        rules.push(rule);
    }
    if bits.at != rule_bits {
        return Err("incorrect native dictionary bit length".into());
    }
    let mut blocks = Vec::with_capacity(block_count);
    for _ in 0..block_count {
        let width = reader.take(1)?[0];
        if !(9..=32).contains(&width) {
            return Err("invalid native block width".into());
        }
        let length = usize::try_from(reader.u64()?).map_err(|_| "native block length overflow")?;
        let bits = length
            .checked_mul(width as usize)
            .ok_or("native block size overflow")?;
        let block = PackedSymbols::from_bytes(width, length, reader.take(bits.div_ceil(8))?)?;
        if block
            .iter()
            .any(|s| s as usize >= BASE_SYMBOLS + rules.len())
        {
            return Err("native block refers to an undefined symbol".into());
        }
        blocks.push(block);
    }
    if reader.at != bytes.len() {
        return Err("trailing bytes after native archive".into());
    }
    Ok(NativeCompressed {
        rules,
        blocks,
        input_bytes,
        field_count,
    })
}
