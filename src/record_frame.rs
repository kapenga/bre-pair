//! Research independent-record framing; no change to production archive blocks.
use crate::{
    huffman::{get_var, put_var},
    tokenizer::fingerprint,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    Fixed20,
    VarintChecked,
    VarintPlain,
    EofChecked,
    EofPlain,
}
impl Format {
    pub fn file_header(self) -> Vec<u8> {
        if self == Self::Fixed20 {
            return Vec::new();
        }
        let mut bytes = b"BRREC001".to_vec();
        let mode: u64 = match self {
            Self::Fixed20 => unreachable!(),
            Self::VarintChecked => 1,
            Self::VarintPlain => 2,
            Self::EofChecked => 3,
            Self::EofPlain => 4,
        };
        bytes.extend_from_slice(&mode.to_le_bytes());
        bytes
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Fixed20 => "fixed20",
            Self::VarintChecked => "varint-checked",
            Self::VarintPlain => "varint-plain",
            Self::EofChecked => "eof-checked",
            Self::EofPlain => "eof-plain",
        }
    }
    pub fn checked(self) -> bool {
        matches!(self, Self::Fixed20 | Self::VarintChecked | Self::EofChecked)
    }
    pub fn has_counters(self) -> bool {
        !matches!(self, Self::EofChecked | Self::EofPlain)
    }
    pub fn encode(
        self,
        bytes: u64,
        bits: u64,
        symbols: u64,
        payload: &[u8],
    ) -> Result<Vec<u8>, String> {
        if bits.div_ceil(8) != payload.len() as u64 || symbols > bits || bytes > 1 << 20 {
            return Err("invalid record totals".into());
        }
        let mut out = Vec::with_capacity(20 + payload.len());
        for n in [bytes, bits, symbols]
            .into_iter()
            .take(if self.has_counters() { 3 } else { 0 })
        {
            if self == Self::Fixed20 {
                out.extend_from_slice(
                    &u32::try_from(n)
                        .map_err(|_| "record counter exceeds u32")?
                        .to_le_bytes(),
                );
            } else {
                put_var(&mut out, n);
            }
        }
        if self.checked() {
            out.extend_from_slice(&fingerprint(payload).to_le_bytes());
        }
        out.extend_from_slice(payload);
        Ok(out)
    }
    pub fn parse(self, mut frame: &[u8]) -> Result<Record<'_>, String> {
        let (bytes, bits, symbols) = if self == Self::Fixed20 {
            if frame.len() < 20 {
                return Err("truncated benchmark row".into());
            }
            let number = |at| u32::from_le_bytes(frame[at..at + 4].try_into().unwrap()) as u64;
            let totals = (number(0), number(4), number(8));
            frame = &frame[12..];
            totals
        } else if self.has_counters() {
            (
                get_var(&mut frame)?,
                get_var(&mut frame)?,
                get_var(&mut frame)?,
            )
        } else {
            (0, 0, 0)
        };
        if self.checked() {
            let checksum = frame.get(..8).ok_or("truncated record checksum")?;
            let expected = u64::from_le_bytes(checksum.try_into().unwrap());
            frame = &frame[8..];
            if fingerprint(frame) != expected {
                return Err("record checksum differs".into());
            }
        }
        if self.has_counters()
            && (bits.div_ceil(8) != frame.len() as u64 || symbols > bits || bytes > 1 << 20)
        {
            return Err("invalid record framing".into());
        }
        Ok(Record {
            has_counters: self.has_counters(),
            bytes,
            bits,
            symbols,
            payload: frame,
        })
    }
}
pub struct Record<'a> {
    /// False for EOF framing: totals below are absent and must not be used.
    pub has_counters: bool,
    pub bytes: u64,
    pub bits: u64,
    pub symbols: u64,
    pub payload: &'a [u8],
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn eof_framing_stores_only_payload_and_optional_checksum() {
        let payload = [0xa0];
        for format in [Format::EofChecked, Format::EofPlain] {
            let encoded = format.encode(129, 3, 2, &payload).unwrap();
            let record = format.parse(&encoded).unwrap();
            assert!(!record.has_counters);
            assert_eq!(record.payload, payload);
            assert_eq!(
                encoded.len(),
                payload.len() + if format.checked() { 8 } else { 0 }
            );
            assert_eq!(format.file_header().len(), 16);
            if format.checked() {
                let mut bad = encoded;
                *bad.last_mut().unwrap() ^= 1;
                assert!(format.parse(&bad).is_err());
            } else {
                assert_eq!(encoded, payload);
            }
        }
    }
    #[test]
    fn compact_counters_preserve_payload_and_checked_variant_integrity() {
        let payload = [0xa0];
        for format in [Format::Fixed20, Format::VarintChecked, Format::VarintPlain] {
            assert_eq!(
                format.file_header().len(),
                if format == Format::Fixed20 { 0 } else { 16 }
            );
            let encoded = format.encode(129, 3, 2, &payload).unwrap();
            let r = format.parse(&encoded).unwrap();
            assert_eq!(
                (r.bytes, r.bits, r.symbols, r.payload),
                (129, 3, 2, payload.as_slice())
            );
            assert_eq!(
                encoded.len(),
                match format {
                    Format::Fixed20 => 21,
                    Format::VarintChecked => 13,
                    Format::VarintPlain => 5,
                    Format::EofChecked | Format::EofPlain => unreachable!(),
                }
            );
            for end in 0..encoded.len() {
                assert!(format.parse(&encoded[..end]).is_err());
            }
            let mut extra = encoded.clone();
            extra.push(0);
            assert!(format.parse(&extra).is_err());
            if format.checked() {
                let mut bad = encoded.clone();
                *bad.last_mut().unwrap() ^= 1;
                assert!(format.parse(&bad).is_err());
            }
        }
        assert!(Format::Fixed20.encode(1 << 21, 1, 1, &[0]).is_err());
        assert!(Format::VarintPlain.encode(1, 1, 2, &[0]).is_err());
        assert!(Format::VarintPlain.parse(&[128; 30]).is_err());
    }
}
