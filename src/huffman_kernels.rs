//! Validated bulk decoder and optional matched hot-loop experiments.
use super::Codebook;

#[cfg(any(test, feature = "huffman-kernels"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kernel {
    Reference,
    OffsetChecked,
    BulkPrefix,
    BulkPrefixOverlap,
    BulkLinearOverlap,
    WindowPrefix,
}
#[cfg(any(test, feature = "huffman-kernels"))]
impl Kernel {
    pub const ALL: [Self; 6] = [
        Self::Reference,
        Self::OffsetChecked,
        Self::BulkPrefix,
        Self::BulkPrefixOverlap,
        Self::BulkLinearOverlap,
        Self::WindowPrefix,
    ];
    pub fn name(self) -> &'static str {
        match self {
            Self::Reference => "reference",
            Self::OffsetChecked => "offset-checked",
            Self::BulkPrefix => "bulk-prefix",
            Self::BulkPrefixOverlap => "bulk-prefix-overlap",
            Self::BulkLinearOverlap => "bulk-linear-overlap",
            Self::WindowPrefix => "window-prefix",
        }
    }
}
#[derive(Default)]
struct Bits {
    word: u64,
    available: u8,
    at: usize,
}
impl Bits {
    #[inline(always)]
    fn refill<const MASK: bool>(&mut self, payload: &[u8], max: u8) {
        if self.available >= max {
            return;
        }
        if payload.len() - self.at >= 8 {
            self.word |= u64::from_be_bytes(payload[self.at..self.at + 8].try_into().unwrap())
                >> self.available;
            let take = (64 - self.available) / 8;
            self.available += take * 8;
            self.at += take as usize;
            if MASK {
                self.word &= u64::MAX << (64 - self.available);
            }
            // With MASK=false, low speculative bits are the prefix of the
            // next unread byte from this immutable payload. Refilling at `at`
            // ORs the same bits in the same positions, so clearing is redundant.
        } else {
            while self.available < max && self.at < payload.len() {
                self.word |= u64::from(payload[self.at]) << (56 - self.available);
                self.available += 8;
                self.at += 1;
            }
        }
    }
    fn consumed(&self) -> Result<u64, String> {
        (self.at as u64)
            .checked_mul(8)
            .and_then(|n| n.checked_sub(u64::from(self.available)))
            .ok_or_else(|| "Huffman bit position overflow".into())
    }
}
impl Codebook {
    /// Decode one byte-bounded field, stopping at an EOF-ending expansion.
    /// Returns (consumed bits, emitted symbols); only zero byte padding may follow.
    pub(crate) fn decode_field_ranks(
        &self,
        payload: &[u8],
        mut emit: impl FnMut(u32) -> Result<bool, String>,
    ) -> Result<(u64, u64), String> {
        let bits = (payload.len() as u64)
            .checked_mul(8)
            .ok_or("payload size overflow")?;
        let mut state = Bits::default();
        let mut consumed = 0u64;
        let mut symbols = 0u64;
        let max = *self.lengths.last().unwrap();
        while consumed < bits {
            state.refill::<false>(payload, max);
            let len = self.length_fast::<true>(state.word);
            if u64::from(len) > bits - consumed {
                return Err("truncated Huffman field code".into());
            }
            let code = state.word >> (64 - len);
            if code.wrapping_sub(self.first[len as usize]) >= u64::from(self.counts[len as usize]) {
                return Err("invalid Huffman field code".into());
            }
            let ended = emit(code.wrapping_sub(self.bias[len as usize]) as u32)?;
            state.word <<= len;
            state.available -= len;
            consumed += u64::from(len);
            symbols += 1;
            if ended {
                if bits - consumed > 7 {
                    return Err("bytes after field EOF".into());
                }
                self.validate_payload(payload, consumed, symbols)?;
                return Ok((consumed, symbols));
            }
        }
        Err("Huffman field lacks EOF".into())
    }
    #[cfg(any(test, feature = "huffman-kernels"))]
    pub fn decode_ranks_variant(
        &self,
        kernel: Kernel,
        payload: &[u8],
        bits: u64,
        count: u64,
        emit: impl FnMut(u32) -> Result<(), String>,
    ) -> Result<(), String> {
        match kernel {
            Kernel::Reference => self.decode_ranks_reference(payload, bits, count, emit),
            Kernel::OffsetChecked => self.decode_offset_checked(payload, bits, count, emit),
            Kernel::BulkPrefix => self.decode_bulk::<true, true>(payload, bits, count, emit),
            Kernel::BulkPrefixOverlap => {
                self.decode_bulk::<true, false>(payload, bits, count, emit)
            }
            Kernel::BulkLinearOverlap => {
                self.decode_bulk::<false, false>(payload, bits, count, emit)
            }
            Kernel::WindowPrefix => self.decode_window(payload, bits, count, emit),
        }
    }
    fn validate_payload(&self, payload: &[u8], bits: u64, count: u64) -> Result<(), String> {
        if bits.div_ceil(8) != payload.len() as u64 || count > bits {
            return Err("invalid Huffman payload size".into());
        }
        if bits % 8 != 0
            && payload
                .last()
                .is_some_and(|b| b & ((1 << (8 - bits % 8)) - 1) != 0)
        {
            return Err("nonzero Huffman padding".into());
        }
        Ok(())
    }
    #[inline(always)]
    fn length_linear(&self, word: u64) -> u8 {
        for &(threshold, len) in &self.boundaries {
            if word >= threshold {
                return len;
            }
        }
        // Valid canonical tables always include the zero lower boundary.
        unreachable!("validated Huffman table lacks a zero boundary")
    }
    #[inline(always)]
    fn length_fast<const PREFIX: bool>(&self, word: u64) -> u8 {
        if PREFIX {
            let len = self.prefix_lengths[(word >> 52) as usize];
            if len != 0 {
                return len;
            }
        }
        self.length_linear(word)
    }
    fn checked_tail<const PREFIX: bool, const MASK: bool>(
        &self,
        payload: &[u8],
        bits: u64,
        count: u64,
        state: &mut Bits,
        mut consumed: u64,
        mut emit: impl FnMut(u32) -> Result<(), String>,
    ) -> Result<(), String> {
        let max = *self.lengths.last().unwrap();
        for _ in 0..count {
            state.refill::<MASK>(payload, max);
            let len = self.length_fast::<PREFIX>(state.word);
            if consumed + u64::from(len) > bits {
                return Err("truncated Huffman code".into());
            }
            let code = state.word >> (64 - len);
            if code - self.first[len as usize] >= u64::from(self.counts[len as usize]) {
                return Err("invalid Huffman code".into());
            }
            emit(code.wrapping_sub(self.bias[len as usize]) as u32)?;
            state.word <<= len;
            state.available -= len;
            consumed += u64::from(len);
        }
        if consumed != bits {
            return Err("unused Huffman payload bits".into());
        }
        Ok(())
    }
    #[cfg(any(test, feature = "huffman-kernels"))]
    fn decode_offset_checked(
        &self,
        payload: &[u8],
        bits: u64,
        count: u64,
        emit: impl FnMut(u32) -> Result<(), String>,
    ) -> Result<(), String> {
        self.validate_payload(payload, bits, count)?;
        self.checked_tail::<true, true>(payload, bits, count, &mut Bits::default(), 0, emit)
    }
    pub(super) fn decode_bulk<const PREFIX: bool, const MASK: bool>(
        &self,
        payload: &[u8],
        bits: u64,
        count: u64,
        mut emit: impl FnMut(u32) -> Result<(), String>,
    ) -> Result<(), String> {
        self.validate_payload(payload, bits, count)?;
        if self.lengths.len() == 1 {
            return self.decode_ranks_reference(payload, bits, count, emit);
        }
        let max = *self.lengths.last().unwrap();
        let (mut state, mut consumed, mut remaining) = (Bits::default(), 0u64, count);
        while remaining != 0 {
            // Complete canonical trees accept every full-width prefix. This
            // batch cannot reach padding even if every code has maximum length.
            // Only the final <= max/min symbols need per-code length checks.
            let batch = ((bits - consumed) / u64::from(max)).min(remaining);
            if batch == 0 {
                break;
            }
            for _ in 0..batch {
                state.refill::<MASK>(payload, max);
                let len = self.length_fast::<PREFIX>(state.word);
                let rank = (state.word >> (64 - len)).wrapping_sub(self.bias[len as usize]) as u32;
                emit(rank)?;
                state.word <<= len;
                state.available -= len;
            }
            remaining -= batch;
            consumed = state.consumed()?;
        }
        self.checked_tail::<PREFIX, MASK>(payload, bits, remaining, &mut state, consumed, emit)
    }
    #[cfg(any(test, feature = "huffman-kernels"))]
    fn decode_window(
        &self,
        payload: &[u8],
        bits: u64,
        count: u64,
        mut emit: impl FnMut(u32) -> Result<(), String>,
    ) -> Result<(), String> {
        self.validate_payload(payload, bits, count)?;
        if self.lengths.len() == 1 {
            return self.decode_ranks_reference(payload, bits, count, emit);
        }
        let max = u64::from(*self.lengths.last().unwrap());
        let (mut position, mut remaining) = (0u64, count);
        while remaining != 0 && bits - position >= max {
            let at = (position / 8) as usize;
            if payload.len() - at < 8 {
                break;
            }
            let word =
                u64::from_be_bytes(payload[at..at + 8].try_into().unwrap()) << (position % 8);
            let len = self.length_fast::<true>(word);
            emit((word >> (64 - len)).wrapping_sub(self.bias[len as usize]) as u32)?;
            position += u64::from(len);
            remaining -= 1;
        }
        let mut state = Bits {
            at: (position / 8) as usize,
            ..Bits::default()
        };
        let skip = (position % 8) as u8;
        if skip != 0 {
            state.word = u64::from(payload[state.at]) << (56 + skip);
            state.available = 8 - skip;
            state.at += 1;
        }
        self.checked_tail::<true, true>(payload, bits, remaining, &mut state, position, emit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn compare(book: &Codebook, ids: &[u32]) {
        let (bytes, bits) = book.encode(ids).unwrap();
        let mut expected = Vec::new();
        book.decode_ranks_reference(&bytes, bits, ids.len() as u64, |r| {
            expected.push(r);
            Ok(())
        })
        .unwrap();
        for kernel in Kernel::ALL {
            let mut actual = Vec::new();
            book.decode_ranks_variant(kernel, &bytes, bits, ids.len() as u64, |r| {
                actual.push(r);
                Ok(())
            })
            .unwrap();
            assert_eq!(actual, expected, "{}", kernel.name());
            assert!(book
                .decode_ranks_variant(kernel, &bytes, bits, ids.len() as u64 + 1, |_| Ok(()))
                .is_err());
            if !ids.is_empty() {
                assert!(book
                    .decode_ranks_variant(kernel, &bytes, bits, ids.len() as u64 - 1, |_| Ok(()))
                    .is_err());
            }
        }
    }
    #[test]
    fn all_kernels_match_reference_on_empty_short_random_and_deep_codes() {
        let mut random = 937u64;
        for alphabet in [1, 2, 3, 257, 32768] {
            let counts: Vec<_> = (0..alphabet).map(|n| (n as u64 % 97 + 1).pow(3)).collect();
            let book = Codebook::from_counts(&counts).unwrap();
            for n in [0, 1, 2, 7, 31, 65, 10003] {
                let ids: Vec<_> = (0..n)
                    .map(|_| {
                        random = random.wrapping_mul(6364136223846793005).wrapping_add(1);
                        (random % alphabet as u64) as u32
                    })
                    .collect();
                compare(&book, &ids);
            }
        }
        let (mut a, mut b) = (1u64, 1u64);
        let mut counts = vec![a, b];
        for _ in 0..80 {
            let c = a + b;
            counts.push(c);
            a = b;
            b = c;
        }
        let book = Codebook::from_counts(&counts).unwrap();
        let ids: Vec<_> = (0..counts.len() as u32).cycle().take(1003).collect();
        compare(&book, &ids);
    }
    #[test]
    fn every_truncated_bit_and_changed_padding_matches_reference() {
        let book = Codebook::from_counts(&[1, 2, 4, 8, 16, 32, 64, 128, 256]).unwrap();
        let ids: Vec<_> = (0..97).map(|n| n % 9).collect();
        let (bytes, bits) = book.encode(&ids).unwrap();
        for cut in 0..=bits {
            let mut data = bytes[..cut.div_ceil(8) as usize].to_vec();
            if cut % 8 != 0 {
                *data.last_mut().unwrap() &= 255u8 << (8 - cut % 8);
            }
            let mut reference = Vec::new();
            let result = book.decode_ranks_reference(&data, cut, ids.len() as u64, |r| {
                reference.push(r);
                Ok(())
            });
            for kernel in Kernel::ALL {
                let mut actual = Vec::new();
                let status = book.decode_ranks_variant(kernel, &data, cut, ids.len() as u64, |r| {
                    actual.push(r);
                    Ok(())
                });
                assert_eq!(
                    status.is_ok(),
                    result.is_ok(),
                    "{} at bit {cut}",
                    kernel.name()
                );
                assert_eq!(actual, reference, "{} at bit {cut}", kernel.name());
                if cut % 8 != 0 {
                    let mut bad = data.clone();
                    *bad.last_mut().unwrap() |= 1;
                    assert!(book
                        .decode_ranks_variant(kernel, &bad, cut, ids.len() as u64, |_| Ok(()))
                        .is_err());
                }
            }
        }
        let singleton = Codebook::from_counts(&[0, 10]).unwrap();
        for kernel in Kernel::ALL {
            assert!(singleton
                .decode_ranks_variant(kernel, &[128], 1, 1, |_| Ok(()))
                .is_err());
        }
    }
}
