//! Experimental nonzero byte codes plus a literal zero field delimiter.
//! All canonical ranks, including EOF-bearing ranks, are preserved. The extra
//! delimiter makes boundaries discoverable without interpreting any rank.
use super::{ByteCodes, Expansions};
use crate::huffman::Codebook;
use std::ops::Range;

#[derive(Clone, Copy, Default)]
struct Lead {
    base: u64,
    bytes: usize,
}

pub struct ScannableCodes {
    lead: [Lead; 256],
    boundaries: [u64; 5],
    pub slots: [u16; 5],
    pub symbols: usize,
}

impl ScannableCodes {
    pub fn with_slots(symbols: usize, slots: [u16; 5]) -> Result<Self, String> {
        if symbols == 0
            || symbols > u32::MAX as usize
            || slots.iter().map(|&x| u32::from(x)).sum::<u32>() != 255
        {
            return Err("invalid scannable alphabet or prefix allocation".into());
        }
        let mut lead = [Lead::default(); 256];
        let mut boundaries = [0; 5];
        let (mut prefix, mut base, mut capacity) = (1, 0u64, 1u64);
        for level in 0..5 {
            for _ in 0..slots[level] {
                if base < symbols as u64 {
                    lead[prefix] = Lead {
                        base,
                        bytes: level + 1,
                    };
                }
                prefix += 1;
                base += capacity;
            }
            boundaries[level] = base;
            capacity *= 255;
        }
        if base < symbols as u64 {
            return Err("insufficient scannable code capacity".into());
        }
        Ok(Self {
            lead,
            boundaries,
            slots,
            symbols,
        })
    }

    /// Fit byte widths using only the frozen Huffman model, never test counts.
    pub fn from_book(book: &Codebook) -> Result<Self, String> {
        let mut cumulative = vec![0u128];
        for (_, length) in book.lengths() {
            cumulative.push(cumulative.last().unwrap() + (1u128 << (book.max_bits - length)));
        }
        let n = book.symbols.len();
        let fifth = u16::from(n as u64 > 255u64.pow(4));
        let available = 255 - fifth;
        let mut best = (u128::MAX, [0; 5]);
        for a in 0..=available {
            for b in 0..=available - a {
                for c in 0..=available - a - b {
                    let d = available - a - b - c;
                    let x = u64::from(a);
                    let y = x + u64::from(b) * 255;
                    let z = y + u64::from(c) * 255 * 255;
                    let w = z + u64::from(d) * 255u64.pow(3);
                    if w + u64::from(fifth) * 255u64.pow(4) < n as u64 {
                        continue;
                    }
                    let mut cost = cumulative[n] * 4
                        - cumulative[(x as usize).min(n)]
                        - cumulative[(y as usize).min(n)]
                        - cumulative[(z as usize).min(n)];
                    if fifth != 0 {
                        cost += cumulative[n] - cumulative[(w as usize).min(n)];
                    }
                    if cost < best.0 {
                        best = (cost, [a, b, c, d, fifth]);
                    }
                }
            }
        }
        Self::with_slots(n, best.1)
    }

    pub fn encode_rank(&self, rank: u32, output: &mut Vec<u8>) -> Result<(), String> {
        if rank as usize >= self.symbols {
            return Err("undefined scannable rank".into());
        }
        let mut prefixes = 1u64;
        let mut before = 0;
        let mut capacity = 1;
        for level in 0..5 {
            if u64::from(rank) < self.boundaries[level] {
                let mut rest = u64::from(rank) - before;
                output.push((prefixes + rest / capacity) as u8);
                rest %= capacity;
                for _ in 0..level {
                    output.push((rest % 255 + 1) as u8);
                    rest /= 255;
                }
                return Ok(());
            }
            prefixes += u64::from(self.slots[level]);
            before = self.boundaries[level];
            capacity *= 255;
        }
        Err("rank outside scannable allocation".into())
    }

    pub fn decode_rank(&self, input: &[u8]) -> Result<(u32, usize), String> {
        let entry = self.lead[usize::from(*input.first().ok_or("missing scannable code")?)];
        if entry.bytes == 0 || input.len() < entry.bytes {
            return Err("invalid or truncated scannable code".into());
        }
        let mut rank = entry.base;
        let mut factor = 1;
        for &digit in &input[1..entry.bytes] {
            if digit == 0 {
                return Err("EOF inside scannable code".into());
            }
            rank += u64::from(digit - 1) * factor;
            factor *= 255;
        }
        if rank >= self.symbols as u64 {
            return Err("undefined scannable rank".into());
        }
        Ok((rank as u32, entry.bytes))
    }

    pub fn transcode_field(
        &self,
        book: &Codebook,
        input: &[u8],
        output: &mut Vec<u8>,
        mut ends: impl FnMut(u32) -> bool,
    ) -> Result<u64, String> {
        if book.symbols.len() != self.symbols {
            return Err("scannable alphabet mismatch".into());
        }
        let start = output.len();
        let result = book.decode_field_ranks(input, |rank| {
            self.encode_rank(rank, output)?;
            Ok(ends(rank))
        });
        match result {
            Ok((_, count)) => {
                output.push(0);
                Ok(count)
            }
            Err(error) => {
                output.truncate(start);
                Err(error)
            }
        }
    }

    pub fn metadata(&self) -> [u8; 32] {
        let mut bytes = [0; 32];
        bytes[..8].copy_from_slice(b"BRSCAN01");
        bytes[8..16].copy_from_slice(&(self.symbols as u64).to_le_bytes());
        for (i, value) in self.slots.iter().enumerate() {
            bytes[16 + 2 * i..18 + 2 * i].copy_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    pub fn from_metadata(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() != 32 || &bytes[..8] != b"BRSCAN01" || bytes[26..].iter().any(|&b| b != 0) {
            return Err("invalid scannable metadata".into());
        }
        let symbols = usize::try_from(u64::from_le_bytes(bytes[8..16].try_into().unwrap()))
            .map_err(|_| "scannable alphabet exceeds address space")?;
        let mut slots = [0; 5];
        for (i, value) in slots.iter_mut().enumerate() {
            *value = u16::from_le_bytes(bytes[16 + 2 * i..18 + 2 * i].try_into().unwrap());
        }
        Self::with_slots(symbols, slots)
    }
}

impl Expansions<'_> {
    /// Fair sparse-index control: parse ordinary byte ranks and EOF flags but
    /// never expand their contents. Input retains the ordinary 3-byte guard.
    pub fn skip_byte_fields(
        &self,
        codes: &ByteCodes,
        input: &[u8],
        range: Range<usize>,
        fields: u64,
    ) -> Result<usize, String> {
        if range.start > range.end
            || range.end.checked_add(3).is_none_or(|end| end > input.len())
            || codes.symbols != self.references.len()
        {
            return Err("invalid ordinary skip bounds".into());
        }
        let mut at = range.start;
        let mut remaining = fields;
        while remaining != 0 && at < range.end {
            // Range plus the ordinary three-byte guard proves this load safe.
            let word = unsafe {
                u32::from_le(std::ptr::read_unaligned(
                    input.as_ptr().add(at).cast::<u32>(),
                ))
            };
            let entry = codes.lead[word as u8 as usize];
            if entry.bytes == 0 || entry.bytes as usize > range.end - at {
                return Err("invalid ordinary skip code".into());
            }
            let rank = u64::from(entry.base) + u64::from((word >> 8) & entry.mask);
            let reference = self
                .references
                .get(rank as usize)
                .ok_or("undefined ordinary skip rank")?;
            at += entry.bytes as usize;
            remaining -= u64::from(reference.1 >> 31);
        }
        if remaining != 0 {
            return Err("insufficient fields for ordinary skip".into());
        }
        Ok(at - range.start)
    }
    /// Decode one field from a concatenated stream; return bytes and input used.
    /// Only the output needs the existing 16-byte private copy guard.
    pub fn scannable_next(
        &self,
        codes: &ScannableCodes,
        input: &[u8],
        output: &mut [u8],
        maximum: usize,
    ) -> Result<(usize, usize), String> {
        if output.len() < 16
            || maximum > output.len() - 16
            || codes.symbols != self.references.len()
        {
            return Err("invalid scannable output bounds or alphabet".into());
        }
        let (mut at, mut used, mut ended) = (0, 0, false);
        loop {
            let byte = *input.get(at).ok_or("scannable field lacks EOF delimiter")?;
            if byte == 0 {
                return if ended {
                    Ok((used, at + 1))
                } else {
                    Err("delimiter before EOF rank".into())
                };
            }
            if ended {
                return Err("content after EOF rank".into());
            }
            let (rank, size) = codes.decode_rank(&input[at..])?;
            ended = self.emit(rank, output, &mut used, maximum)?;
            at += size;
        }
    }
}

pub fn for_each_eof_scalar(input: &[u8], mut visit: impl FnMut(usize)) {
    for (at, &byte) in input.iter().enumerate() {
        if byte == 0 {
            visit(at);
        }
    }
}

/// Runtime-dispatched AVX2 boundary scan with a portable fallback.
pub fn for_each_eof(input: &[u8], visit: impl FnMut(usize)) {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        // CPU support was checked; the SIMD loop never reads past input.
        unsafe {
            for_each_eof_avx2(input, visit);
        }
        return;
    }
    for_each_eof_scalar(input, visit);
}

/// Find an EOF by its zero-based ordinal without decoding intervening symbols.
pub fn nth_eof(input: &[u8], ordinal: u64) -> Option<usize> {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        unsafe {
            return nth_eof_avx2(input, ordinal);
        }
    }
    let mut left = ordinal;
    input.iter().position(|&byte| {
        if byte != 0 {
            false
        } else if left == 0 {
            true
        } else {
            left -= 1;
            false
        }
    })
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn nth_eof_avx2(input: &[u8], mut ordinal: u64) -> Option<usize> {
    use std::arch::x86_64::*;
    let zero = _mm256_setzero_si256();
    let mut at = 0;
    while input.len() - at >= 32 {
        let word = _mm256_loadu_si256(input.as_ptr().add(at).cast());
        let mut mask = _mm256_movemask_epi8(_mm256_cmpeq_epi8(word, zero)) as u32;
        let count = u64::from(mask.count_ones());
        if ordinal < count {
            for _ in 0..ordinal {
                mask &= mask - 1;
            }
            return Some(at + mask.trailing_zeros() as usize);
        }
        ordinal -= count;
        at += 32;
    }
    input[at..]
        .iter()
        .position(|&byte| {
            if byte != 0 {
                false
            } else if ordinal == 0 {
                true
            } else {
                ordinal -= 1;
                false
            }
        })
        .map(|offset| at + offset)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn for_each_eof_avx2(input: &[u8], mut visit: impl FnMut(usize)) {
    use std::arch::x86_64::*;
    let zero = _mm256_setzero_si256();
    let mut at = 0;
    while input.len() - at >= 32 {
        let word = _mm256_loadu_si256(input.as_ptr().add(at).cast());
        let mut mask = _mm256_movemask_epi8(_mm256_cmpeq_epi8(word, zero)) as u32;
        while mask != 0 {
            visit(at + mask.trailing_zeros() as usize);
            mask &= mask - 1;
        }
        at += 32;
    }
    for_each_eof_scalar(&input[at..], |offset| visit(at + offset));
}

/// Borrowed payload plus sparse byte offsets. Building scans markers, not ranks;
/// syntax and expansion limits are validated by the decoder separately.
pub struct SparseFields<'a> {
    payload: &'a [u8],
    offsets: Vec<u64>,
    pub stride: u64,
    pub fields: u64,
}
impl<'a> SparseFields<'a> {
    pub fn build(payload: &'a [u8], stride: u64) -> Result<Self, String> {
        if stride == 0 || (!payload.is_empty() && payload.last() != Some(&0)) {
            return Err("invalid sparse stride or unterminated payload".into());
        }
        let mut fields = 0;
        let mut offsets = if payload.is_empty() { vec![] } else { vec![0] };
        for_each_eof(payload, |at| {
            fields += 1;
            if fields % stride == 0 && at + 1 < payload.len() {
                offsets.push((at + 1) as u64);
            }
        });
        Ok(Self {
            payload,
            offsets,
            stride,
            fields,
        })
    }
    pub fn locate(&self, field: u64) -> Result<Range<usize>, String> {
        if field >= self.fields {
            return Err("field ordinal outside sparse index".into());
        }
        let checkpoint = (field / self.stride) as usize;
        let mut start = self.offsets[checkpoint] as usize;
        // Only the bounded local suffix is inspected, never the whole column.
        let skip = field % self.stride;
        if skip != 0 {
            start += nth_eof(&self.payload[start..], skip - 1).ok_or("missing EOF")? + 1;
        }
        let end = start + nth_eof(&self.payload[start..], 0).ok_or("missing EOF")? + 1;
        Ok(start..end)
    }
    pub fn metadata(&self) -> Vec<u8> {
        let mut bytes = b"BRSPAR01".to_vec();
        for value in [self.stride, self.fields, self.payload.len() as u64] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for value in &self.offsets {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn all_widths_nonzero_and_full_u32_capacity() {
        for slots in [[1, 1, 1, 251, 1], [0, 0, 0, 254, 1], [255, 0, 0, 0, 0]] {
            let n = if slots[0] == 255 {
                255
            } else {
                u32::MAX as usize
            };
            let codes = ScannableCodes::with_slots(n, slots).unwrap();
            let mut ranks = vec![
                0,
                1,
                254,
                255,
                256,
                65024,
                65025,
                16581374,
                16581375,
                u32::MAX - 1,
            ];
            ranks.retain(|&r| (r as usize) < n);
            let mut state = 1234567u32;
            for _ in 0..10000 {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                ranks.push(state % (n as u32));
            }
            let restored = ScannableCodes::from_metadata(&codes.metadata()).unwrap();
            for rank in ranks {
                let mut bytes = vec![];
                codes.encode_rank(rank, &mut bytes).unwrap();
                assert!(!bytes.contains(&0));
                assert_eq!(restored.decode_rank(&bytes).unwrap(), (rank, bytes.len()));
                for cut in 0..bytes.len() {
                    assert!(codes.decode_rank(&bytes[..cut]).is_err());
                }
                for at in 0..bytes.len() {
                    let mut bad = bytes.clone();
                    bad[at] = 0;
                    assert!(codes.decode_rank(&bad).is_err());
                }
            }
        }
    }
    #[test]
    fn boundary_scan_matches_every_alignment_and_dense_markers() {
        let mut bytes = (0..4096).map(|n| (n % 255 + 1) as u8).collect::<Vec<_>>();
        for at in [0, 1, 31, 32, 33, 63, 64, 127, 255, 4095] {
            bytes[at] = 0;
        }
        for start in 0..65 {
            for end in start..(start + 130).min(bytes.len()) {
                let mut scalar = vec![];
                let mut vector = vec![];
                for_each_eof_scalar(&bytes[start..end], |at| scalar.push(at));
                for_each_eof(&bytes[start..end], |at| vector.push(at));
                assert_eq!(scalar, vector);
                for (n, &at) in scalar.iter().enumerate() {
                    assert_eq!(nth_eof(&bytes[start..end], n as u64), Some(at));
                }
                assert_eq!(nth_eof(&bytes[start..end], scalar.len() as u64), None);
            }
        }
        for value in [0, 1, 255] {
            let data = vec![value; 257];
            let mut found = vec![];
            for_each_eof(&data, |at| found.push(at));
            assert_eq!(found.len(), if value == 0 { 257 } else { 0 });
        }
    }
    #[test]
    fn sparse_lookup_exact_ranges_and_bounds() {
        let mut payload = vec![];
        let mut ranges = vec![];
        for n in 0..3001 {
            let start = payload.len();
            payload.extend(std::iter::repeat_n(1, n % 17));
            payload.push(0);
            ranges.push(start..payload.len());
        }
        for stride in [1, 2, 17, 1024, 10000] {
            let sparse = SparseFields::build(&payload, stride).unwrap();
            for (n, range) in ranges.iter().enumerate() {
                assert_eq!(sparse.locate(n as u64).unwrap(), *range);
            }
            assert!(sparse.locate(ranges.len() as u64).is_err());
            assert_eq!(
                sparse.metadata().len(),
                32 + ranges.len().div_ceil(stride as usize) * 8
            );
        }
        assert!(SparseFields::build(&payload, 0).is_err());
        assert!(SparseFields::build(&[1], 1).is_err());
        assert_eq!(SparseFields::build(&[], 1024).unwrap().fields, 0);
    }
    #[test]
    fn field_bytes_merged_eof_empty_binary_and_malformed() {
        let mut arena = vec![0, 255, b'X'];
        arena.extend([0; 16]);
        let references = [(0, 2), (2, 0x80000001), (0, 0x80000000)];
        let expansions = Expansions::new(&arena, &references).unwrap();
        let codes = ScannableCodes::with_slots(3, [255, 0, 0, 0, 0]).unwrap();
        let book = Codebook::from_counts(&[50, 20, 10]).unwrap();
        let ordered = book
            .symbols
            .iter()
            .map(|&r| references[r as usize])
            .collect::<Vec<_>>();
        let ranked = Expansions::new(&arena, &ordered).unwrap();
        let fitted = ScannableCodes::from_book(&book).unwrap();
        let mut stream = vec![];
        for ids in [vec![0, 1], vec![2]] {
            let (encoded, _) = book.encode(&ids).unwrap();
            fitted
                .transcode_field(&book, &encoded, &mut stream, |r| ranked.ends_field(r))
                .unwrap();
        }
        let index = SparseFields::build(&stream, 1024).unwrap();
        assert_eq!(index.fields, 2);
        let mut output = [0xaa; 19];
        let first = index.locate(0).unwrap();
        assert_eq!(
            ranked
                .scannable_next(&fitted, &stream[first.clone()], &mut output, 3)
                .unwrap(),
            (3, first.len())
        );
        assert_eq!(&output[..3], &[0, 255, b'X']);
        assert_eq!(
            ranked
                .scannable_next(&fitted, &stream[index.locate(1).unwrap()], &mut output, 0)
                .unwrap()
                .0,
            0
        );
        assert!(ranked
            .scannable_next(&fitted, &stream, &mut output, 2)
            .is_err());
        for cut in 0..first.end {
            assert!(ranked
                .scannable_next(&fitted, &stream[..cut], &mut output, 3)
                .is_err());
        }
        for malformed in [vec![0], vec![1, 0], vec![2, 1, 0], vec![255, 0]] {
            assert!(expansions
                .scannable_next(&codes, &malformed, &mut output, 3)
                .is_err());
        }
        let mut metadata = fitted.metadata();
        metadata[31] = 1;
        assert!(ScannableCodes::from_metadata(&metadata).is_err());
        assert!(ScannableCodes::with_slots(256, [255, 0, 0, 0, 0]).is_err());
    }

    #[test]
    fn ordinary_skip_matches_field_boundaries_without_expansion() {
        let arena = vec![0; 32];
        let refs = [(0, 1), (0, 0x80000001), (0, 0x80000000)];
        let e = Expansions::new(&arena, &refs).unwrap();
        let codes = ByteCodes::with_slots(3, [0, 256, 0, 0]).unwrap();
        let mut data = vec![];
        for rank in [0, 0, 1, 2, 0, 1] {
            codes.encode_rank(rank, &mut data).unwrap();
        }
        let end = data.len();
        data.extend([0; 4]);
        assert_eq!(e.skip_byte_fields(&codes, &data, 0..end, 0).unwrap(), 0);
        assert_eq!(e.skip_byte_fields(&codes, &data, 0..end, 1).unwrap(), 6);
        assert_eq!(e.skip_byte_fields(&codes, &data, 0..end, 2).unwrap(), 8);
        assert_eq!(e.skip_byte_fields(&codes, &data, 0..end, 3).unwrap(), 12);
        assert!(e.skip_byte_fields(&codes, &data, 0..end, 4).is_err());
        assert!(e.skip_byte_fields(&codes, &data, 0..end - 1, 3).is_err());
        assert!(e.skip_byte_fields(&codes, &data[..end], 0..end, 1).is_err());
    }
}
