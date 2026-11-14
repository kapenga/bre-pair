//! Opt-in byte-aligned speed experiment. Production formats are unchanged.
//! Codes use the existing canonical rank order, avoiding an extra permutation.
use crate::huffman::Codebook;
use std::ops::Range;
pub mod scannable;
pub type Tables<'a> = (&'a [u8], &'a [(u32, u32)]);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DecodedBlock {
    pub bytes: usize,
    pub fields: u64,
    pub symbols: u64,
    pub ended: bool,
}

#[derive(Clone, Copy, Default)]
struct Lead {
    base: u32,
    mask: u32,
    bytes: u32,
}

pub struct ByteCodes {
    lead: [Lead; 256],
    pub slots: [u16; 4],
    pub symbols: usize,
    boundaries: [u64; 4],
}
impl ByteCodes {
    /// Choose byte brackets from the frozen Huffman model, never holdout counts.
    pub fn from_book(book: &Codebook) -> Result<Self, String> {
        let mut cumulative = vec![0u64];
        for (_, length) in book.lengths() {
            let weight = 1u64 << (book.max_bits - length);
            cumulative.push(cumulative.last().unwrap() + weight);
        }
        let n = book.symbols.len();
        let mut best = (u64::MAX, [0u16; 4]);
        for a in 0..=256u16 {
            for b in 0..=256 - a {
                for c in 0..=256 - a - b {
                    let d = 256 - a - b - c;
                    let x = u64::from(a);
                    let y = x + u64::from(b) * 256;
                    let z = y + u64::from(c) * 65536;
                    if z + u64::from(d) * 16777216 < n as u64 {
                        continue;
                    }
                    let saved = cumulative[(x as usize).min(n)]
                        + cumulative[(y as usize).min(n)]
                        + cumulative[(z as usize).min(n)];
                    let cost = cumulative[n] * 4 - saved;
                    if cost < best.0 {
                        best = (cost, [a, b, c, d]);
                    }
                }
            }
        }
        Self::with_slots(n, best.1)
    }
    pub fn with_slots(symbols: usize, slots: [u16; 4]) -> Result<Self, String> {
        if symbols == 0
            || symbols > u32::MAX as usize
            || slots.iter().map(|&x| u32::from(x)).sum::<u32>() != 256
        {
            return Err("invalid byte-code alphabet or prefix allocation".into());
        }
        let mut lead = [Lead::default(); 256];
        let mut prefix = 0;
        let mut base = 0u64;
        let mut boundaries = [0; 4];
        for level in 0..4 {
            let capacity = 1u64 << (level * 8);
            for _ in 0..slots[level] {
                if base < symbols as u64 {
                    lead[prefix] = Lead {
                        base: base as u32,
                        mask: (capacity - 1) as u32,
                        bytes: level as u32 + 1,
                    };
                }
                base += capacity;
                prefix += 1;
            }
            boundaries[level] = base;
        }
        if base < symbols as u64 {
            return Err("insufficient byte-code capacity".into());
        }
        Ok(Self {
            lead,
            slots,
            symbols,
            boundaries,
        })
    }
    pub fn encode_rank(&self, rank: u32, out: &mut Vec<u8>) -> Result<(), String> {
        if rank as usize >= self.symbols {
            return Err("undefined byte-code rank".into());
        }
        let rank = u64::from(rank);
        let mut before = 0;
        let mut prefixes = 0;
        for level in 0..4 {
            if rank < self.boundaries[level] {
                let rest = rank - before;
                out.push((prefixes + (rest >> (level * 8))) as u8);
                out.extend_from_slice(&(rest as u32).to_le_bytes()[..level]);
                return Ok(());
            }
            before = self.boundaries[level];
            prefixes += u64::from(self.slots[level]);
        }
        Err("rank outside byte-code allocation".into())
    }
    pub fn transcode_field(
        &self,
        book: &Codebook,
        input: &[u8],
        out: &mut Vec<u8>,
        mut ends: impl FnMut(u32) -> bool,
    ) -> Result<u64, String> {
        let (_, count) = book.decode_field_ranks(input, |rank| {
            self.encode_rank(rank, out)?;
            Ok(ends(rank))
        })?;
        Ok(count)
    }
    pub fn metadata(&self) -> [u8; 16] {
        let mut out = [0; 16];
        out[..8].copy_from_slice(b"BRBYTE01");
        for (i, value) in self.slots.iter().enumerate() {
            out[8 + i * 2..10 + i * 2].copy_from_slice(&value.to_le_bytes());
        }
        out
    }
    /// Checked standalone rank decode, also used for an oversized cached symbol.
    pub fn decode_rank(&self, input: &[u8]) -> Result<(u32, usize), String> {
        let entry = self.lead[usize::from(*input.first().ok_or("missing byte code")?)];
        let length = entry.bytes as usize;
        if length == 0 || length > input.len() {
            return Err("invalid or truncated byte code".into());
        }
        let mut bytes = [0; 4];
        bytes[..length].copy_from_slice(&input[..length]);
        let rank = entry
            .base
            .checked_add((u32::from_le_bytes(bytes) >> 8) & entry.mask)
            .filter(|&rank| (rank as usize) < self.symbols)
            .ok_or("undefined byte rank")?;
        Ok((rank, length))
    }
    pub fn from_metadata(symbols: usize, data: &[u8]) -> Result<Self, String> {
        if data.len() != 16 || &data[..8] != b"BRBYTE01" {
            return Err("invalid byte-code metadata".into());
        }
        let mut slots = [0; 4];
        for (i, v) in slots.iter_mut().enumerate() {
            *v = u16::from_le_bytes(data[8 + i * 2..10 + i * 2].try_into().unwrap());
        }
        Self::with_slots(symbols, slots)
    }
}

/// Borrow one shared immutable arena and rank-to-(offset,length/EOF) table.
pub struct Expansions<'a> {
    arena: &'a [u8],
    references: &'a [(u32, u32)],
}
impl<'a> Expansions<'a> {
    pub fn new(arena: &'a [u8], references: &'a [(u32, u32)]) -> Result<Self, String> {
        for &(offset, flags) in references {
            if (offset as usize)
                .checked_add((flags & 0x7fffffff) as usize)
                .and_then(|n| n.checked_add(16))
                .is_none_or(|end| end > arena.len())
            {
                return Err("expansion arena lacks required 16-byte guard".into());
            }
        }
        Ok(Self { arena, references })
    }
    pub fn ends_field(&self, rank: u32) -> bool {
        self.references[rank as usize].1 >> 31 != 0
    }
    /// Borrow a fully cached expansion; no temporary giant output allocation.
    pub fn rank_bytes(&self, rank: u32) -> Result<(&[u8], bool), String> {
        let &(offset, flags) = self
            .references
            .get(rank as usize)
            .ok_or("undefined byte rank")?;
        Ok((
            &self.arena[offset as usize..offset as usize + (flags & 0x7fff_ffff) as usize],
            flags >> 31 != 0,
        ))
    }
    #[inline(always)]
    fn emit(
        &self,
        rank: u32,
        out: &mut [u8],
        used: &mut usize,
        maximum: usize,
    ) -> Result<bool, String> {
        let &(offset, flags) = self
            .references
            .get(rank as usize)
            .ok_or("invalid expansion rank")?;
        let length = (flags & 0x7fffffff) as usize;
        if length > maximum - *used {
            return Err("decoded output exceeds limit".into());
        }
        // Constructor validates all source ranges plus a 16-byte guard. Entry
        // points validate the destination guard; this call checks logical size.
        // Safe Rust borrowing also prevents input/output arena aliasing.
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.arena.as_ptr().add(offset as usize),
                out.as_mut_ptr().add(*used),
                if length <= 16 { 16 } else { length },
            );
        }
        *used += length;
        Ok(flags >> 31 != 0)
    }
    pub fn huffman_field(
        &self,
        book: &Codebook,
        input: &[u8],
        out: &mut [u8],
        maximum: usize,
    ) -> Result<usize, String> {
        if maximum > out.len().saturating_sub(16) || out.len() < 16 {
            return Err("missing output guard".into());
        }
        let mut used = 0;
        book.decode_field_ranks(input, |rank| self.emit(rank, out, &mut used, maximum))?;
        Ok(used)
    }
    pub fn huffman_block(
        &self,
        book: &Codebook,
        input: &[u8],
        bits: u64,
        count: u64,
        out: &mut [u8],
        maximum: usize,
    ) -> Result<usize, String> {
        if maximum > out.len().saturating_sub(16) || out.len() < 16 {
            return Err("missing output guard".into());
        }
        let mut used = 0;
        book.decode_ranks(input, bits, count, |rank| {
            self.emit(rank, out, &mut used, maximum).map(|_| ())
        })?;
        Ok(used)
    }
    /// Input includes at least three readable guard bytes after the encoded
    /// range. They may belong to the next field, but never affect decoded ranks.
    /// Output has a private 16-byte guard. All malformed-data checks remain on.
    pub fn byte_field(
        &self,
        codes: &ByteCodes,
        input: &[u8],
        range: Range<usize>,
        out: &mut [u8],
        maximum: usize,
    ) -> Result<usize, String> {
        self.byte_range::<true, false>(codes, input, range, out, maximum)
            .map(|result| result.bytes)
    }
    pub fn byte_block(
        &self,
        codes: &ByteCodes,
        input: &[u8],
        range: Range<usize>,
        out: &mut [u8],
        maximum: usize,
    ) -> Result<usize, String> {
        self.byte_range::<false, false>(codes, input, range, out, maximum)
            .map(|result| result.bytes)
    }
    /// Full-stream validation counts EOF and ranks in the existing decode loop.
    pub fn byte_block_checked(
        &self,
        codes: &ByteCodes,
        input: &[u8],
        range: Range<usize>,
        out: &mut [u8],
        maximum: usize,
    ) -> Result<DecodedBlock, String> {
        self.byte_range::<false, true>(codes, input, range, out, maximum)
    }
    pub fn fixed_field(
        &self,
        input: &[u8],
        range: Range<usize>,
        width: usize,
        out: &mut [u8],
        maximum: usize,
    ) -> Result<usize, String> {
        if !matches!(width, 2..=4)
            || range.start > range.end
            || range.end.checked_add(3).is_none_or(|end| end > input.len())
            || (range.end - range.start) % width != 0
            || out.len() < 16
            || maximum > out.len() - 16
        {
            return Err("invalid fixed-code bounds".into());
        }
        let mut used = 0;
        let mut at = range.start;
        let mask = u32::MAX >> ((4 - width) * 8);
        while at < range.end {
            // Four readable input bytes were proved above; unused suffix bits
            // are masked before indexing the checked expansion table.
            let rank = unsafe {
                u32::from_le(std::ptr::read_unaligned(
                    input.as_ptr().add(at).cast::<u32>(),
                ))
            } & mask;
            at += width;
            if self.emit(rank, out, &mut used, maximum)? {
                if at != range.end {
                    return Err("fixed bytes after EOF".into());
                }
                return Ok(used);
            }
        }
        Err("fixed field lacks EOF".into())
    }
    #[inline]
    fn byte_range<const FIELD: bool, const COUNT: bool>(
        &self,
        codes: &ByteCodes,
        input: &[u8],
        range: Range<usize>,
        out: &mut [u8],
        maximum: usize,
    ) -> Result<DecodedBlock, String> {
        if range.start > range.end
            || range.end.checked_add(3).is_none_or(|end| end > input.len())
            || out.len() < 16
            || maximum > out.len() - 16
            || codes.symbols != self.references.len()
        {
            return Err("invalid encoded/output bounds".into());
        }
        let mut at = range.start;
        let mut used = 0;
        let mut result = DecodedBlock::default();
        while at < range.end {
            // A single wide load covers the lead and up to three suffix bytes.
            // Bounds above prove the speculative bytes are readable.
            let word = unsafe {
                u32::from_le(std::ptr::read_unaligned(
                    input.as_ptr().add(at).cast::<u32>(),
                ))
            };
            let entry = codes.lead[word as u8 as usize];
            if entry.bytes == 0 || entry.bytes as usize > range.end - at {
                return Err("invalid/truncated byte code".into());
            }
            let rank = entry
                .base
                .checked_add((word >> 8) & entry.mask)
                .ok_or("byte-code rank overflow")?;
            at += entry.bytes as usize;
            let ended = self.emit(rank, out, &mut used, maximum)?;
            if COUNT {
                result.symbols += 1;
                result.fields += u64::from(ended);
                result.ended = ended;
            }
            if FIELD && ended {
                if at != range.end {
                    return Err("bytes after field EOF".into());
                }
                result.bytes = used;
                return Ok(result);
            }
        }
        if FIELD {
            Err("byte field lacks EOF".into())
        } else {
            result.bytes = used;
            Ok(result)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn table() -> (Vec<u8>, Vec<(u32, u32)>) {
        let mut arena = vec![b'x'; 100 + 16];
        arena[0] = b'A';
        (arena, vec![(0, 1), (1, 16), (17, 17), (0, 0x80000000)])
    }
    #[test]
    fn bracket_boundaries_roundtrip_all_widths() {
        let codes = ByteCodes::with_slots(200000, [250, 2, 2, 2]).unwrap();
        for rank in [0, 249, 250, 761, 762, 131833, 131834, 199999] {
            let mut encoded = Vec::new();
            codes.encode_rank(rank, &mut encoded).unwrap();
            let n = encoded.len();
            encoded.resize(n + 4, 0);
            let word = u32::from_le_bytes(encoded[..4].try_into().unwrap());
            let e = codes.lead[encoded[0] as usize];
            assert_eq!(e.bytes as usize, n);
            assert_eq!(e.base + ((word >> 8) & e.mask), rank);
        }
    }
    #[test]
    fn exact_bytes_short_long_and_empty_eof() {
        let (arena, refs) = table();
        let expansions = Expansions::new(&arena, &refs).unwrap();
        let codes = ByteCodes::with_slots(4, [256, 0, 0, 0]).unwrap();
        let mut input = Vec::new();
        for rank in [0, 1, 2, 3] {
            codes.encode_rank(rank, &mut input).unwrap();
        }
        let end = input.len();
        input.resize(end + 4, 0);
        let mut output = vec![0xcc; 51];
        let n = expansions
            .byte_field(&codes, &input, 0..end, &mut output, 34)
            .unwrap();
        assert_eq!(n, 34);
        assert_eq!(output[0], b'A');
        assert_eq!(&output[1..34], &[b'x'; 33]);
        assert_eq!(output[50], 0xcc);
        assert!(expansions
            .byte_field(&codes, &input, 0..end, &mut output, 33)
            .is_err());
        for cut in 0..end {
            assert!(expansions
                .byte_field(&codes, &input, 0..cut, &mut output, 34)
                .is_err());
        }
        input[end] = 0;
        assert!(expansions
            .byte_field(&codes, &input, 0..end + 1, &mut output, 34)
            .is_err());
    }
    #[test]
    fn rejects_unused_codes_and_missing_guards() {
        let (arena, refs) = table();
        let e = Expansions::new(&arena, &refs).unwrap();
        let c = ByteCodes::with_slots(4, [256, 0, 0, 0]).unwrap();
        let mut out = [0; 50];
        assert!(e
            .byte_field(&c, &[255, 0, 0, 0], 0..1, &mut out, 34)
            .is_err());
        assert!(e.byte_field(&c, &[3], 0..1, &mut out, 34).is_err());
        assert!(e
            .byte_field(&c, &[3, 0, 0, 0], 0..1, &mut out[..15], 0)
            .is_err());
    }
    #[test]
    fn multi_byte_truncation_metadata_and_fixed_widths() {
        let (arena, refs) = table();
        let e = Expansions::new(&arena, &refs).unwrap();
        let c = ByteCodes::with_slots(4, [0, 256, 0, 0]).unwrap();
        let c = ByteCodes::from_metadata(4, &c.metadata()).unwrap();
        let mut input = Vec::new();
        for r in [0, 3] {
            c.encode_rank(r, &mut input).unwrap();
        }
        let end = input.len();
        input.extend([0; 4]);
        let mut out = [0u8; 17];
        assert_eq!(e.byte_field(&c, &input, 0..end, &mut out, 1).unwrap(), 1);
        for cut in 0..end {
            assert!(e.byte_field(&c, &input, 0..cut, &mut out, 1).is_err());
        }
        for width in 2..=4 {
            let mut input = Vec::new();
            for r in [0u32, 3] {
                input.extend_from_slice(&r.to_le_bytes()[..width]);
            }
            let end = input.len();
            input.extend([0; 4]);
            assert_eq!(
                e.fixed_field(&input, 0..end, width, &mut out, 1).unwrap(),
                1
            );
            for cut in 0..end {
                assert!(e.fixed_field(&input, 0..cut, width, &mut out, 1).is_err());
            }
        }
        assert!(ByteCodes::from_metadata(4, &[0; 16]).is_err());
    }
    #[test]
    fn direct_huffman_preserves_eof_and_bounds() {
        let (arena, refs) = table();
        let book = Codebook::from_counts(&[100, 30, 4, 20]).unwrap();
        let ranked: Vec<_> = book.symbols.iter().map(|&r| refs[r as usize]).collect();
        let e = Expansions::new(&arena, &ranked).unwrap();
        let (input, _) = book.encode(&[0, 3]).unwrap();
        let mut out = [0u8; 17];
        assert_eq!(e.huffman_field(&book, &input, &mut out, 1).unwrap(), 1);
        assert_eq!(out[0], b'A');
        assert!(e.huffman_field(&book, &input, &mut out, 0).is_err());
    }
    #[test]
    fn optimizer_uses_only_frozen_model_and_covers_alphabet() {
        let book = Codebook::from_counts(&[100, 30, 2, 1]).unwrap();
        let c = ByteCodes::from_book(&book).unwrap();
        for r in 0..4 {
            let mut out = Vec::new();
            c.encode_rank(r, &mut out).unwrap();
            assert_eq!(out.len(), 1);
        }
        assert!(ByteCodes::with_slots(65537, [0, 256, 0, 0]).is_err());
    }
}
