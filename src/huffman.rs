//! Canonical, MSB-first Huffman codes. Decoding uses Decoder2-style high-aligned
//! range boundaries, not a tree walk or a table exponential in code length.
//! The release path retains the owner's Decoder2-style range search, a 4 KiB
//! prefix-to-length lookup and bounded wide refills with a checked tail.
use std::{cmp::Reverse, collections::BinaryHeap};

pub const MAX_BITS: usize = 56;
#[path = "huffman_kernels.rs"]
mod kernels;
#[cfg(any(test, feature = "huffman-kernels"))]
pub use kernels::Kernel;

/// Reusable dense encoding lookup. No frequencies are updated while coding.
pub struct Encoding {
    // Canonical codes are at most 56 bits: upper 56 bits hold the code,
    // low byte holds its length. Zero denotes an absent symbol.
    codes: Vec<u64>,
}

#[derive(Clone, Debug)]
pub struct Codebook {
    /// Logical symbol IDs in canonical (length, ID) order.
    pub symbols: Vec<u32>,
    lengths: Vec<u8>,
    counts: [u32; MAX_BITS + 1],
    first: [u64; MAX_BITS + 1],
    ranks: [u32; MAX_BITS + 1],
    bias: [u64; MAX_BITS + 1],
    boundaries: Vec<(u64, u8)>,
    // Coarse prefix -> code LENGTH, not symbol. Only bins straddling a
    // length-range boundary need the compact range scan. Just 4 KiB even
    // for millions of symbols and 56-bit codes.
    prefix_lengths: Box<[u8; 4096]>,
    pub max_bits: u8,
}

impl Codebook {
    pub fn from_counts(counts: &[u64]) -> Result<Self, String> {
        let leaf_count = counts.iter().filter(|&&n| n != 0).count();
        let node_count = leaf_count
            .checked_mul(2)
            .and_then(|n| n.checked_sub(1))
            .ok_or("empty or oversized Huffman alphabet")?;
        // Exact cardinalities avoid power-of-two spare capacity and a second
        // huge allocation during parent/leaf/queue growth.
        let mut heap = BinaryHeap::with_capacity(leaf_count);
        let mut parents = Vec::with_capacity(node_count);
        let mut leaves = Vec::with_capacity(leaf_count);
        for (id, &count) in counts.iter().enumerate().filter(|(_, n)| **n != 0) {
            let id = u32::try_from(id).map_err(|_| "Huffman alphabet exceeds u32")?;
            let node = parents.len();
            parents.push(usize::MAX);
            heap.push(Reverse((count as u128, id, node)));
            leaves.push((count, id, node));
        }
        if leaves.is_empty() {
            return Err("empty Huffman alphabet".into());
        }
        while heap.len() > 1 {
            let Reverse((a, ta, na)) = heap.pop().unwrap();
            let Reverse((b, tb, nb)) = heap.pop().unwrap();
            let node = parents.len();
            parents[na] = node;
            parents[nb] = node;
            parents.push(usize::MAX);
            heap.push(Reverse((a + b, ta.min(tb), node)));
        }
        // The emptied queue can retain hundreds of MB for large alphabets.
        // Release it before deriving lengths or building the canonical table.
        drop(heap);
        // Parent IDs are greater than child IDs: one reverse pass, even for a
        // highly skewed tree. Reuse each consumed parent link as its depth;
        // later children only need the already converted parent's depth.
        for node in (0..parents.len()).rev() {
            let parent = parents[node];
            parents[node] = if parent == usize::MAX {
                0
            } else {
                parents[parent] + 1
            };
        }
        let mut histogram = [0u32; MAX_BITS + 1];
        for &(_, _, node) in &leaves {
            histogram[parents[node].clamp(1, MAX_BITS)] += 1;
        }
        drop(parents);
        if leaves.len() > 1 {
            // Length limiting is only needed for pathological distributions.
            // Split a shorter leaf, remove one maximum-depth leaf, preserving
            // leaf count while reducing the Kraft sum by one 2^-MAX_BITS unit.
            let target = 1u128 << MAX_BITS;
            let mut used: u128 = (1..=MAX_BITS)
                .map(|n| u128::from(histogram[n]) << (MAX_BITS - n))
                .sum();
            while used > target {
                let n = (1..MAX_BITS)
                    .rev()
                    .find(|&n| histogram[n] != 0)
                    .ok_or("cannot limit Huffman lengths")?;
                histogram[n] -= 1;
                histogram[n + 1] += 2;
                histogram[MAX_BITS] -= 1;
                used -= 1;
            }
        }
        leaves.sort_unstable_by_key(|&(count, id, _)| (count, Reverse(id)));
        let mut entries = Vec::with_capacity(leaves.len());
        let mut at = 0;
        for n in (1..=MAX_BITS).rev() {
            for _ in 0..histogram[n] {
                entries.push((leaves[at].1, n as u8));
                at += 1;
            }
        }
        drop(leaves);
        Self::from_lengths(&entries)
    }

    pub fn from_lengths(entries: &[(u32, u8)]) -> Result<Self, String> {
        if entries.is_empty() || entries.len() > u32::MAX as usize {
            return Err("invalid Huffman alphabet size".into());
        }
        let mut entries = entries.to_vec();
        {
            #[cfg(feature = "profile")]
            let _span = crate::profile::Span::new("huffman.table.sort_by_id");
            entries.sort_unstable();
        }
        if entries.windows(2).any(|w| w[0].0 == w[1].0) {
            return Err("duplicate Huffman symbol".into());
        }
        {
            #[cfg(feature = "profile")]
            let _span = crate::profile::Span::new("huffman.table.sort_by_length");
            entries.sort_unstable_by_key(|&(id, len)| (len, id));
        }
        Self::from_canonical(entries)
    }

    /// Private construction after ordering and uniqueness have been validated.
    /// The wire format already supplies this order; general callers still use
    /// from_lengths to sort arbitrary input.
    fn from_canonical(entries: Vec<(u32, u8)>) -> Result<Self, String> {
        if entries.is_empty() || entries.len() > u32::MAX as usize {
            return Err("invalid Huffman alphabet size".into());
        }
        #[cfg(feature = "profile")]
        let _build = crate::profile::Span::new("huffman.table.build_ranges_and_prefix");
        let mut book = Self {
            symbols: Vec::with_capacity(entries.len()),
            lengths: Vec::with_capacity(entries.len()),
            counts: [0; MAX_BITS + 1],
            first: [0; MAX_BITS + 1],
            ranks: [0; MAX_BITS + 1],
            bias: [0; MAX_BITS + 1],
            boundaries: Vec::new(),
            prefix_lengths: Box::new([0; 4096]),
            max_bits: 0,
        };
        for (id, len) in entries {
            if len == 0 || len as usize > MAX_BITS {
                return Err("Huffman length must be 1..=56".into());
            }
            book.counts[len as usize] += 1;
            book.max_bits = book.max_bits.max(len);
            book.symbols.push(id);
            book.lengths.push(len);
        }
        let (mut code, mut rank) = (0u64, 0u32);
        for n in 1..=MAX_BITS {
            code = (code + u64::from(book.counts[n - 1])) << 1;
            book.first[n] = code;
            book.ranks[n] = rank;
            {
                book.bias[n] = code.wrapping_sub(u64::from(rank));
            }
            rank += book.counts[n];
            if code + u64::from(book.counts[n]) > 1u64 << n {
                return Err("oversubscribed Huffman lengths".into());
            }
        }
        let n = book.max_bits as usize;
        if book.symbols.len() == 1 {
            if n != 1 {
                return Err("singleton Huffman code must have length one".into());
            }
        } else if book.first[n] + u64::from(book.counts[n]) != 1u64 << n {
            return Err("incomplete Huffman lengths".into());
        }
        for n in (1..=n).rev().filter(|&n| book.counts[n] != 0) {
            book.boundaries.push((book.first[n] << (64 - n), n as u8));
        }
        for prefix in 0..4096 {
            let lower = (prefix as u64) << 52;
            let upper = lower | ((1u64 << 52) - 1);
            let a = book
                .boundaries
                .iter()
                .find(|&&(b, _)| lower >= b)
                .map(|&(_, len)| len);
            let b = book
                .boundaries
                .iter()
                .find(|&&(b, _)| upper >= b)
                .map(|&(_, len)| len);
            if a == b {
                book.prefix_lengths[prefix] = a.unwrap_or(0);
            }
        }
        Ok(book)
    }

    pub fn lengths(&self) -> impl Iterator<Item = (u32, u8)> + '_ {
        self.symbols
            .iter()
            .copied()
            .zip(self.lengths.iter().copied())
    }

    pub fn encode(&self, ids: &[u32]) -> Result<(Vec<u8>, u64), String> {
        self.encoding().encode(ids)
    }
    pub fn encoding(&self) -> Encoding {
        let largest = *self.symbols.iter().max().unwrap() as usize;
        let mut codes = vec![0; largest + 1];
        for (rank, (id, len)) in self.lengths().enumerate() {
            let code = self.first[len as usize] + rank as u64 - u64::from(self.ranks[len as usize]);
            codes[id as usize] = (code << 8) | u64::from(len);
        }
        Encoding { codes }
    }
}
impl Encoding {
    /// Bounded-memory metadata coding; the main payload encoder stays unchanged.
    pub(crate) fn encode_iter(
        &self,
        ids: impl IntoIterator<Item = Result<u32, String>>,
        expected_bits: u64,
    ) -> Result<Vec<u8>, String> {
        let mut out = Vec::with_capacity(
            usize::try_from(expected_bits.div_ceil(8)).map_err(|_| "payload too large")?,
        );
        let (mut word, mut bits, mut total) = (0u64, 0u8, 0u64);
        for id in ids {
            let packed = *self
                .codes
                .get(id? as usize)
                .ok_or("undefined Huffman symbol")?;
            let (code, len) = (packed >> 8, packed as u8);
            if len == 0 {
                return Err("undefined Huffman symbol".into());
            }
            total = total
                .checked_add(u64::from(len))
                .ok_or("Huffman bit count overflow")?;
            if total > expected_bits {
                return Err("metadata Huffman bit count differs".into());
            }
            let available = 64 - bits;
            if len < available {
                word |= code << (available - len);
                bits += len;
            } else {
                let carry = len - available;
                word |= code >> carry;
                out.extend_from_slice(&word.to_be_bytes());
                bits = carry;
                word = if carry == 0 { 0 } else { code << (64 - carry) };
            }
        }
        if total != expected_bits {
            return Err("metadata Huffman bit count differs".into());
        }
        if bits != 0 {
            out.extend_from_slice(&word.to_be_bytes()[..bits.div_ceil(8) as usize]);
        }
        Ok(out)
    }
    pub fn encode(&self, ids: &[u32]) -> Result<(Vec<u8>, u64), String> {
        let bit_count = ids.iter().try_fold(0u64, |total, &id| {
            let &packed = self
                .codes
                .get(id as usize)
                .ok_or("undefined Huffman symbol")?;
            let len = packed as u8;
            if len == 0 {
                return Err("undefined Huffman symbol");
            }
            total
                .checked_add(u64::from(len))
                .ok_or("Huffman bit count overflow")
        })?;
        let mut out = Vec::with_capacity(
            usize::try_from(bit_count.div_ceil(8)).map_err(|_| "payload too large")?,
        );
        let (mut word, mut bits) = (0u64, 0u8);
        for &id in ids {
            let packed = self.codes[id as usize];
            let (code, len) = (packed >> 8, packed as u8);
            let available = 64 - bits;
            if len < available {
                word |= code << (available - len);
                bits += len;
            } else {
                let carry = len - available;
                word |= code >> carry;
                out.extend_from_slice(&word.to_be_bytes());
                bits = carry;
                word = if carry == 0 { 0 } else { code << (64 - carry) };
            }
        }
        if bits != 0 {
            out.extend_from_slice(&word.to_be_bytes()[..bits.div_ceil(8) as usize]);
        }
        Ok((out, bit_count))
    }
}
impl Codebook {
    /// The caller can fuse rank-to-arena expansion into the hot decoding loop.
    pub fn decode(
        &self,
        payload: &[u8],
        bit_count: u64,
        count: u64,
        mut emit: impl FnMut(u32) -> Result<(), String>,
    ) -> Result<(), String> {
        self.decode_ranks(payload, bit_count, count, |rank| {
            emit(self.symbols[rank as usize])
        })
    }
    /// Canonical ranks let a decoder prepare its slice references in code
    /// order, avoiding a rank -> logical ID -> dictionary lookup per symbol.
    pub fn decode_ranks(
        &self,
        payload: &[u8],
        bit_count: u64,
        count: u64,
        emit: impl FnMut(u32) -> Result<(), String>,
    ) -> Result<(), String> {
        self.decode_bulk::<true, false>(payload, bit_count, count, emit)
    }
    fn decode_ranks_reference(
        &self,
        payload: &[u8],
        bit_count: u64,
        count: u64,
        mut emit: impl FnMut(u32) -> Result<(), String>,
    ) -> Result<(), String> {
        if bit_count.div_ceil(8) != payload.len() as u64 || count > bit_count {
            return Err("invalid Huffman payload size".into());
        }
        if bit_count % 8 != 0
            && payload
                .last()
                .is_some_and(|b| b & ((1 << (8 - bit_count % 8)) - 1) != 0)
        {
            return Err("nonzero Huffman padding".into());
        }
        let (mut word, mut available, mut at, mut consumed) = (0u64, 0u8, 0usize, 0u64);
        for _ in 0..count {
            // At most 55 valid bits before refill, so adding a byte fits u64.
            if available < self.max_bits && payload.len() - at >= 8 {
                word |= u64::from_be_bytes(payload[at..at + 8].try_into().unwrap()) >> available;
                let take = (64 - available) / 8;
                available += take * 8;
                at += take as usize;
                word &= u64::MAX << (64 - available);
            }
            while available < self.max_bits && at < payload.len() {
                word |= u64::from(payload[at]) << (56 - available);
                available += 8;
                at += 1;
            }
            let mut len = self.prefix_lengths[(word >> 52) as usize];
            if len == 0 {
                for &(boundary, length) in &self.boundaries {
                    if word >= boundary {
                        len = length;
                        break;
                    }
                }
            }
            if len == 0 {
                return Err("invalid Huffman prefix".into());
            }
            if consumed + u64::from(len) > bit_count {
                return Err("truncated Huffman code".into());
            }
            let index = (word >> (64 - len)) - self.first[len as usize];
            if index >= u64::from(self.counts[len as usize]) {
                return Err("invalid Huffman code".into());
            }
            emit(self.ranks[len as usize] + index as u32)?;
            word <<= len;
            available -= len;
            consumed += u64::from(len);
        }
        if consumed != bit_count {
            return Err("unused Huffman payload bits".into());
        }
        Ok(())
    }
}

pub(crate) fn put_var(out: &mut Vec<u8>, mut n: u64) {
    while n >= 128 {
        out.push(n as u8 | 128);
        n >>= 7;
    }
    out.push(n as u8);
}
pub(crate) fn get_var(input: &mut &[u8]) -> Result<u64, String> {
    let mut n = 0u64;
    for shift in (0..=63).step_by(7) {
        let (&byte, rest) = input.split_first().ok_or("truncated varint")?;
        *input = rest;
        if shift == 63 && byte > 1 {
            return Err("varint overflow".into());
        }
        n |= u64::from(byte & 127) << shift;
        if byte < 128 {
            if shift != 0 && byte == 0 {
                return Err("noncanonical varint".into());
            }
            return Ok(n);
        }
    }
    Err("varint overflow".into())
}

impl Codebook {
    pub fn to_bytes(&self) -> Vec<u8> {
        let min = self.lengths[0];
        let mut out = vec![min, self.max_bits];
        for n in min..=self.max_bits {
            put_var(&mut out, u64::from(self.counts[n as usize]));
        }
        let (mut previous, mut last_length) = (0u64, 0u8);
        for (id, len) in self.lengths() {
            if len != last_length {
                previous = 0;
                last_length = len;
            }
            put_var(&mut out, u64::from(id) - previous);
            previous = u64::from(id) + 1;
        }
        out
    }
    pub fn from_bytes(mut input: &[u8], used: usize, alphabet: usize) -> Result<Self, String> {
        #[cfg(feature = "profile")]
        let parse = crate::profile::Span::new("huffman.table.parse_wire");
        if input.len() < 2
            || used == 0
            || used > alphabet
            || used > input.len()
            || used > u32::MAX as usize
            || alphabet as u64 > u64::from(u32::MAX) + 1
        {
            return Err("invalid Huffman table size".into());
        }
        let (min, max) = (input[0], input[1]);
        input = &input[2..];
        if min == 0 || max < min || max as usize > MAX_BITS {
            return Err("invalid Huffman table lengths".into());
        }
        let mut histogram = Vec::new();
        let mut total = 0u64;
        for _ in min..=max {
            let n = get_var(&mut input)?;
            total = total.checked_add(n).ok_or("table count overflow")?;
            histogram.push(n);
        }
        if total != used as u64 {
            return Err("Huffman table count mismatch".into());
        }
        let mut entries = Vec::with_capacity(used);
        // At most eight scratch bytes per used symbol. Dense alphabets use
        // a small bitset; sparse u32 IDs use a bounded ID-only uniqueness sort
        // rather than allocating a bitset for the entire ID space.
        let mut seen =
            (alphabet <= used.saturating_mul(64)).then(|| vec![0u64; alphabet.div_ceil(64)]);
        for (offset, n) in histogram.into_iter().enumerate() {
            let mut previous = 0u64;
            for _ in 0..n {
                let id = previous
                    .checked_add(get_var(&mut input)?)
                    .ok_or("symbol ID overflow")?;
                if id >= alphabet as u64 {
                    return Err("undefined Huffman table symbol".into());
                }
                if let Some(seen) = &mut seen {
                    let word = &mut seen[id as usize / 64];
                    let mask = 1u64 << (id % 64);
                    if *word & mask != 0 {
                        return Err("duplicate Huffman symbol".into());
                    }
                    *word |= mask;
                }
                entries.push((id as u32, min + offset as u8));
                previous = id + 1;
            }
        }
        if !input.is_empty() {
            return Err("trailing Huffman table data".into());
        }
        if seen.is_none() {
            let mut ids: Vec<_> = entries.iter().map(|&(id, _)| id).collect();
            ids.sort_unstable();
            if ids.windows(2).any(|w| w[0] == w[1]) {
                return Err("duplicate Huffman symbol".into());
            }
        }
        #[cfg(feature = "profile")]
        drop(parse);
        Self::from_canonical(entries)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ordered_wire_loader_preserves_codes_and_rejects_cross_length_duplicates() {
        for alphabet in [3, 257, 32768] {
            let counts: Vec<_> = (0..alphabet).map(|i| (i as u64 % 113) + 1).collect();
            let original = Codebook::from_counts(&counts).unwrap();
            let loaded = Codebook::from_bytes(&original.to_bytes(), alphabet, alphabet).unwrap();
            assert_eq!(
                original.lengths().collect::<Vec<_>>(),
                loaded.lengths().collect::<Vec<_>>()
            );
            let ids: Vec<_> = (0..alphabet as u32).rev().collect();
            let (payload, bits) = original.encode(&ids).unwrap();
            let mut actual = Vec::new();
            loaded
                .decode(&payload, bits, ids.len() as u64, |id| {
                    actual.push(id);
                    Ok(())
                })
                .unwrap();
            assert_eq!(actual, ids);
        }
        // Complete tree, but symbol 7 appears at both lengths 1 and 2.
        let malformed = [1, 2, 1, 2, 7, 7, 0];
        for alphabet in [10, 1000000] {
            assert!(Codebook::from_bytes(&malformed, 3, alphabet).is_err());
        }
        // A tiny sparse table must not allocate for all 2^32 possible IDs.
        let original = Codebook::from_lengths(&[(u32::MAX, 1)]).unwrap();
        let loaded = Codebook::from_bytes(&original.to_bytes(), 1, u32::MAX as usize + 1).unwrap();
        let mut actual = Vec::new();
        loaded
            .decode(&[0], 1, 1, |id| {
                actual.push(id);
                Ok(())
            })
            .unwrap();
        assert_eq!(actual, [u32::MAX]);
    }
    #[test]
    fn canonical_known_vector_and_singleton() {
        let book = Codebook::from_lengths(&[(0, 1), (1, 2), (2, 3), (3, 3)]).unwrap();
        let (bytes, bits) = book.encode(&[0, 1, 2, 3]).unwrap();
        assert_eq!((bytes.clone(), bits), (vec![0b01011011, 0b10000000], 9));
        let mut decoded = Vec::new();
        book.decode(&bytes, bits, 4, |id| {
            decoded.push(id);
            Ok(())
        })
        .unwrap();
        assert_eq!(decoded, [0, 1, 2, 3]);
        let book = Codebook::from_counts(&[0, 100]).unwrap();
        assert_eq!(book.encode(&[1; 8]).unwrap(), (vec![0], 8));
        assert!(book.decode(&[128], 1, 1, |_| Ok(())).is_err());
    }
    #[test]
    fn distributions_round_trip_and_reject_damage() {
        let mut random = 17u64;
        for alphabet in [1, 2, 3, 257, 32768] {
            let counts: Vec<_> = (0..alphabet).map(|i| (i as u64 % 29 + 1).pow(3)).collect();
            let book = Codebook::from_counts(&counts).unwrap();
            let book = Codebook::from_bytes(&book.to_bytes(), alphabet, alphabet).unwrap();
            let ids: Vec<_> = (0..10003)
                .map(|_| {
                    random = random.wrapping_mul(6364136223846793005).wrapping_add(1);
                    (random % alphabet as u64) as u32
                })
                .collect();
            let (bytes, bits) = book.encode(&ids).unwrap();
            let mut actual = Vec::new();
            book.decode(&bytes, bits, ids.len() as u64, |id| {
                actual.push(id);
                Ok(())
            })
            .unwrap();
            assert_eq!(ids, actual);
            assert!(book
                .decode(&bytes, bits, ids.len() as u64 - 1, |_| Ok(()))
                .is_err());
            if bits % 8 != 0 {
                let mut bad = bytes.clone();
                *bad.last_mut().unwrap() |= 1;
                assert!(book
                    .decode(&bad, bits, ids.len() as u64, |_| Ok(()))
                    .is_err());
            }
        }
    }
    #[test]
    fn skewed_u64_counts_are_length_limited() {
        let (mut a, mut b) = (1u64, 1u64);
        let mut counts = vec![a, b];
        for _ in 0..80 {
            let c = a + b;
            counts.push(c);
            a = b;
            b = c;
        }
        let book = Codebook::from_counts(&counts).unwrap();
        assert_eq!(book.max_bits as usize, MAX_BITS);
        let ids: Vec<_> = (0..counts.len() as u32).collect();
        let (bytes, bits) = book.encode(&ids).unwrap();
        let mut actual = Vec::new();
        book.decode(&bytes, bits, ids.len() as u64, |id| {
            actual.push(id);
            Ok(())
        })
        .unwrap();
        assert_eq!(actual, ids);
        assert!(Codebook::from_lengths(&[(0, 1), (1, 1), (2, 1)]).is_err());
        assert!(Codebook::from_lengths(&[(0, 2), (1, 2)]).is_err());
        assert!(Codebook::from_lengths(&[(0, 1), (0, 1)]).is_err());
    }
}
