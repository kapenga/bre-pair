//! SPDX-License-Identifier: Apache-2.0
//! Adapted from Igor Zavadskyi and Maksym Kovalchuk, Fast-Large-Huffman,
//! licensed revision ada4c1cb10b45b4b9aecd3e88b50756facd4d89e.
//! The historical Rust adaptation is restored under the verified upstream
//! license; local changes include two-code groups, checked tails, immutable
//! tables, CPU dispatch and the original BRe-Pair byte/AI-ID expansion path.
//! See third-party/FAST-LARGE-HUFFMAN-LICENSE-APACHE.txt and
//! docs/huffman-licensed-integration.md for provenance and validation.
//! Grouped refill and staged AVX2 canonical-rank decoding, following
//! Zavadskyi/Kovalchuk (2025), DOI 10.17721/1812-5409.2025/2.25, optimizations
//! 1, 3 and 4. This Rust implementation credits that published algorithm;
//! upstream C++ source is not vendored. See docs/huffman-avx2-adoption.md.
//!
//! Unlike the experimental C++ harness, this accepts ordinary unpadded byte
//! slices, emits no surplus ranks, and preserves the checked tail. It retains
//! the portable decoder for unsupported CPUs, singleton and >31-bit tables.
use super::{Bits, Codebook, AUTO_AVX2_MIN_SYMBOLS};

const BLOCK: usize = 128;

#[derive(Clone, Debug)]
pub(crate) struct GroupedTables {
    start: [u8; 4096],
    upper: [u64; 32],
    offset: [u32; 32],
}

impl GroupedTables {
    pub(crate) fn new(book: &Codebook) -> Option<Self> {
        if !available() || book.symbols.len() == 1 || book.max_bits > 31 {
            return None;
        }
        let mut tables = Self {
            start: [0; 4096],
            upper: [0; 32],
            offset: [0; 32],
        };
        let min = usize::from(book.lengths[0]);
        for len in min..=usize::from(book.max_bits) {
            // Exclusive upper bounds also skip lengths with no codes.
            tables.upper[len] = (book.first[len] + u64::from(book.counts[len])) << (32 - len);
            tables.offset[len] = book.ranks[len].wrapping_sub(book.first[len] as u32);
        }
        for (prefix, start) in tables.start.iter_mut().enumerate() {
            let top = (prefix as u64) << 20;
            let mut len = min;
            while top >= tables.upper[len] {
                len += 1;
            }
            *start = len as u8;
        }
        Some(tables)
    }
}

pub(super) fn available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

impl Codebook {
    pub(in crate::huffman) fn decode_dispatch<const FORCE: bool>(
        &self,
        payload: &[u8],
        bits: u64,
        count: u64,
        emit: impl FnMut(u32) -> Result<(), String>,
    ) -> Result<(), String> {
        #[cfg(target_arch = "x86_64")]
        if let Some(tables) = &self.grouped {
            // The full-archive comparison regressed on medium alphabets. Keep
            // automatic SIMD for very large tables; forced research variants
            // still cover small tables. This threshold is an engineering guard.
            if (FORCE || self.force_grouped || self.symbols.len() >= AUTO_AVX2_MIN_SYMBOLS)
                && available()
            {
                // Tables are private and constructed only for complete,
                // non-singleton codebooks with maximum length <=31. The CPU
                // guard precedes entry to a target_feature AVX2 function.
                return unsafe { self.decode_grouped_avx2(tables, payload, bits, count, emit) };
            }
        }
        self.decode_bulk::<true, false>(payload, bits, count, emit)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn decode_grouped_avx2(
        &self,
        tables: &GroupedTables,
        payload: &[u8],
        bits: u64,
        count: u64,
        mut emit: impl FnMut(u32) -> Result<(), String>,
    ) -> Result<(), String> {
        use std::arch::x86_64::*;

        self.validate_payload(payload, bits, count)?;
        let max = u64::from(self.max_bits);
        // Reserve two extra input words for full, unpadded, in-bounds reads
        // throughout the block. Actual code consumption is <= BLOCK * max.
        let required = BLOCK as u64 * max + 128;
        if count < BLOCK as u64 || bits < required {
            return self.decode_bulk::<true, false>(payload, bits, count, emit);
        }

        // The block admission test proves every load ends before
        // consumed + BLOCK*max + 128 <= bits <= payload.len()*8. Use unaligned
        // word loads inside that bounded region, avoiding per-load slice checks.
        let read =
            |at: usize| u64::from_be(std::ptr::read_unaligned(payload.as_ptr().add(at).cast()));
        let mut word = read(0);
        let (mut base, mut shift, mut consumed, mut remaining) = (0usize, 0u32, 0u64, count);
        let (mut words, mut lengths) = ([0u32; BLOCK], [0u32; BLOCK]);
        let width = _mm256_set1_epi32(32);
        while remaining >= BLOCK as u64 && bits - consumed >= required {
            for i in (0..BLOCK).step_by(2) {
                // The previous refill supplied a full 64-bit window. Two
                // codes need at most 62 bits; no refill test inside this group.
                for j in 0..2 {
                    let top = (word >> 32) as u32;
                    let mut len = usize::from(tables.start[(word >> 52) as usize]);
                    while u64::from(top) >= tables.upper[len] {
                        len += 1;
                    }
                    words[i + j] = top;
                    lengths[i + j] = len as u32;
                    word <<= len;
                    shift += len as u32;
                }

                // After a positive-sized pair, shift is in 1..=126. At most
                // one word boundary is crossed. Both shifts below are <64.
                if shift > 64 {
                    shift -= 64;
                    base += 8;
                    word |= read(base) << shift;
                }
                word |= read(base + 8) >> (64 - shift);
            }

            // AVX2 works on eight already delimited codes. Reuse words for
            // ranks, avoiding a third scratch array or caller output padding.
            for i in (0..BLOCK).step_by(8) {
                let top = _mm256_loadu_si256(words.as_ptr().add(i).cast());
                let len = _mm256_loadu_si256(lengths.as_ptr().add(i).cast());
                let code = _mm256_srlv_epi32(top, _mm256_sub_epi32(width, len));
                // All initialized length lanes are 1..=31, in offset[32].
                let offset = _mm256_i32gather_epi32::<4>(tables.offset.as_ptr().cast(), len);
                let ranks = _mm256_add_epi32(code, offset);
                _mm256_storeu_si256(words.as_mut_ptr().add(i).cast(), ranks);
            }
            for &rank in &words {
                emit(rank)?;
            }
            remaining -= BLOCK as u64;
            consumed = base as u64 * 8 + u64::from(shift);
        }

        // Restart the ordinary byte reader at exactly the consumed bit index.
        // No speculative padding bits or extra ranks escape to the caller.
        let mut tail = Bits {
            at: (consumed / 8) as usize,
            ..Bits::default()
        };
        let skip = (consumed % 8) as u8;
        if skip != 0 {
            tail.word = u64::from(payload[tail.at]) << (56 + skip);
            tail.available = 8 - skip;
            tail.at += 1;
        }
        self.checked_tail::<true, false>(payload, bits, remaining, &mut tail, consumed, emit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn comb(depth: u8) -> Codebook {
        let mut entries: Vec<_> = (1..depth).map(|len| (u32::from(len - 1), len)).collect();
        entries.push((u32::from(depth - 1), depth));
        entries.push((u32::from(depth), depth));
        Codebook::from_lengths(&entries).unwrap()
    }

    fn run(
        book: &Codebook,
        payload: &[u8],
        bits: u64,
        count: u64,
        reference: bool,
    ) -> (bool, Vec<u32>) {
        let mut output = Vec::new();
        let emit = |rank| {
            output.push(rank);
            Ok(())
        };
        let result = if reference {
            book.decode_ranks_reference(payload, bits, count, emit)
        } else {
            book.decode_dispatch::<true>(payload, bits, count, emit)
        };
        (result.is_ok(), output)
    }

    #[test]
    fn grouped_blocks_word_boundaries_unaligned_input_and_checked_tails() {
        for depth in [2, 12, 23, 27, 31, 32, 56] {
            let book = comb(depth);
            assert_eq!(book.grouped.is_some(), available() && depth <= 31);
            for prefix in 0..8 {
                for count in [127, 128, 129, 255, 256, 257, 4097] {
                    let ids: Vec<_> = std::iter::repeat_n(0, prefix)
                        .chain((0..count).map(|i| if i % 3 == 0 { 0 } else { u32::from(depth) }))
                        .collect();
                    let (payload, bits) = book.encode(&ids).unwrap();
                    // Force an unaligned payload start; no trailing padding.
                    let mut unaligned = vec![0];
                    unaligned.extend_from_slice(&payload);
                    let expected = run(&book, &payload, bits, ids.len() as u64, true);
                    assert!(expected.0);
                    assert_eq!(
                        run(&book, &unaligned[1..], bits, ids.len() as u64, false),
                        expected
                    );
                }
            }
            let ids = vec![u32::from(depth); 4097];
            let (payload, bits) = book.encode(&ids).unwrap();
            assert_eq!(
                run(&book, &payload, bits, ids.len() as u64, false),
                run(&book, &payload, bits, ids.len() as u64, true)
            );
            let mut fallback = book.clone();
            fallback.grouped = None;
            assert_eq!(
                run(&fallback, &payload, bits, ids.len() as u64, false),
                run(&book, &payload, bits, ids.len() as u64, true)
            );
        }
    }

    #[test]
    fn grouped_errors_preserve_reference_prefix_and_callback_failure() {
        let book = comb(27);
        let ids: Vec<_> = (0..4097).map(|i| if i % 5 == 0 { 0 } else { 27 }).collect();
        let (payload, bits) = book.encode(&ids).unwrap();
        let bulk = BLOCK as u64 * 27 + 128;
        for cut in [
            1,
            63,
            64,
            65,
            bulk - 1,
            bulk,
            bulk + 1,
            bits - 27,
            bits - 1,
            bits,
        ] {
            let mut data = payload[..cut.div_ceil(8) as usize].to_vec();
            if cut % 8 != 0 {
                *data.last_mut().unwrap() &= 255u8 << (8 - cut % 8);
            }
            for count in [ids.len() as u64 - 1, ids.len() as u64, ids.len() as u64 + 1] {
                assert_eq!(
                    run(&book, &data, cut, count, false),
                    run(&book, &data, cut, count, true),
                    "cut={cut},count={count}"
                );
            }
            if cut % 8 != 0 {
                *data.last_mut().unwrap() |= 1;
                assert!(!run(&book, &data, cut, ids.len() as u64, false).0);
            }
        }
        let mut prefix = Vec::new();
        assert_eq!(
            book.decode_dispatch::<true>(&payload, bits, ids.len() as u64, |rank| {
                if prefix.len() == 129 {
                    return Err("consumer stopped".into());
                }
                prefix.push(rank);
                Ok(())
            }),
            Err("consumer stopped".into())
        );
        assert_eq!(
            prefix,
            run(&book, &payload, bits, ids.len() as u64, true).1[..129]
        );
    }

    #[test]
    fn automatic_large_alphabet_dispatch_matches_portable_decoder() {
        let counts: Vec<_> = (0..AUTO_AVX2_MIN_SYMBOLS + 1)
            .map(|id| 1 + (id % 5) as u64)
            .collect();
        let book = Codebook::from_counts(&counts).unwrap();
        assert_eq!(
            book.decoder_backend(),
            if available() {
                "grouped-avx2"
            } else {
                "decoder2"
            }
        );
        let ids: Vec<_> = (0..32769).map(|i| (i * 73 % counts.len()) as u32).collect();
        let (payload, bits) = book.encode(&ids).unwrap();
        let mut automatic = Vec::new();
        book.decode_ranks(&payload, bits, ids.len() as u64, |rank| {
            automatic.push(rank);
            Ok(())
        })
        .unwrap();
        assert_eq!(
            automatic,
            run(&book, &payload, bits, ids.len() as u64, true).1
        );
    }
}
