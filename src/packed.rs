//! Fixed-width symbols, packed into little-endian 64-bit words.

#[derive(Clone, Debug)]
pub struct PackedSymbols {
    words: Vec<u64>,
    len: usize,
    width: u8,
}

/// Forward compaction state. Full output words are committed only after their
/// input bits have been consumed; the final partial word is written at finish.
pub(crate) struct RewriteCursor {
    word: usize,
    buffer: u64,
    bits: u32,
    len: usize,
}

pub fn symbol_width(alphabet_size: usize) -> u8 {
    (usize::BITS - (alphabet_size.saturating_sub(1)).leading_zeros()).max(1) as u8
}

impl PackedSymbols {
    pub fn new(width: u8) -> Self {
        assert!((1..=32).contains(&width));
        Self {
            words: Vec::new(),
            len: 0,
            width,
        }
    }

    pub fn with_capacity(width: u8, symbols: usize) -> Self {
        assert!((1..=32).contains(&width));
        Self {
            words: Vec::with_capacity(
                symbols
                    .checked_mul(width as usize)
                    .expect("packed capacity overflow")
                    .div_ceil(64),
            ),
            len: 0,
            width,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn width(&self) -> u8 {
        self.width
    }
    pub fn payload_bytes(&self) -> usize {
        (self.len * self.width as usize).div_ceil(8)
    }
    pub fn storage_bytes(&self) -> usize {
        self.words.len() * 8
    }
    /// Actual requested allocation, including spare capacity after compaction.
    pub fn allocated_bytes(&self) -> usize {
        self.words.capacity() * 8
    }

    pub fn shrink_to_fit(&mut self) {
        self.words.shrink_to_fit();
    }

    pub fn get(&self, index: usize) -> u32 {
        assert!(index < self.len);
        let bit = index * self.width as usize;
        let word = bit / 64;
        let offset = bit % 64;
        let mut value = self.words[word] >> offset;
        if offset + self.width as usize > 64 {
            value |= self.words[word + 1] << (64 - offset);
        }
        (value & ((1u64 << self.width) - 1)) as u32
    }

    pub fn push(&mut self, value: u32) {
        assert!((value as u64) < (1u64 << self.width));
        let bit = self.len * self.width as usize;
        if (bit + self.width as usize).div_ceil(64) > self.words.len() {
            self.words.push(0);
        }
        let word = bit / 64;
        let offset = bit % 64;
        self.words[word] |= (value as u64) << offset;
        if offset + self.width as usize > 64 {
            self.words[word + 1] |= (value as u64) >> (64 - offset);
        }
        self.len += 1;
    }

    pub fn iter(&self) -> impl Iterator<Item = u32> + Clone + '_ {
        (0..self.len).map(|i| self.get(i))
    }

    pub(crate) fn rewrite_from(&self, prefix: usize) -> RewriteCursor {
        assert!(prefix <= self.len);
        let bit = prefix * self.width as usize;
        let word = bit / 64;
        let bits = (bit % 64) as u32;
        RewriteCursor {
            word,
            buffer: if bits == 0 {
                0
            } else {
                self.words[word] & ((1u64 << bits) - 1)
            },
            bits,
            len: prefix,
        }
    }

    /// The caller must consume at least as many original bits as it emits.
    /// Keep the original length until finish so the read cursor can use `get`.
    #[inline]
    pub(crate) fn push_rewritten(&mut self, cursor: &mut RewriteCursor, value: u32) {
        debug_assert!(cursor.len < self.len);
        debug_assert!((value as u64) < (1u64 << self.width));
        let offset = cursor.bits;
        cursor.buffer |= (value as u64) << offset;
        cursor.bits += self.width as u32;
        if cursor.bits >= 64 {
            self.words[cursor.word] = cursor.buffer;
            cursor.word += 1;
            cursor.bits -= 64;
            cursor.buffer = if cursor.bits == 0 {
                0
            } else {
                (value as u64) >> (64 - offset)
            };
        }
        cursor.len += 1;
    }

    pub(crate) fn finish_rewrite(&mut self, mut cursor: RewriteCursor) {
        if cursor.bits != 0 {
            self.words[cursor.word] = cursor.buffer;
            cursor.word += 1;
        }
        self.len = cursor.len;
        self.words.truncate(cursor.word);
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.words.len() * 8);
        for word in &self.words {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        bytes.truncate(self.payload_bytes());
        bytes
    }

    pub fn from_bytes(width: u8, len: usize, bytes: &[u8]) -> Result<Self, String> {
        if !(1..=32).contains(&width) {
            return Err("invalid symbol width".into());
        }
        let bits = len
            .checked_mul(width as usize)
            .ok_or("symbol size overflow")?;
        if bits.div_ceil(8) != bytes.len() {
            return Err("incorrect packed payload length".into());
        }
        let mut words = Vec::with_capacity(bytes.len().div_ceil(8));
        for chunk in bytes.chunks(8) {
            let mut word = [0u8; 8];
            word[..chunk.len()].copy_from_slice(chunk);
            words.push(u64::from_le_bytes(word));
        }
        Ok(Self { words, len, width })
    }
}

#[cfg(test)]
mod large_length_tests {
    use super::*;

    #[test]
    fn in_place_compaction_preserves_unread_bits_and_allocation_at_every_width() {
        for width in 1..=32 {
            let mask = ((1u64 << width) - 1) as u32;
            for len in [0, 1, 2, 7, 63, 64, 65, 127, 257] {
                let original: Vec<_> = (0..len)
                    .map(|i| crate::pair_map::mix64(i as u64 + 1) as u32 & mask)
                    .collect();
                for prefix in 0..=len.min(64) {
                    let mut packed = PackedSymbols::with_capacity(width, len);
                    for &value in &original {
                        packed.push(value);
                    }
                    let pointer = packed.words.as_ptr();
                    let capacity = packed.words.capacity();
                    let mut cursor = packed.rewrite_from(prefix);
                    let mut expected = original[..prefix].to_vec();
                    let mut read = prefix;
                    while read < len {
                        assert_eq!(packed.get(read), original[read]);
                        let a = packed.get(read);
                        let b = if read + 1 < len {
                            Some(packed.get(read + 1))
                        } else {
                            None
                        };
                        if let Some(b) = b {
                            assert_eq!(b, original[read + 1]);
                        }
                        let value = if read % 3 != 0 && b.is_some() {
                            read += 2;
                            (a ^ b.unwrap()).wrapping_add(1) & mask
                        } else {
                            read += 1;
                            a
                        };
                        packed.push_rewritten(&mut cursor, value);
                        expected.push(value);
                        // Catch accidental writes into the unread part of a
                        // shared backing word, including cross-word symbols.
                        if read < len {
                            assert_eq!(packed.get(read), original[read]);
                        }
                    }
                    packed.finish_rewrite(cursor);
                    assert_eq!(packed.words.as_ptr(), pointer);
                    assert_eq!(packed.words.capacity(), capacity);
                    assert_eq!(packed.iter().collect::<Vec<_>>(), expected);
                    let mut reference = PackedSymbols::new(width);
                    for &value in &expected {
                        reference.push(value);
                    }
                    assert_eq!(packed.to_bytes(), reference.to_bytes());
                }
            }
        }
    }

    #[test]
    fn packed_size_arithmetic_preserves_lengths_above_u32() {
        // Metadata-only fixture: check the actual size calculation without
        // allocating billions of symbols. No packed read is performed.
        let mut metadata = PackedSymbols {
            words: Vec::new(),
            len: (1usize << 32) + 7,
            width: 9,
        };
        assert_eq!(metadata.len(), 4_294_967_303);
        assert_eq!(metadata.payload_bytes(), 4_831_838_216);
        metadata.width = 32;
        assert_eq!(metadata.payload_bytes(), 17_179_869_212);
        assert!(PackedSymbols::from_bytes(9, metadata.len(), &[]).is_err());
    }
}
