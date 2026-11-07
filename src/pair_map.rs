//! Hash tables for internal numeric symbol pairs. The two u32 IDs form one
//! u64 before mixing; no byte-string or cryptographic hashing is needed here.
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

use crate::grammar::Pair;

pub type PairMap<V> = HashMap<Pair, V, BuildHasherDefault<PairHasher>>;

#[inline]
pub fn mix64(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[derive(Clone, Default)]
pub struct PairHasher(u64);

impl Hasher for PairHasher {
    #[inline]
    fn write_u32(&mut self, value: u32) {
        self.0 = (self.0 << 32) | u64::from(value);
    }

    #[inline]
    fn write_u64(&mut self, value: u64) {
        self.0 = value;
    }

    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 = self.0.rotate_left(8) ^ u64::from(byte);
        }
    }

    #[inline]
    fn finish(&self) -> u64 {
        // Mix both IDs into both table-index and control-tag bits. Identity
        // hashing is particularly poor for consecutive or shared symbol IDs.
        mix64(self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::hash::Hash;

    #[test]
    fn symbol_tuples_hash_as_their_full_packed_64_bit_key() {
        for pair in [
            (0, 0),
            (0, u32::MAX),
            (u32::MAX, 0),
            (u32::MAX, u32::MAX),
            (0x1234_5678, 0x9abc_def0),
        ] {
            let mut tuple = PairHasher::default();
            pair.hash(&mut tuple);
            let mut packed = PairHasher::default();
            packed.write_u64((u64::from(pair.0) << 32) | u64::from(pair.1));
            assert_eq!(tuple.finish(), packed.finish());
        }
    }
}
