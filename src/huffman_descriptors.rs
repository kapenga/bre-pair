//! Optional canonical coding of rule arities, used only at dictionary load time.
use crate::huffman::{get_var, put_var, Codebook};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Coding {
    #[default]
    Auto,
    Varint,
}
impl Coding {
    pub fn name(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Varint => "varint",
        }
    }
}
impl std::str::FromStr for Coding {
    type Err = String;
    fn from_str(value: &str) -> Result<Self, String> {
        match value {
            "auto" => Ok(Self::Auto),
            "varint" => Ok(Self::Varint),
            _ => Err("rule lengths must be auto or varint".into()),
        }
    }
}

pub(super) fn visit_raw(
    mut bytes: &[u8],
    leaves: u64,
    mut emit: impl FnMut(u64, Option<u64>) -> Result<(), String>,
) -> Result<(), String> {
    for _ in 0..leaves {
        let arity = get_var(&mut bytes)?;
        if arity == 1 {
            return Err("invalid shared definition arity".into());
        }
        let literal = if arity == 0 {
            Some(get_var(&mut bytes)?)
        } else {
            None
        };
        emit(arity, literal)?;
    }
    if !bytes.is_empty() {
        return Err("shared descriptors have trailing bytes".into());
    }
    Ok(())
}

/// None means the original byte sequence is smaller (ties retain the old format).
pub(super) fn compress(bytes: &[u8], leaves: u64) -> Result<Option<Vec<u8>>, String> {
    if leaves == 0 {
        return Ok(None);
    }
    let mut histogram = BTreeMap::<u64, u64>::new();
    let mut literals = Vec::new();
    visit_raw(bytes, leaves, |arity, literal| {
        *histogram.entry(arity).or_default() += 1;
        if let Some(id) = literal {
            put_var(&mut literals, id);
        }
        Ok(())
    })?;
    // Dense IDs avoid an allocation indexed by a possibly enormous arity.
    let values: Vec<_> = histogram.keys().copied().collect();
    let counts: Vec<_> = histogram.values().copied().collect();
    let book = Codebook::from_counts(&counts)?;
    let mut lengths = vec![0; counts.len()];
    for (id, length) in book.lengths() {
        lengths[id as usize] = length;
    }
    let mut out = Vec::new();
    put_var(&mut out, values.len() as u64);
    for (&value, &length) in values.iter().zip(&lengths) {
        put_var(&mut out, value);
        out.push(length);
    }
    let bits = counts
        .iter()
        .zip(&lengths)
        .try_fold(0u64, |sum, (&n, &length)| {
            sum.checked_add(
                n.checked_mul(u64::from(length))
                    .ok_or("arity bit count overflow")?,
            )
            .ok_or("arity bit count overflow")
        })?;
    put_var(&mut out, bits);
    let projected = (out.len() as u128) + u128::from(bits.div_ceil(8)) + literals.len() as u128;
    if projected >= bytes.len() as u128 {
        return Ok(None);
    }
    let mut remaining = bytes;
    let ids = (0..leaves).map(|_| {
        let arity = get_var(&mut remaining)?;
        if arity == 0 {
            get_var(&mut remaining)?;
        }
        Ok(values.binary_search(&arity).map_err(|_| "unknown arity")? as u32)
    });
    out.extend(book.encoding().encode_iter(ids, bits)?);
    out.extend(literals);
    Ok(Some(out))
}

pub(super) fn visit_huffman(
    mut bytes: &[u8],
    leaves: u64,
    max_arity: u64,
    mut emit: impl FnMut(u64, Option<u64>) -> Result<(), String>,
) -> Result<(), String> {
    let used = get_var(&mut bytes)?;
    if used == 0 || used > leaves || used > (bytes.len() / 2) as u64 || used > u64::from(u32::MAX) {
        return Err("invalid arity alphabet size".into());
    }
    let mut values = Vec::with_capacity(used as usize);
    let mut entries = Vec::with_capacity(used as usize);
    for id in 0..used {
        let value = get_var(&mut bytes)?;
        if value == 1 || value > max_arity || values.last().is_some_and(|&last| value <= last) {
            return Err("invalid arity alphabet value".into());
        }
        let (&length, rest) = bytes.split_first().ok_or("truncated arity code length")?;
        bytes = rest;
        values.push(value);
        entries.push((id as u32, length));
    }
    let book = Codebook::from_lengths(&entries)?;
    let bits = get_var(&mut bytes)?;
    let size = usize::try_from(bits.div_ceil(8)).map_err(|_| "arity payload too large")?;
    if size > bytes.len() || leaves > bits {
        return Err("truncated arity payload".into());
    }
    let (payload, mut literals) = bytes.split_at(size);
    book.decode(payload, bits, leaves, |id| {
        let arity = values[id as usize];
        let literal = if arity == 0 {
            Some(get_var(&mut literals)?)
        } else {
            None
        };
        emit(arity, literal)
    })?;
    if !literals.is_empty() {
        return Err("arity literal IDs have trailing bytes".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn raw(records: &[(u64, Option<u64>)]) -> Vec<u8> {
        let mut bytes = Vec::new();
        for &(arity, literal) in records {
            put_var(&mut bytes, arity);
            if let Some(id) = literal {
                put_var(&mut bytes, id);
            }
        }
        bytes
    }
    #[test]
    fn fallback_empty_and_small_dictionaries_and_singleton_coding() {
        assert!(compress(&[], 0).unwrap().is_none());
        assert!(compress(&raw(&[(0, Some(256)), (2, None)]), 2)
            .unwrap()
            .is_none());
        let records = vec![(2, None); 1001];
        let packed = compress(&raw(&records), records.len() as u64)
            .unwrap()
            .unwrap();
        let mut decoded = Vec::new();
        visit_huffman(&packed, 1001, 2, |a, l| {
            decoded.push((a, l));
            Ok(())
        })
        .unwrap();
        assert_eq!(decoded, records);
    }
    #[test]
    fn sparse_large_arities_and_literals_round_trip_without_dense_value_allocation() {
        let mut records = vec![(2, None); 1000];
        records.extend([(0, Some(0)), (0, Some(256)), (3, None), (u64::MAX, None)]);
        let packed = compress(&raw(&records), records.len() as u64)
            .unwrap()
            .unwrap();
        let mut decoded = Vec::new();
        visit_huffman(&packed, records.len() as u64, u64::MAX, |a, l| {
            decoded.push((a, l));
            Ok(())
        })
        .unwrap();
        assert_eq!(decoded, records);
        assert!(visit_huffman(&packed, records.len() as u64, 256, |_, _| Ok(())).is_err());
    }
    #[test]
    fn corrupt_tables_payload_lengths_padding_and_literal_streams_fail() {
        let records = [(vec![(2, None); 997]), vec![(0, Some(256)); 3]].concat();
        let packed = compress(&raw(&records), records.len() as u64)
            .unwrap()
            .unwrap();
        let reject = |bytes: &[u8], count| visit_huffman(bytes, count, 2, |_, _| Ok(())).is_err();
        for end in 0..packed.len() {
            assert!(reject(&packed[..end], 1000), "prefix {end}");
        }
        assert!(reject(&packed, 999));
        assert!(reject(&packed, 1001));
        let mut extra = packed.clone();
        extra.push(0);
        assert!(reject(&extra, 1000));
        let mut bad = packed.clone();
        bad[0] = 0;
        assert!(reject(&bad, 1000));
        let mut bad = packed.clone();
        bad[2] = 0;
        assert!(reject(&bad, 1000));
        let mut bad = packed.clone();
        bad[3] = 0;
        assert!(reject(&bad, 1000));
        // All truncations must also fail for the uncompressed representation.
        let unpacked = raw(&records);
        for end in 0..unpacked.len() {
            assert!(visit_raw(&unpacked[..end], 1000, |_, _| Ok(())).is_err());
        }
    }
    #[test]
    fn nonzero_arity_padding_and_oversubscribed_codes_are_rejected() {
        let records = [vec![(2, None); 998], vec![(0, Some(256)); 3]].concat();
        let mut packed = compress(&raw(&records), records.len() as u64)
            .unwrap()
            .unwrap();
        let mut cursor = packed.as_slice();
        let used = get_var(&mut cursor).unwrap();
        for _ in 0..used {
            get_var(&mut cursor).unwrap();
            cursor = &cursor[1..];
        }
        let bits = get_var(&mut cursor).unwrap();
        assert_ne!(bits % 8, 0);
        let end = packed.len() - cursor.len() + bits.div_ceil(8) as usize;
        packed[end - 1] |= 1;
        assert!(visit_huffman(&packed, 1001, 2, |_, _| Ok(())).is_err());
        // Three symbols cannot all have one-bit codes.
        assert!(visit_huffman(&[3, 0, 1, 2, 1, 3, 1, 3, 0], 3, 3, |_, _| Ok(())).is_err());
    }
    #[test]
    fn streaming_bit_packing_matches_existing_encoder_across_word_boundaries() {
        let mut entries: Vec<_> = (0..55).map(|id| (id, id as u8 + 1)).collect();
        entries.extend([(55, 56), (56, 56)]);
        let book = Codebook::from_lengths(&entries).unwrap();
        let encoding = book.encoding();
        let ids: Vec<_> = (0..57).rev().cycle().take(1007).collect();
        let (expected, bits) = encoding.encode(&ids).unwrap();
        assert_eq!(
            encoding
                .encode_iter(ids.iter().copied().map(Ok), bits)
                .unwrap(),
            expected
        );
        assert!(encoding
            .encode_iter(ids.iter().copied().map(Ok), bits - 1)
            .is_err());
        assert!(encoding
            .encode_iter(ids.iter().copied().map(Ok), bits + 1)
            .is_err());
        assert!(encoding.encode_iter([Ok(99)], 1).is_err());
        assert_eq!(
            encoding
                .encode_iter([Err("parse failed".to_owned())], 1)
                .unwrap_err(),
            "parse failed"
        );
    }
}
