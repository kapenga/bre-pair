//! Read-only feasibility study of separate data/reference Huffman tables.
//! This writes no archive and changes no production format. Complete size is
//! projected from exact recoded streams plus an actually encoded length array,
//! preserving every existing model/arity/header/index byte and charging nine
//! extra bytes for length-section framing/mode.
use super::{add, error, get_var, put_var, Codebook, Reader};
use std::{
    collections::BTreeMap,
    fs::File,
    io::{BufReader, Read, Seek},
    path::Path,
};

fn delta_length(length: u8, base: u8, relative: bool) -> u16 {
    if !relative || length == 0 {
        return u16::from(length);
    }
    let d = i16::from(length) - i16::from(base);
    (if d >= 0 { d * 2 } else { -d * 2 - 1 }) as u16 + 1
}
fn restore_length(value: u16, base: u8, relative: bool) -> Result<u8, String> {
    if value == 0 {
        return Ok(0);
    }
    let length = if relative {
        let zig = value as i16 - 1;
        i16::from(base)
            + if zig % 2 == 0 {
                zig / 2
            } else {
                -(zig + 1) / 2
            }
    } else {
        value as i16
    };
    if !(1..=56).contains(&length) {
        return Err("invalid projected code length".into());
    }
    Ok(length as u8)
}
fn encode_lengths(
    base: &[u8],
    data: &[u8],
    refs: &[u8],
    relative: bool,
) -> Result<Vec<u8>, String> {
    if base.len() != data.len() || base.len() != refs.len() {
        return Err("length array sizes differ".into());
    }
    if base.is_empty() {
        return Ok(Vec::new());
    }
    let pairs: Vec<_> = base
        .iter()
        .zip(data)
        .zip(refs)
        .map(|((&b, &d), &r)| (delta_length(d, b, relative) << 7) | delta_length(r, b, relative))
        .collect();
    let mut histogram = BTreeMap::<u16, u64>::new();
    for &pair in &pairs {
        *histogram.entry(pair).or_default() += 1;
    }
    let values: Vec<_> = histogram.keys().copied().collect();
    let counts: Vec<_> = histogram.values().copied().collect();
    let book = Codebook::from_counts(&counts)?;
    let mut lengths = vec![0; values.len()];
    let mut bits = 0u64;
    for (id, len) in book.lengths() {
        lengths[id as usize] = len;
        add(
            &mut bits,
            counts[id as usize]
                .checked_mul(u64::from(len))
                .ok_or("length stream overflow")?,
        )?;
    }
    let mut out = Vec::new();
    put_var(&mut out, values.len() as u64);
    for (&value, &length) in values.iter().zip(&lengths) {
        put_var(&mut out, u64::from(value));
        out.push(length);
    }
    put_var(&mut out, bits);
    out.extend(book.encoding().encode_iter(
        pairs.iter().map(|p| {
            values
                .binary_search(p)
                .map(|id| id as u32)
                .map_err(|_| "missing length pair".into())
        }),
        bits,
    )?);
    Ok(out)
}
fn decode_lengths(
    mut bytes: &[u8],
    base: &[u8],
    relative: bool,
) -> Result<(Vec<u8>, Vec<u8>), String> {
    if base.is_empty() {
        return if bytes.is_empty() {
            Ok((Vec::new(), Vec::new()))
        } else {
            Err("unexpected empty length metadata".into())
        };
    }
    let used = get_var(&mut bytes)?;
    if used == 0 || used > base.len() as u64 || used > bytes.len() as u64 / 2 {
        return Err("invalid length-pair alphabet".into());
    }
    let mut values = Vec::new();
    let mut entries = Vec::new();
    for id in 0..used {
        let value = u16::try_from(get_var(&mut bytes)?).map_err(|_| "length pair exceeds u16")?;
        if value >= 1 << 14 || values.last().is_some_and(|&v| value <= v) {
            return Err("invalid length pair order/range".into());
        }
        let (&length, rest) = bytes.split_first().ok_or("truncated length pair table")?;
        bytes = rest;
        values.push(value);
        entries.push((id as u32, length));
    }
    let book = Codebook::from_lengths(&entries)?;
    let bits = get_var(&mut bytes)?;
    if bits.div_ceil(8) != bytes.len() as u64 {
        return Err("length payload size differs".into());
    }
    let (mut data, mut refs) = (
        Vec::with_capacity(base.len()),
        Vec::with_capacity(base.len()),
    );
    book.decode(bytes, bits, base.len() as u64, |id| {
        let value = values[id as usize];
        let b = base[data.len()];
        data.push(restore_length(value >> 7, b, relative)?);
        refs.push(restore_length(value & 127, b, relative)?);
        Ok(())
    })?;
    Ok((data, refs))
}
fn optimal(counts: &[u64]) -> Result<(Vec<u8>, u64), String> {
    let mut lengths = vec![0; counts.len()];
    if !counts.iter().any(|&n| n != 0) {
        return Ok((lengths, 0));
    }
    let book = Codebook::from_counts(counts)?;
    let mut bits = 0;
    for (id, len) in book.lengths() {
        lengths[id as usize] = len;
        add(
            &mut bits,
            counts[id as usize]
                .checked_mul(u64::from(len))
                .ok_or("coded count overflow")?,
        )?;
    }
    Ok((lengths, bits))
}

fn restore_book(lengths: &[u8]) -> Result<Option<Codebook>, String> {
    let entries: Vec<_> = lengths
        .iter()
        .enumerate()
        .filter(|(_, len)| **len != 0)
        .map(|(id, &len)| (id as u32, len))
        .collect();
    if entries.is_empty() {
        Ok(None)
    } else {
        Ok(Some(Codebook::from_lengths(&entries)?))
    }
}

pub struct TableProjection {
    archive: u64,
    projected: u64,
    nodes: usize,
    zero_data: usize,
    old_data_bits: u64,
    data_bits: u64,
    old_ref_bits: u64,
    ref_bits: u64,
    old_data_bytes: u64,
    data_bytes: u64,
    absolute: usize,
    relative: usize,
}
impl TableProjection {
    pub fn json(&self) -> String {
        format!(concat!("{{\"scope\":\"projection only; no new archive format or production changes\",",
            "\"source_archive_bytes\":{},\"projected_archive_bytes\":{},\"projected_saving_bytes\":{},",
            "\"nodes\":{},\"zero_data_helpers\":{},\"shared_data_bits\":{},\"separate_data_bits\":{},",
            "\"shared_reference_bits\":{},\"separate_reference_bits\":{},\"shared_data_padded_bytes\":{},",
            "\"separate_data_padded_bytes\":{},\"absolute_lengths_bytes\":{},\"relative_lengths_bytes\":{},",
            "\"extra_section_framing_bytes\":9,\"all_existing_model_arity_header_index_bytes_retained\":true,",
            "\"length_array_round_trip\":true,\"all_recoded_symbol_streams_verified\":true}}"),
            self.archive,self.projected,i128::from(self.archive)-i128::from(self.projected),self.nodes,self.zero_data,
            self.old_data_bits,self.data_bits,self.old_ref_bits,self.ref_bits,self.old_data_bytes,self.data_bytes,self.absolute,self.relative)
    }
}
pub fn project_tables(source: &Path) -> Result<TableProjection, String> {
    project(&mut Reader::new(BufReader::new(
        File::open(source).map_err(error)?,
    ))?)
}
fn project<R: Read + Seek>(reader: &mut Reader<R>) -> Result<TableProjection, String> {
    let n = reader.stats.leaves;
    let (mut data, mut refs, mut base) = (vec![0u64; n], vec![0u64; n], vec![0u8; n]);
    for &id in &reader.graph.children {
        add(&mut refs[id as usize], 1)?;
    }
    if let Some(book) = &reader.book {
        for (id, length) in book.lengths() {
            base[id as usize] = length;
        }
    }
    let mut old_data_bytes = 0;
    for i in 0..reader.index.len() {
        let block = reader.block(i)?;
        add(&mut old_data_bytes, block.bits.div_ceil(8))?;
        block
            .book
            .decode(&block.payload, block.bits, block.totals.symbols, |id| {
                add(&mut data[id as usize], 1)
            })?;
    }
    let weighted = |counts: &[u64]| {
        counts
            .iter()
            .zip(&base)
            .map(|(&n, &l)| u128::from(n) * u128::from(l))
            .sum::<u128>()
    };
    if weighted(&data) != u128::from(reader.stats.payload_bits)
        || weighted(&refs) != u128::from(reader.stats.reference_bits)
    {
        return Err("projection source weighted bits differ".into());
    }
    let (data_lengths, data_bits) = optimal(&data)?;
    let (ref_lengths, ref_bits) = optimal(&refs)?;
    let absolute = encode_lengths(&base, &data_lengths, &ref_lengths, false)?;
    let relative = encode_lengths(&base, &data_lengths, &ref_lengths, true)?;
    for (encoded, is_relative) in [(&absolute, false), (&relative, true)] {
        let restored = decode_lengths(encoded, &base, is_relative)?;
        if restored != (data_lengths.clone(), ref_lengths.clone()) {
            return Err("length array round trip differs".into());
        }
    }
    let (encoded, is_relative) = if absolute.len() <= relative.len() {
        (&absolute, false)
    } else {
        (&relative, true)
    };
    let (restored_data, restored_refs) = decode_lengths(encoded, &base, is_relative)?;
    let data_book = restore_book(&restored_data)?;
    let ref_book = restore_book(&restored_refs)?;
    // Verify actual canonical recoding, not just a Shannon entropy bound.
    if let Some(book) = ref_book {
        let (payload, bits) = book.encode(&reader.graph.children)?;
        if bits != ref_bits {
            return Err("reference cost differs".into());
        }
        let mut at = 0;
        book.decode(&payload, bits, reader.graph.children.len() as u64, |id| {
            if reader.graph.children[at] != id {
                return Err("recoded reference differs".into());
            }
            at += 1;
            Ok(())
        })?;
    }
    let mut data_bytes = 0;
    let mut observed_data_bits = 0;
    let data_encoding = data_book.as_ref().map(Codebook::encoding);
    for i in 0..reader.index.len() {
        let block = reader.block(i)?;
        let mut ids = Vec::with_capacity(block.totals.symbols as usize);
        block
            .book
            .decode(&block.payload, block.bits, block.totals.symbols, |id| {
                ids.push(id);
                Ok(())
            })?;
        let book = data_book.as_ref().ok_or("missing projected data table")?;
        let (payload, bits) = data_encoding
            .as_ref()
            .ok_or("missing projected encoding table")?
            .encode(&ids)?;
        add(&mut observed_data_bits, bits)?;
        add(&mut data_bytes, bits.div_ceil(8))?;
        let mut at = 0;
        book.decode(&payload, bits, ids.len() as u64, |id| {
            if ids[at] != id {
                return Err("recoded data differs".into());
            }
            at += 1;
            Ok(())
        })?;
    }
    if observed_data_bits != data_bits {
        return Err("projected data cost differs".into());
    }
    let old = reader
        .stats
        .reference_bits
        .div_ceil(8)
        .checked_add(old_data_bytes)
        .ok_or("projection size overflow")?;
    let new = ref_bits
        .div_ceil(8)
        .checked_add(data_bytes)
        .and_then(|v| v.checked_add(absolute.len().min(relative.len()) as u64))
        .and_then(|v| v.checked_add(9))
        .ok_or("projection size overflow")?;
    let projected = reader
        .stats
        .archive_bytes
        .checked_sub(old)
        .and_then(|v| v.checked_add(new))
        .ok_or("projection size overflow")?;
    Ok(TableProjection {
        archive: reader.stats.archive_bytes,
        projected,
        nodes: n,
        zero_data: data
            .iter()
            .zip(&refs)
            .filter(|(d, r)| **d == 0 && **r != 0)
            .count(),
        old_data_bits: reader.stats.payload_bits,
        data_bits,
        old_ref_bits: reader.stats.reference_bits,
        ref_bits,
        old_data_bytes,
        data_bytes,
        absolute: absolute.len(),
        relative: relative.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::super::{prepare, RuleLengths, State, Totals, Writer};
    use super::*;
    use crate::huffman_dictionary::{byte_model, Dictionary};
    use std::io::Cursor;
    #[test]
    fn projected_length_metadata_preserves_zero_lengths_deep_codes_and_padding() {
        let base: Vec<_> = (0..2003).map(|i| (i % 56 + 1) as u8).collect();
        let data: Vec<_> = base
            .iter()
            .enumerate()
            .map(|(i, &b)| if i % 3 == 0 { 0 } else { b })
            .collect();
        let refs: Vec<_> = base
            .iter()
            .enumerate()
            .map(|(i, &b)| {
                if i % 5 == 0 {
                    56
                } else {
                    b.saturating_sub(1).max(1)
                }
            })
            .collect();
        for mode in [false, true] {
            let bytes = encode_lengths(&base, &data, &refs, mode).unwrap();
            assert_eq!(
                decode_lengths(&bytes, &base, mode).unwrap(),
                (data.clone(), refs.clone())
            );
            for end in 0..bytes.len() {
                assert!(decode_lengths(&bytes[..end], &base, mode).is_err());
            }
            let mut extra = bytes;
            extra.push(0);
            assert!(decode_lengths(&extra, &base, mode).is_err());
        }
        assert!(decode_lengths(&[0], &[1], false).is_err());
        assert_eq!(
            decode_lengths(&[], &[], false).unwrap(),
            (Vec::new(), Vec::new())
        );
    }
    #[test]
    fn split_table_projection_verifies_all_symbols_helpers_and_empty_streams() {
        let dictionary =
            Dictionary::new(byte_model(), vec![(97, 98), (257, 99), (258, 256)], false).unwrap();
        for roots in [vec![], vec![256], vec![259; 700]] {
            let mut counts = vec![0; 260];
            let mut totals = Totals::default();
            for &id in &roots {
                counts[id as usize] += 1;
                super::super::totals_add(&mut totals, dictionary.expansions[id as usize]).unwrap();
            }
            let state = State::new(&dictionary, counts).unwrap();
            let prepared = prepare(&state, &dictionary, RuleLengths::Auto).unwrap();
            let mut output = Cursor::new(Vec::new());
            let mut writer = Writer::new(&mut output, prepared, &dictionary, 17).unwrap();
            for id in roots {
                writer.push(id).unwrap();
            }
            writer.finish(totals).unwrap();
            let bytes = output.into_inner();
            let mut reader = Reader::new(Cursor::new(bytes.clone())).unwrap();
            let result = project(&mut reader).unwrap();
            assert_eq!(result.archive, bytes.len() as u64);
            assert!(result.data_bits <= result.old_data_bits);
            assert!(result.ref_bits <= result.old_ref_bits);
            let mut raw = Vec::new();
            reader.decode_to(&mut raw, false).unwrap();
            assert_eq!(raw, b"abc".repeat(700 * usize::from(totals.bytes != 0)));
        }
    }
}
