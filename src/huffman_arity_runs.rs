//! Canonical Huffman run coding of rule arities; metadata initialization only.
use super::{add, get_var, put_var, Codebook};
use std::collections::BTreeMap;

pub(super) fn scalar(values: &[u64]) -> Result<Vec<u8>, String> {
    let mut histogram = BTreeMap::<u64, u64>::new();
    for &v in values {
        *histogram.entry(v).or_default() += 1;
    }
    let mut out = Vec::new();
    put_var(&mut out, histogram.len() as u64);
    if values.is_empty() {
        return Ok(out);
    }
    let symbols: Vec<_> = histogram.keys().copied().collect();
    let counts: Vec<_> = histogram.values().copied().collect();
    let book = Codebook::from_counts(&counts)?;
    let mut lengths = vec![0; symbols.len()];
    let mut bits = 0;
    for (id, len) in book.lengths() {
        lengths[id as usize] = len;
        add(
            &mut bits,
            counts[id as usize]
                .checked_mul(u64::from(len))
                .ok_or("arity stream overflow")?,
        )?;
    }
    for (&v, &len) in symbols.iter().zip(&lengths) {
        put_var(&mut out, v);
        out.push(len);
    }
    put_var(&mut out, bits);
    out.extend(book.encoding().encode_iter(
        values.iter().map(|v| {
            symbols
                .binary_search(v)
                .map(|id| id as u32)
                .map_err(|_| "missing arity value".into())
        }),
        bits,
    )?);
    Ok(out)
}
pub(super) fn read_scalar(input: &mut &[u8], count: usize) -> Result<Vec<u64>, String> {
    let used = get_var(input)?;
    if count == 0 {
        return if used == 0 {
            Ok(Vec::new())
        } else {
            Err("unexpected nonempty arity table".into())
        };
    }
    if used == 0 || used > count as u64 || used > input.len() as u64 / 2 {
        return Err("invalid scalar alphabet size".into());
    }
    let mut values = Vec::new();
    let mut entries = Vec::new();
    for id in 0..used {
        let v = get_var(input)?;
        if values.last().is_some_and(|&last| v <= last) {
            return Err("invalid scalar value order".into());
        }
        let (&len, rest) = input.split_first().ok_or("truncated scalar table")?;
        *input = rest;
        values.push(v);
        entries.push((id as u32, len));
    }
    let book = Codebook::from_lengths(&entries)?;
    let bits = get_var(input)?;
    let size = usize::try_from(bits.div_ceil(8)).map_err(|_| "scalar payload too large")?;
    if size > input.len() {
        return Err("truncated scalar payload".into());
    }
    let (payload, rest) = input.split_at(size);
    *input = rest;
    if count as u64 > bits {
        return Err("scalar count exceeds coded bits".into());
    }
    let mut out = Vec::with_capacity(count);
    book.decode(payload, bits, count as u64, |id| {
        out.push(values[id as usize]);
        Ok(())
    })?;
    Ok(out)
}
pub(super) fn runs(values: &[u64]) -> (Vec<u64>, Vec<u64>) {
    let (mut types, mut lengths) = (Vec::new(), Vec::<u64>::new());
    for &v in values {
        if types.last() == Some(&v) {
            *lengths.last_mut().unwrap() += 1;
        } else {
            types.push(v);
            lengths.push(1);
        }
    }
    (types, lengths)
}
pub(super) fn encode_runs(values: &[u64], dominant: Option<u64>) -> Result<Vec<u8>, String> {
    let mut out = vec![u8::from(dominant.is_some())];
    if let Some(d) = dominant {
        put_var(&mut out, d);
        let (mut others, mut gaps) = (Vec::new(), vec![0u64]);
        for &v in values {
            if v == d {
                *gaps.last_mut().unwrap() += 1;
            } else {
                others.push(v);
                gaps.push(0);
            }
        }
        put_var(&mut out, others.len() as u64);
        out.extend(scalar(&gaps)?);
        out.extend(scalar(&others)?);
    } else {
        let (types, lengths) = runs(values);
        put_var(&mut out, types.len() as u64);
        out.extend(scalar(&types)?);
        out.extend(scalar(&lengths)?);
    }
    Ok(out)
}
fn read_runs(input: &mut &[u8], n: usize) -> Result<Vec<u64>, String> {
    let (&mode, rest) = input.split_first().ok_or("missing arity run mode")?;
    *input = rest;
    let mut out = Vec::with_capacity(n);
    let mut repeat = |v, count: u64| -> Result<(), String> {
        let count = usize::try_from(count).map_err(|_| "arity run too long")?;
        if count > n - out.len() {
            return Err("arity runs exceed leaf count".into());
        }
        out.resize(out.len() + count, v);
        Ok(())
    };
    match mode {
        0 => {
            let count = usize::try_from(get_var(input)?).map_err(|_| "too many arity runs")?;
            if count > n {
                return Err("too many arity runs".into());
            }
            let types = read_scalar(input, count)?;
            let lengths = read_scalar(input, count)?;
            for (&v, &count) in types.iter().zip(&lengths) {
                if count == 0 {
                    return Err("zero arity run".into());
                }
                repeat(v, count)?;
            }
        }
        1 => {
            let d = get_var(input)?;
            let count = usize::try_from(get_var(input)?).map_err(|_| "too many other arities")?;
            if count > n {
                return Err("too many other arities".into());
            }
            let gaps = read_scalar(input, count.checked_add(1).ok_or("gap count overflow")?)?;
            let others = read_scalar(input, count)?;
            for (i, &gap) in gaps.iter().enumerate() {
                repeat(d, gap)?;
                if let Some(&v) = others.get(i) {
                    if v == d {
                        return Err("dominant arity in other stream".into());
                    }
                    repeat(v, 1)?;
                }
            }
        }
        _ => return Err("unknown arity run mode".into()),
    }
    if out.len() != n {
        return Err("arity run size/trailing bytes differ".into());
    }
    Ok(out)
}

pub(super) fn decode_runs(mut input: &[u8], n: usize) -> Result<Vec<u64>, String> {
    let result = read_runs(&mut input, n)?;
    if !input.is_empty() {
        return Err("arity runs have trailing bytes".into());
    }
    Ok(result)
}

pub(super) fn compress(raw: &[u8], leaves: u64) -> Result<Vec<u8>, String> {
    let mut arities = Vec::new();
    let mut literals = Vec::new();
    let mut histogram = BTreeMap::<u64, u64>::new();
    super::descriptors::visit_raw(raw, leaves, |arity, literal| {
        arities.push(arity);
        *histogram.entry(arity).or_default() += 1;
        if let Some(id) = literal {
            put_var(&mut literals, id);
        }
        Ok(())
    })?;
    let dominant = histogram
        .iter()
        .max_by_key(|(&v, &n)| (n, std::cmp::Reverse(v)))
        .map_or(0, |(&v, _)| v);
    let all = encode_runs(&arities, None)?;
    let gaps = encode_runs(&arities, Some(dominant))?;
    let mut result = if all.len() <= gaps.len() { all } else { gaps };
    result.extend(literals);
    Ok(result)
}

pub(super) fn visit(
    mut bytes: &[u8],
    leaves: u64,
    max_arity: u64,
    mut emit: impl FnMut(u64, Option<u64>) -> Result<(), String>,
) -> Result<(), String> {
    let n = usize::try_from(leaves).map_err(|_| "arity count exceeds platform")?;
    let arities = read_runs(&mut bytes, n)?;
    for arity in arities {
        if arity == 1 || arity > max_arity {
            return Err("invalid run-coded arity".into());
        }
        let literal = if arity == 0 {
            Some(get_var(&mut bytes)?)
        } else {
            None
        };
        emit(arity, literal)?;
    }
    if !bytes.is_empty() {
        return Err("run-coded descriptors have trailing bytes".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn descriptor_literals_bounds_and_malformed_streams_are_checked() {
        let mut raw = Vec::new();
        let mut expected = Vec::new();
        for i in 0..600 {
            let arity = if i % 100 == 0 { 0 } else { 2 };
            put_var(&mut raw, arity);
            let literal = if arity == 0 {
                put_var(&mut raw, (i / 100) as u64);
                Some((i / 100) as u64)
            } else {
                None
            };
            expected.push((arity, literal));
        }
        let bytes = compress(&raw, 600).unwrap();
        let mut observed = Vec::new();
        visit(&bytes, 600, 2000, |a, l| {
            observed.push((a, l));
            Ok(())
        })
        .unwrap();
        assert_eq!(observed, expected);
        for end in 0..bytes.len() {
            assert!(visit(&bytes[..end], 600, 2000, |_, _| Ok(())).is_err());
        }
        let mut extra = bytes.clone();
        extra.push(0);
        assert!(visit(&extra, 600, 2000, |_, _| Ok(())).is_err());
        assert!(visit(&bytes, 600, 1, |_, _| Ok(())).is_err());
        assert!(visit(&bytes, 599, 2000, |_, _| Ok(())).is_err());
        let mut bad = bytes;
        bad[0] = 255;
        assert!(visit(&bad, 600, 2000, |_, _| Ok(())).is_err());
        let empty = compress(&[], 0).unwrap();
        visit(&empty, 0, 0, |_, _| Ok(())).unwrap();
    }
}
