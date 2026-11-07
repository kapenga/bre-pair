//! Read-only run-coding feasibility for the canonical-order arity sequence.
//! Alternative descriptors are serialized and round-tripped but no archive
//! format is added. Literal IDs and all non-descriptor bytes stay charged.
use super::arity_runs::{decode_runs, encode_runs, runs, scalar};
use super::{error, put_var, Reader, HEADER, NONE};
use std::{
    collections::BTreeMap,
    fs::File,
    io::{BufReader, Read, Seek, SeekFrom},
    path::Path,
};

pub fn project_arities(source: &Path) -> Result<String, String> {
    let mut reader = Reader::new(BufReader::new(File::open(source).map_err(error)?))?;
    let mut arities = Vec::with_capacity(reader.stats.leaves);
    let mut literals = Vec::new();
    let mut raw = Vec::new();
    let mut histogram = BTreeMap::<u64, u64>::new();
    for id in 0..reader.stats.leaves {
        let arity = if reader.graph.terminal[id] == NONE {
            (reader.graph.offsets[id + 1] - reader.graph.offsets[id]) as u64
        } else {
            0
        };
        arities.push(arity);
        *histogram.entry(arity).or_default() += 1;
        put_var(&mut raw, arity);
        if arity == 0 {
            put_var(&mut literals, u64::from(reader.graph.terminal[id]));
            put_var(&mut raw, u64::from(reader.graph.terminal[id]));
        }
    }
    let original = if reader.stats.run_rule_lengths {
        super::arity_runs::compress(&raw, reader.stats.leaves as u64)?
    } else if reader.stats.huffman_rule_lengths {
        let mut v = scalar(&arities)?;
        v.extend(&literals);
        v
    } else {
        raw
    };
    let mut stored = vec![0; reader.stats.descriptor_bytes as usize];
    let at = (HEADER as u64) + reader.stats.dictionary_bytes
        - reader.stats.reference_bits.div_ceil(8)
        - reader.stats.descriptor_bytes;
    reader.input.seek(SeekFrom::Start(at)).map_err(error)?;
    reader.input.read_exact(&mut stored).map_err(error)?;
    if stored != original {
        return Err("source descriptor reconstruction differs".into());
    }
    let dominant = histogram
        .iter()
        .max_by_key(|(&value, &count)| (count, std::cmp::Reverse(value)))
        .map_or(0, |(&v, _)| v);
    let all = encode_runs(&arities, None)?;
    let gaps = encode_runs(&arities, Some(dominant))?;
    if decode_runs(&all, arities.len())? != arities || decode_runs(&gaps, arities.len())? != arities
    {
        return Err("arity run round trip differs".into());
    }
    let literal_bytes = literals.len() as u64;
    let all_bytes = all.len() as u64 + literal_bytes;
    let gap_bytes = gaps.len() as u64 + literal_bytes;
    let projected =
        reader.stats.archive_bytes - reader.stats.descriptor_bytes + all_bytes.min(gap_bytes);
    Ok(format!(concat!("{{\"scope\":\"descriptor projection only; no new archive format\",\"source_archive_bytes\":{},",
        "\"source_descriptor_bytes\":{},\"leaves\":{},\"runs\":{},\"dominant_arity\":{},\"dominant_count\":{},",
        "\"literal_bytes\":{},\"all_run_descriptor_bytes\":{},\"dominant_gap_descriptor_bytes\":{},",
        "\"projected_archive_bytes\":{},\"projected_saving_bytes\":{},\"source_descriptor_byte_identical\":true,",
        "\"alternative_descriptor_round_trips\":true,\"all_non_descriptor_bytes_retained\":true}}"),reader.stats.archive_bytes,
        reader.stats.descriptor_bytes,arities.len(),runs(&arities).0.len(),dominant,histogram.get(&dominant).copied().unwrap_or(0),literal_bytes,
        all_bytes,gap_bytes,projected,i128::from(reader.stats.archive_bytes)-i128::from(projected)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared_huffman::arity_runs::read_scalar;
    #[test]
    fn scalar_arrays_are_sparse_and_preserve_empty_and_u64_values() {
        for values in [vec![], vec![0; 1001], vec![0, 1, u64::MAX, 17, u64::MAX]] {
            let bytes = scalar(&values).unwrap();
            let mut input = bytes.as_slice();
            assert_eq!(read_scalar(&mut input, values.len()).unwrap(), values);
            assert!(input.is_empty());
        }
    }
    #[test]
    fn run_modes_preserve_zero_arities_empty_data_and_boundaries() {
        let mut mixed = vec![2; 300];
        mixed.extend([0, 3, 0, 4, 4, 2]);
        mixed.extend(vec![2; 150]);
        for values in [vec![], vec![2; 1001], vec![0; 1001], mixed] {
            for mode in [None, Some(0), Some(2)] {
                let bytes = encode_runs(&values, mode).unwrap();
                assert_eq!(decode_runs(&bytes, values.len()).unwrap(), values);
                for end in 0..bytes.len() {
                    assert!(decode_runs(&bytes[..end], values.len()).is_err());
                }
                let mut extra = bytes;
                extra.push(0);
                assert!(decode_runs(&extra, values.len()).is_err());
            }
        }
    }
}
