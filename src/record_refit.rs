//! Training-only fixed-grammar reparse/refit for the prepared-record experiment.
use crate::{
    codec::{Frozen, MatchScratch},
    huffman::Codebook,
    huffman_archive::Totals,
    huffman_dictionary::{byte_model, Dictionary},
    shared_huffman::{RuleLengths, StaticModel},
};
use std::{
    fs::File,
    io::{BufRead, BufReader, Read},
    path::Path,
    time::Instant,
};

pub struct FitPoint {
    pub iteration: usize,
    pub payload_bits: u128,
    pub payload_bytes: u64,
    pub fields: u64,
    pub symbols: u64,
    pub legacy_model_bytes: usize,
    pub compact_model_bytes: usize,
    pub seconds: f64,
}
pub struct Refit {
    pub model: Frozen,
    pub points: Vec<FitPoint>,
}

/// The only corpus parameter is the training file. Holdout never enters fitting.
/// Iterations are fixed in advance; candidates and fallback bytes stay intact.
pub fn training(
    source: &Frozen,
    input: &Path,
    refits: usize,
    shared_weights: bool,
) -> Result<Refit, String> {
    if refits > 16
        || source.dictionary.model.to_bytes() != byte_model().to_bytes()
        || source.book.symbols.len() != source.dictionary.expansions.len()
    {
        return Err("record refit requires an all-candidate byte model and 0..16 refits".into());
    }
    let mut model = Frozen::from_bytes(&source.to_bytes()?)?;
    let mut points = Vec::new();
    for iteration in 0..=refits {
        let start = Instant::now();
        let matcher = model.prepare()?;
        let mut scratch = MatchScratch::default();
        let mut bytes = Vec::new();
        let mut terminals = Vec::new();
        let mut reader = BufReader::new(File::open(input).map_err(|e| e.to_string())?);
        let mut counts = vec![0u64; model.dictionary.expansions.len()];
        let mut lengths = vec![0u8; counts.len()];
        for (id, len) in model.book.lengths() {
            lengths[id as usize] = len;
        }
        let mut actual = Totals::default();
        let mut payload_bits = 0u128;
        let mut payload_bytes = 0u64;
        loop {
            bytes.clear();
            reader
                .by_ref()
                .take((1 << 20) + 1)
                .read_until(b'\n', &mut bytes)
                .map_err(|e| e.to_string())?;
            if bytes.is_empty() {
                break;
            }
            if bytes.len() > 1 << 20 {
                return Err("record exceeds refit bound".into());
            }
            terminals.clear();
            terminals.extend(bytes.iter().map(|&b| u32::from(b)));
            terminals.push(256);
            let symbols = matcher.segment_with(&terminals, &mut scratch)?;
            let mut field_bits = 0u64;
            for &id in symbols {
                counts[id as usize] = counts[id as usize]
                    .checked_add(1)
                    .ok_or("training count overflow")?;
                field_bits = field_bits
                    .checked_add(u64::from(lengths[id as usize]))
                    .ok_or("field bit overflow")?;
                actual.add_symbol(&model.dictionary, id)?;
            }
            payload_bits += u128::from(field_bits);
            payload_bytes = payload_bytes
                .checked_add(field_bits.div_ceil(8))
                .ok_or("payload overflow")?;
        }
        if (actual.bytes, actual.fields, actual.tokens)
            != (
                source.training.bytes,
                source.training.fields,
                source.training.tokens,
            )
        {
            return Err("refit training bytes/fields/terminals differ from frozen model".into());
        }
        model.training = actual;
        let point = FitPoint {
            iteration,
            payload_bits,
            payload_bytes,
            fields: actual.fields,
            symbols: actual.symbols,
            legacy_model_bytes: model.to_bytes()?.len(),
            compact_model_bytes: StaticModel::encode_frozen(&model, RuleLengths::Auto)?.len(),
            seconds: start.elapsed().as_secs_f64(),
        };
        eprintln!("training refit {iteration}: {} fields, {payload_bits} bits, {} padded payload bytes, {:.3}s",actual.fields,payload_bytes,point.seconds);
        points.push(point);
        if iteration == refits {
            break;
        }
        if shared_weights {
            for &(a, b) in &model.dictionary.rules {
                for id in [a, b] {
                    counts[id as usize] = counts[id as usize]
                        .checked_add(1)
                        .ok_or("model reference overflow")?;
                }
            }
        }
        for n in &mut counts {
            *n = (*n).max(1);
        }
        model = Frozen {
            dictionary: Dictionary::new(
                model.dictionary.model.clone(),
                model.dictionary.rules.clone(),
                true,
            )?,
            book: Codebook::from_counts(&counts)?,
            training: actual,
        };
    }
    Ok(Refit { model, points })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        fs,
        sync::atomic::{AtomicU64, Ordering},
    };
    static NEXT: AtomicU64 = AtomicU64::new(0);
    #[test]
    fn training_refit_keeps_grammar_and_full_byte_fallback_and_checks_input_totals() {
        let dictionary = Dictionary::new(byte_model(), vec![(97, 98), (257, 99)], true).unwrap();
        let mut counts = vec![1; 259];
        counts[257] = 50;
        counts[258] = 1;
        let raw = b"abcabc\nabc\n";
        let model = Frozen {
            dictionary,
            book: Codebook::from_counts(&counts).unwrap(),
            training: Totals {
                bytes: raw.len() as u64,
                fields: 2,
                tokens: raw.len() as u64 + 2,
                symbols: 9,
            },
        };
        let path = std::env::temp_dir().join(format!(
            "record-refit-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::write(&path, raw).unwrap();
        for shared in [false, true] {
            let result = training(&model, &path, 2, shared).unwrap();
            assert_eq!(result.points.len(), 3);
            assert_eq!(result.model.dictionary.rules, model.dictionary.rules);
            assert_eq!(result.model.book.symbols.len(), 259);
            assert_eq!(result.model.training.fields, 2);
            assert_eq!(result.model.training.bytes, raw.len() as u64);
            let first = result.model.to_bytes().unwrap();
            assert_eq!(
                training(&model, &path, 2, shared)
                    .unwrap()
                    .model
                    .to_bytes()
                    .unwrap(),
                first
            );
        }
        fs::write(&path, b"other\n").unwrap();
        assert!(training(&model, &path, 2, true).is_err());
        fs::remove_file(path).unwrap();
    }
}
