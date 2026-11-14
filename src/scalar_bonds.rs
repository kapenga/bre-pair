//! Frozen training-only Unicode-scalar NPMI for the forward-loss experiment.
//! BRBONDS1 counts and Q24 rounding match the C# tokenizer workbench.
use crate::pair_map::PairMap;
use std::io::Write;

pub const SCALE: u32 = 1 << 24;
const SCALARS: usize = 0x110000;

pub struct Statistics {
    pairs: PairMap<u64>,
    left: Vec<u64>,
    right: Vec<u64>,
    pub observations: u64,
}

pub struct Model {
    weights: PairMap<u32>,
}

/// Invalid UTF-8 breaks scalar adjacency. Valid neighboring runs are kept;
/// replacement characters are never invented. Offsets refer to original bytes.
fn visit(
    field: &[u8],
    mut scalar: impl FnMut(usize, Option<char>) -> Result<(), String>,
) -> Result<(), String> {
    let mut at = 0;
    while at < field.len() {
        let (text, invalid) = match std::str::from_utf8(&field[at..]) {
            Ok(text) => (text, 0),
            Err(error) => (
                std::str::from_utf8(&field[at..at + error.valid_up_to()]).unwrap(),
                error
                    .error_len()
                    .unwrap_or(field.len() - at - error.valid_up_to()),
            ),
        };
        for (offset, value) in text.char_indices() {
            scalar(at + offset, Some(value))?;
        }
        at += text.len();
        if invalid != 0 {
            scalar(at, None)?;
            at += invalid;
        }
    }
    Ok(())
}

impl Statistics {
    pub fn count(fields: &[Vec<u8>]) -> Result<Self, String> {
        let mut result = Self {
            pairs: PairMap::default(),
            left: vec![0; SCALARS],
            right: vec![0; SCALARS],
            observations: 0,
        };
        for field in fields {
            let mut previous = None;
            visit(field, |_, value| {
                if let (Some(a), Some(b)) = (previous, value) {
                    let (a, b) = (a as u32, b as u32);
                    let count = result.pairs.entry((a, b)).or_default();
                    *count = count.checked_add(1).ok_or("scalar pair count overflow")?;
                    // Marginals cannot exceed the checked total event count.
                    result.observations = result
                        .observations
                        .checked_add(1)
                        .ok_or("scalar adjacency total overflow")?;
                    result.left[a as usize] += 1;
                    result.right[b as usize] += 1;
                }
                previous = value;
                Ok(())
            })?;
        }
        Ok(result)
    }

    pub fn pair_count(&self) -> usize {
        self.pairs.len()
    }

    pub fn write_counts(&self, mut output: impl Write) -> Result<(), String> {
        let mut write = || -> std::io::Result<()> {
            output.write_all(b"BRBONDS1")?;
            output.write_all(&(self.pairs.len() as u64).to_le_bytes())?;
            // Stable serialization, independently loadable by CharacterBonds.
            let mut pairs: Vec<_> = self.pairs.iter().collect();
            pairs.sort_unstable_by_key(|&(pair, _)| *pair);
            for (&(a, b), &count) in pairs {
                output.write_all(&((u64::from(a) << 32) | u64::from(b)).to_le_bytes())?;
                output.write_all(&count.to_le_bytes())?;
            }
            Ok(())
        };
        write().map_err(|error| error.to_string())
    }

    pub fn freeze(self) -> Model {
        let mut weights = PairMap::with_capacity_and_hasher(self.pairs.len(), Default::default());
        for ((a, b), count) in self.pairs {
            let probability = count as f64 / self.observations as f64;
            let strength = if probability == 1.0 {
                1.0
            } else {
                let pmi = (count as f64 * self.observations as f64
                    / (self.left[a as usize] as f64 * self.right[b as usize] as f64))
                    .ln();
                (pmi / -probability.ln()).clamp(0.0, 1.0)
            };
            weights.insert(
                (a, b),
                (strength * f64::from(SCALE)).round_ties_even() as u32,
            );
        }
        Model { weights }
    }
}

impl Model {
    /// Incoming scores: starts/EOF cost zero; valid UTF-8 interiors cost one.
    pub fn field_weights(&self, field: &[u8]) -> Result<Vec<f32>, String> {
        let mut result = vec![0.0; field.len() + 1];
        let mut previous = None;
        visit(field, |at, value| {
            if let Some(value) = value {
                if let Some(a) = previous {
                    result[at] = self
                        .weights
                        .get(&(a as u32, value as u32))
                        .copied()
                        .unwrap_or(0) as f32
                        / SCALE as f32;
                }
                result[at + 1..at + value.len_utf8()].fill(1.0);
            }
            previous = value;
            Ok(())
        })?;
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_marginals_include_overlaps_and_reset_at_invalid_bytes_and_fields() {
        let fields = vec![
            b"aaaaa".to_vec(),
            b"ab".to_vec(),
            b"b\xffa".to_vec(),
            vec![],
        ];
        let stats = Statistics::count(&fields).unwrap();
        assert_eq!(stats.observations, 5);
        assert_eq!(stats.pairs.get(&(97, 97)), Some(&4));
        assert_eq!(stats.left[97], 5);
        assert_eq!(stats.right[97], 4);
        assert!(!stats.pairs.contains_key(&(98, 97)));
        let mut bytes = Vec::new();
        stats.write_counts(&mut bytes).unwrap();
        assert_eq!(&bytes[..8], b"BRBONDS1");
        assert_eq!(u64::from_le_bytes(bytes[8..16].try_into().unwrap()), 2);
        let model = stats.freeze();
        assert_eq!(model.field_weights(b"b\xffa").unwrap(), vec![0.0; 4]);
        assert_eq!(model.field_weights(b"").unwrap(), [0.0]);
        assert_eq!(model.field_weights(b"unseen").unwrap(), vec![0.0; 7]);
    }

    #[test]
    fn clipped_npmi_matches_independent_scalar_formula_and_utf8_cuts() {
        let fields = ["甲乙甲乙", "甲丙", "😀甲", "甲甲"].map(|s| s.as_bytes().to_vec());
        let stats = Statistics::count(&fields).unwrap();
        assert_eq!(stats.observations, 6);
        let model = stats.freeze();
        let scores = model.field_weights("甲乙".as_bytes()).unwrap();
        // count=2, total=6, outgoing(甲)=4, incoming(乙)=2.
        let expected = ((1.5f64.ln() / 3f64.ln()) * f64::from(SCALE)).round_ties_even() as u32;
        assert_eq!(
            scores,
            [0.0, 1.0, 1.0, expected as f32 / SCALE as f32, 1.0, 1.0, 0.0]
        );
        assert_eq!(
            model.field_weights("😀".as_bytes()).unwrap(),
            [0.0, 1.0, 1.0, 1.0, 0.0]
        );
        let constant = Statistics::count(&[b"aaaaa".to_vec()]).unwrap().freeze();
        assert_eq!(
            constant.field_weights(b"aaa").unwrap(),
            [0.0, 1.0, 1.0, 0.0]
        );
    }
}
