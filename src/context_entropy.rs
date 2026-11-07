//! Offline byte-context entropy: count the complete corpus, then freeze scores.
//! This supplies a training signal, not an adaptive coder or decoder model.
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::hash::BuildHasherDefault;

use crate::grammar::EOF;
use crate::pair_map::PairHasher;

type NumericMap<V> = HashMap<u64, V, BuildHasherDefault<PairHasher>>;

#[derive(Clone, Copy, Debug)]
pub struct Options {
    pub order: usize,
    pub minimum_context_count: u64,
    /// Fail explicitly at this limit; never silently omit a context follower.
    pub maximum_ngrams: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            order: 3,
            minimum_context_count: 32,
            maximum_ngrams: 20_000_000,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Statistics {
    pub observed_ngrams: usize,
    pub retained_contexts: usize,
    pub contexts_by_order: [usize; 7],
}

pub struct Model {
    options: Options,
    weights: NumericMap<f32>,
    pub statistics: Statistics,
}

#[derive(Default)]
pub(crate) struct History {
    bytes: u64,
    length: usize,
}

fn context_key(history: &History, order: usize) -> u64 {
    let marker = 1u64 << (8 * order);
    marker | (history.bytes & (marker - 1))
}

impl History {
    pub(crate) fn push(&mut self, byte: u8, maximum: usize) {
        self.bytes = ((self.bytes << 8) | u64::from(byte)) & ((1u64 << (8 * maximum)) - 1);
        self.length = (self.length + 1).min(maximum);
    }
}

impl Model {
    pub fn train(fields: &[Vec<u8>], options: Options) -> Result<Self, String> {
        if options.order > 6 || options.minimum_context_count == 0 || options.maximum_ngrams == 0 {
            return Err(
                "context order must be 0..=6; support and n-gram limit must be positive".into(),
            );
        }
        let mut counts: NumericMap<u64> = NumericMap::default();
        for field in fields {
            let mut history = History::default();
            for symbol in field.iter().map(|&b| u32::from(b)).chain([EOF]) {
                for order in 0..=history.length {
                    let key = (context_key(&history, order) << 9) | u64::from(symbol);
                    let full = counts.len() >= options.maximum_ngrams;
                    match counts.entry(key) {
                        Entry::Occupied(mut entry) => {
                            *entry.get_mut() =
                                entry.get().checked_add(1).ok_or("context count overflow")?;
                        }
                        Entry::Vacant(entry) => {
                            if full {
                                return Err("context n-gram limit exceeded; increase --max-ngrams or reduce --order".into());
                            }
                            entry.insert(1);
                        }
                    }
                }
                if symbol != EOF {
                    history.push(symbol as u8, options.order);
                }
            }
        }
        let observed_ngrams = counts.len();
        // Consume the one statistics table. Sorted followers make entropy
        // summation independent of corpus order and hash iteration order.
        let mut entries: Vec<_> = counts.into_iter().collect();
        entries.sort_unstable_by_key(|&(key, _)| key);
        let mut weights = NumericMap::default();
        let mut contexts_by_order = [0usize; 7];
        let mut at = 0;
        while at < entries.len() {
            let context = entries[at].0 >> 9;
            let mut end = at;
            let mut total = 0u64;
            while end < entries.len() && entries[end].0 >> 9 == context {
                total = total
                    .checked_add(entries[end].1)
                    .ok_or("context total overflow")?;
                end += 1;
            }
            if context == 1 || total >= options.minimum_context_count {
                let entropy = entries[at..end]
                    .iter()
                    .map(|&(_, count)| {
                        let p = count as f64 / total as f64;
                        -p * p.log2()
                    })
                    .sum::<f64>();
                // Compute the requested signal in float32. These exact f32
                // differences can be summed reproducibly in binary units.
                let normalized = (entropy / 257f64.log2()).clamp(0.0, 1.0) as f32;
                weights.insert(context, 1.0f32 - normalized);
                let order = (63 - context.leading_zeros() as usize) / 8;
                contexts_by_order[order] += 1;
            }
            at = end;
        }
        // Empty corpora still have a defined order-zero fallback.
        if let Entry::Vacant(entry) = weights.entry(1) {
            entry.insert(1.0);
            contexts_by_order[0] += 1;
        }
        let statistics = Statistics {
            observed_ngrams,
            retained_contexts: weights.len(),
            contexts_by_order,
        };
        Ok(Self {
            options,
            weights,
            statistics,
        })
    }

    pub(crate) fn weight(&self, history: &History) -> (f32, usize) {
        for order in (0..=history.length.min(self.options.order)).rev() {
            if let Some(&weight) = self.weights.get(&context_key(history, order)) {
                return (weight, order);
            }
        }
        unreachable!("order-zero context is always retained")
    }

    /// Score incoming boundaries, including the legal final byte/EOF join.
    /// Field starts have no incoming join; no context crosses EOF.
    pub fn field_weights(&self, field: &[u8]) -> Vec<f32> {
        let mut result = Vec::with_capacity(field.len() + 1);
        let mut history = History::default();
        for (index, symbol) in field.iter().map(|&b| u32::from(b)).chain([EOF]).enumerate() {
            result.push(if index == 0 {
                0.0
            } else {
                self.weight(&history).0
            });
            if symbol != EOF {
                history.push(symbol as u8, self.options.order);
            }
        }
        result
    }

    pub(crate) fn order(&self) -> usize {
        self.options.order
    }
}

/// Float32 subtraction from one gives multiples of 2^-24 over [0,1]. Keep
/// all those bits when summing; this is not byte quantization. The stored
/// boundary buffer remains f32, and pair totals are independent of scheduling.
pub(crate) const WEIGHT_SCALE: u64 = 1 << 24;

#[inline]
pub(crate) fn weight_units(weight: f32) -> u32 {
    let scaled = weight * WEIGHT_SCALE as f32;
    debug_assert!((0.0..=1.0).contains(&weight));
    debug_assert_eq!(scaled, scaled.trunc());
    scaled as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn float32_entropy_differences_keep_all_bits_without_byte_quantization() {
        for i in 0..10001 {
            let weight = 1.0f32 - i as f32 / 10000.0;
            assert_eq!(weight_units(weight) as f32 / WEIGHT_SCALE as f32, weight);
        }
        let a = 1.0f32 - 0.25;
        let b = 1.0f32 - (0.25 + 1.0 / WEIGHT_SCALE as f32);
        assert_eq!(weight_units(a) - weight_units(b), 1);
        assert_ne!(a.to_bits(), b.to_bits());
    }
}
