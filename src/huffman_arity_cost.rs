//! Research pruning costs for the already implemented canonical arity section.
//! Freeze lengths between refits; maintain the histogram and section overhead
//! exactly under those lengths. Previously unseen values use a one-count
//! Shannon length until refitting, plus their full table-entry cost.
use super::{var_bytes, Codebook, Proposal, RuleLengths, State};
use std::collections::BTreeMap;

pub(super) struct Costs {
    histogram: BTreeMap<u64, u64>,
    lengths: BTreeMap<u64, u8>,
    bits: u128,
    unseen_length: u8,
    huffman: bool,
}
pub(super) struct Change {
    deltas: BTreeMap<u64, i128>,
    bits: u128,
    pub delta: f64,
}
impl Costs {
    pub fn new(state: &State, coding: RuleLengths) -> Result<Self, String> {
        let mut histogram = BTreeMap::<u64, u64>::new();
        for id in 0..state.data.len() {
            if state.data[id] != 0 || state.refs[id] != 0 {
                *histogram.entry(state.arities[id]).or_default() += 1;
            }
        }
        let mut result = Self {
            histogram,
            lengths: BTreeMap::new(),
            bits: 0,
            unseen_length: 1,
            huffman: false,
        };
        if coding == RuleLengths::Varint || result.histogram.is_empty() {
            return Ok(result);
        }
        let values: Vec<_> = result.histogram.keys().copied().collect();
        let counts: Vec<_> = result.histogram.values().copied().collect();
        let leaves: u64 = counts.iter().sum();
        result.unseen_length = (64 - leaves.leading_zeros()).clamp(1, 56) as u8;
        let book = Codebook::from_counts(&counts)?;
        for (id, length) in book.lengths() {
            result.lengths.insert(values[id as usize], length);
            result.bits += u128::from(counts[id as usize]) * u128::from(length);
        }
        let table_bytes = var_bytes(values.len() as u64) as u128
            + values
                .iter()
                .map(|&v| var_bytes(v) as u128 + 1)
                .sum::<u128>()
            + var_bytes(u64::try_from(result.bits).map_err(|_| "arity bits exceed u64")?) as u128;
        let raw_bytes: u128 = result
            .histogram
            .iter()
            .map(|(&v, &n)| var_bytes(v) as u128 * u128::from(n))
            .sum();
        result.huffman = table_bytes + result.bits.div_ceil(8) < raw_bytes;
        Ok(result)
    }
    fn length(&self, arity: u64) -> u8 {
        self.lengths
            .get(&arity)
            .copied()
            .unwrap_or(self.unseen_length)
    }
    pub fn change(&self, state: &State, proposal: &Proposal) -> Result<Change, String> {
        let mut deltas = BTreeMap::<u64, i128>::new();
        *deltas.entry(state.arities[proposal.id]).or_default() -= 1;
        for &(parent, count) in &proposal.parents {
            let old = state.arities[parent as usize];
            let new = old
                .checked_add(
                    count
                        .checked_mul(state.arities[proposal.id] - 1)
                        .ok_or("arity estimate overflow")?,
                )
                .ok_or("arity estimate overflow")?;
            *deltas.entry(old).or_default() -= 1;
            *deltas.entry(new).or_default() += 1;
        }
        let mut bit_delta = 0i128;
        let mut table_delta = 0i128;
        let mut used = self.histogram.len() as i128;
        for (&arity, &delta) in &deltas {
            let old = i128::from(self.histogram.get(&arity).copied().unwrap_or(0));
            let new = old.checked_add(delta).ok_or("arity histogram overflow")?;
            if !(0..=i128::from(u64::MAX)).contains(&new) {
                return Err("arity histogram under/overflow".into());
            }
            if self.huffman {
                bit_delta += delta * i128::from(self.length(arity));
                let presence = i128::from(new != 0) - i128::from(old != 0);
                table_delta += presence * (var_bytes(arity) as i128 + 1) * 8;
                used += presence;
            } else {
                bit_delta += delta * var_bytes(arity) as i128 * 8;
            }
        }
        let bits = if self.huffman {
            let new = i128::try_from(self.bits)
                .map_err(|_| "arity estimate exceeds i128")?
                .checked_add(bit_delta)
                .ok_or("arity estimate overflow")?;
            u128::try_from(new).map_err(|_| "negative arity bit estimate")?
        } else {
            0
        };
        let delta = if self.huffman {
            let bits64 = u64::try_from(bits).map_err(|_| "arity estimate exceeds u64")?;
            let old64 = u64::try_from(self.bits).map_err(|_| "arity estimate exceeds u64")?;
            let used64 = u64::try_from(used).map_err(|_| "negative arity alphabet size")?;
            table_delta
                + (var_bytes(used64) as i128 - var_bytes(self.histogram.len() as u64) as i128) * 8
                + (var_bytes(bits64) as i128 - var_bytes(old64) as i128) * 8
                + (bits.div_ceil(8) as i128 - self.bits.div_ceil(8) as i128) * 8
        } else {
            bit_delta
        };
        Ok(Change {
            deltas,
            bits,
            delta: delta as f64,
        })
    }
    pub fn apply(&mut self, change: Change) -> Result<(), String> {
        for (arity, delta) in change.deltas {
            let old = i128::from(self.histogram.get(&arity).copied().unwrap_or(0));
            let count = u64::try_from(old + delta).map_err(|_| "invalid updated arity count")?;
            if count == 0 {
                self.histogram.remove(&arity);
            } else {
                self.histogram.insert(arity, count);
            }
        }
        self.bits = change.bits;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        huffman_dictionary::{byte_model, Dictionary},
        shared_huffman::{prepare, Options},
    };

    fn state(rules: Vec<(u32, u32)>, roots: &[u32]) -> (Dictionary, State) {
        let dictionary = Dictionary::new(byte_model(), rules, false).unwrap();
        let mut data = vec![0; dictionary.expansions.len()];
        for &id in roots {
            data[id as usize] += 1;
        }
        let state = State::new(&dictionary, data).unwrap();
        (dictionary, state)
    }
    #[test]
    fn fixed_arity_cost_matches_actual_descriptor_bytes_when_lengths_stay_equal() {
        let roots: Vec<_> = (257..457).chain([256]).collect();
        let (dictionary, mut state) = state(vec![(97, 98); 200], &roots);
        let mut costs = Costs::new(&state, RuleLengths::Auto).unwrap();
        assert!(costs.huffman);
        let mut nonzero = 0;
        for id in 257..274 {
            let before = prepare(&state, &dictionary, RuleLengths::Auto)
                .unwrap()
                .stats
                .descriptor_bytes;
            let p = state.proposal(id, &Options::default()).unwrap().unwrap();
            let change = costs.change(&state, &p).unwrap();
            let predicted = change.delta;
            costs.apply(change).unwrap();
            state.apply(p).unwrap();
            let after = prepare(&state, &dictionary, RuleLengths::Auto)
                .unwrap()
                .stats
                .descriptor_bytes;
            assert_eq!(predicted, (after as i64 - before as i64) as f64 * 8.0);
            nonzero += usize::from(predicted != 0.0);
            assert_eq!(
                costs.histogram,
                Costs::new(&state, RuleLengths::Auto).unwrap().histogram
            );
        }
        assert!(
            nonzero >= 2,
            "exercise actual byte-padding boundary crossings"
        );
    }
    #[test]
    fn new_and_disappearing_arities_multiple_parents_and_refits_match_full_recount() {
        let mut rules = vec![(97, 98)];
        for i in 0..160 {
            rules.push(if i % 3 == 0 { (257, 257) } else { (257, 99) });
        }
        let roots: Vec<_> = (258..418).chain([256]).collect();
        let (_, mut state) = state(rules, &roots);
        let mut costs = Costs::new(&state, RuleLengths::Auto).unwrap();
        assert!(costs.huffman);
        // Independent full-section objective under the current frozen lengths.
        let size = |c: &Costs| -> i128 {
            if !c.huffman {
                return c
                    .histogram
                    .iter()
                    .map(|(&a, &n)| var_bytes(a) as i128 * i128::from(n) * 8)
                    .sum();
            }
            let bits: u128 = c
                .histogram
                .iter()
                .map(|(&a, &n)| u128::from(n) * u128::from(c.length(a)))
                .sum();
            (var_bytes(c.histogram.len() as u64) as i128
                + c.histogram
                    .keys()
                    .map(|&a| var_bytes(a) as i128 + 1)
                    .sum::<i128>()
                + var_bytes(u64::try_from(bits).unwrap()) as i128
                + bits.div_ceil(8) as i128)
                * 8
        };
        for id in 257..418 {
            let Some(p) = state.proposal(id, &Options::default()).unwrap() else {
                continue;
            };
            let before = size(&costs);
            let change = costs.change(&state, &p).unwrap();
            let predicted = change.delta;
            costs.apply(change).unwrap();
            state.apply(p).unwrap();
            let rebuilt = Costs::new(&state, RuleLengths::Auto).unwrap();
            assert_eq!(costs.histogram, rebuilt.histogram);
            assert_eq!(predicted, (size(&costs) - before) as f64);
            if id % 11 == 0 {
                costs = rebuilt;
            }
        }
        let (_, small) = self::state(vec![(97, 98)], &[257, 256]);
        assert!(!Costs::new(&small, RuleLengths::Auto).unwrap().huffman);
        let c = Costs::new(&small, RuleLengths::Varint).unwrap();
        let p = small.proposal(257, &Options::default()).unwrap().unwrap();
        assert_eq!(c.change(&small, &p).unwrap().delta, p.arity_delta);
    }
}
