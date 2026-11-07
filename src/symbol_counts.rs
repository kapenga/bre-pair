//! Global symbol statistics, independent of which pairs remain in the registry.
//! Frequencies change at batch barriers; workers only mutate field starts and
//! raw self-overlap corrections in these shared dense arrays.
use std::sync::atomic::{AtomicU64, Ordering};

use crate::grammar::{Pair, Symbol, BASE_SYMBOLS, EOF};

pub(crate) struct SymbolCounts {
    pub uses: Vec<u64>,
    starts: Vec<AtomicU64>,
    overlaps: Vec<AtomicU64>,
    eligible: Vec<u8>,
    minimum: u64,
}

impl SymbolCounts {
    /// Empty state for the unfiltered specialization; no dense arrays allocate.
    pub fn disabled() -> Self {
        Self {
            uses: Vec::new(),
            starts: Vec::new(),
            overlaps: Vec::new(),
            eligible: Vec::new(),
            minimum: 0,
        }
    }

    pub fn new(minimum: u64, track_overlaps: bool) -> Self {
        Self {
            uses: vec![0; BASE_SYMBOLS],
            starts: (0..BASE_SYMBOLS).map(|_| AtomicU64::new(0)).collect(),
            overlaps: if track_overlaps {
                (0..BASE_SYMBOLS).map(|_| AtomicU64::new(0)).collect()
            } else {
                Vec::new()
            },
            minimum,
            eligible: vec![0; BASE_SYMBOLS],
        }
    }

    pub fn push_initial(&mut self, symbol: Symbol, first: bool) {
        self.uses[symbol as usize] += 1;
        if first {
            *self.starts[symbol as usize].get_mut() += 1;
        }
    }

    pub fn resize(&mut self, alphabet: usize) {
        self.uses.resize(alphabet, 0);
        self.eligible.resize(alphabet, 0);
        self.starts.resize_with(alphabet, || AtomicU64::new(0));
        if !self.overlaps.is_empty() {
            self.overlaps.resize_with(alphabet, || AtomicU64::new(0));
        }
    }

    pub fn refresh_eligibility(&mut self) {
        for symbol in 0..self.uses.len() {
            self.refresh_symbol(symbol);
        }
    }

    fn refresh_symbol(&mut self, symbol: usize) {
        let uses = self.uses[symbol];
        self.eligible[symbol] =
            u8::from(uses >= self.minimum) | (u8::from(uses / 2 >= self.minimum) << 1);
    }

    /// Non-overlapping self-pairs require twice as many symbol occurrences.
    /// This stronger bound must not be applied to distinct-symbol pairs.
    #[inline]
    pub fn allows(&self, pair: Pair) -> bool {
        if pair.0 == EOF {
            return false;
        }
        let common = self.eligible[pair.0 as usize] & self.eligible[pair.1 as usize];
        common & 1 != 0 && (pair.0 != pair.1 || common & 2 != 0)
    }

    pub fn replace(&mut self, pair: Pair, symbol: Symbol, count: u64) -> Result<(), String> {
        for child in [pair.0, pair.1] {
            self.uses[child as usize] = self.uses[child as usize]
                .checked_sub(count)
                .ok_or("symbol frequency underflow")?;
        }
        self.uses[symbol as usize] = self.uses[symbol as usize]
            .checked_add(count)
            .ok_or("symbol frequency overflow")?;
        for affected in [pair.0, pair.1, symbol] {
            self.refresh_symbol(affected as usize);
        }
        Ok(())
    }

    pub fn replace_start(&self, old: Symbol, new: Symbol) -> Result<(), String> {
        if old != new {
            change(&self.starts[old as usize], -1)?;
            change(&self.starts[new as usize], 1)?;
        }
        Ok(())
    }

    pub fn start_count(&self, symbol: Symbol) -> u64 {
        self.starts[symbol as usize].load(Ordering::Relaxed)
    }

    pub fn overlap_count(&self, symbol: Symbol) -> u64 {
        self.overlaps[symbol as usize].load(Ordering::Relaxed)
    }

    pub fn change_overlap(&self, symbol: Symbol, delta: i64) -> Result<(), String> {
        change(&self.overlaps[symbol as usize], delta)
    }

    pub fn clear_overlaps(&mut self) {
        for count in &mut self.overlaps {
            *count.get_mut() = 0;
        }
    }

    pub fn storage_bytes(&self) -> usize {
        (self.uses.capacity() + self.starts.capacity() + self.overlaps.capacity()) * 8
            + self.eligible.capacity()
    }
}

fn change(counter: &AtomicU64, delta: i64) -> Result<(), String> {
    counter
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |old| {
            if delta >= 0 {
                old.checked_add(delta as u64)
            } else {
                old.checked_sub(delta.unsigned_abs())
            }
        })
        .map(|_| ())
        .map_err(|_| "shared symbol counter overflow or underflow".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        batch::PairCounter,
        grammar::Grammar,
        registry::Registry,
        scoring::{Model, Statistics},
    };

    #[test]
    fn frequency_bounds_keep_distinct_pairs_and_use_full_width_counts() {
        let mut counts = SymbolCounts::new(8, true);
        counts.uses[1] = 8;
        counts.uses[2] = 8;
        counts.refresh_eligibility();
        assert!(counts.allows((1, 2)));
        assert!(!counts.allows((1, 1)));
        counts.uses[1] = 15;
        counts.refresh_eligibility();
        assert!(!counts.allows((1, 1)));
        counts.uses[1] = 16;
        counts.refresh_eligibility();
        assert!(counts.allows((1, 1)));
        counts.uses[2] = 7;
        counts.refresh_eligibility();
        assert!(!counts.allows((1, 2)));
        assert!(!counts.allows((2, 1)));
        counts.uses[1] = 1u64 << 35;
        counts.replace((1, 1), 2, 1u64 << 33).unwrap();
        assert_eq!(counts.uses[1], 1u64 << 34);
        assert_eq!(counts.uses[2], (1u64 << 33) + 7);
        counts.minimum = u64::MAX;
        counts.uses[1] = u64::MAX;
        counts.refresh_eligibility();
        assert!(!counts.allows((1, 1)));
        counts.uses[2] = u64::MAX;
        counts.refresh_eligibility();
        assert!(counts.allows((1, 2)));
        *counts.overlaps[1].get_mut() = u64::MAX;
        assert!(counts.change_overlap(1, 1).is_err());
        counts.change_overlap(1, -1).unwrap();
        assert_eq!(counts.overlap_count(1), u64::MAX - 1);
    }

    #[test]
    fn dense_statistics_match_all_scores_with_rare_pairs_and_absorbed_eof() {
        let mut grammar = Grammar::default();
        let ending = grammar.add((122, EOF)).unwrap();
        let whole = grammar.add((113, ending)).unwrap();
        let mut fields = vec![vec![97, 97, 97, 97, 97, 98, 122, EOF]; 8];
        fields.extend([
            vec![97, 98, 113, EOF],
            vec![97, 98, 114, EOF],
            vec![EOF],
            vec![whole],
            vec![97, ending],
            vec![98, 97, ending],
        ]);
        let mut counts = SymbolCounts::new(8, true);
        counts.resize(grammar.alphabet_size());
        let mut reference = PairCounter::new(true);
        for field in fields {
            for (index, symbol) in field.into_iter().enumerate() {
                counts.push_initial(symbol, index == 0);
                reference.push(symbol, &grammar);
            }
        }
        let mut registry = Registry::new(2).unwrap();
        counts.refresh_eligibility();
        let mut writer = registry.writer();
        let mut omitted = 0;
        for (&pair, &count) in &reference.counts {
            if counts.allows(pair) {
                writer.pair(pair, count as i64).unwrap();
            } else {
                omitted += 1;
            }
        }
        for (&symbol, &count) in &reference.self_overlaps {
            counts.change_overlap(symbol, count as i64).unwrap();
        }
        writer.finish().unwrap();
        assert!(omitted > 0);
        let frozen = registry.freeze();
        let actual = Statistics::from_symbols(&frozen, &counts, &grammar);
        let expected = Statistics::from_counter(&reference, grammar.alphabet_size());
        assert_eq!(actual.left, expected.left);
        assert_eq!(actual.right, expected.right);
        assert_eq!(actual.edges, expected.edges);
        for (&pair, &count) in &reference.counts {
            if !counts.allows(pair) {
                assert_eq!(frozen.get(pair), 0);
                continue;
            }
            assert_eq!(actual.observed(pair), expected.observed(pair));
            for model in [
                Model::Frequency,
                Model::Conditional,
                Model::Pmi,
                Model::WeightedPmi,
                Model::Likelihood,
            ] {
                for alpha in [0.0, 0.5, 1.0, 2.0, 4.0] {
                    assert_eq!(
                        actual.score_with_alpha(model, pair, count, alpha).to_bits(),
                        expected
                            .score_with_alpha(model, pair, count, alpha)
                            .to_bits()
                    );
                }
            }
        }
    }
}
