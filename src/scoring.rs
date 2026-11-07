//! Experimental merge priorities. Replacement counts remain non-overlapping;
//! association statistics use all legal overlapping adjacency observations.
use std::collections::HashMap;
use std::str::FromStr;

use crate::batch::{Counts, PairCounter};
use crate::grammar::{Grammar, Pair, Symbol, EOF};
use crate::registry::Frozen;
use crate::symbol_counts::SymbolCounts;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Model {
    #[default]
    Frequency,
    Conditional,
    Pmi,
    WeightedPmi,
    Likelihood,
}

impl Model {
    pub fn name(self) -> &'static str {
        match self {
            Self::Frequency => "frequency",
            Self::Conditional => "conditional",
            Self::Pmi => "pmi",
            Self::WeightedPmi => "weighted-pmi",
            Self::Likelihood => "likelihood",
        }
    }
}

impl FromStr for Model {
    type Err = String;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "frequency" => Ok(Self::Frequency),
            "conditional" => Ok(Self::Conditional),
            "pmi" => Ok(Self::Pmi),
            "weighted-pmi" => Ok(Self::WeightedPmi),
            "likelihood" => Ok(Self::Likelihood),
            _ => {
                Err("score must be frequency, conditional, pmi, weighted-pmi, or likelihood".into())
            }
        }
    }
}

pub struct Statistics<'a> {
    source: Source<'a>,
    pub left: Vec<u64>,
    pub right: Vec<u64>,
    pub edges: u64,
}

enum Source<'a> {
    Local(&'a Counts, &'a HashMap<Symbol, u64>),
    Shared(&'a Frozen<'a>),
    Filtered(&'a Frozen<'a>, &'a SymbolCounts),
}

impl<'a> Statistics<'a> {
    pub fn from_counter(counter: &'a PairCounter, alphabet_size: usize) -> Self {
        assert!(
            counter.tracks_self_overlaps(),
            "association statistics require raw self-overlap tracking"
        );
        Self::from_counts(&counter.counts, &counter.self_overlaps, alphabet_size)
    }

    pub fn from_counts(
        pairs: &'a Counts,
        self_overlaps: &'a HashMap<Symbol, u64>,
        alphabet_size: usize,
    ) -> Self {
        let mut result = Self {
            source: Source::Local(pairs, self_overlaps),
            left: vec![0; alphabet_size],
            right: vec![0; alphabet_size],
            edges: 0,
        };
        // Marginals derive from the existing count table, without another text
        // scan or a duplicate pair table. Dense symbol IDs permit direct indexing.
        for (&pair, &count) in pairs {
            let observed = count
                + if pair.0 == pair.1 {
                    self_overlaps.get(&pair.0).copied().unwrap_or(0)
                } else {
                    0
                };
            result.left[pair.0 as usize] += observed;
            result.right[pair.1 as usize] += observed;
            result.edges += observed;
        }
        result
    }

    pub fn from_registry(registry: &'a Frozen<'a>, alphabet_size: usize) -> Self {
        Self::from_registry_with_visit(registry, alphabet_size, |_, _| {})
    }

    /// Exact marginals for complete EOF-terminated fields, even when impossible
    /// pairs have been omitted. Pair lookups are used only for retained pairs.
    pub(crate) fn from_symbols(
        registry: &'a Frozen<'a>,
        counts: &'a SymbolCounts,
        grammar: &Grammar,
    ) -> Self {
        let mut left = Vec::with_capacity(counts.uses.len());
        let mut right = Vec::with_capacity(counts.uses.len());
        let mut edges = 0;
        for (symbol, &uses) in counts.uses.iter().enumerate() {
            let symbol = symbol as Symbol;
            let outgoing = if grammar.ends_field(symbol) { 0 } else { uses };
            left.push(outgoing);
            right.push(uses - counts.start_count(symbol));
            edges += outgoing;
        }
        Self {
            source: Source::Filtered(registry, counts),
            left,
            right,
            edges,
        }
    }

    /// Accumulate all marginals and visit each positive registry entry in one
    /// table walk. Candidate filtering belongs in the visitor; rare pairs and
    /// raw self-overlaps must still contribute to the association statistics.
    /// Score collected candidates only after this returns, when marginals are
    /// complete. The visitor may see the reserved EOF-left overlap namespace.
    pub fn from_registry_with_visit(
        registry: &'a Frozen<'a>,
        alphabet_size: usize,
        mut visit: impl FnMut(Pair, u64),
    ) -> Self {
        let mut result = Self {
            source: Source::Shared(registry),
            left: vec![0; alphabet_size],
            right: vec![0; alphabet_size],
            edges: 0,
        };
        registry.visit(|pair, count| {
            let (left, right) = if pair.0 == EOF {
                (pair.1, pair.1)
            } else {
                pair
            };
            result.left[left as usize] += count;
            result.right[right as usize] += count;
            result.edges += count;
            visit(pair, count);
        });
        result
    }

    fn overlap(&self, symbol: Symbol) -> u64 {
        match self.source {
            Source::Local(_, overlaps) => overlaps.get(&symbol).copied().unwrap_or(0),
            Source::Shared(registry) => registry.get((EOF, symbol)),
            Source::Filtered(_, counts) => counts.overlap_count(symbol),
        }
    }

    pub fn observed(&self, pair: Pair) -> u64 {
        if pair.0 == EOF {
            return 0;
        }
        (match self.source {
            Source::Local(pairs, _) => pairs.get(&pair).copied().unwrap_or(0),
            Source::Shared(registry) | Source::Filtered(registry, _) => registry.get(pair),
        }) + if pair.0 == pair.1 {
            self.overlap(pair.0)
        } else {
            0
        }
    }

    pub fn score(&self, model: Model, pair: Pair, replacements: u64) -> f64 {
        if model == Model::Frequency {
            return replacements as f64;
        }
        let observed = (replacements
            + if pair.0 == pair.1 {
                self.overlap(pair.0)
            } else {
                0
            }) as f64;
        let left = self.left[pair.0 as usize] as f64;
        let right = self.right[pair.1 as usize] as f64;
        let total = self.edges as f64;
        let expected = left * right / total;
        match model {
            Model::Frequency => unreachable!(),
            Model::Conditional => observed / left,
            Model::Pmi => (observed / expected).log2(),
            Model::WeightedPmi => replacements as f64 * (observed / expected).log2(),
            Model::Likelihood => {
                // Signed G^2 for a 2x2 table of legal adjacency observations.
                // Sign prevents a strongly *avoided* pair from ranking highly.
                let cells = [
                    observed,
                    left - observed,
                    right - observed,
                    total - left - right + observed,
                ];
                let expected_cells = [
                    expected,
                    left * (total - right) / total,
                    (total - left) * right / total,
                    (total - left) * (total - right) / total,
                ];
                let g2 = 2.0
                    * cells
                        .iter()
                        .zip(expected_cells)
                        .map(|(&o, e)| {
                            if o > 0.0 && e > 0.0 {
                                o * (o / e).ln()
                            } else {
                                0.0
                            }
                        })
                        .sum::<f64>();
                (observed - expected).signum() * g2.max(0.0)
            }
        }
    }

    /// Tune support versus association; alpha=1 preserves the original path.
    pub fn score_with_alpha(&self, model: Model, pair: Pair, replacements: u64, alpha: f64) -> f64 {
        if model == Model::WeightedPmi && alpha != 1.0 {
            (replacements as f64).powf(alpha) * self.score(Model::Pmi, pair, replacements)
        } else {
            self.score(model, pair, replacements)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grammar::Grammar;
    use crate::packed::PackedSymbols;
    #[test]
    fn association_keeps_raw_self_overlap_distinct_from_replacements() {
        let mut buffer = PackedSymbols::new(9);
        for symbol in [97, 97, 97, 97, 97, 256, 98, 256] {
            buffer.push(symbol);
        }
        let counter = PairCounter::from_buffer(&buffer, &Grammar::default());
        let stats = Statistics::from_counter(&counter, 257);
        assert_eq!(stats.observed((97, 97)), 4);
        assert_eq!(stats.edges, 6);
        assert_eq!(stats.observed((256, 98)), 0);
        assert_eq!(stats.score(Model::Frequency, (97, 97), 2), 2.0);
        assert!((stats.score(Model::Conditional, (97, 97), 2) - 0.8).abs() < 1e-12);
    }
    #[test]
    fn likelihood_detects_independence_and_attraction_without_nan() {
        let overlaps = HashMap::new();
        let pairs = Counts::from_iter([((1, 2), 10), ((1, 3), 10), ((4, 2), 40), ((4, 3), 40)]);
        let independent = Statistics::from_counts(&pairs, &overlaps, 5);
        assert_eq!(independent.score(Model::Pmi, (1, 2), 10), 0.0);
        assert_eq!(independent.score(Model::Likelihood, (1, 2), 10), 0.0);
        let strong_pairs = Counts::from_iter([((1, 2), 20), ((4, 2), 30), ((4, 3), 50)]);
        let strong = Statistics::from_counts(&strong_pairs, &overlaps, 5);
        assert!(strong.score(Model::Likelihood, (1, 2), 20) > 0.0);
        assert!(strong.score(Model::Pmi, (1, 2), 20) > 0.0);
    }

    #[test]
    fn reused_counts_match_an_independent_raw_adjacency_scan() {
        let mut grammar = Grammar::default();
        let ending = grammar.add((97, 256)).unwrap();
        for pattern in 0..3usize.pow(7) {
            let mut remaining = pattern;
            let mut symbols = Vec::new();
            for _ in 0..7 {
                symbols.push([97, 98, ending][remaining % 3]);
                remaining /= 3;
            }
            symbols.push(256);
            let mut buffer = PackedSymbols::new(9);
            for &s in &symbols {
                buffer.push(s);
            }
            let counter = PairCounter::from_buffer(&buffer, &grammar);
            let stats = Statistics::from_counter(&counter, grammar.alphabet_size());
            let mut raw = HashMap::new();
            let mut left = vec![0; grammar.alphabet_size()];
            let mut right = left.clone();
            for pair in symbols.windows(2) {
                if !grammar.ends_field(pair[0]) {
                    *raw.entry((pair[0], pair[1])).or_insert(0) += 1;
                    left[pair[0] as usize] += 1;
                    right[pair[1] as usize] += 1;
                }
            }
            assert_eq!(stats.edges, raw.values().sum());
            assert_eq!(stats.left, left);
            assert_eq!(stats.right, right);
            for (&pair, &count) in &raw {
                assert_eq!(stats.observed(pair), count);
            }
        }
    }
}
