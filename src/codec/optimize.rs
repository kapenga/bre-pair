//! Training-only code eligibility search. AI IDs remain fixed; uncoded recipes
//! survive only when a coded parent needs their decoding structure.
use super::{Codebook, Dictionary, Frozen, Totals, NONE};
use crate::huffman_archive::{Encoder, Reader, DEFAULT_BLOCK_SYMBOLS};
use std::{
    fs::File,
    io::{BufReader, BufWriter},
    path::Path,
};

#[derive(Clone, Copy, Debug)]
pub struct OptimizeOptions {
    /// Whole-alphabet Huffman rebuilds, including reweighting and zero-use batch.
    pub max_trials: usize,
}
impl Default for OptimizeOptions {
    fn default() -> Self {
        Self { max_trials: 64 }
    }
}
#[derive(Debug)]
pub struct OptimizeReport {
    pub trials: usize,
    pub accepted: usize,
    pub before_codes: usize,
    pub after_codes: usize,
    pub before_rules: usize,
    pub after_rules: usize,
    pub removed_zero_use_codes: usize,
    pub removed_positive_use_codes: usize,
    pub before_payload_bits: u128,
    pub after_payload_bits: u128,
    pub before_metadata_bytes: usize,
    pub after_metadata_bytes: usize,
}
pub struct Optimization {
    pub model: Frozen,
    /// Old symbol -> coded new symbol, or NONE when occurrences must expand.
    pub mapping: Vec<u32>,
    pub report: OptimizeReport,
}

pub(super) fn totals(dictionary: &Dictionary, counts: &[u64]) -> Result<Totals, String> {
    if counts.len() != dictionary.expansions.len() {
        return Err("count alphabet differs from dictionary".into());
    }
    let (mut bytes, mut tokens, mut fields, mut symbols) = (0u128, 0u128, 0u128, 0u128);
    for (e, &n) in dictionary.expansions.iter().zip(counts) {
        bytes += u128::from(e.bytes) * u128::from(n);
        tokens += u128::from(e.terminals) * u128::from(n);
        fields += u128::from(e.ends_field) * u128::from(n);
        symbols += u128::from(n);
        if bytes > u128::from(u64::MAX)
            || tokens > u128::from(u64::MAX)
            || fields > u128::from(u64::MAX)
            || symbols > u128::from(u64::MAX)
        {
            return Err("training histogram totals exceed u64".into());
        }
    }
    Ok(Totals {
        bytes: bytes as u64,
        tokens: tokens as u64,
        fields: fields as u64,
        symbols: symbols as u64,
    })
}
pub(super) fn cost(model: &Frozen, counts: &[u64]) -> Result<(u128, usize), String> {
    let mut bits = vec![0u8; counts.len()];
    for (id, n) in model.book.lengths() {
        bits[id as usize] = n;
    }
    let mut payload = 0u128;
    for (&n, &length) in counts.iter().zip(&bits) {
        if n != 0 && length == 0 {
            return Err("training histogram uses an uncoded symbol".into());
        }
        payload += u128::from(n) * u128::from(length);
    }
    Ok((
        payload,
        model.dictionary.to_bytes().len() + model.book.to_bytes().len(),
    ))
}
pub(super) fn objective(value: (u128, usize)) -> u128 {
    value.0 + value.1 as u128 * 8
}

pub(super) fn candidate(
    original: &Frozen,
    input: &[u64],
    active: &[bool],
) -> Result<(Frozen, Vec<u32>, Vec<u64>), String> {
    candidate_dictionary(&original.dictionary, input, active)
}

pub(super) fn candidate_dictionary(
    original: &Dictionary,
    input: &[u64],
    active: &[bool],
) -> Result<(Frozen, Vec<u32>, Vec<u64>), String> {
    let basis = original.model.tokens.len();
    let n = input.len();
    let mut counts = input.to_vec();
    for id in (basis..n).rev() {
        if !active[id] {
            let uses = std::mem::take(&mut counts[id]);
            let (a, b) = original.rules[id - basis];
            for child in [a, b] {
                counts[child as usize] = counts[child as usize]
                    .checked_add(uses)
                    .ok_or("collapsed count exceeds u64")?;
            }
        }
    }
    let training = totals(original, &counts)?;
    let mut live = active.to_vec();
    for id in (basis..n).rev() {
        if live[id] {
            let (a, b) = original.rules[id - basis];
            live[a as usize] = true;
            live[b as usize] = true;
        }
    }
    let mut mapping = vec![NONE; n];
    for (id, slot) in mapping[..basis].iter_mut().enumerate() {
        *slot = id as u32;
    }
    let mut rules = vec![];
    for id in basis..n {
        if live[id] {
            let (a, b) = original.rules[id - basis];
            if mapping[a as usize] == NONE || mapping[b as usize] == NONE {
                return Err("missing live recipe child".into());
            }
            mapping[id] = (basis + rules.len()) as u32;
            rules.push((mapping[a as usize], mapping[b as usize]));
        }
    }
    let dictionary = Dictionary::new(original.model.clone(), rules, true)?;
    let mut weights = vec![0; dictionary.expansions.len()];
    let mut compact_counts = vec![0; weights.len()];
    for id in 0..n {
        if active[id] {
            let new = mapping[id] as usize;
            weights[new] = counts[id].max(1);
            compact_counts[new] = counts[id];
        }
    }
    for (id, enabled) in active.iter().enumerate() {
        if !enabled {
            mapping[id] = NONE;
        }
    }
    let model = Frozen {
        dictionary,
        book: Codebook::from_counts(&weights)?,
        training,
    };
    model.validate()?;
    Ok((model, mapping, compact_counts))
}

impl Frozen {
    /// Counts must describe an encoding of the training corpus, never holdout.
    /// Scores include all payload bits and exact dictionary/table serialization;
    /// archive block framing and padding are validated by a final actual recode.
    pub fn optimize_counts(
        &self,
        counts: &[u64],
        options: OptimizeOptions,
    ) -> Result<Optimization, String> {
        self.validate()?;
        if options.max_trials > 10000 {
            return Err("optimizer trial limit must be <=10000".into());
        }
        let actual = totals(&self.dictionary, counts)?;
        if (actual.bytes, actual.tokens, actual.fields)
            != (
                self.training.bytes,
                self.training.tokens,
                self.training.fields,
            )
        {
            return Err("histogram does not match training byte/token/field totals".into());
        }
        let before = cost(self, counts)?;
        let mut best_cost = before;
        let n = counts.len();
        let basis = self.dictionary.model.tokens.len();
        let mut active = vec![false; n];
        for &id in &self.book.symbols {
            active[id as usize] = true;
        }
        let mut model = Frozen::from_bytes(&self.to_bytes()?)?;
        let mut mapping: Vec<_> = (0..n)
            .map(|id| if active[id] { id as u32 } else { NONE })
            .collect();
        let (mut trials, mut accepted) = (0, 0);
        // First test reweighting the actual minimum-bit training parse. Its
        // counts can differ from the original BRe-Pair construction stream.
        let mut proposals = vec![active.clone()];
        let mut zeros = active.clone();
        for id in basis..n {
            if counts[id] == 0 {
                zeros[id] = false;
            }
        }
        if zeros != active {
            proposals.push(zeros);
        }
        for proposed in proposals {
            if trials == options.max_trials {
                break;
            }
            trials += 1;
            let (trial, trial_mapping, histogram) = candidate(self, counts, &proposed)?;
            let trial_cost = cost(&trial, &histogram)?;
            if objective(trial_cost) < objective(best_cost) {
                model = trial;
                mapping = trial_mapping;
                active = proposed;
                best_cost = trial_cost;
                accepted += 1;
            }
        }
        // Bounded heuristic order only. Every accepted removal rebuilds the
        // complete Huffman table and measures all symbols, not a local estimate.
        let lengths: Vec<_> = {
            let mut v = vec![0u8; n];
            for (id, bits) in self.book.lengths() {
                v[id as usize] = bits;
            }
            v
        };
        let mut fallback = vec![0u128; n];
        for id in 0..n {
            let expanded = if id < basis {
                u128::MAX
            } else {
                let (a, b) = self.dictionary.rules[id - basis];
                fallback[a as usize].saturating_add(fallback[b as usize])
            };
            fallback[id] = if lengths[id] == 0 {
                expanded
            } else {
                expanded.min(u128::from(lengths[id]))
            };
        }
        let mut order: Vec<_> = (basis..n)
            .filter(|&id| active[id] && counts[id] != 0)
            .map(|id| {
                let (a, b) = self.dictionary.rules[id - basis];
                let expanded = fallback[a as usize].saturating_add(fallback[b as usize]);
                let penalty = expanded
                    .saturating_sub(u128::from(lengths[id]))
                    .saturating_mul(u128::from(counts[id]));
                (penalty, counts[id], id)
            })
            .collect();
        order.sort_unstable();
        for (_, _, id) in order {
            if trials == options.max_trials {
                break;
            }
            let mut proposed = active.clone();
            proposed[id] = false;
            trials += 1;
            let (trial, trial_mapping, histogram) = candidate(self, counts, &proposed)?;
            let trial_cost = cost(&trial, &histogram)?;
            if objective(trial_cost) < objective(best_cost) {
                model = trial;
                mapping = trial_mapping;
                active = proposed;
                best_cost = trial_cost;
                accepted += 1;
            }
        }
        let report = OptimizeReport {
            trials,
            accepted,
            before_codes: self.book.symbols.len(),
            after_codes: model.book.symbols.len(),
            before_rules: self.dictionary.rules.len(),
            after_rules: model.dictionary.rules.len(),
            removed_zero_use_codes: (basis..n)
                .filter(|&id| mapping[id] == NONE && lengths[id] != 0 && counts[id] == 0)
                .count(),
            removed_positive_use_codes: (basis..n)
                .filter(|&id| mapping[id] == NONE && lengths[id] != 0 && counts[id] != 0)
                .count(),
            before_payload_bits: before.0,
            after_payload_bits: best_cost.0,
            before_metadata_bytes: before.1,
            after_metadata_bytes: best_cost.1,
        };
        Ok(Optimization {
            model,
            mapping,
            report,
        })
    }

    pub fn optimize_archive(
        &self,
        input: &Path,
        options: OptimizeOptions,
    ) -> Result<Optimization, String> {
        let mut reader = Reader::new(BufReader::new(
            File::open(input).map_err(|e| e.to_string())?,
        ))?;
        if reader.dictionary.to_bytes() != self.dictionary.to_bytes()
            || reader.static_codebook().map(Codebook::to_bytes) != Some(self.book.to_bytes())
        {
            return Err("training archive does not use this frozen dictionary/table".into());
        }
        let mut counts = vec![0u64; self.dictionary.expansions.len()];
        let mut sum = Totals::default();
        for i in 0..reader.index.len() {
            let block = reader.block(i)?;
            let mut found = Totals::default();
            block
                .book
                .decode(&block.payload, block.bits, block.totals.symbols, |id| {
                    counts[id as usize] = counts[id as usize]
                        .checked_add(1)
                        .ok_or("training symbol count overflow")?;
                    found.add_symbol(&self.dictionary, id).map(|_| ())
                })?;
            if found != block.totals {
                return Err("training block expansion totals differ".into());
            }
            sum.add(found)?;
        }
        if sum != reader.totals {
            return Err("training archive totals differ".into());
        }
        self.optimize_counts(&counts, options)
    }
}

impl Optimization {
    /// Rewrite old roots through the selected subset. Bounded decoded blocks,
    /// one shared model, and one dense histogram; no per-thread dictionaries.
    pub fn recode_archive(
        &self,
        original: &Frozen,
        input: &Path,
        output: &Path,
    ) -> Result<Totals, String> {
        let mut reader = Reader::new(BufReader::new(
            File::open(input).map_err(|e| e.to_string())?,
        ))?;
        if reader.dictionary.to_bytes() != original.dictionary.to_bytes()
            || reader.static_codebook().map(Codebook::to_bytes) != Some(original.book.to_bytes())
            || self.mapping.len() != original.dictionary.expansions.len()
        {
            return Err("recode source differs from optimization source".into());
        }
        let out = BufWriter::new(File::create_new(output).map_err(|e| e.to_string())?);
        let mut encoder = Encoder::with_codebook(
            out,
            &self.model.dictionary,
            DEFAULT_BLOCK_SYMBOLS,
            &self.model.book,
        )?;
        let mut stack = vec![];
        let mut batch = Vec::with_capacity(65536);
        let basis = original.dictionary.model.tokens.len();
        for i in 0..reader.index.len() {
            let block = reader.block(i)?;
            block
                .book
                .decode(&block.payload, block.bits, block.totals.symbols, |id| {
                    stack.push(id);
                    while let Some(id) = stack.pop() {
                        let next = self.mapping[id as usize];
                        if next != NONE {
                            batch.push(next);
                        } else {
                            if (id as usize) < basis {
                                return Err("optimization removed an AI terminal".into());
                            }
                            let (a, b) = original.dictionary.rules[id as usize - basis];
                            stack.push(b);
                            stack.push(a);
                        }
                        if batch.len() == 65536 {
                            encoder.extend(&batch)?;
                            batch.clear();
                        }
                    }
                    Ok(())
                })?;
        }
        encoder.extend(&batch)?;
        let found = encoder.finish(reader.totals.bytes, reader.totals.fields)?;
        if found.tokens != reader.totals.tokens {
            return Err("optimization changed AI token count".into());
        }
        Ok(found)
    }
}
