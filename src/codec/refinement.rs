//! Bounded research experiment: evaluate code removal by reparsing complete
//! training fields, rather than only expanding the removed roots' children.
//! The calibration archive must fit an explicit AI-ID storage budget. This is
//! deliberately separate from streaming production discovery and its defaults.
use super::{optimize, Codebook, Frozen, MatchScratch, Totals, NONE};
use crate::{
    huffman_archive::{Encoder, Reader, DEFAULT_BLOCK_SYMBOLS},
    tokenizer::fingerprint,
};
use std::{
    fs::File,
    io::{BufReader, BufWriter, Read, Seek},
    path::Path,
};

pub struct Calibration {
    terminals: Vec<u32>,
    ends: Vec<usize>,
    dictionary_hash: u64,
    book_hash: u64,
    vocabulary_hash: u64,
    pub totals: Totals,
    pub original_payload_bits: u128,
}

impl Calibration {
    /// Loads *all* fields or fails before allocating the corpus buffers. The
    /// budget covers the u32 AI IDs and usize field ends, not the shared model.
    pub fn read<R: Read + Seek>(
        model: &Frozen,
        mut reader: Reader<R>,
        maximum_storage: u64,
    ) -> Result<Self, String> {
        model.validate()?;
        let dictionary = model.dictionary.to_bytes();
        let book = model.book.to_bytes();
        if reader.dictionary.to_bytes() != dictionary
            || reader.static_codebook().map(Codebook::to_bytes) != Some(book.clone())
        {
            return Err("calibration archive differs from frozen dictionary/table".into());
        }
        let expected = reader.totals;
        if (expected.bytes, expected.fields, expected.tokens)
            != (
                model.training.bytes,
                model.training.fields,
                model.training.tokens,
            )
        {
            return Err("calibration must represent the complete training input".into());
        }
        let storage = u128::from(expected.tokens) * 4
            + u128::from(expected.fields) * std::mem::size_of::<usize>() as u128;
        if storage > u128::from(maximum_storage) {
            return Err("complete training calibration exceeds AI-ID storage budget".into());
        }
        let tokens = usize::try_from(expected.tokens).map_err(|_| "too many calibration IDs")?;
        let fields = usize::try_from(expected.fields).map_err(|_| "too many calibration fields")?;
        let mut terminals = Vec::with_capacity(tokens);
        let mut ends = Vec::with_capacity(fields);
        let mut counts = vec![0u64; model.dictionary.expansions.len()];
        let mut stack = Vec::new();
        let mut found = Totals::default();
        for index in 0..reader.index.len() {
            let block = reader.block(index)?;
            let mut block_found = Totals::default();
            block
                .book
                .decode(&block.payload, block.bits, block.totals.symbols, |id| {
                    counts[id as usize] = counts[id as usize]
                        .checked_add(1)
                        .ok_or("calibration count overflow")?;
                    block_found.add_symbol(&model.dictionary, id)?;
                    if block_found.exceeds(&block.totals) {
                        return Err("calibration block exceeds declared totals".into());
                    }
                    stack.push(id);
                    while let Some(id) = stack.pop() {
                        if (id as usize) < model.dictionary.model.tokens.len() {
                            if terminals.len() == tokens {
                                return Err("calibration AI IDs exceed declared total".into());
                            }
                            terminals.push(id);
                            if model.dictionary.model.tokens[id as usize].ends_field {
                                if ends.len() == fields {
                                    return Err("calibration fields exceed declared total".into());
                                }
                                ends.push(terminals.len());
                            }
                        } else {
                            let (a, b) = model.dictionary.rules
                                [id as usize - model.dictionary.model.tokens.len()];
                            stack.push(b);
                            stack.push(a);
                        }
                    }
                    Ok(())
                })?;
            if block_found != block.totals {
                return Err("calibration block totals differ".into());
            }
            found.add(block_found)?;
        }
        if found != expected
            || terminals.len() != tokens
            || ends.len() != fields
            || ends.last().copied().unwrap_or(0) != terminals.len()
        {
            return Err("calibration training totals or final EOF differ".into());
        }
        Ok(Self {
            original_payload_bits: optimize::cost(model, &counts)?.0,
            terminals,
            ends,
            dictionary_hash: fingerprint(&dictionary),
            book_hash: fingerprint(&book),
            vocabulary_hash: fingerprint(&model.dictionary.model.to_bytes()),
            totals: found,
        })
    }

    fn check(&self, model: &Frozen) -> Result<(), String> {
        if fingerprint(&model.dictionary.to_bytes()) != self.dictionary_hash
            || fingerprint(&model.book.to_bytes()) != self.book_hash
        {
            return Err("calibration belongs to a different frozen model".into());
        }
        Ok(())
    }

    pub fn storage_bytes(&self) -> usize {
        self.terminals.capacity() * 4 + self.ends.capacity() * std::mem::size_of::<usize>()
    }

    fn histogram(&self, model: &Frozen) -> Result<Vec<u64>, String> {
        let matcher = model.prepare()?;
        let mut scratch = MatchScratch::default();
        let mut counts = vec![0u64; model.dictionary.expansions.len()];
        let mut begin = 0;
        for &end in &self.ends {
            for &id in matcher.segment_with(&self.terminals[begin..end], &mut scratch)? {
                counts[id as usize] = counts[id as usize]
                    .checked_add(1)
                    .ok_or("reparsed frequency overflow")?;
            }
            begin = end;
        }
        let found = optimize::totals(&model.dictionary, &counts)?;
        if (found.bytes, found.fields, found.tokens)
            != (self.totals.bytes, self.totals.fields, self.totals.tokens)
        {
            return Err("reparse changed training expansions".into());
        }
        Ok(counts)
    }

    /// Writes a normal BRHUFF02 archive. Only the code choices change; these
    /// are precisely the AI IDs recovered from the verified training archive.
    pub fn write_archive(&self, model: &Frozen, output: &Path) -> Result<Totals, String> {
        if fingerprint(&model.dictionary.model.to_bytes()) != self.vocabulary_hash {
            return Err("calibration AI vocabulary differs from output model".into());
        }
        let matcher = model.prepare()?;
        let mut scratch = MatchScratch::default();
        let out = BufWriter::new(File::create_new(output).map_err(|e| e.to_string())?);
        let mut encoder =
            Encoder::with_codebook(out, &model.dictionary, DEFAULT_BLOCK_SYMBOLS, &model.book)?;
        let mut begin = 0;
        for &end in &self.ends {
            encoder.extend(matcher.segment_with(&self.terminals[begin..end], &mut scratch)?)?;
            begin = end;
        }
        encoder.finish(self.totals.bytes, self.totals.fields)
    }
}

#[derive(Clone, Copy)]
pub struct Options {
    /// Includes the reweighting-only control and the zero-use group proposal.
    pub max_trials: usize,
    /// Further parse -> Huffman rebuild -> parse steps per proposal.
    pub refits: usize,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            max_trials: 64,
            refits: 2,
        }
    }
}

#[derive(Debug)]
pub struct Trial {
    pub number: usize,
    pub removed_id: Option<u32>,
    pub zero_group: bool,
    pub accepted: bool,
    pub parses: usize,
    pub payload_bits: u128,
    pub metadata_bytes: usize,
    /// First fixed-table parse versus expansion-only under that same table.
    pub boundary_gain_bits: u128,
}

pub struct Report {
    pub trials: usize,
    pub accepted: usize,
    pub parses: usize,
    pub before_payload_bits: u128,
    pub after_payload_bits: u128,
    pub before_metadata_bytes: usize,
    pub after_metadata_bytes: usize,
    pub removed_zero_codes: usize,
    pub removed_positive_codes: usize,
}

fn fit(
    mut model: Frozen,
    calibration: &Calibration,
    refits: usize,
) -> Result<(Frozen, Vec<u64>, usize, u128), String> {
    let mut counts = calibration.histogram(&model)?;
    model.training = optimize::totals(&model.dictionary, &counts)?;
    let mut best = model.to_bytes()?;
    let mut best_counts = counts.clone();
    let mut best_cost = optimize::objective(optimize::cost(&model, &counts)?);
    let first_bits = optimize::cost(&model, &counts)?.0;
    let mut parses = 1;
    for _ in 0..refits {
        let mut weights = vec![0; counts.len()];
        for &id in &model.book.symbols {
            weights[id as usize] = counts[id as usize].max(1);
        }
        let book = Codebook::from_counts(&weights)?;
        if book.to_bytes() == model.book.to_bytes() {
            break;
        }
        model.book = book;
        counts = calibration.histogram(&model)?;
        parses += 1;
        model.training = optimize::totals(&model.dictionary, &counts)?;
        let cost = optimize::objective(optimize::cost(&model, &counts)?);
        if cost < best_cost {
            best_cost = cost;
            best = model.to_bytes()?;
            best_counts.clone_from(&counts);
        }
    }
    Ok((Frozen::from_bytes(&best)?, best_counts, parses, first_bits))
}

impl Frozen {
    pub fn refine_codes(
        &self,
        calibration: &Calibration,
        options: Options,
        mut on_trial: impl FnMut(&Trial),
    ) -> Result<(Frozen, Report), String> {
        self.validate()?;
        calibration.check(self)?;
        if options.max_trials > 10000 || options.refits > 16 {
            return Err("refinement limits: at most 10000 trials and 16 refits".into());
        }
        let basis = self.dictionary.model.tokens.len();
        let n = self.dictionary.expansions.len();
        let mut current_counts = calibration.histogram(self)?;
        let before = optimize::cost(self, &current_counts)?;
        let mut best_cost = before;
        let mut best = Frozen::from_bytes(&self.to_bytes()?)?;
        let mut active = vec![false; n];
        for &id in &self.book.symbols {
            active[id as usize] = true;
        }
        let mut lengths = vec![0; n];
        for (id, bits) in self.book.lengths() {
            lengths[id as usize] = bits;
        }
        let initial_counts = current_counts.clone();
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
            .filter(|&id| active[id] && current_counts[id] != 0)
            .map(|id| {
                let (a, b) = self.dictionary.rules[id - basis];
                let penalty = fallback[a as usize]
                    .saturating_add(fallback[b as usize])
                    .saturating_sub(u128::from(lengths[id]))
                    .saturating_mul(u128::from(current_counts[id]));
                (penalty, current_counts[id], id)
            })
            .collect();
        order.sort_unstable();
        let mut trials = 0;
        let mut accepted = 0;
        let mut parses = 1;
        // Identity/reweight, zero-use batch, then individual weak codes. A zero
        // in the old parse is only a proposal, never proof of no future value.
        let proposals = [None, None]
            .into_iter()
            .chain(order.into_iter().map(|(_, _, id)| Some(id)));
        for (proposal_index, removed) in proposals.enumerate() {
            if trials == options.max_trials {
                break;
            }
            let zero_group = proposal_index == 1;
            let mut proposed = active.clone();
            if zero_group {
                for id in basis..n {
                    if current_counts[id] == 0 {
                        proposed[id] = false;
                    }
                }
                if proposed == active {
                    continue;
                }
            } else if let Some(id) = removed {
                if !active[id] {
                    continue;
                }
                proposed[id] = false;
            }
            trials += 1;
            let (trial, mapping, expanded) = optimize::candidate(self, &current_counts, &proposed)?;
            let expanded_bits = optimize::cost(&trial, &expanded)?.0;
            let (trial, counts, evaluations, first_bits) = fit(trial, calibration, options.refits)?;
            if first_bits > expanded_bits {
                return Err("minimum-bit parsing lost to feasible expansion-only cover".into());
            }
            parses += evaluations;
            let trial_cost = optimize::cost(&trial, &counts)?;
            let improved = optimize::objective(trial_cost) < optimize::objective(best_cost);
            on_trial(&Trial {
                number: trials,
                removed_id: removed.map(|id| id as u32),
                zero_group,
                accepted: improved,
                parses: evaluations,
                payload_bits: trial_cost.0,
                metadata_bytes: trial_cost.1,
                boundary_gain_bits: expanded_bits - first_bits,
            });
            if improved {
                best_cost = trial_cost;
                best = trial;
                active = proposed;
                current_counts.fill(0);
                for (id, &new) in mapping.iter().enumerate() {
                    if new != NONE {
                        current_counts[id] = counts[new as usize];
                    }
                }
                accepted += 1;
            }
        }
        let report = Report {
            trials,
            accepted,
            parses,
            before_payload_bits: before.0,
            after_payload_bits: best_cost.0,
            before_metadata_bytes: before.1,
            after_metadata_bytes: best_cost.1,
            removed_zero_codes: (basis..n)
                .filter(|&id| lengths[id] != 0 && !active[id] && initial_counts[id] == 0)
                .count(),
            removed_positive_codes: (basis..n)
                .filter(|&id| lengths[id] != 0 && !active[id] && initial_counts[id] != 0)
                .count(),
        };
        Ok((best, report))
    }
}

pub fn read_calibration(model: &Frozen, path: &Path, budget: u64) -> Result<Calibration, String> {
    Calibration::read(
        model,
        Reader::new(BufReader::new(File::open(path).map_err(|e| e.to_string())?))?,
        budget,
    )
}
