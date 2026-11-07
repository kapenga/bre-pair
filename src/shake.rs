//! Reproducible random expansion pulses at completed batch barriers.
use crate::{
    grammar::{Grammar, Symbol, BASE_SYMBOLS},
    packed::PackedSymbols,
    pair_map::mix64,
};

#[derive(Clone, Copy, Debug)]
pub struct Options {
    pub fraction: f64,
    pub seed: u64,
    pub pulses: usize,
    pub first_batch: usize,
    pub interval: usize,
    pub block_rounds: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            fraction: 0.10,
            seed: 1,
            pulses: 2,
            first_batch: 1,
            interval: 1,
            block_rounds: 1,
        }
    }
}

impl Options {
    pub(crate) fn validate(&self) -> Result<(), String> {
        if !self.fraction.is_finite()
            || self.fraction <= 0.0
            || self.fraction > 1.0
            || self.pulses == 0
            || self.first_batch == 0
            || self.interval == 0
            || self.block_rounds == 0
        {
            return Err("invalid random shake fraction, schedule, or blocked warm-up".into());
        }
        self.interval
            .checked_mul(self.pulses - 1)
            .and_then(|n| n.checked_add(self.first_batch))
            .ok_or("shake schedule overflow")?;
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct Event {
    pub after_batch: usize,
    pub selected_symbols: Vec<Symbol>,
    pub before: usize,
    pub after: usize,
    pub seconds: f64,
}

pub(crate) struct Plan {
    pub symbols: Vec<Symbol>,
    pub indices: Vec<u32>,
    pub expansions: Vec<Vec<Symbol>>,
    pub before: usize,
    pub after: usize,
}

pub(crate) fn plan(
    blocks: &[PackedSymbols],
    grammar: &Grammar,
    options: &Options,
    pulse: usize,
) -> Result<Plan, String> {
    options.validate()?;
    let mut counts = vec![0u64; grammar.alphabet_size()];
    let mut before = 0usize;
    for block in blocks {
        before = before
            .checked_add(block.len())
            .ok_or("shake length overflow")?;
        for symbol in block.iter() {
            counts[symbol as usize] = counts[symbol as usize]
                .checked_add(1)
                .ok_or("shake count overflow")?;
        }
    }
    let mut symbols: Vec<_> = (BASE_SYMBOLS..counts.len())
        .filter(|&id| counts[id] != 0)
        .map(|id| id as Symbol)
        .collect();
    let key = mix64(options.seed ^ (pulse as u64).wrapping_mul(0x9e3779b97f4a7c15));
    symbols.sort_unstable_by_key(|&id| (mix64(key ^ id as u64), id));
    symbols.truncate((symbols.len() as f64 * options.fraction).ceil() as usize);
    let mut after = before as u128;
    let mut indices = vec![u32::MAX; grammar.alphabet_size()];
    let mut expansions = Vec::with_capacity(symbols.len());
    for (index, &symbol) in symbols.iter().enumerate() {
        after = after
            .checked_add(counts[symbol as usize] as u128 * (grammar.terminals(symbol) - 1) as u128)
            .ok_or("shake expansion overflow")?;
        indices[symbol as usize] = u32::try_from(index).map_err(|_| "shake index overflow")?;
        let mut labels = Vec::with_capacity(grammar.terminals(symbol));
        grammar.visit_terminals(symbol, |terminal| labels.push(terminal));
        expansions.push(labels);
    }
    Ok(Plan {
        symbols,
        indices,
        expansions,
        before,
        after: usize::try_from(after).map_err(|_| "shake length overflow")?,
    })
}
