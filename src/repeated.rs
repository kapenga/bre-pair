//! Restart discovery from a flat frozen vocabulary's exact segmentation.
//! Binary construction helpers are storage details, never implicit candidates.
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::time::Instant;

use crate::engine::{Compressed, Config, Round};
use crate::field_engine::{self, MergeBlock, Options, ParallelTraining};
use crate::grammar::{Grammar, Symbol, BASE_SYMBOLS, EOF};
use crate::packed::{symbol_width, PackedSymbols};
use crate::tokenizer::{Model, Tokenizer};

pub struct Discovery {
    pub pool: Model,
    pub support: Vec<u64>,
    pub training: ParallelTraining,
    pub initial_tokens: u64,
    pub helper_rules: usize,
    pub segmentation_seconds: f64,
    pub blocked_candidates: u64,
    pub block_lifted_early: bool,
}

pub struct DiscoveryOptions {
    pub workers: usize,
    pub candidate_limit: usize,
    pub max_token_bytes: usize,
    pub collapse: Vec<Symbol>,
    pub block_rounds: usize,
    pub shake: Option<crate::shake::Options>,
}

fn intern(labels: &[Symbol], grammar: &mut Grammar) -> Result<Symbol, String> {
    if labels.len() == 1 {
        return Ok(labels[0]);
    }
    let middle = labels.len() / 2;
    let left = intern(&labels[..middle], grammar)?;
    let right = intern(&labels[middle..], grammar)?;
    grammar.add((left, right))
}

pub(crate) fn seed_grammar(seed: &Model) -> Result<(Grammar, Vec<Symbol>), String> {
    let mut grammar = Grammar::default();
    let mut mapping = Vec::with_capacity(seed.tokens.len());
    for token in &seed.tokens {
        let labels: Vec<_> = token
            .bytes
            .iter()
            .map(|&b| b as Symbol)
            .chain(token.ends_field.then_some(EOF))
            .collect();
        mapping.push(intern(&labels, &mut grammar)?);
    }
    Ok((grammar, mapping))
}

/// Exact parse each original raw line, preserving delimiters and virtual EOF.
/// Only the selected seed tokens occur in the initial packed stream.
pub fn prepare(
    training: &Path,
    seed: &Model,
    chunk_symbols: usize,
    buffer_bytes: usize,
) -> Result<(Compressed, Vec<bool>), String> {
    prepare_collapsed(training, seed, &[], chunk_symbols, buffer_bytes)
}

pub fn prepare_collapsed(
    training: &Path,
    seed: &Model,
    collapse: &[Symbol],
    chunk_symbols: usize,
    buffer_bytes: usize,
) -> Result<(Compressed, Vec<bool>), String> {
    if chunk_symbols == 0 {
        return Err("chunk size must be positive".into());
    }
    let tokenizer = Tokenizer::new(seed.clone())?;
    let mut active = vec![1; seed.tokens.len()];
    for &id in collapse {
        if (id as usize) < BASE_SYMBOLS || id as usize >= active.len() || active[id as usize] == 0 {
            return Err("collapse IDs must be distinct learned tokens".into());
        }
        active[id as usize] = 0;
    }
    let (grammar, mapping) = seed_grammar(seed)?;
    let mut members = vec![false; grammar.alphabet_size()];
    for &symbol in &mapping {
        members[symbol as usize] = true;
    }
    debug_assert!(members[..BASE_SYMBOLS].iter().all(|&b| b));
    let width = symbol_width(grammar.alphabet_size());
    let mut blocks = Vec::new();
    let mut block = PackedSymbols::new(width);
    let mut input = BufReader::new(File::open(training).map_err(|e| e.to_string())?);
    let progress = crate::training_progress::enabled();
    let total_bytes = if progress {
        input.get_ref().metadata().map_err(|e| e.to_string())?.len()
    } else {
        0
    };
    crate::training_progress::phase("packing", 0, 0, total_bytes, "bytes");
    let mut next_progress = 64 * 1024 * 1024u64;
    let mut field = Vec::new();
    let (mut input_bytes, mut field_count, mut packed_bytes) = (0u64, 0u64, 0usize);
    loop {
        field.clear();
        if input
            .read_until(b'\n', &mut field)
            .map_err(|e| e.to_string())?
            == 0
        {
            break;
        }
        input_bytes = input_bytes
            .checked_add(field.len() as u64)
            .ok_or("input size overflow")?;
        field_count += 1;
        let ids = if collapse.is_empty() {
            tokenizer.encode(&field)
        } else {
            tokenizer.encode_active(&field, Some(&active))
        };
        for id in ids {
            block.push(mapping[id as usize]);
        }
        if progress && input_bytes >= next_progress {
            crate::training_progress::phase("packing", 0, input_bytes, total_bytes, "bytes");
            next_progress = input_bytes.saturating_add(64 * 1024 * 1024);
        }
        if packed_bytes
            .checked_add(block.storage_bytes())
            .is_none_or(|n| n > buffer_bytes)
        {
            return Err("seed segmentation exceeds packed budget".into());
        }
        if block.len() >= chunk_symbols {
            packed_bytes += block.storage_bytes();
            blocks.push(block);
            block = PackedSymbols::new(width);
        }
    }
    if !block.is_empty() {
        blocks.push(block);
    }
    crate::training_progress::phase("packing", 0, input_bytes, total_bytes, "bytes");
    Ok((
        Compressed {
            grammar,
            blocks,
            input_bytes,
            field_count,
        },
        members,
    ))
}

/// Grow the collapsed alphabet again using the existing central registry.
/// candidate_limit counts admitted symbol IDs, including the seed/basis;
/// helper-only nodes do not consume that budget. Deduplication and the byte
/// length bound may make the exported pool smaller than the limit.
pub fn discover(
    training: &Path,
    seed: &Model,
    config: &Config,
    workers: usize,
    candidate_limit: usize,
    max_token_bytes: usize,
    on_round: impl FnMut(&Round),
) -> Result<Discovery, String> {
    discover_with_options(
        training,
        seed,
        config,
        &DiscoveryOptions {
            workers,
            candidate_limit,
            max_token_bytes,
            collapse: vec![],
            block_rounds: 0,
            shake: None,
        },
        on_round,
    )
}

pub fn discover_with_options(
    training: &Path,
    seed: &Model,
    config: &Config,
    options: &DiscoveryOptions,
    on_round: impl FnMut(&Round),
) -> Result<Discovery, String> {
    let DiscoveryOptions {
        workers,
        candidate_limit,
        max_token_bytes,
        collapse,
        block_rounds,
        shake,
    } = options;
    let (workers, candidate_limit, max_token_bytes) =
        (*workers, *candidate_limit, *max_token_bytes);
    if candidate_limit <= seed.tokens.len()
        || max_token_bytes == 0
        || seed.tokens.iter().any(|t| t.bytes.len() > max_token_bytes)
    {
        return Err(
            "candidate limit must exceed seed size; byte limit must retain every seed".into(),
        );
    }
    let started = Instant::now();
    if !collapse.is_empty() && *block_rounds == 0 {
        return Err("targeted discovery needs a positive blocked warm-up".into());
    }
    if shake.is_some() && !collapse.is_empty() {
        return Err("random shakes and initial targeted collapses are separate experiments".into());
    }
    if let Some(shake) = shake {
        shake.validate()?;
    }
    let (source, members) =
        prepare_collapsed(training, seed, collapse, 65536, config.buffer_bytes)?;
    let initial_tokens = source.blocks.iter().map(|b| b.len() as u64).sum();
    let helper_rules = source.grammar.rules.len();
    let segmentation_seconds = started.elapsed().as_secs_f64();
    let result = field_engine::compress_seeded(
        source,
        members,
        config,
        workers,
        Options {
            max_rules: Some(candidate_limit),
            shake: *shake,
            ..Options::default()
        },
        if collapse.is_empty() {
            None
        } else {
            Some(MergeBlock::new(
                collapse
                    .iter()
                    .map(|&id| {
                        let token = &seed.tokens[id as usize];
                        token
                            .bytes
                            .iter()
                            .map(|&b| b as Symbol)
                            .chain(token.ends_field.then_some(EOF))
                            .collect()
                    })
                    .collect(),
                *block_rounds,
            ))
        },
        on_round,
    )?;
    let (pool, support) = Model::candidates_filtered(
        &result.training.compressed,
        max_token_bytes,
        result.candidate_membership.as_deref(),
    )?;
    Ok(Discovery {
        pool,
        support,
        blocked_candidates: result.blocked_candidates,
        block_lifted_early: result.block_lifted_early,
        training: result,
        initial_tokens,
        helper_rules,
        segmentation_seconds,
    })
}
