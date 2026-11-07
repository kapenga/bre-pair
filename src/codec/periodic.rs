//! Research trainer: periodically refit a temporary Huffman table and reparse
//! the active packed stream. Each epoch drops its registry before reparsing,
//! then rebuilds one central registry and exact marginals, including revived
//! rare symbols. Production training defaults are unchanged.
use super::{Codebook, Dictionary, Frozen, MatchScratch, TrainOptions, NONE};
use crate::{
    engine::{Compressed, Config, Round},
    field_engine,
    grammar::BASE_SYMBOLS,
    huffman_archive::{Encoder, Totals, DEFAULT_BLOCK_SYMBOLS},
    packed::{symbol_width, PackedSymbols},
    repeated,
    tokenizer::Model,
};
use std::{fs::File, io::BufWriter, path::Path, time::Instant};

pub struct Event {
    pub after_pass: usize,
    pub before_symbols: u64,
    pub after_symbols: u64,
    pub before_bits: u128,
    pub after_bits: u128,
    pub changed_fields: u64,
    pub maximum_field_tokens: usize,
    pub seconds: f64,
}

pub struct Run {
    pub model: Frozen,
    pub rounds: usize,
    pub events: Vec<Event>,
    pub segmentation_seconds: f64,
    pub merging_seconds: f64,
    pub seconds: f64,
    pub registry_peak_bytes: usize,
    pub stopped_at_round_limit: bool,
    source: Compressed,
    mapping: Vec<u32>,
}

struct Snapshot {
    model: Frozen,
    mapping: Vec<u32>,
    inverse: Vec<u32>,
}

fn snapshot(
    vocabulary: &Model,
    seeds: &[u32],
    source: &Compressed,
    members: &[bool],
    initial_tokens: u64,
) -> Result<Snapshot, String> {
    let mut mapping = vec![NONE; source.grammar.alphabet_size()];
    for (id, &symbol) in seeds.iter().enumerate() {
        mapping[symbol as usize] = id as u32;
    }
    let mut rules = Vec::new();
    for (i, rule) in source.grammar.rules.iter().enumerate() {
        let id = BASE_SYMBOLS + i;
        if !members[id] || mapping[id] != NONE {
            continue;
        }
        let (a, b) = (mapping[rule.left as usize], mapping[rule.right as usize]);
        if a == NONE || b == NONE {
            return Err("periodic rule splits a frozen AI token".into());
        }
        mapping[id] = u32::try_from(vocabulary.tokens.len() + rules.len())
            .map_err(|_| "periodic IDs exhausted")?;
        rules.push((a, b));
    }
    let dictionary = Dictionary::new(vocabulary.clone(), rules, true)?;
    let mut inverse = vec![NONE; dictionary.expansions.len()];
    for (id, &next) in mapping.iter().enumerate() {
        if next != NONE {
            inverse[next as usize] = id as u32;
        }
    }
    let mut counts = vec![0u64; dictionary.expansions.len()];
    let mut actual = Totals::default();
    for block in &source.blocks {
        for symbol in block.iter() {
            let id = mapping[symbol as usize];
            if id == NONE {
                return Err("periodic stream contains construction helper".into());
            }
            counts[id as usize] = counts[id as usize]
                .checked_add(1)
                .ok_or("periodic count overflow")?;
            actual.add_symbol(&dictionary, id)?;
        }
    }
    if (actual.bytes, actual.fields, actual.tokens)
        != (source.input_bytes, source.field_count, initial_tokens)
    {
        return Err("periodic stream changed training totals".into());
    }
    for n in &mut counts {
        *n = (*n).max(1);
    }
    let model = Frozen {
        dictionary,
        book: Codebook::from_counts(&counts)?,
        training: actual,
    };
    Ok(Snapshot {
        model,
        mapping,
        inverse,
    })
}

fn reparse(
    source: &mut Compressed,
    view: &Snapshot,
    pass: usize,
    budget: usize,
) -> Result<Event, String> {
    let start = Instant::now();
    let matcher = view.model.prepare()?;
    let basis = view.model.dictionary.model.tokens.len();
    let mut bits = vec![0u8; view.inverse.len()];
    for (id, n) in view.model.book.lengths() {
        bits[id as usize] = n;
    }
    let mut scratch = MatchScratch::default();
    let (mut field, mut roots, mut stack) = (Vec::new(), Vec::new(), Vec::new());
    let mut event = Event {
        after_pass: pass,
        before_symbols: 0,
        after_symbols: 0,
        before_bits: 0,
        after_bits: 0,
        changed_fields: 0,
        maximum_field_tokens: 0,
        seconds: 0.0,
    };
    let width = symbol_width(source.grammar.alphabet_size());
    let mut packed = source
        .blocks
        .iter()
        .map(PackedSymbols::allocated_bytes)
        .sum::<usize>();
    for block in &mut source.blocks {
        let old_bytes = block.allocated_bytes();
        let mut output = PackedSymbols::with_capacity(width, block.len());
        for symbol in block.iter() {
            let id = view.mapping[symbol as usize];
            roots.push(id);
            stack.push(id);
            while let Some(id) = stack.pop() {
                if (id as usize) < basis {
                    field.push(id);
                } else {
                    let (a, b) = view.model.dictionary.rules[id as usize - basis];
                    stack.push(b);
                    stack.push(a);
                }
            }
            if view.model.dictionary.expansions[id as usize].ends_field {
                let next = matcher.segment_with(&field, &mut scratch)?;
                let before = roots
                    .iter()
                    .map(|&id| u128::from(bits[id as usize]))
                    .sum::<u128>();
                let after = next
                    .iter()
                    .map(|&id| u128::from(bits[id as usize]))
                    .sum::<u128>();
                if after > before {
                    return Err("minimum-bit reparse lost to old training cover".into());
                }
                // Avoid equal-cost/equal-size boundary churn between epochs.
                let improve = after < before || (after == before && next.len() < roots.len());
                let selected = if improve { next } else { &roots };
                #[cfg(feature = "audit")]
                {
                    let mut expanded = Vec::new();
                    for &id in selected {
                        stack.push(id);
                        while let Some(id) = stack.pop() {
                            if (id as usize) < basis {
                                expanded.push(id);
                            } else {
                                let (a, b) = view.model.dictionary.rules[id as usize - basis];
                                stack.push(b);
                                stack.push(a);
                            }
                        }
                    }
                    if expanded != field {
                        return Err("periodic reparse changed exact AI IDs".into());
                    }
                }
                for &id in selected {
                    output.push(view.inverse[id as usize]);
                }
                event.before_symbols = event
                    .before_symbols
                    .checked_add(roots.len() as u64)
                    .ok_or("periodic symbol total overflow")?;
                event.after_symbols = event
                    .after_symbols
                    .checked_add(selected.len() as u64)
                    .ok_or("periodic symbol total overflow")?;
                event.before_bits += before;
                event.after_bits += if improve { after } else { before };
                event.changed_fields += u64::from(improve && next != roots);
                event.maximum_field_tokens = event.maximum_field_tokens.max(field.len());
                field.clear();
                roots.clear();
            }
            if packed - old_bytes + output.allocated_bytes() > budget {
                return Err("periodic reparse exceeds packed buffer budget".into());
            }
        }
        if !field.is_empty() {
            return Err("periodic chunk does not end at EOF".into());
        }
        output.shrink_to_fit();
        packed = packed - old_bytes + output.allocated_bytes();
        *block = output;
    }
    event.seconds = start.elapsed().as_secs_f64();
    Ok(event)
}

/// A finite-round research run. every=0 disables checkpoints; reparse=false
/// tests checkpoint/recount equivalence without changing the active sequence.
#[allow(clippy::too_many_arguments)]
pub fn train(
    vocabulary: &Model,
    input: &Path,
    config: &Config,
    workers: usize,
    options: &TrainOptions,
    every: usize,
    reparse_enabled: bool,
    mut on_round: impl FnMut(usize, &Round),
    mut on_event: impl FnMut(&Event),
) -> Result<Run, String> {
    let start = Instant::now();
    if options.document_counting == super::DocumentCounting::Boundary {
        return Err(
            "boundary document cleanup belongs to uninterrupted codec-train, not periodic research"
                .into(),
        );
    }
    let cap = options
        .max_rounds
        .filter(|&n| n != 0)
        .ok_or("periodic research requires a finite positive round limit")?;
    let mut config = config.clone();
    let size = std::fs::metadata(input).map_err(|e| e.to_string())?.len();
    config.minimum_count = options.effective_minimum(config.minimum_count, size)?;
    let (mut source, mut members) =
        repeated::prepare(input, vocabulary, 65536, config.buffer_bytes)?;
    if source.input_bytes != size {
        return Err("training changed during initial segmentation".into());
    }
    let (_, seeds) = repeated::seed_grammar(vocabulary)?;
    let initial_tokens = source.blocks.iter().map(|b| b.len() as u64).sum();
    let segmentation_seconds = start.elapsed().as_secs_f64();
    let mut rounds = 0;
    let mut events = Vec::new();
    let mut merging_seconds = 0.0;
    let mut registry_peak_bytes = 0;
    let max_rules = options
        .max_rules
        .map(|n| {
            n.checked_add(vocabulary.tokens.len())
                .ok_or("periodic rule limit overflow")
        })
        .transpose()?;
    let stopped_at_round_limit;
    loop {
        let remaining = cap - rounds;
        let length = if every == 0 {
            remaining
        } else {
            every.min(remaining)
        };
        let limits = field_engine::Options {
            max_rounds: Some(length),
            max_rules,
            frequency_cutoff: true,
            memory_saving: true,
            last_seen_documents: options.document_counting == super::DocumentCounting::LastSeen,
            ..Default::default()
        };
        let callback = |r: &Round| on_round(rounds + r.index, r);
        let mut result = if let Some(documents) = &options.documents {
            field_engine::compress_seeded_documents(
                source, members, &config, workers, limits, documents, callback,
            )?
        } else {
            field_engine::compress_seeded(
                source, members, &config, workers, limits, None, callback,
            )?
        };
        rounds += result.training.rounds.len();
        merging_seconds += result.training.seconds;
        registry_peak_bytes = registry_peak_bytes.max(result.registry_peak_slot_bytes);
        let reached_end =
            rounds == cap || result.stopped_at_rule_limit || !result.stopped_at_round_limit;
        members = result
            .candidate_membership
            .take()
            .ok_or("periodic membership absent")?;
        source = result.training.compressed;
        if reached_end {
            stopped_at_round_limit = rounds == cap;
            break;
        }
        if reparse_enabled {
            let view = snapshot(vocabulary, &seeds, &source, &members, initial_tokens)?;
            let event = reparse(&mut source, &view, rounds, config.buffer_bytes)?;
            on_event(&event);
            events.push(event);
        }
        // The epoch's central pair registry was dropped on return. The next
        // epoch rebuilds exact endpoint marginals and pair counts from scratch;
        // merge-only rare-symbol eligibility cannot survive a reparse.
    }
    let Snapshot { model, mapping, .. } =
        snapshot(vocabulary, &seeds, &source, &members, initial_tokens)?;
    Ok(Run {
        model,
        rounds,
        events,
        segmentation_seconds,
        merging_seconds,
        seconds: start.elapsed().as_secs_f64(),
        registry_peak_bytes,
        stopped_at_round_limit,
        source,
        mapping,
    })
}

impl Run {
    pub fn write_archive(&self, output: &Path) -> Result<Totals, String> {
        let out = BufWriter::new(File::create_new(output).map_err(|e| e.to_string())?);
        let mut encoder = Encoder::with_codebook(
            out,
            &self.model.dictionary,
            DEFAULT_BLOCK_SYMBOLS,
            &self.model.book,
        )?;
        let mut batch = Vec::with_capacity(65536);
        for block in &self.source.blocks {
            for symbol in block.iter() {
                batch.push(self.mapping[symbol as usize]);
                if batch.len() == 65536 {
                    encoder.extend(&batch)?;
                    batch.clear();
                }
            }
        }
        encoder.extend(&batch)?;
        encoder.finish(self.source.input_bytes, self.source.field_count)
    }
}
