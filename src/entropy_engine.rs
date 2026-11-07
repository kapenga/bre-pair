//! Experimental batched Re-Pair guided by frozen float32 byte-boundary scores.
//! The ordinary engine is independent of this experimental bookkeeping.
use crate::batch::{self, Candidate, Edge, Position};
use crate::context_entropy::{self, weight_units, History, Model};
use crate::engine::{Compressed, Round, Training};
use crate::entropy_registry::{Registry, Writer};
use crate::grammar::{Grammar, Symbol, EOF};
use crate::packed::{symbol_width, PackedSymbols};
use crate::pair_map::PairMap;
use crate::search;
use std::sync::Mutex;
use std::time::Instant;

#[derive(Clone, Debug)]
pub struct Options {
    pub context: context_entropy::Options,
    pub minimum_count: u64,
    pub workers: usize,
    pub chunk_symbols: usize,
    pub buffer_bytes: usize,
    pub max_rules: Option<usize>,
    pub max_rounds: Option<usize>,
    pub search: search::Options,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            context: context_entropy::Options::default(),
            minimum_count: 8,
            workers: 4,
            chunk_symbols: 65536,
            buffer_bytes: 134217728,
            max_rules: None,
            max_rounds: None,
            search: search::Options {
                certify: false,
                ..search::Options::default()
            },
        }
    }
}

pub struct ResultSet {
    pub training: Training,
    pub context_statistics: context_entropy::Statistics,
    pub statistics_seconds: f64,
    pub annotation_seconds: f64,
    pub discovery_seconds: f64,
    pub initial_weight_bytes: usize,
    pub final_weight_bytes: usize,
    pub initial_weight_capacity_bytes: usize,
    pub final_weight_capacity_bytes: usize,
    pub mean_initial_weight: f64,
    pub zero_weights: u64,
    pub backoff_by_order: [u64; 7],
    pub peak_pair_entries: usize,
    pub stopped_at_rule_limit: bool,
    pub stopped_at_round_limit: bool,
}

struct Buffers {
    blocks: Vec<PackedSymbols>,
    weights: Vec<Vec<f32>>,
}

fn rewrite(
    block: &mut PackedSymbols,
    weights: &mut Vec<f32>,
    grammar: &Grammar,
    replacements: &PairMap<Symbol>,
    width: u8,
    writer: &mut Writer<'_>,
) -> Result<(), String> {
    debug_assert_eq!(block.len(), weights.len());
    let end = block.len();
    let mut output = (width != block.width()).then(|| PackedSymbols::with_capacity(width, end));
    let mut cursor = output.is_none().then(|| block.rewrite_from(0));
    let mut read = 0;
    let mut written = 0;
    let mut position = Position::default();
    while read < end {
        let a = block.get(read);
        let weight = weights[read]; // Incoming boundary of the first child.
        let replacement = (read + 1 < end)
            .then(|| block.get(read + 1))
            .and_then(|b| replacements.get(&(a, b)).copied());
        let emitted = if let Some(symbol) = replacement {
            read += 2;
            symbol
        } else {
            read += 1;
            a
        };
        // Only the join between the two children disappears. The next
        // symbol retains its original incoming weight on the next iteration.
        weights[written] = weight;
        written += 1;
        if let Edge::Pair(pair) = position.advance(emitted, grammar, false) {
            writer.push(pair, weight_units(weight))?;
        }
        if let Some(output) = &mut output {
            output.push(emitted);
        } else {
            block.push_rewritten(cursor.as_mut().unwrap(), emitted);
        }
    }
    if let Some(output) = output {
        *block = output;
    } else {
        block.finish_rewrite(cursor.unwrap());
    }
    weights.truncate(written);
    // Reclaim only after a substantial reduction, not on every rewrite.
    if weights.capacity() / 2 > weights.len() {
        weights.shrink_to_fit();
    }
    Ok(())
}

fn parallel_rewrite(
    buffers: Buffers,
    workers: usize,
    registry: &Registry,
    grammar: &Grammar,
    replacements: &PairMap<Symbol>,
    width: u8,
) -> Result<Buffers, String> {
    let count = buffers.blocks.len();
    let jobs = Mutex::new(buffers.blocks.into_iter().zip(buffers.weights).enumerate());
    let mut results = std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for _ in 0..workers.min(count.max(1)) {
            let jobs = &jobs;
            handles.push(scope.spawn(move || {
                let mut writer = registry.writer();
                let mut result = Vec::new();
                loop {
                    let job = jobs
                        .lock()
                        .map_err(|_| "poisoned entropy chunk queue")?
                        .next();
                    let Some((index, (mut block, mut weights))) = job else {
                        break;
                    };
                    rewrite(
                        &mut block,
                        &mut weights,
                        grammar,
                        replacements,
                        width,
                        &mut writer,
                    )?;
                    result.push((index, block, weights));
                }
                writer.flush()?;
                Ok::<_, String>(result)
            }));
        }
        let mut result = Vec::with_capacity(count);
        for handle in handles {
            result.extend(handle.join().map_err(|_| "entropy worker panicked")??);
        }
        Ok::<_, String>(result)
    })?;
    results.sort_unstable_by_key(|&(index, _, _)| index);
    let (blocks, weights) = results
        .into_iter()
        .map(|(_, block, weights)| (block, weights))
        .unzip();
    Ok(Buffers { blocks, weights })
}

pub fn compress(
    fields: &[Vec<u8>],
    options: &Options,
    mut on_round: impl FnMut(&Round),
) -> Result<ResultSet, String> {
    if !(1..=256).contains(&options.workers)
        || options.chunk_symbols == 0
        || options.minimum_count < 2
        || options.max_rules == Some(0)
        || options.max_rounds == Some(0)
    {
        return Err("invalid entropy trainer workers, chunks, minimum or stopping limit".into());
    }
    let symbols = fields
        .iter()
        .try_fold(0usize, |n, f| n.checked_add(f.len())?.checked_add(1))
        .ok_or("entropy input size overflow")?;
    if (symbols as u128 * 9).div_ceil(8) > options.buffer_bytes as u128 {
        return Err(
            "entropy experiment requires the initial input to fit the packed budget".into(),
        );
    }
    let initial_weight_bytes = symbols
        .checked_mul(4)
        .ok_or("weight buffer size overflow")?;
    let started = Instant::now();
    let model = Model::train(fields, options.context)?;
    let context_statistics = model.statistics;
    let statistics_seconds = started.elapsed().as_secs_f64();
    let annotated = Instant::now();
    let mut registry = Registry::new();
    let mut writer = registry.writer();
    let mut buffers = Buffers {
        blocks: Vec::new(),
        weights: Vec::new(),
    };
    // Plan EOF-aligned capacities from field lengths, without another byte
    // scan. Geometric growth would otherwise nearly double the f32 buffer
    // when the last complete field takes a chunk just past its target.
    let mut chunk_lengths = Vec::with_capacity(symbols.div_ceil(options.chunk_symbols));
    let mut length = 0;
    for field in fields {
        length += field.len() + 1;
        if length >= options.chunk_symbols {
            chunk_lengths.push(length);
            length = 0;
        }
    }
    if length != 0 {
        chunk_lengths.push(length);
    }
    let mut capacities = chunk_lengths.into_iter();
    let capacity = capacities.next().unwrap_or(0);
    let mut block = PackedSymbols::with_capacity(9, capacity);
    let mut weights = Vec::with_capacity(capacity);
    let mut position = Position::default();
    let grammar = Grammar::default();
    let mut weight_sum = 0f64;
    let mut zero_weights = 0;
    let mut backoff_by_order = [0u64; 7];
    for field in fields {
        let mut history = History::default();
        for (index, symbol) in field.iter().map(|&b| u32::from(b)).chain([EOF]).enumerate() {
            let weight = if index == 0 {
                0.0
            } else {
                let (weight, order) = model.weight(&history);
                backoff_by_order[order] += 1;
                weight_sum += f64::from(weight);
                zero_weights += u64::from(weight == 0.0);
                weight
            };
            block.push(symbol);
            weights.push(weight);
            if let Edge::Pair(pair) = position.advance(symbol, &grammar, false) {
                writer.push(pair, weight_units(weight))?;
            }
            if symbol != EOF {
                history.push(symbol as u8, model.order());
            }
        }
        if block.len() >= options.chunk_symbols {
            buffers.blocks.push(block);
            buffers.weights.push(weights);
            let capacity = capacities.next().unwrap_or(0);
            block = PackedSymbols::with_capacity(9, capacity);
            weights = Vec::with_capacity(capacity);
        }
    }
    if !block.is_empty() {
        buffers.blocks.push(block);
        buffers.weights.push(weights);
    }
    writer.flush()?;
    drop(writer);
    drop(model); // No predictor is needed after the second source pass.
    let initial_weight_capacity_bytes = buffers.weights.iter().map(|w| w.capacity() * 4).sum();
    let annotation_seconds = annotated.elapsed().as_secs_f64();
    let discovery_started = Instant::now();
    let mut grammar = grammar;
    let mut rounds = Vec::new();
    let mut peak_pair_entries = 0;
    let mut max_buffer_bytes: usize = buffers
        .blocks
        .iter()
        .map(PackedSymbols::storage_bytes)
        .sum();
    let mut stopped_at_rule_limit = false;
    let mut stopped_at_round_limit = false;
    #[cfg(any(test, feature = "audit"))]
    let original_weights = buffers.weights.clone();
    loop {
        let round_started = Instant::now();
        #[cfg(any(test, feature = "audit"))]
        audit(&buffers, &original_weights, &grammar, &registry)?;
        let mut candidates = Vec::new();
        let mut entries = 0;
        registry.visit(|pair, value| {
            entries += 1;
            if value.count >= options.minimum_count {
                candidates.push(Candidate {
                    pair,
                    count: value.count,
                    priority: value.priority(),
                });
            }
        });
        peak_pair_entries = peak_pair_entries.max(entries);
        let score_seconds = round_started.elapsed().as_secs_f64();
        let selection_started = Instant::now();
        let mut selection = batch::select_candidates_fields(
            &buffers.blocks,
            candidates,
            &options.search,
            options.workers,
        );
        if let Some(limit) = options.max_rules {
            let available = limit.saturating_sub(grammar.rules.len());
            let mut added = 0;
            let mut keep = 0;
            for candidate in &selection.pairs {
                if grammar.find(candidate.pair).is_none() {
                    if added == available {
                        break;
                    }
                    added += 1;
                }
                keep += 1;
            }
            if keep < selection.pairs.len() {
                selection.pairs.truncate(keep);
                selection.score = selection.pairs.iter().map(|c| c.count).sum();
                selection.greedy_score = selection.score;
            }
        }
        let selection_seconds = selection_started.elapsed().as_secs_f64();
        if selection.pairs.is_empty() {
            break;
        }
        let before = buffers.blocks.iter().map(PackedSymbols::len).sum::<usize>();
        let removed =
            usize::try_from(selection.score).map_err(|_| "weighted replacement size overflow")?;
        let predicted = before
            .checked_sub(removed)
            .ok_or("weighted replacement count underflow")?;
        let rewrite_started = Instant::now();
        let previous_rules = grammar.rules.len();
        let mut replacements = PairMap::default();
        let mut minimum_selected_count = u64::MAX;
        let mut maximum_selected_count = 0;
        for candidate in &selection.pairs {
            minimum_selected_count = minimum_selected_count.min(candidate.count);
            maximum_selected_count = maximum_selected_count.max(candidate.count);
            replacements.insert(candidate.pair, grammar.add(candidate.pair)?);
        }
        let width = symbol_width(grammar.alphabet_size());
        if (predicted as u128 * u128::from(width)).div_ceil(8) > options.buffer_bytes as u128 {
            return Err("entropy rewrite exceeds the packed budget".into());
        }
        registry.clear();
        buffers = parallel_rewrite(
            buffers,
            options.workers,
            &registry,
            &grammar,
            &replacements,
            width,
        )?;
        let rewritten = buffers.blocks.iter().map(PackedSymbols::len).sum::<usize>();
        if rewritten != predicted {
            return Err("weighted replacement counts disagree with output".into());
        }
        let packed_bytes = buffers
            .blocks
            .iter()
            .map(PackedSymbols::storage_bytes)
            .sum();
        max_buffer_bytes = max_buffer_bytes.max(packed_bytes);
        let round = Round {
            index: rounds.len() + 1,
            candidates: selection.candidates,
            eligible: selection.eligible,
            tie_edges: selection.tie_edges,
            selected: selection.pairs.len(),
            minimum_selected_count,
            maximum_selected_count,
            new_rules: grammar.rules.len() - previous_rules,
            before,
            rewritten,
            after_refill: rewritten,
            removed,
            refill_bytes: 0,
            total_rules: grammar.rules.len(),
            width,
            packed_bytes,
            seconds: round_started.elapsed().as_secs_f64(),
            components: selection.components,
            largest_component: selection.largest_component,
            greedy_score: selection.greedy_score,
            score_upper_bound: selection.score_upper_bound,
            proven_optimal: false,
            search_nodes: selection.search_nodes,
            search_seconds: selection.search_seconds,
            improved_components: selection.improved_components,
            score_seconds,
            selection_seconds,
            rewrite_seconds: rewrite_started.elapsed().as_secs_f64(),
            refill_seconds: 0.0,
        };
        on_round(&round);
        rounds.push(round);
        stopped_at_rule_limit = options
            .max_rules
            .is_some_and(|limit| grammar.rules.len() >= limit);
        stopped_at_round_limit = options
            .max_rounds
            .is_some_and(|limit| rounds.len() >= limit);
        if stopped_at_rule_limit || stopped_at_round_limit {
            break;
        }
    }
    #[cfg(any(test, feature = "audit"))]
    audit(&buffers, &original_weights, &grammar, &registry)?;
    if stopped_at_rule_limit || stopped_at_round_limit {
        let mut entries = 0;
        registry.visit(|_, _| entries += 1);
        peak_pair_entries = peak_pair_entries.max(entries);
    }
    let final_weight_bytes = buffers.weights.iter().map(|w| w.len() * 4).sum();
    let final_weight_capacity_bytes = buffers.weights.iter().map(|w| w.capacity() * 4).sum();
    let discovery_seconds = discovery_started.elapsed().as_secs_f64();
    let observations = backoff_by_order.iter().sum::<u64>();
    Ok(ResultSet {
        training: Training {
            compressed: Compressed {
                grammar,
                blocks: buffers.blocks,
                input_bytes: fields.iter().map(|f| f.len() as u64).sum(),
                field_count: fields.len() as u64,
            },
            rounds,
            refill_calls: 1,
            width_flushes: 0,
            stalled_flushes: 0,
            max_buffer_bytes,
            seconds: started.elapsed().as_secs_f64(),
            pruning: None,
        },
        context_statistics,
        statistics_seconds,
        annotation_seconds,
        discovery_seconds,
        initial_weight_bytes,
        final_weight_bytes,
        initial_weight_capacity_bytes,
        final_weight_capacity_bytes,
        mean_initial_weight: if observations == 0 {
            0.0
        } else {
            weight_sum / observations as f64
        },
        zero_weights,
        backoff_by_order,
        peak_pair_entries,
        stopped_at_rule_limit,
        stopped_at_round_limit,
    })
}

#[cfg(any(test, feature = "audit"))]
fn audit(
    buffers: &Buffers,
    original: &[Vec<f32>],
    grammar: &Grammar,
    registry: &Registry,
) -> Result<(), String> {
    let mut expected: PairMap<(u64, u128)> = PairMap::default();
    for ((block, weights), original) in buffers.blocks.iter().zip(&buffers.weights).zip(original) {
        if block.len() != weights.len() {
            return Err("boundary length audit failed".into());
        }
        let mut offset = 0;
        let symbols: Vec<_> = block.iter().collect();
        for (&symbol, &weight) in symbols.iter().zip(weights) {
            if original.get(offset).map(|w| w.to_bits()) != Some(weight.to_bits()) {
                return Err("original byte boundary score was lost during a merge".into());
            }
            offset += grammar.terminals(symbol);
        }
        if offset != original.len() {
            return Err("terminal coverage audit failed".into());
        }
        let mut add = |a, b, right: usize| {
            let entry = expected.entry((a, b)).or_default();
            entry.0 += 1;
            entry.1 += u128::from(weight_units(weights[right]));
        };
        let mut start = 0;
        while start < symbols.len() {
            let symbol = symbols[start];
            let mut end = start + 1;
            while end < symbols.len() && symbols[end] == symbol {
                end += 1;
            }
            if !grammar.ends_field(symbol) {
                for right in (start + 1..end).step_by(2) {
                    add(symbol, symbol, right);
                }
                if end < symbols.len() {
                    add(symbol, symbols[end], end);
                }
            }
            start = end;
        }
    }
    let mut mismatch = false;
    registry.visit(|pair, value| {
        mismatch |= expected.remove(&pair) != Some((value.count, value.units()));
    });
    if mismatch || !expected.is_empty() {
        return Err("weighted count/sum audit failed".into());
    }
    Ok(())
}
