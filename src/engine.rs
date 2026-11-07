use std::time::Instant;

use crate::batch::{self, PairCounter};
use crate::grammar::{Grammar, Symbol, TokenTrie, BASE_SYMBOLS, EOF};
use crate::packed::{symbol_width, PackedSymbols};
use crate::pair_map::PairMap;
use crate::prune;
use crate::scoring::{Model, Statistics};
use crate::search;

#[derive(Clone, Debug)]
pub struct Config {
    /// Active packed sequence budget; auxiliary memory is separate in this prototype.
    pub buffer_bytes: usize,
    pub minimum_count: u64,
    pub pair_counting: batch::Counting,
    pub search: search::Options,
    pub score: Model,
    /// Replacement-count exponent for weighted PMI. Valid range: 0..=4.
    pub pmi_alpha: f64,
    /// Whole-input experiment: prune once after a replacement checkpoint.
    pub pruning: Option<prune::Checkpoint>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            buffer_bytes: 1024 * 1024,
            minimum_count: 4,
            pair_counting: batch::Counting::Adaptive,
            search: search::Options::default(),
            score: Model::Frequency,
            pmi_alpha: 1.0,
            pruning: None,
        }
    }
}

#[derive(Debug)]
pub struct Round {
    pub index: usize,
    pub candidates: usize,
    pub eligible: usize,
    pub tie_edges: usize,
    pub selected: usize,
    /// Non-overlapping occurrence counts of the selected pairs, before rewrite.
    pub minimum_selected_count: u64,
    pub maximum_selected_count: u64,
    pub new_rules: usize,
    pub before: usize,
    pub rewritten: usize,
    pub after_refill: usize,
    pub removed: usize,
    pub refill_bytes: usize,
    pub total_rules: usize,
    pub width: u8,
    /// Packed active-buffer storage after rewrite/refill, including word padding.
    pub packed_bytes: usize,
    pub seconds: f64,
    pub components: usize,
    pub largest_component: usize,
    pub greedy_score: u64,
    pub score_upper_bound: u64,
    pub proven_optimal: bool,
    pub search_nodes: usize,
    pub search_seconds: f64,
    pub improved_components: usize,
    pub score_seconds: f64,
    pub selection_seconds: f64,
    pub rewrite_seconds: f64,
    pub refill_seconds: f64,
}

pub struct Compressed {
    pub grammar: Grammar,
    pub blocks: Vec<PackedSymbols>,
    pub input_bytes: u64,
    pub field_count: u64,
}

impl Compressed {
    pub fn token_count(&self) -> usize {
        self.blocks.iter().map(PackedSymbols::len).sum()
    }
    pub fn data_bytes(&self) -> usize {
        self.blocks.iter().map(PackedSymbols::payload_bytes).sum()
    }
    pub fn dictionary_bits(&self) -> usize {
        (0..self.grammar.rules.len())
            .map(|i| 2 * symbol_width(BASE_SYMBOLS + i) as usize)
            .sum()
    }

    pub fn decode(&self) -> Result<Vec<Vec<u8>>, String> {
        let size =
            usize::try_from(self.input_bytes).map_err(|_| "input too large for this platform")?;
        let field_count = usize::try_from(self.field_count).map_err(|_| "too many fields")?;
        let mut fields = Vec::new();
        let mut field = Vec::new();
        let mut emitted = 0;
        // Explicit stack also handles deep grammars without recursive calls.
        let mut stack = Vec::new();
        for block in &self.blocks {
            for symbol in block.iter() {
                if symbol as usize >= self.grammar.alphabet_size() {
                    return Err("data refers to an undefined symbol".into());
                }
                stack.push(symbol);
                while let Some(symbol) = stack.pop() {
                    if symbol == EOF {
                        if fields.len() >= field_count {
                            return Err("too many field markers".into());
                        }
                        fields.push(std::mem::take(&mut field));
                    } else if symbol < EOF {
                        if emitted == size || fields.len() >= field_count {
                            return Err("decoded data exceeds header lengths".into());
                        }
                        field.push(symbol as u8);
                        emitted += 1;
                    } else {
                        let rule = &self.grammar.rules[symbol as usize - BASE_SYMBOLS];
                        stack.push(rule.right);
                        stack.push(rule.left);
                    }
                }
            }
        }
        if emitted != size || fields.len() != field_count || !field.is_empty() {
            return Err("decoded lengths or final field marker do not match header".into());
        }
        Ok(fields)
    }
}

pub struct Training {
    pub compressed: Compressed,
    pub rounds: Vec<Round>,
    pub refill_calls: usize,
    pub width_flushes: usize,
    pub stalled_flushes: usize,
    pub max_buffer_bytes: usize,
    pub seconds: f64,
    pub pruning: Option<prune::State>,
}

#[derive(Default)]
struct Cursor {
    field: usize,
    offset: usize,
}

impl Cursor {
    fn done(&self, fields: &[Vec<u8>]) -> bool {
        self.field == fields.len()
    }

    fn refill(
        &mut self,
        fields: &[Vec<u8>],
        grammar: &Grammar,
        buffer: &mut PackedSymbols,
        counter: &mut PairCounter,
        capacity: usize,
    ) -> usize {
        if self.done(fields) || buffer.len() == capacity {
            return 0;
        }
        let trie = if grammar.rules.is_empty() {
            None
        } else {
            Some(TokenTrie::new(grammar))
        };
        let mut added_bytes = 0;
        while !self.done(fields) && buffer.len() < capacity {
            let field = &fields[self.field];
            if let Some(trie) = &trie {
                // Exact over the complete unread suffix. Only a fitting prefix is
                // appended; the unconsumed suffix is reparsed against the next grammar.
                for symbol in trie.segment_prefix(&field[self.offset..], capacity - buffer.len()) {
                    if buffer.len() == capacity {
                        break;
                    }
                    buffer.push(symbol);
                    counter.push(symbol, grammar);
                    let ends = grammar.ends_field(symbol);
                    let bytes = grammar.terminals(symbol) - usize::from(ends);
                    self.offset += bytes;
                    added_bytes += bytes;
                    if ends {
                        self.field += 1;
                        self.offset = 0;
                        break;
                    }
                }
            } else {
                while self.offset < field.len() && buffer.len() < capacity {
                    let symbol = field[self.offset] as Symbol;
                    buffer.push(symbol);
                    counter.push(symbol, grammar);
                    self.offset += 1;
                    added_bytes += 1;
                }
                if self.offset == field.len() && buffer.len() < capacity {
                    buffer.push(EOF);
                    counter.push(EOF, grammar);
                    self.field += 1;
                    self.offset = 0;
                }
            }
        }
        added_bytes
    }
}

pub fn compress(
    fields: &[Vec<u8>],
    config: &Config,
    mut on_round: impl FnMut(&Round),
) -> Result<Training, String> {
    if config.minimum_count < 2 {
        return Err("minimum count must be at least 2".into());
    }
    if let Some(checkpoint) = config.pruning {
        if checkpoint.after_round == 0 || checkpoint.max_uses >= config.minimum_count {
            return Err(
                "pruning checkpoint must be positive and max uses below the merge minimum".into(),
            );
        }
    }
    if !config.pmi_alpha.is_finite() || !(0.0..=4.0).contains(&config.pmi_alpha) {
        return Err("PMI alpha must be finite and between 0 and 4".into());
    }
    if config.score != Model::WeightedPmi && config.pmi_alpha != 1.0 {
        return Err("PMI alpha applies only to weighted-pmi scoring".into());
    }
    if config.buffer_bytes < 8 {
        return Err("buffer must contain at least 8 bytes".into());
    }
    if config.search.beam_width == 0 || config.search.node_limit == 0 {
        return Err("beam width and search-node budget must be positive".into());
    }
    let buffer_bits = (config.buffer_bytes / 8)
        .checked_mul(64)
        .ok_or("buffer size overflow")?;
    let started = Instant::now();
    let track_overlaps = config.score != Model::Frequency;
    let mut grammar = Grammar::default();
    let initial_symbols = fields
        .iter()
        .fold(0usize, |n, field| {
            n.saturating_add(field.len()).saturating_add(1)
        })
        .min(buffer_bits / 9);
    let mut buffer = PackedSymbols::with_capacity(9, initial_symbols);
    let mut counter = PairCounter::new(track_overlaps);
    let mut cursor = Cursor::default();
    let mut blocks = Vec::new();
    let mut rounds = Vec::new();
    let mut refill_calls = 1;
    let mut width_flushes = 0;
    let mut stalled_flushes = 0;
    cursor.refill(fields, &grammar, &mut buffer, &mut counter, buffer_bits / 9);
    if config.pruning.is_some() && !cursor.done(fields) {
        return Err("during-training pruning requires the complete input to fit initially".into());
    }
    let mut max_buffer_bytes = buffer.storage_bytes();
    let mut pruning: Option<prune::State> = None;
    loop {
        if let Some(checkpoint) = config.pruning {
            if pruning.is_none() && rounds.len() == checkpoint.after_round {
                if !blocks.is_empty() {
                    return Err("pruning checkpoint requires an unsealed whole-input buffer".into());
                }
                let pruning_started = Instant::now();
                let plan =
                    prune::plan(&grammar, std::slice::from_ref(&buffer), checkpoint.max_uses)?;
                let after_tokens = plan
                    .before_tokens
                    .checked_add(plan.added_tokens)
                    .ok_or("pruning length overflow")?;
                if after_tokens > buffer_bits / buffer.width() as usize {
                    return Err("pruning expansion exceeds the packed buffer budget".into());
                }
                if plan.added_tokens > 0 {
                    drop(counter);
                    let (output, counted) =
                        prune::rewrite_active(&buffer, &grammar, &plan.retired, track_overlaps);
                    if output.len() != after_tokens {
                        return Err("pruning prediction disagrees with output".into());
                    }
                    buffer = output;
                    counter = counted;
                }
                max_buffer_bytes = max_buffer_bytes.max(buffer.storage_bytes());
                pruning = Some(prune::State {
                    retired: plan.retired,
                    event: prune::Event {
                        after_round: checkpoint.after_round,
                        max_uses: checkpoint.max_uses,
                        selected_rules: plan.selected_rules,
                        before_tokens: plan.before_tokens,
                        after_tokens,
                        seconds: pruning_started.elapsed().as_secs_f64(),
                    },
                    reactivated: 0,
                });
            }
        }
        let round_started = Instant::now();
        let score_started = Instant::now();
        let statistics = if config.score == Model::Frequency {
            None
        } else {
            Some(Statistics::from_counter(&counter, grammar.alphabet_size()))
        };
        let score_seconds = score_started.elapsed().as_secs_f64();
        let selection_started = Instant::now();
        let selection = batch::select_scored_with_alpha(
            &buffer,
            &counter.counts,
            config.minimum_count,
            &config.search.for_round(rounds.len()),
            statistics.as_ref().map(|stats| (config.score, stats)),
            config.pmi_alpha,
        );
        let selection_seconds = selection_started.elapsed().as_secs_f64();
        drop(statistics);
        if selection.pairs.is_empty() {
            if !buffer.is_empty() {
                blocks.push(buffer);
            }
            if cursor.done(fields) {
                break;
            }
            // Seal a stalled block and continue reading with the persistent grammar.
            // This creates a block boundary, never an artificial field marker.
            stalled_flushes += 1;
            let width = symbol_width(grammar.alphabet_size());
            buffer = PackedSymbols::new(width);
            counter = PairCounter::new(track_overlaps);
            cursor.refill(
                fields,
                &grammar,
                &mut buffer,
                &mut counter,
                buffer_bits / width as usize,
            );
            refill_calls += 1;
            max_buffer_bytes = max_buffer_bytes.max(buffer.storage_bytes());
            continue;
        }
        let new_rules = selection
            .pairs
            .iter()
            .filter(|c| grammar.find(c.pair).is_none())
            .count();
        let new_alphabet = grammar
            .alphabet_size()
            .checked_add(new_rules)
            .ok_or("alphabet size overflow")?;
        if new_alphabet > u32::MAX as usize {
            return Err("symbol IDs exhausted".into());
        }
        let width = symbol_width(new_alphabet);
        let removed = selection
            .pairs
            .iter()
            .map(|c| c.count as usize)
            .sum::<usize>();
        let rewritten = buffer
            .len()
            .checked_sub(removed)
            .ok_or("invalid predicted batch reduction")?;
        if rewritten > buffer_bits / width as usize {
            // A width transition can exceed the budget despite fewer tokens. Seal
            // a prefix, then recount and reselect on the retained snapshot.
            let keep = (buffer.len() / 2).max(1);
            let split = buffer.len() - keep;
            let mut prefix = PackedSymbols::new(buffer.width());
            let mut suffix = PackedSymbols::new(buffer.width());
            for (i, symbol) in buffer.iter().enumerate() {
                if i < split {
                    prefix.push(symbol);
                } else {
                    suffix.push(symbol);
                }
            }
            if prefix.is_empty() {
                return Err("buffer is too small for the next symbol width".into());
            }
            blocks.push(prefix);
            buffer = suffix;
            counter = PairCounter::from_buffer_with_tracking(&buffer, &grammar, track_overlaps);
            width_flushes += 1;
            continue;
        }
        let before = buffer.len();
        let rewrite_started = Instant::now();
        let mut replacements =
            PairMap::with_capacity_and_hasher(selection.pairs.len(), Default::default());
        let mut minimum_selected_count = u64::MAX;
        let mut maximum_selected_count = 0;
        for candidate in &selection.pairs {
            minimum_selected_count = minimum_selected_count.min(candidate.count);
            maximum_selected_count = maximum_selected_count.max(candidate.count);
            let symbol = grammar.add(candidate.pair)?;
            if let Some(state) = &mut pruning {
                if let Some(retired) = state.retired.get_mut(symbol as usize - BASE_SYMBOLS) {
                    if *retired {
                        *retired = false;
                        state.reactivated += 1;
                    }
                }
            }
            replacements.insert(candidate.pair, symbol);
        }
        let (output, output_counter) = batch::rewrite_counted(
            &buffer,
            &replacements,
            &grammar,
            width,
            counter,
            config.pair_counting,
        );
        if output.len() != rewritten {
            return Err("batch counts disagree with actual replacements".into());
        }
        buffer = output;
        counter = output_counter;
        let rewrite_seconds = rewrite_started.elapsed().as_secs_f64();
        let refill_started = Instant::now();
        let refill_bytes = if !cursor.done(fields) {
            refill_calls += 1;
            cursor.refill(
                fields,
                &grammar,
                &mut buffer,
                &mut counter,
                buffer_bits / width as usize,
            )
        } else {
            0
        };
        let refill_seconds = refill_started.elapsed().as_secs_f64();
        max_buffer_bytes = max_buffer_bytes.max(buffer.storage_bytes());
        let round = Round {
            index: rounds.len() + 1,
            candidates: selection.candidates,
            eligible: selection.eligible,
            tie_edges: selection.tie_edges,
            selected: selection.pairs.len(),
            minimum_selected_count,
            maximum_selected_count,
            new_rules,
            before,
            rewritten,
            after_refill: buffer.len(),
            removed,
            refill_bytes,
            total_rules: grammar.rules.len(),
            width,
            packed_bytes: buffer.storage_bytes(),
            seconds: round_started.elapsed().as_secs_f64(),
            components: selection.components,
            largest_component: selection.largest_component,
            greedy_score: selection.greedy_score,
            score_upper_bound: selection.score_upper_bound,
            proven_optimal: selection.proven_optimal,
            search_nodes: selection.search_nodes,
            search_seconds: selection.search_seconds,
            improved_components: selection.improved_components,
            score_seconds,
            selection_seconds,
            rewrite_seconds,
            refill_seconds,
        };
        on_round(&round);
        rounds.push(round);
    }
    Ok(Training {
        compressed: Compressed {
            grammar,
            blocks,
            input_bytes: fields.iter().map(|f| f.len() as u64).sum(),
            field_count: fields.len() as u64,
        },
        rounds,
        refill_calls,
        width_flushes,
        stalled_flushes,
        max_buffer_bytes,
        seconds: started.elapsed().as_secs_f64(),
        pruning,
    })
}
