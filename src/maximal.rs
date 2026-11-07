//! Whole-file sequential reference experiment, separate from the packed batch engine.
//! Maximal repeats are found by extending every occurrence of a maximum-frequency
//! pair. Frequencies here include overlaps, as in the MR-RePair paper; the minimum
//! threshold is checked against the replacements that can actually be made.
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
use std::time::Instant;

use crate::engine::Compressed;
use crate::grammar::{Grammar, Pair, Symbol, BASE_SYMBOLS, EOF};
use crate::packed::{symbol_width, PackedSymbols};

/// Native rules have arbitrary right-hand sides, rather than binary trees.
pub struct NativeCompressed {
    pub rules: Vec<Vec<Symbol>>,
    pub blocks: Vec<PackedSymbols>,
    pub input_bytes: u64,
    pub field_count: u64,
}

impl NativeCompressed {
    pub fn token_count(&self) -> usize {
        self.blocks.iter().map(PackedSymbols::len).sum()
    }

    pub fn data_bytes(&self) -> usize {
        self.blocks.iter().map(PackedSymbols::payload_bytes).sum()
    }

    /// This conversion also validates topological order and field boundaries.
    /// It is used for verification, not to measure the native dictionary size.
    pub fn to_binary(&self) -> Result<Compressed, String> {
        let mut grammar = Grammar::default();
        let mut mapping: Vec<Symbol> = (0..BASE_SYMBOLS as Symbol).collect();
        for rhs in &self.rules {
            if rhs.len() < 2 {
                return Err("a native rule must have at least two children".into());
            }
            let mut children = Vec::with_capacity(rhs.len());
            for &child in rhs {
                children.push(
                    *mapping
                        .get(child as usize)
                        .ok_or("undefined native child")?,
                );
            }
            let mut result = children[0];
            for &child in &children[1..] {
                result = grammar.add((result, child))?;
            }
            mapping.push(result);
        }
        let width = symbol_width(grammar.alphabet_size());
        let mut blocks = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            let mut output = PackedSymbols::new(width);
            for symbol in block.iter() {
                output.push(
                    *mapping
                        .get(symbol as usize)
                        .ok_or("undefined native token")?,
                );
            }
            blocks.push(output);
        }
        Ok(Compressed {
            grammar,
            blocks,
            input_bytes: self.input_bytes,
            field_count: self.field_count,
        })
    }
}

#[derive(Debug)]
pub struct Round {
    pub index: usize,
    pub seed: Pair,
    pub seed_frequency: usize,
    pub rhs_length: usize,
    pub replacements: usize,
    pub trimmed: bool,
    pub before: usize,
    pub after: usize,
}

pub struct Training {
    pub compressed: NativeCompressed,
    pub rounds: Vec<Round>,
    pub seconds: f64,
}

const NONE: usize = usize::MAX;

struct Node {
    symbol: Symbol,
    previous: usize,
    next: usize,
}

#[derive(Default)]
struct Occurrences {
    positions: HashSet<usize>,
    version: u64,
}

/// Every live, legal adjacency is indexed, including overlapping self-pairs.
/// Stable node indices retain text order after deletion. Versioned heap entries
/// allow local frequency updates without searching or rebuilding the text.
struct Index {
    nodes: Vec<Node>,
    ends_field: Vec<bool>,
    occurrences: HashMap<Pair, Occurrences>,
    queue: BinaryHeap<(usize, Reverse<Pair>, u64)>,
    dirty: HashSet<Pair>,
}

impl Index {
    fn new(fields: &[Vec<u8>]) -> Self {
        let mut nodes = Vec::new();
        for field in fields {
            for symbol in field.iter().map(|&b| b as Symbol).chain([EOF]) {
                let i = nodes.len();
                nodes.push(Node {
                    symbol,
                    previous: i.checked_sub(1).unwrap_or(NONE),
                    next: i + 1,
                });
            }
        }
        if let Some(last) = nodes.last_mut() {
            last.next = NONE;
        }
        let mut ends_field = vec![false; BASE_SYMBOLS];
        ends_field[EOF as usize] = true;
        let mut index = Self {
            nodes,
            ends_field,
            occurrences: HashMap::new(),
            queue: BinaryHeap::new(),
            dirty: HashSet::new(),
        };
        for i in 0..index.nodes.len() {
            index.add_edge(i);
        }
        index.flush();
        index
    }

    fn pair_at(&self, at: usize) -> Option<Pair> {
        if at == NONE {
            return None;
        }
        let node = &self.nodes[at];
        if node.next == NONE || self.ends_field[node.symbol as usize] {
            return None;
        }
        Some((node.symbol, self.nodes[node.next].symbol))
    }

    fn add_edge(&mut self, at: usize) {
        if let Some(pair) = self.pair_at(at) {
            let inserted = self
                .occurrences
                .entry(pair)
                .or_default()
                .positions
                .insert(at);
            assert!(inserted, "adjacency was already indexed");
            self.dirty.insert(pair);
        }
    }

    fn remove_edge(&mut self, at: usize) {
        if let Some(pair) = self.pair_at(at) {
            let removed = self
                .occurrences
                .get_mut(&pair)
                .unwrap()
                .positions
                .remove(&at);
            assert!(removed, "adjacency was missing from index");
            self.dirty.insert(pair);
        }
    }

    fn flush(&mut self) {
        for pair in self.dirty.drain() {
            let entry = self.occurrences.get_mut(&pair).unwrap();
            entry.version += 1;
            if entry.positions.len() >= 2 {
                self.queue
                    .push((entry.positions.len(), Reverse(pair), entry.version));
            }
        }
    }

    fn pop(&mut self, minimum: usize) -> Option<(Pair, usize)> {
        while let Some((count, Reverse(pair), version)) = self.queue.pop() {
            if count < minimum {
                return None;
            }
            let current = &self.occurrences[&pair];
            if current.version == version && current.positions.len() == count {
                return Some((pair, count));
            }
        }
        None
    }

    fn extend(&self, pair: Pair, maximal: bool) -> (Vec<Symbol>, Vec<(usize, usize)>, bool) {
        let mut starts: Vec<_> = self.occurrences[&pair].positions.iter().copied().collect();
        starts.sort_unstable();
        let mut ends: Vec<_> = starts.iter().map(|&at| self.nodes[at].next).collect();
        let mut rhs = VecDeque::from([pair.0, pair.1]);
        if maximal {
            loop {
                let previous = self.nodes[starts[0]].previous;
                if previous == NONE || self.ends_field[self.nodes[previous].symbol as usize] {
                    break;
                }
                let symbol = self.nodes[previous].symbol;
                if starts.iter().any(|&at| {
                    let p = self.nodes[at].previous;
                    p == NONE || self.nodes[p].symbol != symbol
                }) {
                    break;
                }
                for at in &mut starts {
                    *at = self.nodes[*at].previous;
                }
                rhs.push_front(symbol);
            }
            loop {
                if self.ends_field[*rhs.back().unwrap() as usize] {
                    break;
                }
                let next = self.nodes[ends[0]].next;
                if next == NONE {
                    break;
                }
                let symbol = self.nodes[next].symbol;
                if ends.iter().any(|&at| {
                    let n = self.nodes[at].next;
                    n == NONE || self.nodes[n].symbol != symbol
                }) {
                    break;
                }
                for at in &mut ends {
                    *at = self.nodes[*at].next;
                }
                rhs.push_back(symbol);
            }
        }
        // MR-RePair's endpoint adjustment avoids the one-symbol overlaps of
        // maximum-frequency maximal repeats. Pairs retain the usual run handling.
        let trimmed = maximal && rhs.len() > 2 && rhs.front() == rhs.back();
        if trimmed {
            rhs.pop_front();
            for at in &mut starts {
                *at = self.nodes[*at].next;
            }
        }
        let mut ranges = Vec::new();
        let mut previous_end = None;
        for (start, end) in starts.into_iter().zip(ends) {
            if previous_end.is_none_or(|last| start > last) {
                ranges.push((start, end));
                previous_end = Some(end);
            }
        }
        (rhs.into(), ranges, trimmed)
    }

    fn replace(&mut self, start: usize, end: usize, symbol: Symbol) {
        let previous = self.nodes[start].previous;
        let next = self.nodes[end].next;
        self.remove_edge(previous);
        let mut at = start;
        loop {
            self.remove_edge(at);
            if at == end {
                break;
            }
            at = self.nodes[at].next;
        }
        self.nodes[start].symbol = symbol;
        self.nodes[start].next = next;
        if next != NONE {
            self.nodes[next].previous = start;
        }
        self.add_edge(previous);
        self.add_edge(start);
    }
}

/// `maximal=false` is a sequential pair control using the same data structures,
/// occurrence frequency, minimum-replacement threshold and deterministic ties.
/// This reference loads the whole input and does not implement batching/refill.
pub fn compress(
    fields: &[Vec<u8>],
    minimum: usize,
    maximal: bool,
    mut on_round: impl FnMut(&Round),
) -> Result<Training, String> {
    if minimum < 2 {
        return Err("minimum count must be at least 2".into());
    }
    let started = Instant::now();
    let mut index = Index::new(fields);
    let mut tokens = index.nodes.len();
    let mut rules = Vec::new();
    let mut rounds = Vec::new();
    while let Some((seed, frequency)) = index.pop(minimum) {
        let (rhs, ranges, trimmed) = index.extend(seed, maximal);
        if ranges.len() < minimum {
            // A self-pair's raw frequency includes overlap. Reconsider it when
            // local updates change its occurrences, rather than accepting fewer
            // than the requested number of actual replacements.
            continue;
        }
        let symbol =
            u32::try_from(BASE_SYMBOLS + rules.len()).map_err(|_| "symbol IDs exhausted")?;
        let ends = index.ends_field[*rhs.last().unwrap() as usize];
        index.ends_field.push(ends);
        let before = tokens;
        tokens -= (rhs.len() - 1) * ranges.len();
        for &(start, end) in &ranges {
            index.replace(start, end, symbol);
        }
        index.flush();
        let round = Round {
            index: rounds.len() + 1,
            seed,
            seed_frequency: frequency,
            rhs_length: rhs.len(),
            replacements: ranges.len(),
            trimmed,
            before,
            after: tokens,
        };
        rules.push(rhs);
        on_round(&round);
        rounds.push(round);
    }
    let width = symbol_width(BASE_SYMBOLS + rules.len());
    let mut block = PackedSymbols::new(width);
    let mut at = if index.nodes.is_empty() { NONE } else { 0 };
    while at != NONE {
        block.push(index.nodes[at].symbol);
        at = index.nodes[at].next;
    }
    if block.len() != tokens {
        return Err("reference token counts disagree".into());
    }
    Ok(Training {
        compressed: NativeCompressed {
            rules,
            blocks: if block.is_empty() {
                vec![]
            } else {
                vec![block]
            },
            input_bytes: fields.iter().map(|f| f.len() as u64).sum(),
            field_count: fields.len() as u64,
        },
        rounds,
        seconds: started.elapsed().as_secs_f64(),
    })
}
