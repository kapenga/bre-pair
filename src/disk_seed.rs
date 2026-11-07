//! Experimental two-pass raw-byte initialization. The first pass retains only
//! byte-pair counts and observed byte triples; the second applies that same
//! compatible batch while creating the first packed in-memory stream.
use crate::{
    batch::{Candidate, PairCounter},
    batch_experiments,
    engine::{Compressed, Config},
    grammar::{Grammar, Symbol, BASE_SYMBOLS, EOF},
    packed::{symbol_width, PackedSymbols},
    scoring::{Model, Statistics},
};
use std::{
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
    time::Instant,
};

const BYTES: usize = 256;
const PAIRS: usize = BASE_SYMBOLS * BASE_SYMBOLS;
const TRIPLES: usize = BYTES * BYTES * BASE_SYMBOLS;
const MISSING: Symbol = u32::MAX;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Stamp {
    bytes: u64,
    fields: u64,
    symbols: u64,
    fingerprint: u64,
}

fn visit(
    mut input: impl BufRead,
    mut push: impl FnMut(Symbol) -> Result<(), String>,
) -> Result<Stamp, String> {
    let mut stamp = Stamp {
        bytes: 0,
        fields: 0,
        symbols: 0,
        fingerprint: 0xcbf29ce484222325,
    };
    let mut open = false;
    loop {
        let bytes = input.fill_buf().map_err(|e| e.to_string())?;
        if bytes.is_empty() {
            break;
        }
        for &byte in bytes {
            stamp.fingerprint = (stamp.fingerprint ^ u64::from(byte)).wrapping_mul(0x100000001b3);
            push(u32::from(byte))?;
            open = true;
            if byte == b'\n' {
                push(EOF)?;
                stamp.fields = stamp.fields.checked_add(1).ok_or("field count overflow")?;
                open = false;
            }
        }
        let n = bytes.len();
        stamp.bytes = stamp
            .bytes
            .checked_add(n as u64)
            .ok_or("input byte count overflow")?;
        input.consume(n);
    }
    if open {
        push(EOF)?;
        stamp.fields = stamp.fields.checked_add(1).ok_or("field count overflow")?;
    }
    stamp.symbols = stamp
        .bytes
        .checked_add(stamp.fields)
        .ok_or("input symbol count overflow")?;
    Ok(stamp)
}

struct ByteStatistics {
    counts: Vec<u64>,
    omitted_self: [u64; BASE_SYMBOLS],
    triples: Vec<u64>,
    previous: Option<Symbol>,
    older: Option<Symbol>,
    run: u64,
}
impl ByteStatistics {
    fn new() -> Self {
        Self {
            counts: vec![0; PAIRS],
            omitted_self: [0; BASE_SYMBOLS],
            triples: vec![0; TRIPLES.div_ceil(64)],
            previous: None,
            older: None,
            run: 0,
        }
    }
    fn push(&mut self, symbol: Symbol) {
        self.run = if self.previous == Some(symbol) {
            self.run + 1
        } else {
            1
        };
        if let Some(a) = self.previous {
            if a != EOF {
                if a == symbol && self.run % 2 == 1 {
                    self.omitted_self[symbol as usize] += 1;
                } else {
                    self.counts[a as usize * BASE_SYMBOLS + symbol as usize] += 1;
                }
            }
            if let Some(older) = self.older {
                if older != EOF && a != EOF {
                    let bit =
                        (older as usize * BYTES + a as usize) * BASE_SYMBOLS + symbol as usize;
                    self.triples[bit / 64] |= 1 << (bit % 64);
                }
            }
        }
        self.older = self.previous;
        self.previous = Some(symbol);
    }
    fn visit_triples(&self, mut visit: impl FnMut(Symbol, Symbol, Symbol)) {
        for (word, &value) in self.triples.iter().enumerate() {
            let mut bits = value;
            while bits != 0 {
                let at = word * 64 + bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let c = at % BASE_SYMBOLS;
                let pair = at / BASE_SYMBOLS;
                visit((pair / BYTES) as u32, (pair % BYTES) as u32, c as u32);
            }
        }
    }
    fn choose(&self, config: &Config) -> Vec<Candidate> {
        let mut counter = PairCounter::new(true);
        for (at, &count) in self.counts.iter().enumerate() {
            if count != 0 {
                counter.counts.insert(
                    ((at / BASE_SYMBOLS) as u32, (at % BASE_SYMBOLS) as u32),
                    count,
                );
            }
        }
        for (id, &count) in self.omitted_self.iter().enumerate() {
            if count != 0 {
                counter.self_overlaps.insert(id as u32, count);
            }
        }
        let stats = Statistics::from_counter(&counter, BASE_SYMBOLS);
        let candidates = counter
            .counts
            .iter()
            .filter(|(_, n)| **n >= config.minimum_count)
            .map(|(&pair, &count)| Candidate {
                pair,
                count,
                priority: if config.score == Model::Frequency {
                    count as f64
                } else {
                    stats.score_with_alpha(config.score, pair, count, config.pmi_alpha)
                },
            })
            .collect();
        batch_experiments::select(candidates, &config.search.for_round(0), |index, edge| {
            self.visit_triples(|a, b, c| {
                if let (Some(&p), Some(&q)) = (index.get(&(a, b)), index.get(&(b, c))) {
                    if p != q {
                        edge(p, q);
                    }
                }
            });
        })
        .pairs
    }
}

#[derive(Clone, Debug)]
pub struct Report {
    pub input_bytes: u64,
    pub fields: u64,
    pub symbols_before: u64,
    pub symbols_after: u64,
    pub selected_pairs: usize,
    pub replacements: u64,
    pub width: u8,
    pub dense_statistics_bytes: usize,
    pub packed_payload_bytes: usize,
    pub packed_allocation_bytes: usize,
    pub first_scan_seconds: f64,
    pub selection_seconds: f64,
    pub second_scan_seconds: f64,
}
pub struct Seed {
    pub compressed: Compressed,
    pub selected: Vec<Candidate>,
    pub report: Report,
}

struct PackedOutput {
    blocks: Vec<PackedSymbols>,
    current: PackedSymbols,
    allocated: usize,
    limit: usize,
    chunk_symbols: usize,
    expected: u64,
    emitted: u64,
    width: u8,
}
impl PackedOutput {
    fn new(width: u8, chunk_symbols: usize, expected: u64, limit: usize) -> Result<Self, String> {
        let reserve = chunk_symbols
            .min(usize::try_from(expected).map_err(|_| "symbol count exceeds address space")?);
        let requested = reserve
            .checked_mul(width as usize)
            .ok_or("packed capacity overflow")?
            .div_ceil(64)
            * 8;
        if requested > limit {
            return Err("initial packed chunk exceeds buffer budget".into());
        }
        Ok(Self {
            blocks: Vec::new(),
            current: PackedSymbols::with_capacity(width, reserve),
            allocated: 0,
            limit,
            chunk_symbols,
            expected,
            emitted: 0,
            width,
        })
    }
    fn push(&mut self, id: Symbol, grammar: &Grammar) -> Result<(), String> {
        if self.emitted == self.expected {
            return Err("second pass exceeds predicted symbol count".into());
        }
        self.current.push(id);
        self.emitted += 1;
        if self
            .allocated
            .saturating_add(self.current.allocated_bytes())
            > self.limit
        {
            return Err("second-pass packed stream exceeds buffer budget".into());
        }
        if self.current.len() >= self.chunk_symbols && grammar.ends_field(id) {
            self.flush();
            let remaining = usize::try_from(self.expected - self.emitted)
                .map_err(|_| "remaining symbols exceed address space")?;
            self.current =
                PackedSymbols::with_capacity(self.width, self.chunk_symbols.min(remaining));
        }
        Ok(())
    }
    fn flush(&mut self) {
        if !self.current.is_empty() {
            let mut block = std::mem::replace(&mut self.current, PackedSymbols::new(self.width));
            block.shrink_to_fit();
            self.allocated += block.allocated_bytes();
            self.blocks.push(block);
        }
    }
    fn finish(mut self) -> Result<Vec<PackedSymbols>, String> {
        if self.emitted != self.expected {
            return Err("second-pass symbol count differs from prediction".into());
        }
        self.flush();
        Ok(self.blocks)
    }
}

/// Two raw-byte scans with LF-retaining fields, exactly like compress_file.
/// The returned stream has already applied one global batch. Input fingerprints,
/// all selected replacement counts and the predicted output length are checked.
pub fn prepare(
    first: impl BufRead,
    second: impl BufRead,
    config: &Config,
    chunk_symbols: usize,
) -> Result<Seed, String> {
    if chunk_symbols == 0
        || config.minimum_count < 2
        || config.pruning.is_some()
        || config.search.frequency_override_percent != 0
        || !(1..=8).contains(&config.search.fill_waves)
        || !config.pmi_alpha.is_finite()
        || !(0.0..=4.0).contains(&config.pmi_alpha)
        || (config.score != Model::WeightedPmi && config.pmi_alpha != 1.0)
    {
        return Err("invalid two-pass initialization configuration".into());
    }
    let start = Instant::now();
    let mut stats = ByteStatistics::new();
    let stamp = visit(first, |symbol| {
        stats.push(symbol);
        Ok(())
    })?;
    let first_scan_seconds = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let selected = stats.choose(config);
    let selection_seconds = start.elapsed().as_secs_f64();
    let replacements = selected.iter().try_fold(0u64, |total, c| {
        total
            .checked_add(c.count)
            .ok_or("replacement count overflow")
    })?;
    let after = stamp
        .symbols
        .checked_sub(replacements)
        .ok_or("invalid predicted replacement count")?;
    let mut grammar = Grammar::default();
    let mut mapping = vec![MISSING; PAIRS];
    for c in &selected {
        mapping[c.pair.0 as usize * BASE_SYMBOLS + c.pair.1 as usize] = grammar.add(c.pair)?;
    }
    let width = symbol_width(grammar.alphabet_size());
    let mut out = PackedOutput::new(width, chunk_symbols, after, config.buffer_bytes)?;
    let dense_statistics_bytes =
        stats.counts.len() * 8 + stats.omitted_self.len() * 8 + stats.triples.len() * 8;
    // The observed-triple bitmap and selection temporaries are not kept during
    // packing. Only selected expected counts are needed to verify the rewrite.
    drop(stats);
    let start = Instant::now();
    let mut actual = vec![0u64; PAIRS];
    let mut pending = None;
    let checked = visit(second, |symbol| {
        if let Some(a) = pending.take() {
            let at = a as usize * BASE_SYMBOLS + symbol as usize;
            let replacement = mapping[at];
            if replacement != MISSING {
                actual[at] += 1;
                out.push(replacement, &grammar)?;
            } else {
                out.push(a, &grammar)?;
                pending = Some(symbol);
            }
        } else {
            pending = Some(symbol);
        }
        Ok(())
    })?;
    if let Some(symbol) = pending {
        out.push(symbol, &grammar)?;
    }
    if stamp != checked {
        return Err("input differs between initialization passes".into());
    }
    for c in &selected {
        if actual[c.pair.0 as usize * BASE_SYMBOLS + c.pair.1 as usize] != c.count {
            return Err("selected pair count differs during second-pass rewrite".into());
        }
    }
    let blocks = out.finish()?;
    let second_scan_seconds = start.elapsed().as_secs_f64();
    let report = Report {
        input_bytes: stamp.bytes,
        fields: stamp.fields,
        symbols_before: stamp.symbols,
        symbols_after: after,
        selected_pairs: selected.len(),
        replacements,
        width,
        dense_statistics_bytes,
        packed_payload_bytes: blocks.iter().map(PackedSymbols::payload_bytes).sum(),
        packed_allocation_bytes: blocks.iter().map(PackedSymbols::allocated_bytes).sum(),
        first_scan_seconds,
        selection_seconds,
        second_scan_seconds,
    };
    Ok(Seed {
        compressed: Compressed {
            grammar,
            blocks,
            input_bytes: stamp.bytes,
            field_count: stamp.fields,
        },
        selected,
        report,
    })
}

pub fn prepare_file(path: &Path, config: &Config, chunk_symbols: usize) -> Result<Seed, String> {
    let first = BufReader::with_capacity(65536, File::open(path).map_err(|e| e.to_string())?);
    let second = BufReader::with_capacity(65536, File::open(path).map_err(|e| e.to_string())?);
    prepare(first, second, config, chunk_symbols)
}
