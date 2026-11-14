//! Shared dictionary/data canonical Huffman experiment. BRHUFV01/02 assign IDs
//! by canonical rank and permits variable-arity recipes and forward references.
//! Pruning uses exact reference counts and an incremental Shannon objective;
//! complete serialized checkpoints, not estimated gains, select the winner.
use crate::{
    huffman::{get_var, put_var, Codebook, Encoding},
    huffman_archive::{
        read_compact_encoded, Block, BlockSource, EncodedBlock, IndexEntry, PreparedDecoder,
        Reader as SourceReader, Totals, DEFAULT_BLOCK_SYMBOLS,
    },
    huffman_dictionary::{byte_model, DecodeGraph, Dictionary, Expansion},
    tokenizer::{fingerprint, Model},
};
use std::{
    fs::{self, File},
    io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write},
    path::Path,
    sync::Arc,
    time::Instant,
};

#[path = "huffman_descriptors.rs"]
mod descriptors;
pub use descriptors::Coding as RuleLengths;
#[path = "huffman_arity_cost.rs"]
mod arity_cost;
#[path = "record_model.rs"]
mod record_model;
pub use record_model::StaticModel;
#[cfg(feature = "bytecode-experiments")]
#[path = "bytecode/archive.rs"]
pub mod byte_archive;
#[path = "huffman_table_projection.rs"]
mod table_projection;
pub use table_projection::{project_tables, TableProjection};
#[path = "huffman_arity_projection.rs"]
mod arity_projection;
pub use arity_projection::project_arities;
#[path = "huffman_arity_runs.rs"]
mod arity_runs;

pub(crate) const MAGIC: &[u8; 8] = b"BRHUFV01";
const HUFFMAN_LENGTHS_MAGIC: &[u8; 8] = b"BRHUFV02";
const RUN_LENGTHS_MAGIC: &[u8; 8] = b"BRHUFV03";
pub fn is_format(magic: &[u8; 8]) -> bool {
    magic == MAGIC || magic == HUFFMAN_LENGTHS_MAGIC || magic == RUN_LENGTHS_MAGIC
}
const INDEX_MAGIC: &[u8; 8] = b"BRHVIDX1";
const HEADER: usize = 128;
const BLOCK_HEADER: usize = 24;
const NONE: u32 = u32::MAX;

fn error(e: std::io::Error) -> String {
    e.to_string()
}
fn read_array<const N: usize>(r: &mut impl Read) -> Result<[u8; N], String> {
    let mut b = [0; N];
    r.read_exact(&mut b).map_err(error)?;
    Ok(b)
}
fn number(b: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(b[at..at + 8].try_into().unwrap())
}
fn var_bytes(n: u64) -> usize {
    ((64 - n.leading_zeros()).max(1) as usize).div_ceil(7)
}
fn xlog(n: u128) -> f64 {
    if n == 0 {
        0.0
    } else {
        n as f64 * (n as f64).log2()
    }
}
fn add(total: &mut u64, n: u64) -> Result<(), String> {
    *total = total
        .checked_add(n)
        .ok_or("shared Huffman count overflow")?;
    Ok(())
}
fn totals_add(t: &mut Totals, e: Expansion) -> Result<(), String> {
    add(&mut t.bytes, e.bytes)?;
    add(&mut t.tokens, e.terminals)?;
    add(&mut t.fields, u64::from(e.ends_field))?;
    add(&mut t.symbols, 1)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CostModel {
    Shannon,
    HuffmanLocal,
    HuffmanCorrected,
}
impl CostModel {
    pub fn name(self) -> &'static str {
        match self {
            Self::Shannon => "shannon",
            Self::HuffmanLocal => "huffman-local",
            Self::HuffmanCorrected => "huffman-corrected",
        }
    }
}
impl std::str::FromStr for CostModel {
    type Err = String;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "shannon" => Ok(Self::Shannon),
            "huffman-local" => Ok(Self::HuffmanLocal),
            "huffman-corrected" => Ok(Self::HuffmanCorrected),
            _ => Err("use shannon, huffman-local or huffman-corrected".into()),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Options {
    pub rounds: usize,
    /// General proposals expand at most this many live children. The cheap
    /// single-reference/no-data case does not need to visit children at all.
    pub max_children: usize,
    pub max_parent_visits: usize,
    pub minimum_gain_bits: f64,
    pub block_symbols: usize,
    /// Research scoring policies; the production default remains Shannon.
    pub cost_model: CostModel,
    pub shannon_warmup: usize,
    /// Refit after roughly 1/slices of the current coded alphabet is removed.
    pub refit_slices: usize,
    /// Automatically compress rule lengths if the complete descriptor is smaller.
    pub rule_lengths: RuleLengths,
    /// Research-only: price the compressed arity section during entropy pruning.
    /// The common single-helper prepass and production default remain unchanged.
    pub huffman_arity_cost: bool,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            rounds: 2,
            max_children: 256,
            max_parent_visits: 4096,
            minimum_gain_bits: 1.0,
            block_symbols: DEFAULT_BLOCK_SYMBOLS,
            cost_model: CostModel::Shannon,
            shannon_warmup: 2,
            refit_slices: 8,
            rule_lengths: RuleLengths::Auto,
            huffman_arity_cost: false,
        }
    }
}

struct State {
    basis: usize,
    rules: Vec<(u32, u32)>,
    reachable: Vec<bool>,
    retired: Vec<bool>,
    data: Vec<u64>,
    refs: Vec<u64>,
    arities: Vec<u64>,
    parent_offsets: Vec<usize>,
    parents: Vec<u32>,
    total: u128,
    sum: f64,
}

/// Read-only diagnostics for the selected shared-dictionary checkpoint.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct PruningAudit {
    pub live_rules: usize,
    pub data_once_unreferenced_rules: usize,
    pub data_once_unreferenced_over_child_limit: usize,
    pub max_arity_data_once_unreferenced: u64,
    pub single_use_helpers: usize,
    pub single_use_helpers_over_parent_walk_limit: usize,
    pub over_child_limit_rules: usize,
    pub over_reference_limit_rules: usize,
}

struct Proposal {
    id: usize,
    parents: Vec<(u32, u64)>,
    children: Vec<(u32, u64)>,
    total: u128,
    sum_delta: f64,
    gain: f64,
    arity_delta: f64,
}

fn multiplicities(mut ids: Vec<u32>) -> Vec<(u32, u64)> {
    ids.sort_unstable();
    let mut result: Vec<(u32, u64)> = Vec::new();
    for id in ids {
        if let Some((_, count)) = result.last_mut().filter(|(last, _)| *last == id) {
            *count += 1;
        } else {
            result.push((id, 1));
        }
    }
    result
}

impl State {
    fn new(dictionary: &Dictionary, data: Vec<u64>) -> Result<Self, String> {
        let n = dictionary.expansions.len();
        if data.len() != n {
            return Err("shared histogram alphabet differs".into());
        }
        let basis = dictionary.model.tokens.len();
        let mut reachable: Vec<_> = data.iter().map(|&n| n != 0).collect();
        for id in (basis..n).rev() {
            if reachable[id] {
                let (a, b) = dictionary.rules[id - basis];
                reachable[a as usize] = true;
                reachable[b as usize] = true;
            }
        }
        let mut refs = vec![0u64; n];
        let mut arities = vec![0u64; n];
        for (i, &(a, b)) in dictionary.rules.iter().enumerate() {
            if reachable[basis + i] {
                arities[basis + i] = 2;
                add(&mut refs[a as usize], 1)?;
                add(&mut refs[b as usize], 1)?;
            }
        }
        let mut parent_offsets = vec![0usize; n + 1];
        for id in 0..n {
            parent_offsets[id + 1] = parent_offsets[id]
                .checked_add(usize::try_from(refs[id]).map_err(|_| "too many parent references")?)
                .ok_or("parent index overflow")?;
        }
        let mut parents = vec![0; parent_offsets[n]];
        let mut at = parent_offsets[..n].to_vec();
        for (i, &(a, b)) in dictionary.rules.iter().enumerate() {
            if reachable[basis + i] {
                for child in [a, b] {
                    parents[at[child as usize]] = (basis + i) as u32;
                    at[child as usize] += 1;
                }
            }
        }
        drop(at);
        let mut total = 0u128;
        let mut sum = 0.0;
        for (&d, &r) in data.iter().zip(&refs) {
            let f = d.checked_add(r).ok_or("combined frequency exceeds u64")?;
            total += u128::from(f);
            sum += xlog(u128::from(f));
        }
        Ok(Self {
            basis,
            rules: dictionary.rules.clone(),
            retired: reachable.iter().map(|&b| !b).collect(),
            reachable,
            data,
            refs,
            arities,
            parent_offsets,
            parents,
            total,
            sum,
        })
    }
    fn live(&self, id: usize) -> bool {
        self.reachable[id] && !self.retired[id]
    }
    fn audit(&self, options: &Options) -> PruningAudit {
        let mut audit = PruningAudit::default();
        for id in self.basis..self.data.len() {
            if !self.live(id) {
                continue;
            }
            audit.live_rules += 1;
            let over_children = self.arities[id] > options.max_children as u64;
            audit.over_child_limit_rules += usize::from(over_children);
            audit.over_reference_limit_rules +=
                usize::from(self.refs[id] > options.max_parent_visits as u64);
            if self.data[id] == 1 && self.refs[id] == 0 {
                audit.data_once_unreferenced_rules += 1;
                audit.data_once_unreferenced_over_child_limit += usize::from(over_children);
                audit.max_arity_data_once_unreferenced =
                    audit.max_arity_data_once_unreferenced.max(self.arities[id]);
            }
            if self.data[id] == 0 && self.refs[id] == 1 {
                audit.single_use_helpers += 1;
                audit.single_use_helpers_over_parent_walk_limit +=
                    usize::from(self.active_parents(id, options.max_parent_visits).is_none());
            }
        }
        audit
    }
    fn children(&self, id: usize, limit: usize) -> Option<Vec<u32>> {
        let (a, b) = self.rules[id - self.basis];
        let mut stack = vec![b, a];
        let mut out = Vec::new();
        while let Some(child) = stack.pop() {
            if child as usize >= self.basis && self.retired[child as usize] {
                let (a, b) = self.rules[child as usize - self.basis];
                stack.push(b);
                stack.push(a);
            } else {
                out.push(child);
            }
            if out.len().checked_add(stack.len())? > limit {
                return None;
            }
        }
        Some(out)
    }
    fn active_parents(&self, id: usize, limit: usize) -> Option<Vec<(u32, u64)>> {
        let mut stack = self.parents[self.parent_offsets[id]..self.parent_offsets[id + 1]].to_vec();
        let mut result = Vec::new();
        let mut visits = 0;
        while let Some(parent) = stack.pop() {
            visits += 1;
            if visits > limit || stack.len() > limit {
                return None;
            }
            let p = parent as usize;
            if self.retired[p] {
                stack.extend_from_slice(
                    &self.parents[self.parent_offsets[p]..self.parent_offsets[p + 1]],
                );
            } else {
                result.push(parent);
            }
        }
        (result.len() as u64 == self.refs[id]).then(|| multiplicities(result))
    }
    fn proposal(&self, id: usize, options: &Options) -> Result<Option<Proposal>, String> {
        if id < self.basis || !self.live(id) {
            return Ok(None);
        }
        let d = self.data[id];
        let r = self.refs[id];
        let f = d.checked_add(r).ok_or("combined frequency exceeds u64")?;
        if f == 0 {
            return Ok(None);
        }
        let cheap = d == 0 && r == 1;
        if !cheap && self.arities[id] > options.max_children as u64 {
            return Ok(None);
        }
        if r > options.max_parent_visits as u64 {
            return Ok(None);
        }
        let Some(parents) = self.active_parents(id, options.max_parent_visits) else {
            return Ok(None);
        };
        let mut arity_delta = -(var_bytes(self.arities[id]) as f64 * 8.0);
        for &(parent, count) in &parents {
            let old = self.arities[parent as usize];
            let extra = count
                .checked_mul(self.arities[id] - 1)
                .ok_or("definition arity overflow")?;
            let new = old.checked_add(extra).ok_or("definition arity overflow")?;
            arity_delta += (var_bytes(new) as f64 - var_bytes(old) as f64) * 8.0;
        }
        let children = if cheap {
            Vec::new()
        } else {
            let Some(children) = self.children(id, options.max_children) else {
                return Ok(None);
            };
            multiplicities(children)
        };
        let mut sum_delta = -xlog(u128::from(f));
        let mut total = self.total - u128::from(f);
        for &(child, m) in &children {
            let old = self.data[child as usize]
                .checked_add(self.refs[child as usize])
                .ok_or("combined frequency exceeds u64")?;
            let extra = (f - 1)
                .checked_mul(m)
                .ok_or("combined frequency exceeds u64")?;
            let new = old
                .checked_add(extra)
                .ok_or("combined frequency exceeds u64")?;
            sum_delta += xlog(u128::from(new)) - xlog(u128::from(old));
            total += u128::from(extra);
        }
        let delta = xlog(total) - xlog(self.total) - sum_delta + arity_delta;
        Ok(Some(Proposal {
            id,
            parents,
            children,
            total,
            sum_delta,
            gain: -delta,
            arity_delta,
        }))
    }

    fn code_costs(&self, model: CostModel) -> Result<(Vec<f64>, usize), String> {
        let weights = self.weights()?;
        let mut costs = vec![0.0; weights.len()];
        if self.total == 0 {
            return Ok((costs, 0));
        }
        let book = Codebook::from_counts(&weights)?;
        for (id, length) in book.lengths() {
            costs[id as usize] = match model {
                CostModel::HuffmanCorrected => {
                    f64::from(length) - (self.total as f64 / weights[id as usize] as f64).log2()
                }
                _ => f64::from(length),
            };
        }
        Ok((costs, book.symbols.len()))
    }

    fn code_gain(&self, proposal: &Proposal, model: CostModel, costs: &[f64]) -> f64 {
        if model == CostModel::Shannon {
            return proposal.gain;
        }
        let f = self.data[proposal.id] + self.refs[proposal.id];
        let mut change = -(f as f64) * costs[proposal.id];
        for &(child, multiplicity) in &proposal.children {
            change +=
                ((u128::from(f - 1) * u128::from(multiplicity)) as f64) * costs[child as usize];
        }
        match model {
            CostModel::HuffmanLocal => -change - proposal.arity_delta,
            CostModel::HuffmanCorrected => proposal.gain - change,
            CostModel::Shannon => unreachable!(),
        }
    }
    fn apply(&mut self, p: Proposal) -> Result<(), String> {
        let (d, r) = (self.data[p.id], self.refs[p.id]);
        for (child, m) in p.children {
            let child = child as usize;
            add(
                &mut self.data[child],
                d.checked_mul(m).ok_or("data frequency overflow")?,
            )?;
            if r == 0 {
                self.refs[child] = self.refs[child]
                    .checked_sub(m)
                    .ok_or("missing recipe reference")?;
            } else {
                add(
                    &mut self.refs[child],
                    (r - 1).checked_mul(m).ok_or("recipe frequency overflow")?,
                )?;
            }
        }
        for (parent, m) in p.parents {
            let extra = m
                .checked_mul(self.arities[p.id] - 1)
                .ok_or("definition arity overflow")?;
            add(&mut self.arities[parent as usize], extra)?;
        }
        self.data[p.id] = 0;
        self.refs[p.id] = 0;
        self.retired[p.id] = true;
        self.total = p.total;
        self.sum += p.sum_delta;
        Ok(())
    }
    fn single_helpers(&mut self, options: &Options) -> Result<usize, String> {
        let mut accepted = 0;
        // Creation order keeps live parents ahead, avoiding long walks through
        // retired ancestors. The special case never flattens a helper's RHS.
        for id in self.basis..self.data.len() {
            if self.data[id] == 0 && self.refs[id] == 1 {
                if let Some(p) = self.proposal(id, options)? {
                    if p.gain > 0.0 {
                        self.apply(p)?;
                        accepted += 1;
                    }
                }
            }
        }
        Ok(accepted)
    }
    fn prune(&mut self, options: &Options) -> Result<(usize, usize), String> {
        let mut order: Vec<_> = (self.basis..self.data.len())
            .filter(|&id| self.live(id))
            .map(|id| id as u32)
            .collect();
        // Retain only IDs, instead of a padded (u128, usize) per live symbol.
        // Sorting observes exactly the same combined-frequency/ID order.
        order.sort_unstable_by_key(|&id| {
            let id = id as usize;
            (self.data[id] as u128 + self.refs[id] as u128, id)
        });
        let (mut tried, mut accepted) = (0, 0);
        let mut arity = if options.huffman_arity_cost {
            Some(arity_cost::Costs::new(self, options.rule_lengths)?)
        } else {
            None
        };
        let (mut costs, mut next_refit) =
            if options.cost_model == CostModel::Shannon && arity.is_none() {
                (Vec::new(), usize::MAX)
            } else {
                let (costs, leaves) = if options.cost_model == CostModel::Shannon {
                    (
                        Vec::new(),
                        self.data
                            .iter()
                            .zip(&self.refs)
                            .filter(|(d, r)| **d != 0 || **r != 0)
                            .count(),
                    )
                } else {
                    self.code_costs(options.cost_model)?
                };
                (costs, (leaves / options.refit_slices.max(1)).max(1))
            };
        for id in order {
            let id = id as usize;
            if let Some(mut p) = self.proposal(id, options)? {
                tried += 1;
                let descriptor_change = arity.as_ref().map(|a| a.change(self, &p)).transpose()?;
                if let Some(change) = &descriptor_change {
                    p.gain += p.arity_delta - change.delta;
                    p.arity_delta = change.delta;
                }
                if self.code_gain(&p, options.cost_model, &costs) >= options.minimum_gain_bits {
                    if let Some(change) = descriptor_change {
                        arity.as_mut().unwrap().apply(change)?;
                    }
                    self.apply(p)?;
                    accepted += 1;
                    if accepted >= next_refit {
                        let (updated, leaves) = if options.cost_model == CostModel::Shannon {
                            (
                                Vec::new(),
                                self.data
                                    .iter()
                                    .zip(&self.refs)
                                    .filter(|(d, r)| **d != 0 || **r != 0)
                                    .count(),
                            )
                        } else {
                            self.code_costs(options.cost_model)?
                        };
                        costs = updated;
                        if arity.is_some() {
                            arity = Some(arity_cost::Costs::new(self, options.rule_lengths)?);
                        }
                        next_refit =
                            accepted.saturating_add((leaves / options.refit_slices.max(1)).max(1));
                    }
                }
            }
        }
        Ok((tried, accepted))
    }
    fn weights(&self) -> Result<Vec<u64>, String> {
        self.data
            .iter()
            .zip(&self.refs)
            .map(|(&d, &r)| {
                d.checked_add(r)
                    .ok_or_else(|| "combined frequency exceeds u64".into())
            })
            .collect()
    }
    fn roots(
        &self,
        id: u32,
        stack: &mut Vec<u32>,
        mut emit: impl FnMut(u32) -> Result<(), String>,
    ) -> Result<(), String> {
        stack.push(id);
        while let Some(id) = stack.pop() {
            if id as usize >= self.basis && self.retired[id as usize] {
                let (a, b) = self.rules[id as usize - self.basis];
                stack.push(b);
                stack.push(a);
            } else {
                emit(id)?;
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct Stats {
    pub archive_bytes: u64,
    pub dictionary_bytes: u64,
    pub codebook_bytes: u64,
    pub descriptor_bytes: u64,
    pub huffman_rule_lengths: bool,
    pub run_rule_lengths: bool,
    pub reference_bits: u64,
    pub references: u64,
    pub payload_bits: u64,
    pub leaves: usize,
    pub rules: usize,
    pub nonbinary_rules: usize,
    pub totals: Totals,
    pub blocks: usize,
}
impl Stats {
    pub fn format(&self) -> &'static str {
        if self.run_rule_lengths {
            "BRHUFV03"
        } else if self.huffman_rule_lengths {
            "BRHUFV02"
        } else {
            "BRHUFV01"
        }
    }
}

struct Prepared {
    bytes: Vec<u8>,
    book: Option<Codebook>,
    stats: Stats,
}

fn prepare(
    state: &State,
    dictionary: &Dictionary,
    coding: RuleLengths,
) -> Result<Prepared, String> {
    let weights = state.weights()?;
    let book = if weights.iter().any(|&f| f != 0) {
        Some(Codebook::from_counts(&weights)?)
    } else {
        None
    };
    let mut table = Vec::new();
    if let Some(book) = &book {
        let mut hist = [0u64; 57];
        for (_, length) in book.lengths() {
            hist[length as usize] += 1;
        }
        let min = (1..hist.len()).find(|&i| hist[i] != 0).unwrap();
        table.extend([min as u8, book.max_bits]);
        for &n in &hist[min..=book.max_bits as usize] {
            put_var(&mut table, n);
        }
    }
    let mut descriptors = Vec::new();
    let mut references = Vec::new();
    let mut observed = vec![0u64; weights.len()];
    let (mut rules, mut nonbinary) = (0, 0);
    if let Some(book) = &book {
        for &id in &book.symbols {
            if (id as usize) < state.basis {
                put_var(&mut descriptors, 0);
                put_var(&mut descriptors, u64::from(id));
            } else {
                let rhs = state
                    .children(id as usize, usize::MAX)
                    .ok_or("definition size overflow")?;
                if rhs.len() as u64 != state.arities[id as usize] {
                    return Err("maintained definition arity differs".into());
                }
                rules += 1;
                nonbinary += usize::from(rhs.len() != 2);
                put_var(&mut descriptors, rhs.len() as u64);
                for &child in &rhs {
                    add(&mut observed[child as usize], 1)?;
                }
                references.extend(rhs);
            }
        }
    }
    if observed != state.refs {
        return Err("maintained dictionary reference counts differ".into());
    }
    let compressed = if coding == RuleLengths::Auto {
        descriptors::compress(
            &descriptors,
            book.as_ref().map_or(0, |b| b.symbols.len() as u64),
        )?
    } else {
        None
    };
    let huffman_rule_lengths = compressed.is_some();
    let descriptors = compressed.unwrap_or(descriptors);
    let (encoded, bits) = match &book {
        Some(book) => book.encode(&references)?,
        None => (Vec::new(), 0),
    };
    let mut bytes = Vec::new();
    if dictionary.ai_tokens {
        let model = dictionary.model.to_bytes();
        put_var(&mut bytes, model.len() as u64);
        bytes.extend(model);
    } else {
        put_var(&mut bytes, 0);
    }
    bytes.extend(&table);
    bytes.extend(&descriptors);
    bytes.extend(encoded);
    let stats = Stats {
        archive_bytes: 0,
        dictionary_bytes: bytes.len() as u64,
        codebook_bytes: table.len() as u64,
        descriptor_bytes: descriptors.len() as u64,
        huffman_rule_lengths,
        run_rule_lengths: false,
        reference_bits: bits,
        references: references.len() as u64,
        payload_bits: 0,
        leaves: book.as_ref().map_or(0, |b| b.symbols.len()),
        rules,
        nonbinary_rules: nonbinary,
        totals: Totals::default(),
        blocks: 0,
    };
    Ok(Prepared { bytes, book, stats })
}

struct Writer<'a, W> {
    out: W,
    encoding: Option<Encoding>,
    dictionary: &'a Dictionary,
    ids: Vec<u32>,
    index: Vec<IndexEntry>,
    stats: Stats,
    ai: bool,
    dictionary_hash: u64,
    block_symbols: usize,
    ended: bool,
}
impl<'a, W: Write + Seek> Writer<'a, W> {
    fn new(
        mut out: W,
        prepared: Prepared,
        dictionary: &'a Dictionary,
        block_symbols: usize,
    ) -> Result<Self, String> {
        out.write_all(&[0; HEADER])
            .and_then(|_| out.write_all(&prepared.bytes))
            .map_err(error)?;
        Ok(Self {
            out,
            encoding: prepared.book.as_ref().map(Codebook::encoding),
            dictionary,
            ids: Vec::with_capacity(block_symbols),
            index: Vec::new(),
            stats: prepared.stats,
            ai: dictionary.ai_tokens,
            dictionary_hash: fingerprint(&prepared.bytes),
            block_symbols,
            ended: true,
        })
    }
    fn push(&mut self, id: u32) -> Result<(), String> {
        self.ids.push(id);
        if self.ids.len() == self.block_symbols {
            self.flush()?;
        }
        Ok(())
    }
    fn flush(&mut self) -> Result<(), String> {
        if self.ids.is_empty() {
            return Ok(());
        }
        let (payload, bits) = self
            .encoding
            .as_ref()
            .ok_or("missing shared codebook")?
            .encode(&self.ids)?;
        self.index.push(IndexEntry {
            offset: self.out.stream_position().map_err(error)?,
            before: self.stats.totals,
        });
        let mut header = [0u8; BLOCK_HEADER];
        header[..8].copy_from_slice(&(self.ids.len() as u64).to_le_bytes());
        header[8..16].copy_from_slice(&bits.to_le_bytes());
        let check = fingerprint(&payload);
        header[16..24].copy_from_slice(&check.to_le_bytes());
        self.out
            .write_all(&header)
            .and_then(|_| self.out.write_all(&payload))
            .map_err(error)?;
        add(&mut self.stats.payload_bits, bits)?;
        for &id in &self.ids {
            let e = self.dictionary.expansions[id as usize];
            totals_add(&mut self.stats.totals, e)?;
            self.ended = e.ends_field;
        }
        self.ids.clear();
        Ok(())
    }
    fn finish(mut self, expected: Totals) -> Result<Stats, String> {
        self.flush()?;
        if (
            self.stats.totals.bytes,
            self.stats.totals.fields,
            self.stats.totals.tokens,
        ) != (expected.bytes, expected.fields, expected.tokens)
            || !self.ended
        {
            return Err("rewritten shared archive totals differ".into());
        }
        let offset = self.out.stream_position().map_err(error)?;
        let mut index = INDEX_MAGIC.to_vec();
        for e in &self.index {
            for n in [
                e.offset,
                e.before.bytes,
                e.before.fields,
                e.before.tokens,
                e.before.symbols,
            ] {
                index.extend(n.to_le_bytes());
            }
        }
        index.extend(fingerprint(&index).to_le_bytes());
        self.out.write_all(&index).map_err(error)?;
        self.stats.blocks = self.index.len();
        self.stats.archive_bytes = self.out.stream_position().map_err(error)?;
        let mut h = [0u8; HEADER];
        h[..8].copy_from_slice(self.stats.format().as_bytes());
        for (at, n) in [
            (8, u64::from(self.ai)),
            (16, self.stats.totals.bytes),
            (24, self.stats.totals.fields),
            (32, self.stats.totals.tokens),
            (40, self.stats.totals.symbols),
            (48, self.stats.dictionary_bytes),
            (56, self.stats.reference_bits),
            (64, self.stats.references),
            (72, self.stats.descriptor_bytes),
            (80, self.stats.codebook_bytes),
            (88, self.stats.leaves as u64),
            (96, self.stats.blocks as u64),
            (104, offset),
            (112, self.dictionary_hash),
        ] {
            h[at..at + 8].copy_from_slice(&n.to_le_bytes());
        }
        let check = fingerprint(&h[..120]);
        h[120..128].copy_from_slice(&check.to_le_bytes());
        self.out
            .seek(SeekFrom::Start(0))
            .and_then(|_| self.out.write_all(&h))
            .and_then(|_| self.out.flush())
            .map_err(error)?;
        Ok(self.stats)
    }
}

fn checkpoint(
    source: &Path,
    output: &Path,
    state: &State,
    block: usize,
    coding: RuleLengths,
) -> Result<Stats, String> {
    let reader = SourceReader::new(BufReader::new(File::open(source).map_err(error)?))?;
    let (dictionary, totals, mut source) = reader.into_source();
    let prepared = prepare(state, &dictionary, coding)?;
    let output = crate::atomic_output::Output::new(output)?;
    let mut writer = Writer::new(
        BufWriter::with_capacity(1 << 20, File::create_new(output.path()).map_err(error)?),
        prepared,
        &dictionary,
        block,
    )?;
    let mut remaining = state.data.clone();
    let mut stack = Vec::new();
    let mut raw = EncodedBlock::default();
    for i in 0..source.len() {
        source.read(i, &mut raw)?;
        let book = raw.codebook(dictionary.expansions.len())?;
        book.decode(&raw.payload, raw.bits, raw.totals.symbols, |id| {
            state.roots(id, &mut stack, |root| {
                remaining[root as usize] = remaining[root as usize]
                    .checked_sub(1)
                    .ok_or("maintained data frequency differs")?;
                writer.push(root)
            })
        })?;
    }
    if remaining.iter().any(|&n| n != 0) {
        return Err("maintained data frequency differs".into());
    }
    let stats = writer.finish(totals)?;
    output.commit()?;
    Ok(stats)
}

#[derive(Debug)]
pub struct Variant {
    pub name: String,
    pub stats: Stats,
    pub candidates: usize,
    pub removed: usize,
    pub seconds: f64,
}
pub struct Report {
    pub original_bytes: u64,
    pub selected: String,
    pub variants: Vec<Variant>,
}

pub fn optimize(source: &Path, directory: &Path, options: Options) -> Result<Report, String> {
    Ok(optimize_inner(source, directory, options, false)?.0)
}

/// Optimize normally and inspect the smallest emitted checkpoint without
/// changing any decisions. The audit is absent if the original source wins.
pub fn optimize_audited(
    source: &Path,
    directory: &Path,
    options: Options,
) -> Result<(Report, Option<PruningAudit>), String> {
    optimize_inner(source, directory, options, true)
}

fn optimize_inner(
    source: &Path,
    directory: &Path,
    options: Options,
    audit_pruning: bool,
) -> Result<(Report, Option<PruningAudit>), String> {
    if options.block_symbols == 0
        || options.block_symbols > 1 << 24
        || options.max_children < 2
        || options.max_parent_visits == 0
        || !options.minimum_gain_bits.is_finite()
        || options.minimum_gain_bits < 0.0
        || options.refit_slices == 0
        || (options.huffman_arity_cost && options.rule_lengths != RuleLengths::Auto)
        || (options.cost_model != CostModel::Shannon && options.rounds <= options.shannon_warmup)
    {
        return Err("invalid shared Huffman options".into());
    }
    let mut reader = SourceReader::new(BufReader::new(File::open(source).map_err(error)?))?;
    let mut data = vec![0u64; reader.dictionary.expansions.len()];
    let mut totals = Totals::default();
    for i in 0..reader.index.len() {
        let b = reader.block(i)?;
        let mut actual = Totals::default();
        b.book.decode(&b.payload, b.bits, b.totals.symbols, |id| {
            add(&mut data[id as usize], 1)?;
            totals_add(&mut actual, reader.dictionary.expansions[id as usize])
        })?;
        if actual != b.totals {
            return Err("source Huffman block totals differ".into());
        }
        add(&mut totals.bytes, actual.bytes)?;
        add(&mut totals.fields, actual.fields)?;
        add(&mut totals.tokens, actual.tokens)?;
        add(&mut totals.symbols, actual.symbols)?;
    }
    if totals != reader.totals {
        return Err("source Huffman totals differ".into());
    }
    let mut state = State::new(&reader.dictionary, data)?;
    drop(reader);
    fs::create_dir(directory).map_err(error)?;
    let original_bytes = fs::metadata(source).map_err(error)?.len();
    let mut report = Report {
        original_bytes,
        selected: "original".into(),
        variants: Vec::new(),
    };
    let mut best = original_bytes;
    let mut best_audit = None;
    for step in 0..options.rounds + 2 {
        let started = Instant::now();
        let (name, candidates, removed) = match step {
            0 => ("binary-shared".to_owned(), 0, 0),
            1 => {
                let n = state.single_helpers(&options)?;
                ("inlined-shared".to_owned(), n, n)
            }
            _ => {
                let mut step_options = options;
                if step - 1 <= options.shannon_warmup {
                    step_options.cost_model = CostModel::Shannon;
                }
                let (tried, n) = state.prune(&step_options)?;
                (format!("entropy-{}", step - 1), tried, n)
            }
        };
        let stats = checkpoint(
            source,
            &directory.join(format!("{name}.brhv")),
            &state,
            options.block_symbols,
            options.rule_lengths,
        )?;
        eprintln!(
            "{name}: {} bytes, {} rules, {removed} removed, {:.3}s",
            stats.archive_bytes,
            stats.rules,
            started.elapsed().as_secs_f64()
        );
        if stats.archive_bytes < best {
            best = stats.archive_bytes;
            report.selected = name.clone();
            if audit_pruning {
                best_audit = Some(state.audit(&options));
            }
        }
        report.variants.push(Variant {
            name,
            stats,
            candidates,
            removed,
            seconds: started.elapsed().as_secs_f64(),
        });
        if step >= 2
            && removed == 0
            && (options.cost_model == CostModel::Shannon || step - 1 > options.shannon_warmup)
        {
            break;
        }
    }
    Ok((report, best_audit))
}

struct Graph {
    model: Model,
    terminal: Vec<u32>,
    offsets: Vec<usize>,
    children: Vec<u32>,
    expansions: Vec<Expansion>,
    ai_tokens: bool,
}
impl DecodeGraph for Graph {
    fn model(&self) -> &Model {
        &self.model
    }
    fn expansions(&self) -> &[Expansion] {
        &self.expansions
    }
    fn ai_tokens(&self) -> bool {
        self.ai_tokens
    }
    fn has_rules(&self) -> bool {
        !self.children.is_empty()
    }
    fn terminal(&self, id: usize) -> Option<u32> {
        (self.terminal[id] != NONE).then_some(self.terminal[id])
    }
    fn children(&self, id: usize) -> impl DoubleEndedIterator<Item = u32> + ExactSizeIterator {
        self.children[self.offsets[id]..self.offsets[id + 1]]
            .iter()
            .copied()
    }
}
impl Graph {
    fn resolve(&mut self) -> Result<(), String> {
        let n = self.terminal.len();
        let mut colors = vec![0u8; n];
        let mut stack = Vec::new();
        for root in 0..n {
            if colors[root] == 2 {
                continue;
            }
            stack.push((root, self.offsets[root]));
            colors[root] = 1;
            while let Some((id, next)) = stack.last_mut() {
                if self.terminal[*id] != NONE {
                    let token = &self.model.tokens[self.terminal[*id] as usize];
                    self.expansions[*id] = Expansion {
                        bytes: token.bytes.len() as u64,
                        terminals: 1,
                        ends_field: token.ends_field,
                    };
                    colors[*id] = 2;
                    stack.pop();
                    continue;
                }
                if *next < self.offsets[*id + 1] {
                    let child = self.children[*next] as usize;
                    *next += 1;
                    match colors[child] {
                        1 => return Err("cyclic shared definition".into()),
                        0 => {
                            colors[child] = 1;
                            stack.push((child, self.offsets[child]));
                        }
                        _ => (),
                    }
                } else {
                    let id = *id;
                    let mut e = Expansion {
                        bytes: 0,
                        terminals: 0,
                        ends_field: false,
                    };
                    for &child in &self.children[self.offsets[id]..self.offsets[id + 1]] {
                        if e.ends_field {
                            return Err("shared definition crosses EOF".into());
                        }
                        let c = self.expansions[child as usize];
                        add(&mut e.bytes, c.bytes)?;
                        add(&mut e.terminals, c.terminals)?;
                        e.ends_field = c.ends_field;
                    }
                    self.expansions[id] = e;
                    colors[id] = 2;
                    stack.pop();
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct MetadataSpec {
    ai: bool,
    huffman_rule_lengths: bool,
    run_rule_lengths: bool,
    bits: u64,
    refs: u64,
    desc: u64,
    table: u64,
    leaves: u64,
}

fn parse_metadata(
    metadata: &[u8],
    spec: &MetadataSpec,
) -> Result<(Graph, Option<Codebook>, usize, usize), String> {
    let MetadataSpec {
        ai,
        huffman_rule_lengths,
        run_rule_lengths,
        bits,
        refs,
        desc,
        table,
        leaves,
    } = *spec;
    if leaves > u64::from(u32::MAX)
        || refs > bits
        || leaves
            > if run_rule_lengths {
                (metadata.len() as u64).saturating_mul(8)
            } else if huffman_rule_lengths {
                desc.saturating_mul(8)
            } else {
                desc
            }
    {
        return Err("invalid shared metadata sizes".into());
    }
    let mut remaining = metadata;
    let model_len = usize::try_from(get_var(&mut remaining)?).map_err(|_| "AI model too large")?;
    if (model_len != 0) != ai || model_len > remaining.len() {
        return Err("invalid shared AI model".into());
    }
    let model = if ai {
        Model::from_bytes(&remaining[..model_len])?
    } else {
        byte_model()
    };
    remaining = &remaining[model_len..];
    if table
        .checked_add(desc)
        .and_then(|n| n.checked_add(bits.div_ceil(8)))
        != Some(remaining.len() as u64)
    {
        return Err("shared dictionary sizes differ".into());
    }
    let mut hist = &remaining[..table as usize];
    let mut entries = Vec::new();
    if leaves != 0 {
        if hist.len() < 2
            || hist[0] == 0
            || hist[1] < hist[0]
            || hist[1] as usize > crate::huffman::MAX_BITS
        {
            return Err("invalid shared canonical lengths".into());
        }
        let (min, max) = (hist[0], hist[1]);
        hist = &hist[2..];
        for length in min..=max {
            let count = get_var(&mut hist)?;
            if count > leaves - entries.len() as u64 {
                return Err("shared leaf count differs".into());
            }
            for _ in 0..count {
                entries.push((entries.len() as u32, length));
            }
        }
    }
    if !hist.is_empty() || entries.len() as u64 != leaves {
        return Err("shared leaf count differs".into());
    }
    let book = if leaves == 0 {
        None
    } else {
        Some(Codebook::from_lengths(&entries)?)
    };
    let records = &remaining[table as usize..(table + desc) as usize];
    let mut terminal = Vec::with_capacity(leaves as usize);
    let mut offsets = vec![0usize];
    let mut literals = vec![false; model.tokens.len()];
    let mut rules = 0;
    let mut nonbinary = 0;
    let mut emit_descriptor = |arity: u64, literal: Option<u64>| {
        if arity == 1 || arity > refs {
            return Err("invalid shared definition arity".into());
        }
        if arity == 0 {
            let id = usize::try_from(literal.ok_or("missing literal ID")?)
                .map_err(|_| "literal ID overflow")?;
            if id >= literals.len() || std::mem::replace(&mut literals[id], true) {
                return Err("undefined or duplicate shared literal".into());
            }
            terminal.push(id as u32);
        } else {
            terminal.push(NONE);
            rules += 1;
            nonbinary += usize::from(arity != 2);
        }
        offsets.push(
            offsets
                .last()
                .unwrap()
                .checked_add(usize::try_from(arity).map_err(|_| "arity overflow")?)
                .ok_or("reference count overflow")?,
        );
        if *offsets.last().unwrap() as u64 > refs {
            return Err("shared reference count differs".into());
        }
        Ok(())
    };
    if run_rule_lengths {
        arity_runs::visit(records, leaves, refs, &mut emit_descriptor)?;
    } else if huffman_rule_lengths {
        descriptors::visit_huffman(records, leaves, refs, &mut emit_descriptor)?;
    } else {
        descriptors::visit_raw(records, leaves, &mut emit_descriptor)?;
    }
    if *offsets.last().unwrap() as u64 != refs {
        return Err("shared descriptors differ".into());
    }
    let mut children = Vec::with_capacity(refs as usize);
    if let Some(book) = &book {
        book.decode_ranks(&remaining[(table + desc) as usize..], bits, refs, |rank| {
            children.push(rank);
            Ok(())
        })?;
    } else if bits != 0 || refs != 0 {
        return Err("empty shared dictionary has references".into());
    }
    let mut graph = Graph {
        model,
        ai_tokens: ai,
        terminal,
        offsets,
        children,
        expansions: vec![
            Expansion {
                bytes: 0,
                terminals: 0,
                ends_field: false
            };
            leaves as usize
        ],
    };
    graph.resolve()?;
    Ok((graph, book, rules, nonbinary))
}

pub struct Reader<R> {
    input: R,
    graph: Graph,
    book: Option<Arc<Codebook>>,
    pub stats: Stats,
    pub ai_tokens: bool,
    index: Vec<IndexEntry>,
    index_offset: u64,
}
impl<R: Read + Seek> Reader<R> {
    pub fn static_codebook(&self) -> Option<&Codebook> {
        self.book.as_deref()
    }
    pub fn new(mut input: R) -> Result<Self, String> {
        let size = input.seek(SeekFrom::End(0)).map_err(error)?;
        input.seek(SeekFrom::Start(0)).map_err(error)?;
        let h = read_array::<HEADER>(&mut input)?;
        let run_rule_lengths = &h[..8] == RUN_LENGTHS_MAGIC;
        let huffman_rule_lengths = run_rule_lengths || &h[..8] == HUFFMAN_LENGTHS_MAGIC;
        if (&h[..8] != MAGIC && !huffman_rule_lengths)
            || number(&h, 8) > 1
            || fingerprint(&h[..120]) != number(&h, 120)
        {
            return Err("invalid shared Huffman header".into());
        }
        let ai = number(&h, 8) != 0;
        let (dlen, bits, refs, desc, table, leaves, blocks, index_offset) = (
            number(&h, 48),
            number(&h, 56),
            number(&h, 64),
            number(&h, 72),
            number(&h, 80),
            number(&h, 88),
            number(&h, 96),
            number(&h, 104),
        );
        let metadata_end = (HEADER as u64)
            .checked_add(dlen)
            .ok_or("dictionary overflow")?;
        if metadata_end > index_offset
            || index_offset > size
            || leaves > u64::from(u32::MAX)
            || leaves
                > if run_rule_lengths {
                    dlen.saturating_mul(8)
                } else if huffman_rule_lengths {
                    desc.saturating_mul(8)
                } else {
                    desc
                }
            || refs > bits
            || desc
                .checked_add(table)
                .and_then(|n| n.checked_add(bits.div_ceil(8)))
                .is_none_or(|n| n >= dlen)
            || blocks
                .checked_mul(40)
                .and_then(|n| n.checked_add(16))
                .and_then(|n| n.checked_add(index_offset))
                != Some(size)
            || (leaves == 0) != (number(&h, 40) == 0)
            || (blocks == 0) != (number(&h, 40) == 0)
        {
            return Err("invalid shared Huffman sizes".into());
        }
        let mut metadata = vec![0; usize::try_from(dlen).map_err(|_| "dictionary too large")?];
        input.read_exact(&mut metadata).map_err(error)?;
        if fingerprint(&metadata) != number(&h, 112) {
            return Err("shared dictionary checksum differs".into());
        }
        let (graph, book, rules, nonbinary) = parse_metadata(
            &metadata,
            &MetadataSpec {
                ai,
                huffman_rule_lengths,
                run_rule_lengths,
                bits,
                refs,
                desc,
                table,
                leaves,
            },
        )?;
        input.seek(SeekFrom::Start(index_offset)).map_err(error)?;
        let mut index_bytes = vec![0; (size - index_offset) as usize];
        input.read_exact(&mut index_bytes).map_err(error)?;
        if &index_bytes[..8] != INDEX_MAGIC
            || fingerprint(&index_bytes[..index_bytes.len() - 8])
                != number(&index_bytes, index_bytes.len() - 8)
        {
            return Err("shared index checksum differs".into());
        }
        let mut index = Vec::with_capacity(blocks as usize);
        for b in index_bytes[8..index_bytes.len() - 8].chunks_exact(40) {
            let e = IndexEntry {
                offset: number(b, 0),
                before: Totals {
                    bytes: number(b, 8),
                    fields: number(b, 16),
                    tokens: number(b, 24),
                    symbols: number(b, 32),
                },
            };
            if e.offset < metadata_end
                || e.offset >= index_offset
                || index
                    .last()
                    .is_some_and(|p: &IndexEntry| p.offset >= e.offset)
            {
                return Err("invalid shared block index".into());
            }
            index.push(e);
        }
        if index
            .first()
            .is_some_and(|e| e.offset != metadata_end || e.before != Totals::default())
            || (index.is_empty() && index_offset != metadata_end)
        {
            return Err("shared first block differs".into());
        }
        let mut payload_bits = 0;
        for i in 0..index.len() {
            input
                .seek(SeekFrom::Start(index[i].offset))
                .map_err(error)?;
            let header = read_array::<BLOCK_HEADER>(&mut input)?;
            let (symbols, bits) = (number(&header, 0), number(&header, 8));
            let next = index.get(i + 1).map_or(index_offset, |e| e.offset);
            if symbols == 0
                || symbols > 1 << 24
                || symbols > bits
                || index[i]
                    .offset
                    .checked_add(BLOCK_HEADER as u64)
                    .and_then(|n| n.checked_add(bits.div_ceil(8)))
                    != Some(next)
            {
                return Err("invalid shared data block".into());
            }
            add(&mut payload_bits, bits)?;
        }
        let stats = Stats {
            archive_bytes: size,
            dictionary_bytes: dlen,
            codebook_bytes: table,
            descriptor_bytes: desc,
            huffman_rule_lengths,
            run_rule_lengths,
            reference_bits: bits,
            references: refs,
            payload_bits,
            leaves: leaves as usize,
            rules,
            nonbinary_rules: nonbinary,
            totals: Totals {
                bytes: number(&h, 16),
                fields: number(&h, 24),
                tokens: number(&h, 32),
                symbols: number(&h, 40),
            },
            blocks: blocks as usize,
        };
        Ok(Self {
            input,
            graph,
            book: book.map(Arc::new),
            stats,
            ai_tokens: ai,
            index,
            index_offset,
        })
    }
    pub fn terminal_vocabulary(&self) -> usize {
        self.graph.model.tokens.len()
    }

    pub fn prepare_decoder(
        &self,
        ids: bool,
        arena_budget: usize,
    ) -> Result<PreparedDecoder, String> {
        PreparedDecoder::from_graph(&self.graph, self.book.clone(), ids, arena_budget)
    }

    pub fn block(&mut self, index: usize) -> Result<Block, String> {
        let entry = *self
            .index
            .get(index)
            .ok_or("shared block index out of range")?;
        let next = self.index.get(index + 1).copied().unwrap_or(IndexEntry {
            offset: self.index_offset,
            before: self.stats.totals,
        });
        let mut raw = EncodedBlock::default();
        read_compact_encoded(
            &mut self.input,
            entry,
            next,
            self.stats.totals,
            self.book.as_ref(),
            &mut raw,
        )?;
        Ok(Block {
            book: raw.codebook(self.graph.expansions.len())?,
            payload: raw.payload,
            bits: raw.bits,
            totals: raw.totals,
        })
    }

    pub fn decode_block(
        &self,
        decoder: &PreparedDecoder,
        block: &Block,
        out: &mut impl Write,
    ) -> Result<bool, String> {
        decoder.decode_graph_block(&self.graph, block, out)
    }

    pub fn decode_to(&mut self, out: &mut impl Write, ids: bool) -> Result<Totals, String> {
        self.decode_to_with_arena(out, ids, 64 << 20)
    }
    pub fn decode_to_with_arena(
        &mut self,
        out: &mut impl Write,
        ids: bool,
        arena_bytes: usize,
    ) -> Result<Totals, String> {
        if ids && !self.ai_tokens {
            return Err("byte archive has no frozen AI token IDs".into());
        }
        let decoder = self.prepare_decoder(ids, arena_bytes)?;
        let mut totals = Totals::default();
        let mut ended = true;
        for i in 0..self.index.len() {
            let entry = self.index[i];
            if totals != entry.before {
                return Err("shared block index totals differ".into());
            }
            let block = self.block(i)?;
            ended = self.decode_block(&decoder, &block, out)?;
            totals.add(block.totals)?;
        }
        if totals != self.stats.totals || !ended {
            return Err("shared decoded totals differ".into());
        }
        out.flush().map_err(error)?;
        Ok(totals)
    }
}

/// Uses the production canonical decoder, reference buffers and output pipeline.
pub fn decode_pipeline<R: Read + Seek + Send>(
    reader: Reader<R>,
    token_ids: bool,
    out: &mut impl Write,
    options: crate::huffman_pipeline::Options,
) -> Result<crate::huffman_pipeline::Report, String> {
    if token_ids && !reader.ai_tokens {
        return Err("byte archive has no frozen AI token IDs".into());
    }
    let source = BlockSource::compact(
        reader.input,
        reader.index,
        reader.index_offset,
        reader.stats.totals,
        reader.graph.expansions.len(),
        reader.book,
    );
    crate::huffman_pipeline::decode_graph(
        reader.graph,
        reader.stats.totals,
        source,
        token_ids,
        out,
        options,
    )
}

pub fn decode_file(source: &Path, output: &Path, ids: bool) -> Result<Totals, String> {
    let reader = Reader::new(BufReader::with_capacity(
        1 << 20,
        File::open(source).map_err(error)?,
    ))?;
    let output = crate::atomic_output::Output::new(output)?;
    let mut out =
        BufWriter::with_capacity(1 << 20, File::create_new(output.path()).map_err(error)?);
    let totals = decode_pipeline(
        reader,
        ids,
        &mut out,
        crate::huffman_pipeline::Options::default(),
    )?
    .totals;
    out.flush().map_err(error)?;
    drop(out);
    output.commit()?;
    Ok(totals)
}

pub struct RepackReport {
    pub before: Stats,
    pub after: Stats,
}

/// Change only descriptor storage. Canonical codes, references, data blocks and
/// their checksums remain unchanged; absolute block/index offsets are adjusted.
pub fn repack_rule_lengths(
    source: &Path,
    destination: &Path,
    coding: RuleLengths,
) -> Result<RepackReport, String> {
    repack_descriptors(source, destination, coding, false)
}

/// Choose run-coded metadata only when it beats the existing representations.
/// The shared primary table and all payload/reference bits remain unchanged.
pub fn repack_arity_runs(source: &Path, destination: &Path) -> Result<RepackReport, String> {
    repack_descriptors(source, destination, RuleLengths::Auto, true)
}

fn repack_descriptors(
    source: &Path,
    destination: &Path,
    coding: RuleLengths,
    allow_runs: bool,
) -> Result<RepackReport, String> {
    let Reader {
        mut input,
        index,
        index_offset,
        stats: before,
        ..
    } = Reader::new(BufReader::with_capacity(
        1 << 20,
        File::open(source).map_err(error)?,
    ))?;
    input.seek(SeekFrom::Start(0)).map_err(error)?;
    let mut header = read_array::<HEADER>(&mut input)?;
    let mut metadata =
        vec![0; usize::try_from(before.dictionary_bytes).map_err(|_| "dictionary too large")?];
    input.read_exact(&mut metadata).map_err(error)?;
    if fingerprint(&metadata) != number(&header, 112) {
        return Err("dictionary changed during repack".into());
    }
    let mut prefix = metadata.as_slice();
    let model_len = usize::try_from(get_var(&mut prefix)?).map_err(|_| "AI model too large")?;
    let start = metadata.len() - prefix.len() + model_len + before.codebook_bytes as usize;
    let end = start + before.descriptor_bytes as usize;
    if end > metadata.len() {
        return Err("dictionary changed during repack".into());
    }
    let mut raw = Vec::new();
    let records = if before.run_rule_lengths {
        arity_runs::visit(
            &metadata[start..end],
            before.leaves as u64,
            before.references,
            |arity, literal| {
                put_var(&mut raw, arity);
                if let Some(id) = literal {
                    put_var(&mut raw, id);
                }
                Ok(())
            },
        )?;
        raw.as_slice()
    } else if before.huffman_rule_lengths {
        descriptors::visit_huffman(
            &metadata[start..end],
            before.leaves as u64,
            before.references,
            |arity, literal| {
                put_var(&mut raw, arity);
                if let Some(id) = literal {
                    put_var(&mut raw, id);
                }
                Ok(())
            },
        )?;
        raw.as_slice()
    } else {
        &metadata[start..end]
    };
    let compressed = if coding == RuleLengths::Auto {
        descriptors::compress(records, before.leaves as u64)?
    } else {
        None
    };
    let mut huffman_rule_lengths = compressed.is_some();
    let mut chosen = compressed.as_deref().unwrap_or(records);
    let run_candidate = if allow_runs {
        Some(arity_runs::compress(records, before.leaves as u64)?)
    } else {
        None
    };
    let mut run_rule_lengths = false;
    if let Some(bytes) = &run_candidate {
        if bytes.len() < chosen.len() {
            chosen = bytes;
            run_rule_lengths = true;
            huffman_rule_lengths = true;
        }
    }
    let mut replacement =
        Vec::with_capacity(metadata.len() - before.descriptor_bytes as usize + chosen.len());
    replacement.extend_from_slice(&metadata[..start]);
    replacement.extend_from_slice(chosen);
    replacement.extend_from_slice(&metadata[end..]);
    let mut after = before.clone();
    after.dictionary_bytes = replacement.len() as u64;
    after.descriptor_bytes = chosen.len() as u64;
    after.huffman_rule_lengths = huffman_rule_lengths;
    after.run_rule_lengths = run_rule_lengths;
    let old_end = HEADER as u64 + before.dictionary_bytes;
    let new_end = HEADER as u64 + after.dictionary_bytes;
    let new_index = new_end
        .checked_add(index_offset - old_end)
        .ok_or("repacked index overflow")?;
    after.archive_bytes = new_index
        .checked_add(before.archive_bytes - index_offset)
        .ok_or("repacked archive overflow")?;
    header[..8].copy_from_slice(after.format().as_bytes());
    for (at, n) in [
        (48, after.dictionary_bytes),
        (72, after.descriptor_bytes),
        (104, new_index),
        (112, fingerprint(&replacement)),
    ] {
        header[at..at + 8].copy_from_slice(&n.to_le_bytes());
    }
    let check = fingerprint(&header[..120]);
    header[120..].copy_from_slice(&check.to_le_bytes());
    let reservation = crate::atomic_output::Output::new(destination)?;
    let mut output = BufWriter::with_capacity(
        1 << 20,
        File::create_new(reservation.path()).map_err(error)?,
    );
    output
        .write_all(&header)
        .and_then(|_| output.write_all(&replacement))
        .map_err(error)?;
    input.seek(SeekFrom::Start(old_end)).map_err(error)?;
    let payload_len = index_offset - old_end;
    let copied = std::io::copy(&mut input.take(payload_len), &mut output).map_err(error)?;
    if copied != payload_len {
        return Err("truncated data during descriptor repack".into());
    }
    let mut bytes = INDEX_MAGIC.to_vec();
    for entry in index {
        let offset = new_end
            .checked_add(entry.offset - old_end)
            .ok_or("repacked block overflow")?;
        for n in [
            offset,
            entry.before.bytes,
            entry.before.fields,
            entry.before.tokens,
            entry.before.symbols,
        ] {
            bytes.extend_from_slice(&n.to_le_bytes());
        }
    }
    bytes.extend_from_slice(&fingerprint(&bytes).to_le_bytes());
    output
        .write_all(&bytes)
        .and_then(|_| output.flush())
        .map_err(error)?;
    if output.stream_position().map_err(error)? != after.archive_bytes {
        return Err("repacked size differs".into());
    }
    drop(output);
    reservation.commit()?;
    Ok(RepackReport { before, after })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn state(dictionary: &Dictionary, sequence: &[u32]) -> State {
        let mut counts = vec![0; dictionary.expansions.len()];
        for &id in sequence {
            counts[id as usize] += 1;
        }
        State::new(dictionary, counts).unwrap()
    }
    #[test]
    fn pruning_audit_reports_single_data_occurrence_blocked_by_child_limit() {
        let dictionary = Dictionary::new(byte_model(), vec![(97, 98), (257, 99)], false).unwrap();
        let mut s = state(&dictionary, &[258, 256]);
        let options = Options {
            max_children: 2,
            ..Options::default()
        };
        assert_eq!(s.single_helpers(&options).unwrap(), 1);
        assert!(s.proposal(258, &options).unwrap().is_none());
        let before_weights = s.weights().unwrap();
        let before_total = s.total;
        assert_eq!(
            s.audit(&options),
            PruningAudit {
                live_rules: 1,
                data_once_unreferenced_rules: 1,
                data_once_unreferenced_over_child_limit: 1,
                max_arity_data_once_unreferenced: 3,
                over_child_limit_rules: 1,
                ..PruningAudit::default()
            }
        );
        assert_eq!(s.weights().unwrap(), before_weights);
        assert_eq!(s.total, before_total);
        verify(&s, &dictionary, &[258, 256]);
    }
    #[test]
    fn local_huffman_gain_matches_fixed_code_lengths_and_exact_descriptor_delta() {
        let dictionary = Dictionary::new(
            byte_model(),
            vec![(97, 98), (257, 99), (258, 258), (259, 256)],
            false,
        )
        .unwrap();
        let mut sequence = vec![260; 17];
        sequence.extend([258, 258, 256, 97, 98, 256]);
        for id in 257..261 {
            let original = state(&dictionary, &sequence);
            let (costs, _) = original.code_costs(CostModel::HuffmanLocal).unwrap();
            let Some(proposal) = original.proposal(id, &Options::default()).unwrap() else {
                continue;
            };
            let predicted = original.code_gain(&proposal, CostModel::HuffmanLocal, &costs);
            let size = |s: &State| -> f64 {
                let bit_cost = s
                    .weights()
                    .unwrap()
                    .iter()
                    .zip(&costs)
                    .map(|(&f, &l)| f as f64 * l)
                    .sum::<f64>();
                bit_cost
                    + (s.basis..s.data.len())
                        .filter(|&id| s.live(id))
                        .map(|id| var_bytes(s.arities[id]) as f64 * 8.0)
                        .sum::<f64>()
            };
            let before = size(&original);
            let mut changed = state(&dictionary, &sequence);
            changed.apply(proposal).unwrap();
            assert!((predicted - (before - size(&changed))).abs() < 1e-9);
            verify(&changed, &dictionary, &sequence);
        }
    }
    #[test]
    fn huffman_polishing_preserves_exact_counts_bytes_and_field_endings() {
        let dictionary = Dictionary::new(
            byte_model(),
            vec![(97, 98), (257, 99), (258, 258), (259, 256)],
            false,
        )
        .unwrap();
        let mut sequence = vec![260; 31];
        sequence.extend([258, 258, 256, 97, 98, 256]);
        let original = state(&dictionary, &sequence);
        let mut baseline =
            Reader::new(Cursor::new(bytes(&dictionary, &original, &sequence, 7))).unwrap();
        let mut expected = Vec::new();
        baseline.decode_to(&mut expected, false).unwrap();
        for (model, huffman_arity_cost) in [
            (CostModel::HuffmanLocal, false),
            (CostModel::HuffmanCorrected, false),
            (CostModel::Shannon, true),
            (CostModel::HuffmanLocal, true),
            (CostModel::HuffmanCorrected, true),
        ] {
            let mut s = state(&dictionary, &sequence);
            let options = Options {
                cost_model: model,
                refit_slices: 16,
                rounds: 8,
                huffman_arity_cost,
                ..Default::default()
            };
            for _ in 0..4 {
                s.prune(&options).unwrap();
                verify(&s, &dictionary, &sequence);
            }
            let mut reader =
                Reader::new(Cursor::new(bytes(&dictionary, &s, &sequence, 7))).unwrap();
            let mut actual = Vec::new();
            reader.decode_to(&mut actual, false).unwrap();
            assert_eq!(actual, expected);
        }
        let s = state(&dictionary, &sequence);
        let p = s.proposal(258, &Options::default()).unwrap().unwrap();
        assert_eq!(
            s.code_gain(&p, CostModel::HuffmanCorrected, &vec![0.0; s.data.len()]),
            p.gain
        );
    }
    fn bytes(dictionary: &Dictionary, state: &State, sequence: &[u32], block: usize) -> Vec<u8> {
        let mut cursor = Cursor::new(Vec::new());
        let prepared = prepare(state, dictionary, RuleLengths::Auto).unwrap();
        let mut writer = Writer::new(&mut cursor, prepared, dictionary, block).unwrap();
        let mut expected = Totals::default();
        let mut stack = Vec::new();
        for &id in sequence {
            totals_add(&mut expected, dictionary.expansions[id as usize]).unwrap();
            state.roots(id, &mut stack, |id| writer.push(id)).unwrap();
        }
        writer.finish(expected).unwrap();
        cursor.into_inner()
    }
    fn verify(state: &State, dictionary: &Dictionary, sequence: &[u32]) {
        prepare(state, dictionary, RuleLengths::Auto).unwrap();
        let mut actual = vec![0; state.data.len()];
        let mut stack = Vec::new();
        for &id in sequence {
            state
                .roots(id, &mut stack, |id| {
                    actual[id as usize] += 1;
                    Ok(())
                })
                .unwrap();
        }
        assert_eq!(actual, state.data);
        let weights = state.weights().unwrap();
        assert_eq!(
            weights.iter().map(|&n| n as u128).sum::<u128>(),
            state.total
        );
        let sum: f64 = weights.iter().map(|&n| xlog(n as u128)).sum();
        assert!((state.sum - sum).abs() < 0.00001);
    }
    #[test]
    fn common_engine_handles_variable_arity_forward_references_and_partial_caches() {
        for ai in [false, true] {
            let dictionary = Dictionary::new(
                byte_model(),
                vec![(97, 98), (257, 99), (258, 100), (259, 256)],
                ai,
            )
            .unwrap();
            let sequence = vec![260; 37];
            let mut s = state(&dictionary, &sequence);
            s.single_helpers(&Options::default()).unwrap();
            let encoded = bytes(&dictionary, &s, &sequence, 3);
            let reader = Reader::new(Cursor::new(encoded.clone())).unwrap();
            assert_eq!(reader.stats.nonbinary_rules, 1);
            assert!(reader
                .graph
                .children
                .iter()
                .enumerate()
                .any(|(at, &child)| {
                    let parent = reader.graph.offsets.partition_point(|&offset| offset <= at) - 1;
                    child as usize > parent
                }));
            for ids in [false, true] {
                if ids && !ai {
                    continue;
                }
                let expected = if ids {
                    [97u32, 98, 99, 100, 256]
                        .repeat(37)
                        .into_iter()
                        .flat_map(u32::to_le_bytes)
                        .collect()
                } else {
                    b"abcd".repeat(37)
                };
                for budget in [0, 2, 16, 65536] {
                    let reader = Reader::new(Cursor::new(encoded.clone())).unwrap();
                    let mut output = Vec::new();
                    let options = crate::huffman_pipeline::Options {
                        arena_bytes: budget,
                        buffers_per_stage: 1,
                        references_per_buffer: 7,
                        output_buffer_bytes: 3,
                    };
                    let report = decode_pipeline(reader, ids, &mut output, options).unwrap();
                    assert_eq!(output, expected);
                    assert_eq!(report.totals.fields, 37);
                    let mut reader = Reader::new(Cursor::new(encoded.clone())).unwrap();
                    let decoder = reader.prepare_decoder(ids, budget).unwrap();
                    output.clear();
                    for index in 0..reader.stats.blocks {
                        let block = reader.block(index).unwrap();
                        reader.decode_block(&decoder, &block, &mut output).unwrap();
                    }
                    assert_eq!(output, expected);
                }
            }
        }
    }
    #[test]
    fn common_pipeline_handles_empty_archives_and_propagates_write_failures() {
        struct Reject;
        impl Write for Reject {
            fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
                Err(std::io::Error::other("injected output failure"))
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let dictionary = Dictionary::new(byte_model(), vec![], false).unwrap();
        for sequence in [vec![], vec![97, 256]] {
            let encoded = bytes(&dictionary, &state(&dictionary, &sequence), &sequence, 1);
            let reader = Reader::new(Cursor::new(encoded.clone())).unwrap();
            let mut out = Vec::new();
            decode_pipeline(reader, false, &mut out, Default::default()).unwrap();
            assert_eq!(
                out,
                if sequence.is_empty() {
                    &b""[..]
                } else {
                    &b"a"[..]
                }
            );
            if !sequence.is_empty() {
                let reader = Reader::new(Cursor::new(encoded)).unwrap();
                let err = decode_pipeline(reader, false, &mut Reject, Default::default())
                    .err()
                    .unwrap();
                assert!(err.contains("injected output failure"), "{err}");
            }
        }
    }
    #[test]
    fn single_helpers_inline_without_changing_data_or_child_weights() {
        let dictionary =
            Dictionary::new(byte_model(), vec![(97, 98), (257, 99), (258, 256)], false).unwrap();
        let sequence = vec![259; 20];
        let mut s = state(&dictionary, &sequence);
        let previous = s.refs.clone();
        assert_eq!(s.single_helpers(&Options::default()).unwrap(), 2);
        assert_eq!(s.data[259], 20);
        for id in [97, 98, 99, 256] {
            assert_eq!(s.refs[id], previous[id]);
        }
        assert_eq!(s.arities[259], 4);
        verify(&s, &dictionary, &sequence);
        let encoded = bytes(&dictionary, &s, &sequence, 3);
        let mut reader = Reader::new(Cursor::new(encoded)).unwrap();
        assert_eq!(reader.stats.nonbinary_rules, 1);
        assert_eq!(reader.stats.rules, 1);
        let mut out = Vec::new();
        reader.decode_to(&mut out, false).unwrap();
        assert_eq!(out, b"abc".repeat(20));
    }
    #[test]
    fn shared_uncoded_helper_gets_a_code_from_dictionary_references() {
        let dictionary = Dictionary::new(
            byte_model(),
            vec![(97, 98), (257, 99), (257, 100), (258, 256), (259, 256)],
            false,
        )
        .unwrap();
        let sequence = [vec![260; 30], vec![261; 30]].concat();
        let mut s = state(&dictionary, &sequence);
        s.single_helpers(&Options::default()).unwrap();
        assert!(!s.retired[257]);
        assert_eq!((s.data[257], s.refs[257]), (0, 2));
        assert!(prepare(&s, &dictionary, RuleLengths::Auto)
            .unwrap()
            .book
            .unwrap()
            .symbols
            .contains(&257));
        verify(&s, &dictionary, &sequence);
        let mut reader = Reader::new(Cursor::new(bytes(&dictionary, &s, &sequence, 7))).unwrap();
        let mut out = Vec::new();
        reader.decode_to(&mut out, false).unwrap();
        assert_eq!(out, [b"abc".repeat(30), b"abd".repeat(30)].concat());
    }
    #[test]
    fn positive_usage_counts_duplicate_children_and_definition_removal_exactly() {
        let dictionary = Dictionary::new(byte_model(), vec![(97, 97), (257, 256)], false).unwrap();
        for sequence in [vec![257, 257, 258], vec![257, 257, 257, 257, 256]] {
            let mut s = state(&dictionary, &sequence);
            let p = s.proposal(257, &Options::default()).unwrap().unwrap();
            let predicted = xlog(s.total) - s.sum - p.gain;
            let old_arity: usize = (257..s.data.len())
                .filter(|&id| s.live(id))
                .map(|id| var_bytes(s.arities[id]))
                .sum();
            s.apply(p).unwrap();
            let new_arity: usize = (257..s.data.len())
                .filter(|&id| s.live(id))
                .map(|id| var_bytes(s.arities[id]))
                .sum();
            let measured = xlog(s.total) - s.sum + (new_arity as f64 - old_arity as f64) * 8.0;
            assert!((predicted - measured).abs() < 1e-9);
            verify(&s, &dictionary, &sequence);
            let mut reader =
                Reader::new(Cursor::new(bytes(&dictionary, &s, &sequence, 2))).unwrap();
            let mut out = Vec::new();
            reader.decode_to(&mut out, false).unwrap();
            assert_eq!(out, vec![97; if sequence.contains(&258) { 6 } else { 8 }]);
        }
    }
    #[test]
    fn byte_alphabet_eof_empty_input_and_singleton_round_trip() {
        let dictionary = Dictionary::new(byte_model(), vec![], false).unwrap();
        for sequence in [Vec::new(), vec![256], (0..257).collect()] {
            let s = state(&dictionary, &sequence);
            let mut reader =
                Reader::new(Cursor::new(bytes(&dictionary, &s, &sequence, 17))).unwrap();
            let mut out = Vec::new();
            reader.decode_to(&mut out, false).unwrap();
            let expected: Vec<_> = sequence
                .iter()
                .filter(|&&id| id < 256)
                .map(|&id| id as u8)
                .collect();
            assert_eq!(out, expected);
        }
    }
    #[test]
    fn stable_ai_ids_survive_equal_bytes_with_different_token_boundaries() {
        let mut tokens = byte_model().tokens;
        tokens.push(crate::tokenizer::Token {
            bytes: b"ab".to_vec(),
            ends_field: false,
        });
        tokens.push(crate::tokenizer::Token {
            bytes: b"ab".to_vec(),
            ends_field: true,
        });
        let dictionary =
            Dictionary::new(Model::new(tokens).unwrap(), vec![(257, 256)], true).unwrap();
        let sequence = vec![258, 259];
        let s = state(&dictionary, &sequence);
        let encoded = bytes(&dictionary, &s, &sequence, 1);
        let mut reader = Reader::new(Cursor::new(encoded.clone())).unwrap();
        let mut out = Vec::new();
        reader.decode_to(&mut out, true).unwrap();
        let expected: Vec<_> = [258u32, 257, 256]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect();
        assert_eq!(out, expected);
        let mut reader = Reader::new(Cursor::new(encoded)).unwrap();
        out.clear();
        reader.decode_to(&mut out, false).unwrap();
        assert_eq!(out, b"abab");
    }
    #[test]
    fn cyclic_or_cross_field_definitions_are_rejected() {
        let empty = Expansion {
            bytes: 0,
            terminals: 0,
            ends_field: false,
        };
        let mut graph = Graph {
            ai_tokens: false,
            model: byte_model(),
            terminal: vec![NONE, 97],
            offsets: vec![0, 2, 2],
            children: vec![0, 1],
            expansions: vec![empty; 2],
        };
        assert!(graph.resolve().unwrap_err().contains("cyclic"));
        graph.terminal = vec![NONE, 256, 97];
        graph.offsets = vec![0, 2, 2, 2];
        graph.children = vec![1, 2];
        graph.expansions = vec![empty; 3];
        assert!(graph.resolve().unwrap_err().contains("EOF"));
    }
    #[test]
    fn corrupt_and_truncated_archives_fail_validation() {
        let dictionary = Dictionary::new(byte_model(), vec![(97, 256)], false).unwrap();
        let sequence = vec![257; 9];
        let s = state(&dictionary, &sequence);
        let encoded = bytes(&dictionary, &s, &sequence, 3);
        for len in [0, 8, 127, 128, encoded.len() - 1] {
            assert!(Reader::new(Cursor::new(encoded[..len].to_vec())).is_err());
        }
        for at in [8, HEADER, encoded.len() - 1] {
            let mut corrupt = encoded.clone();
            corrupt[at] ^= 1;
            assert!(Reader::new(Cursor::new(corrupt)).is_err());
        }
        let reader = Reader::new(Cursor::new(encoded.clone())).unwrap();
        let offset = reader.index[0].offset as usize + BLOCK_HEADER;
        let mut corrupt = encoded;
        corrupt[offset] ^= 128;
        let mut reader = Reader::new(Cursor::new(corrupt)).unwrap();
        assert!(reader.decode_to(&mut Vec::new(), false).is_err());
    }
    #[test]
    fn declared_output_limit_is_checked_before_emitting_expansions() {
        let dictionary = Dictionary::new(byte_model(), vec![(97, 256)], false).unwrap();
        let sequence = vec![257; 9];
        let s = state(&dictionary, &sequence);
        let mut encoded = bytes(&dictionary, &s, &sequence, 3);
        encoded[16..24].copy_from_slice(&0u64.to_le_bytes());
        let checksum = fingerprint(&encoded[..120]);
        encoded[120..128].copy_from_slice(&checksum.to_le_bytes());
        let mut reader = Reader::new(Cursor::new(encoded)).unwrap();
        let mut out = Vec::new();
        assert!(reader.decode_to(&mut out, false).is_err());
        assert!(out.is_empty());
    }
    #[test]
    fn combined_counters_exceed_u32_and_parent_arity_byte_boundary_is_charged() {
        let dictionary = Dictionary::new(byte_model(), vec![(97, 98), (257, 256)], false).unwrap();
        let mut counts = vec![0; 260];
        counts.truncate(259);
        counts[258] = (1u64 << 34) + 7;
        let s = State::new(&dictionary, counts).unwrap();
        assert_eq!(s.data[258], (1u64 << 34) + 7);
        let p = s.proposal(257, &Options::default()).unwrap().unwrap();
        assert!(p.gain.is_finite());
        let mut boundary = state(&dictionary, &[258]);
        let base_gain = boundary
            .proposal(257, &Options::default())
            .unwrap()
            .unwrap()
            .gain;
        boundary.arities[258] = 127;
        let crossing = boundary
            .proposal(257, &Options::default())
            .unwrap()
            .unwrap()
            .gain;
        assert!((base_gain - crossing - 8.0).abs() < 1e-9);
    }
    #[test]
    fn successive_pruning_matches_recount_under_varied_graphs_and_orders() {
        for seed in 1..=12 {
            let mut random = seed as u64;
            let mut next = |limit: usize| {
                random ^= random << 13;
                random ^= random >> 7;
                random ^= random << 17;
                random as usize % limit
            };
            let mut rules = Vec::new();
            let mut available = vec![97u32, 98, 99, 100];
            let mut lengths = vec![1u64; 257];
            for i in 0..96 {
                let (a, b) = (
                    available[next(available.len())],
                    available[next(available.len())],
                );
                rules.push((a, b));
                lengths.push(lengths[a as usize] + lengths[b as usize]);
                if lengths[257 + i] < 32 {
                    available.push((257 + i) as u32);
                }
            }
            let dictionary = Dictionary::new(byte_model(), rules, false).unwrap();
            let mut sequence: Vec<_> = (0..64).map(|_| available[next(available.len())]).collect();
            sequence.push(256);
            let mut s = state(&dictionary, &sequence);
            let options = Options {
                max_children: 128,
                ..Options::default()
            };
            let mut order: Vec<_> = (257..s.data.len()).collect();
            if seed % 2 == 0 {
                order.reverse();
            }
            for id in order {
                if let Some(p) = s.proposal(id, &options).unwrap() {
                    // Exercise count updates even for intentionally unfavorable
                    // proposals; exact checkpoints protect actual selection.
                    s.apply(p).unwrap();
                    verify(&s, &dictionary, &sequence);
                }
            }
            let mut reader =
                Reader::new(Cursor::new(bytes(&dictionary, &s, &sequence, 11))).unwrap();
            let mut out = Vec::new();
            reader.decode_to(&mut out, false).unwrap();
            let original = state(&dictionary, &sequence);
            let mut reader =
                Reader::new(Cursor::new(bytes(&dictionary, &original, &sequence, 11))).unwrap();
            let mut reference = Vec::new();
            reader.decode_to(&mut reference, false).unwrap();
            assert_eq!(out, reference);
        }
    }
}
