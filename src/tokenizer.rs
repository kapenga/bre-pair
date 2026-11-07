//! Frozen byte vocabularies and exact minimum-token segmentation. Learned
//! tokens are independent expansions: their construction children need no IDs.
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};

use crate::archive::Reader;
use crate::engine::Compressed;
use crate::grammar::{Symbol, BASE_SYMBOLS, EOF};
use crate::pair_map::PairMap;

const NONE: u32 = u32::MAX;
const MAGIC: &[u8; 8] = b"BRTOKN01";
mod byte_parser;
pub use byte_parser::{ByteParser, ParseScratch};
mod greedy;
pub use greedy::GreedyParser;

/// One reusable dictionary-DP workspace, independent of corpus size.
#[derive(Default)]
pub(crate) struct ReplacementScratch {
    costs: Vec<u64>,
    choices: Vec<Symbol>,
}
impl ReplacementScratch {
    pub(crate) fn allocated_bytes(&self) -> usize {
        self.costs.capacity() * std::mem::size_of::<u64>()
            + self.choices.capacity() * std::mem::size_of::<Symbol>()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Token {
    pub bytes: Vec<u8>,
    pub ends_field: bool,
}

impl Token {
    pub fn terminals(&self) -> usize {
        self.bytes.len() + usize::from(self.ends_field)
    }
    fn labels(&self) -> Vec<u16> {
        self.bytes
            .iter()
            .map(|&b| b as u16)
            .chain(self.ends_field.then_some(EOF as u16))
            .collect()
    }
}

#[derive(Clone, Debug)]
pub struct Model {
    pub tokens: Vec<Token>,
}

pub fn fingerprint(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325u64, |hash, &byte| {
        (hash ^ byte as u64).wrapping_mul(0x100000001b3)
    })
}

impl Model {
    pub fn new(tokens: Vec<Token>) -> Result<Self, String> {
        if tokens.len() < BASE_SYMBOLS || tokens.len() >= NONE as usize {
            return Err("vocabulary must contain the 257-symbol basis and fit u32 IDs".into());
        }
        let mut unique = HashSet::with_capacity(tokens.len());
        for (id, token) in tokens.iter().enumerate() {
            if id < EOF as usize && (token.bytes != [id as u8] || token.ends_field) {
                return Err("byte token IDs must equal their byte values".into());
            }
            if id == EOF as usize && (!token.bytes.is_empty() || !token.ends_field) {
                return Err("token 256 must be the empty EOF token".into());
            }
            if token.bytes.len() > u32::MAX as usize
                || token.terminals() == 0
                || (id >= BASE_SYMBOLS && token.bytes.is_empty())
                || !unique.insert(token)
            {
                return Err("empty or duplicate token expansion".into());
            }
        }
        Ok(Self { tokens })
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend(MAGIC);
        out.extend((self.tokens.len() as u32).to_le_bytes());
        for token in &self.tokens {
            out.extend((token.bytes.len() as u32).to_le_bytes());
            out.push(u8::from(token.ends_field));
            out.extend(&token.bytes);
        }
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        let mut r = Reader { bytes, at: 0 };
        if r.take(8)? != MAGIC {
            return Err("not a BRTOKN01 model".into());
        }
        let count = r.u32()? as usize;
        if count > bytes.len() / 5 {
            return Err("model counts exceed file size".into());
        }
        let mut tokens = Vec::with_capacity(count);
        for _ in 0..count {
            let len = r.u32()? as usize;
            let flag = r.take(1)?[0];
            if flag > 1 {
                return Err("invalid token flags".into());
            }
            tokens.push(Token {
                bytes: r.take(len)?.to_vec(),
                ends_field: flag != 0,
            });
        }
        if r.at != bytes.len() {
            return Err("trailing model data".into());
        }
        Self::new(tokens)
    }

    /// Flatten, deduplicate and bound an existing enwik8 candidate grammar.
    /// Support includes uses through every parent in the encoded training DAG.
    pub fn candidates(source: &Compressed, max_bytes: usize) -> Result<(Self, Vec<u64>), String> {
        Self::candidates_filtered(source, max_bytes, None)
    }

    /// Construction helpers still contribute through the support DAG, but
    /// only explicitly admitted entries become vocabulary candidates.
    pub fn candidates_filtered(
        source: &Compressed,
        max_bytes: usize,
        included: Option<&[bool]>,
    ) -> Result<(Self, Vec<u64>), String> {
        let (model, support, _) = Self::candidates_impl(source, max_bytes, included, None)?;
        Ok((model, support))
    }

    /// Research: retain a second dense construction-support vector for one
    /// labeled subset of root occurrences, sharing the same deduplicated pool.
    pub fn candidates_with_language_support(
        source: &Compressed,
        max_bytes: usize,
        included: Option<&[bool]>,
        language_roots: &[u64],
    ) -> Result<(Self, Vec<u64>, Vec<u64>), String> {
        let (model, support, language) =
            Self::candidates_impl(source, max_bytes, included, Some(language_roots))?;
        Ok((
            model,
            support,
            language.expect("requested language support"),
        ))
    }

    #[allow(clippy::type_complexity)]
    fn candidates_impl(
        source: &Compressed,
        max_bytes: usize,
        included: Option<&[bool]>,
        language_roots: Option<&[u64]>,
    ) -> Result<(Self, Vec<u64>, Option<Vec<u64>>), String> {
        let grammar = &source.grammar;
        if included.is_some_and(|m| {
            m.len() != grammar.alphabet_size() || m[..BASE_SYMBOLS].contains(&false)
        }) {
            return Err("candidate membership must retain the basis and match the grammar".into());
        }
        let mut support = vec![0u64; grammar.alphabet_size()];
        for block in &source.blocks {
            for symbol in block.iter() {
                support[symbol as usize] += 1;
            }
        }
        if language_roots.is_some_and(|roots| {
            roots.len() != support.len() || roots.iter().zip(&support).any(|(a, b)| a > b)
        }) {
            return Err("language root counts must be a subset of total root counts".into());
        }
        let mut language_support = language_roots.map(<[u64]>::to_vec);
        for (offset, rule) in grammar.rules.iter().enumerate().rev() {
            let uses = support[BASE_SYMBOLS + offset];
            for child in [rule.left, rule.right] {
                support[child as usize] = support[child as usize]
                    .checked_add(uses)
                    .ok_or("candidate support overflow")?;
                if let Some(language) = &mut language_support {
                    language[child as usize] = language[child as usize]
                        .checked_add(language[BASE_SYMBOLS + offset])
                        .ok_or("language candidate support overflow")?;
                }
            }
        }
        let mut expansions: Vec<Option<Token>> = (0..EOF)
            .map(|byte| {
                Some(Token {
                    bytes: vec![byte as u8],
                    ends_field: false,
                })
            })
            .chain([Some(Token {
                bytes: vec![],
                ends_field: true,
            })])
            .collect();
        for rule in &grammar.rules {
            let token = match (
                &expansions[rule.left as usize],
                &expansions[rule.right as usize],
            ) {
                (Some(left), Some(right)) if left.bytes.len() + right.bytes.len() <= max_bytes => {
                    let mut bytes = left.bytes.clone();
                    bytes.extend(&right.bytes);
                    Some(Token {
                        bytes,
                        ends_field: right.ends_field,
                    })
                }
                _ => None,
            };
            expansions.push(token);
        }
        let mut unique: HashMap<Token, usize> = HashMap::new();
        let mut tokens = Vec::new();
        let mut counts: Vec<u64> = Vec::new();
        let mut language_counts: Option<Vec<u64>> = language_support.as_ref().map(|_| Vec::new());
        for (source_id, token) in expansions.into_iter().enumerate() {
            if included.is_some_and(|m| !m[source_id]) {
                continue;
            }
            if let Some(token) = token {
                if let Some(&id) = unique.get(&token) {
                    counts[id] = counts[id]
                        .checked_add(support[source_id])
                        .ok_or("support overflow")?;
                    if let (Some(values), Some(language)) =
                        (&mut language_counts, &language_support)
                    {
                        values[id] = values[id]
                            .checked_add(language[source_id])
                            .ok_or("language support overflow")?;
                    }
                } else {
                    unique.insert(token.clone(), tokens.len());
                    tokens.push(token);
                    counts.push(support[source_id]);
                    if let (Some(values), Some(language)) =
                        (&mut language_counts, &language_support)
                    {
                        values.push(language[source_id]);
                    }
                }
            }
        }
        Ok((Self::new(tokens)?, counts, language_counts))
    }

    pub fn subset(&self, active: &[u8]) -> Result<Self, String> {
        if active.len() != self.tokens.len() || active[..BASE_SYMBOLS].contains(&0) {
            return Err("subset must retain every byte and EOF".into());
        }
        Self::new(
            self.tokens
                .iter()
                .zip(active)
                .filter(|(_, flag)| **flag != 0)
                .map(|(token, _)| token.clone())
                .collect(),
        )
    }

    pub fn decode_field(&self, ids: &[Symbol]) -> Result<Vec<u8>, String> {
        let mut bytes = Vec::new();
        let mut ended = false;
        for &id in ids {
            if ended {
                return Err("token after field ending".into());
            }
            let token = self.tokens.get(id as usize).ok_or("undefined token ID")?;
            bytes.extend(&token.bytes);
            ended = token.ends_field;
        }
        if !ended {
            return Err("missing field ending".into());
        }
        Ok(bytes)
    }
}

struct BuildNode {
    first: u32,
    last: u32,
    token: u32,
}
impl Default for BuildNode {
    fn default() -> Self {
        Self {
            first: NONE,
            last: NONE,
            token: NONE,
        }
    }
}
struct BuildEdge {
    label: u16,
    child: u32,
    next: u32,
}
struct Node {
    first: u32,
    count: u16,
    token: u32,
}
struct Edge {
    label: u16,
    child: u32,
}

/// A flat sorted trie: no per-node heap allocations in the runtime index.
pub struct Tokenizer {
    pub(crate) model: Model,
    nodes: Vec<Node>,
    edges: Vec<Edge>,
    roots: [u32; BASE_SYMBOLS],
    longest: usize,
}

impl Tokenizer {
    pub fn model(&self) -> &Model {
        &self.model
    }

    pub fn new(model: Model) -> Result<Self, String> {
        Self::build(model, false)
    }

    fn build(model: Model, reverse: bool) -> Result<Self, String> {
        let model = Model::new(model.tokens)?;
        let mut sorted: Vec<_> = model
            .tokens
            .iter()
            .enumerate()
            .map(|(id, t)| {
                let mut labels = t.labels();
                if reverse {
                    labels.reverse();
                }
                (labels, id as u32)
            })
            .collect();
        sorted.sort_unstable();
        let mut build = vec![BuildNode::default()];
        let mut links: Vec<BuildEdge> = Vec::new();
        let mut stack = vec![0u32];
        let mut previous: Vec<u16> = Vec::new();
        let mut longest = 1;
        for (labels, id) in sorted {
            longest = longest.max(labels.len());
            let prefix = labels
                .iter()
                .zip(&previous)
                .take_while(|(a, b)| a == b)
                .count();
            stack.truncate(prefix + 1);
            for &label in &labels[prefix..] {
                let child = u32::try_from(build.len()).map_err(|_| "trie exceeds u32 nodes")?;
                let edge = u32::try_from(links.len()).map_err(|_| "trie exceeds u32 edges")?;
                if child == NONE || edge == NONE {
                    return Err("trie index exhausted".into());
                }
                build.push(BuildNode::default());
                links.push(BuildEdge {
                    label,
                    child,
                    next: NONE,
                });
                let parent = &mut build[*stack.last().unwrap() as usize];
                if parent.last == NONE {
                    parent.first = edge;
                } else {
                    links[parent.last as usize].next = edge;
                }
                parent.last = edge;
                stack.push(child);
            }
            build[*stack.last().unwrap() as usize].token = id;
            previous = labels;
        }
        let mut nodes = Vec::with_capacity(build.len());
        let mut edges = Vec::with_capacity(links.len());
        for node in build {
            let first = edges.len() as u32;
            let mut at = node.first;
            while at != NONE {
                let edge = &links[at as usize];
                edges.push(Edge {
                    label: edge.label,
                    child: edge.child,
                });
                at = edge.next;
            }
            nodes.push(Node {
                first,
                count: (edges.len() - first as usize) as u16,
                token: node.token,
            });
        }
        let mut roots = [NONE; BASE_SYMBOLS];
        for edge in &edges[..nodes[0].count as usize] {
            roots[edge.label as usize] = edge.child;
        }
        Ok(Self {
            model,
            nodes,
            edges,
            roots,
            longest,
        })
    }

    fn child(&self, node: u32, label: u16) -> Option<u32> {
        let node = &self.nodes[node as usize];
        let slice = &self.edges[node.first as usize..node.first as usize + node.count as usize];
        slice
            .binary_search_by_key(&label, |e| e.label)
            .ok()
            .map(|i| slice[i].child)
    }

    /// Backward shortest-path DP. Equal-cost parses prefer a longer first token,
    /// then the smaller ID. Byte positions and corpus totals remain 64-bit.
    fn choices(
        &self,
        bytes: &[u8],
        eof: bool,
        active: Option<&[u8]>,
        exclude: u32,
    ) -> (u64, Vec<Symbol>) {
        let mut scratch = ReplacementScratch::default();
        let cost = self.choices_into(bytes, eof, active, exclude, &mut scratch);
        (cost, scratch.choices)
    }

    fn choices_into(
        &self,
        bytes: &[u8],
        eof: bool,
        active: Option<&[u8]>,
        exclude: u32,
        scratch: &mut ReplacementScratch,
    ) -> u64 {
        let len = bytes.len() + usize::from(eof);
        let window = self.longest.min(len) + 1;
        scratch.costs.resize(window, u64::MAX);
        scratch.costs.fill(u64::MAX);
        scratch.choices.resize(len, 0);
        let costs = &mut scratch.costs;
        let choices = &mut scratch.choices;
        costs[len % window] = 0;
        for start in (0..len).rev() {
            let first = if start < bytes.len() {
                bytes[start] as u16
            } else {
                EOF as u16
            };
            let mut node = self.roots[first as usize];
            let mut best = costs[(start + 1) % window] + 1;
            let mut best_len = 1;
            let mut choice = first as u32;
            for end in start + 1..len.min(start + self.longest) {
                let label = if end < bytes.len() {
                    bytes[end] as u16
                } else {
                    EOF as u16
                };
                let Some(next) = self.child(node, label) else {
                    break;
                };
                node = next;
                let token = self.nodes[node as usize].token;
                if token != NONE
                    && token != exclude
                    && active.is_none_or(|a| a[token as usize] != 0)
                {
                    let cost = costs[(end + 1) % window] + 1;
                    let length = end + 1 - start;
                    if cost < best
                        || (cost == best
                            && (length > best_len || (length == best_len && token < choice)))
                    {
                        best = cost;
                        best_len = length;
                        choice = token;
                    }
                }
            }
            costs[start % window] = best;
            choices[start] = choice;
        }
        costs[0]
    }

    pub fn encode(&self, bytes: &[u8]) -> Vec<Symbol> {
        self.encode_active(bytes, None)
    }

    /// Parse a non-final fragment without inventing a field ending. This uses
    /// the same exact objective and tie order as complete-field encoding.
    pub fn encode_fragment(&self, bytes: &[u8]) -> Vec<Symbol> {
        let (cost, choices) = self.choices(bytes, false, None, NONE);
        let mut ids = Vec::with_capacity(cost as usize);
        let mut at = 0;
        while at < choices.len() {
            let id = choices[at];
            ids.push(id);
            at += self.model.tokens[id as usize].terminals();
        }
        ids
    }

    pub(crate) fn encode_active(&self, bytes: &[u8], active: Option<&[u8]>) -> Vec<Symbol> {
        let (cost, choices) = self.choices(bytes, true, active, NONE);
        let mut ids = Vec::with_capacity(cost as usize);
        let mut at = 0;
        while at < choices.len() {
            let id = choices[at];
            ids.push(id);
            at += self.model.tokens[id as usize].terminals();
        }
        ids
    }

    pub(crate) fn replacement_cost(&self, id: Symbol, active: &[u8]) -> u64 {
        let token = &self.model.tokens[id as usize];
        self.choices(&token.bytes, token.ends_field, Some(active), id)
            .0
    }

    /// Learned pieces strictly inside a target; leaves a terminal on each side.
    pub(crate) fn visit_internal_matches(
        &self,
        id: Symbol,
        active: &[u8],
        mut visit: impl FnMut(Symbol, usize),
    ) {
        let token = &self.model.tokens[id as usize];
        let limit = token.terminals() - 1;
        for start in 1..limit {
            let mut node = self.roots[token.bytes[start] as usize];
            for end in start + 1..limit {
                let Some(next) = self.child(node, u16::from(token.bytes[end])) else {
                    break;
                };
                node = next;
                let candidate = self.nodes[node as usize].token;
                if candidate != NONE
                    && candidate as usize >= BASE_SYMBOLS
                    && active[candidate as usize] != 0
                {
                    visit(candidate, end + 1 - start);
                }
            }
        }
    }

    /// Cheapest literal replacement, without adding an EOF to non-ending pieces.
    pub(crate) fn replacement(
        &self,
        id: Symbol,
        active: &[u8],
        scratch: &mut ReplacementScratch,
        ids: &mut Vec<Symbol>,
    ) {
        let token = &self.model.tokens[id as usize];
        self.choices_into(&token.bytes, token.ends_field, Some(active), id, scratch);
        ids.clear();
        let mut at = 0;
        while at < scratch.choices.len() {
            let next = scratch.choices[at];
            ids.push(next);
            at += self.model.tokens[next as usize].terminals();
        }
    }

    /// Exact whole-field cost with one learned entry unavailable. Fields whose
    /// chosen optimum omits it need no counterfactual reparse.
    pub(crate) fn count_excluding(&self, bytes: &[u8], id: Symbol) -> u64 {
        self.choices(bytes, true, None, id).0
    }

    /// All matching edges, including alternatives crossing the chosen parse.
    /// Positions include the final EOF and never cross a field boundary.
    pub(crate) fn matching_edges(
        &self,
        bytes: &[u8],
        mut visit: impl FnMut(usize, usize, Symbol) -> Result<(), String>,
    ) -> Result<(), String> {
        let len = bytes.len() + 1;
        for start in 0..len {
            let first = bytes.get(start).map_or(256, |&b| b as usize);
            let mut node = self.roots[first];
            visit(start, start + 1, first as Symbol)?;
            for end in start + 1..len.min(start + self.longest) {
                let label = bytes.get(end).map_or(256, |&b| b as u16);
                let Some(next) = self.child(node, label) else {
                    break;
                };
                node = next;
                let token = self.nodes[node as usize].token;
                if token != NONE {
                    visit(start, end + 1, token)?;
                }
            }
        }
        Ok(())
    }

    /// Propose inactive entries spanning at least two tokens of the current
    /// optimal parse. Scores order proposals only; whole-field reparsing must
    /// measure their actual joint benefit and final usage.
    pub(crate) fn insertion_candidates(
        &self,
        bytes: &[u8],
        ids: &[Symbol],
        active: &[u8],
        mut visit: impl FnMut(Symbol, u64) -> Result<(), String>,
    ) -> Result<(), String> {
        let mut boundaries = Vec::with_capacity(ids.len() + 1);
        boundaries.push(0);
        for &id in ids {
            boundaries
                .push(boundaries.last().unwrap() + self.model.tokens[id as usize].terminals());
        }
        let len = bytes.len() + 1;
        for (i, &start) in boundaries[..ids.len()].iter().enumerate() {
            let first = bytes.get(start).map_or(256, |&b| b as usize);
            let mut node = self.roots[first];
            for end in start + 1..len.min(start + self.longest) {
                let label = bytes.get(end).map_or(256, |&b| b as u16);
                let Some(next) = self.child(node, label) else {
                    break;
                };
                node = next;
                let token = self.nodes[node as usize].token;
                if token != NONE && active[token as usize] == 0 {
                    if let Ok(j) = boundaries.binary_search(&(end + 1)) {
                        if j > i + 1 {
                            visit(token, (j - i - 1) as u64)?;
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

pub struct Bpe {
    pub(crate) model: Model,
    ranks: PairMap<(u32, u32)>,
}

impl Bpe {
    pub fn model(&self) -> &Model {
        &self.model
    }

    pub fn from_bytes(model: Model, bytes: &[u8]) -> Result<Self, String> {
        let model = Model::new(model.tokens)?;
        let mut r = Reader { bytes, at: 0 };
        if r.take(8)? != b"BRTBPE01" || r.u64()? != fingerprint(&model.to_bytes()) {
            return Err("BPE ranks do not match the vocabulary".into());
        }
        let count = r.u32()? as usize;
        if count > bytes.len() / 12 {
            return Err("BPE rank count exceeds file size".into());
        }
        let mut ranks = PairMap::default();
        for rank in 0..count {
            let left = r.u32()?;
            let right = r.u32()?;
            let out = r.u32()?;
            let a = model
                .tokens
                .get(left as usize)
                .ok_or("undefined BPE child")?;
            let b = model
                .tokens
                .get(right as usize)
                .ok_or("undefined BPE child")?;
            let output = model
                .tokens
                .get(out as usize)
                .ok_or("undefined BPE output")?;
            if a.ends_field
                || output.ends_field != b.ends_field
                || output.bytes != [a.bytes.as_slice(), b.bytes.as_slice()].concat()
                || ranks.insert((left, right), (rank as u32, out)).is_some()
            {
                return Err("invalid BPE merge definition".into());
            }
        }
        if r.at != bytes.len() {
            return Err("trailing BPE rank data".into());
        }
        Ok(Self { model, ranks })
    }

    pub fn encode(&self, bytes: &[u8]) -> Vec<Symbol> {
        let mut ids: Vec<_> = bytes.iter().map(|&b| b as u32).chain([EOF]).collect();
        let n = ids.len();
        let mut prev: Vec<_> = (0..n)
            .map(|i| i.checked_sub(1).unwrap_or(usize::MAX))
            .collect();
        let mut next: Vec<_> = (1..=n).collect();
        let mut heap = BinaryHeap::new();
        let add = |at: usize, ids: &[Symbol], next: &[usize], heap: &mut BinaryHeap<_>| {
            if at < n && next[at] < n {
                let pair = (ids[at], ids[next[at]]);
                if let Some(&(rank, out)) = self.ranks.get(&pair) {
                    heap.push(Reverse((rank, at, next[at], pair.0, pair.1, out)));
                }
            }
        };
        for at in 0..n {
            add(at, &ids, &next, &mut heap);
        }
        while let Some(Reverse((_, at, right, a, b, out))) = heap.pop() {
            if ids[at] != a || ids[right] != b || next[at] != right {
                continue;
            }
            ids[at] = out;
            ids[right] = NONE;
            next[at] = next[right];
            if next[at] < n {
                prev[next[at]] = at;
            }
            add(prev[at], &ids, &next, &mut heap);
            add(at, &ids, &next, &mut heap);
        }
        let mut output = Vec::new();
        let mut at = 0;
        while at < n {
            output.push(ids[at]);
            at = next[at];
        }
        output
    }
}
