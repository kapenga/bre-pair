//! Frozen AI vocabulary + token-compatible BRe-Pair recipes + training Huffman
//! table. Application never learns from its input or rebuilds Huffman codes.
use crate::{
    engine::{Config, Round},
    field_engine,
    grammar::BASE_SYMBOLS,
    huffman::Codebook,
    huffman_archive::{Encoder, Totals},
    huffman_dictionary::{DecodeGraph, Dictionary},
    pair_map::PairMap,
    repeated,
    tokenizer::{fingerprint, ByteParser, GreedyParser, Model, ParseScratch},
};
use std::{
    fs::File,
    io::{BufRead, BufReader, BufWriter, Seek, Write},
    path::Path,
    time::Instant,
};

const MAGIC: &[u8; 8] = b"BRCODE01";
const HEADER: usize = 56;
const NONE: u32 = u32::MAX;
mod parallel;
pub(crate) mod spooled;
pub use parallel::EncodeOptions;
mod optimize;
pub mod periodic;
pub use optimize::{Optimization, OptimizeOptions, OptimizeReport};
pub mod recipe_compaction;
pub mod refinement;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EncodeMode {
    /// Exact minimum AI-token count, then minimum Huffman bits for those IDs.
    #[default]
    Advanced,
    /// Longest-prefix matches in both stages; AI IDs may differ from Advanced.
    Greedy,
}
impl EncodeMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Advanced => "advanced",
            Self::Greedy => "greedy",
        }
    }
}
impl std::str::FromStr for EncodeMode {
    type Err = String;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "advanced" => Ok(Self::Advanced),
            "greedy" => Ok(Self::Greedy),
            _ => Err("encoding mode must be advanced or greedy".into()),
        }
    }
}

enum Parser {
    Advanced(ByteParser),
    Greedy(GreedyParser),
}
impl Parser {
    fn new(model: Model, mode: EncodeMode) -> Result<Self, String> {
        match mode {
            EncodeMode::Advanced => Ok(Self::Advanced(ByteParser::new(model)?)),
            EncodeMode::Greedy => Ok(Self::Greedy(GreedyParser::new(model)?)),
        }
    }
    fn encode<'a>(&self, bytes: &[u8], scratch: &'a mut ParseScratch) -> &'a [u32] {
        match self {
            Self::Advanced(parser) => parser.encode(bytes, scratch),
            Self::Greedy(parser) => parser.encode(bytes, scratch),
        }
    }
}

pub struct Frozen {
    pub dictionary: Dictionary,
    pub book: Codebook,
    pub training: Totals,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DocumentCounting {
    #[default]
    Exact,
    LastSeen,
    Boundary,
}
impl DocumentCounting {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Exact => "exact",
            Self::LastSeen => "last-seen",
            Self::Boundary => "boundary",
        }
    }
}
impl std::str::FromStr for DocumentCounting {
    type Err = String;
    fn from_str(value: &str) -> Result<Self, String> {
        match value {
            "exact" => Ok(Self::Exact),
            "last-seen" => Ok(Self::LastSeen),
            "boundary" => Ok(Self::Boundary),
            _ => Err("document counting must be exact, last-seen, or boundary".into()),
        }
    }
}

#[derive(Default)]
pub struct TrainReport {
    pub rounds: usize,
    pub segmentation_seconds: f64,
    pub merging_seconds: f64,
    pub huffman_seconds: f64,
    pub helper_rules: usize,
    pub zero_usage_codes: usize,
    pub registry_peak_slot_bytes: usize,
    pub max_packed_bytes: usize,
    pub stopped_at_limit: bool,
    pub stopped_at_round_limit: bool,
    pub minimum_count: u64,
    pub document_filtered_candidates: u64,
    pub document_support_seconds: f64,
    pub document_cleanup_seconds: f64,
    pub document_removed_codes: usize,
    pub document_removed_used_codes: usize,
    pub document_cleanup_before_symbols: u64,
    pub document_cleanup_after_symbols: u64,
}

#[derive(Clone, Debug, Default)]
pub struct TrainOptions {
    /// Additional compression recipes beyond the fixed AI vocabulary.
    pub max_rules: Option<usize>,
    /// Completed merge/rewrite passes. Hitting the cap still builds a model.
    pub max_rounds: Option<usize>,
    /// Occurrence floor per decimal GB, rounded down to whole occurrences.
    pub minimum_count_per_gb: u64,
    pub documents: Option<crate::document_support::Policy>,
    pub document_counting: DocumentCounting,
}
impl TrainOptions {
    pub fn effective_minimum(&self, base: u64, input_bytes: u64) -> Result<u64, String> {
        if base < 2 {
            return Err("codec minimum count must be at least two".into());
        }
        if self.document_counting != DocumentCounting::Exact && self.documents.is_none() {
            return Err("document counting mode requires a training document index".into());
        }
        let scaled =
            u128::from(input_bytes) * u128::from(self.minimum_count_per_gb) / 1_000_000_000;
        let scaled = u64::try_from(scaled).map_err(|_| "scaled minimum count exceeds u64")?;
        let documents = if let Some(policy) = &self.documents {
            if !(1..=64).contains(&policy.minimum_groups) {
                return Err("rule support requires 1..64 document groups".into());
            }
            policy.minimum_groups as u64
        } else {
            1
        };
        Ok(base.max(scaled).max(documents))
    }
}

impl Frozen {
    pub fn train(
        vocabulary: &Model,
        input: &Path,
        config: &Config,
        workers: usize,
        maximum_rules: Option<usize>,
        on_round: impl FnMut(&Round),
    ) -> Result<(Self, TrainReport), String> {
        Self::train_with_options(
            vocabulary,
            input,
            config,
            workers,
            TrainOptions {
                max_rules: maximum_rules,
                ..Default::default()
            },
            on_round,
        )
    }

    pub fn train_with_options(
        vocabulary: &Model,
        input: &Path,
        config: &Config,
        workers: usize,
        options: TrainOptions,
        on_round: impl FnMut(&Round),
    ) -> Result<(Self, TrainReport), String> {
        if options.max_rounds == Some(0) {
            return Err("codec round limit must be positive".into());
        }
        let mut config = config.clone();
        let input_bytes = std::fs::metadata(input).map_err(|e| e.to_string())?.len();
        config.minimum_count = options.effective_minimum(config.minimum_count, input_bytes)?;
        let start = Instant::now();
        let (mut source, membership) =
            repeated::prepare(input, vocabulary, 65536, config.buffer_bytes)?;
        if source.input_bytes != input_bytes {
            return Err("training input size changed during packing".into());
        }
        let (_, seed_ids) = repeated::seed_grammar(vocabulary)?;
        let initial_tokens = source.blocks.iter().map(|b| b.len() as u64).sum();
        let helper_rules = source.grammar.rules.len();
        // AI terminals are atomic in this stage. Keep their byte construction
        // graph for metadata/decoding, with a fresh lookup for compression
        // recipes. Reusing a helper ID would revive old pair counts and force
        // the merge engine's full-recount safeguard.
        source.grammar.reset_merge_lookup();
        if crate::training_progress::enabled() {
            crate::training_progress::seed(
                helper_rules,
                initial_tokens,
                source.blocks.iter().map(|b| b.storage_bytes()).sum(),
                crate::packed::symbol_width(source.grammar.alphabet_size()),
            );
        }
        let segmentation_seconds = start.elapsed().as_secs_f64();
        let limit = options
            .max_rules
            .map(|n| {
                vocabulary
                    .tokens
                    .len()
                    .checked_add(n)
                    .ok_or("codec rule limit overflow")
            })
            .transpose()?;
        let limits = field_engine::Options {
            max_rules: limit,
            max_rounds: options.max_rounds,
            frequency_cutoff: true,
            memory_saving: true,
            last_seen_documents: options.document_counting == DocumentCounting::LastSeen,
            ..Default::default()
        };
        let result = if let Some(documents) = options
            .documents
            .as_ref()
            .filter(|_| options.document_counting != DocumentCounting::Boundary)
        {
            field_engine::compress_seeded_documents(
                source, membership, &config, workers, limits, documents, on_round,
            )?
        } else {
            field_engine::compress_seeded(
                source, membership, &config, workers, limits, None, on_round,
            )?
        };
        crate::training_progress::phase(
            "building_model",
            result.training.rounds.len(),
            0,
            0,
            "symbols",
        );
        let source = &result.training.compressed;
        let cleanup = if options.document_counting == DocumentCounting::Boundary {
            let mut protected = vec![false; source.grammar.alphabet_size()];
            for &id in &seed_ids {
                protected[id as usize] = true;
            }
            Some(crate::document_support::Cleanup::plan_at_round(
                source,
                options.documents.as_ref().unwrap(),
                &protected,
                result.training.rounds.len(),
            )?)
        } else {
            None
        };
        let members = result
            .candidate_membership
            .as_ref()
            .ok_or("missing codec membership")?;
        let mut mapping = vec![NONE; source.grammar.alphabet_size()];
        for (ai_id, &seed_id) in seed_ids.iter().enumerate() {
            mapping[seed_id as usize] = ai_id as u32;
        }
        let mut rules = Vec::new();
        for (i, rule) in source.grammar.rules.iter().enumerate() {
            let id = BASE_SYMBOLS + i;
            if !members[id] || mapping[id] != NONE {
                continue;
            }
            let (left, right) = (mapping[rule.left as usize], mapping[rule.right as usize]);
            if left == NONE || right == NONE {
                return Err("compression rule would split a frozen AI terminal".into());
            }
            mapping[id] = u32::try_from(vocabulary.tokens.len() + rules.len())
                .map_err(|_| "codec IDs exhausted")?;
            rules.push((left, right));
        }
        let dictionary = Dictionary::new(vocabulary.clone(), rules, true)?;
        let mut counts = vec![0u64; dictionary.expansions.len()];
        let actual = if let Some(cleanup) = &cleanup {
            // Cleanup already counted the final expanded stream. Map its
            // histogram instead of walking the entire packed input again.
            for (old, &count) in cleanup.uses.iter().enumerate() {
                if count != 0 {
                    let id = mapping[old];
                    if id == NONE {
                        return Err("cleanup exposed a non-AI construction helper".into());
                    }
                    counts[id as usize] = counts[id as usize]
                        .checked_add(count)
                        .ok_or("codec count overflow")?;
                }
            }
            optimize::totals(&dictionary, &counts)?
        } else {
            let mut actual = Totals::default();
            for block in &source.blocks {
                for symbol in block.iter() {
                    let id = mapping[symbol as usize];
                    if id == NONE {
                        return Err("unmapped codec symbol".into());
                    }
                    counts[id as usize] = counts[id as usize]
                        .checked_add(1)
                        .ok_or("codec count overflow")?;
                    actual.add_symbol(&dictionary, id)?;
                }
            }
            actual
        };
        if actual.bytes != source.input_bytes
            || actual.fields != source.field_count
            || actual.tokens != initial_tokens
        {
            return Err("codec training changed byte, field, or frozen AI token totals".into());
        }
        let (mut document_removed_codes, mut document_removed_used_codes) = (0, 0);
        let mut active = cleanup.as_ref().map(|_| vec![true; counts.len()]);
        if let Some(cleanup) = &cleanup {
            for (old, &new) in mapping.iter().enumerate() {
                if new != NONE && !cleanup.active[old] {
                    active.as_mut().unwrap()[new as usize] = false;
                    document_removed_codes += 1;
                    document_removed_used_codes += usize::from(cleanup.groups[old] != 0);
                }
            }
        }
        let start = Instant::now();
        crate::training_progress::phase(
            "huffman_fit",
            result.training.rounds.len(),
            0,
            counts.len() as u64,
            "symbols",
        );
        let (model, zero_usage_codes) = if let Some(active) = active {
            let (model, _, histogram) =
                optimize::candidate_dictionary(&dictionary, &counts, &active)?;
            if (
                model.training.bytes,
                model.training.tokens,
                model.training.fields,
            ) != (actual.bytes, actual.tokens, actual.fields)
            {
                return Err("document cleanup changed bytes, fields, or frozen AI IDs".into());
            }
            let zeros = model
                .book
                .symbols
                .iter()
                .filter(|&&id| histogram[id as usize] == 0)
                .count();
            (model, zeros)
        } else {
            let zeros = counts.iter().filter(|&&n| n == 0).count();
            for count in &mut counts {
                *count = (*count).max(1);
            }
            (
                Self {
                    dictionary,
                    book: Codebook::from_counts(&counts)?,
                    training: actual,
                },
                zeros,
            )
        };
        let huffman_seconds = start.elapsed().as_secs_f64();
        let report = TrainReport {
            rounds: result.training.rounds.len(),
            segmentation_seconds,
            merging_seconds: result.training.seconds,
            huffman_seconds,
            helper_rules,
            zero_usage_codes,
            registry_peak_slot_bytes: result.registry_peak_slot_bytes,
            max_packed_bytes: result.training.max_buffer_bytes,
            stopped_at_limit: result.stopped_at_rule_limit,
            stopped_at_round_limit: result.stopped_at_round_limit,
            minimum_count: config.minimum_count,
            document_filtered_candidates: result.document_filtered_candidates,
            document_support_seconds: result.document_support_seconds,
            document_cleanup_seconds: cleanup.as_ref().map_or(0.0, |p| p.seconds),
            document_removed_codes,
            document_removed_used_codes,
            document_cleanup_before_symbols: cleanup.as_ref().map_or(0, |p| p.before_symbols),
            document_cleanup_after_symbols: cleanup.as_ref().map_or(0, |p| p.after_symbols),
        };
        Ok((model, report))
    }

    fn validate(&self) -> Result<(), String> {
        let n = self.dictionary.expansions.len();
        if !self.dictionary.ai_tokens {
            return Err("frozen codec requires an AI dictionary".into());
        }
        let mut seen = vec![false; n];
        for &id in &self.book.symbols {
            let flag = seen
                .get_mut(id as usize)
                .ok_or("codec table ID exceeds dictionary")?;
            if std::mem::replace(flag, true) {
                return Err("duplicate codec table ID".into());
            }
        }
        if seen[..self.dictionary.model.tokens.len()].contains(&false) {
            return Err("frozen codec requires a code for every AI terminal".into());
        }
        if self.training.fields > self.training.tokens
            || self.training.symbols > self.training.tokens
        {
            return Err("invalid codec training totals".into());
        }
        Ok(())
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, String> {
        self.validate()?;
        let dictionary = self.dictionary.to_bytes();
        let table = self.book.to_bytes();
        let mut out = Vec::new();
        out.extend(MAGIC);
        out.extend((dictionary.len() as u64).to_le_bytes());
        out.extend((self.book.symbols.len() as u32).to_le_bytes());
        out.extend(
            u32::try_from(table.len())
                .map_err(|_| "codec table too large")?
                .to_le_bytes(),
        );
        for n in [
            self.training.bytes,
            self.training.fields,
            self.training.tokens,
            self.training.symbols,
        ] {
            out.extend(n.to_le_bytes());
        }
        out.extend(dictionary);
        out.extend(table);
        out.extend(fingerprint(&out).to_le_bytes());
        Ok(out)
    }

    pub fn from_bytes(input: &[u8]) -> Result<Self, String> {
        if input.len() < HEADER + 8 || &input[..8] != MAGIC {
            return Err("not a BRCODE01 model".into());
        }
        let number = |at| u64::from_le_bytes(input[at..at + 8].try_into().unwrap());
        let dictionary_len =
            usize::try_from(number(8)).map_err(|_| "codec dictionary too large")?;
        let used = u32::from_le_bytes(input[16..20].try_into().unwrap()) as usize;
        let table_len = u32::from_le_bytes(input[20..24].try_into().unwrap()) as usize;
        let end = HEADER
            .checked_add(dictionary_len)
            .and_then(|n| n.checked_add(table_len))
            .ok_or("codec size overflow")?;
        if end.checked_add(8) != Some(input.len()) || fingerprint(&input[..end]) != number(end) {
            return Err("frozen codec length or checksum mismatch".into());
        }
        let dictionary = Dictionary::from_bytes(&input[HEADER..HEADER + dictionary_len], true)?;
        let book = Codebook::from_bytes(
            &input[HEADER + dictionary_len..end],
            used,
            dictionary.expansions.len(),
        )?;
        let model = Self {
            dictionary,
            book,
            training: Totals {
                bytes: number(24),
                fields: number(32),
                tokens: number(40),
                symbols: number(48),
            },
        };
        model.validate()?;
        Ok(model)
    }

    pub fn prepare(&self) -> Result<Matcher, String> {
        self.validate()?;
        Matcher::new(&self.dictionary, &self.book)
    }

    /// One read of the new corpus. Tokenization and grammar matching use only
    /// this immutable model; Encoder reuses its exact canonical codes.
    pub fn encode_file(
        &self,
        input: &Path,
        output: &Path,
        block_size: usize,
    ) -> Result<Totals, String> {
        self.encode_file_with_options(input, output, block_size, EncodeOptions::default())
    }

    pub fn encode_file_with_options(
        &self,
        input: &Path,
        output: &Path,
        block_size: usize,
        options: EncodeOptions,
    ) -> Result<Totals, String> {
        let source =
            BufReader::with_capacity(1 << 20, File::open(input).map_err(|e| e.to_string())?);
        let file = BufWriter::with_capacity(
            1 << 20,
            File::create_new(output).map_err(|e| e.to_string())?,
        );
        self.encode_with_options(source, file, block_size, options)
    }

    pub fn encode<R: BufRead, W: Write + Seek>(
        &self,
        source: R,
        output: W,
        block_size: usize,
    ) -> Result<Totals, String> {
        self.encode_with_options(source, output, block_size, EncodeOptions::default())
    }

    pub fn encode_with_options<R: BufRead, W: Write + Seek>(
        &self,
        mut source: R,
        output: W,
        block_size: usize,
        options: EncodeOptions,
    ) -> Result<Totals, String> {
        options.validate()?;
        let tokenizer = {
            #[cfg(feature = "profile")]
            let _span = crate::profile::Span::new("codec.startup.tokenizer");
            Parser::new(self.dictionary.model.clone(), options.mode)?
        };
        let matcher = {
            #[cfg(feature = "profile")]
            let _span = crate::profile::Span::new("codec.startup.grammar_matcher");
            self.prepare()?
        };
        if options.workers > 1 {
            return parallel::encode(
                self, &tokenizer, &matcher, source, output, block_size, options,
            );
        }
        let mut encoder = Encoder::with_codebook(output, &self.dictionary, block_size, &self.book)?;
        #[cfg(feature = "profile")]
        let mut times = [std::time::Duration::ZERO; 4];
        let (mut bytes, mut fields) = (0u64, 0u64);
        let mut field = Vec::new();
        let mut scratch = ParseScratch::default();
        let mut matching = MatchScratch::default();
        loop {
            field.clear();
            #[cfg(feature = "profile")]
            let start = Instant::now();
            let read = source
                .read_until(b'\n', &mut field)
                .map_err(|e| e.to_string())?;
            #[cfg(feature = "profile")]
            {
                times[0] += start.elapsed();
            }
            if read == 0 {
                break;
            }
            bytes = bytes
                .checked_add(field.len() as u64)
                .ok_or("input size overflow")?;
            fields += 1;
            #[cfg(feature = "profile")]
            let start = Instant::now();
            let terminals = tokenizer.encode(&field, &mut scratch);
            #[cfg(feature = "profile")]
            {
                times[1] += start.elapsed();
            }
            #[cfg(feature = "profile")]
            let start = Instant::now();
            let symbols = matcher.segment_using(terminals, &mut matching, options.mode)?;
            #[cfg(feature = "profile")]
            {
                times[2] += start.elapsed();
            }
            #[cfg(feature = "profile")]
            let start = Instant::now();
            encoder.extend(symbols)?;
            #[cfg(feature = "profile")]
            {
                times[3] += start.elapsed();
            }
        }
        #[cfg(feature = "profile")]
        let start = Instant::now();
        let result = encoder.finish(bytes, fields);
        #[cfg(feature = "profile")]
        {
            times[3] += start.elapsed();
            for (name, time) in [
                "codec.input",
                "codec.tokenize",
                "codec.grammar_match",
                "codec.huffman_write",
            ]
            .into_iter()
            .zip(times)
            {
                crate::profile::elapsed(name, fields, time.as_secs_f64());
            }
        }
        result
    }
}

struct Node {
    first: u32,
    count: u32,
    token: u32,
}
struct Edge {
    label: u32,
    child: u32,
}

/// Compact immutable trie over AI IDs, not byte expansions. Minimum-bit paths
/// preserve every frozen AI boundary, including alternatives with equal bytes.
pub struct Matcher {
    nodes: Vec<Node>,
    edges: Vec<Edge>,
    roots: Vec<u32>,
    lengths: Vec<usize>,
    bits: Vec<u8>,
    longest: usize,
}

#[derive(Default)]
pub struct MatchScratch {
    costs: Vec<u64>,
    choices: Vec<u32>,
    symbols: Vec<u32>,
}

impl Matcher {
    fn new(dictionary: &Dictionary, book: &Codebook) -> Result<Self, String> {
        Self::from_graph(dictionary, book)
    }
    pub(crate) fn from_graph<G: DecodeGraph>(
        dictionary: &G,
        book: &Codebook,
    ) -> Result<Self, String> {
        let n = dictionary.expansions().len();
        let basis = dictionary.model().tokens.len();
        let mut bits = vec![0; n];
        for (id, length) in book.lengths() {
            bits[id as usize] = length;
        }
        let mut nodes = vec![Node {
            first: 0,
            count: 0,
            token: NONE,
        }];
        let mut links = Vec::new();
        let mut by_edge = PairMap::default();
        let mut stack = Vec::new();
        for id in 0..n {
            // Uncoded recipes can still be children of coded rules. They are
            // decoding structure, never zero-bit candidates for the parser.
            if bits[id] == 0 {
                continue;
            }
            let mut node = 0;
            stack.push(id as u32);
            while let Some(symbol) = stack.pop() {
                if let Some(terminal) = dictionary.terminal(symbol as usize) {
                    node = if let Some(&child) = by_edge.get(&(node, terminal)) {
                        child
                    } else {
                        let child = u32::try_from(nodes.len())
                            .map_err(|_| "codec trie exceeds u32 nodes")?;
                        if child == NONE {
                            return Err("codec trie IDs exhausted".into());
                        }
                        nodes.push(Node {
                            first: 0,
                            count: 0,
                            token: NONE,
                        });
                        by_edge.insert((node, terminal), child);
                        links.push((node, terminal, child));
                        child
                    };
                } else {
                    stack.extend(dictionary.children(symbol as usize).rev());
                }
            }
            let old = &mut nodes[node as usize].token;
            if *old == NONE || (bits[id], id) < (bits[*old as usize], *old as usize) {
                *old = id as u32;
            }
        }
        drop(by_edge);
        links.sort_unstable();
        let mut edges = Vec::with_capacity(links.len());
        for (parent, label, child) in links {
            let node = &mut nodes[parent as usize];
            if node.count == 0 {
                node.first = edges.len() as u32;
            }
            node.count += 1;
            edges.push(Edge { label, child });
        }
        let mut roots = vec![NONE; basis];
        for edge in &edges[..nodes[0].count as usize] {
            roots[edge.label as usize] = edge.child;
        }
        let lengths: Vec<_> = dictionary
            .expansions()
            .iter()
            .map(|e| usize::try_from(e.terminals).map_err(|_| "codec expansion too long"))
            .collect::<Result<_, _>>()?;
        let longest = lengths
            .iter()
            .zip(&bits)
            .filter(|(_, bits)| **bits != 0)
            .map(|(&length, _)| length)
            .max()
            .unwrap_or(1);
        Ok(Self {
            nodes,
            edges,
            roots,
            lengths,
            bits,
            longest,
        })
    }

    pub fn segment(&self, terminals: &[u32]) -> Result<Vec<u32>, String> {
        Ok(self
            .segment_with(terminals, &mut MatchScratch::default())?
            .to_vec())
    }

    fn segment_using<'a>(
        &self,
        terminals: &[u32],
        scratch: &'a mut MatchScratch,
        mode: EncodeMode,
    ) -> Result<&'a [u32], String> {
        match mode {
            EncodeMode::Advanced => self.segment_with(terminals, scratch),
            EncodeMode::Greedy => self.segment_greedy_with(terminals, scratch),
        }
    }

    /// Longest AI-ID prefix at each position. Identical expansions use the
    /// shortest code, then smaller ID. No token-count or bit-cost lookahead.
    pub fn segment_greedy_with<'a>(
        &self,
        terminals: &[u32],
        scratch: &'a mut MatchScratch,
    ) -> Result<&'a [u32], String> {
        if terminals.iter().any(|&id| id as usize >= self.roots.len()) {
            return Err("undefined frozen AI token".into());
        }
        scratch.symbols.clear();
        let mut at = 0;
        while at < terminals.len() {
            let mut node = self.roots[terminals[at] as usize];
            let mut choice = self.nodes[node as usize].token;
            let mut length = 1;
            for (end, &label) in terminals
                .iter()
                .enumerate()
                .take(terminals.len().min(at.saturating_add(self.longest)))
                .skip(at + 1)
            {
                let parent = &self.nodes[node as usize];
                let children =
                    &self.edges[parent.first as usize..(parent.first + parent.count) as usize];
                let Ok(index) = children.binary_search_by_key(&label, |e| e.label) else {
                    break;
                };
                node = children[index].child;
                let id = self.nodes[node as usize].token;
                if id != NONE {
                    choice = id;
                    length = end + 1 - at;
                }
            }
            scratch.symbols.push(choice);
            at += length;
        }
        Ok(&scratch.symbols)
    }

    pub fn segment_with<'a>(
        &self,
        terminals: &[u32],
        scratch: &'a mut MatchScratch,
    ) -> Result<&'a [u32], String> {
        if terminals.iter().any(|&id| id as usize >= self.roots.len()) {
            return Err("undefined frozen AI token".into());
        }
        let window = (self.longest.min(terminals.len()) + 1).next_power_of_two();
        let mask = window - 1;
        if scratch.costs.len() < window {
            scratch.costs.resize(window, 0);
        }
        if scratch.choices.len() < terminals.len() {
            scratch.choices.resize(terminals.len(), 0);
        }
        let costs = &mut scratch.costs;
        let choices = &mut scratch.choices;
        costs[terminals.len() & mask] = 0;
        for start in (0..terminals.len()).rev() {
            let base = terminals[start];
            let mut node = self.roots[base as usize];
            let mut best = costs[(start + 1) & mask]
                .checked_add(self.bits[base as usize] as u64)
                .ok_or("encoded field bit count overflow")?;
            let (mut best_len, mut choice) = (1, base);
            for end in start + 1..terminals.len().min(start.saturating_add(self.longest)) {
                let parent = &self.nodes[node as usize];
                let children =
                    &self.edges[parent.first as usize..(parent.first + parent.count) as usize];
                let Ok(index) = children.binary_search_by_key(&terminals[end], |e| e.label) else {
                    break;
                };
                node = children[index].child;
                let id = self.nodes[node as usize].token;
                if id != NONE {
                    let cost = costs[(end + 1) & mask]
                        .checked_add(self.bits[id as usize] as u64)
                        .ok_or("encoded field bit count overflow")?;
                    let length = end + 1 - start;
                    if cost < best
                        || (cost == best
                            && (length > best_len || (length == best_len && id < choice)))
                    {
                        best = cost;
                        best_len = length;
                        choice = id;
                    }
                }
            }
            choices[start] = choice;
            costs[start & mask] = best;
        }
        let output = &mut scratch.symbols;
        output.clear();
        let mut at = 0;
        while at < terminals.len() {
            let id = choices[at];
            output.push(id);
            at += self.lengths[id as usize];
        }
        Ok(output)
    }
}
