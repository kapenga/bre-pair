//! Research-only variable-arity dictionary prototype. Existing BRCODE01 and
//! BRHUFF02 readers/writers are unchanged. AI-ID expansions and Huffman lengths
//! are invariant; serialized model bytes decide between construction variants.
use super::{Codebook, Dictionary, Frozen, Totals, NONE};
use crate::{
    huffman::{get_var, put_var},
    huffman_dictionary::Expansion,
    pair_map::PairMap,
    tokenizer::{fingerprint, Model},
};

const MAGIC: &[u8; 8] = b"BRMGP001";

pub struct Compact {
    pub model: Model,
    pub rules: Vec<Vec<u32>>,
    pub book: Codebook,
    pub training: Totals,
    expansions: Vec<Expansion>,
}

fn reference_bytes(id: u32) -> usize {
    ((32 - id.leading_zeros()).max(1) as usize).div_ceil(7)
}

impl Compact {
    fn new(
        model: Model,
        rules: Vec<Vec<u32>>,
        book: Codebook,
        training: Totals,
    ) -> Result<Self, String> {
        let mut expansions: Vec<_> = model
            .tokens
            .iter()
            .map(|t| Expansion {
                bytes: t.bytes.len() as u64,
                terminals: 1,
                ends_field: t.ends_field,
            })
            .collect();
        for rhs in &rules {
            if rhs.len() < 2 {
                return Err("compact recipe needs at least two children".into());
            }
            let mut total = Expansion {
                bytes: 0,
                terminals: 0,
                ends_field: false,
            };
            for &id in rhs {
                if total.ends_field {
                    return Err("compact recipe crosses EOF".into());
                }
                let child = expansions
                    .get(id as usize)
                    .ok_or("compact child is not earlier")?;
                total.bytes = total
                    .bytes
                    .checked_add(child.bytes)
                    .ok_or("compact byte length overflow")?;
                total.terminals = total
                    .terminals
                    .checked_add(child.terminals)
                    .ok_or("compact token length overflow")?;
                total.ends_field = child.ends_field;
            }
            expansions.push(total);
        }
        let mut covered = vec![false; expansions.len()];
        for &id in &book.symbols {
            *covered
                .get_mut(id as usize)
                .ok_or("compact code exceeds dictionary")? = true;
        }
        if covered[..model.tokens.len()].contains(&false) {
            return Err("compact dictionary lost AI fallback".into());
        }
        Ok(Self {
            model,
            rules,
            book,
            training,
            expansions,
        })
    }

    /// A self-contained *experimental* model, not a production archive. Binary
    /// arity is implicit; a sparse exception table describes longer recipes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = MAGIC.to_vec();
        for n in [
            self.training.bytes,
            self.training.fields,
            self.training.tokens,
            self.training.symbols,
        ] {
            out.extend(n.to_le_bytes());
        }
        let model = self.model.to_bytes();
        put_var(&mut out, model.len() as u64);
        out.extend(model);
        put_var(&mut out, self.rules.len() as u64);
        put_var(
            &mut out,
            self.rules.iter().filter(|r| r.len() != 2).count() as u64,
        );
        let mut previous = 0;
        for (i, rhs) in self.rules.iter().enumerate() {
            if rhs.len() != 2 {
                put_var(&mut out, (i - previous) as u64);
                put_var(&mut out, rhs.len() as u64);
                previous = i;
            }
        }
        for rhs in &self.rules {
            for &id in rhs {
                put_var(&mut out, u64::from(id));
            }
        }
        let book = self.book.to_bytes();
        put_var(&mut out, self.book.symbols.len() as u64);
        put_var(&mut out, book.len() as u64);
        out.extend(book);
        out.extend(fingerprint(&out).to_le_bytes());
        out
    }

    pub fn from_bytes(input: &[u8]) -> Result<Self, String> {
        if input.len() < 48 || &input[..8] != MAGIC {
            return Err("invalid experimental recipe model".into());
        }
        let end = input.len() - 8;
        if fingerprint(&input[..end]) != u64::from_le_bytes(input[end..].try_into().unwrap()) {
            return Err("experimental recipe checksum differs".into());
        }
        let number = |at| u64::from_le_bytes(input[at..at + 8].try_into().unwrap());
        let training = Totals {
            bytes: number(8),
            fields: number(16),
            tokens: number(24),
            symbols: number(32),
        };
        let mut input = &input[40..end];
        let model_len =
            usize::try_from(get_var(&mut input)?).map_err(|_| "compact vocabulary too large")?;
        let model_bytes = input
            .get(..model_len)
            .ok_or("truncated compact vocabulary")?;
        let model = Model::from_bytes(model_bytes)?;
        input = &input[model_len..];
        let n = usize::try_from(get_var(&mut input)?).map_err(|_| "too many compact recipes")?;
        if n > input.len() / 2 {
            return Err("compact recipe count exceeds input".into());
        }
        let exceptions =
            usize::try_from(get_var(&mut input)?).map_err(|_| "too many arity exceptions")?;
        if exceptions > n {
            return Err("too many arity exceptions".into());
        }
        let mut arities = vec![2; n];
        let mut previous = 0usize;
        for i in 0..exceptions {
            let delta =
                usize::try_from(get_var(&mut input)?).map_err(|_| "arity delta overflow")?;
            if i != 0 && delta == 0 {
                return Err("duplicate arity exception".into());
            }
            let index = previous.checked_add(delta).ok_or("arity index overflow")?;
            let arity = usize::try_from(get_var(&mut input)?).map_err(|_| "arity overflow")?;
            if arity < 3 || arity > input.len() {
                return Err("invalid arity exception".into());
            }
            *arities
                .get_mut(index)
                .ok_or("arity index exceeds recipes")? = arity;
            previous = index;
        }
        let mut rules = Vec::with_capacity(n);
        for arity in arities {
            if arity > input.len() {
                return Err("truncated compact children".into());
            }
            let mut rhs = Vec::with_capacity(arity);
            for _ in 0..arity {
                rhs.push(
                    u32::try_from(get_var(&mut input)?).map_err(|_| "compact child ID overflow")?,
                );
            }
            rules.push(rhs);
        }
        let used =
            usize::try_from(get_var(&mut input)?).map_err(|_| "compact code count overflow")?;
        let bytes =
            usize::try_from(get_var(&mut input)?).map_err(|_| "compact table size overflow")?;
        if bytes != input.len() {
            return Err("compact table size differs".into());
        }
        let book = Codebook::from_bytes(input, used, model.tokens.len() + rules.len())?;
        Self::new(model, rules, book, training)
    }

    fn next_terminal(&self, stack: &mut Vec<u32>) -> Option<u32> {
        while let Some(id) = stack.pop() {
            if (id as usize) < self.model.tokens.len() {
                return Some(id);
            }
            stack.extend(
                self.rules[id as usize - self.model.tokens.len()]
                    .iter()
                    .rev()
                    .copied(),
            );
        }
        None
    }

    pub fn emit(
        &self,
        id: u32,
        token_ids: bool,
        mut write: impl FnMut(&[u8]) -> Result<(), String>,
    ) -> Result<(), String> {
        if id as usize >= self.expansions.len() {
            return Err("undefined compact symbol".into());
        }
        let mut stack = vec![id];
        while let Some(id) = self.next_terminal(&mut stack) {
            if token_ids {
                write(&id.to_le_bytes())?;
            } else {
                write(&self.model.tokens[id as usize].bytes)?;
            }
        }
        Ok(())
    }
}

pub struct Variant {
    pub name: &'static str,
    pub model_bytes: usize,
    pub rules: usize,
    pub nonbinary_rules: usize,
    pub child_references: usize,
    pub changed_definitions: usize,
    pub removed_helpers: usize,
}
pub struct Analysis {
    pub variants: Vec<Variant>,
    pub selected: &'static str,
    pub compact: Compact,
    /// Original coded IDs to compact IDs. Unreachable helpers map to NONE.
    pub mapping: Vec<u32>,
}

fn source_next(dictionary: &Dictionary, stack: &mut Vec<u32>) -> Option<u32> {
    while let Some(id) = stack.pop() {
        if (id as usize) < dictionary.model.tokens.len() {
            return Some(id);
        }
        let (a, b) = dictionary.rules[id as usize - dictionary.model.tokens.len()];
        stack.push(b);
        stack.push(a);
    }
    None
}

fn compact(source: &Frozen, definitions: &[Vec<u32>]) -> Result<(Compact, Vec<u32>), String> {
    let basis = source.dictionary.model.tokens.len();
    let mut needed = vec![false; source.dictionary.expansions.len()];
    let mut stack = source.book.symbols.clone();
    while let Some(id) = stack.pop() {
        if std::mem::replace(&mut needed[id as usize], true) {
            continue;
        }
        if id as usize >= basis {
            stack.extend(&definitions[id as usize - basis]);
        }
    }
    let mut order: Vec<_> = (basis..needed.len()).filter(|&id| needed[id]).collect();
    order.sort_unstable_by_key(|&id| (source.dictionary.expansions[id].terminals, id));
    let mut mapping = vec![NONE; needed.len()];
    for (id, slot) in mapping[..basis].iter_mut().enumerate() {
        *slot = id as u32;
    }
    let mut rules = Vec::with_capacity(order.len());
    for id in order {
        let rhs = definitions[id - basis]
            .iter()
            .map(|&child| {
                if mapping[child as usize] == NONE {
                    Err("recipe child is not strictly shorter")
                } else {
                    Ok(mapping[child as usize])
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        mapping[id] = (basis + rules.len()) as u32;
        rules.push(rhs);
    }
    let lengths: Vec<_> = source
        .book
        .lengths()
        .map(|(id, n)| (mapping[id as usize], n))
        .collect();
    let result = Compact::new(
        source.dictionary.model.clone(),
        rules,
        Codebook::from_lengths(&lengths)?,
        source.training,
    )?;
    // Verify *AI ID sequences*, not only byte equality, for every coded root.
    let (mut left, mut right) = (Vec::new(), Vec::new());
    for &id in &source.book.symbols {
        left.push(id);
        right.push(mapping[id as usize]);
        loop {
            let a = source_next(&source.dictionary, &mut left);
            let b = result.next_terminal(&mut right);
            if a != b {
                return Err("recipe optimization changed an AI ID sequence".into());
            }
            if a.is_none() {
                break;
            }
        }
    }
    Ok((result, mapping))
}

fn inline(source: &Frozen, definitions: &[Vec<u32>], maximum_rhs: usize) -> Vec<Vec<u32>> {
    let basis = source.dictionary.model.tokens.len();
    let mut coded = vec![false; source.dictionary.expansions.len()];
    for &id in &source.book.symbols {
        coded[id as usize] = true;
    }
    let mut live = coded.clone();
    let mut stack = source.book.symbols.clone();
    while let Some(id) = stack.pop() {
        if id as usize >= basis {
            for &child in &definitions[id as usize - basis] {
                if !std::mem::replace(&mut live[child as usize], true) {
                    stack.push(child);
                }
            }
        }
    }
    let mut uses = vec![0u64; live.len()];
    for (id, rhs) in definitions.iter().enumerate() {
        if live[basis + id] {
            for &child in rhs {
                uses[child as usize] += 1;
            }
        }
    }
    let mut output = Vec::with_capacity(definitions.len());
    for rhs in definitions {
        let mut next = Vec::new();
        stack.extend(rhs.iter().rev().copied());
        while let Some(id) = stack.pop() {
            if id as usize >= basis && !coded[id as usize] && uses[id as usize] == 1 {
                stack.extend(definitions[id as usize - basis].iter().rev().copied());
            } else {
                next.push(id);
            }
            if next.len() + stack.len() > maximum_rhs {
                next.clone_from(rhs);
                stack.clear();
                break;
            }
        }
        output.push(next);
    }
    output
}

struct Node {
    first: usize,
    count: usize,
    symbol: u32,
}
struct Trie {
    nodes: Vec<Node>,
    edges: Vec<(u32, u32)>,
    roots: Vec<u32>,
}
impl Trie {
    fn new(source: &Frozen, maximum_terms: usize) -> Result<Self, String> {
        let mut nodes = vec![Node {
            first: 0,
            count: 0,
            symbol: NONE,
        }];
        let mut links = Vec::new();
        let mut index = PairMap::default();
        let mut stack = Vec::new();
        for (id, expansion) in source.dictionary.expansions.iter().enumerate() {
            if expansion.terminals > maximum_terms as u64 {
                continue;
            }
            stack.push(id as u32);
            let mut node = 0u32;
            while let Some(symbol) = source_next(&source.dictionary, &mut stack) {
                node = if let Some(&child) = index.get(&(node, symbol)) {
                    child
                } else {
                    let child =
                        u32::try_from(nodes.len()).map_err(|_| "recipe trie exceeds u32")?;
                    if child == NONE {
                        return Err("recipe trie exhausted IDs".into());
                    }
                    nodes.push(Node {
                        first: 0,
                        count: 0,
                        symbol: NONE,
                    });
                    index.insert((node, symbol), child);
                    links.push((node, symbol, child));
                    child
                };
            }
            let old = &mut nodes[node as usize].symbol;
            if *old == NONE
                || (reference_bytes(id as u32), id as u32) < (reference_bytes(*old), *old)
            {
                *old = id as u32;
            }
        }
        drop(index);
        links.sort_unstable();
        let mut edges = Vec::with_capacity(links.len());
        for (parent, label, child) in links {
            let node = &mut nodes[parent as usize];
            if node.count == 0 {
                node.first = edges.len();
            }
            node.count += 1;
            edges.push((label, child));
        }
        let mut roots = vec![NONE; source.dictionary.model.tokens.len()];
        for &(label, child) in &edges[..nodes[0].count] {
            roots[label as usize] = child;
        }
        Ok(Self {
            nodes,
            edges,
            roots,
        })
    }
    fn reparse(&self, terminals: &[u32]) -> Vec<u32> {
        let n = terminals.len();
        let mut costs = vec![(0usize, 0usize); n + 1];
        let mut choices = vec![(0u32, 0usize); n];
        for start in (0..n).rev() {
            let id = terminals[start];
            let mut best = (
                costs[start + 1].0 + reference_bytes(id),
                costs[start + 1].1 + 1,
            );
            choices[start] = (id, 1);
            let mut node = self.roots[id as usize];
            for end in start + 1..n {
                let parent = &self.nodes[node as usize];
                let children = &self.edges[parent.first..parent.first + parent.count];
                let Ok(found) = children.binary_search_by_key(&terminals[end], |e| e.0) else {
                    break;
                };
                node = children[found].1;
                let id = self.nodes[node as usize].symbol;
                let length = end + 1 - start;
                if id != NONE && length < n {
                    let cost = (costs[end + 1].0 + reference_bytes(id), costs[end + 1].1 + 1);
                    if cost < best
                        || (cost == best
                            && (length > choices[start].1
                                || (length == choices[start].1 && id < choices[start].0)))
                    {
                        best = cost;
                        choices[start] = (id, length);
                    }
                }
            }
            costs[start] = best;
        }
        let mut output = Vec::new();
        let mut at = 0;
        while at < n {
            output.push(choices[at].0);
            at += choices[at].1;
        }
        output
    }
}

pub fn analyze(source: &Frozen, maximum_terms: usize) -> Result<Analysis, String> {
    source.validate()?;
    if !(2..=4096).contains(&maximum_terms) {
        return Err("recipe reparse bound must be 2..4096 AI IDs".into());
    }
    let original: Vec<_> = source
        .dictionary
        .rules
        .iter()
        .map(|&(a, b)| vec![a, b])
        .collect();
    let trie = Trie::new(source, maximum_terms)?;
    let mut reparsed = Vec::with_capacity(original.len());
    let mut stack = Vec::new();
    let mut terminals = Vec::new();
    let basis = source.dictionary.model.tokens.len();
    for (i, rhs) in original.iter().enumerate() {
        if source.dictionary.expansions[basis + i].terminals > maximum_terms as u64 {
            reparsed.push(rhs.clone());
            continue;
        }
        terminals.clear();
        stack.push((basis + i) as u32);
        while let Some(id) = source_next(&source.dictionary, &mut stack) {
            terminals.push(id);
        }
        let next = trie.reparse(&terminals);
        let cost = |r: &[u32]| {
            (
                r.iter().map(|&id| reference_bytes(id)).sum::<usize>(),
                r.len(),
            )
        };
        if (cost(&next), &next) < (cost(rhs), rhs) {
            reparsed.push(next);
        } else {
            reparsed.push(rhs.clone());
        }
    }
    let inlined = inline(source, &original, maximum_terms);
    let reparsed_inlined = inline(source, &reparsed, maximum_terms);
    let candidates = [
        ("binary", original.clone()),
        ("inline", inlined),
        ("reparse", reparsed),
        ("reparse-inline", reparsed_inlined),
    ];
    let mut variants = Vec::new();
    let mut best = None;
    let mut selected = "binary";
    for (name, definitions) in candidates {
        let (compact, mapping) = compact(source, &definitions)?;
        let bytes = compact.to_bytes();
        // Round-trip through the prototype's actual format, then check all roots
        // again; cost statistics alone are not a decoder correctness check.
        let decoded = Compact::from_bytes(&bytes)?;
        let mut left = Vec::new();
        let mut right = Vec::new();
        for &id in &source.book.symbols {
            left.push(id);
            right.push(mapping[id as usize]);
            loop {
                let a = source_next(&source.dictionary, &mut left);
                let b = decoded.next_terminal(&mut right);
                if a != b {
                    return Err("serialized recipe changed AI IDs".into());
                }
                if a.is_none() {
                    break;
                }
            }
        }
        variants.push(Variant {
            name,
            model_bytes: bytes.len(),
            rules: compact.rules.len(),
            nonbinary_rules: compact.rules.iter().filter(|r| r.len() != 2).count(),
            child_references: compact.rules.iter().map(Vec::len).sum(),
            changed_definitions: definitions
                .iter()
                .zip(&original)
                .filter(|(a, b)| a != b)
                .count(),
            removed_helpers: source.dictionary.rules.len() - compact.rules.len(),
        });
        if best.as_ref().is_none_or(|(n, _, _)| bytes.len() < *n) {
            best = Some((bytes.len(), compact, mapping));
            selected = name;
        }
    }
    let (_, compact, mapping) = best.unwrap();
    Ok(Analysis {
        variants,
        selected,
        compact,
        mapping,
    })
}
