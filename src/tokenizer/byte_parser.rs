//! Reversed Aho-Corasick matching fused with exact backward segmentation.
//! Failure/output links reuse suffix matches instead of restarting a trie at
//! every byte. The frozen model is shared; only scratch belongs to a worker.
use super::{Edge, Model, Symbol, Tokenizer, BASE_SYMBOLS, EOF, NONE};

#[derive(Default)]
pub struct ParseScratch {
    costs: Vec<u64>,
    choices: Vec<Symbol>,
    pub(super) ids: Vec<Symbol>,
}

pub struct ByteParser {
    model: Model,
    nodes: Vec<ParseNode>,
    edges: Vec<Edge>,
    roots: [u32; BASE_SYMBOLS],
    second: Box<[u32]>,
    longest: usize,
    lengths: Vec<usize>,
}

struct ParseNode {
    first: u32,
    count: u16,
    token: u32,
    failure: u32,
    matched: u32,
}

impl ByteParser {
    pub fn new(model: Model) -> Result<Self, String> {
        let Tokenizer {
            model,
            nodes,
            edges,
            roots,
            longest,
        } = Tokenizer::build(model, true)?;
        let lengths = model.tokens.iter().map(|t| t.terminals()).collect();
        // Only depth-one nodes have basis-token IDs. Cache their second
        // transitions in 257x257 u32 entries (264 KiB), shared by all workers.
        let mut second = vec![NONE; BASE_SYMBOLS * BASE_SYMBOLS].into_boxed_slice();
        for (label, &root) in roots.iter().enumerate() {
            let node = &nodes[root as usize];
            for edge in &edges[node.first as usize..node.first as usize + node.count as usize] {
                second[label * BASE_SYMBOLS + edge.label as usize] = edge.child;
            }
        }
        let mut parser = Self {
            nodes: nodes
                .into_iter()
                .map(|n| ParseNode {
                    first: n.first,
                    count: n.count,
                    token: n.token,
                    failure: 0,
                    matched: NONE,
                })
                .collect(),
            model,
            edges,
            roots,
            longest,
            lengths,
            second,
        };
        let root = &parser.nodes[0];
        let mut queue: Vec<u32> = parser.edges[..root.count as usize]
            .iter()
            .map(|e| e.child)
            .collect();
        let mut at = 0;
        while at < queue.len() {
            let parent = queue[at] as usize;
            let node = &parser.nodes[parent];
            let range = node.first as usize..node.first as usize + node.count as usize;
            for i in range {
                let edge = &parser.edges[i];
                let (label, child) = (edge.label, edge.child);
                let failure = parser.advance(parser.nodes[parent].failure, label);
                parser.nodes[child as usize].failure = failure;
                let token = parser.nodes[child as usize].token;
                parser.nodes[child as usize].matched =
                    if token != NONE && token as usize >= BASE_SYMBOLS {
                        child
                    } else {
                        parser.nodes[failure as usize].matched
                    };
                queue.push(child);
            }
            at += 1;
        }
        Ok(parser)
    }

    pub fn model(&self) -> &Model {
        &self.model
    }

    /// Research overlap audit: report all non-basis byte matches without a DP
    /// or invented EOF. The compact automaton is shared with exact parsing.
    pub fn visit_fragment_matches(&self, bytes: &[u8], mut visit: impl FnMut(Symbol)) {
        let mut state = 0;
        for &byte in bytes.iter().rev() {
            state = self.advance(state, u16::from(byte));
            let mut matched = self.nodes[state as usize].matched;
            while matched != NONE {
                let node = &self.nodes[matched as usize];
                if node.token as usize >= BASE_SYMBOLS {
                    visit(node.token);
                }
                matched = self.nodes[node.failure as usize].matched;
            }
        }
    }

    #[inline]
    fn advance(&self, mut node: u32, label: u16) -> u32 {
        loop {
            if node == 0 {
                return self.roots[label as usize];
            }
            let state = &self.nodes[node as usize];
            if (state.token as usize) < BASE_SYMBOLS {
                let child = self.second[state.token as usize * BASE_SYMBOLS + label as usize];
                return if child == NONE {
                    self.roots[label as usize]
                } else {
                    child
                };
            }
            let edges =
                &self.edges[state.first as usize..state.first as usize + state.count as usize];
            if let Ok(i) = edges.binary_search_by_key(&label, |e| e.label) {
                return edges[i].child;
            }
            node = state.failure;
        }
    }

    /// Same minimum-token objective and tie order as Tokenizer::encode,
    /// including the virtual EOF. No normalized text or approximate matches.
    pub fn encode<'a>(&self, bytes: &[u8], scratch: &'a mut ParseScratch) -> &'a [Symbol] {
        let n = bytes.len() + 1;
        let window = (self.longest.min(n) + 1).next_power_of_two();
        if scratch.costs.len() < window {
            scratch.costs.resize(window, 0);
        }
        if scratch.choices.len() < n {
            scratch.choices.resize(n, 0);
        }
        let mask = window - 1;
        scratch.costs[n & mask] = 0;
        let mut state = 0;
        for start in (0..n).rev() {
            let label = bytes.get(start).map_or(EOF as u16, |&b| u16::from(b));
            state = self.advance(state, label);
            let mut best = scratch.costs[(start + 1) & mask] + 1;
            let (mut best_len, mut choice) = (1, label as Symbol);
            let mut matched = self.nodes[state as usize].matched;
            while matched != NONE {
                let node = &self.nodes[matched as usize];
                let token = node.token;
                let length = self.lengths[token as usize];
                let cost = scratch.costs[(start + length) & mask] + 1;
                if cost < best
                    || (cost == best
                        && (length > best_len || (length == best_len && token < choice)))
                {
                    best = cost;
                    best_len = length;
                    choice = token;
                }
                matched = self.nodes[node.failure as usize].matched;
            }
            scratch.choices[start] = choice;
            scratch.costs[start & mask] = best;
        }
        scratch.ids.clear();
        scratch.ids.reserve(scratch.costs[0] as usize);
        let mut at = 0;
        while at < n {
            let id = scratch.choices[at];
            scratch.ids.push(id);
            at += self.lengths[id as usize];
        }
        &scratch.ids
    }
}

#[cfg(test)]
mod overlap_tests {
    use super::*;
    use crate::tokenizer::Token;
    #[test]
    fn compact_matcher_reports_all_overlaps_and_no_synthetic_eof() {
        let mut tokens: Vec<_> = (0..256)
            .map(|b| Token {
                bytes: vec![b as u8],
                ends_field: false,
            })
            .collect();
        tokens.push(Token {
            bytes: Vec::new(),
            ends_field: true,
        });
        for (raw, ends_field) in [
            (b"aa".as_slice(), false),
            (b"aaa", false),
            (b"a", true),
            (b"\0a", false),
        ] {
            tokens.push(Token {
                bytes: raw.to_vec(),
                ends_field,
            });
        }
        let model = Model::new(tokens).unwrap();
        let simple = Tokenizer::new(model.clone()).unwrap();
        let parser = ByteParser::new(model).unwrap();
        for raw in [b"aaaaa".as_slice(), b"\0aaa\0a", b"", b"a\xff"] {
            let mut expected = Vec::new();
            simple
                .matching_edges(raw, |_, _, id| {
                    if id >= 257 && !simple.model.tokens[id as usize].ends_field {
                        expected.push(id);
                    }
                    Ok(())
                })
                .unwrap();
            let mut actual = Vec::new();
            parser.visit_fragment_matches(raw, |id| actual.push(id));
            expected.sort_unstable();
            actual.sort_unstable();
            assert_eq!(actual, expected);
        }
    }
}
