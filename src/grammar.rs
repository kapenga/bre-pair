use crate::pair_map::PairMap;

pub type Symbol = u32;
pub type Pair = (Symbol, Symbol);
pub const EOF: Symbol = 256;
pub const BASE_SYMBOLS: usize = 257;

#[derive(Clone, Debug)]
pub struct Rule {
    pub left: Symbol,
    pub right: Symbol,
    /// Length of the terminal expansion, including an optional final EOF.
    pub terminals: usize,
    pub ends_field: bool,
}

#[derive(Clone, Debug, Default)]
pub struct Grammar {
    pub rules: Vec<Rule>,
    by_pair: PairMap<Symbol>,
}

impl Grammar {
    pub fn alphabet_size(&self) -> usize {
        BASE_SYMBOLS + self.rules.len()
    }
    pub fn find(&self, pair: Pair) -> Option<Symbol> {
        self.by_pair.get(&pair).copied()
    }

    /// Begin codec discovery above an immutable vocabulary. Existing nodes
    /// remain available for expansion/EOF metadata, but their construction
    /// pairs must not alias newly learned compression recipes to old IDs.
    /// The codec exports only seed roots and subsequently learned recipes;
    /// the internal helper graph is not a canonical BRPAIR01 dictionary.
    pub(crate) fn reset_merge_lookup(&mut self) {
        self.by_pair = PairMap::default();
    }

    pub fn ends_field(&self, symbol: Symbol) -> bool {
        if symbol < BASE_SYMBOLS as Symbol {
            symbol == EOF
        } else {
            self.rules[symbol as usize - BASE_SYMBOLS].ends_field
        }
    }

    pub fn terminals(&self, symbol: Symbol) -> usize {
        if symbol < BASE_SYMBOLS as Symbol {
            1
        } else {
            self.rules[symbol as usize - BASE_SYMBOLS].terminals
        }
    }

    pub fn add(&mut self, pair: Pair) -> Result<Symbol, String> {
        if let Some(symbol) = self.find(pair) {
            return Ok(symbol);
        }
        let id = u32::try_from(self.alphabet_size()).map_err(|_| "symbol IDs exhausted")?;
        if pair.0 >= id || pair.1 >= id {
            return Err("rule refers to an undefined symbol".into());
        }
        if self.ends_field(pair.0) {
            return Err("rule would cross a field boundary".into());
        }
        let terminals = self
            .terminals(pair.0)
            .checked_add(self.terminals(pair.1))
            .ok_or("rule expansion length overflow")?;
        self.rules.push(Rule {
            left: pair.0,
            right: pair.1,
            terminals,
            ends_field: self.ends_field(pair.1),
        });
        self.by_pair.insert(pair, id);
        Ok(id)
    }

    pub fn visit_terminals(&self, symbol: Symbol, mut visit: impl FnMut(Symbol)) {
        let mut stack = vec![symbol];
        while let Some(symbol) = stack.pop() {
            if symbol < BASE_SYMBOLS as Symbol {
                visit(symbol);
            } else {
                let rule = &self.rules[symbol as usize - BASE_SYMBOLS];
                stack.push(rule.right);
                stack.push(rule.left);
            }
        }
    }
}

#[derive(Default)]
struct TrieNode {
    edges: Vec<(Symbol, usize)>,
    token: Option<Symbol>,
}

/// A prototype matching index. It expands dictionary entries, not the active buffer.
pub struct TokenTrie {
    nodes: Vec<TrieNode>,
    lengths: Vec<usize>,
    roots: [usize; BASE_SYMBOLS],
}

impl TokenTrie {
    pub fn new(grammar: &Grammar) -> Self {
        let mut trie = Self {
            nodes: vec![TrieNode::default()],
            lengths: (0..grammar.alphabet_size() as Symbol)
                .map(|symbol| grammar.terminals(symbol))
                .collect(),
            roots: [0; BASE_SYMBOLS],
        };
        for symbol in 0..grammar.alphabet_size() as Symbol {
            let mut node = 0;
            grammar.visit_terminals(symbol, |terminal| {
                let next = trie.nodes[node]
                    .edges
                    .iter()
                    .find(|(label, _)| *label == terminal)
                    .map(|(_, child)| *child);
                node = match next {
                    Some(child) => child,
                    None => {
                        let child = trie.nodes.len();
                        trie.nodes.push(TrieNode::default());
                        trie.nodes[node].edges.push((terminal, child));
                        child
                    }
                };
            });
            // Smallest ID wins when different rules have the same expansion.
            if trie.nodes[node].token.is_none() {
                trie.nodes[node].token = Some(symbol);
            }
        }
        for node in &mut trie.nodes {
            node.edges.sort_unstable();
        }
        for &(terminal, child) in &trie.nodes[0].edges {
            trie.roots[terminal as usize] = child;
        }
        trie
    }

    /// Exact minimum-token parse of the whole unread field suffix, including EOF.
    /// Ties prefer a longer first token, then the smallest symbol ID.
    pub fn segment(&self, bytes: &[u8]) -> Vec<Symbol> {
        self.segment_prefix(bytes, usize::MAX)
    }

    pub fn segment_prefix(&self, bytes: &[u8], max_tokens: usize) -> Vec<Symbol> {
        let length = bytes.len() + 1;
        let terminal = |i: usize| {
            if i < bytes.len() {
                bytes[i] as Symbol
            } else {
                EOF
            }
        };
        // Only the next longest-token-length costs are needed. Keep the exact
        // choices for reconstruction, but do not allocate full cost/step arrays.
        let window = self.lengths.iter().copied().max().unwrap_or(1).min(length) + 1;
        let mut cost = vec![usize::MAX; window];
        let mut choice = vec![0; length];
        let mut position = length % window;
        cost[position] = 0;
        for start in (0..length).rev() {
            position = if position == 0 {
                window - 1
            } else {
                position - 1
            };
            let mut next = if position + 1 == window {
                0
            } else {
                position + 1
            };
            let first = terminal(start);
            let mut node = self.roots[first as usize];
            let mut best_cost = 1 + cost[next];
            let mut best_length = 1;
            choice[start] = first;
            for end in start + 1..length {
                next += 1;
                if next == window {
                    next = 0;
                }
                let edges = &self.nodes[node].edges;
                let Ok(edge) = edges.binary_search_by_key(&terminal(end), |&(label, _)| label)
                else {
                    break;
                };
                node = edges[edge].1;
                if let Some(symbol) = self.nodes[node].token {
                    let next_cost = 1 + cost[next];
                    let token_len = end + 1 - start;
                    if next_cost < best_cost
                        || (next_cost == best_cost
                            && (token_len > best_length
                                || (token_len == best_length && symbol < choice[start])))
                    {
                        best_cost = next_cost;
                        choice[start] = symbol;
                        best_length = token_len;
                    }
                }
            }
            cost[position] = best_cost;
        }
        let mut tokens = Vec::with_capacity(cost[0].min(max_tokens));
        let mut at = 0;
        while at < length && tokens.len() < max_tokens {
            tokens.push(choice[at]);
            at += self.lengths[choice[at] as usize];
        }
        tokens
    }
}
