//! Longest-prefix parsing with reusable output and one shared forward trie.
use super::{Model, ParseScratch, Symbol, Tokenizer, BASE_SYMBOLS, EOF, NONE};

pub struct GreedyParser {
    tokenizer: Tokenizer,
    second: Box<[u32]>,
}

impl GreedyParser {
    pub fn new(model: Model) -> Result<Self, String> {
        let tokenizer = Tokenizer::new(model)?;
        let mut second = vec![NONE; BASE_SYMBOLS * BASE_SYMBOLS].into_boxed_slice();
        for (label, &root) in tokenizer.roots.iter().enumerate() {
            let node = &tokenizer.nodes[root as usize];
            for edge in
                &tokenizer.edges[node.first as usize..node.first as usize + node.count as usize]
            {
                second[label * BASE_SYMBOLS + edge.label as usize] = edge.child;
            }
        }
        Ok(Self { tokenizer, second })
    }

    pub fn model(&self) -> &Model {
        self.tokenizer.model()
    }

    /// Choose the longest matching token at each position, including virtual
    /// EOF. This is lossless but need not minimize token count or match exact IDs.
    pub fn encode<'a>(&self, bytes: &[u8], scratch: &'a mut ParseScratch) -> &'a [Symbol] {
        scratch.ids.clear();
        let n = bytes.len() + 1;
        let mut at = 0;
        while at < n {
            let label = bytes.get(at).map_or(EOF as u16, |&b| u16::from(b));
            let mut choice = u32::from(label);
            let mut length = 1;
            let stop = n.min(at.saturating_add(self.tokenizer.longest));
            if at + 1 < stop {
                let next = bytes.get(at + 1).map_or(EOF as u16, |&b| u16::from(b));
                let mut node = self.second[label as usize * BASE_SYMBOLS + next as usize];
                if node != NONE {
                    let id = self.tokenizer.nodes[node as usize].token;
                    if id != NONE {
                        choice = id;
                        length = 2;
                    }
                    for end in at + 2..stop {
                        let label = bytes.get(end).map_or(EOF as u16, |&b| u16::from(b));
                        let Some(child) = self.tokenizer.child(node, label) else {
                            break;
                        };
                        node = child;
                        let id = self.tokenizer.nodes[node as usize].token;
                        if id != NONE {
                            choice = id;
                            length = end + 1 - at;
                        }
                    }
                }
            }
            scratch.ids.push(choice);
            at += length;
        }
        &scratch.ids
    }
}
