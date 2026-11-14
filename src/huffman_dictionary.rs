//! Codec symbols are separate from stable AI terminal IDs. Recipes may join
//! terminals or earlier codec symbols, but may never cross an EOF boundary.
use crate::{
    grammar::Grammar,
    huffman::{get_var, put_var},
    tokenizer::{Model, Token},
};

#[derive(Clone, Copy, Debug)]
pub struct Expansion {
    pub bytes: u64,
    pub terminals: u64,
    pub ends_field: bool,
}

pub struct Dictionary {
    pub model: Model,
    pub rules: Vec<(u32, u32)>,
    pub expansions: Vec<Expansion>,
    /// True only when terminals identify a frozen AI vocabulary.
    pub ai_tokens: bool,
}

/// Decoder input after format-specific validation. Methods are statically
/// dispatched during preparation/fallback; the hot loop uses primitive arrays.
pub(crate) trait DecodeGraph: Sync {
    fn model(&self) -> &Model;
    fn expansions(&self) -> &[Expansion];
    fn ai_tokens(&self) -> bool;
    fn has_rules(&self) -> bool;
    fn terminal(&self, id: usize) -> Option<u32>;
    fn children(&self, id: usize) -> impl DoubleEndedIterator<Item = u32> + ExactSizeIterator;
}

impl DecodeGraph for Dictionary {
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
        !self.rules.is_empty()
    }
    fn terminal(&self, id: usize) -> Option<u32> {
        (id < self.model.tokens.len()).then_some(id as u32)
    }
    fn children(&self, id: usize) -> impl DoubleEndedIterator<Item = u32> + ExactSizeIterator {
        let rule = id
            .checked_sub(self.model.tokens.len())
            .map(|at| self.rules[at]);
        let (a, b) = rule.unwrap_or((0, 0));
        [a, b].into_iter().take(if rule.is_some() { 2 } else { 0 })
    }
}

pub fn byte_model() -> Model {
    Model::new(
        (0..257)
            .map(|id| Token {
                bytes: if id < 256 { vec![id as u8] } else { vec![] },
                ends_field: id == 256,
            })
            .collect(),
    )
    .unwrap()
}

impl Dictionary {
    pub fn new(model: Model, rules: Vec<(u32, u32)>, ai_tokens: bool) -> Result<Self, String> {
        if !ai_tokens && model.to_bytes() != byte_model().to_bytes() {
            return Err("byte grammar must have the 257-symbol basis".into());
        }
        if model
            .tokens
            .len()
            .checked_add(rules.len())
            .ok_or("dictionary overflow")?
            >= u32::MAX as usize
        {
            return Err("dictionary exceeds u32 IDs".into());
        }
        let mut expansions: Vec<_> = model
            .tokens
            .iter()
            .map(|token| Expansion {
                bytes: token.bytes.len() as u64,
                terminals: 1,
                ends_field: token.ends_field,
            })
            .collect();
        for &(left, right) in &rules {
            let a = *expansions
                .get(left as usize)
                .ok_or("undefined recipe child")?;
            let b = *expansions
                .get(right as usize)
                .ok_or("undefined recipe child")?;
            if a.ends_field {
                return Err("recipe crosses a field boundary".into());
            }
            expansions.push(Expansion {
                bytes: a
                    .bytes
                    .checked_add(b.bytes)
                    .ok_or("recipe byte length overflow")?,
                terminals: a
                    .terminals
                    .checked_add(b.terminals)
                    .ok_or("recipe token length overflow")?,
                ends_field: b.ends_field,
            });
        }
        Ok(Self {
            model,
            rules,
            expansions,
            ai_tokens,
        })
    }
    pub fn from_grammar(grammar: &Grammar) -> Result<Self, String> {
        Self::new(
            byte_model(),
            grammar.rules.iter().map(|r| (r.left, r.right)).collect(),
            false,
        )
    }
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        if self.ai_tokens {
            let bytes = self.model.to_bytes();
            put_var(&mut out, bytes.len() as u64);
            out.extend(bytes);
        }
        put_var(&mut out, self.rules.len() as u64);
        for &(left, right) in &self.rules {
            put_var(&mut out, u64::from(left));
            put_var(&mut out, u64::from(right));
        }
        out
    }
    pub fn from_bytes(mut input: &[u8], ai_tokens: bool) -> Result<Self, String> {
        let model = if ai_tokens {
            let size = usize::try_from(get_var(&mut input)?).map_err(|_| "model too large")?;
            let bytes = input.get(..size).ok_or("truncated Huffman model")?;
            let model = Model::from_bytes(bytes)?;
            input = &input[size..];
            model
        } else {
            byte_model()
        };
        let count = usize::try_from(get_var(&mut input)?).map_err(|_| "too many recipes")?;
        if count > input.len() / 2 {
            return Err("recipe counts exceed dictionary size".into());
        }
        let mut rules = Vec::with_capacity(count);
        for _ in 0..count {
            let a = u32::try_from(get_var(&mut input)?).map_err(|_| "child ID overflow")?;
            let b = u32::try_from(get_var(&mut input)?).map_err(|_| "child ID overflow")?;
            rules.push((a, b));
        }
        if !input.is_empty() {
            return Err("trailing dictionary bytes".into());
        }
        Self::new(model, rules, ai_tokens)
    }
}

#[derive(Clone, Copy)]
struct Reference {
    offset: u32,
    length: u32,
}
pub(crate) struct ExpansionReference {
    pub offset: u32,
    pub length: u32,
    pub byte_flags: u32,
    pub terminals: u32,
}
const MISSING: u32 = u32::MAX;

/// Longest expansions first; children reuse slices inside their parent. Eight
/// bytes per reference. Uncached large expansions use an iterative fallback.
pub struct Arena {
    bytes: Vec<u8>,
    references: Vec<Reference>,
    pub token_ids: bool,
}
impl Arena {
    pub fn new(dictionary: &Dictionary, token_ids: bool, budget: usize) -> Result<Self, String> {
        Self::from_graph(dictionary, token_ids, budget)
    }
    pub(crate) fn from_graph<G: DecodeGraph>(
        dictionary: &G,
        token_ids: bool,
        budget: usize,
    ) -> Result<Self, String> {
        if token_ids && !dictionary.ai_tokens() {
            return Err("byte grammar has no frozen AI token IDs".into());
        }
        let length = |id: usize| -> Result<u64, String> {
            let e = dictionary.expansions()[id];
            if token_ids {
                e.terminals
                    .checked_mul(4)
                    .ok_or_else(|| "token byte length overflow".into())
            } else {
                Ok(e.bytes)
            }
        };
        let limit = budget.min((u32::MAX - 1) as usize);
        let mut references = vec![
            Reference {
                offset: MISSING,
                length: 0
            };
            dictionary.expansions().len()
        ];
        let mut order: Vec<_> = (0..references.len()).collect();
        {
            #[cfg(feature = "profile")]
            let _sort = crate::profile::Span::new("huffman.arena.sort_length");
            order.sort_unstable_by_key(|&id| {
                std::cmp::Reverse(if token_ids {
                    dictionary.expansions()[id].terminals
                } else {
                    dictionary.expansions()[id].bytes
                })
            });
        }
        let mut used = 0usize;
        let mut placements = Vec::new();
        let mut stack = Vec::new();
        #[cfg(feature = "profile")]
        let planning = crate::profile::Span::new("huffman.arena.plan_slices");
        for id in order {
            if references[id].offset != MISSING {
                continue;
            }
            let len = length(id)?;
            if len > (limit - used) as u64 {
                continue;
            }
            placements.push((id, used));
            stack.push((id, used));
            while let Some((child, offset)) = stack.pop() {
                if references[child].offset != MISSING {
                    continue;
                }
                references[child] = Reference {
                    offset: offset as u32,
                    length: length(child)? as u32,
                };
                if dictionary.terminal(child).is_none() {
                    let mut end = offset + length(child)? as usize;
                    for c in dictionary.children(child).rev() {
                        end -= length(c as usize)? as usize;
                        stack.push((c as usize, end));
                    }
                }
            }
            used += len as usize;
        }
        #[cfg(feature = "profile")]
        drop(planning);
        // Padding permits one fixed 16-byte copy for a short expansion, while
        // only its logical length is exposed. Allocate it with the arena.
        let mut bytes = {
            #[cfg(feature = "profile")]
            let _allocate = crate::profile::Span::new("huffman.arena.allocate_bytes");
            vec![0u8; used + 16]
        };
        let mut pending = Vec::new();
        #[cfg(feature = "profile")]
        let filling = crate::profile::Span::new("huffman.arena.fill_bytes");
        for (id, start) in placements {
            let mut cursor = start;
            pending.push(id);
            while let Some(child) = pending.pop() {
                let r = references[child];
                let end = r.offset as usize + r.length as usize;
                // Bytes before cursor are already filled, including earlier
                // children of this placement. Reuse those slices too: repeated
                // subtrees must not expand one literal at a time during setup.
                if end <= cursor {
                    bytes.copy_within(r.offset as usize..end, cursor);
                    cursor += r.length as usize;
                } else if let Some(terminal) = dictionary.terminal(child) {
                    if token_ids {
                        bytes[cursor..cursor + 4].copy_from_slice(&terminal.to_le_bytes());
                        cursor += 4;
                    } else {
                        let expansion = &dictionary.model().tokens[terminal as usize].bytes;
                        bytes[cursor..cursor + expansion.len()].copy_from_slice(expansion);
                        cursor += expansion.len();
                    }
                } else {
                    pending.extend(dictionary.children(child).rev().map(|c| c as usize));
                }
            }
            if cursor != start + length(id)? as usize {
                return Err("arena construction length mismatch".into());
            }
        }
        #[cfg(feature = "profile")]
        drop(filling);
        Ok(Self {
            bytes,
            references,
            token_ids,
        })
    }
    pub fn allocated_bytes(&self) -> usize {
        self.bytes.len() + self.references.len() * 8
    }
    pub fn cached(&self, id: u32) -> Option<&[u8]> {
        let r = self.references[id as usize];
        (r.offset != MISSING)
            .then(|| &self.bytes[r.offset as usize..r.offset as usize + r.length as usize])
    }
    pub(crate) fn ranked_references<G: DecodeGraph>(
        &self,
        dictionary: &G,
        ids: &[u32],
    ) -> Option<Vec<(u32, u32)>> {
        if dictionary.ai_tokens() && dictionary.has_rules() {
            return None;
        }
        self.references_for(dictionary, ids)
    }
    /// EOF byte fields do not infer AI-token totals from coded-symbol counts.
    pub(crate) fn field_references<G: DecodeGraph>(
        &self,
        dictionary: &G,
        ids: &[u32],
    ) -> Option<Vec<(u32, u32)>> {
        if self.token_ids {
            return None;
        }
        self.references_for(dictionary, ids)
    }
    fn references_for<G: DecodeGraph>(
        &self,
        dictionary: &G,
        ids: &[u32],
    ) -> Option<Vec<(u32, u32)>> {
        ids.iter()
            .map(|&id| {
                if self.token_ids {
                    let e = dictionary.expansions()[id as usize];
                    return (e.bytes <= 0x7fff_ffff).then_some((
                        dictionary.terminal(id as usize)?,
                        e.bytes as u32 | (u32::from(e.ends_field) << 31),
                    ));
                }
                let r = self.references[id as usize];
                if r.offset == MISSING || r.length > 0x7fff_ffff {
                    None
                } else {
                    Some((
                        r.offset,
                        r.length
                            | (u32::from(dictionary.expansions()[id as usize].ends_field) << 31),
                    ))
                }
            })
            .collect()
    }
    pub(crate) fn data(&self) -> &[u8] {
        &self.bytes
    }
    pub(crate) fn ranked_expansions<G: DecodeGraph>(
        &self,
        dictionary: &G,
        ids: &[u32],
    ) -> Option<Vec<ExpansionReference>> {
        Some(
            ids.iter()
                .map(|&id| {
                    let e = dictionary.expansions()[id as usize];
                    let r = self.references[id as usize];
                    if r.offset == MISSING || e.bytes > 0x7fff_ffff || e.terminals > u32::MAX as u64
                    {
                        ExpansionReference {
                            offset: id,
                            length: u32::MAX,
                            byte_flags: 0,
                            terminals: 0,
                        }
                    } else {
                        ExpansionReference {
                            offset: r.offset,
                            length: r.length,
                            byte_flags: e.bytes as u32 | (u32::from(e.ends_field) << 31),
                            terminals: e.terminals as u32,
                        }
                    }
                })
                .collect(),
        )
    }
    pub fn emit(
        &self,
        dictionary: &Dictionary,
        id: u32,
        stack: &mut Vec<u32>,
        write: impl FnMut(&[u8]) -> Result<(), String>,
    ) -> Result<(), String> {
        self.emit_graph(dictionary, id, stack, write)
    }
    pub(crate) fn emit_graph<G: DecodeGraph>(
        &self,
        dictionary: &G,
        id: u32,
        stack: &mut Vec<u32>,
        mut write: impl FnMut(&[u8]) -> Result<(), String>,
    ) -> Result<(), String> {
        if let Some(bytes) = self.cached(id) {
            return write(bytes);
        }
        stack.push(id);
        while let Some(id) = stack.pop() {
            if let Some(bytes) = self.cached(id) {
                write(bytes)?;
            } else if let Some(terminal) = dictionary.terminal(id as usize) {
                if self.token_ids {
                    write(&terminal.to_le_bytes())?;
                } else {
                    write(&dictionary.model().tokens[terminal as usize].bytes)?;
                }
            } else {
                stack.extend(dictionary.children(id as usize).rev());
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn eof_byte_fields_can_cache_ai_grammar_without_changing_block_or_id_path() {
        let dictionary = Dictionary::new(byte_model(), vec![(97, 98), (257, 256)], true).unwrap();
        let arena = Arena::new(&dictionary, false, 65536).unwrap();
        assert!(arena.ranked_references(&dictionary, &[258]).is_none());
        let field = arena.field_references(&dictionary, &[258]).unwrap();
        assert_eq!(field[0].1, 0x80000002);
        let ids = Arena::new(&dictionary, true, 65536).unwrap();
        assert!(ids.field_references(&dictionary, &[258]).is_none());
        assert!(ids.ranked_references(&dictionary, &[258]).is_none());
        let uncached = Arena::new(&dictionary, false, 0).unwrap();
        assert!(uncached.field_references(&dictionary, &[258]).is_none());
    }
    #[test]
    fn shared_subslices_and_uncached_fallback_preserve_both_outputs() {
        let model = byte_model();
        let d = Dictionary::new(model, vec![(97, 98), (257, 99), (258, 256)], true).unwrap();
        for budget in [0, 2, 2048] {
            for ids in [false, true] {
                let arena = Arena::new(&d, ids, budget).unwrap();
                let mut out: Vec<u8> = Vec::new();
                arena
                    .emit(&d, 259, &mut vec![], |b| {
                        out.extend(b);
                        Ok(())
                    })
                    .unwrap();
                let expected = if ids {
                    [97u32, 98, 99, 256]
                        .into_iter()
                        .flat_map(u32::to_le_bytes)
                        .collect()
                } else {
                    b"abc".to_vec()
                };
                assert_eq!(out, expected);
            }
        }
        let arena = Arena::new(&d, false, 2048).unwrap();
        assert!(arena.allocated_bytes() < 260 * 8 + 260 + 16); // abc and child slices share storage.
        assert!(Dictionary::new(byte_model(), vec![(256, 97)], false).is_err());
        assert!(Dictionary::new(byte_model(), vec![(257, 97)], false).is_err());
    }
    #[test]
    fn deep_recipes_use_no_recursive_expansion() {
        let mut rules = vec![(97, 98)];
        for id in 258..10257 {
            rules.push((id - 1, 97));
        }
        let d = Dictionary::new(byte_model(), rules, false).unwrap();
        let arena = Arena::new(&d, false, 128).unwrap();
        let mut n = 0;
        arena
            .emit(&d, 10256, &mut vec![], |b| {
                n += b.len();
                Ok(())
            })
            .unwrap();
        assert_eq!(n, 10001);
    }
}
