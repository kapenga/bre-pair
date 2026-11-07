//! Fixed-vocabulary estimates from a complete, unchanged training grammar.
//! These counts never read raw training text or use measured parser frequencies.
use std::collections::HashMap;

use crate::{
    engine::Compressed,
    grammar::{BASE_SYMBOLS, EOF},
    tokenizer::{Model, Token},
    vocabulary::Totals,
};

pub struct Estimates {
    /// All derivation occurrences, including inside larger retained pieces.
    /// Includes all aliases, even if some aliases failed candidate admission.
    /// This is construction support, not an emitted-token partition.
    pub inclusive_uses: Vec<u64>,
    /// Stop at each retained piece; otherwise expand its construction children.
    /// This is a valid partition, but does not discover new cross-boundary matches.
    pub collapsed_uses: Vec<u64>,
    pub collapsed: Totals,
    pub matched_rules: usize,
    pub absent_model_tokens: usize,
}

fn add(count: &mut u64, value: u64) -> Result<(), String> {
    *count = count.checked_add(value).ok_or("frequency count overflow")?;
    Ok(())
}

struct Prepared {
    roots: Vec<u64>,
    matched: Vec<u32>,
    present: Vec<bool>,
}

fn prepare(source: &Compressed, model: &Model) -> Result<Prepared, String> {
    Model::new(model.tokens.clone())?;
    let alphabet = source.grammar.alphabet_size();
    let lookup: HashMap<_, _> = model
        .tokens
        .iter()
        .enumerate()
        .map(|(id, t)| (t, id as u32))
        .collect();
    let longest = model.tokens.iter().map(Token::terminals).max().unwrap();
    let mut matched = vec![u32::MAX; alphabet];
    for (id, entry) in matched.iter_mut().take(BASE_SYMBOLS).enumerate() {
        *entry = id as u32;
    }
    let mut probe = Token {
        bytes: Vec::new(),
        ends_field: false,
    };
    let mut stack = Vec::new();
    for (offset, rule) in source.grammar.rules.iter().enumerate() {
        if rule.terminals > longest {
            continue;
        }
        probe.bytes.clear();
        probe.ends_field = rule.ends_field;
        stack.push((BASE_SYMBOLS + offset) as u32);
        while let Some(id) = stack.pop() {
            if id < EOF {
                probe.bytes.push(id as u8);
            } else if id > EOF {
                let child = &source.grammar.rules[id as usize - BASE_SYMBOLS];
                stack.push(child.right);
                stack.push(child.left);
            }
        }
        matched[BASE_SYMBOLS + offset] = lookup.get(&probe).copied().unwrap_or(u32::MAX);
    }
    let mut present = vec![false; model.tokens.len()];
    for &id in &matched {
        if id != u32::MAX {
            present[id as usize] = true;
        }
    }
    let mut roots = vec![0u64; alphabet];
    let total = source.blocks.iter().map(|b| b.len() as u64).sum();
    let progress = crate::training_progress::enabled();
    let mut seen = 0;
    let mut next_progress = 16 * 1024 * 1024;
    crate::training_progress::phase("frequency_estimation", 0, 0, total, "symbols");
    for block in &source.blocks {
        for id in block.iter() {
            add(&mut roots[id as usize], 1)?;
            seen += 1;
            if progress && seen >= next_progress {
                crate::training_progress::phase("frequency_estimation", 0, seen, total, "symbols");
                next_progress = seen.saturating_add(16 * 1024 * 1024);
            }
        }
    }
    crate::training_progress::phase("frequency_estimation", 0, seen, total, "symbols");
    Ok(Prepared {
        roots,
        matched,
        present,
    })
}

/// Reusable dense grammar counters. Building this reads the packed stream once;
/// subsequent vocabulary changes touch only dictionary nodes and counters.
pub struct Counter {
    roots: Vec<u64>,
    matched: Vec<u32>,
    children: Vec<(u32, u32)>,
    tokens: Vec<(u32, bool)>,
    bytes: u64,
    fields: u64,
}
impl Counter {
    pub fn new(source: &Compressed, model: &Model) -> Result<Self, String> {
        let p = prepare(source, model)?;
        Ok(Self {
            roots: p.roots,
            matched: p.matched,
            children: source
                .grammar
                .rules
                .iter()
                .map(|r| (r.left, r.right))
                .collect(),
            tokens: model
                .tokens
                .iter()
                .map(|t| (t.bytes.len() as u32, t.ends_field))
                .collect(),
            bytes: source.input_bytes,
            fields: source.field_count,
        })
    }
    pub fn count(&self, active: &[u8]) -> Result<(Vec<u64>, Totals), String> {
        if active.len() != self.tokens.len() || active[..BASE_SYMBOLS].contains(&0) {
            return Err(
                "estimated counts require a matching active vocabulary and full basis".into(),
            );
        }
        let mut pending = self.roots.clone();
        let mut uses = vec![0; active.len()];
        for id in (0..pending.len()).rev() {
            let n = pending[id];
            if n == 0 {
                continue;
            }
            let target = self.matched[id];
            if target != u32::MAX && active[target as usize] != 0 {
                add(&mut uses[target as usize], n)?;
            } else {
                let (left, right) = self.children[id - BASE_SYMBOLS];
                add(&mut pending[left as usize], n)?;
                add(&mut pending[right as usize], n)?;
            }
        }
        let mut totals = Totals::default();
        for (id, (&(length, ends), &n)) in self.tokens.iter().zip(&uses).enumerate() {
            add(&mut totals.tokens, n)?;
            add(
                &mut totals.bytes,
                n.checked_mul(u64::from(length))
                    .ok_or("estimated byte count overflow")?,
            )?;
            if ends {
                add(&mut totals.fields, n)?;
            }
            if id < 256 {
                add(&mut totals.byte_tokens, n)?;
            }
        }
        if totals.bytes != self.bytes || totals.fields != self.fields {
            return Err("cached grammar frequencies do not conserve bytes and EOFs".into());
        }
        Ok((uses, totals))
    }
}

pub fn from_grammar(source: &Compressed, model: &Model) -> Result<Estimates, String> {
    let Prepared {
        mut roots,
        matched,
        present,
    } = prepare(source, model)?;
    let alphabet = source.grammar.alphabet_size();
    let mut support = roots.clone();
    for (offset, rule) in source.grammar.rules.iter().enumerate().rev() {
        let n = support[BASE_SYMBOLS + offset];
        add(&mut support[rule.left as usize], n)?;
        add(&mut support[rule.right as usize], n)?;
    }
    let mut inclusive_uses = vec![0; model.tokens.len()];
    for (id, &target) in matched.iter().enumerate() {
        if target != u32::MAX {
            add(&mut inclusive_uses[target as usize], support[id])?;
        }
    }
    let mut collapsed_uses = vec![0; model.tokens.len()];
    for id in (0..alphabet).rev() {
        let n = roots[id];
        let target = matched[id];
        if target != u32::MAX {
            add(&mut collapsed_uses[target as usize], n)?;
        } else {
            let rule = &source.grammar.rules[id - BASE_SYMBOLS];
            add(&mut roots[rule.left as usize], n)?;
            add(&mut roots[rule.right as usize], n)?;
        }
    }
    let mut collapsed = Totals::default();
    for (id, (token, &n)) in model.tokens.iter().zip(&collapsed_uses).enumerate() {
        add(&mut collapsed.tokens, n)?;
        add(
            &mut collapsed.bytes,
            n.checked_mul(token.bytes.len() as u64)
                .ok_or("frequency byte total overflow")?,
        )?;
        if token.ends_field {
            add(&mut collapsed.fields, n)?;
        }
        if id < 256 {
            add(&mut collapsed.byte_tokens, n)?;
        }
    }
    if collapsed.bytes != source.input_bytes || collapsed.fields != source.field_count {
        return Err("collapsed frequencies do not conserve original bytes and EOFs".into());
    }
    Ok(Estimates {
        inclusive_uses,
        collapsed_uses,
        collapsed,
        matched_rules: matched[BASE_SYMBOLS..]
            .iter()
            .filter(|&&id| id != u32::MAX)
            .count(),
        absent_model_tokens: present.iter().filter(|&&x| !x).count(),
    })
}
