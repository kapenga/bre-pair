//! Exact document eligibility at a stage boundary, never in the merge loop.
//! Rejected parents propagate their occurrences and document sets to children.
//! Reverse grammar order reaches the collapse fixed point without reparsing.
use super::{last_seen::Documents, Policy};
use crate::{engine::Compressed, grammar::BASE_SYMBOLS};
use std::time::Instant;

const NONE: usize = usize::MAX;

#[derive(Clone, Copy)]
struct Witness {
    document: u64,
    next: usize,
}

struct Support {
    heads: Vec<usize>,
    counts: Vec<u8>,
    last: Vec<u64>,
    nodes: Vec<Witness>,
    free: Vec<usize>,
    minimum: u8,
}

impl Support {
    fn new(symbols: usize, minimum: u8) -> Self {
        Self {
            heads: vec![NONE; symbols],
            counts: vec![0; symbols],
            last: vec![0; symbols],
            nodes: vec![],
            free: vec![],
            minimum,
        }
    }

    fn observe(&mut self, id: usize, document: u64) {
        if self.counts[id] >= self.minimum || self.last[id] == document {
            return;
        }
        self.last[id] = document;
        let mut at = self.heads[id];
        while at != NONE {
            if self.nodes[at].document == document {
                return;
            }
            at = self.nodes[at].next;
        }
        self.counts[id] += 1;
        if self.counts[id] == self.minimum {
            self.release(id);
        } else {
            let node = Witness {
                document,
                next: self.heads[id],
            };
            let at = if let Some(at) = self.free.pop() {
                self.nodes[at] = node;
                at
            } else {
                self.nodes.push(node);
                self.nodes.len() - 1
            };
            self.heads[id] = at;
        }
    }

    fn release(&mut self, id: usize) {
        let mut at = std::mem::replace(&mut self.heads[id], NONE);
        while at != NONE {
            self.free.push(at);
            at = self.nodes[at].next;
        }
    }

    fn saturate(&mut self, id: usize) {
        self.counts[id] = self.minimum;
        self.release(id);
    }

    fn propagate(&mut self, id: usize, children: [usize; 2], protected: &[bool]) {
        let mut at = std::mem::replace(&mut self.heads[id], NONE);
        while at != NONE {
            let Witness { document, next } = self.nodes[at];
            for child in children {
                if !protected[child] {
                    self.observe(child, document);
                }
            }
            self.free.push(at);
            at = next;
        }
    }
}

fn collect_support(
    source: &Compressed,
    policy: &Policy,
    protected: &[bool],
    round: usize,
) -> Result<(Support, Vec<u64>), String> {
    let alphabet = source.grammar.alphabet_size();
    let documents = Documents::new_at_round(policy, source.input_bytes, source.field_count, round)?;
    let mut cursor = documents.cursor(0);
    let mut support = Support::new(alphabet, policy.minimum_groups as u8);
    let mut uses = vec![0u64; alphabet];
    let total = source.blocks.len() as u64;
    crate::training_progress::phase("document_cleanup_count", round, 0, total, "chunks");
    for (i, block) in source.blocks.iter().enumerate() {
        for symbol in block.iter() {
            let id = symbol as usize;
            if id >= alphabet {
                return Err("undefined document-cleanup symbol".into());
            }
            uses[id] = uses[id]
                .checked_add(1)
                .ok_or("document cleanup count overflow")?;
            if !protected[id] {
                support.observe(id, cursor.document());
            }
            cursor.validate_symbol(symbol, source.grammar.ends_field(symbol), &source.grammar)?;
        }
        if (i + 1) % 64 == 0 {
            crate::training_progress::phase(
                "document_cleanup_count",
                round,
                (i + 1) as u64,
                total,
                "chunks",
            );
        }
    }
    cursor.finish(source.input_bytes, source.field_count)?;
    documents.validate_file()?;
    crate::training_progress::phase("document_cleanup_count", round, total, total, "chunks");
    // Document coordinates are no longer needed during DAG propagation.
    drop(documents);
    Ok((support, uses))
}

/// Candidate admission by document presence anywhere in the encoded derivation,
/// including inside retained parents. Final exact tokenizer usage must still
/// pass the document floor; this is not a raw substring-support inventory.
pub struct CandidateSupport {
    pub included: Vec<bool>,
    pub groups: Vec<u8>,
    pub witness_slots: usize,
    pub seconds: f64,
}

impl CandidateSupport {
    pub fn plan(source: &Compressed, policy: &Policy) -> Result<Self, String> {
        let started = Instant::now();
        let mut protected = vec![false; source.grammar.alphabet_size()];
        protected[..BASE_SYMBOLS].fill(true);
        let (mut support, uses) = collect_support(source, policy, &protected, 0)?;
        drop(uses);
        let minimum = policy.minimum_groups as u8;
        for (i, rule) in source.grammar.rules.iter().enumerate().rev() {
            let id = BASE_SYMBOLS + i;
            let children = [rule.left as usize, rule.right as usize];
            if support.counts[id] >= minimum {
                // Every document containing the parent contains each child.
                // Saturation is sufficient even though its witnesses were freed.
                for child in children {
                    if !protected[child] {
                        support.saturate(child);
                    }
                }
            } else {
                support.propagate(id, children, &protected);
            }
        }
        let included = support
            .counts
            .iter()
            .enumerate()
            .map(|(id, &n)| protected[id] || n >= minimum)
            .collect();
        Ok(Self {
            included,
            groups: support.counts,
            witness_slots: support.nodes.len(),
            seconds: started.elapsed().as_secs_f64(),
        })
    }
}

/// Eligibility and exact output histogram after recursively expanding rejected
/// symbols. Uncoded structural helpers may remain in the decoding grammar.
pub struct Cleanup {
    pub active: Vec<bool>,
    pub uses: Vec<u64>,
    /// Capped exact support at the parent-first decision; protected IDs are 0.
    pub groups: Vec<u8>,
    pub before_symbols: u64,
    pub after_symbols: u64,
    pub removed_rules: usize,
    pub removed_used_rules: usize,
    pub witness_slots: usize,
    pub seconds: f64,
}

impl Cleanup {
    pub fn plan(source: &Compressed, policy: &Policy, protected: &[bool]) -> Result<Self, String> {
        Self::plan_at_round(source, policy, protected, 0)
    }

    pub(crate) fn plan_at_round(
        source: &Compressed,
        policy: &Policy,
        protected: &[bool],
        round: usize,
    ) -> Result<Self, String> {
        let started = Instant::now();
        let alphabet = source.grammar.alphabet_size();
        if protected.len() != alphabet || protected[..BASE_SYMBOLS].contains(&false) {
            return Err(
                "document cleanup must protect the byte/EOF basis and match the grammar".into(),
            );
        }
        let (mut support, mut uses) = collect_support(source, policy, protected, round)?;
        let sum = |counts: &[u64]| {
            counts.iter().try_fold(0u64, |n, &c| {
                n.checked_add(c).ok_or("document cleanup total overflow")
            })
        };
        let before_symbols = sum(&uses)?;
        let mut active = protected.to_vec();
        let (mut removed_rules, mut removed_used_rules) = (0usize, 0usize);
        crate::training_progress::phase(
            "document_cleanup_collapse",
            round,
            0,
            source.grammar.rules.len() as u64,
            "rules",
        );
        for (i, rule) in source.grammar.rules.iter().enumerate().rev() {
            let done = source.grammar.rules.len() - i;
            if done % 65536 == 0 {
                crate::training_progress::phase(
                    "document_cleanup_collapse",
                    round,
                    done as u64,
                    source.grammar.rules.len() as u64,
                    "rules",
                );
            }
            let id = BASE_SYMBOLS + i;
            if protected[id] || support.counts[id] >= policy.minimum_groups as u8 {
                active[id] = true;
            } else {
                removed_rules += 1;
                let count = std::mem::take(&mut uses[id]);
                removed_used_rules += usize::from(count != 0);
                let children = [rule.left as usize, rule.right as usize];
                for child in children {
                    uses[child] = uses[child]
                        .checked_add(count)
                        .ok_or("collapsed document frequency overflow")?;
                }
                support.propagate(id, children, protected);
            }
        }
        let after_symbols = sum(&uses)?;
        crate::training_progress::phase(
            "document_cleanup_collapse",
            round,
            source.grammar.rules.len() as u64,
            source.grammar.rules.len() as u64,
            "rules",
        );
        Ok(Self {
            active,
            uses,
            groups: support.counts,
            before_symbols,
            after_symbols,
            removed_rules,
            removed_used_rules,
            witness_slots: support.nodes.len(),
            seconds: started.elapsed().as_secs_f64(),
        })
    }
}
