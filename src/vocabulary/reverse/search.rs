//! Equal-cardinality random proposals inside the unchanged cheap candidate pool.
//! One baseline plus random independent-set exchanges; full training parses
//! choose the winner. No corpus copies or holdout measurements are retained.
use super::{Scan, Symbol, Tokenizer, Totals, Workspace};
use std::time::Instant;

#[derive(Clone, Copy, Debug)]
pub struct Options {
    /// Includes the ordinary deterministic batch. Range1..64.
    pub trials: usize,
    pub seed: u64,
    /// Attempt this many exchanges per batch member in each proposal.
    pub swaps_per_token: usize,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            trials: 8,
            seed: 1,
            swaps_per_token: 4,
        }
    }
}
impl Options {
    pub(super) fn valid(self) -> bool {
        (1..=64).contains(&self.trials) && (1..=64).contains(&self.swaps_per_token)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Stats {
    pub requested: usize,
    pub evaluated: usize,
    pub duplicates: usize,
    pub baseline_tokens: u64,
    pub chosen_tokens: u64,
    pub chosen_trial: usize,
    pub chosen_exchanges: usize,
    pub workspace_bytes: usize,
    pub seconds: f64,
}

struct Random(u64);
impl Random {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut x = self.0;
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
        x ^ (x >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        let bound = n as u64;
        let threshold = bound.wrapping_neg() % bound;
        loop {
            let x = self.next();
            if x >= threshold {
                return (x % bound) as usize;
            }
        }
    }
}

struct Graph {
    ids: Vec<Symbol>,
    offsets: Vec<usize>,
    neighbors: Vec<Symbol>,
    slots: Vec<Symbol>,
}
impl Graph {
    fn new(work: &Workspace, prefix: usize) -> Self {
        let ids: Vec<_> = work.ranked[..prefix].iter().map(|&(_, id)| id).collect();
        let mut offsets = vec![0usize; prefix + 1];
        for (rank, &id) in ids.iter().enumerate() {
            for &child in work.path(id) {
                let other = work.ranks[child as usize];
                if other != Symbol::MAX {
                    offsets[rank + 1] += 1;
                    offsets[other as usize + 1] += 1;
                }
            }
        }
        for at in 0..prefix {
            offsets[at + 1] += offsets[at];
        }
        let mut next = offsets[..prefix].to_vec();
        let mut neighbors = vec![0; offsets[prefix]];
        for (rank, &id) in ids.iter().enumerate() {
            for &child in work.path(id) {
                let other = work.ranks[child as usize];
                if other != Symbol::MAX {
                    neighbors[next[rank]] = other;
                    next[rank] += 1;
                    neighbors[next[other as usize]] = rank as Symbol;
                    next[other as usize] += 1;
                }
            }
        }
        Self {
            ids,
            offsets,
            neighbors,
            slots: vec![Symbol::MAX; prefix],
        }
    }
    fn adjacent(&self, rank: usize) -> &[Symbol] {
        &self.neighbors[self.offsets[rank]..self.offsets[rank + 1]]
    }
    fn reset(&mut self, batch: &[Symbol]) {
        self.slots.fill(Symbol::MAX);
        for (slot, &rank) in batch.iter().enumerate() {
            self.slots[rank as usize] = slot as Symbol;
        }
    }
    fn exchange(&mut self, batch: &mut [Symbol], rank: usize, rng: &mut Random) -> bool {
        if self.slots[rank] != Symbol::MAX {
            return false;
        }
        let mut conflict = None;
        for &other in self.adjacent(rank) {
            if self.slots[other as usize] != Symbol::MAX {
                if conflict.is_some_and(|previous| previous != other) {
                    return false;
                }
                conflict = Some(other);
            }
        }
        let slot = conflict.map_or_else(
            || rng.below(batch.len()),
            |other| self.slots[other as usize] as usize,
        );
        self.slots[batch[slot] as usize] = Symbol::MAX;
        batch[slot] = rank as Symbol;
        self.slots[rank] = slot as Symbol;
        true
    }
    fn independent(&self, batch: &[Symbol]) -> bool {
        batch.iter().all(|&rank| {
            self.adjacent(rank as usize)
                .iter()
                .all(|&other| self.slots[other as usize] == Symbol::MAX)
        })
    }
    fn bytes(&self) -> usize {
        (self.ids.capacity() + self.neighbors.capacity() + self.slots.capacity()) * 4
            + self.offsets.capacity() * std::mem::size_of::<usize>()
    }
}

pub(super) fn choose(
    work: &mut Workspace,
    tokenizer: &Tokenizer,
    active: &mut [u8],
    uses: &[u64],
    context: (Totals, usize),
    options: Options,
    refresh: &mut impl FnMut(&Tokenizer, &[u8]) -> Result<Scan, String>,
) -> Result<(Scan, Stats), String> {
    let started = Instant::now();
    let (before, prefix) = context;
    let mut graph = Graph::new(work, prefix.min(work.ranked.len()));
    let baseline: Vec<_> = work
        .batch
        .iter()
        .map(|&id| work.ranks[id as usize])
        .collect();
    let mut best_ids = work.batch.clone();
    let mut best_scan: Option<Scan> = None;
    let mut seen = Vec::<Vec<Symbol>>::new();
    let mut trial = baseline.clone();
    let mut stats = Stats {
        requested: options.trials,
        ..Default::default()
    };
    let mut rng = Random(options.seed ^ (work.round as u64).wrapping_mul(0xd1342543de82ef95));
    for index in 0..options.trials {
        trial.clone_from(&baseline);
        graph.reset(&trial);
        let mut exchanges = 0;
        if index != 0 {
            for _ in 0..trial.len().saturating_mul(options.swaps_per_token) {
                let rank = rng.below(graph.ids.len());
                exchanges += usize::from(graph.exchange(&mut trial, rank, &mut rng));
            }
        }
        if trial.len() != baseline.len() || !graph.independent(&trial) {
            return Err("random reverse batch changed cardinality or touched a dependency".into());
        }
        let mut ids: Vec<_> = trial.iter().map(|&rank| graph.ids[rank as usize]).collect();
        ids.sort_unstable();
        if seen.contains(&ids) {
            stats.duplicates += 1;
            continue;
        }
        seen.push(ids.clone());
        let mut upper = u128::from(before.tokens);
        for &id in &ids {
            upper += u128::from(uses[id as usize]) * (work.path(id).len() - 1) as u128;
            active[id as usize] = 0;
        }
        let measured = refresh(tokenizer, active);
        for &id in &ids {
            active[id as usize] = 1;
        }
        let measured = measured?;
        if measured.all.bytes != before.bytes
            || measured.all.fields != before.fields
            || measured.all.tokens < before.tokens
            || u128::from(measured.all.tokens) > upper
        {
            return Err("random reverse candidate violates exact parse bounds".into());
        }
        stats.evaluated += 1;
        if index == 0 {
            stats.baseline_tokens = measured.all.tokens;
        }
        // Strict improvement only: ties preserve the deterministic baseline.
        if best_scan
            .as_ref()
            .is_none_or(|best| measured.all.tokens < best.all.tokens)
        {
            best_ids = ids;
            stats.chosen_tokens = measured.all.tokens;
            stats.chosen_trial = index;
            stats.chosen_exchanges = exchanges;
            best_scan = Some(measured);
        }
    }
    stats.workspace_bytes = graph.bytes()
        + baseline.capacity() * 4
        + trial.capacity() * 4
        + best_ids.capacity() * 4
        + seen.iter().map(|v| v.capacity() * 4).sum::<usize>()
        + seen.capacity() * std::mem::size_of::<Vec<Symbol>>();
    work.selected.fill(0);
    for &id in &best_ids {
        work.selected[id as usize] = 1;
    }
    work.batch = best_ids;
    stats.seconds = started.elapsed().as_secs_f64();
    Ok((
        best_scan.ok_or("random reverse search did not evaluate its baseline")?,
        stats,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chooser_returns_minimum_of_every_evaluated_training_parse_and_restores_mask() {
        use crate::{
            grammar::EOF,
            tokenizer::{Model, Token},
        };
        let mut tokens: Vec<_> = (0..=255)
            .map(|b| Token {
                bytes: vec![b],
                ends_field: false,
            })
            .chain([Token {
                bytes: vec![],
                ends_field: true,
            }])
            .collect();
        for bytes in [b"ab".as_slice(), b"abc", b"bc", b"abcd", b"xy"] {
            tokens.push(Token {
                bytes: bytes.to_vec(),
                ends_field: false,
            });
        }
        let tokenizer = Tokenizer::new(Model::new(tokens).unwrap()).unwrap();
        for field in [b"abcabcdxyabc".as_slice(), b"!?"] {
            let mut active = vec![1; tokenizer.model().tokens.len()];
            let parse = |mask: &[u8]| {
                let mut uses = vec![0; mask.len()];
                for id in tokenizer.encode_active(field, Some(mask)) {
                    uses[id as usize] += 1;
                }
                let all = super::super::totals(&tokenizer, &uses, mask).unwrap();
                assert_eq!(uses[EOF as usize], 1);
                Scan {
                    all,
                    uses,
                    heldout: Default::default(),
                    seconds: 0.0,
                }
            };
            let initial = parse(&active);
            let mut work = Workspace::default();
            work.plan(
                &tokenizer,
                &active,
                &initial.uses,
                &vec![0; active.len()],
                5,
            )
            .unwrap();
            let mut baseline = work.batch.clone();
            baseline.sort_unstable();
            let mut observed = Vec::new();
            let (best, stats) = choose(
                &mut work,
                &tokenizer,
                &mut active,
                &initial.uses,
                (initial.all, 5),
                Options::default(),
                &mut |_, mask| {
                    assert_eq!(mask.iter().filter(|&&on| on == 0).count(), baseline.len());
                    let parsed = parse(mask);
                    observed.push(parsed.all.tokens);
                    Ok(parsed)
                },
            )
            .unwrap();
            assert!(active.iter().all(|&on| on == 1));
            assert_eq!(best.all.tokens, *observed.iter().min().unwrap());
            assert_eq!(stats.chosen_tokens, best.all.tokens);
            assert_eq!(stats.evaluated, observed.len());
            for &id in &work.batch {
                assert_eq!(best.uses[id as usize], 0);
            }
            if field == b"!?" {
                assert_eq!(stats.chosen_trial, 0);
                work.batch.sort_unstable();
                assert_eq!(work.batch, baseline);
            }
        }
    }

    #[test]
    fn exchanges_preserve_size_and_bidirectional_independence_with_repeated_edges() {
        let mut g = Graph {
            ids: vec![257, 258, 259, 260],
            offsets: vec![0, 2, 5, 6, 6],
            neighbors: vec![1, 1, 0, 0, 2, 1],
            slots: vec![Symbol::MAX; 4],
        };
        let mut batch = vec![0, 2];
        g.reset(&batch);
        let mut rng = Random(1);
        assert!(!g.exchange(&mut batch, 1, &mut rng)); // touches two chosen nodes
        assert!(g.exchange(&mut batch, 3, &mut rng)); // isolated alternate
        assert!(g.independent(&batch));
        for _ in 0..1000 {
            let rank = rng.below(4);
            g.exchange(&mut batch, rank, &mut rng);
            assert_eq!(batch.len(), 2);
            assert!(g.independent(&batch));
        }
    }
}
