//! Independent-set searches for equal-frequency conflict components.
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::str::FromStr;

/// Extra-wave floor per decimal GB, rounded up without narrowing corpus sizes.
pub fn scaled_fill_minimum(absolute: u64, per_gb: u64, bytes: u64) -> Result<u64, String> {
    let scaled = (u128::from(per_gb) * u128::from(bytes)).div_ceil(1_000_000_000);
    let scaled = u64::try_from(scaled).map_err(|_| "scaled fill count exceeds u64")?;
    Ok(absolute.max(scaled))
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Strategy {
    #[default]
    Greedy,
    Beam,
    Hybrid,
}

impl Strategy {
    pub fn name(self) -> &'static str {
        match self {
            Self::Greedy => "greedy",
            Self::Beam => "beam",
            Self::Hybrid => "hybrid",
        }
    }
}

impl FromStr for Strategy {
    type Err = String;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "greedy" => Ok(Self::Greedy),
            "beam" => Ok(Self::Beam),
            "hybrid" => Ok(Self::Hybrid),
            _ => Err("selector must be greedy, beam, or hybrid".into()),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TieBreak {
    #[default]
    Lex,
    Reverse,
    Seeded,
}

impl TieBreak {
    pub fn name(self) -> &'static str {
        match self {
            Self::Lex => "lex",
            Self::Reverse => "reverse",
            Self::Seeded => "seeded",
        }
    }
}

impl FromStr for TieBreak {
    type Err = String;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "lex" => Ok(Self::Lex),
            "reverse" => Ok(Self::Reverse),
            "seeded" => Ok(Self::Seeded),
            _ => Err("tie break must be lex, reverse, or seeded".into()),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BatchPriority {
    #[default]
    Score,
    Bands2,
    Bands4,
    Bounded2,
    Bounded4,
    Bounded8,
}

impl BatchPriority {
    pub fn name(self) -> &'static str {
        match self {
            Self::Score => "score",
            Self::Bands2 => "bands2",
            Self::Bands4 => "bands4",
            Self::Bounded2 => "bounded2",
            Self::Bounded4 => "bounded4",
            Self::Bounded8 => "bounded8",
        }
    }
}

impl FromStr for BatchPriority {
    type Err = String;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "score" => Ok(Self::Score),
            "bands2" => Ok(Self::Bands2),
            "bands4" => Ok(Self::Bands4),
            "bounded2" => Ok(Self::Bounded2),
            "bounded4" => Ok(Self::Bounded4),
            "bounded8" => Ok(Self::Bounded8),
            _ => Err(
                "batch priority must be score, bands2, bands4, bounded2, bounded4, or bounded8"
                    .into(),
            ),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Options {
    pub strategy: Strategy,
    pub tie_break: TieBreak,
    pub seed: u64,
    pub beam_width: usize,
    /// Per-component search-node budget, never a claim of unconditional optimality.
    pub node_limit: usize,
    /// Disable optional clique certificates for a greedy timing run.
    pub certify: bool,
    /// Experimental: on an observed conflict, prefer frequency when the weaker
    /// count is at most this percentage of the stronger count. Zero disables it.
    /// Valid range: 0..=100. Other conflicts retain the configured score priority.
    pub frequency_override_percent: u8,
    /// Globally consistent experimental priorities; default preserves old output.
    pub batch_priority: BatchPriority,
    /// Number of compatible local-maximum waves before one rewrite, 1..=8.
    pub fill_waves: u8,
    /// Count floor for extra waves only; wave one keeps all normal candidates.
    pub fill_min_count: u64,
    /// Preserve the first N normal training rounds before enabling fill waves.
    pub fill_after_rounds: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            strategy: Strategy::Greedy,
            tie_break: TieBreak::Lex,
            seed: 0,
            beam_width: 32,
            node_limit: 200_000,
            certify: true,
            frequency_override_percent: 0,
            batch_priority: BatchPriority::Score,
            fill_waves: 1,
            fill_min_count: 0,
            fill_after_rounds: 0,
        }
    }
}

impl Options {
    pub fn for_round(self, completed_rounds: usize) -> Self {
        if completed_rounds < self.fill_after_rounds {
            Self {
                fill_waves: 1,
                ..self
            }
        } else {
            self
        }
    }
}

pub struct Solution {
    pub vertices: Vec<usize>,
    pub greedy_size: usize,
    pub upper_bound: usize,
    pub nodes: usize,
}

/// Minimum-degree greedy, restricted to a sorted subset of vertices.
fn priority(vertex: usize, options: &Options) -> u64 {
    match options.tie_break {
        TieBreak::Lex => vertex as u64,
        TieBreak::Reverse => u64::MAX - vertex as u64,
        TieBreak::Seeded => {
            let mut x = (vertex as u64 ^ options.seed).wrapping_add(0x9e3779b97f4a7c15);
            x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
            x ^ (x >> 31)
        }
    }
}

fn greedy(adj: &[Vec<usize>], available: &[usize], options: &Options) -> Vec<usize> {
    let mut alive = vec![false; adj.len()];
    for &v in available {
        alive[v] = true;
    }
    let mut degrees = vec![0; adj.len()];
    let mut heap = BinaryHeap::new();
    for &v in available {
        degrees[v] = adj[v].iter().filter(|&&u| alive[u]).count();
        heap.push(Reverse((degrees[v], priority(v, options), v)));
    }
    let mut selected = Vec::new();
    while let Some(Reverse((degree, _, v))) = heap.pop() {
        if !alive[v] || degrees[v] != degree {
            continue;
        }
        selected.push(v);
        let mut remove = vec![v];
        remove.extend(adj[v].iter().copied().filter(|&u| alive[u]));
        for &u in &remove {
            alive[u] = false;
        }
        for u in remove {
            for &w in &adj[u] {
                if alive[w] {
                    degrees[w] -= 1;
                    heap.push(Reverse((degrees[w], priority(w, options), w)));
                }
            }
        }
    }
    selected.sort_unstable();
    selected
}

/// A partition into cliques bounds an independent set by one vertex per clique.
fn upper_bound(adj: &[Vec<usize>], available: &[usize]) -> usize {
    // Avoid quadratic work on very large components; singleton cliques are valid.
    if available.len() > 256 {
        return available.len();
    }
    let mut cliques: Vec<Vec<usize>> = Vec::new();
    for &v in available {
        if let Some(clique) = cliques
            .iter_mut()
            .find(|clique| clique.iter().all(|u| adj[v].binary_search(u).is_ok()))
        {
            clique.push(v);
        } else {
            cliques.push(vec![v]);
        }
    }
    cliques.len()
}

fn improve(best: &mut Vec<usize>, mut candidate: Vec<usize>) {
    candidate.sort_unstable();
    // Preserve the baseline on equal scores; different equally good batches can
    // change all later rounds, so avoid hiding changes behind a tie convention.
    if candidate.len() > best.len() {
        *best = candidate;
    }
}

struct Exact<'a> {
    adj: &'a [u64],
    best: u64,
    nodes: usize,
    limit: usize,
    exhausted: bool,
}

impl Exact<'_> {
    fn bound(&self, mut remaining: u64) -> u32 {
        let mut cliques = 0;
        while remaining != 0 {
            cliques += 1;
            let mut common = remaining;
            while common != 0 {
                let v = common.trailing_zeros() as usize;
                let bit = 1u64 << v;
                remaining &= !bit;
                common &= self.adj[v] & remaining;
            }
        }
        cliques
    }

    fn visit(&mut self, mut remaining: u64, mut chosen: u64) {
        if self.nodes >= self.limit {
            self.exhausted = true;
            return;
        }
        self.nodes += 1;
        // Isolated vertices and degree-one vertices can safely be included in
        // some maximum-cardinality independent set of an unweighted graph.
        loop {
            let mut scan = remaining;
            let mut reduced = false;
            while scan != 0 {
                let v = scan.trailing_zeros() as usize;
                scan &= scan - 1;
                if (self.adj[v] & remaining).count_ones() <= 1 {
                    chosen |= 1u64 << v;
                    remaining &= !(self.adj[v] | (1u64 << v));
                    reduced = true;
                    break;
                }
            }
            if !reduced {
                break;
            }
        }
        if chosen.count_ones() > self.best.count_ones() {
            self.best = chosen;
        }
        if remaining == 0 || chosen.count_ones() + self.bound(remaining) <= self.best.count_ones() {
            return;
        }
        let v = (0..self.adj.len())
            .filter(|&v| remaining & (1u64 << v) != 0)
            .max_by_key(|&v| ((self.adj[v] & remaining).count_ones(), Reverse(v)))
            .unwrap();
        let bit = 1u64 << v;
        self.visit(remaining & !(self.adj[v] | bit), chosen | bit);
        self.visit(remaining & !bit, chosen);
    }
}

#[derive(Clone)]
struct State {
    remaining: Vec<usize>,
    chosen: Vec<usize>,
    upper: usize,
}

fn beam(adj: &[Vec<usize>], options: &Options, seed: Vec<usize>) -> Solution {
    let n = adj.len();
    let greedy_size = seed.len();
    if options.strategy == Strategy::Greedy && !options.certify {
        return Solution {
            vertices: seed,
            greedy_size,
            upper_bound: adj.len(),
            nodes: 0,
        };
    }
    let mut best = seed;
    let all: Vec<_> = (0..n).collect();
    let root_upper = upper_bound(adj, &all);
    let mut frontier = vec![State {
        remaining: all,
        chosen: Vec::new(),
        upper: root_upper,
    }];
    let mut dropped_upper = 0;
    let mut nodes = 0;
    while !frontier.is_empty() {
        let mut next = Vec::new();
        for state in frontier {
            if state.upper <= best.len() {
                continue;
            }
            if nodes >= options.node_limit {
                dropped_upper = dropped_upper.max(state.upper);
                continue;
            }
            nodes += 1;
            let mut alive = vec![false; n];
            for &v in &state.remaining {
                alive[v] = true;
            }
            let v = *state
                .remaining
                .iter()
                .max_by_key(|&&v| (adj[v].iter().filter(|&&u| alive[u]).count(), Reverse(v)))
                .unwrap();
            for include in [true, false] {
                let remaining: Vec<_> = state
                    .remaining
                    .iter()
                    .copied()
                    .filter(|&u| u != v && (!include || adj[v].binary_search(&u).is_err()))
                    .collect();
                let mut chosen = state.chosen.clone();
                if include {
                    chosen.push(v);
                }
                let mut completion = chosen.clone();
                completion.extend(greedy(adj, &remaining, options));
                improve(&mut best, completion);
                let upper = chosen.len() + upper_bound(adj, &remaining);
                if upper > best.len() {
                    next.push(State {
                        remaining,
                        chosen,
                        upper,
                    });
                }
            }
        }
        next.retain(|s| s.upper > best.len());
        next.sort_by(|a, b| {
            b.upper
                .cmp(&a.upper)
                .then_with(|| b.chosen.len().cmp(&a.chosen.len()))
                .then_with(|| a.remaining.cmp(&b.remaining))
                .then_with(|| a.chosen.cmp(&b.chosen))
        });
        if next.len() > options.beam_width {
            for state in &next[options.beam_width..] {
                dropped_upper = dropped_upper.max(state.upper);
            }
            next.truncate(options.beam_width);
        }
        frontier = next;
    }
    let upper = dropped_upper.max(best.len());
    Solution {
        vertices: best,
        greedy_size,
        upper_bound: upper,
        nodes,
    }
}

/// Adjacency lists must describe a simple undirected graph and be sorted.
pub fn solve(adj: &[Vec<usize>], options: &Options) -> Solution {
    let all: Vec<_> = (0..adj.len()).collect();
    let seed = greedy(adj, &all, options);
    let greedy_size = seed.len();
    let root_upper = upper_bound(adj, &all);
    if options.strategy == Strategy::Greedy || greedy_size == root_upper {
        return Solution {
            vertices: seed,
            greedy_size,
            upper_bound: root_upper,
            nodes: 0,
        };
    }
    if options.strategy == Strategy::Hybrid && adj.len() <= 64 {
        let masks: Vec<_> = adj
            .iter()
            .map(|neighbors| neighbors.iter().fold(0, |m, &v| m | (1u64 << v)))
            .collect();
        let initial = seed.iter().fold(0, |m, &v| m | (1u64 << v));
        let mut exact = Exact {
            adj: &masks,
            best: initial,
            nodes: 0,
            limit: options.node_limit,
            exhausted: false,
        };
        let remaining = if adj.len() == 64 {
            u64::MAX
        } else {
            (1u64 << adj.len()) - 1
        };
        exact.visit(remaining, 0);
        let vertices: Vec<_> = (0..adj.len())
            .filter(|&v| exact.best & (1u64 << v) != 0)
            .collect();
        let upper = if exact.exhausted {
            root_upper
        } else {
            vertices.len()
        };
        return Solution {
            vertices,
            greedy_size,
            upper_bound: upper,
            nodes: exact.nodes,
        };
    }
    beam(adj, options, seed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn graph(n: usize, edges: u64) -> Vec<Vec<usize>> {
        let mut adj = vec![Vec::new(); n];
        let mut bit = 0;
        for a in 0..n {
            for b in a + 1..n {
                if edges & (1u64 << bit) != 0 {
                    adj[a].push(b);
                    adj[b].push(a);
                }
                bit += 1;
            }
        }
        for neighbors in &mut adj {
            neighbors.sort_unstable();
        }
        adj
    }

    fn oracle(adj: &[Vec<usize>]) -> usize {
        (0..1usize << adj.len())
            .filter(|&mask| {
                (0..adj.len())
                    .all(|v| mask & (1 << v) == 0 || adj[v].iter().all(|&u| mask & (1 << u) == 0))
            })
            .map(|mask| mask.count_ones() as usize)
            .max()
            .unwrap()
    }

    #[test]
    fn exact_search_matches_all_six_vertex_graphs_and_exposes_greedy_failures() {
        let options = Options {
            strategy: Strategy::Hybrid,
            ..Options::default()
        };
        let mut improvements = 0;
        for edges in 0..1u64 << 15 {
            let adj = graph(6, edges);
            let optimum = oracle(&adj);
            let result = solve(&adj, &options);
            assert_eq!(result.vertices.len(), optimum);
            assert_eq!(result.upper_bound, optimum);
            if result.vertices.len() > result.greedy_size {
                improvements += 1;
            }
        }
        assert!(
            improvements > 0,
            "minimum-degree greedy is not always optimal"
        );
    }

    #[test]
    fn beam_bounds_and_budgeted_exact_search_are_honest() {
        let mut unproven = 0;
        for edges in 0..1u64 << 15 {
            let adj = graph(6, edges);
            let optimum = oracle(&adj);
            for strategy in [Strategy::Greedy, Strategy::Beam, Strategy::Hybrid] {
                let options = Options {
                    strategy,
                    beam_width: 1,
                    node_limit: 1,
                    ..Options::default()
                };
                let result = solve(&adj, &options);
                assert!(result.vertices.len() >= result.greedy_size);
                assert!(result.vertices.len() <= optimum);
                assert!(result.upper_bound >= optimum);
                if result.upper_bound > result.vertices.len() {
                    unproven += 1;
                }
                for &v in &result.vertices {
                    assert!(adj[v].iter().all(|u| !result.vertices.contains(u)));
                }
            }
        }
        assert!(
            unproven > 0,
            "a tiny budget must not falsely certify all graphs"
        );
    }

    #[test]
    fn hybrid_handles_64_bit_masks_and_large_component_fallback() {
        let small = (0..1u64 << 15)
            .map(|edges| graph(6, edges))
            .find(|adj| {
                let greedy = solve(adj, &Options::default());
                greedy.vertices.len() < oracle(adj)
            })
            .unwrap();
        let optimum = oracle(&small);
        let mut full_word = small;
        full_word.resize(64, Vec::new());
        let options = Options {
            strategy: Strategy::Hybrid,
            ..Options::default()
        };
        let result = solve(&full_word, &options);
        assert_eq!(result.vertices.len(), optimum + 58);
        assert_eq!(result.upper_bound, result.vertices.len());
        assert!(result.nodes > 0);
        let mut cycle = vec![Vec::new(); 65];
        for (v, neighbors) in cycle.iter_mut().enumerate() {
            neighbors.extend([(v + 64) % 65, (v + 1) % 65]);
            neighbors.sort_unstable();
        }
        let result = solve(&cycle, &options);
        assert_eq!(result.vertices.len(), 32);
        assert_eq!(result.upper_bound, 32);
    }

    #[test]
    fn branching_and_beam_bounds_match_larger_brute_force_instances() {
        let mut random = 2026u64;
        for case in 0..64 {
            let mut adj = vec![Vec::new(); 12];
            for a in 0..12 {
                for b in a + 1..12 {
                    random ^= random << 13;
                    random ^= random >> 7;
                    random ^= random << 17;
                    if random % 8 < 1 + case % 7 {
                        adj[a].push(b);
                        adj[b].push(a);
                    }
                }
            }
            for neighbors in &mut adj {
                neighbors.sort_unstable();
            }
            let optimum = oracle(&adj);
            for strategy in [Strategy::Beam, Strategy::Hybrid] {
                let result = solve(
                    &adj,
                    &Options {
                        strategy,
                        beam_width: 4,
                        ..Options::default()
                    },
                );
                assert!(result.vertices.len() <= optimum);
                assert!(result.upper_bound >= optimum);
                if strategy == Strategy::Hybrid {
                    assert_eq!(result.vertices.len(), optimum);
                    assert_eq!(result.upper_bound, optimum);
                }
            }
        }
    }
}
