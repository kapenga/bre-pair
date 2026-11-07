//! Dictionary-only absorption prior. Exact overlap support, not a probability
//! that an extension exists at a particular corpus position.
use crate::{
    grammar::{Symbol, BASE_SYMBOLS, EOF},
    tokenizer::{Model, Token, Tokenizer},
};

const ALPHABET: usize = 257;
const KEYS: usize = ALPHABET * ALPHABET;
pub(crate) const SCALE: u128 = 1_000_000;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum OverlapWeight {
    Presence,
    #[default]
    Linear,
    Square,
}
impl OverlapWeight {
    pub fn name(self) -> &'static str {
        match self {
            Self::Presence => "presence",
            Self::Linear => "linear",
            Self::Square => "square",
        }
    }
    fn mass(self, n: usize) -> u128 {
        match self {
            Self::Presence => u128::from(n != 0),
            Self::Linear => n as u128,
            Self::Square => (n as u128) * (n as u128),
        }
    }
}
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum FrequencyWeight {
    #[default]
    Count,
    Presence,
    Log,
    Availability,
}
impl FrequencyWeight {
    pub fn name(self) -> &'static str {
        match self {
            Self::Count => "count",
            Self::Presence => "presence",
            Self::Log => "log",
            Self::Availability => "availability",
        }
    }
    fn weight(self, n: u64) -> u64 {
        match self {
            Self::Count => n,
            Self::Presence => u64::from(n != 0),
            Self::Log => u64::from(u64::BITS - n.leading_zeros()),
            Self::Availability => 1,
        }
    }
}
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Normalization {
    #[default]
    Local,
    Global,
}
impl Normalization {
    pub fn name(self) -> &'static str {
        match self {
            Self::Local => "local",
            Self::Global => "global",
        }
    }
}

/// Research controls for distinguishing useful overlap information from jitter.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Signal {
    #[default]
    Overlap,
    Shuffle,
    Stratified,
    Uniform,
}
impl Signal {
    pub fn name(self) -> &'static str {
        match self {
            Self::Overlap => "overlap",
            Self::Shuffle => "shuffle",
            Self::Stratified => "stratified",
            Self::Uniform => "uniform",
        }
    }
}

// SplitMix64, with rejection sampling for an unbiased bounded integer.
struct Random(u64);
impl Random {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    fn below(&mut self, bound: u64) -> u64 {
        let threshold = bound.wrapping_neg() % bound;
        loop {
            let n = self.next();
            if n >= threshold {
                return n % bound;
            }
        }
    }
}

fn randomize(discounts: &mut [u32], signal: Signal, seed: u64, round: u64, bound: u32) {
    let mut rng = Random(seed ^ round.wrapping_mul(0xd1342543de82ef95));
    match signal {
        Signal::Overlap => {}
        Signal::Shuffle | Signal::Stratified => {
            for i in (1..discounts.len()).rev() {
                let j = rng.below((i + 1) as u64) as usize;
                discounts.swap(i, j);
            }
        }
        Signal::Uniform => {
            for d in discounts {
                *d = rng.below(u64::from(bound) + 1) as u32;
            }
        }
    }
}

/// Preserve exact byte length and log2 usage bucket while removing the
/// within-bucket relationship between token content and its discount.
fn stratified_shuffle(
    discounts: &mut [u32],
    strata: &[(usize, u32, usize)],
    seed: u64,
    round: u64,
) {
    let mut rng = Random(seed ^ round.wrapping_mul(0xd1342543de82ef95));
    let mut start = 0;
    while start < strata.len() {
        let mut end = start + 1;
        while end < strata.len()
            && (strata[end].0, strata[end].1) == (strata[start].0, strata[start].1)
        {
            end += 1;
        }
        for i in (start + 1..end).rev() {
            let j = start + rng.below((i - start + 1) as u64) as usize;
            discounts.swap(strata[i].2, strata[j].2);
        }
        start = end;
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Settings {
    /// Bit1: start of target; bit2: strict middle containment; bit4: target end.
    pub types: u8,
    pub overlap_weight: OverlapWeight,
    pub frequency_weight: FrequencyWeight,
    pub minimum_overlap: usize,
    pub normalization: Normalization,
    pub internal: bool,
    pub signal: Signal,
    pub seed: u64,
}
impl Default for Settings {
    fn default() -> Self {
        Self {
            types: 7,
            overlap_weight: OverlapWeight::Linear,
            frequency_weight: FrequencyWeight::Count,
            minimum_overlap: 1,
            normalization: Normalization::Local,
            internal: false,
            signal: Signal::Overlap,
            seed: 0,
        }
    }
}
impl Settings {
    pub fn valid(self) -> bool {
        ((1..=7).contains(&self.types) || (self.types == 0 && self.internal))
            && self.minimum_overlap != 0
    }
}

fn label(t: &Token, at: usize) -> u16 {
    t.bytes.get(at).map_or(EOF as u16, |&b| u16::from(b))
}
fn key(a: u16, b: u16) -> usize {
    usize::from(a) * ALPHABET + usize::from(b)
}

#[derive(Clone, Copy)]
enum Kind {
    Prefix,
    Suffix,
    Contains,
}

fn keys(t: &Token, kind: Kind, out: &mut Vec<usize>) {
    out.clear();
    let n = t.terminals();
    match kind {
        Kind::Prefix => out.push(key(label(t, 0), label(t, 1))),
        Kind::Suffix => out.push(key(label(t, n - 2), label(t, n - 1))),
        Kind::Contains => {
            for at in 1..n {
                out.push(key(label(t, at - 1), label(t, at)));
            }
            out.sort_unstable();
            out.dedup();
        }
    }
}

struct Buckets {
    offsets: Vec<usize>,
    ids: Vec<Symbol>,
}
impl Buckets {
    fn new(model: &Model, order: &[Symbol], kind: Kind) -> Self {
        let mut offsets = vec![0usize; KEYS + 1];
        let mut scratch = Vec::new();
        for &id in order {
            keys(&model.tokens[id as usize], kind, &mut scratch);
            for &k in &scratch {
                offsets[k + 1] += 1;
            }
        }
        for k in 0..KEYS {
            offsets[k + 1] += offsets[k];
        }
        let mut next = offsets[..KEYS].to_vec();
        let mut ids = vec![0; offsets[KEYS]];
        for &id in order {
            keys(&model.tokens[id as usize], kind, &mut scratch);
            for &k in &scratch {
                ids[next[k]] = id;
                next[k] += 1;
            }
        }
        Self { offsets, ids }
    }
    fn get(&self, k: usize) -> &[Symbol] {
        &self.ids[self.offsets[k]..self.offsets[k + 1]]
    }
    fn bytes(&self) -> usize {
        self.offsets.capacity() * 8 + self.ids.capacity() * 4
    }
}

#[derive(Default)]
pub(crate) struct Stats {
    pub targets: usize,
    pub pair_checks: u64,
    pub ratio_ppm_sum: u128,
    pub ratio_ppm_max: u128,
    pub saturated_targets: usize,
    pub zero_support_targets: usize,
    pub one_overlap_share_ppm_sum: u128,
    pub internal_share_ppm_sum: u128,
}

pub(crate) struct Index {
    pub settings: Settings,
    order: Vec<Symbol>,
    prefix: Buckets,
    suffix: Buckets,
    contains: Buckets,
    failure_offsets: Vec<usize>,
    failures: Vec<u32>,
    seen: Vec<u32>,
    stamp: u32,
    first: Vec<u64>,
    last: Vec<u64>,
    endpoints: Vec<u64>,
    grams: Vec<usize>,
    queries: Vec<usize>,
    discounts: Vec<u32>,
    strata: Vec<(usize, u32, usize)>,
    random_round: u64,
    one_mass: u128,
}

impl Index {
    pub fn new(model: &Model) -> Self {
        let mut order: Vec<_> = (BASE_SYMBOLS..model.tokens.len())
            .map(|id| id as Symbol)
            .collect();
        order.sort_unstable_by_key(|&id| {
            (std::cmp::Reverse(model.tokens[id as usize].bytes.len()), id)
        });
        let prefix = Buckets::new(model, &order, Kind::Prefix);
        let suffix = Buckets::new(model, &order, Kind::Suffix);
        let contains = Buckets::new(model, &order, Kind::Contains);
        let mut failure_offsets = Vec::with_capacity(model.tokens.len() + 1);
        let mut failures = Vec::<u32>::new();
        for (id, t) in model.tokens.iter().enumerate() {
            let start = failures.len();
            failure_offsets.push(start);
            if id < BASE_SYMBOLS {
                continue;
            }
            failures.push(0);
            for at in 1..t.terminals() {
                let mut matched = failures[start + at - 1] as usize;
                while matched != 0 && label(t, matched) != label(t, at) {
                    matched = failures[start + matched - 1] as usize;
                }
                if label(t, matched) == label(t, at) {
                    matched += 1;
                }
                failures.push(matched as u32);
            }
        }
        failure_offsets.push(failures.len());
        Self {
            settings: Settings::default(),
            order,
            prefix,
            suffix,
            contains,
            failure_offsets,
            failures,
            seen: vec![0; model.tokens.len()],
            stamp: 0,
            first: vec![0; ALPHABET],
            last: vec![0; ALPHABET],
            endpoints: vec![0; KEYS],
            grams: Vec::new(),
            queries: Vec::new(),
            discounts: Vec::new(),
            strata: Vec::new(),
            random_round: 0,
            one_mass: 0,
        }
    }

    pub fn allocated_bytes(&self) -> usize {
        self.order.capacity() * 4
            + self.prefix.bytes()
            + self.suffix.bytes()
            + self.contains.bytes()
            + self.failure_offsets.capacity() * 8
            + self.failures.capacity() * 4
            + self.seen.capacity() * 4
            + (self.first.capacity() + self.last.capacity() + self.endpoints.capacity()) * 8
            + (self.grams.capacity() + self.queries.capacity()) * 8
            + self.discounts.capacity() * 4
            + self.strata.capacity() * std::mem::size_of::<(usize, u32, usize)>()
    }

    fn failure(&self, id: Symbol) -> &[u32] {
        &self.failures[self.failure_offsets[id as usize]..self.failure_offsets[id as usize + 1]]
    }

    /// Longest compatible alignment crossing either outer boundary of t.
    /// Full containment is included; shorter, wholly internal pieces are not.
    fn overlap_parts(&self, model: &Model, tid: Symbol, uid: Symbol) -> [usize; 3] {
        let t = &model.tokens[tid as usize];
        let u = &model.tokens[uid as usize];
        let tf = self.failure(tid);
        let uf = self.failure(uid);
        let mut left = 0;
        let mut parts = [0; 3];
        for at in 0..u.terminals() {
            while left != 0 && label(t, left) != label(u, at) {
                left = tf[left - 1] as usize;
            }
            if label(t, left) == label(u, at) {
                left += 1;
            }
            if left == t.terminals() {
                let start = at + 1 - t.terminals();
                if start == 0 && at + 1 < u.terminals() {
                    parts[2] = left;
                } else if start > 0 && at + 1 == u.terminals() {
                    parts[0] = left;
                } else if start > 0 && at + 1 < u.terminals() {
                    parts[1] = left;
                }
                left = tf[left - 1] as usize;
            }
        }
        if left == u.terminals() {
            left = tf[left - 1] as usize;
        }
        let mut right = 0;
        for at in 0..t.terminals() {
            while right != 0 && (right == u.terminals() || label(u, right) != label(t, at)) {
                right = uf[right - 1] as usize;
            }
            if label(u, right) == label(t, at) {
                right += 1;
            }
        }
        if right == u.terminals() {
            right = uf[right - 1] as usize;
        }
        parts[0] = parts[0].max(left);
        parts[2] = parts[2].max(right);
        parts
    }

    fn overlap(&self, model: &Model, tid: Symbol, uid: Symbol) -> usize {
        self.overlap_parts(model, tid, uid)
            .into_iter()
            .enumerate()
            .filter(|(i, _)| self.settings.types & (1 << i) != 0)
            .map(|(_, n)| n)
            .max()
            .unwrap_or(0)
    }

    #[allow(clippy::too_many_arguments)]
    fn visit(
        &mut self,
        model: &Model,
        tid: Symbol,
        active: &[u8],
        uses: &[u64],
        kind: Kind,
        k: usize,
        numerator: &mut u128,
        stats: &mut Stats,
    ) {
        let bucket = match kind {
            Kind::Prefix => &self.prefix,
            Kind::Suffix => &self.suffix,
            Kind::Contains => &self.contains,
        };
        let t = &model.tokens[tid as usize];
        for &uid in bucket.get(k) {
            let u = &model.tokens[uid as usize];
            if u.bytes.len() < t.bytes.len() {
                break;
            }
            if self.seen[uid as usize] == self.stamp {
                continue;
            }
            self.seen[uid as usize] = self.stamp;
            if uid == tid
                || active[uid as usize] == 0
                || self.settings.frequency_weight.weight(uses[uid as usize]) == 0
            {
                continue;
            }
            stats.pair_checks += 1;
            let overlap = self.overlap(model, tid, uid);
            if overlap < 2 || overlap < self.settings.minimum_overlap {
                continue;
            }
            let base = usize::from(
                self.settings.minimum_overlap == 1
                    && ((self.settings.types & 4 != 0
                        && label(t, t.terminals() - 1) == label(u, 0))
                        || (self.settings.types & 1 != 0
                            && label(u, u.terminals() - 1) == label(t, 0))),
            );
            let weight = u128::from(self.settings.frequency_weight.weight(uses[uid as usize]));
            *numerator += weight * (self.settings.overlap_weight.mass(overlap) - base as u128);
            self.one_mass -= weight * base as u128;
        }
    }

    /// All one-terminal overlaps are aggregated by length and endpoints. Only
    /// stronger matches visit posting lists, once per distinct alternative.
    fn numerator(
        &mut self,
        model: &Model,
        tid: Symbol,
        active: &[u8],
        uses: &[u64],
        stats: &mut Stats,
    ) -> u128 {
        self.stamp = self.stamp.wrapping_add(1);
        if self.stamp == 0 {
            self.seen.fill(0);
            self.stamp = 1;
        }
        let t = &model.tokens[tid as usize];
        let first = label(t, 0);
        let last = label(t, t.terminals() - 1);
        let mut result = 0;
        if self.settings.minimum_overlap == 1 {
            if self.settings.types & 4 != 0 {
                result += u128::from(self.first[last as usize]);
            }
            if self.settings.types & 1 != 0 {
                result += u128::from(self.last[first as usize]);
            }
            if self.settings.types & 5 == 5 {
                result -= u128::from(self.endpoints[key(last, first)]);
            }
            if self.settings.types & 5 != 0 && first == last {
                result -= u128::from(self.settings.frequency_weight.weight(uses[tid as usize]));
            }
        }
        self.one_mass = result;
        keys(t, Kind::Contains, &mut self.grams);
        let rarest = self
            .grams
            .iter()
            .copied()
            .min_by_key(|&k| self.contains.get(k).len())
            .unwrap();
        for at in 0..self.grams.len() {
            let k = self.grams[at];
            if self.settings.types & 4 != 0 {
                self.visit(
                    model,
                    tid,
                    active,
                    uses,
                    Kind::Prefix,
                    k,
                    &mut result,
                    stats,
                );
            }
            if self.settings.types & 1 != 0 {
                self.visit(
                    model,
                    tid,
                    active,
                    uses,
                    Kind::Suffix,
                    k,
                    &mut result,
                    stats,
                );
            }
        }
        if self.settings.types & 2 != 0 {
            self.visit(
                model,
                tid,
                active,
                uses,
                Kind::Contains,
                rarest,
                &mut result,
                stats,
            );
        }
        result
    }

    pub fn adjust(
        &mut self,
        tokenizer: &Tokenizer,
        active: &[u8],
        uses: &[u64],
        ranked: &mut [(u128, Symbol)],
        prefix: usize,
        penalty_ppm: u32,
    ) -> Stats {
        let model = tokenizer.model();
        let mut stats = Stats::default();
        self.discounts.clear();
        self.random_round += 1;
        let kth = prefix.min(ranked.len()) - 1;
        ranked.select_nth_unstable(kth);
        let boundary = ranked[kth].0 * SCALE;
        self.queries.clear();
        for (at, (score, _)) in ranked.iter_mut().enumerate() {
            // Even the maximum discount cannot bring an omitted candidate into
            // the selected prefix. This preserves full-reranking semantics.
            if *score * (SCALE - u128::from(penalty_ppm)) <= boundary {
                self.queries.push(at);
            }
            *score *= SCALE;
        }
        self.queries.sort_unstable_by_key(|&at| {
            (
                std::cmp::Reverse(model.tokens[ranked[at].1 as usize].bytes.len()),
                ranked[at].1,
            )
        });
        self.first.fill(0);
        self.last.fill(0);
        self.endpoints.fill(0);
        let mut cursor = 0;
        let mut eligible_weight = 0u128;
        let all_weight = if self.settings.internal {
            self.order
                .iter()
                .filter(|&&id| active[id as usize] != 0)
                .map(|&id| u128::from(self.settings.frequency_weight.weight(uses[id as usize])))
                .sum()
        } else {
            0
        };
        for q in 0..self.queries.len() {
            let at = self.queries[q];
            let tid = ranked[at].1;
            let length = model.tokens[tid as usize].bytes.len();
            while cursor < self.order.len() {
                let uid = self.order[cursor];
                let u = &model.tokens[uid as usize];
                if u.bytes.len() < length {
                    break;
                }
                cursor += 1;
                if active[uid as usize] == 0 {
                    continue;
                }
                let n = self.settings.frequency_weight.weight(uses[uid as usize]);
                eligible_weight += u128::from(n);
                let a = label(u, 0);
                let b = label(u, u.terminals() - 1);
                self.first[a as usize] += n;
                self.last[b as usize] += n;
                self.endpoints[key(a, b)] += n;
            }
            let outer = self.numerator(model, tid, active, uses, &mut stats);
            let mut numerator = outer;
            if self.settings.internal {
                tokenizer.visit_internal_matches(tid, active, |uid, length| {
                    if self.seen[uid as usize] == self.stamp {
                        return;
                    }
                    self.seen[uid as usize] = self.stamp;
                    let weight = self.settings.frequency_weight.weight(uses[uid as usize]);
                    if length < self.settings.minimum_overlap || weight == 0 {
                        return;
                    }
                    stats.pair_checks += 1;
                    numerator += u128::from(weight) * self.settings.overlap_weight.mass(length);
                });
            }
            let own = self
                .settings
                .overlap_weight
                .mass(model.tokens[tid as usize].terminals());
            let denominator = match self.settings.normalization {
                Normalization::Local => {
                    numerator
                        + u128::from(self.settings.frequency_weight.weight(uses[tid as usize]))
                            * own
                }
                Normalization::Global => {
                    if self.settings.internal {
                        all_weight * own
                    } else {
                        eligible_weight * own
                    }
                }
            };
            let discount = if denominator == 0 {
                0
            } else {
                numerator * u128::from(penalty_ppm) / denominator
            };
            if self.settings.signal == Signal::Overlap {
                ranked[at].0 = ranked[at].0 / SCALE * (SCALE - discount);
            } else {
                self.discounts.push(discount as u32);
            }
            let ratio = if denominator == 0 {
                0
            } else {
                numerator * SCALE / denominator
            };
            stats.ratio_ppm_sum += ratio;
            stats.ratio_ppm_max = stats.ratio_ppm_max.max(ratio);
            stats.saturated_targets += usize::from(ratio >= 950000);
            stats.zero_support_targets += usize::from(numerator == 0);
            stats.one_overlap_share_ppm_sum += if numerator == 0 {
                0
            } else {
                self.one_mass * SCALE / numerator
            };
            stats.internal_share_ppm_sum += if numerator == 0 {
                0
            } else {
                (numerator - outer) * SCALE / numerator
            };
            stats.targets += 1;
        }
        if self.settings.signal != Signal::Overlap {
            if self.settings.signal == Signal::Stratified {
                self.strata.clear();
                for (q, &at) in self.queries.iter().enumerate() {
                    let id = ranked[at].1 as usize;
                    self.strata.push((
                        model.tokens[id].bytes.len(),
                        u64::BITS - uses[id].leading_zeros(),
                        q,
                    ));
                }
                self.strata.sort_unstable();
                stratified_shuffle(
                    &mut self.discounts,
                    &self.strata,
                    self.settings.seed,
                    self.random_round,
                );
            } else {
                randomize(
                    &mut self.discounts,
                    self.settings.signal,
                    self.settings.seed,
                    self.random_round,
                    penalty_ppm,
                );
            }
            for (&at, &discount) in self.queries.iter().zip(&self.discounts) {
                ranked[at].0 = ranked[at].0 / SCALE * (SCALE - u128::from(discount));
            }
        }
        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stratified_placebo_cannot_mix_length_or_usage_buckets() {
        let strata = [(2, 0, 1), (2, 0, 3), (2, 1, 0), (2, 1, 4), (3, 1, 2)];
        let original = [10, 20, 30, 40, 50];
        for seed in 0..20 {
            let mut d = original;
            stratified_shuffle(&mut d, &strata, seed, 1);
            let mut a = [d[1], d[3]];
            let mut b = [d[0], d[4]];
            a.sort_unstable();
            b.sort_unstable();
            assert_eq!(a, [20, 40]);
            assert_eq!(b, [10, 50]);
            assert_eq!(d[2], 30);
        }
    }

    #[test]
    fn placebo_rng_and_shuffle_preserve_distribution_and_are_reproducible() {
        let mut rng = Random(0);
        assert_eq!(rng.next(), 0xe220a8397b1dcdaf);
        assert_eq!(rng.next(), 0x6e789e6aa1b965f4);
        let original: Vec<_> = (0..200).map(|n| n * 500).collect();
        for seed in [0, 1, u64::MAX] {
            let mut a = original.clone();
            let mut b = original.clone();
            randomize(&mut a, Signal::Shuffle, seed, 1, 100_000);
            randomize(&mut b, Signal::Shuffle, seed, 1, 100_000);
            assert_eq!(a, b);
            assert_ne!(a, original);
            a.sort_unstable();
            assert_eq!(a, original);
            randomize(&mut b, Signal::Uniform, seed, 1, 100_000);
            assert!(b.iter().all(|&d| d <= 100_000));
        }
    }

    #[test]
    fn shuffled_signal_preserves_real_discounts_with_different_assignments() {
        let m = model();
        let n = m.tokens.len();
        let tokenizer = Tokenizer::new(m).unwrap();
        let active = vec![1; n];
        let uses: Vec<_> = (0..n).map(|id| (id % 17) as u64).collect();
        // Equal base costs make the complete candidate universe eligible, and
        // expose discounts directly in the adjusted integer score.
        let base: Vec<_> = (BASE_SYMBOLS..n).map(|id| (1, id as Symbol)).collect();
        let mut real = base.clone();
        let mut index = Index::new(tokenizer.model());
        index.adjust(&tokenizer, &active, &uses, &mut real, 7, 100_000);
        let mut shuffled = base.clone();
        index.settings.signal = Signal::Shuffle;
        index.settings.seed = 123;
        index.adjust(&tokenizer, &active, &uses, &mut shuffled, 7, 100_000);
        let scores = |v: &[(u128, Symbol)]| {
            let mut s: Vec<_> = v.iter().map(|&(score, _)| score).collect();
            s.sort_unstable();
            s
        };
        assert_eq!(scores(&real), scores(&shuffled));
        real.sort_unstable_by_key(|&(_, id)| id);
        shuffled.sort_unstable_by_key(|&(_, id)| id);
        assert_ne!(real, shuffled);
    }

    fn model() -> Model {
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
        for length in 1..=4 {
            for bits in 0..(1 << length) {
                let bytes: Vec<_> = (0..length)
                    .map(|at| b'a' + ((bits >> at) & 1) as u8)
                    .collect();
                if length > 1 {
                    tokens.push(Token {
                        bytes: bytes.clone(),
                        ends_field: false,
                    });
                }
                tokens.push(Token {
                    bytes,
                    ends_field: true,
                });
            }
        }
        for bytes in [
            b"abcde".as_slice(),
            b"cdefg",
            b"zzabc",
            b"xxabcdeyy",
            b"defg",
            b"abcdefg",
            b"bcdef",
            b"ghij",
            &[0, 255, 0],
        ] {
            tokens.push(Token {
                bytes: bytes.to_vec(),
                ends_field: false,
            });
        }
        Model::new(tokens).unwrap()
    }

    // Independent alignment oracle: test every relative placement and every
    // shared terminal, including EOF. Neither KMP nor posting lists are used.
    fn oracle_parts(t: &Token, u: &Token) -> [usize; 3] {
        let labels = |p: &Token| {
            p.bytes
                .iter()
                .map(|&b| u16::from(b))
                .chain(p.ends_field.then_some(256))
                .collect::<Vec<_>>()
        };
        let a = labels(t);
        let b = labels(u);
        let mut best = [0; 3];
        for start in -(b.len() as isize - 1)..a.len() as isize {
            let end = start + b.len() as isize;
            if start >= 0 && end <= a.len() as isize {
                continue;
            }
            let left = start.max(0) as usize;
            let right = end.min(a.len() as isize) as usize;
            if (left..right).all(|at| a[at] == b[(at as isize - start) as usize]) {
                let kind = if start < 0 && end > a.len() as isize {
                    1
                } else if start < 0 {
                    0
                } else {
                    2
                };
                best[kind] = best[kind].max(right - left);
            }
        }
        best
    }

    fn overlap(t: &Token, u: &Token) -> usize {
        oracle_parts(t, u).into_iter().max().unwrap()
    }

    #[test]
    fn directional_parts_match_all_alignments_including_periodic_multiple_matches() {
        let m = model();
        let index = Index::new(&m);
        for tid in BASE_SYMBOLS..m.tokens.len() {
            for uid in BASE_SYMBOLS..m.tokens.len() {
                if uid == tid {
                    continue;
                }
                assert_eq!(
                    index.overlap_parts(&m, tid as Symbol, uid as Symbol),
                    oracle_parts(&m.tokens[tid], &m.tokens[uid]),
                    "t={tid},u={uid}"
                );
            }
        }
    }

    #[test]
    fn every_scoring_ablation_matches_independent_full_scoring() {
        let m = model();
        let n = m.tokens.len();
        let mut settings = vec![Settings::default()];
        for types in [1, 2, 4, 5] {
            settings.push(Settings {
                types,
                ..Default::default()
            });
        }
        for overlap_weight in [OverlapWeight::Presence, OverlapWeight::Square] {
            settings.push(Settings {
                overlap_weight,
                ..Default::default()
            });
        }
        for frequency_weight in [
            FrequencyWeight::Presence,
            FrequencyWeight::Log,
            FrequencyWeight::Availability,
        ] {
            settings.push(Settings {
                frequency_weight,
                ..Default::default()
            });
        }
        for minimum_overlap in [2, 3] {
            settings.push(Settings {
                minimum_overlap,
                ..Default::default()
            });
        }
        settings.push(Settings {
            normalization: Normalization::Global,
            ..Default::default()
        });
        settings.push(Settings {
            types: 2,
            overlap_weight: OverlapWeight::Square,
            frequency_weight: FrequencyWeight::Presence,
            minimum_overlap: 3,
            normalization: Normalization::Global,
            internal: false,
            ..Default::default()
        });
        settings.push(Settings {
            types: 0,
            internal: true,
            ..Default::default()
        });
        settings.push(Settings {
            internal: true,
            ..Default::default()
        });
        settings.push(Settings {
            internal: true,
            normalization: Normalization::Global,
            frequency_weight: FrequencyWeight::Availability,
            ..Default::default()
        });
        for config in settings {
            let mut index = Index::new(&m);
            index.settings = config;
            let active: Vec<_> = (0..n)
                .map(|id| u8::from(id < BASE_SYMBOLS || id % 5 != 0))
                .collect();
            let uses: Vec<_> = (0..n).map(|id| (id % 17) as u64).collect();
            let weight = |count: u64| match config.frequency_weight {
                FrequencyWeight::Count => count,
                FrequencyWeight::Presence => u64::from(count > 0),
                FrequencyWeight::Log => u64::from(u64::BITS - count.leading_zeros()),
                FrequencyWeight::Availability => 1,
            };
            let mass = |overlap: usize| match config.overlap_weight {
                OverlapWeight::Presence => u128::from(overlap > 0),
                OverlapWeight::Linear => overlap as u128,
                OverlapWeight::Square => (overlap as u128).pow(2),
            };
            let base: Vec<_> = (BASE_SYMBOLS..n)
                .filter(|&id| active[id] != 0)
                .map(|id| ((id % 7 + 1) as u128, id as Symbol))
                .collect();
            let mut expected: Vec<_> = base
                .iter()
                .map(|&(score, tid)| {
                    let t = &m.tokens[tid as usize];
                    let mut numerator = 0;
                    let mut eligible = 0;
                    for (uid, u) in m.tokens.iter().enumerate().skip(BASE_SYMBOLS) {
                        if active[uid] == 0 {
                            continue;
                        }
                        let w = u128::from(weight(uses[uid]));
                        if config.internal || u.bytes.len() >= t.bytes.len() {
                            eligible += w;
                        }
                        if uid == tid as usize {
                            continue;
                        }
                        let mut overlap = if u.bytes.len() >= t.bytes.len() {
                            oracle_parts(t, u)
                                .into_iter()
                                .enumerate()
                                .filter(|(i, _)| config.types & (1 << i) != 0)
                                .map(|(_, v)| v)
                                .max()
                                .unwrap_or(0)
                        } else {
                            0
                        };
                        if config.internal && u.terminals() + 2 <= t.terminals() {
                            let a: Vec<_> = t
                                .bytes
                                .iter()
                                .map(|&b| u16::from(b))
                                .chain(t.ends_field.then_some(256))
                                .collect();
                            let b: Vec<_> = u
                                .bytes
                                .iter()
                                .map(|&b| u16::from(b))
                                .chain(u.ends_field.then_some(256))
                                .collect();
                            if (1..a.len() - b.len()).any(|at| a[at..at + b.len()] == b) {
                                overlap = overlap.max(b.len());
                            }
                        }
                        if overlap >= config.minimum_overlap {
                            numerator += w * mass(overlap);
                        }
                    }
                    let denominator = match config.normalization {
                        Normalization::Local => {
                            numerator + u128::from(weight(uses[tid as usize])) * mass(t.terminals())
                        }
                        Normalization::Global => eligible * mass(t.terminals()),
                    };
                    let discount = if denominator == 0 {
                        0
                    } else {
                        numerator * 30000 / denominator
                    };
                    (score * (SCALE - discount), tid)
                })
                .collect();
            expected.sort_unstable();
            let mut actual = base;
            let stats = index.adjust(
                &Tokenizer::new(m.clone()).unwrap(),
                &active,
                &uses,
                &mut actual,
                7,
                30000,
            );
            actual.sort_unstable();
            assert_eq!(actual[..7], expected[..7], "{config:?}");
            assert!(stats.ratio_ppm_max <= SCALE);
            assert!(stats.one_overlap_share_ppm_sum <= stats.targets as u128 * SCALE);
        }
    }

    fn support(m: &Model, active: &[u8], uses: &[u64], tid: usize) -> u128 {
        let t = &m.tokens[tid];
        m.tokens
            .iter()
            .enumerate()
            .filter(|(uid, u)| *uid != tid && active[*uid] != 0 && u.bytes.len() >= t.bytes.len())
            .map(|(uid, u)| u128::from(uses[uid]) * overlap(t, u) as u128)
            .sum()
    }

    #[test]
    fn indexed_coverage_matches_all_alignments_masks_lengths_and_repeated_matches() {
        let m = model();
        let n = m.tokens.len();
        let mut index = Index::new(&m);
        for seed in 0..8 {
            let mut active = vec![1; n];
            for (id, kept) in active.iter_mut().enumerate().skip(BASE_SYMBOLS) {
                *kept = u8::from((id + seed) % 5 != 0);
            }
            let uses: Vec<_> = (0..n).map(|id| ((id * 7 + seed) % 13) as u64).collect();
            for tid in BASE_SYMBOLS..n {
                if active[tid] == 0 {
                    continue;
                }
                index.first.fill(0);
                index.last.fill(0);
                index.endpoints.fill(0);
                for &uid in &index.order {
                    let u = &m.tokens[uid as usize];
                    if active[uid as usize] == 0 || u.bytes.len() < m.tokens[tid].bytes.len() {
                        continue;
                    }
                    let a = label(u, 0);
                    let b = label(u, u.terminals() - 1);
                    let count = uses[uid as usize];
                    index.first[a as usize] += count;
                    index.last[b as usize] += count;
                    index.endpoints[key(a, b)] += count;
                }
                let actual =
                    index.numerator(&m, tid as Symbol, &active, &uses, &mut Stats::default());
                assert_eq!(
                    actual,
                    support(&m, &active, &uses, tid),
                    "seed={seed}, target={tid}"
                );
            }
        }
    }

    #[test]
    fn shortlisting_and_integer_discounts_match_full_oracle_and_scale_with_corpus() {
        let m = model();
        let n = m.tokens.len();
        let active = vec![1; n];
        let uses: Vec<_> = (0..n).map(|id| (1u64 << 53) + (id % 7) as u64).collect();
        let base: Vec<_> = (BASE_SYMBOLS..n)
            .map(|id| ((1u128 << 70) + (id % 11) as u128 * 1000, id as Symbol))
            .collect();
        for penalty in [1000, 10000, 1_000_000] {
            let mut expected: Vec<_> = base
                .iter()
                .map(|&(score, id)| {
                    let numerator = support(&m, &active, &uses, id as usize);
                    let denominator = numerator
                        + u128::from(uses[id as usize]) * m.tokens[id as usize].terminals() as u128;
                    (
                        score * (SCALE - numerator * u128::from(penalty) / denominator),
                        id,
                    )
                })
                .collect();
            expected.sort_unstable();
            for limit in [1, 5, base.len()] {
                let mut index = Index::new(&m);
                let mut ranked = base.clone();
                index.adjust(
                    &Tokenizer::new(m.clone()).unwrap(),
                    &active,
                    &uses,
                    &mut ranked,
                    limit,
                    penalty,
                );
                ranked.sort_unstable();
                assert_eq!(ranked[..limit], expected[..limit]);
                let triple: Vec<_> = uses.iter().map(|x| x * 3).collect();
                let mut larger: Vec<_> = base.iter().map(|&(score, id)| (score * 3, id)).collect();
                index.adjust(
                    &Tokenizer::new(m.clone()).unwrap(),
                    &active,
                    &triple,
                    &mut larger,
                    limit,
                    penalty,
                );
                larger.sort_unstable();
                for at in 0..limit {
                    assert_eq!(larger[at], (ranked[at].0 * 3, ranked[at].1));
                }
            }
        }
    }

    #[test]
    fn absorption_works_in_both_directions_containment_and_rejects_short_alternatives() {
        let m = model();
        let index = Index::new(&m);
        let id = |value: &[u8]| {
            m.tokens
                .iter()
                .position(|t| t.bytes == value && !t.ends_field)
                .unwrap() as Symbol
        };
        let t = id(b"abcde");
        assert_eq!(index.overlap(&m, t, id(b"cdefg")), 3);
        assert_eq!(index.overlap(&m, t, id(b"zzabc")), 3);
        assert_eq!(index.overlap(&m, t, id(b"xxabcdeyy")), 5);
        let mut uses = vec![0; m.tokens.len()];
        uses[id(b"defg") as usize] = 100000;
        assert_eq!(support(&m, &vec![1; uses.len()], &uses, t as usize), 0);
    }

    #[test]
    fn shorter_internal_fragment_is_distinct_from_outer_overlap_and_edge_touching_pieces() {
        let m = model();
        let id = |value: &[u8]| {
            m.tokens
                .iter()
                .position(|t| t.bytes == value && !t.ends_field)
                .unwrap() as Symbol
        };
        let target = id(b"abcdefg");
        let middle = id(b"bcdef");
        let tokenizer = Tokenizer::new(m.clone()).unwrap();
        let active = vec![1; m.tokens.len()];
        let mut found = vec![];
        tokenizer.visit_internal_matches(target, &active, |uid, n| found.push((uid, n)));
        assert!(found.contains(&(middle, 5)));
        assert!(found
            .iter()
            .all(|&(uid, _)| uid != id(b"abcde") && uid != id(b"defg")));
        let mut uses = vec![0; m.tokens.len()];
        uses[target as usize] = 5;
        uses[middle as usize] = 12;
        let mut index = Index::new(&m);
        index.settings = Settings {
            types: 0,
            internal: true,
            ..Default::default()
        };
        let mut ranked = vec![(10, target)];
        let stats = index.adjust(&tokenizer, &active, &uses, &mut ranked, 1, 100000);
        assert_eq!(ranked[0].0, 10 * (SCALE - 60 * 100000 / 95));
        assert_eq!(stats.internal_share_ppm_sum, SCALE);
    }
}
