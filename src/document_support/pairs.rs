//! Exact selection-time support. One central sparse witness pool, no worker
//! maps or per-field document array; saturated candidates release witnesses.
use super::{u64_at, Policy};
use crate::{batch::Candidate, grammar::Grammar, packed::PackedSymbols, pair_map::PairMap};
use std::{
    collections::{hash_map::Entry, HashMap},
    fs::File,
    io::{BufReader, Read, Seek, SeekFrom},
    time::SystemTime,
};

const NONE: usize = usize::MAX;
struct Witness {
    group: [u8; 32],
    next: usize,
}
struct Support {
    heads: Vec<usize>,
    counts: Vec<u8>,
    /// Last physical document record checked for each candidate, not a group
    /// fingerprint. Full group identities below still deduplicate other records.
    last_records: Vec<u64>,
    nodes: Vec<Witness>,
    free: Vec<usize>,
    minimum: u8,
    left: usize,
}
impl Support {
    fn new(n: usize, minimum: usize) -> Self {
        Self {
            heads: vec![NONE; n],
            counts: vec![0; n],
            last_records: vec![0; n],
            nodes: vec![],
            free: vec![],
            minimum: minimum as u8,
            left: n,
        }
    }
    fn reset(&mut self, n: usize) {
        self.heads.resize(n, NONE);
        self.heads.fill(NONE);
        self.counts.resize(n, 0);
        self.counts.fill(0);
        self.last_records.resize(n, 0);
        self.last_records.fill(0);
        self.nodes.clear();
        self.free.clear();
        self.left = n;
    }
    fn observe(&mut self, id: usize, group: [u8; 32], record: u64, repeated_group: bool) {
        debug_assert_ne!(record, 0, "document record ordinals start at one");
        if self.counts[id] >= self.minimum || self.last_records[id] == record {
            return;
        }
        // Cache even a previously witnessed normalized group. Without this,
        // every repeat in a duplicate document can walk the old witness chain.
        self.last_records[id] = record;
        if repeated_group {
            let mut at = self.heads[id];
            while at != NONE {
                if self.nodes[at].group == group {
                    return;
                }
                at = self.nodes[at].next;
            }
        }
        self.counts[id] += 1;
        if self.counts[id] == self.minimum {
            self.left -= 1;
            let mut at = std::mem::replace(&mut self.heads[id], NONE);
            while at != NONE {
                let next = self.nodes[at].next;
                self.free.push(at);
                at = next;
            }
        } else if repeated_group {
            let node = Witness {
                group,
                next: self.heads[id],
            };
            let at = if let Some(at) = self.free.pop() {
                self.nodes[at] = node;
                at
            } else {
                let at = self.nodes.len();
                self.nodes.push(node);
                at
            };
            self.heads[id] = at;
        }
    }
}

pub(crate) struct PairGate {
    input: BufReader<File>,
    documents: u64,
    bytes: u64,
    fields: u64,
    minimum: usize,
    length: u64,
    modified: Option<SystemTime>,
    validated_boundaries: bool,
    lookup: PairMap<usize>,
    support: Support,
    /// One bit per physical record: its normalized group occurs more than once.
    /// The temporary full-group map is dropped before building the pair registry.
    repeated_records: Vec<u64>,
}
impl PairGate {
    pub fn new(policy: &Policy, bytes: u64, fields: u64) -> Result<Self, String> {
        if !(1..=64).contains(&policy.minimum_groups) {
            return Err("rule support requires 1..64 document groups".into());
        }
        let mut input =
            BufReader::with_capacity(65536, File::open(&policy.index).map_err(|e| e.to_string())?);
        let metadata = input.get_ref().metadata().map_err(|e| e.to_string())?;
        let mut header = [0; 24];
        input.read_exact(&mut header).map_err(|e| e.to_string())?;
        if &header[..8] != b"BREDOC01"
            || header[8..12] != 104u32.to_le_bytes()
            || header[12..16] != 0u32.to_le_bytes()
        {
            return Err(
                "rule support requires a BREDOC01 training index, never a holdout index".into(),
            );
        }
        let documents = u64_at(&header, 16);
        if documents.checked_mul(104).and_then(|n| n.checked_add(24)) != Some(metadata.len()) {
            return Err("document index length differs from record count".into());
        }
        let records = usize::try_from(documents).map_err(|_| "document index is too large")?;
        let mut repeated_records = if policy.minimum_groups > 1 {
            vec![0u64; records.div_ceil(64)]
        } else {
            vec![]
        };
        let mut first_records = HashMap::<[u8; 32], usize>::new();
        crate::training_progress::phase("document_index", 0, 0, documents, "documents");
        let (mut offset, mut found_fields) = (0u64, 0u64);
        for ordinal in 0..records {
            let mut raw = [0; 104];
            input.read_exact(&mut raw).map_err(|e| e.to_string())?;
            let (n, f) = (u64_at(&raw, 84), u64_at(&raw, 92));
            if u64_at(&raw, 76) != offset || n == 0 || f == 0 {
                return Err("invalid training document offset/length/fields".into());
            }
            offset = offset
                .checked_add(n)
                .ok_or("document byte total overflow")?;
            found_fields = found_fields
                .checked_add(f)
                .ok_or("document field total overflow")?;
            if policy.minimum_groups > 1 {
                let group: [u8; 32] = raw[32..64].try_into().unwrap();
                match first_records.entry(group) {
                    Entry::Vacant(entry) => {
                        entry.insert(ordinal);
                    }
                    Entry::Occupied(entry) => {
                        let first = *entry.get();
                        repeated_records[first / 64] |= 1 << (first % 64);
                        repeated_records[ordinal / 64] |= 1 << (ordinal % 64);
                    }
                }
            }
            if (ordinal + 1) % 65536 == 0 {
                crate::training_progress::phase(
                    "document_index",
                    0,
                    (ordinal + 1) as u64,
                    documents,
                    "documents",
                );
            }
        }
        drop(first_records);
        crate::training_progress::phase("document_index", 0, documents, documents, "documents");
        if offset != bytes || found_fields != fields {
            return Err("document index differs from training byte/field totals".into());
        }
        Ok(Self {
            input,
            documents,
            bytes,
            fields,
            minimum: policy.minimum_groups,
            length: metadata.len(),
            modified: metadata.modified().ok(),
            validated_boundaries: false,
            lookup: PairMap::default(),
            support: Support::new(0, policy.minimum_groups),
            repeated_records,
        })
    }

    pub fn filter(
        &mut self,
        blocks: &[PackedSymbols],
        grammar: &Grammar,
        candidates: &mut Vec<Candidate>,
        round: usize,
    ) -> Result<(), String> {
        let metadata = self.input.get_ref().metadata().map_err(|e| e.to_string())?;
        if metadata.len() != self.length || metadata.modified().ok() != self.modified {
            return Err("training document index changed".into());
        }
        if self.minimum == 1 || candidates.is_empty() {
            return Ok(());
        }
        crate::training_progress::phase("document_support", round, 0, self.bytes, "bytes");
        let progress = crate::training_progress::enabled();
        let mut next_progress = 64 * 1024 * 1024u64;
        self.lookup.clear();
        self.lookup.reserve(candidates.len());
        for (i, c) in candidates.iter().enumerate() {
            if self.lookup.insert(c.pair, i).is_some() {
                return Err("duplicate document-support candidate".into());
            }
        }
        self.support.reset(candidates.len());
        self.input
            .seek(SeekFrom::Start(24))
            .map_err(|e| e.to_string())?;
        let (mut left_documents, mut left_bytes, mut left_fields) = (self.documents, 0u64, 0u64);
        let (mut seen_bytes, mut seen_fields) = (0u64, 0u64);
        let mut group = [0; 32];
        let mut record = 0u64;
        let mut repeated_group = false;
        let mut previous = None;
        let mut early = false;
        'blocks: for block in blocks {
            for symbol in block.iter() {
                if left_fields == 0 {
                    if left_bytes != 0 || left_documents == 0 {
                        return Err("training document boundary differs from packed fields".into());
                    }
                    let mut raw = [0; 104];
                    self.input.read_exact(&mut raw).map_err(|e| e.to_string())?;
                    if u64_at(&raw, 76) != seen_bytes {
                        return Err("document offset differs from packed corpus".into());
                    }
                    group.copy_from_slice(&raw[32..64]);
                    left_bytes = u64_at(&raw, 84);
                    left_fields = u64_at(&raw, 92);
                    left_documents -= 1;
                    record = self.documents - left_documents;
                    let ordinal = record as usize - 1; // Document count was checked against usize above.
                    repeated_group =
                        self.repeated_records[ordinal / 64] & (1 << (ordinal % 64)) != 0;
                }
                if let Some(left) = previous {
                    if let Some(&id) = self.lookup.get(&(left, symbol)) {
                        self.support.observe(id, group, record, repeated_group);
                    }
                }
                let ending = grammar.ends_field(symbol);
                let bytes = (grammar.terminals(symbol) - usize::from(ending)) as u64;
                left_bytes = left_bytes
                    .checked_sub(bytes)
                    .ok_or("document boundary cuts a packed symbol")?;
                seen_bytes = seen_bytes
                    .checked_add(bytes)
                    .ok_or("packed document byte overflow")?;
                previous = if ending { None } else { Some(symbol) };
                if ending {
                    left_fields -= 1;
                    seen_fields = seen_fields
                        .checked_add(1)
                        .ok_or("packed document field overflow")?;
                    if left_fields == 0 && left_bytes != 0 {
                        return Err("document length differs from packed field boundaries".into());
                    }
                }
                if self.validated_boundaries && self.support.left == 0 {
                    early = true;
                    break 'blocks;
                }
            }
            if progress && seen_bytes >= next_progress {
                crate::training_progress::phase(
                    "document_support",
                    round,
                    seen_bytes,
                    self.bytes,
                    "bytes",
                );
                next_progress = seen_bytes.saturating_add(64 * 1024 * 1024);
            }
        }
        crate::training_progress::phase("document_support", round, seen_bytes, self.bytes, "bytes");
        if !early {
            if left_documents != 0
                || left_bytes != 0
                || left_fields != 0
                || seen_bytes != self.bytes
                || seen_fields != self.fields
            {
                return Err("packed input ended inside training document index".into());
            }
            self.validated_boundaries = true;
        }
        let mut id = 0;
        candidates.retain(|_| {
            let keep = self.support.counts[id] as usize >= self.minimum;
            id += 1;
            keep
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn repeated_groups_do_not_count_twice_and_saturation_recycles_nodes() {
        let mut s = Support::new(2, 3);
        s.observe(0, [1; 32], 1, true);
        s.observe(0, [2; 32], 2, true);
        s.observe(0, [1; 32], 3, true);
        assert_eq!(s.counts[0], 2);
        s.observe(0, [3; 32], 4, true);
        assert_eq!(s.counts[0], 3);
        assert_eq!(s.free.len(), 2);
        let allocated = s.nodes.len();
        s.observe(1, [4; 32], 5, true);
        s.observe(1, [5; 32], 6, true);
        assert_eq!(s.nodes.len(), allocated);
        s.observe(0, [9; 32], 7, true);
        assert_eq!(s.counts[0], 3);
        s.observe(1, [6; 32], 8, true);
        assert_eq!(s.left, 0);
        let capacity = s.nodes.capacity();
        s.reset(1);
        assert_eq!(s.nodes.capacity(), capacity);
        assert_eq!(s.left, 1);
        assert_eq!(s.counts, [0]);
        s.observe(0, [1; 32], 1, true);
        assert_eq!(s.counts, [1]);
    }

    #[test]
    fn record_markers_match_full_group_sets_across_repeats_resets_and_wide_ordinals() {
        use std::collections::HashSet;
        for minimum in [2, 8, 64] {
            let mut actual = Support::new(73, minimum);
            for epoch in 0..2 {
                actual.reset(73);
                let mut expected = vec![HashSet::new(); 73];
                let mut rng = 19u64 + epoch;
                let groups: Vec<_> = (1..=400u64)
                    .map(|ordinal| {
                        let mut group = [42; 32];
                        let key = if ordinal % 5 == 0 {
                            ordinal + 1000
                        } else {
                            ordinal % 97
                        };
                        group[24..].copy_from_slice(&key.to_le_bytes());
                        group
                    })
                    .collect();
                let mut multiplicities = HashMap::new();
                for &group in &groups {
                    *multiplicities.entry(group).or_insert(0usize) += 1;
                }
                for (index, &group) in groups.iter().enumerate() {
                    let ordinal = index as u64 + 1;
                    // Mix globally unique groups and nonadjacent repeats;
                    // all group keys share their first 24 bytes.
                    let record = (1u64 << 32) + ordinal;
                    for _ in 0..24 {
                        rng ^= rng << 13;
                        rng ^= rng >> 7;
                        rng ^= rng << 17;
                        let id = rng as usize % expected.len();
                        expected[id].insert(group);
                        // Repeated instances/fields inside the same record
                        // contribute only one distinct normalized group.
                        for _ in 0..7 {
                            actual.observe(id, group, record, multiplicities[&group] > 1);
                        }
                    }
                    for (id, groups) in expected.iter().enumerate() {
                        assert_eq!(actual.counts[id] as usize, groups.len().min(minimum));
                    }
                    assert_eq!(
                        actual.left,
                        expected.iter().filter(|g| g.len() < minimum).count()
                    );
                }
            }
        }
    }

    #[test]
    fn unique_document_groups_need_no_witness_nodes() {
        let mut support = Support::new(1, 64);
        for record in 1..=64u64 {
            let mut group = [0; 32];
            group[24..].copy_from_slice(&record.to_le_bytes());
            for _ in 0..10 {
                support.observe(0, group, record, false);
            }
            assert_eq!(support.counts[0] as u64, record);
            assert!(support.nodes.is_empty());
        }
        assert_eq!(support.left, 0);
        assert!(support.free.is_empty());
    }
}
