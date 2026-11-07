//! One concurrent pair registry. Shards partition entries; workers never own
//! pair maps. Atomic counts mutate under shared shard access, and growth takes
//! exclusive access to one shard. A global growth lock bounds resize overlap.
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, RwLock};

use crate::grammar::{Pair, Symbol, EOF};
use crate::pair_map::mix64;

const EMPTY: u64 = ((EOF as u64) << 32) | EOF as u64;
const LOG_SIZE: usize = 512;

fn slot_size(minimum_documents: u8) -> usize {
    std::mem::size_of::<Slot>() + usize::from(minimum_documents != 0) * 8
}

/// Deliberately approximate: interleaved A/B/A observations count three times.
/// The ID and capped counter share one atomic word; there is no witness set.
fn observe_document(state: &AtomicU64, document: u64, minimum: u8) {
    use crate::document_support::last_seen::DOCUMENT_MASK;
    let mut old = state.load(Ordering::Relaxed);
    loop {
        let count = (old >> 56) as u8;
        if count >= minimum || old & DOCUMENT_MASK == document {
            return;
        }
        let new = (u64::from(count + 1) << 56) | document;
        match state.compare_exchange_weak(old, new, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => return,
            Err(actual) => old = actual,
        }
    }
}

#[inline]
fn pack(pair: Pair) -> u64 {
    ((pair.0 as u64) << 32) | pair.1 as u64
}
#[inline]
fn unpack(key: u64) -> Pair {
    ((key >> 32) as u32, key as u32)
}

struct Slot {
    key: AtomicU64,
    count: AtomicU64,
}
struct Table {
    slots: Box<[Slot]>,
    documents: Option<Box<[AtomicU64]>>,
    occupied: AtomicUsize,
}

enum Update {
    Done,
    Grow,
}

impl Table {
    fn new(capacity: usize, documented: bool) -> Self {
        Self {
            slots: (0..capacity)
                .map(|_| Slot {
                    key: AtomicU64::new(EMPTY),
                    count: AtomicU64::new(0),
                })
                .collect(),
            occupied: AtomicUsize::new(0),
            documents: documented.then(|| (0..capacity).map(|_| AtomicU64::new(0)).collect()),
        }
    }

    fn apply(
        &self,
        change: Delta,
        ignore_missing: bool,
        minimum_documents: u8,
        #[cfg(feature = "profile")] metrics: &mut crate::profile::RegistryMetrics,
    ) -> Result<Update, String> {
        let Delta {
            key,
            hash,
            delta,
            document,
        } = change;
        let mask = self.slots.len() - 1;
        let mut at = hash as usize & mask;
        for _ in 0..self.slots.len() {
            #[cfg(feature = "profile")]
            {
                metrics.probes += 1;
            }
            let slot = &self.slots[at];
            let mut found = slot.key.load(Ordering::Acquire);
            if found == EMPTY {
                if ignore_missing {
                    return Ok(Update::Done);
                }
                if delta < 0 {
                    return Err("registry decrement of an absent pair".into());
                }
                if self.occupied.load(Ordering::Relaxed) >= self.slots.len() / 8 * 7 {
                    return Ok(Update::Grow);
                }
                match slot
                    .key
                    .compare_exchange(EMPTY, key, Ordering::AcqRel, Ordering::Acquire)
                {
                    Ok(_) => {
                        #[cfg(feature = "profile")]
                        {
                            metrics.registrations += 1;
                        }
                        self.occupied.fetch_add(1, Ordering::Relaxed);
                        found = key;
                    }
                    Err(actual) => {
                        #[cfg(feature = "profile")]
                        {
                            metrics.registration_collisions += 1;
                        }
                        found = actual;
                    }
                }
            }
            if found == key {
                #[cfg(feature = "profile")]
                {
                    metrics.updates += 1;
                }
                #[cfg(feature = "profile")]
                let mut attempts = 0u64;
                if delta != 0 {
                    slot.count
                        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |old| {
                            #[cfg(feature = "profile")]
                            {
                                attempts += 1;
                            }
                            if delta >= 0 {
                                old.checked_add(delta as u64)
                            } else {
                                old.checked_sub(delta.unsigned_abs())
                            }
                        })
                        .map_err(|_| "registry counter overflow or underflow")?;
                }
                #[cfg(feature = "profile")]
                {
                    metrics.counter_retries += attempts.saturating_sub(1);
                }
                if document != 0 {
                    observe_document(
                        &self.documents.as_ref().expect("documented registry")[at],
                        document,
                        minimum_documents,
                    );
                }
                return Ok(Update::Done);
            }
            at = (at + 1) & mask;
        }
        Ok(Update::Grow)
    }

    fn get(&self, key: u64, hash: u64) -> u64 {
        let mask = self.slots.len() - 1;
        let mut at = hash as usize & mask;
        for _ in 0..self.slots.len() {
            let slot = &self.slots[at];
            let found = slot.key.load(Ordering::Acquire);
            if found == key {
                return slot.count.load(Ordering::Relaxed);
            }
            if found == EMPTY {
                return 0;
            }
            at = (at + 1) & mask;
        }
        0
    }

    fn insert_quiescent(&mut self, key: u64, count: u64, document: u64) {
        let mask = self.slots.len() - 1;
        let mut at = mix64(key) as usize & mask;
        while *self.slots[at].key.get_mut() != EMPTY {
            at = (at + 1) & mask;
        }
        *self.slots[at].key.get_mut() = key;
        *self.slots[at].count.get_mut() = count;
        if let Some(documents) = &mut self.documents {
            *documents[at].get_mut() = document;
        }
        *self.occupied.get_mut() += 1;
    }
}

pub struct Registry {
    shards: Vec<RwLock<Table>>,
    shift: u32,
    growth: Mutex<()>,
    growths: AtomicUsize,
    /// Peak slot bytes, including the one old/new shard overlap during growth.
    peak_slot_bytes: AtomicUsize,
    slot_bytes: AtomicUsize,
    retired_pairs: usize,
    compactions: usize,
    /// At a merge-only barrier, missing pairs of older IDs can never recover.
    retired_before: AtomicU64,
    minimum_documents: u8,
}

impl Registry {
    pub fn new(shards: usize) -> Result<Self, String> {
        Self::with_documents(shards, 0)
    }

    /// Optional approximate support: one packed last ID/count per central slot.
    pub fn with_documents(shards: usize, minimum_documents: u8) -> Result<Self, String> {
        if minimum_documents > 64 {
            return Err("document minimum exceeds 64".into());
        }
        if !shards.is_power_of_two() || !(2..=65536).contains(&shards) {
            return Err("registry shard count must be a power of two from 2 to 65536".into());
        }
        let bytes = shards * 64 * slot_size(minimum_documents);
        Ok(Self {
            shards: (0..shards)
                .map(|_| RwLock::new(Table::new(64, minimum_documents != 0)))
                .collect(),
            shift: 64 - shards.trailing_zeros(),
            growth: Mutex::new(()),
            growths: AtomicUsize::new(0),
            peak_slot_bytes: AtomicUsize::new(bytes),
            slot_bytes: AtomicUsize::new(bytes),
            retired_pairs: 0,
            compactions: 0,
            retired_before: AtomicU64::new(0),
            minimum_documents,
        })
    }

    fn shard(&self, hash: u64) -> usize {
        (hash >> self.shift) as usize
    }

    fn grow(&self, shard: usize) -> Result<(), String> {
        let _growth = self
            .growth
            .lock()
            .map_err(|_| "poisoned registry growth lock")?;
        let mut table = self.shards[shard]
            .write()
            .map_err(|_| "poisoned registry shard")?;
        // Another worker may already have made room while this worker waited.
        if table.occupied.load(Ordering::Relaxed) < table.slots.len() / 8 * 7 {
            return Ok(());
        }
        let retired_before = self.retired_before.load(Ordering::Relaxed);
        let retain = |slot: &Slot| {
            let key = slot.key.load(Ordering::Relaxed);
            slot.count.load(Ordering::Relaxed) > 0
                || (retired_before != 0
                    && key != EMPTY
                    && (key >> 32) < retired_before
                    && (key as u32 as u64) < retired_before)
        };
        let live = table.slots.iter().filter(|s| retain(s)).count();
        let capacity = if live + 1 < table.slots.len() / 2 {
            table.slots.len()
        } else {
            table
                .slots
                .len()
                .checked_mul(2)
                .ok_or("registry capacity overflow")?
        };
        let new_bytes = capacity
            .checked_mul(slot_size(self.minimum_documents))
            .ok_or("registry size overflow")?;
        let old_bytes = table.slots.len() * slot_size(self.minimum_documents);
        let current = self.slot_bytes.load(Ordering::Relaxed);
        self.peak_slot_bytes.fetch_max(
            current
                .checked_add(new_bytes)
                .ok_or("registry size overflow")?,
            Ordering::Relaxed,
        );
        let mut replacement = Table::new(capacity, self.minimum_documents != 0);
        for (index, slot) in table.slots.iter().enumerate() {
            let count = slot.count.load(Ordering::Relaxed);
            if retain(slot) {
                replacement.insert_quiescent(
                    slot.key.load(Ordering::Relaxed),
                    count,
                    table
                        .documents
                        .as_ref()
                        .map_or(0, |d| d[index].load(Ordering::Relaxed)),
                );
            }
        }
        *table = replacement;
        self.slot_bytes
            .store(current - old_bytes + new_bytes, Ordering::Relaxed);
        self.growths.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn apply_sorted(
        &self,
        deltas: &mut [Delta],
        #[cfg(feature = "profile")] metrics: &mut crate::profile::RegistryMetrics,
    ) -> Result<(), String> {
        #[cfg(feature = "profile")]
        let sort_started = std::time::Instant::now();
        if self.minimum_documents == 0 {
            deltas.sort_unstable_by_key(|d| d.hash);
        } else {
            // Keep repeated observations from one document adjacent within the
            // bounded update log, without a worker-private map or ID array.
            deltas.sort_unstable_by_key(|d| (d.hash, d.document));
        }
        #[cfg(feature = "profile")]
        {
            metrics.flushes += 1;
            metrics.deltas += deltas.len() as u64;
            metrics.sort_seconds += sort_started.elapsed().as_secs_f64();
        }
        #[cfg(feature = "profile")]
        let apply_started = std::time::Instant::now();
        let mut at = 0;
        while at < deltas.len() {
            let shard = self.shard(deltas[at].hash);
            #[cfg(feature = "profile")]
            let lock_started = std::time::Instant::now();
            let mut table = self.shards[shard]
                .read()
                .map_err(|_| "poisoned registry shard")?;
            #[cfg(feature = "profile")]
            {
                metrics.locks += 1;
                metrics.lock_seconds += lock_started.elapsed().as_secs_f64();
            }
            while at < deltas.len() && self.shard(deltas[at].hash) == shard {
                let key = deltas[at].key;
                let hash = deltas[at].hash;
                let mut end = at + 1;
                let mut delta = deltas[at].delta;
                while end < deltas.len() && deltas[end].key == key {
                    let Some(combined) = delta.checked_add(deltas[end].delta) else {
                        break;
                    };
                    delta = combined;
                    end += 1;
                }
                let first_document = if self.minimum_documents == 0 {
                    0
                } else {
                    deltas[at..end]
                        .iter()
                        .find_map(|d| (d.document != 0).then_some(d.document))
                        .unwrap_or(0)
                };
                if delta != 0 || first_document != 0 {
                    let before = self.retired_before.load(Ordering::Relaxed);
                    let ignore_missing = (key >> 32) < before && (key as u32 as u64) < before;
                    let mut document = first_document;
                    let mut next_document = at;
                    let mut change = delta;
                    loop {
                        if let Update::Done = table.apply(
                            Delta {
                                key,
                                hash,
                                delta: change,
                                document,
                            },
                            ignore_missing,
                            self.minimum_documents,
                            #[cfg(feature = "profile")]
                            metrics,
                        )? {
                            while next_document < end && deltas[next_document].document <= document
                            {
                                next_document += 1;
                            }
                            if next_document == end {
                                break;
                            }
                            document = deltas[next_document].document;
                            change = 0;
                            continue;
                        }
                        drop(table);
                        #[cfg(feature = "profile")]
                        let grow_started = std::time::Instant::now();
                        self.grow(shard)?;
                        #[cfg(feature = "profile")]
                        {
                            metrics.growth_requests += 1;
                            metrics.grow_seconds += grow_started.elapsed().as_secs_f64();
                        }
                        #[cfg(feature = "profile")]
                        let lock_started = std::time::Instant::now();
                        table = self.shards[shard]
                            .read()
                            .map_err(|_| "poisoned registry shard")?;
                        #[cfg(feature = "profile")]
                        {
                            metrics.locks += 1;
                            metrics.lock_seconds += lock_started.elapsed().as_secs_f64();
                        }
                    }
                }
                at = end;
            }
        }
        #[cfg(feature = "profile")]
        {
            metrics.apply_seconds += apply_started.elapsed().as_secs_f64();
        }
        Ok(())
    }

    pub fn writer(&self) -> Writer<'_> {
        Writer {
            registry: self,
            log: Vec::with_capacity(LOG_SIZE),
            #[cfg(feature = "profile")]
            metrics: crate::profile::RegistryMetrics::default(),
        }
    }

    /// Requires exclusive ownership: readers see one globally quiescent epoch.
    pub fn freeze(&mut self) -> Frozen<'_> {
        Frozen { registry: self }
    }

    pub fn clear(&mut self) {
        self.retired_before.store(0, Ordering::Relaxed);
        for shard in &mut self.shards {
            let table = shard.get_mut().expect("poisoned registry shard");
            for slot in &mut table.slots {
                *slot.key.get_mut() = EMPTY;
                *slot.count.get_mut() = 0;
            }
            if let Some(documents) = &mut table.documents {
                for document in documents {
                    *document.get_mut() = 0;
                }
            }
            *table.occupied.get_mut() = 0;
        }
    }

    /// Exact merge-only retirement: an adjacency of two old IDs cannot gain
    /// occurrences when replacements emit NEW IDs. Dense symbol statistics
    /// must supply PMI marginals separately. A full recount resets this epoch.
    /// Return the surviving entry count while already walking the shards, so
    /// callers can reserve candidate storage without another table scan.
    pub fn compact_frequent(&mut self, minimum: u64, alphabet: usize) -> Result<usize, String> {
        let mut total_live = 0usize;
        for shard in &mut self.shards {
            let table = shard.get_mut().map_err(|_| "poisoned registry shard")?;
            let mut live = 0usize;
            for slot in &mut table.slots {
                let count = *slot.count.get_mut();
                if count >= minimum {
                    live += 1;
                } else if count != 0 {
                    *slot.count.get_mut() = 0;
                    self.retired_pairs += 1;
                }
            }
            total_live = total_live
                .checked_add(live)
                .ok_or("registry entry count overflow")?;
            let capacity = live
                .checked_mul(8)
                .and_then(|n| n.checked_div(7))
                .and_then(|n| n.checked_add(2))
                .and_then(usize::checked_next_power_of_two)
                .ok_or("registry compaction size overflow")?
                .max(64);
            let occupied = *table.occupied.get_mut();
            if capacity < table.slots.len() || occupied != live {
                let new_bytes = capacity * slot_size(self.minimum_documents);
                let old_bytes = table.slots.len() * slot_size(self.minimum_documents);
                let current = *self.slot_bytes.get_mut();
                *self.peak_slot_bytes.get_mut() =
                    (*self.peak_slot_bytes.get_mut()).max(current + new_bytes);
                let mut replacement = Table::new(capacity, self.minimum_documents != 0);
                for (index, slot) in table.slots.iter_mut().enumerate() {
                    let count = *slot.count.get_mut();
                    if count != 0 {
                        replacement.insert_quiescent(
                            *slot.key.get_mut(),
                            count,
                            table
                                .documents
                                .as_ref()
                                .map_or(0, |d| d[index].load(Ordering::Relaxed)),
                        );
                    }
                }
                *table = replacement;
                *self.slot_bytes.get_mut() = current - old_bytes + new_bytes;
                self.compactions += 1;
            }
        }
        self.retired_before
            .store(alphabet as u64, Ordering::Relaxed);
        Ok(total_live)
    }

    /// Filter at a quiescent barrier and collect candidates in the same walk.
    /// Deleted keys retain their probe positions until a shard is rebuilt.
    /// Sparse shards shrink one at a time, bounding old/new allocation overlap.
    pub fn retain_and_visit(
        &mut self,
        mut keep: impl FnMut(Pair) -> bool,
        mut visitor: impl FnMut(Pair, u64),
    ) {
        self.retain_and_visit_documented(&mut keep, |pair, count, _| visitor(pair, count));
    }

    /// Read and reset support in the existing candidate-collection walk.
    pub fn retain_and_visit_documented(
        &mut self,
        mut keep: impl FnMut(Pair) -> bool,
        mut visitor: impl FnMut(Pair, u64, u8),
    ) {
        #[cfg(feature = "profile")]
        let (mut scanned, mut live_entries) = (0u64, 0u64);
        for shard in &mut self.shards {
            let table = shard.get_mut().expect("poisoned registry shard");
            #[cfg(feature = "profile")]
            {
                scanned += table.slots.len() as u64;
            }
            let mut live = 0;
            for (index, slot) in table.slots.iter_mut().enumerate() {
                let support = table.documents.as_mut().map_or(0, |documents| {
                    let word = documents[index].get_mut();
                    let support = (*word >> 56) as u8;
                    *word = 0;
                    support
                });
                let count = *slot.count.get_mut();
                if count == 0 {
                    continue;
                }
                let pair = unpack(*slot.key.get_mut());
                if keep(pair) {
                    live += 1;
                    visitor(pair, count, support);
                } else {
                    *slot.count.get_mut() = 0;
                    self.retired_pairs += 1;
                }
            }
            #[cfg(feature = "profile")]
            {
                live_entries += live as u64;
            }
            if table.slots.len() > 64 && live < table.slots.len() / 4 {
                let capacity = (live * 2).next_power_of_two().max(64);
                let new_bytes = capacity * slot_size(self.minimum_documents);
                let old_bytes = table.slots.len() * slot_size(self.minimum_documents);
                let current = *self.slot_bytes.get_mut();
                *self.peak_slot_bytes.get_mut() =
                    (*self.peak_slot_bytes.get_mut()).max(current + new_bytes);
                let mut replacement = Table::new(capacity, self.minimum_documents != 0);
                for slot in &mut table.slots {
                    let count = *slot.count.get_mut();
                    if count > 0 {
                        replacement.insert_quiescent(*slot.key.get_mut(), count, 0);
                    }
                }
                *table = replacement;
                *self.slot_bytes.get_mut() = current - old_bytes + new_bytes;
                self.compactions += 1;
            }
        }
        #[cfg(feature = "profile")]
        {
            crate::profile::counter("registry.scan_slots", scanned);
            crate::profile::counter("registry.scan_live_entries", live_entries);
        }
    }

    pub fn retired_pairs(&self) -> usize {
        self.retired_pairs
    }
    pub fn compactions(&self) -> usize {
        self.compactions
    }

    pub fn slot_bytes(&self) -> usize {
        self.slot_bytes.load(Ordering::Relaxed)
    }
    pub fn peak_slot_bytes(&self) -> usize {
        self.peak_slot_bytes.load(Ordering::Relaxed)
    }
    pub fn growths(&self) -> usize {
        self.growths.load(Ordering::Relaxed)
    }
    pub fn worker_log_bytes() -> usize {
        LOG_SIZE * std::mem::size_of::<Delta>()
    }
}

#[derive(Clone, Copy)]
struct Delta {
    key: u64,
    hash: u64,
    delta: i64,
    document: u64,
}

/// A bounded 16 KiB update log, not a private count map. Full logs are combined
/// and applied centrally; their size is independent of corpus/table size.
#[must_use = "flush or finish the update log before taking a registry snapshot"]
pub struct Writer<'a> {
    registry: &'a Registry,
    log: Vec<Delta>,
    #[cfg(feature = "profile")]
    metrics: crate::profile::RegistryMetrics,
}

impl Writer<'_> {
    #[cfg(feature = "profile")]
    pub(crate) fn record_endpoint_check(&mut self, allowed: bool) {
        self.metrics.endpoint_checks += 1;
        self.metrics.skipped_pair_deltas += u64::from(!allowed);
    }
    #[cfg(feature = "profile")]
    pub(crate) fn record_start_change(&mut self) {
        self.metrics.start_counter_updates += 2;
    }
    #[cfg(feature = "profile")]
    pub(crate) fn record_dense_overlap(&mut self) {
        self.metrics.overlap_counter_updates += 1;
    }

    pub fn finish(mut self) -> Result<(), String> {
        self.flush()?;
        #[cfg(feature = "profile")]
        self.metrics.record();
        Ok(())
    }
    fn change(&mut self, pair: Pair, delta: i64) -> Result<(), String> {
        self.change_documented(pair, delta, 0)
    }
    fn change_documented(&mut self, pair: Pair, delta: i64, document: u64) -> Result<(), String> {
        let key = pack(pair);
        if key == EMPTY {
            return Err("reserved registry key".into());
        }
        self.log.push(Delta {
            key,
            hash: mix64(key),
            delta,
            document,
        });
        if self.log.len() == LOG_SIZE {
            self.flush()?;
        }
        Ok(())
    }
    pub fn pair(&mut self, pair: Pair, delta: i64) -> Result<(), String> {
        if pair.0 == EOF {
            return Err("EOF cannot start a legal pair".into());
        }
        self.change(pair, delta)
    }
    pub fn overlap(&mut self, symbol: Symbol, delta: i64) -> Result<(), String> {
        self.change((EOF, symbol), delta)
    }
    pub fn pair_document(&mut self, pair: Pair, delta: i64, document: u64) -> Result<(), String> {
        if pair.0 == EOF
            || document == 0
            || document > crate::document_support::last_seen::DOCUMENT_MASK
            || self.registry.minimum_documents == 0
        {
            return Err("invalid last-seen pair observation".into());
        }
        self.change_documented(pair, delta, document)
    }
    pub fn flush(&mut self) -> Result<(), String> {
        self.registry.apply_sorted(
            &mut self.log,
            #[cfg(feature = "profile")]
            &mut self.metrics,
        )?;
        self.log.clear();
        Ok(())
    }
}

pub struct Frozen<'a> {
    registry: &'a Registry,
}

impl Frozen<'_> {
    #[cfg(feature = "audit")]
    pub(crate) fn retires_low_counts(&self) -> bool {
        self.registry.retired_before.load(Ordering::Relaxed) != 0
    }
    pub fn get(&self, pair: Pair) -> u64 {
        let key = pack(pair);
        if key == EMPTY {
            return 0;
        }
        let hash = mix64(key);
        self.registry.shards[self.registry.shard(hash)]
            .read()
            .expect("poisoned registry shard")
            .get(key, hash)
    }
    /// Raw-overlap entries use the otherwise forbidden EOF-left namespace.
    pub fn visit(&self, mut visitor: impl FnMut(Pair, u64)) {
        #[cfg(feature = "profile")]
        let (mut scanned, mut live) = (0u64, 0u64);
        for shard in &self.registry.shards {
            let table = shard.read().expect("poisoned registry shard");
            #[cfg(feature = "profile")]
            {
                scanned += table.slots.len() as u64;
            }
            for slot in &table.slots {
                let count = slot.count.load(Ordering::Relaxed);
                if count > 0 {
                    #[cfg(feature = "profile")]
                    {
                        live += 1;
                    }
                    visitor(unpack(slot.key.load(Ordering::Relaxed)), count);
                }
            }
        }
        #[cfg(feature = "profile")]
        {
            crate::profile::counter("registry.scan_slots", scanned);
            crate::profile::counter("registry.scan_live_entries", live);
        }
    }
}
