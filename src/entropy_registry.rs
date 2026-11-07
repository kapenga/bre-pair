//! One shared, sharded count/score namespace for the boundary experiment.
//! Workers have bounded contribution logs, never private pair tables.
use crate::context_entropy::WEIGHT_SCALE;
use crate::grammar::Pair;
use crate::pair_map::{mix64, PairMap};
use std::sync::Mutex;

const LOG_SIZE: usize = 512;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Aggregate {
    pub count: u64,
    low: u64,
    high: u64,
}

impl Aggregate {
    pub fn units(self) -> u128 {
        (u128::from(self.high) << 64) | u128::from(self.low)
    }
    pub fn priority(self) -> f64 {
        self.units() as f64 / WEIGHT_SCALE as f64
    }
    pub fn add(&mut self, count: u64, units: u64) -> Result<(), String> {
        let count = self
            .count
            .checked_add(count)
            .ok_or("weighted pair count overflow")?;
        let (low, carry) = self.low.overflowing_add(units);
        let high = self
            .high
            .checked_add(u64::from(carry))
            .ok_or("weighted pair sum overflow")?;
        *self = Self { count, low, high };
        Ok(())
    }
}

pub(crate) struct Registry {
    shards: Vec<Mutex<PairMap<Aggregate>>>,
}

#[derive(Clone, Copy)]
struct Contribution {
    pair: Pair,
    units: u32,
    hash: u64,
}

pub(crate) struct Writer<'a> {
    registry: &'a Registry,
    log: Vec<Contribution>,
}

impl Registry {
    pub fn new() -> Self {
        Self {
            shards: (0..64).map(|_| Mutex::new(PairMap::default())).collect(),
        }
    }
    pub fn writer(&self) -> Writer<'_> {
        Writer {
            registry: self,
            log: Vec::with_capacity(LOG_SIZE),
        }
    }
    pub fn clear(&mut self) {
        for shard in &mut self.shards {
            shard.get_mut().expect("poisoned weighted shard").clear();
        }
    }
    pub fn visit(&self, mut visit: impl FnMut(Pair, Aggregate)) {
        for shard in &self.shards {
            for (&pair, &aggregate) in shard.lock().expect("poisoned weighted shard").iter() {
                visit(pair, aggregate);
            }
        }
    }
}

fn hash(pair: Pair) -> u64 {
    mix64((u64::from(pair.0) << 32) | u64::from(pair.1))
}

impl Writer<'_> {
    pub fn push(&mut self, pair: Pair, units: u32) -> Result<(), String> {
        self.log.push(Contribution {
            pair,
            units,
            hash: hash(pair),
        });
        if self.log.len() == LOG_SIZE {
            self.flush()?;
        }
        Ok(())
    }
    pub fn flush(&mut self) -> Result<(), String> {
        self.log.sort_unstable_by_key(|entry| entry.hash);
        let mut at = 0;
        while at < self.log.len() {
            let shard = (self.log[at].hash >> 58) as usize;
            let mut table = self.registry.shards[shard]
                .lock()
                .map_err(|_| "poisoned weighted shard")?;
            while at < self.log.len() && (self.log[at].hash >> 58) as usize == shard {
                let pair = self.log[at].pair;
                let mut end = at + 1;
                let mut units = u64::from(self.log[at].units);
                while end < self.log.len() && self.log[end].pair == pair {
                    units += u64::from(self.log[end].units);
                    end += 1;
                }
                table
                    .entry(pair)
                    .or_default()
                    .add((end - at) as u64, units)?;
                at = end;
            }
        }
        self.log.clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn weighted_registry_preserves_full_counts_exact_sums_and_concurrent_updates() {
        let registry = Registry::new();
        std::thread::scope(|scope| {
            for worker in 0..4u32 {
                let registry = &registry;
                scope.spawn(move || {
                    let mut writer = registry.writer();
                    for i in 0..4096u32 {
                        writer
                            .push((u32::MAX, u32::MAX), (i % 4) * (WEIGHT_SCALE as u32 / 4))
                            .unwrap();
                        writer.push((worker, i), 1).unwrap();
                    }
                    writer.flush().unwrap();
                });
            }
        });
        let mut hot = None;
        registry.visit(|pair, value| {
            if pair == (u32::MAX, u32::MAX) {
                hot = Some(value);
            }
        });
        let hot = hot.unwrap();
        assert_eq!(hot.count, 4 * 4096);
        assert_eq!(hot.units(), u128::from(4 * 1024 * 6 * WEIGHT_SCALE / 4));
        let mut large = Aggregate {
            count: u64::from(u32::MAX),
            low: u64::MAX - 10,
            high: 7,
        };
        large.add(2, 20).unwrap();
        assert_eq!(large.count, u64::from(u32::MAX) + 2);
        assert_eq!(large.units(), (8u128 << 64) | 9);
        let mut maximum = Aggregate {
            count: u64::MAX,
            ..Default::default()
        };
        let before = maximum;
        assert!(maximum.add(1, 1).is_err());
        assert_eq!(maximum, before);
    }
}
