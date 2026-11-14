//! Research-only document-labelled allocation prior for forward discovery.
//! The production registry keeps exact raw counts. One bounded candidate index
//! and dense marginal arrays collect the extra weighted mass in a separate pass.
use crate::{
    batch::Candidate, grammar::Grammar, packed::PackedSymbols, pair_map::PairMap,
    scoring::Statistics,
};
use std::{
    io::Read,
    path::Path,
    sync::{
        atomic::{AtomicU64, AtomicUsize, Ordering},
        OnceLock,
    },
};

struct Document {
    end: u64,
    first_field: u64,
    fields: u64,
    extra: f64,
}

/// Chunk byte/field extents never change during merge-only training.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Chunk {
    pub start: u64,
    pub end: u64,
    pub first_field: u64,
    pub fields: u64,
}

/// A bounded update log, not a worker-local pair table. Integer reductions
/// are exact and independent of scheduling; one central counter array survives.
struct CounterLog<'a> {
    counters: &'a [AtomicU64],
    pending: Vec<usize>,
}
impl<'a> CounterLog<'a> {
    fn new(counters: &'a [AtomicU64]) -> Self {
        Self {
            counters,
            pending: Vec::with_capacity(1024),
        }
    }
    fn add(&mut self, index: usize) {
        self.pending.push(index);
        if self.pending.len() == 1024 {
            self.flush();
        }
    }
    fn flush(&mut self) {
        self.pending.sort_unstable();
        let mut at = 0;
        while at < self.pending.len() {
            let index = self.pending[at];
            let mut end = at + 1;
            while end < self.pending.len() && self.pending[end] == index {
                end += 1;
            }
            self.counters[index].fetch_add((end - at) as u64, Ordering::Relaxed);
            at = end;
        }
        self.pending.clear();
    }
}

pub struct Policy {
    documents: Vec<Document>,
    bytes: u64,
    fields: u64,
    weighted: bool,
    extra: f64,
    chunks: OnceLock<Vec<Chunk>>,
}

fn word(raw: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(raw[offset..offset + 8].try_into().unwrap())
}

impl Policy {
    /// Input is the existing BREDOC01 training index, never evaluation labels.
    pub fn read(path: &Path, category: u16, multiplier: f64) -> Result<Self, String> {
        if !multiplier.is_finite() || multiplier <= 0.0 {
            return Err("language multiplier must be finite and positive".into());
        }
        let mut source =
            std::io::BufReader::new(std::fs::File::open(path).map_err(|e| e.to_string())?);
        let mut header = [0; 24];
        source.read_exact(&mut header).map_err(|e| e.to_string())?;
        if &header[..8] != b"BREDOC01"
            || header[8..12] != 104u32.to_le_bytes()
            || header[12..16] != 0u32.to_le_bytes()
        {
            return Err("language weighting requires a BREDOC01 training index".into());
        }
        let count = usize::try_from(word(&header, 16))
            .map_err(|_| "document count exceeds address space")?;
        let mut documents = Vec::with_capacity(count);
        let (mut bytes, mut fields, mut found) = (0u64, 0u64, false);
        for _ in 0..count {
            let mut row = [0; 104];
            source.read_exact(&mut row).map_err(|e| e.to_string())?;
            let length = word(&row, 84);
            let n = word(&row, 92);
            if word(&row, 76) != bytes || length == 0 || n == 0 || row[103] != 0 {
                return Err("invalid language document extent".into());
            }
            bytes = bytes.checked_add(length).ok_or("document byte overflow")?;
            fields = fields.checked_add(n).ok_or("document field overflow")?;
            let matching = u16::from_le_bytes(row[100..102].try_into().unwrap()) == category;
            found |= matching;
            documents.push(Document {
                end: bytes,
                first_field: fields - n,
                fields: n,
                extra: if matching { multiplier - 1.0 } else { 0.0 },
            });
        }
        if source.read(&mut [0]).map_err(|e| e.to_string())? != 0 || !found {
            return Err("trailing index bytes or requested language absent".into());
        }
        Ok(Self {
            documents,
            bytes,
            fields,
            weighted: multiplier != 1.0,
            extra: multiplier - 1.0,
            chunks: OnceLock::new(),
        })
    }

    pub fn validate_totals(&self, bytes: u64, fields: u64) -> Result<(), String> {
        if (bytes, fields) != (self.bytes, self.fields) {
            return Err("training text and language index totals differ".into());
        }
        Ok(())
    }

    pub(crate) fn set_chunks(&self, chunks: Vec<Chunk>) -> Result<(), String> {
        let (mut end, mut fields) = (0, 0);
        for c in &chunks {
            if c.start != end || c.first_field != fields || c.end < c.start || c.fields == 0 {
                return Err("invalid language chunk extent".into());
            }
            end = c.end;
            fields = fields
                .checked_add(c.fields)
                .ok_or("language field overflow")?;
        }
        self.validate_totals(end, fields)?;
        if let Some(old) = self.chunks.get() {
            if old != &chunks {
                return Err("language policy reused with different chunks".into());
            }
        } else {
            self.chunks
                .set(chunks)
                .map_err(|_| "language chunks already initialized")?;
        }
        Ok(())
    }

    // Keep the scorer's shared training context explicit, matching field_engine.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn score(
        &self,
        candidates: &mut [Candidate],
        stats: &Statistics<'_>,
        blocks: &[PackedSymbols],
        grammar: &Grammar,
        round: usize,
        alpha: f64,
        workers: usize,
    ) -> Result<(), String> {
        // Unit weighting follows the original arithmetic and tie ordering exactly.
        if !self.weighted || candidates.is_empty() {
            return Ok(());
        }
        let mut index = PairMap::with_capacity_and_hasher(candidates.len(), Default::default());
        for (i, candidate) in candidates.iter().enumerate() {
            index.insert(candidate.pair, i);
        }
        let alphabet = grammar.alphabet_size();
        let counters: Vec<AtomicU64> = (0..3 * alphabet + candidates.len())
            .map(|_| AtomicU64::new(0))
            .collect();
        let chunks = self.chunks.get().ok_or("language chunks not initialized")?;
        if chunks.len() != blocks.len() || workers == 0 {
            return Err("language chunk count or worker mismatch".into());
        }
        let mut lengths = Vec::with_capacity(grammar.alphabet_size());
        for symbol in 0..grammar.alphabet_size() as u32 {
            lengths
                .push((grammar.terminals(symbol) - usize::from(grammar.ends_field(symbol))) as u64);
        }
        crate::training_progress::phase(
            "language_weighted_statistics",
            round,
            0,
            blocks.len() as u64,
            "chunks",
        );
        let next = AtomicUsize::new(0);
        let completed = AtomicUsize::new(0);
        std::thread::scope(|scope| -> Result<(), String> {
            let mut handles = Vec::new();
            for _ in 0..workers.min(blocks.len().max(1)) {
                handles.push(scope.spawn(|| -> Result<(), String> {
                    let mut log = CounterLog::new(&counters);
                    loop {
                        let block_id = next.fetch_add(1, Ordering::Relaxed);
                        if block_id >= blocks.len() {
                            break;
                        }
                        let chunk = chunks[block_id];
                        let block = &blocks[block_id];
                        let mut document = self
                            .documents
                            .partition_point(|d| d.first_field <= chunk.first_field)
                            .checked_sub(1)
                            .ok_or("missing chunk document")?;
                        let mut remaining = self.documents[document].first_field
                            + self.documents[document].fields
                            - chunk.first_field;
                        let mut offset = chunk.start;
                        let mut fields = 0;
                        let (mut previous, mut run) = (None, 0u64);
                        for symbol in block.iter() {
                            let doc = self
                                .documents
                                .get(document)
                                .ok_or("symbols exceed language index")?;
                            run = if previous == Some(symbol) { run + 1 } else { 1 };
                            if let Some(a) = previous {
                                if !grammar.ends_field(a) && doc.extra != 0.0 {
                                    log.add(a as usize);
                                    log.add(alphabet + symbol as usize);
                                    if a == symbol && run % 2 == 1 {
                                        log.add(2 * alphabet + symbol as usize);
                                    } else if let Some(&i) = index.get(&(a, symbol)) {
                                        log.add(3 * alphabet + i);
                                    }
                                }
                            }
                            offset = offset
                                .checked_add(lengths[symbol as usize])
                                .ok_or("language byte count overflow")?;
                            if offset > doc.end {
                                return Err("symbol crosses indexed document boundary".into());
                            }
                            if grammar.ends_field(symbol) {
                                fields += 1;
                                remaining =
                                    remaining.checked_sub(1).ok_or("too many indexed fields")?;
                                if remaining == 0 {
                                    if offset != doc.end {
                                        return Err("indexed document field/byte mismatch".into());
                                    }
                                    document += 1;
                                    remaining =
                                        self.documents.get(document).map_or(0, |d| d.fields);
                                }
                            }
                            previous = Some(symbol);
                        }
                        if !previous.is_some_and(|s| grammar.ends_field(s)) {
                            return Err("weighted chunks must be EOF-aligned".into());
                        }
                        if offset != chunk.end || fields != chunk.fields {
                            return Err("weighted chunk bytes/fields changed".into());
                        }
                        let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
                        if done % 64 == 0 {
                            crate::training_progress::phase(
                                "language_weighted_statistics",
                                round,
                                done as u64,
                                blocks.len() as u64,
                                "chunks",
                            );
                        }
                    }
                    log.flush();
                    Ok(())
                }));
            }
            for h in handles {
                h.join()
                    .map_err(|_| "language statistics worker panicked")??;
            }
            Ok(())
        })?;
        let get = |i: usize| counters[i].load(Ordering::Relaxed);
        let extra_edges = (0..alphabet)
            .map(&get)
            .try_fold(0u64, |a, b| a.checked_add(b))
            .ok_or("language edge count overflow")?;
        let total = stats.edges as f64 + extra_edges as f64 * self.extra;
        for (i, candidate) in candidates.iter_mut().enumerate() {
            let (a, b) = candidate.pair;
            let replacements = get(3 * alphabet + i);
            let observed = replacements
                + if a == b {
                    get(2 * alphabet + a as usize)
                } else {
                    0
                };
            if replacements > candidate.count || observed > stats.observed(candidate.pair) {
                return Err("language pair counts exceed raw totals".into());
            }
            let joint = stats.observed(candidate.pair) as f64 + observed as f64 * self.extra;
            let count = candidate.count as f64 + replacements as f64 * self.extra;
            let marginal = (stats.left[a as usize] as f64 + get(a as usize) as f64 * self.extra)
                * (stats.right[b as usize] as f64 + get(alphabet + b as usize) as f64 * self.extra);
            candidate.priority = count.powf(alpha) * (joint * total / marginal).log2();
            if !candidate.priority.is_finite() {
                return Err("nonfinite language-weighted priority".into());
            }
        }
        crate::training_progress::phase(
            "language_weighted_statistics",
            round,
            blocks.len() as u64,
            blocks.len() as u64,
            "chunks",
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{batch::PairCounter, engine::Config, field_engine, grammar::EOF, scoring::Model};

    fn packed(fields: &[&[u8]]) -> Vec<PackedSymbols> {
        fields
            .iter()
            .map(|field| {
                let mut p = PackedSymbols::new(9);
                for &b in *field {
                    p.push(b.into());
                }
                p.push(EOF);
                p
            })
            .collect()
    }
    fn counter(blocks: &[PackedSymbols]) -> PairCounter {
        let mut counts = PairCounter::default();
        for b in blocks {
            for s in b.iter() {
                counts.push(s, &Grammar::default());
            }
        }
        counts
    }
    fn policy(fields: &[&[u8]], weights: &[f64]) -> Policy {
        let mut end = 0;
        let documents = fields
            .iter()
            .zip(weights)
            .enumerate()
            .map(|(i, (f, &w))| {
                end += f.len() as u64;
                Document {
                    end,
                    first_field: i as u64,
                    fields: 1,
                    extra: w - 1.0,
                }
            })
            .collect();
        let chunks = fields
            .iter()
            .enumerate()
            .scan(0u64, |start, (i, f)| {
                let chunk = Chunk {
                    start: *start,
                    end: *start + f.len() as u64,
                    first_field: i as u64,
                    fields: 1,
                };
                *start = chunk.end;
                Some(chunk)
            })
            .collect();
        let p = Policy {
            documents,
            bytes: end,
            fields: fields.len() as u64,
            weighted: weights.iter().any(|&w| w != 1.0),
            extra: weights.iter().find(|&&w| w != 1.0).copied().unwrap_or(1.0) - 1.0,
            chunks: OnceLock::new(),
        };
        p.set_chunks(chunks).unwrap();
        p
    }
    #[test]
    fn weighted_statistics_match_explicit_document_replication_with_rare_edges_and_runs() {
        let fields: Vec<&[u8]> = vec![b"aaaaaabc\n", b"bcxy\n"];
        let blocks = packed(&fields);
        let counts = counter(&blocks);
        let stats = Statistics::from_counter(&counts, 257);
        let mut candidates: Vec<_> = counts
            .counts
            .iter()
            .filter(|(_, n)| **n >= 2)
            .map(|(&pair, &count)| Candidate {
                pair,
                count,
                priority: 0.0,
            })
            .collect();
        policy(&fields, &[2.0, 1.0])
            .score(
                &mut candidates,
                &stats,
                &blocks,
                &Grammar::default(),
                1,
                1.0,
                4,
            )
            .unwrap();
        let repeated = packed(&[fields[0], fields[0], fields[1]]);
        let expected = counter(&repeated);
        let wanted = Statistics::from_counter(&expected, 257);
        for c in candidates {
            let score = wanted.score(Model::WeightedPmi, c.pair, expected.counts[&c.pair]);
            assert!(
                (c.priority - score).abs() < 1e-10,
                "{:?}: {} vs {}",
                c.pair,
                c.priority,
                score
            );
            assert_eq!(c.count, counts.counts[&c.pair]);
        }
    }
    #[test]
    fn mismatched_document_extents_are_rejected() {
        let blocks = packed(&[b"abc\n"]);
        let counts = counter(&blocks);
        let stats = Statistics::from_counter(&counts, 257);
        let mut p = policy(&[b"abc\n"], &[1.5]);
        p.documents[0].end += 1;
        let mut candidates = vec![Candidate {
            pair: (97, 98),
            count: 1,
            priority: 0.0,
        }];
        assert!(p
            .score(
                &mut candidates,
                &stats,
                &blocks,
                &Grammar::default(),
                1,
                1.0,
                2
            )
            .is_err());
        assert!(p.validate_totals(5, 1).is_err());
    }
    #[test]
    fn unit_prior_is_identical_and_weighted_training_reconstructs_all_fields() {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "brepair-language-{}-{}.txt",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        let fields: Vec<&[u8]> = vec![b"abcabc abc\n", b"xyxy abcabc\n", b"abc xyxy\n"];
        std::fs::write(&path, fields.concat()).unwrap();
        let config = Config {
            minimum_count: 2,
            buffer_bytes: 8 << 20,
            score: Model::WeightedPmi,
            ..Default::default()
        };
        let options = field_engine::Options {
            frequency_cutoff: true,
            memory_saving: true,
            ..Default::default()
        };
        let original = field_engine::compress_file(&path, &config, 2, 12, options, |_| {}).unwrap();
        let unit = field_engine::compress_file_language_weighted(
            &path,
            &policy(&fields, &[1.0; 3]),
            &config,
            2,
            12,
            options,
            |_| {},
        )
        .unwrap();
        let mut a = Vec::new();
        let mut b = Vec::new();
        crate::archive_io::write(&original.training.compressed, &mut a).unwrap();
        crate::archive_io::write(&unit.training.compressed, &mut b).unwrap();
        assert_eq!(a, b);
        let boosted = field_engine::compress_file_language_weighted(
            &path,
            &policy(&fields, &[1.0, 1.295720025165964, 1.0]),
            &config,
            2,
            12,
            options,
            |_| {},
        )
        .unwrap();
        assert_eq!(
            boosted.training.compressed.decode().unwrap(),
            fields.iter().map(|f| f.to_vec()).collect::<Vec<_>>()
        );
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn parallel_counts_are_deterministic_when_chunks_split_and_cross_documents() {
        let fields: Vec<Vec<u8>> = (0..240)
            .map(|i| {
                if i % 3 == 0 {
                    b"aaaaabcabc\n".to_vec()
                } else {
                    b"xyxyabcabc\n".to_vec()
                }
            })
            .collect();
        let path = std::env::temp_dir().join(format!(
            "brepair-language-parallel-{}.txt",
            std::process::id()
        ));
        std::fs::write(&path, fields.concat()).unwrap();
        let first = fields[..97].iter().map(|f| f.len() as u64).sum();
        let bytes = fields.iter().map(|f| f.len() as u64).sum();
        let make = || Policy {
            documents: vec![
                Document {
                    end: first,
                    first_field: 0,
                    fields: 97,
                    extra: 0.295720025165964,
                },
                Document {
                    end: bytes,
                    first_field: 97,
                    fields: 143,
                    extra: 0.0,
                },
            ],
            bytes,
            fields: 240,
            weighted: true,
            extra: 0.295720025165964,
            chunks: OnceLock::new(),
        };
        let config = Config {
            minimum_count: 2,
            buffer_bytes: 8 << 20,
            score: Model::WeightedPmi,
            ..Default::default()
        };
        let options = field_engine::Options {
            frequency_cutoff: true,
            memory_saving: true,
            ..Default::default()
        };
        let a = field_engine::compress_file_language_weighted(
            &path,
            &make(),
            &config,
            1,
            45,
            options,
            |_| {},
        )
        .unwrap();
        let b = field_engine::compress_file_language_weighted(
            &path,
            &make(),
            &config,
            4,
            45,
            options,
            |_| {},
        )
        .unwrap();
        let mut one = Vec::new();
        let mut four = Vec::new();
        crate::archive_io::write(&a.training.compressed, &mut one).unwrap();
        crate::archive_io::write(&b.training.compressed, &mut four).unwrap();
        assert_eq!(one, four);
        assert_eq!(b.training.compressed.decode().unwrap(), fields);
        std::fs::remove_file(path).unwrap();
    }
}
