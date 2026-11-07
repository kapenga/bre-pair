//! Training-only, capped document support for a selected vocabulary.
//! One central set per active learned token; no worker-private maps.
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use crate::grammar::{Symbol, BASE_SYMBOLS};
use crate::tokenizer::Tokenizer;
use crate::vocabulary::{Scan, Totals};
mod cleanup;
pub(crate) mod last_seen;
mod pairs;
pub use cleanup::{CandidateSupport, Cleanup};
pub(crate) use pairs::PairGate;

#[derive(Clone, Debug)]
pub struct Policy {
    pub index: PathBuf,
    pub minimum_groups: usize,
}

pub struct DocumentScan {
    pub scan: Scan,
    /// Exact until saturated at the requested minimum; indexed by pool ID.
    pub groups: Vec<u32>,
}

struct Support {
    count: AtomicUsize,
    keys: Mutex<Vec<[u8; 32]>>,
}

impl Support {
    fn observe(&self, group: [u8; 32], minimum: usize) -> Result<(), String> {
        if self.count.load(Ordering::Relaxed) >= minimum {
            return Ok(());
        }
        let mut keys = self.keys.lock().map_err(|_| "poisoned document support")?;
        if keys.len() < minimum && !keys.contains(&group) {
            keys.push(group);
            self.count.store(keys.len(), Ordering::Relaxed);
        }
        Ok(())
    }
}

struct Field {
    bytes: Vec<u8>,
    group: [u8; 32],
}

struct Input {
    source: BufReader<File>,
    index: BufReader<File>,
    documents_left: u64,
    bytes_left: u64,
    fields_left: u64,
    offset: u64,
    group: [u8; 32],
    total_bytes: u64,
    next_progress: u64,
    progress: bool,
}

fn u64_at(raw: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(raw[at..at + 8].try_into().unwrap())
}

impl Input {
    fn new(source: &Path, index: &Path) -> Result<Self, String> {
        let mut index = BufReader::new(File::open(index).map_err(|e| e.to_string())?);
        let mut header = [0; 24];
        index.read_exact(&mut header).map_err(|e| e.to_string())?;
        if &header[..8] != b"BREDOC01"
            || header[8..12] != 104u32.to_le_bytes()
            || header[12..16] != 0u32.to_le_bytes()
        {
            return Err(
                "document support requires a BREDOC01 training index, never a holdout index".into(),
            );
        }
        let source_file = File::open(source).map_err(|e| e.to_string())?;
        let total_bytes = source_file.metadata().map_err(|e| e.to_string())?.len();
        let progress = crate::training_progress::enabled();
        crate::training_progress::phase("vocabulary_documents", 0, 0, total_bytes, "bytes");
        Ok(Self {
            source: BufReader::new(source_file),
            index,
            documents_left: u64_at(&header, 16),
            bytes_left: 0,
            fields_left: 0,
            offset: 0,
            group: [0; 32],
            total_bytes,
            next_progress: 64 * 1024 * 1024,
            progress,
        })
    }

    fn batch(&mut self) -> Result<Vec<Field>, String> {
        let mut result = Vec::new();
        let mut size = 0;
        while result.len() < 512 && size < 65536 {
            if self.bytes_left == 0 {
                if self.fields_left != 0 {
                    return Err("document field count differs from its byte length".into());
                }
                if self.documents_left == 0 {
                    if self.source.read(&mut [0]).map_err(|e| e.to_string())? != 0
                        || self.index.read(&mut [0]).map_err(|e| e.to_string())? != 0
                    {
                        return Err("trailing training bytes or document index records".into());
                    }
                    break;
                }
                let mut raw = [0; 104];
                self.index.read_exact(&mut raw).map_err(|e| e.to_string())?;
                if u64_at(&raw, 76) != self.offset || u64_at(&raw, 84) == 0 {
                    return Err("document index offset/length differs from training input".into());
                }
                self.bytes_left = u64_at(&raw, 84);
                self.fields_left = u64_at(&raw, 92);
                self.group = raw[32..64].try_into().unwrap();
                self.documents_left -= 1;
            }
            if self.fields_left == 0 {
                return Err("document contains more fields than its index declares".into());
            }
            let mut bytes = Vec::new();
            let read = (&mut self.source)
                .take(self.bytes_left)
                .read_until(b'\n', &mut bytes)
                .map_err(|e| e.to_string())?;
            if read == 0 {
                return Err("training input ended inside a document".into());
            }
            self.bytes_left -= read as u64;
            self.fields_left -= 1;
            self.offset += read as u64;
            if self.bytes_left == 0 && self.documents_left != 0 && !bytes.ends_with(b"\n") {
                return Err("document boundary cuts an original line field".into());
            }
            size += read;
            result.push(Field {
                bytes,
                group: self.group,
            });
        }
        if self.progress && (self.offset >= self.next_progress || result.is_empty()) {
            crate::training_progress::phase(
                "vocabulary_documents",
                0,
                self.offset,
                self.total_bytes,
                "bytes",
            );
            self.next_progress = self.offset.saturating_add(64 * 1024 * 1024);
        }
        Ok(result)
    }
}

pub fn scan(
    training: &Path,
    tokenizer: &Tokenizer,
    active: Option<&[u8]>,
    workers: usize,
    policy: &Policy,
) -> Result<DocumentScan, String> {
    if !(1..=256).contains(&workers) || !(1..=64).contains(&policy.minimum_groups) {
        return Err("document support requires 1..256 workers and 1..64 document groups".into());
    }
    if active.is_some_and(|a| {
        a.len() != tokenizer.model().tokens.len() || a[..BASE_SYMBOLS].contains(&0)
    }) {
        return Err("document support active set must retain the byte/EOF basis".into());
    }
    let started = Instant::now();
    let ids: Vec<_> = (0..tokenizer.model().tokens.len())
        .filter(|&id| active.is_none_or(|a| a[id] != 0))
        .collect();
    let compact = if let Some(active) = active {
        tokenizer.model().subset(active)?
    } else {
        tokenizer.model().clone()
    };
    let compact = Tokenizer::new(compact)?;
    let input = Mutex::new(Input::new(training, &policy.index)?);
    let uses: Vec<_> = ids.iter().map(|_| AtomicU64::new(0)).collect();
    let support: Vec<_> = ids
        .iter()
        .map(|&id| Support {
            count: AtomicUsize::new(0),
            keys: Mutex::new(Vec::with_capacity(if id < BASE_SYMBOLS {
                0
            } else {
                policy.minimum_groups
            })),
        })
        .collect();
    let totals = std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for _ in 0..workers {
            let (input, compact, uses, support) = (&input, &compact, &uses, &support);
            handles.push(scope.spawn(move || {
                let mut totals = Totals::default();
                let mut log = Vec::<Symbol>::with_capacity(4096);
                let flush = |log: &mut Vec<Symbol>| {
                    log.sort_unstable();
                    let mut at = 0;
                    while at < log.len() {
                        let mut end = at + 1;
                        while end < log.len() && log[end] == log[at] {
                            end += 1;
                        }
                        uses[log[at] as usize].fetch_add((end - at) as u64, Ordering::Relaxed);
                        at = end;
                    }
                    log.clear();
                };
                loop {
                    let batch = input
                        .lock()
                        .map_err(|_| "poisoned document reader")?
                        .batch()?;
                    if batch.is_empty() {
                        break;
                    }
                    for field in batch {
                        let encoded = compact.encode(&field.bytes);
                        if compact.model().decode_field(&encoded)? != field.bytes {
                            return Err("document-aware parser changed original bytes".into());
                        }
                        totals.add(Totals {
                            bytes: field.bytes.len() as u64,
                            tokens: encoded.len() as u64,
                            fields: 1,
                            byte_tokens: encoded.iter().filter(|&&id| id < 256).count() as u64,
                        });
                        for id in encoded {
                            if id as usize >= BASE_SYMBOLS {
                                support[id as usize].observe(field.group, policy.minimum_groups)?;
                            }
                            log.push(id);
                            if log.len() == 4096 {
                                flush(&mut log);
                            }
                        }
                    }
                }
                flush(&mut log);
                Ok::<_, String>(totals)
            }));
        }
        let mut totals = Totals::default();
        for handle in handles {
            totals.add(
                handle
                    .join()
                    .map_err(|_| "document support worker panicked")??,
            );
        }
        Ok::<_, String>(totals)
    })?;
    let mut all_uses = vec![0; tokenizer.model().tokens.len()];
    let mut groups = vec![0; all_uses.len()];
    for (compact_id, &pool_id) in ids.iter().enumerate() {
        all_uses[pool_id] = uses[compact_id].load(Ordering::Relaxed);
        groups[pool_id] = support[compact_id].count.load(Ordering::Relaxed) as u32;
    }
    Ok(DocumentScan {
        scan: Scan {
            all: totals,
            heldout: totals,
            uses: all_uses,
            seconds: started.elapsed().as_secs_f64(),
        },
        groups,
    })
}
