//! Training-only ranking of inexpensive token collapses. One shared tokenizer,
//! dense global statistics, and complete-field counterfactual reparsing.
use crate::{
    grammar::{Symbol, BASE_SYMBOLS},
    tokenizer::{Model, Tokenizer},
};
use std::{
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
    sync::{
        atomic::{AtomicU64, Ordering},
        Mutex,
    },
    time::Instant,
};

#[derive(Debug)]
pub struct Removal {
    pub id: Symbol,
    pub uses: u64,
    pub replacement_tokens: u64,
    pub upper_bound: u64,
    pub sample_uses: u64,
    pub sample_penalty: u64,
    pub estimated_penalty: f64,
}

pub struct Ranking {
    pub entries: Vec<Removal>,
    pub sample_bytes: u64,
    pub sample_fields: u64,
    pub sample_stride: u64,
    pub counterfactual_fields: u64,
    pub seconds: f64,
}

/// Shortlist twice the largest requested collapse count using the existing
/// upper bound, then measure every shortlisted token's individual removal
/// penalty on sampled complete fields. Only fields using that token in the
/// chosen baseline optimum can incur a penalty. This avoids K corpus passes.
pub fn rank(
    training: &Path,
    model: &Model,
    uses: &[u64],
    maximum: usize,
    workers: usize,
    sample_bytes: u64,
) -> Result<Ranking, String> {
    if uses.len() != model.tokens.len()
        || maximum == 0
        || workers == 0
        || workers > 256
        || sample_bytes == 0
        || maximum > model.tokens.len() - BASE_SYMBOLS
    {
        return Err("invalid collapse ranking budget, counts, sample, or worker count".into());
    }
    let started = Instant::now();
    let tokenizer = Tokenizer::new(model.clone())?;
    let active = vec![1; model.tokens.len()];
    let mut entries = Vec::new();
    for (id, &count) in uses.iter().enumerate().skip(BASE_SYMBOLS) {
        let replacement_tokens = tokenizer.replacement_cost(id as Symbol, &active);
        let upper_bound = count
            .checked_mul(replacement_tokens.saturating_sub(1))
            .ok_or("removal score overflow")?;
        entries.push(Removal {
            id: id as Symbol,
            uses: count,
            replacement_tokens,
            upper_bound,
            sample_uses: 0,
            sample_penalty: 0,
            estimated_penalty: upper_bound as f64,
        });
    }
    entries.sort_unstable_by_key(|r| (r.upper_bound, r.id));
    entries.truncate(maximum.saturating_mul(2).min(entries.len()));
    let mut shortlisted = vec![false; model.tokens.len()];
    for entry in &entries {
        shortlisted[entry.id as usize] = true;
    }
    let observed: Vec<_> = uses.iter().map(|_| AtomicU64::new(0)).collect();
    let penalties: Vec<_> = uses.iter().map(|_| AtomicU64::new(0)).collect();
    let input_bytes = std::fs::metadata(training)
        .map_err(|e| e.to_string())?
        .len();
    let stride = input_bytes.div_ceil(sample_bytes).max(1);
    let input = Mutex::new((
        BufReader::new(File::open(training).map_err(|e| e.to_string())?),
        0u64,
    ));
    let bytes = AtomicU64::new(0);
    let fields = AtomicU64::new(0);
    let reparses = AtomicU64::new(0);
    let add = |counter: &AtomicU64, value: u64| {
        counter
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |old| {
                old.checked_add(value)
            })
            .map(|_| ())
            .map_err(|_| "ranking counter overflow".to_string())
    };
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for _ in 0..workers {
            let (input, tokenizer, shortlisted, observed, penalties, bytes, fields, reparses, add) = (
                &input,
                &tokenizer,
                &shortlisted,
                &observed,
                &penalties,
                &bytes,
                &fields,
                &reparses,
                &add,
            );
            handles.push(scope.spawn(move || {
                loop {
                    let mut batch = Vec::new();
                    let mut selected_bytes = 0;
                    {
                        let mut source = input.lock().map_err(|_| "poisoned ranking reader")?;
                        while batch.len() < 512 && selected_bytes < 65536 {
                            let mut field = Vec::new();
                            let length = source
                                .0
                                .read_until(b'\n', &mut field)
                                .map_err(|e| e.to_string())?;
                            if length == 0 {
                                break;
                            }
                            let selected = source.1 % stride == 0;
                            source.1 += 1;
                            if selected {
                                selected_bytes += length;
                                batch.push(field);
                            }
                        }
                    }
                    if batch.is_empty() {
                        break;
                    }
                    for field in batch {
                        add(bytes, field.len() as u64)?;
                        add(fields, 1)?;
                        let mut ids = tokenizer.encode(&field);
                        let baseline = ids.len() as u64;
                        ids.retain(|&id| shortlisted[id as usize]);
                        ids.sort_unstable();
                        let mut at = 0;
                        while at < ids.len() {
                            let id = ids[at];
                            let mut end = at + 1;
                            while end < ids.len() && ids[end] == id {
                                end += 1;
                            }
                            let cost = tokenizer.count_excluding(&field, id);
                            add(&observed[id as usize], (end - at) as u64)?;
                            add(
                                &penalties[id as usize],
                                cost.checked_sub(baseline)
                                    .ok_or("removal improved an exact optimum")?,
                            )?;
                            add(reparses, 1)?;
                            at = end;
                        }
                    }
                }
                Ok::<(), String>(())
            }));
        }
        for handle in handles {
            handle.join().map_err(|_| "ranking worker panicked")??;
        }
        Ok::<(), String>(())
    })?;
    for entry in &mut entries {
        entry.sample_uses = observed[entry.id as usize].load(Ordering::Relaxed);
        entry.sample_penalty = penalties[entry.id as usize].load(Ordering::Relaxed);
        if entry.sample_uses != 0 {
            entry.estimated_penalty =
                entry.uses as f64 * entry.sample_penalty as f64 / entry.sample_uses as f64;
        }
    }
    entries.sort_unstable_by(|a, b| {
        a.estimated_penalty
            .total_cmp(&b.estimated_penalty)
            .then_with(|| a.upper_bound.cmp(&b.upper_bound))
            .then_with(|| a.id.cmp(&b.id))
    });
    Ok(Ranking {
        entries,
        sample_bytes: bytes.load(Ordering::Relaxed),
        sample_fields: fields.load(Ordering::Relaxed),
        sample_stride: stride,
        counterfactual_fields: reparses.load(Ordering::Relaxed),
        seconds: started.elapsed().as_secs_f64(),
    })
}
