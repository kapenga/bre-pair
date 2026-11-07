//! Bounded calibration graphs for joint vocabulary selection. The corpus is
//! streamed; one shared candidate index and one dense statistics array suffice.
use crate::{
    grammar::BASE_SYMBOLS,
    tokenizer::{fingerprint, Model, Tokenizer},
    vocabulary,
};
use std::{
    collections::HashMap,
    fs::{self, File},
    io::{BufRead, BufReader, BufWriter, Seek, SeekFrom, Write},
    path::Path,
};

pub struct Options {
    pub sample_bytes: u64,
    pub maximum_edges: u64,
    pub workers: usize,
    pub minimum_uses: u64,
}

pub fn prepare(
    pool: Model,
    initial: &Model,
    training: &Path,
    output: &Path,
    options: &Options,
) -> Result<(), String> {
    if options.sample_bytes == 0 || options.maximum_edges == 0 || options.minimum_uses == 0 {
        return Err("joint calibration budgets and minimum usage must be positive".into());
    }
    let lookup: HashMap<_, _> = pool
        .tokens
        .iter()
        .enumerate()
        .map(|(i, t)| (t, i))
        .collect();
    let mut active = vec![0u8; pool.tokens.len()];
    for token in &initial.tokens {
        active[*lookup
            .get(token)
            .ok_or("initial token absent from candidate pool")?] = 1;
    }
    drop(lookup);
    let tokenizer = Tokenizer::new(pool)?;
    let scored = vocabulary::scan(
        training,
        &tokenizer,
        None,
        Some(&active),
        options.workers,
        1,
        0,
        true,
        true,
    )?;
    if active
        .iter()
        .zip(&scored.uses)
        .skip(BASE_SYMBOLS)
        .any(|(&a, &n)| a != 0 && n < options.minimum_uses)
    {
        return Err("initial vocabulary has entries below minimum training usage".into());
    }
    fs::create_dir(output).map_err(|e| e.to_string())?;
    let io = |e: std::io::Error| e.to_string();
    fs::write(output.join("pool.brt"), tokenizer.model().to_bytes()).map_err(io)?;
    let mut stats = BufWriter::new(File::create(output.join("statistics.bin")).map_err(io)?);
    stats.write_all(b"BRTSTA01").map_err(io)?;
    stats
        .write_all(&(active.len() as u32).to_le_bytes())
        .map_err(io)?;
    stats
        .write_all(&(initial.tokens.len() as u32).to_le_bytes())
        .map_err(io)?;
    for (id, &flag) in active.iter().enumerate() {
        let loss = if flag != 0 && id >= BASE_SYMBOLS {
            scored.uses[id]
                .checked_mul(
                    tokenizer
                        .replacement_cost(id as u32, &active)
                        .saturating_sub(1),
                )
                .ok_or("removal loss overflow")?
        } else {
            0
        };
        stats
            .write_all(&scored.uses[id].to_le_bytes())
            .map_err(io)?;
        stats.write_all(&loss.to_le_bytes()).map_err(io)?;
        stats.write_all(&[flag]).map_err(io)?;
    }
    stats.flush().map_err(io)?;
    let stride = scored.all.bytes.div_ceil(options.sample_bytes).max(1);
    let mut source = BufReader::new(File::open(training).map_err(io)?);
    let mut graph = BufWriter::new(File::create(output.join("graph.bin")).map_err(io)?);
    graph.write_all(b"BRTGRF01").map_err(io)?;
    graph.write_all(&0u64.to_le_bytes()).map_err(io)?;
    let (mut field_index, mut offset, mut fields, mut bytes, mut edge_count) =
        (0u64, 0u64, 0u64, 0u64, 0u64);
    let (mut skipped_large, mut skipped_edges) = (0u64, 0u64);
    let mut field = Vec::new();
    let mut edges = Vec::new();
    loop {
        field.clear();
        let n = source.read_until(b'\n', &mut field).map_err(io)?;
        if n == 0 {
            break;
        }
        let selected = field_index % stride == 0;
        if selected && n as u64 <= options.sample_bytes - bytes {
            edges.clear();
            let complete = tokenizer
                .matching_edges(&field, |start, end, id| {
                    if edges.len() as u64 >= options.maximum_edges - edge_count {
                        return Err("edge budget".into());
                    }
                    edges.push((start as u64, end as u64, id));
                    Ok(())
                })
                .is_ok();
            if complete {
                for value in [field_index, offset, n as u64, edges.len() as u64] {
                    graph.write_all(&value.to_le_bytes()).map_err(io)?;
                }
                graph.write_all(&field).map_err(io)?;
                for &(start, end, id) in &edges {
                    graph.write_all(&start.to_le_bytes()).map_err(io)?;
                    graph.write_all(&end.to_le_bytes()).map_err(io)?;
                    graph.write_all(&id.to_le_bytes()).map_err(io)?;
                }
                fields += 1;
                bytes += n as u64;
                edge_count += edges.len() as u64;
            } else {
                skipped_edges += 1;
            }
        } else if selected {
            skipped_large += 1;
        }
        offset += n as u64;
        field_index += 1;
    }
    graph.seek(SeekFrom::Start(8)).map_err(io)?;
    graph.write_all(&fields.to_le_bytes()).map_err(io)?;
    graph.flush().map_err(io)?;
    if fields == 0 {
        return Err("calibration budget retained no complete fields".into());
    }
    let report = format!(
        concat!(
            "{{\"pool\":{},\"vocabulary\":{},\"training_bytes\":{},\"training_tokens\":{},",
            "\"training_fields\":{},\"initial_fingerprint\":{},\"sample_bytes\":{},",
            "\"sample_fields\":{},\"sample_stride\":{},\"edges\":{},",
            "\"skipped_large_fields\":{},\"skipped_edge_budget_fields\":{},",
            "\"source_verified\":true,\"round_trip_verified\":true}}\n"
        ),
        active.len(),
        initial.tokens.len(),
        scored.all.bytes,
        scored.all.tokens,
        scored.all.fields,
        fingerprint(&initial.to_bytes()),
        bytes,
        fields,
        stride,
        edge_count,
        skipped_large,
        skipped_edges
    );
    fs::write(output.join("preparation.json"), report).map_err(io)?;
    Ok(())
}
