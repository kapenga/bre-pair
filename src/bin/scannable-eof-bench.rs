//! Frozen-record comparison; no training or parser changes.
#[cfg(feature = "bytecode-experiments")]
mod experiment {
    use bre_pair::{
        bytecode::{
            scannable::{for_each_eof, for_each_eof_scalar, ScannableCodes, SparseFields},
            ByteCodes, Expansions,
        },
        shared_huffman::StaticModel,
    };
    use std::{fs, hint::black_box, path::Path, time::Instant};
    fn index(path: &str) -> Result<Vec<u64>, String> {
        let bytes = fs::read(path).map_err(|e| e.to_string())?;
        if bytes.len() % 8 != 0 {
            return Err("invalid index size".into());
        }
        Ok(bytes
            .chunks_exact(8)
            .map(|v| u64::from_le_bytes(v.try_into().unwrap()))
            .collect())
    }
    pub fn run() -> Result<(), String> {
        let args = std::env::args().collect::<Vec<_>>();
        if args.len() != 6 {
            return Err(
                "usage: scannable-eof-bench MODEL FRAMES INDEX RAW NEW_OUTPUT_DIRECTORY".into(),
            );
        }
        let folder = Path::new(&args[5]);
        fs::create_dir(folder).map_err(|e| e.to_string())?;
        let model_bytes = fs::read(&args[1]).map_err(|e| e.to_string())?;
        let model = StaticModel::from_bytes(&model_bytes)?;
        let book = model.codebook();
        let prepared = model.prepare_decoder(false, 512 << 20)?;
        let (arena, refs) = prepared.bytecode_tables()?;
        let expansions = Expansions::new(arena, refs)?;
        let original = fs::read(&args[2]).map_err(|e| e.to_string())?;
        let original_offsets = index(&args[3])?;
        if original_offsets.len() < 2
            || original_offsets[0] != 16
            || *original_offsets.last().unwrap() != original.len() as u64
            || !original_offsets.windows(2).all(|v| v[0] < v[1])
        {
            return Err("invalid source offsets".into());
        }
        let raw = fs::read(&args[4]).map_err(|e| e.to_string())?;
        let mut raw_offsets = vec![0];
        raw_offsets.extend(
            raw.iter()
                .enumerate()
                .filter_map(|(at, &b)| (b == b'\n').then_some(at + 1)),
        );
        if raw.last() != Some(&b'\n') || raw_offsets.len() != original_offsets.len() {
            return Err("raw record count mismatch".into());
        }
        let fields = original_offsets.len() - 1;
        let maximum = raw_offsets.windows(2).map(|v| v[1] - v[0]).max().unwrap();
        let ordinary = ByteCodes::from_book(&book)?;
        let codes = ScannableCodes::from_book(&book)?;
        let mut bytes = ordinary.metadata().to_vec();
        let mut byte_offsets = vec![16];
        let started = Instant::now();
        for v in original_offsets.windows(2) {
            ordinary.transcode_field(
                &book,
                &original[v[0] as usize..v[1] as usize],
                &mut bytes,
                |r| expansions.ends_field(r),
            )?;
            byte_offsets.push(bytes.len());
        }
        let byte_encode = started.elapsed().as_secs_f64();
        bytes.extend([0; 4]);
        let mut scan = vec![];
        let mut scan_offsets = vec![0];
        let started = Instant::now();
        let mut symbols = 0u64;
        for v in original_offsets.windows(2) {
            symbols += codes.transcode_field(
                &book,
                &original[v[0] as usize..v[1] as usize],
                &mut scan,
                |r| expansions.ends_field(r),
            )?;
            scan_offsets.push(scan.len());
        }
        let scan_encode = started.elapsed().as_secs_f64();
        // The persisted stream, not an in-memory-only representation, is tested.
        let mut saved = codes.metadata().to_vec();
        saved.extend_from_slice(&scan);
        fs::write(folder.join("scannable.frames"), &saved).map_err(|e| e.to_string())?;
        let saved = fs::read(folder.join("scannable.frames")).map_err(|e| e.to_string())?;
        let codes = ScannableCodes::from_metadata(&saved[..32])?;
        let scan = &saved[32..];
        let sparse = SparseFields::build(scan, 1024)?;
        fs::write(folder.join("sparse-1024.index"), sparse.metadata())
            .map_err(|e| e.to_string())?;
        if sparse.fields != fields as u64 {
            return Err("EOF scan field count mismatch".into());
        }
        let mut boundary = 0;
        let mut markers_ok = true;
        for_each_eof(scan, |at| {
            boundary += 1;
            markers_ok &= Some(&(at + 1)) == scan_offsets.get(boundary);
        });
        if !markers_ok || boundary != fields {
            return Err("false or missing EOF marker".into());
        }
        // Every stored canonical rank is compared with the frozen Huffman parse.
        let mut proof = Vec::new();
        for (field, v) in original_offsets.windows(2).enumerate() {
            let mut at = scan_offsets[field];
            let mut mismatch = false;
            proof.clear();
            codes.transcode_field(
                &book,
                &original[v[0] as usize..v[1] as usize],
                &mut proof,
                |rank| {
                    match codes.decode_rank(&scan[at..]) {
                        Ok((got, size)) => {
                            mismatch |= got != rank;
                            at += size;
                        }
                        Err(_) => mismatch = true,
                    }
                    expansions.ends_field(rank)
                },
            )?;
            if mismatch || at + 1 != scan_offsets[field + 1] || scan[at] != 0 {
                return Err("rank/EOF mismatch".into());
            }
        }
        let mut output = vec![0u8; raw.len() + 16]; // prefaulted, outside decode timer
        let mut points = vec![];
        for repeat in 0..4 {
            for step in 0..2 {
                let scannable = (step + repeat) % 2 == 1;
                let started = Instant::now();
                let mut used = 0;
                if scannable {
                    let mut at = 0;
                    let mut count = 0;
                    while at < scan.len() {
                        let (n, taken) = expansions.scannable_next(
                            &codes,
                            &scan[at..],
                            &mut output[used..],
                            maximum.min(raw.len() - used),
                        )?;
                        used += n;
                        at += taken;
                        count += 1;
                    }
                    if count != fields {
                        return Err("decoded count mismatch".into());
                    }
                } else {
                    for v in byte_offsets.windows(2) {
                        used += expansions.byte_field(
                            &ordinary,
                            &bytes,
                            v[0]..v[1],
                            &mut output[used..],
                            maximum.min(raw.len() - used),
                        )?;
                    }
                }
                let seconds = started.elapsed().as_secs_f64();
                if used != raw.len() || output[..used] != raw {
                    return Err("decoded bytes differ".into());
                }
                let profile = if scannable {
                    "scannable-base255"
                } else {
                    "ordinary-base256"
                };
                let line=format!("{{\"kind\":\"decode\",\"profile\":\"{profile}\",\"repeat\":{repeat},\"seconds\":{seconds},\"raw_MBps\":{},\"verified\":true}}",raw.len() as f64/seconds/1e6);
                println!("{line}");
                points.push(line);
            }
        }
        // Repeated compressed-byte scans; no output expansion or dense index.
        for repeat in 0..4 {
            for step in 0..3 {
                let profile_id = (step + repeat) % 3;
                let started = Instant::now();
                let mut count = 0u64;
                for _ in 0..32 {
                    if profile_id == 2 {
                        let end = *byte_offsets.last().unwrap();
                        let used = expansions.skip_byte_fields(
                            &ordinary,
                            black_box(&bytes),
                            16..end,
                            fields as u64,
                        )?;
                        if used != end - 16 {
                            return Err("ordinary skip length mismatch".into());
                        }
                        count += fields as u64;
                    } else if profile_id == 1 {
                        for_each_eof(black_box(scan), |_| count += 1);
                    } else {
                        for_each_eof_scalar(black_box(scan), |_| count += 1);
                    }
                }
                let seconds = started.elapsed().as_secs_f64();
                if black_box(count) != fields as u64 * 32 {
                    return Err("scan count mismatch".into());
                }
                let profile = ["scalar", "runtime-dispatch", "ordinary-rank-skip"][profile_id];
                let scanned = if profile_id == 2 {
                    *byte_offsets.last().unwrap() - 16
                } else {
                    scan.len()
                };
                let line=format!("{{\"kind\":\"scan\",\"profile\":\"{profile}\",\"repeat\":{repeat},\"seconds\":{seconds},\"compressed_MBps\":{},\"verified\":true}}",scanned as f64*32.0/seconds/1e6);
                println!("{line}");
                points.push(line);
            }
        }
        let mut seed = 7351u64;
        let queries = (0..4096)
            .map(|_| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                (seed % fields as u64) as usize
            })
            .collect::<Vec<_>>();
        let ordinary_sparse = byte_offsets[..fields]
            .iter()
            .step_by(1024)
            .copied()
            .collect::<Vec<_>>();
        let mut ordinary_sparse_metadata = b"BROSP001".to_vec();
        for value in [1024, fields as u64, bytes.len() as u64] {
            ordinary_sparse_metadata.extend_from_slice(&value.to_le_bytes());
        }
        for &offset in &ordinary_sparse {
            ordinary_sparse_metadata.extend_from_slice(&(offset as u64).to_le_bytes());
        }
        fs::write(
            folder.join("ordinary-sparse-1024.index"),
            &ordinary_sparse_metadata,
        )
        .map_err(|e| e.to_string())?;
        let ordinary_lookup = |field: usize| -> Result<std::ops::Range<usize>, String> {
            let mut start = ordinary_sparse[field / 1024];
            let end = *byte_offsets.last().unwrap();
            start += expansions.skip_byte_fields(
                &ordinary,
                &bytes,
                start..end,
                (field % 1024) as u64,
            )?;
            let taken = expansions.skip_byte_fields(&ordinary, &bytes, start..end, 1)?;
            Ok(start..start + taken)
        };
        for &field in &queries {
            let range = sparse.locate(field as u64)?;
            if range != (scan_offsets[field]..scan_offsets[field + 1]) {
                return Err("sparse range mismatch".into());
            }
            if ordinary_lookup(field)? != (byte_offsets[field]..byte_offsets[field + 1]) {
                return Err("ordinary sparse range mismatch".into());
            }
        }
        for repeat in 0..4 {
            for step in 0..3 {
                let profile_id = (step + repeat) % 3;
                let started = Instant::now();
                for &field in black_box(&queries) {
                    let range = if profile_id == 2 {
                        ordinary_lookup(field)?
                    } else if profile_id == 1 {
                        sparse.locate(field as u64)?
                    } else {
                        scan_offsets[field]..scan_offsets[field + 1]
                    };
                    black_box(range);
                }
                let seconds = started.elapsed().as_secs_f64();
                let profile = ["dense", "sparse-1024", "ordinary-sparse-1024"][profile_id];
                let line=format!("{{\"kind\":\"boundary-lookup\",\"profile\":\"{profile}\",\"repeat\":{repeat},\"seconds\":{seconds},\"ns_per_lookup\":{},\"verified\":true}}",seconds*1e9/queries.len() as f64);
                println!("{line}");
                points.push(line);
            }
        }
        let avx2 = cfg!(target_arch = "x86_64") && {
            #[cfg(target_arch = "x86_64")]
            {
                std::is_x86_feature_detected!("avx2")
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                false
            }
        };
        let summary=format!("{{\"verified\":true,\"raw_bytes\":{},\"fields\":{fields},\"coded_symbols\":{symbols},\"alphabet\":{},\"ordinary_slots\":{:?},\"scannable_slots\":{:?},\"model_bytes\":{},\"ordinary_stream_bytes\":{},\"scannable_stream_bytes\":{},\"dense_index_bytes\":{},\"sparse_index_bytes\":{},\"ordinary_complete_bytes\":{},\"scannable_dense_complete_bytes\":{},\"scannable_sparse_complete_bytes\":{},\"ordinary_transcode_seconds\":{byte_encode},\"scannable_transcode_seconds\":{scan_encode},\"avx2_scanner\":{avx2},\"timing_scope\":\"one worker, prepared RAM, no IO/startup in decode; scans report compressed bytes; lookup is boundary-only; transcode includes Huffman rank decoding\",\"symbol_sequence_unchanged\":true,\"trials\":[{}]}}",
            raw.len(),book.symbols.len(),ordinary.slots,codes.slots,model_bytes.len(),bytes.len(),saved.len(),byte_offsets.len()*8,sparse.metadata().len(),
            model_bytes.len()+bytes.len()+byte_offsets.len()*8,model_bytes.len()+saved.len()+scan_offsets.len()*8,model_bytes.len()+saved.len()+sparse.metadata().len(),points.join(","));
        let ordinary_sparse_complete =
            model_bytes.len() + bytes.len() + ordinary_sparse_metadata.len();
        let summary = summary.replacen(
            "\"verified\":true,",
            &format!(
                "\"verified\":true,\"ordinary_sparse_complete_bytes\":{ordinary_sparse_complete},"
            ),
            1,
        );
        fs::write(folder.join("summary.json"), summary).map_err(|e| e.to_string())?;
        Ok(())
    }
}
fn main() {
    #[cfg(feature = "bytecode-experiments")]
    if let Err(error) = experiment::run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
    #[cfg(not(feature = "bytecode-experiments"))]
    {
        eprintln!("enable --features bytecode-experiments");
        std::process::exit(2);
    }
}
