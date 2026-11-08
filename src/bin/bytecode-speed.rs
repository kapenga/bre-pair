//! Research-only speed/size profile, enabled with bytecode-experiments.
#[cfg(feature = "bytecode-experiments")]
mod experiment {
    use bre_pair::{
        bytecode::{ByteCodes, Expansions},
        huffman::Codebook,
        huffman_archive::PreparedDecoder,
        shared_huffman::StaticModel,
    };
    use std::{fs, io::Write, path::Path, thread, time::Instant};

    #[derive(Clone, Copy, Debug)]
    enum Mode {
        Original,
        Cached,
        Direct,
        Byte,
        Fixed,
    }
    struct Input<'a> {
        bytes: &'a [u8],
        offsets: &'a [u64],
    }
    struct DiskRun<'a> {
        path: &'a Path,
        started: Instant,
        partitions: Vec<[u64; 4]>,
    }
    struct Context<'a> {
        model: &'a StaticModel,
        prepared: &'a PreparedDecoder,
        legacy: &'a PreparedDecoder,
        book: &'a Codebook,
        expansions: Expansions<'a>,
        codes: &'a ByteCodes,
        fixed_width: usize,
    }
    fn pin_worker(core: usize) -> Result<(), String> {
        #[cfg(target_os = "linux")]
        {
            extern "C" {
                fn sched_setaffinity(pid: i32, size: usize, mask: *const usize) -> i32;
            }
            let mut mask = [0usize; 16];
            mask[core / usize::BITS as usize] |= 1usize << (core % usize::BITS as usize);
            // Local Linux ABI; only the calling worker's affinity changes.
            if unsafe { sched_setaffinity(0, std::mem::size_of_val(&mask), mask.as_ptr()) } != 0 {
                return Err(std::io::Error::last_os_error().to_string());
            }
        }
        #[cfg(not(target_os = "linux"))]
        let _ = core;
        Ok(())
    }
    fn offsets(path: &Path) -> Result<Vec<u64>, String> {
        let bytes = fs::read(path).map_err(|e| e.to_string())?;
        if bytes.len() % 8 != 0 {
            return Err("invalid offset index length".into());
        }
        Ok(bytes
            .chunks_exact(8)
            .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
            .collect())
    }
    fn save_index(path: &Path, index: &[u64]) -> Result<(), String> {
        let mut out = std::io::BufWriter::new(fs::File::create(path).map_err(|e| e.to_string())?);
        for x in index {
            out.write_all(&x.to_le_bytes()).map_err(|e| e.to_string())?;
        }
        out.flush().map_err(|e| e.to_string())
    }
    fn measure(
        ctx: &Context<'_>,
        mode: Mode,
        input: Input<'_>,
        raw: &[u8],
        raw_offsets: &[u64],
        workers: usize,
        disk: Option<DiskRun<'_>>,
    ) -> Result<f64, String> {
        let n = input.offsets.len() - 1;
        let mut buffers = Vec::new();
        for w in 0..workers {
            let (from, to, capacity) = if let Some(io) = &disk {
                let [from, to, begin, end] = io.partitions[w];
                (from as usize, to as usize, (end - begin) as usize)
            } else {
                let from = n * w / workers;
                let to = n * (w + 1) / workers;
                (from, to, (raw_offsets[to] - raw_offsets[from]) as usize)
            };
            let mut output = vec![0u8; capacity + 16];
            output.fill(0xa5);
            std::hint::black_box(&output);
            buffers.push((from, to, output));
        }
        let start = Instant::now();
        let parts = thread::scope(|scope| {
            let mut jobs = Vec::new();
            for (worker, (from, to, mut output)) in buffers.into_iter().enumerate() {
                let input = &input;
                jobs.push(
                    scope.spawn(move || -> Result<(usize, usize, Vec<u8>), String> {
                        pin_worker(if workers == 1 {
                            8
                        } else {
                            [4, 6, 8, 10][worker]
                        })?;
                        let mut at = 0;
                        let mut scratch = Vec::with_capacity(1 << 16);
                        for i in from..to {
                            let range = input.offsets[i] as usize..input.offsets[i + 1] as usize;
                            let maximum = (output.len() - 16 - at).min(1 << 20);
                            let used = match mode {
                                Mode::Original => {
                                    ctx.model.decode_field(
                                        ctx.legacy,
                                        &input.bytes[range],
                                        maximum,
                                        &mut scratch,
                                    )?;
                                    output[at..at + scratch.len()].copy_from_slice(&scratch);
                                    scratch.len()
                                }
                                Mode::Cached => {
                                    ctx.model.decode_field(
                                        ctx.prepared,
                                        &input.bytes[range],
                                        maximum,
                                        &mut scratch,
                                    )?;
                                    output[at..at + scratch.len()].copy_from_slice(&scratch);
                                    scratch.len()
                                }
                                Mode::Direct => ctx.expansions.huffman_field(
                                    ctx.book,
                                    &input.bytes[range],
                                    &mut output[at..],
                                    maximum,
                                )?,
                                Mode::Byte => ctx.expansions.byte_field(
                                    ctx.codes,
                                    input.bytes,
                                    range,
                                    &mut output[at..],
                                    maximum,
                                )?,
                                Mode::Fixed => ctx.expansions.fixed_field(
                                    input.bytes,
                                    range,
                                    ctx.fixed_width,
                                    &mut output[at..],
                                    maximum,
                                )?,
                            };
                            at += used;
                        }
                        if at != output.len() - 16 {
                            return Err("partition output length mismatch".into());
                        }
                        Ok((from, to, output))
                    }),
                );
            }
            jobs.into_iter()
                .map(|j| {
                    j.join()
                        .map_err(|_| "decoder worker panicked".to_string())?
                })
                .collect::<Result<Vec<_>, _>>()
        })?;
        if let Some(io) = &disk {
            let mut file = fs::File::create_new(io.path).map_err(|e| e.to_string())?;
            for (_, _, output) in &parts {
                file.write_all(&output[..output.len() - 16])
                    .map_err(|e| e.to_string())?;
            }
            file.flush().map_err(|e| e.to_string())?;
        }
        let elapsed = disk.as_ref().map_or_else(
            || start.elapsed().as_secs_f64(),
            |io| io.started.elapsed().as_secs_f64(),
        );
        // Every produced byte is checked, outside the throughput timer. Workers
        // own distinct output buffers/guards; the dictionary is shared read-only.
        for (from, to, output) in parts {
            if output[..output.len() - 16]
                != raw[raw_offsets[from] as usize..raw_offsets[to] as usize]
            {
                return Err("decoded bytes differ".into());
            }
        }
        if let Some(io) = disk {
            if fs::read(io.path).map_err(|e| e.to_string())? != raw {
                return Err("file readback differs".into());
            }
        }
        Ok(elapsed)
    }
    pub fn run() -> Result<(), String> {
        let args: Vec<_> = std::env::args().collect();
        let io_only = args.len() == 7 && args[6] == "--io-only";
        if args.len() != 6 && !io_only {
            return Err("usage: bytecode-speed MODEL FRAMES INDEX RAW OUTPUT_DIRECTORY".into());
        }
        let folder = Path::new(&args[5]);
        fs::create_dir(folder).map_err(|e| e.to_string())?;
        let model_bytes = fs::read(&args[1]).map_err(|e| e.to_string())?;
        let model = StaticModel::from_bytes(&model_bytes)?;
        let book = model.codebook();
        let prepared = model.prepare_decoder(false, 512 << 20)?;
        let mut legacy = model.prepare_decoder(false, 512 << 20)?;
        legacy.experiment_force_graph_fallback();
        let (arena, references) = prepared.bytecode_tables()?;
        let expansions = Expansions::new(arena, references)?;
        let codes = ByteCodes::from_book(&book)?;
        let fixed_width = if book.symbols.len() <= 65536 {
            2
        } else if book.symbols.len() <= 16777216 {
            3
        } else {
            4
        };
        let original = fs::read(&args[2]).map_err(|e| e.to_string())?;
        let original_offsets = offsets(Path::new(&args[3]))?;
        if original_offsets.len() < 2
            || original_offsets[0] != 16
            || *original_offsets.last().unwrap() != original.len() as u64
            || !original_offsets.windows(2).all(|p| p[0] < p[1])
        {
            return Err("invalid source field index".into());
        }
        fs::write(folder.join("model.brhm"), &model_bytes).map_err(|e| e.to_string())?;
        let mut bytes = codes.metadata().to_vec();
        let mut byte_offsets = vec![16u64];
        let mut fixed = b"BRFIX001".to_vec();
        fixed.extend_from_slice(&(fixed_width as u64).to_le_bytes());
        let mut fixed_offsets = vec![16u64];
        let mut symbol_count = 0u64;
        for range in original_offsets.windows(2) {
            symbol_count += codes.transcode_field(
                &book,
                &original[range[0] as usize..range[1] as usize],
                &mut bytes,
                |rank| {
                    fixed.extend_from_slice(&rank.to_le_bytes()[..fixed_width]);
                    expansions.ends_field(rank)
                },
            )?;
            byte_offsets.push(bytes.len() as u64);
            fixed_offsets.push(fixed.len() as u64);
        }
        bytes.extend_from_slice(&[0; 4]);
        fixed.extend_from_slice(&[0; 4]);
        fs::write(folder.join("byte.frames"), &bytes).map_err(|e| e.to_string())?;
        fs::write(folder.join("fixed.frames"), &fixed).map_err(|e| e.to_string())?;
        save_index(&folder.join("byte.offsets.u64"), &byte_offsets)?;
        save_index(&folder.join("fixed.offsets.u64"), &fixed_offsets)?;
        drop(bytes);
        drop(fixed);
        let bytes = fs::read(folder.join("byte.frames")).map_err(|e| e.to_string())?;
        let fixed = fs::read(folder.join("fixed.frames")).map_err(|e| e.to_string())?;
        let codes = ByteCodes::from_metadata(book.symbols.len(), &bytes[..16])?;
        let byte_offsets = offsets(&folder.join("byte.offsets.u64"))?;
        let fixed_offsets = offsets(&folder.join("fixed.offsets.u64"))?;
        let raw = fs::read(&args[4]).map_err(|e| e.to_string())?;
        if raw.last() != Some(&b'\n') {
            return Err("control requires LF-retaining records".into());
        }
        let mut raw_offsets = vec![0u64];
        raw_offsets.extend(
            raw.iter()
                .enumerate()
                .filter_map(|(i, &b)| (b == b'\n').then_some(i as u64 + 1)),
        );
        if raw_offsets.len() != original_offsets.len() {
            return Err("raw and coded record counts differ".into());
        }
        let ctx = Context {
            model: &model,
            prepared: &prepared,
            legacy: &legacy,
            book: &book,
            expansions,
            codes: &codes,
            fixed_width,
        };
        let file = fs::File::create(folder.join("trials.jsonl")).map_err(|e| e.to_string())?;
        let mut log = std::io::BufWriter::new(file);
        for repeat in 0..if io_only { 0 } else { 4 } {
            let modes = [
                Mode::Original,
                Mode::Cached,
                Mode::Direct,
                Mode::Byte,
                Mode::Fixed,
            ];
            for j in 0..5 {
                let mode = modes[(j + repeat) % 5];
                let (stream, index) = match mode {
                    Mode::Original | Mode::Cached | Mode::Direct => (&original, &original_offsets),
                    Mode::Byte => (&bytes, &byte_offsets),
                    Mode::Fixed => (&fixed, &fixed_offsets),
                };
                for workers in [1, 4] {
                    let elapsed = measure(
                        &ctx,
                        mode,
                        Input {
                            bytes: stream,
                            offsets: index,
                        },
                        &raw,
                        &raw_offsets,
                        workers,
                        None,
                    )?;
                    let complete = model_bytes.len() + stream.len() + index.len() * 8;
                    let line=format!("{{\"profile\":\"{mode:?}\",\"repeat\":{repeat},\"warmup\":{},\"workers\":{workers},\"seconds\":{elapsed},\"MBps\":{},\"complete_bytes\":{complete},\"verified\":true}}",repeat==0,raw.len() as f64/elapsed/1e6);
                    writeln!(log, "{line}").map_err(|e| e.to_string())?;
                    log.flush().map_err(|e| e.to_string())?;
                    println!("{line}");
                }
            }
        }
        if io_only {
            let io_folder = folder.join("io");
            fs::create_dir(&io_folder).map_err(|e| e.to_string())?;
            for workers in [1, 4] {
                let mut meta = b"BRPART01".to_vec();
                meta.extend_from_slice(&(workers as u64).to_le_bytes());
                let n = original_offsets.len() - 1;
                for w in 0..workers {
                    let from = n * w / workers;
                    let to = n * (w + 1) / workers;
                    for value in [from as u64, to as u64, raw_offsets[from], raw_offsets[to]] {
                        meta.extend_from_slice(&value.to_le_bytes());
                    }
                }
                fs::write(folder.join(format!("io-partitions-{workers}.u64")), meta)
                    .map_err(|e| e.to_string())?;
            }
            for repeat in 0..4 {
                let modes = [Mode::Direct, Mode::Byte, Mode::Fixed];
                for j in 0..3 {
                    let mode = modes[(j + repeat) % 3];
                    let (frame_path, index_path) = match mode {
                        Mode::Direct => (
                            Path::new(&args[2]).to_path_buf(),
                            Path::new(&args[3]).to_path_buf(),
                        ),
                        Mode::Byte => (folder.join("byte.frames"), folder.join("byte.offsets.u64")),
                        Mode::Fixed => (
                            folder.join("fixed.frames"),
                            folder.join("fixed.offsets.u64"),
                        ),
                        _ => unreachable!(),
                    };
                    for workers in [1, 4] {
                        let output = io_folder.join(format!("{mode:?}-{workers}-{repeat}.raw"));
                        let started = Instant::now();
                        let input = fs::read(&frame_path).map_err(|e| e.to_string())?;
                        let index = offsets(&index_path)?;
                        let meta = offsets(&folder.join(format!("io-partitions-{workers}.u64")))?;
                        if meta.len() != 2 + workers * 4 || meta[1] != workers as u64 {
                            return Err("invalid stored partition map".into());
                        }
                        let partitions = meta[2..]
                            .chunks_exact(4)
                            .map(|v| [v[0], v[1], v[2], v[3]])
                            .collect();
                        let elapsed = measure(
                            &ctx,
                            mode,
                            Input {
                                bytes: &input,
                                offsets: &index,
                            },
                            &raw,
                            &raw_offsets,
                            workers,
                            Some(DiskRun {
                                path: &output,
                                started,
                                partitions,
                            }),
                        )?;
                        let complete =
                            model_bytes.len() + input.len() + index.len() * 8 + meta.len() * 8;
                        let line=format!("{{\"profile\":\"{mode:?}\",\"repeat\":{repeat},\"warmup\":{},\"workers\":{workers},\"seconds\":{elapsed},\"MBps\":{},\"complete_bytes\":{complete},\"verified\":true,\"io\":true}}",repeat==0,raw.len() as f64/elapsed/1e6);
                        writeln!(log, "{line}").map_err(|e| e.to_string())?;
                        log.flush().map_err(|e| e.to_string())?;
                        println!("{line}");
                    }
                }
            }
        }
        let report=format!("{{\"verified\":true,\"raw_bytes\":{},\"rows\":{},\"coded_symbols\":{symbol_count},\"alphabet\":{},\"slots\":{:?},\"fixed_width\":{fixed_width},\"arena_and_references_bytes\":{},\"model_bytes\":{},\"byte_payload_bytes\":{},\"fixed_payload_bytes\":{},\"index_bytes\":{},\"source_discovery_unchanged\":true,\"allocation_uses_only_frozen_huffman_model\":true}}",raw.len(),original_offsets.len()-1,book.symbols.len(),codes.slots,prepared.arena_bytes(),model_bytes.len(),bytes.len(),fixed.len(),byte_offsets.len()*8);
        fs::write(folder.join("summary.json"), report).map_err(|e| e.to_string())?;
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
