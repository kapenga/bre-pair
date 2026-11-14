//! Matched byte-rank record transport; uses the existing frozen model and loop.
#[cfg(feature = "bytecode-experiments")]
mod enabled {
    use bre_pair::{
        bytecode::{ByteCodes, Expansions},
        shared_huffman::StaticModel,
    };
    use std::{fs, path::Path, thread, time::Instant};

    fn cpu() -> Result<f64, String> {
        #[cfg(target_os = "linux")]
        {
            #[repr(C)]
            struct Timespec {
                seconds: std::os::raw::c_long,
                nanoseconds: std::os::raw::c_long,
            }
            extern "C" {
                fn clock_gettime(clock: i32, value: *mut Timespec) -> i32;
            }
            let mut value = Timespec {
                seconds: 0,
                nanoseconds: 0,
            };
            if unsafe { clock_gettime(2, &mut value) } != 0 {
                return Err("process CPU clock failed".into());
            }
            Ok(value.seconds as f64 + value.nanoseconds as f64 * 1e-9)
        }
        #[cfg(not(target_os = "linux"))]
        {
            Err("matched process CPU measurement requires Linux".into())
        }
    }

    fn pin(core: usize) -> Result<(), String> {
        #[cfg(target_os = "linux")]
        {
            extern "C" {
                fn sched_setaffinity(pid: i32, size: usize, mask: *const usize) -> i32;
            }
            let mut mask = [0usize; 16];
            mask[core / usize::BITS as usize] |= 1usize << (core % usize::BITS as usize);
            if unsafe { sched_setaffinity(0, std::mem::size_of_val(&mask), mask.as_ptr()) } != 0 {
                return Err(std::io::Error::last_os_error().to_string());
            }
        }
        #[cfg(not(target_os = "linux"))]
        let _ = core;
        Ok(())
    }

    fn read(path: &str) -> Result<Vec<u8>, String> {
        fs::read(path).map_err(|e| e.to_string())
    }

    fn offsets(path: &str, payload: usize, first: Option<usize>) -> Result<Vec<usize>, String> {
        let data = read(path)?;
        if data.len() % 8 != 0 {
            return Err("bad index length".into());
        }
        let result = data
            .chunks_exact(8)
            .map(|b| {
                usize::try_from(u64::from_le_bytes(b.try_into().unwrap()))
                    .map_err(|_| "offset overflow".into())
            })
            .collect::<Result<Vec<_>, String>>()?;
        if result.len() < 2
            || result.last() != Some(&payload)
            || result.windows(2).any(|w| w[0] >= w[1])
            || first.is_some_and(|f| result.first() != Some(&f))
            || result[0] >= payload
        {
            return Err("invalid field offsets".into());
        }
        Ok(result)
    }

    fn pack(args: &[String]) -> Result<(), String> {
        let stored = read(&args[1])?;
        let model = StaticModel::from_bytes(&stored)?;
        let book = model.codebook();
        let decoder = model.prepare_decoder(false, 1024 << 20)?;
        let (arena, references) = decoder.bytecode_tables()?;
        let expansions = Expansions::new(arena, references)?;
        let codes = ByteCodes::from_book(&book)?;
        let original = read(&args[2])?;
        let index = offsets(&args[3], original.len(), None)?;
        let mut bytes = codes.metadata().to_vec();
        let mut positions = vec![16u64];
        let mut symbols = 0;
        for range in index.windows(2) {
            symbols += codes.transcode_field(
                &book,
                &original[range[0]..range[1]],
                &mut bytes,
                |rank| expansions.ends_field(rank),
            )?;
            positions.push(bytes.len() as u64);
        }
        let folder = Path::new(&args[4]);
        fs::create_dir(folder).map_err(|e| e.to_string())?;
        fs::write(folder.join("model.bin"), &stored).map_err(|e| e.to_string())?;
        fs::write(folder.join("payload.bin"), &bytes).map_err(|e| e.to_string())?;
        let saved: Vec<_> = positions.iter().flat_map(|n| n.to_le_bytes()).collect();
        fs::write(folder.join("offsets.u64"), &saved).map_err(|e| e.to_string())?;
        println!("{{\"rows\":{},\"symbols\":{symbols},\"model_bytes\":{},\"payload_bytes\":{},\"index_bytes\":{},\"complete_bytes\":{},\"frozen_rank_transcode\":true}}", positions.len()-1,stored.len(),bytes.len(),saved.len(),stored.len()+bytes.len()+saved.len());
        Ok(())
    }

    fn decode(args: &[String]) -> Result<(), String> {
        let workers: usize = args[5].parse().map_err(|_| "invalid workers")?;
        if ![1, 4].contains(&workers) {
            return Err("workers must be 1 or 4".into());
        }
        let stored = read(&args[1])?;
        let model = StaticModel::from_bytes(&stored)?;
        let decoder = model.prepare_decoder(false, 1024 << 20)?;
        let (arena, references) = decoder.bytecode_tables()?;
        let expansions = Expansions::new(arena, references)?;
        let mut coded = read(&args[2])?;
        let payload_size = coded.len();
        if payload_size < 16 {
            return Err("missing byte-code header".into());
        }
        let codes = ByteCodes::from_metadata(model.codebook().symbols.len(), &coded[..16])?;
        let index = offsets(&args[3], payload_size, Some(16))?;
        // Private runtime guards are not serialized or included in stored size.
        coded.extend_from_slice(&[0; 3]);
        let raw = read(&args[4])?;
        let mut raw_index = vec![0usize];
        raw_index.extend(
            raw.iter()
                .enumerate()
                .filter_map(|(i, &b)| (b == b'\n').then_some(i + 1)),
        );
        if raw_index.len() != index.len() || raw_index.last() != Some(&raw.len()) {
            return Err("expected LF-retaining records".into());
        }
        let n = index.len() - 1;
        for repeat in 0..4 {
            let mut buffers = Vec::new();
            for w in 0..workers {
                let from = n * w / workers;
                let to = n * (w + 1) / workers;
                let mut output = vec![0xa5; raw_index[to] - raw_index[from] + 16];
                output.fill(0x5a);
                std::hint::black_box(&output);
                buffers.push((from, to, output, vec![0usize; to - from]));
            }
            let before = cpu()?;
            let start = Instant::now();
            let parts = thread::scope(|scope| {
                let mut jobs = Vec::new();
                for (w, (from, to, mut output, mut lengths)) in buffers.into_iter().enumerate() {
                    let (coded, index, codes, expansions) = (&coded, &index, &codes, &expansions);
                    jobs.push(scope.spawn(move || -> Result<_, String> {
                        pin(if workers == 1 { 8 } else { [4, 6, 8, 10][w] })?;
                        let logical = output.len() - 16;
                        let mut at = 0;
                        for i in from..to {
                            let maximum = (logical - at).min(1 << 20);
                            let used = expansions.byte_field(
                                codes,
                                coded,
                                index[i]..index[i + 1],
                                &mut output[at..],
                                maximum,
                            )?;
                            lengths[i - from] = used;
                            at += used;
                        }
                        if at != logical {
                            return Err("partition length differs".into());
                        }
                        Ok((from, to, output, lengths))
                    }));
                }
                jobs.into_iter()
                    .map(|job| job.join().map_err(|_| "worker panic".to_string())?)
                    .collect::<Result<Vec<_>, _>>()
            })?;
            let seconds = start.elapsed().as_secs_f64();
            let cpu_seconds = cpu()? - before;
            for (from, to, output, lengths) in parts {
                if lengths
                    .iter()
                    .enumerate()
                    .any(|(j, &len)| len != raw_index[from + j + 1] - raw_index[from + j])
                {
                    return Err("record length differs".into());
                }
                if output[..output.len() - 16] != raw[raw_index[from]..raw_index[to]] {
                    return Err("full decoded bytes differ".into());
                }
            }
            println!("{{\"profile\":\"byte-direct\",\"repeat\":{repeat},\"warmup\":{},\"workers\":{workers},\"cpu_seconds\":{cpu_seconds:.9},\"seconds\":{seconds:.9},\"MBps\":{:.6},\"raw_bytes\":{},\"rows\":{n},\"complete_bytes\":{},\"verified\":true}}", repeat==0,raw.len() as f64/seconds/1e6,raw.len(),stored.len()+payload_size+index.len()*8);
        }
        Ok(())
    }

    pub fn run() -> Result<(), String> {
        let args: Vec<_> = std::env::args().skip(1).collect();
        match args.first().map(String::as_str) {
            Some("pack") if args.len()==5=>pack(&args),
            Some("decode") if args.len()==6=>decode(&args),
            _=>Err("record-byte pack MODEL HUFFMAN INDEX NEW_DIR | decode MODEL BYTE_PAYLOAD INDEX RAW WORKERS(1|4)".into())
        }
    }
}

fn main() {
    #[cfg(feature = "bytecode-experiments")]
    if let Err(error) = enabled::run() {
        eprintln!("record-byte: {error}");
        std::process::exit(1);
    }
    #[cfg(not(feature = "bytecode-experiments"))]
    {
        eprintln!("enable --features bytecode-experiments");
        std::process::exit(2);
    }
}
