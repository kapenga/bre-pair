//! Matched resident-output benchmark of the production bounded field API.
//! Input/model setup and byte comparison are outside each throughput timer.
use bre_pair::shared_huffman::StaticModel;
use std::{fs, thread, time::Instant};

fn pin(core: usize) -> Result<(), String> {
    #[cfg(target_os = "linux")]
    {
        extern "C" {
            fn sched_setaffinity(pid: i32, size: usize, mask: *const usize) -> i32;
        }
        let mut mask = [0usize; 16];
        mask[core / usize::BITS as usize] |= 1usize << (core % usize::BITS as usize);
        // Linux cpu_set_t-compatible local mask, valid for this call's duration.
        if unsafe { sched_setaffinity(0, std::mem::size_of_val(&mask), mask.as_ptr()) } != 0 {
            return Err(std::io::Error::last_os_error().to_string());
        }
    }
    #[cfg(not(target_os = "linux"))]
    let _ = core;
    Ok(())
}
fn run() -> Result<(), String> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 5 {
        return Err("record-decode MODEL FRAMES INDEX RAW WORKERS(1|4)".into());
    }
    let workers: usize = args[4].parse().map_err(|_| "invalid workers")?;
    if ![1, 4].contains(&workers) {
        return Err("workers must be 1 or 4".into());
    }
    let read = |p: &str| fs::read(p).map_err(|e| e.to_string());
    let stored = read(&args[0])?;
    let model = StaticModel::from_bytes(&stored)?;
    let decoder = model.prepare_decoder(false, 64 << 20)?;
    let coded = read(&args[1])?;
    let saved = read(&args[2])?;
    if saved.len() % 8 != 0 {
        return Err("bad index length".into());
    }
    let offsets: Vec<usize> = saved
        .chunks_exact(8)
        .map(|b| {
            usize::try_from(u64::from_le_bytes(b.try_into().unwrap()))
                .map_err(|_| "offset overflow")
        })
        .collect::<Result<_, _>>()?;
    if offsets.len() < 2
        || offsets.last() != Some(&coded.len())
        || offsets.windows(2).any(|w| w[0] >= w[1])
    {
        return Err("invalid field offsets".into());
    }
    let raw = read(&args[3])?;
    let mut raw_offsets = vec![0usize];
    raw_offsets.extend(
        raw.iter()
            .enumerate()
            .filter_map(|(i, &b)| (b == b'\n').then_some(i + 1)),
    );
    if raw_offsets.len() != offsets.len() || raw_offsets.last() != Some(&raw.len()) {
        return Err("benchmark requires one EOF field per LF-retaining row".into());
    }
    let n = offsets.len() - 1;
    for repeat in 0..4 {
        let mut buffers = Vec::new();
        for w in 0..workers {
            let from = n * w / workers;
            let to = n * (w + 1) / workers;
            let mut output = vec![0xa5u8; raw_offsets[to] - raw_offsets[from] + 16];
            output.fill(0x5a);
            std::hint::black_box(&output);
            buffers.push((from, to, output));
        }
        let start = Instant::now();
        let parts = thread::scope(|scope| {
            let jobs: Vec<_> = buffers
                .into_iter()
                .enumerate()
                .map(|(w, (from, to, mut output))| {
                    let (model, decoder, coded, offsets) = (&model, &decoder, &coded, &offsets);
                    scope.spawn(move || -> Result<_, String> {
                        pin(if workers == 1 { 8 } else { [4, 6, 8, 10][w] })?;
                        let logical = output.len() - 16;
                        let mut at = 0;
                        for i in from..to {
                            // Limit by remaining capacity, never by the true size of
                            // this row. Raw offsets are only partition/validation data.
                            let maximum = (logical - at).min(1 << 20);
                            let (used, _, _) = model.decode_field_into(
                                decoder,
                                &coded[offsets[i]..offsets[i + 1]],
                                &mut output[at..at + maximum],
                            )?;
                            at += used;
                        }
                        if at != logical {
                            return Err("partition length mismatch".into());
                        }
                        Ok((from, to, output))
                    })
                })
                .collect();
            jobs.into_iter()
                .map(|j| j.join().map_err(|_| "worker panic".to_string())?)
                .collect::<Result<Vec<_>, _>>()
        })?;
        let seconds = start.elapsed().as_secs_f64();
        for (from, to, output) in parts {
            if output[..output.len() - 16] != raw[raw_offsets[from]..raw_offsets[to]] {
                return Err("full decoded bytes differ".into());
            }
        }
        println!("{{\"profile\":\"production-direct\",\"repeat\":{repeat},\"warmup\":{},\"workers\":{workers},\"seconds\":{seconds:.9},\"MBps\":{:.6},\"raw_bytes\":{},\"rows\":{n},\"complete_bytes\":{},\"verified\":true}}",
            repeat == 0, raw.len() as f64 / seconds / 1e6, raw.len(), stored.len()+coded.len()+saved.len());
    }
    Ok(())
}
fn main() {
    if let Err(error) = run() {
        eprintln!("record-decode: {error}");
        std::process::exit(1);
    }
}
