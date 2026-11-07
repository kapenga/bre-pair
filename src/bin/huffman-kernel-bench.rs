//! Matched in-memory hot-loop trials. No corpus IO or expanded-byte copying
//! inside timing. Every variant must reproduce the reference rank sequence.
#[cfg(not(feature = "huffman-kernels"))]
fn main() {
    eprintln!("build with --features huffman-kernels --bin huffman-kernel-bench");
    std::process::exit(1);
}
#[cfg(feature = "huffman-kernels")]
fn run() -> Result<(), String> {
    use bre_pair::{huffman::Kernel, huffman_archive::Reader, profile::thread_cpu_seconds};
    use std::{fs::File, io::BufReader, time::Instant};
    let mut args: Vec<_> = std::env::args().skip(1).collect();
    let fixture = if args.len() >= 3 && args[args.len() - 2] == "--fixture" {
        let path = args.pop().unwrap();
        args.pop();
        Some(path)
    } else {
        None
    };
    if args.is_empty() || args.len() > 4 {
        return Err(
            "huffman-kernel-bench INPUT.brh [BLOCK=0] [ROUNDS=7] [TARGET_SYMBOLS=16000000]".into(),
        );
    }
    let parse = |i: usize, default: usize| {
        args.get(i).map_or(Ok(default), |s| {
            s.parse::<usize>()
                .map_err(|_| "invalid numeric argument".to_string())
        })
    };
    let block_number = parse(1, 0)?;
    let rounds = parse(2, 7)?;
    let target = parse(3, 16000000)?;
    if !(3..=31).contains(&rounds) || target == 0 {
        return Err("rounds must be 3..31 and target positive".into());
    }
    let pinned_cpu = bre_pair::profile::pin_benchmark_thread()?;
    let mut reader = Reader::new(BufReader::new(
        File::open(&args[0]).map_err(|e| e.to_string())?,
    ))?;
    let block = reader.block(block_number)?;
    let loops = target.div_ceil(block.totals.symbols as usize).max(1);
    let mut expected = Vec::new();
    block.book.decode_ranks_variant(
        Kernel::Reference,
        &block.payload,
        block.bits,
        block.totals.symbols,
        |r| {
            expected.push(r);
            Ok(())
        },
    )?;
    let mut output = vec![0u32; expected.len()];
    for kernel in Kernel::ALL {
        let mut at = 0;
        block.book.decode_ranks_variant(
            kernel,
            &block.payload,
            block.bits,
            block.totals.symbols,
            |r| {
                output[at] = r;
                at += 1;
                Ok(())
            },
        )?;
        if output != expected || at != expected.len() {
            return Err(format!("{} differs from reference", kernel.name()));
        }
    }
    if let Some(path) = fixture {
        use std::io::Write;
        let entries: Vec<_> = block.book.lengths().collect();
        let min = entries.iter().map(|&(_, n)| n).min().unwrap();
        let max = block.book.max_bits;
        let mut counts = vec![0u32; (max - min + 1) as usize];
        for &(_, len) in &entries {
            counts[(len - min) as usize] += 1;
        }
        let mut out = std::io::BufWriter::new(File::create_new(path).map_err(|e| e.to_string())?);
        out.write_all(b"HUFCORE1")
            .and_then(|_| out.write_all(&u32::from(min).to_le_bytes()))
            .and_then(|_| out.write_all(&u32::from(max).to_le_bytes()))
            .and_then(|_| out.write_all(&block.bits.to_le_bytes()))
            .and_then(|_| out.write_all(&block.totals.symbols.to_le_bytes()))
            .map_err(|e| e.to_string())?;
        for n in counts {
            out.write_all(&n.to_le_bytes()).map_err(|e| e.to_string())?;
        }
        out.write_all(&block.payload).map_err(|e| e.to_string())?;
        for rank in expected {
            out.write_all(&rank.to_le_bytes())
                .map_err(|e| e.to_string())?;
        }
        out.flush().map_err(|e| e.to_string())?;
        println!(
            "{{\"fixture_written\":true,\"symbols\":{}}}",
            block.totals.symbols
        );
        return Ok(());
    }
    let mut samples = Vec::new();
    let mut random = 91823u64;
    for round in 0..rounds {
        let mut order = Kernel::ALL;
        for i in (1..order.len()).rev() {
            random = random.wrapping_mul(6364136223846793005).wrapping_add(1);
            order.swap(i, (random % (i + 1) as u64) as usize);
        }
        for kernel in order {
            let cpu = thread_cpu_seconds();
            let start = Instant::now();
            for _ in 0..loops {
                let mut at = 0;
                block.book.decode_ranks_variant(
                    kernel,
                    &block.payload,
                    block.bits,
                    block.totals.symbols,
                    |r| {
                        output[at] = r;
                        at += 1;
                        Ok(())
                    },
                )?;
                std::hint::black_box(&output);
            }
            let elapsed = start.elapsed().as_secs_f64();
            let cpu = cpu.zip(thread_cpu_seconds()).map_or(0.0, |(a, b)| b - a);
            if output != expected {
                return Err(format!("{} output changed", kernel.name()));
            }
            samples.push((kernel, round, elapsed, cpu));
        }
    }
    println!("{{\"scope\":\"in-memory first/preselected block; canonical ranks only; fixed u32 output buffer; no IO, arena expansion or table construction\",\"pinned_cpu\":{},\"block\":{block_number},\"rounds\":{rounds},\"loops_per_trial\":{loops},\"block_symbols\":{},\"block_original_bytes\":{},\"codebook_symbols\":{},\"max_code_bits\":{},\"samples\":[",pinned_cpu.map_or("null".into(),|n|n.to_string()),block.totals.symbols,block.totals.bytes,block.book.symbols.len(),block.book.max_bits);
    for (i, (kernel, round, elapsed, cpu)) in samples.iter().enumerate() {
        println!("{}{{\"kernel\":\"{}\",\"round\":{round},\"seconds\":{elapsed:.9},\"thread_cpu_seconds\":{cpu:.9},\"million_symbols_per_second\":{:.3},\"equivalent_original_MB_per_second\":{:.3}}}",if i==0{""}else{","},kernel.name(),block.totals.symbols as f64*loops as f64/elapsed/1e6,block.totals.bytes as f64*loops as f64/elapsed/1e6);
    }
    println!("]}}");
    Ok(())
}
#[cfg(feature = "huffman-kernels")]
fn main() {
    if let Err(e) = run() {
        eprintln!("huffman-kernel-bench: {e}");
        std::process::exit(1);
    }
}
