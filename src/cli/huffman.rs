use bre_pair::{
    atomic_output::Output,
    huffman_archive::{self, Reader, DEFAULT_BLOCK_SYMBOLS},
    huffman_pipeline,
    tokenizer::{Model, Tokenizer},
};
use std::{
    fs,
    io::{BufReader, Read, Seek, SeekFrom},
    path::Path,
    time::Instant,
};

pub fn run(args: &[String]) -> Result<(), String> {
    let start = Instant::now();
    if args[0] == "huffman-repack" {
        if args.len() != 3 && args.len() != 5 {
            return Err(
                "huffman-repack INPUT.brhv OUTPUT.brhv [--rule-lengths auto|varint|runs]".into(),
            );
        }
        let coding_name = if args.len() == 5 {
            if args[3] != "--rule-lengths" {
                return Err("unknown Huffman repack option".into());
            }
            args[4].as_str()
        } else {
            "auto"
        };
        let report = if coding_name == "runs" {
            bre_pair::shared_huffman::repack_arity_runs(Path::new(&args[1]), Path::new(&args[2]))?
        } else {
            bre_pair::shared_huffman::repack_rule_lengths(
                Path::new(&args[1]),
                Path::new(&args[2]),
                coding_name.parse()?,
            )?
        };
        println!("{{\"format\":\"{}\",\"before_bytes\":{},\"archive_bytes\":{},\"before_descriptor_bytes\":{},\"descriptor_bytes\":{},\"rule_lengths\":\"{}\",\"payload_unchanged\":true,\"seconds\":{:.6}}}",report.after.format(),report.before.archive_bytes,report.after.archive_bytes,report.before.descriptor_bytes,report.after.descriptor_bytes,coding_name,start.elapsed().as_secs_f64());
        return Ok(());
    }
    if args[0] == "huffman-inspect" {
        if args.len() != 2 {
            return Err("huffman-inspect INPUT.brh".into());
        }
        let mut input = BufReader::new(fs::File::open(&args[1]).map_err(|e| e.to_string())?);
        let mut magic = [0; 8];
        input.read_exact(&mut magic).map_err(|e| e.to_string())?;
        input.seek(SeekFrom::Start(0)).map_err(|e| e.to_string())?;
        if bre_pair::shared_huffman::is_format(&magic) {
            let reader = bre_pair::shared_huffman::Reader::new(input)?;
            let s = &reader.stats;
            println!("{{\"format\":\"{}\",\"static_huffman\":true,\"static_table_bytes\":{},\"static_codes\":{},\"ai_tokens\":{},\"terminal_vocabulary\":{},\"codec_rules\":{},\"dictionary_bytes\":{},\"descriptor_bytes\":{},\"huffman_rule_lengths\":{},\"blocks\":{},\"original_bytes\":{},\"fields\":{},\"terminal_tokens\":{},\"codec_symbols\":{},\"default_decoder\":\"{}\",\"maximum_code_bits\":{},\"initialization_seconds\":{:.6}}}", s.format(),s.codebook_bytes, s.leaves, reader.ai_tokens, reader.terminal_vocabulary(), s.rules, s.dictionary_bytes,s.descriptor_bytes,s.huffman_rule_lengths,s.blocks, s.totals.bytes, s.totals.fields, s.totals.tokens, s.totals.symbols,reader.static_codebook().map_or("none",|book|book.decoder_backend()),reader.static_codebook().map_or(0,|book|book.max_bits),start.elapsed().as_secs_f64());
            return Ok(());
        }
        let reader = Reader::new(input)?;
        println!("{{\"format\":\"{}\",\"static_huffman\":{},\"static_table_bytes\":{},\"static_codes\":{},\"ai_tokens\":{},\"terminal_vocabulary\":{},\"codec_rules\":{},\"dictionary_bytes\":{},\"blocks\":{},\"original_bytes\":{},\"fields\":{},\"terminal_tokens\":{},\"codec_symbols\":{}}}",reader.format(),reader.format()=="BRHUFF02",reader.static_table_bytes(),reader.static_codebook().map_or(0,|b|b.symbols.len()),reader.dictionary.ai_tokens,reader.dictionary.model.tokens.len(),reader.dictionary.rules.len(),reader.dictionary_bytes(),reader.index.len(),reader.totals.bytes,reader.totals.fields,reader.totals.tokens,reader.totals.symbols);
        return Ok(());
    }
    let encode = args[0] == "huffman-encode";
    let pack = args[0] == "huffman-pack";
    let positional = if encode { 4 } else { 3 };
    if args.len() < positional {
        return Err("missing Huffman input or output; see --help".into());
    }
    let mut size = DEFAULT_BLOCK_SYMBOLS;
    let mut threads = 1usize;
    let mut decoder = None;
    let mut arena_mib = 64usize;
    for option in args[positional..].chunks(2) {
        let value = option.get(1).ok_or("missing Huffman option value")?;
        match option[0].as_str() {
            "--block-symbols" if encode || pack => {
                size = value.parse().map_err(|_| "invalid block symbol count")?
            }
            "--threads" if !(encode || pack) => {
                threads = value.parse().map_err(|_| "invalid Huffman thread count")?;
                if threads != 1 && threads != 4 {
                    return Err("Huffman decoding supports --threads 1 or 4".into());
                }
            }
            "--decoder" if !(encode || pack) => {
                if !matches!(value.as_str(), "auto" | "scalar" | "avx2") {
                    return Err("decoder must be auto, scalar or avx2".into());
                }
                decoder = Some(value.as_str());
            }
            "--arena-mib" if !(encode || pack) => {
                arena_mib = value.parse().map_err(|_| "invalid arena MiB")?;
                if arena_mib > 1024 {
                    return Err("arena MiB must be 0..=1024".into());
                }
            }
            _ => return Err("unknown Huffman option; see --help".into()),
        }
    }
    if let Some(decoder) = decoder {
        // This CLI process has not started decoder workers yet. Models capture
        // the override at construction; it never mutates a live codebook.
        std::env::set_var("BREPAIR_HUFFMAN_DECODER", decoder);
    }
    let output = Output::new(Path::new(&args[positional - 1]))?;
    let mut pipeline = None;
    let totals = if encode {
        let model = Model::from_bytes(&fs::read(&args[1]).map_err(|e| e.to_string())?)?;
        huffman_archive::encode_file(
            &Tokenizer::new(model)?,
            Path::new(&args[2]),
            output.path(),
            size,
        )?
    } else if pack {
        huffman_archive::pack_file(Path::new(&args[1]), output.path(), size)?
    } else if threads == 4 {
        let report = huffman_pipeline::decode_file(
            Path::new(&args[1]),
            output.path(),
            args[0] == "huffman-ids",
            huffman_pipeline::Options {
                arena_bytes: arena_mib << 20,
                ..Default::default()
            },
        )?;
        let totals = report.totals;
        pipeline = Some(report);
        totals
    } else {
        huffman_archive::decode_file_with_arena(
            Path::new(&args[1]),
            output.path(),
            args[0] == "huffman-ids",
            arena_mib << 20,
        )?
    };
    output.commit()?;
    let seconds = start.elapsed().as_secs_f64();
    let process = bre_pair::memory::process();
    let details = pipeline.map_or_else(String::new,|report|format!(",\"pipeline_seconds\":{:.6},\"pipeline_active_seconds\":[{:.6},{:.6},{:.6},{:.6}],\"pipeline_wait_seconds\":[{:.6},{:.6},{:.6},{:.6}]",report.seconds,report.stages[0].active_seconds,report.stages[1].active_seconds,report.stages[2].active_seconds,report.stages[3].active_seconds,report.stages[0].wait_seconds,report.stages[1].wait_seconds,report.stages[2].wait_seconds,report.stages[3].wait_seconds));
    println!("{{\"original_bytes\":{},\"fields\":{},\"terminal_tokens\":{},\"codec_symbols\":{},\"seconds\":{:.6},\"original_MB_per_second\":{:.3},\"threads\":{},\"peak_heap_bytes\":{},\"peak_resident_bytes\":{},\"peak_private_bytes\":{}{details}}}",totals.bytes,totals.fields,totals.tokens,totals.symbols,seconds,totals.bytes as f64/seconds/1e6,threads,bre_pair::memory::peak_heap_bytes(),process.peak_resident,process.peak_private);
    Ok(())
}
