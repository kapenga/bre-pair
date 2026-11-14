//! Public command-line entry. Research executables stay separate.
#[path = "../cli/codec.rs"]
mod codec_cli;
#[path = "../cli/compress.rs"]
mod compress_cli;
#[path = "../cli/huffman.rs"]
mod huffman_cli;
#[path = "../cli/tokenizer.rs"]
mod tokenizer_cli;
use bre_pair::{
    archive_io, atomic_output::Output, engine::Config, field_engine, memory, scoring::Model,
};
use std::{
    fs,
    io::{BufReader, BufWriter, Read, Write},
    path::Path,
    process::ExitCode,
    time::Instant,
};

#[global_allocator]
static ALLOCATOR: memory::Allocator = memory::Allocator;
const HELP: &str = "BRe-Pair: bounded-memory byte tokenizer training\n\n\
brepair compress INPUT OUTPUT.brh [--min-count 2] [--threads 4]\n\
  [--count-floors 8,4,2] (explicit progressive schedule, replaces --min-count)\n\
  [--cleanup-rounds 2] (1..64; always retain the smallest actual checkpoint)\n\
  [--cleanup-arity-cost varint|huffman|both] (estimate only; default varint; both compares complete archives)\n\
  [--rule-lengths auto|varint] (auto uses Huffman metadata when smaller)\n\
  [--fill-waves 2] (compatible selection waves per rewrite; 1..8)\n\
  [--reparse-refits 0] (optional minimum-bit reparse after discovery; 0..16)\n\
  [--max-reparse-field-bytes 134217728] (explicit field bound; large-field choices use scratch disk)\n\
  [--score weighted-pmi|frequency] [--fields lines|single] [--report FILE.json]\n\
  [--keep-work NEW_DIRECTORY] (retain discovery and cleanup checkpoints)\n\
  Maximum-compression profile: two waves and mandatory Huffman cleanup.\n\
brepair train INPUT CANDIDATES.brp [--memory 200MB] [--threads 4]\n\
  [--min-count 8] [--score weighted-pmi|frequency] [--max-rules N]\n\
  [--stop-max-pair-count N] (stop when every eligible pair has count <=N)\n\
  [--frequency-override-percent 0] [--batch-priority score] [--fill-waves 1]\n\
  [--fill-min-count N] [--fill-min-count-per-gb N] [--fill-after-rounds N]\n\
  [--verify] [--report FILE.json] [--rounds FILE.csv]\n\
brepair vocab CANDIDATES.brp TRAIN MODEL.brt [--vocab-size 8192] [--threads 4]\n\
  [--selection statistics|sampled] (default: statistics)\n\
  [--min-token-uses 8] [--memory 200MB] [--report FILE.json]\n\
  [--document-index TRAIN.docs --min-token-documents 64]\n\
  [--candidate-document-support stream|grammar] (default: stream)\n\
brepair refine CANDIDATES.brp TRAIN INITIAL.brt OUTPUT.brt [--threads 4]\n\
  [--document-index TRAIN.docs --min-token-documents 8] [--min-token-uses 8]\n\
  [--memory SIZE] [--report FILE.json] [--counts FILE.json]\n\
brepair evaluate MODEL.brt INPUT [--threads 4] [--overlap-bytes N] [--verify]\n\
brepair encode MODEL.brt INPUT OUTPUT.bts\n\
brepair decode MODEL.brt INPUT.bts OUTPUT\n\
brepair verify MODEL.brt INPUT.bts ORIGINAL\n\
brepair ids MODEL.brt INPUT OUTPUT.ids\n\
brepair decompress CANDIDATES.brp OUTPUT\n\
brepair inspect MODEL.brt\n\n\
brepair columns split INPUT NEW_DIRECTORY COLUMNS (unquoted PSV)\n\
brepair columns restore COLUMN_DIRECTORY NEW_OUTPUT\n\n\
brepair huffman-pack INPUT.brp OUTPUT.brh [--block-symbols 1048576]\n\
brepair huffman-encode MODEL.brt INPUT OUTPUT.brh [--block-symbols 1048576]\n\
brepair huffman-decode INPUT.brh OUTPUT [--threads 1|4] [--decoder auto|scalar|avx2] [--arena-mib 64]\n\
brepair huffman-ids INPUT.brh OUTPUT.ids [--threads 1|4]\n\
brepair huffman-inspect INPUT.brh\n\n\
brepair huffman-repack INPUT.brhv OUTPUT.brhv [--rule-lengths auto|varint|runs]\n\n\
brepair codec-train MODEL.brt TRAIN OUTPUT.brc [--min-count 8] [--threads 4]\n\
  [--score weighted-pmi|frequency] [--max-rules N] [--max-rounds N] [--report FILE.json]\n\
  [--min-count-per-gb N] [--document-index TRAIN.docs --min-rule-documents N]\n\
  [--document-counting exact|last-seen|boundary]\n\
  [--fill-waves 2 --fill-min-count 32 --fill-min-count-per-gb 1280]\n\
brepair codec-encode MODEL.brc INPUT OUTPUT.brh [--threads 1] [--block-symbols 1048576]\n\
  [--mode advanced|greedy] (default: advanced)\n\
brepair codec-inspect MODEL.brc\n\n\
brepair codec-optimize MODEL.brc TRAINING.brh OUTPUT.brc [--max-trials 64]\n\
  [--recode TRAINING-OPTIMIZED.brh] [--report FILE.json]\n\n\
Every command accepts --memory SIZE. MB/GB are decimal; MiB/GiB are binary.\n\
Default: twice the input/training-file size, at least 64MB. The limit includes\n\
headroom for runtime memory. Windows also enforces a private-commit cap.\n\
Input fields are raw lines retaining newline bytes, each followed by mergeable\n\
EOF. Bytes are never normalized. Extract JSONL text before training.\n\
Output files must be new. Completed outputs are published atomically.\n\
--verbose shows per-batch training and detailed vocabulary-selection progress.\n\
--version prints the executable version.\n";

struct Compare {
    source: BufReader<fs::File>,
    bytes: u64,
}
impl Write for Compare {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let mut expected = [0; 65536];
        for chunk in bytes.chunks(expected.len()) {
            self.source.read_exact(&mut expected[..chunk.len()])?;
            if chunk != &expected[..chunk.len()] {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "archive differs from source",
                ));
            }
        }
        self.bytes += bytes.len() as u64;
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn scaled_fill_minimum(absolute: u64, per_gb: u64, bytes: u64) -> Result<u64, String> {
    bre_pair::search::scaled_fill_minimum(absolute, per_gb, bytes)
}

fn train(args: &[String], limit: &memory::Limit, verbose: bool) -> Result<(), String> {
    if args.len() < 3 {
        return Err(HELP.into());
    }
    let input = Path::new(&args[1]);
    let destination = Path::new(&args[2]);
    let mut workers = 4usize;
    let mut minimum = 8u64;
    let mut score = Model::WeightedPmi;
    let mut maximum = None;
    let mut max_rounds = None;
    let mut stop_max_pair_count = None;
    let mut report = None;
    let mut verify = false;
    let mut frequency_override_percent = 0u8;
    let mut batch_priority = bre_pair::search::BatchPriority::Score;
    let mut fill_waves = 1u8;
    let mut fill_min_count = 0u64;
    let mut fill_min_count_per_gb = 0u64;
    let mut fill_after_rounds = 0usize;
    let mut rounds = None;
    let mut at = 3;
    while at < args.len() {
        if args[at] == "--verify" {
            verify = true;
            at += 1;
            continue;
        }
        let value = args.get(at + 1).ok_or("missing option value")?;
        match args[at].as_str() {
            "--threads" => workers = value.parse().map_err(|_| "invalid worker count")?,
            "--min-count" => minimum = value.parse().map_err(|_| "invalid minimum")?,
            "--score" => {
                score = value.parse()?;
                if !matches!(score, Model::Frequency | Model::WeightedPmi) {
                    return Err("use frequency or weighted-pmi".into());
                }
            }
            "--max-rules" => maximum = Some(value.parse().map_err(|_| "invalid rule limit")?),
            "--max-rounds" => max_rounds = Some(value.parse().map_err(|_| "invalid round limit")?),
            "--stop-max-pair-count" => {
                stop_max_pair_count = Some(
                    value
                        .parse::<u64>()
                        .map_err(|_| "invalid maximum remaining pair count")?,
                )
            }
            "--frequency-override-percent" => {
                frequency_override_percent =
                    value.parse().map_err(|_| "invalid override percentage")?;
                if frequency_override_percent > 100 {
                    return Err("frequency override percentage must be 0..=100".into());
                }
            }
            "--report" => report = Some(value.as_str()),
            "--rounds" => rounds = Some(value.as_str()),
            "--batch-priority" => batch_priority = value.parse()?,
            "--fill-waves" => fill_waves = value.parse().map_err(|_| "invalid fill-wave count")?,
            "--fill-min-count" => {
                fill_min_count = value.parse().map_err(|_| "invalid fill count floor")?
            }
            "--fill-after-rounds" => {
                fill_after_rounds = value.parse().map_err(|_| "invalid fill warmup")?
            }
            "--fill-min-count-per-gb" => {
                fill_min_count_per_gb =
                    value.parse().map_err(|_| "invalid normalized fill count")?
            }
            _ => return Err(format!("unknown train option {}", args[at])),
        }
        at += 2;
    }
    if minimum < 2 || maximum == Some(0) || max_rounds == Some(0) {
        return Err("minimum count must be >=2 and rule/batch limits must be positive".into());
    }
    if stop_max_pair_count.is_some_and(|n| n < minimum) {
        return Err("stop maximum pair count must be at least the minimum count".into());
    }
    if !(1..=8).contains(&fill_waves) {
        return Err("fill waves must be 1..=8".into());
    }
    if frequency_override_percent != 0
        && (batch_priority != bre_pair::search::BatchPriority::Score || fill_waves != 1)
    {
        return Err(
            "local frequency override cannot be combined with batch priorities or fill waves"
                .into(),
        );
    }
    fill_min_count = scaled_fill_minimum(
        fill_min_count,
        fill_min_count_per_gb,
        fs::metadata(input).map_err(|e| e.to_string())?.len(),
    )?;
    let output = Output::new(destination)?;
    let report_output = report.map(|p| Output::new(Path::new(p))).transpose()?;
    let rounds_output = rounds.map(|p| Output::new(Path::new(p))).transpose()?;
    if report_output
        .as_ref()
        .is_some_and(|p| p.destination() == output.destination())
    {
        return Err("report must differ from input/output".into());
    }
    if rounds_output.as_ref().is_some_and(|p| {
        p.destination() == output.destination()
            || report_output
                .as_ref()
                .is_some_and(|r| p.destination() == r.destination())
    }) {
        return Err("rounds must differ from archive/report".into());
    }
    let started = Instant::now();
    let _progress = bre_pair::training_progress::enable();
    let mut rules_created = 0usize;
    eprintln!(
        "training {} with a {} byte process-memory budget",
        input.display(),
        limit.total
    );
    let config = Config {
        buffer_bytes: limit.heap,
        minimum_count: minimum,
        score,
        search: bre_pair::search::Options {
            certify: false,
            frequency_override_percent,
            batch_priority,
            fill_waves,
            fill_min_count,
            fill_after_rounds,
            ..Default::default()
        },
        ..Default::default()
    };
    let result = field_engine::compress_file(
        input,
        &config,
        workers,
        1048576,
        field_engine::Options {
            max_rounds,
            max_rules: maximum,
            stop_max_pair_count,
            frequency_cutoff: true,
            memory_saving: true,
            ..Default::default()
        },
        |round| {
            rules_created += round.new_rules;
            bre_pair::training_progress::completed(round, rules_created);
            eprintln!("codec progress {{\"event\":\"round\",\"round\":{},\"symbols\":{},\"grammar_rules\":{},\"new_rules\":{},\"rules_created\":{},\"packed_bytes\":{},\"symbol_bits\":{},\"selected_pairs\":{},\"minimum_selected_count\":{},\"maximum_selected_count\":{},\"removed_symbols\":{},\"removed_percent\":{:.6},\"seconds\":{:.3}}}",round.index,round.after_refill,round.total_rules,round.new_rules,rules_created,round.packed_bytes,round.width,round.selected,round.minimum_selected_count,round.maximum_selected_count,round.removed,100.0*round.removed as f64/round.before.max(1) as f64,started.elapsed().as_secs_f64());
            if verbose || round.index % 10 == 0 {
                eprintln!(
                    "batch {}: {} rules, {} symbols; selected {}; elapsed {:.3}s; selection {:.3}s; rewrite/count {:.3}s",
                    round.index, round.total_rules, round.rewritten, round.selected,
                    started.elapsed().as_secs_f64(), round.selection_seconds, round.rewrite_seconds
                );
            }
        },
    )?;
    let training_seconds = started.elapsed().as_secs_f64();
    if verify {
        bre_pair::vocabulary::verify_training_source(&result.training.compressed, input)?;
    }
    {
        let file = fs::File::create_new(output.path()).map_err(|e| e.to_string())?;
        let mut writer = BufWriter::with_capacity(65536, file);
        archive_io::write(&result.training.compressed, &mut writer)?;
    }
    if verify {
        let mut source = BufReader::new(fs::File::open(output.path()).map_err(|e| e.to_string())?);
        let mut compare = Compare {
            source: BufReader::new(fs::File::open(input).map_err(|e| e.to_string())?),
            bytes: 0,
        };
        archive_io::decompress(&mut source, &mut compare)?;
        let mut tail = [0];
        if compare.source.read(&mut tail).map_err(|e| e.to_string())? != 0 {
            return Err("source has unencoded trailing bytes".into());
        }
    }
    let process = memory::process();
    let row=format!(concat!("{{\"version\":\"{}\",\"input_bytes\":{},\"fields\":{},\"rules\":{},\"symbols\":{},\"score\":\"{}\",\"minimum_count\":{},\"threads\":{},\"frequency_override_percent\":{},\"batches\":{},\"batch_priority\":\"{}\",\"fill_waves\":{},\"fill_min_count\":{},\"fill_min_count_per_gb\":{},\"fill_after_rounds\":{},\"selection_seconds\":{:.6},\"rewrite_seconds\":{:.6},",
        "\"memory_limit_bytes\":{},\"heap_limit_bytes\":{},\"peak_heap_bytes\":{},\"peak_working_set_bytes\":{},\"peak_private_commit_bytes\":{},",
        "\"registry_peak_slot_bytes\":{},\"retired_pairs\":{},\"registry_compactions\":{},\"training_seconds\":{:.6},\"total_seconds\":{:.6},\"stopped_at_rule_limit\":{},\"stopped_at_round_limit\":{},\"stop_max_pair_count\":{},\"stopped_at_pair_count\":{},\"maximum_pair_count_at_stop\":{},\"source_and_archive_verified\":{}}}\n"),
        env!("CARGO_PKG_VERSION"),result.training.compressed.input_bytes,result.training.compressed.field_count,result.training.compressed.grammar.rules.len(),result.training.compressed.token_count(),score.name(),minimum,workers,frequency_override_percent,result.training.rounds.len(),batch_priority.name(),fill_waves,fill_min_count,fill_min_count_per_gb,fill_after_rounds,
        result.training.rounds.iter().map(|r| r.selection_seconds).sum::<f64>(),result.training.rounds.iter().map(|r| r.rewrite_seconds).sum::<f64>(),
        limit.total,limit.heap,memory::peak_heap_bytes(),process.peak_resident,process.peak_private,
        result.registry_peak_slot_bytes,result.registry_retired_pairs,result.registry_compactions,training_seconds,started.elapsed().as_secs_f64(),result.stopped_at_rule_limit,result.stopped_at_round_limit,
        stop_max_pair_count.map_or("null".into(),|n|n.to_string()),result.stopped_at_pair_count,result.maximum_pair_count_at_stop.map_or("null".into(),|n|n.to_string()),verify);
    output.commit()?;
    if let Some(output) = rounds_output {
        {
            let file = fs::File::create_new(output.path()).map_err(|e| e.to_string())?;
            let mut writer = BufWriter::new(file);
            writeln!(writer, "batch,before,after,candidates,eligible,selected,new_rules,total_rules,selection_seconds,rewrite_seconds").map_err(|e| e.to_string())?;
            for r in &result.training.rounds {
                writeln!(
                    writer,
                    "{},{},{},{},{},{},{},{},{:.6},{:.6}",
                    r.index,
                    r.before,
                    r.rewritten,
                    r.candidates,
                    r.eligible,
                    r.selected,
                    r.new_rules,
                    r.total_rules,
                    r.selection_seconds,
                    r.rewrite_seconds
                )
                .map_err(|e| e.to_string())?;
            }
            writer.flush().map_err(|e| e.to_string())?;
        }
        output.commit()?;
    }
    if let Some(output) = report_output {
        fs::write(output.path(), row.as_bytes()).map_err(|e| e.to_string())?;
        output.commit()?;
    }
    print!("{row}");
    Ok(())
}

fn run() -> Result<(), String> {
    let mut args: Vec<_> = std::env::args().skip(1).collect();
    if args.is_empty()
        || args
            .iter()
            .any(|arg| matches!(arg.as_str(), "--help" | "-h"))
        || args[0] == "help"
    {
        print!("{HELP}");
        return Ok(());
    }
    if args[0] == "--version" {
        println!("brepair {}", env!("CARGO_PKG_VERSION"));
        return Ok(());
    }
    if args[0] == "columns" {
        return bre_pair::columns::run_cli(&args[1..]);
    }
    let mut requested = None;
    let mut verbose = false;
    let mut at = 0;
    while at < args.len() {
        if args[at] == "--verbose" {
            verbose = true;
            args.remove(at);
        } else if args[at] == "--memory" {
            requested = Some(memory::parse_bytes(
                args.get(at + 1).ok_or("missing --memory value")?,
            )?);
            args.drain(at..at + 2);
        } else {
            at += 1;
        }
    }
    let workers = args
        .windows(2)
        .find(|p| p[0] == "--threads")
        .and_then(|p| p[1].parse::<usize>().ok())
        .unwrap_or(4);
    if workers == 0 || workers > 256 {
        return Err("threads must be 1..=256".into());
    }
    let source_index = match args[0].as_str() {
        "compress" | "train" | "decompress" | "inspect" | "huffman-pack" | "huffman-decode"
        | "huffman-ids" | "huffman-inspect" | "huffman-repack" | "codec-inspect" => 1,
        "vocab" | "refine" | "evaluate" | "encode" | "decode" | "ids" | "huffman-encode" => 2,
        "codec-train" | "codec-encode" | "codec-optimize" => 2,
        "verify" => 3,
        _ => return Err(HELP.into()),
    };
    let source = args.get(source_index).ok_or(HELP)?;
    let default = usize::try_from(fs::metadata(source).map_err(|e| e.to_string())?.len())
        .ok()
        .and_then(|n| n.checked_mul(2))
        .ok_or("input size overflow")?
        .max(64000000);
    let limit = memory::Limit::set(requested.unwrap_or(default), workers)?;
    match args[0].as_str() {
        "compress" => compress_cli::run(&args, &limit, verbose),
        command if command.starts_with("codec-") => codec_cli::run(&args, &limit, verbose),
        command if command.starts_with("huffman-") => huffman_cli::run(&args),
        "train" => train(&args, &limit, verbose),
        "decompress" => {
            if args.len() != 3 {
                return Err("brepair decompress INPUT.brp OUTPUT [--memory SIZE]".into());
            }
            let output = Output::new(Path::new(&args[2]))?;
            let mut input = BufReader::with_capacity(
                65536,
                fs::File::open(&args[1]).map_err(|e| e.to_string())?,
            );
            let mut writer = BufWriter::with_capacity(
                65536,
                fs::File::create_new(output.path()).map_err(|e| e.to_string())?,
            );
            let result = archive_io::decompress(&mut input, &mut writer)?;
            drop(writer);
            output.commit()?;
            println!("decoded {} bytes, {} fields", result.bytes, result.fields);
            Ok(())
        }
        "inspect" => {
            if args.len() != 2 {
                return Err("brepair inspect MODEL.brt".into());
            }
            let model = bre_pair::tokenizer::Model::from_bytes(
                &fs::read(&args[1]).map_err(|e| e.to_string())?,
            )?;
            println!(
                "{{\"tokens\":{},\"learned_tokens\":{},\"bits_per_token\":{}}}",
                model.tokens.len(),
                model.tokens.len() - 257,
                bre_pair::packed::symbol_width(model.tokens.len())
            );
            Ok(())
        }
        _ => {
            if args[0] == "vocab" {
                args[0] = "build".into();
            }
            let output_index = match args[0].as_str() {
                "refine" => Some(4),
                "build" | "encode" | "decode" | "ids" => Some(3),
                _ => None,
            };
            let output = output_index
                .map(|i| {
                    args.get(i)
                        .ok_or_else(|| "missing output".to_string())
                        .and_then(|path| Output::new(Path::new(path)))
                })
                .transpose()?;
            let auxiliary_indices: Vec<_> = args
                .windows(2)
                .enumerate()
                .filter(|(_, pair)| matches!(pair[0].as_str(), "--report" | "--counts"))
                .map(|(i, _)| i + 1)
                .collect();
            let mut auxiliary: Vec<Output> = Vec::new();
            for index in auxiliary_indices {
                let file = Output::new(Path::new(&args[index]))?;
                if output
                    .as_ref()
                    .is_some_and(|p| p.destination() == file.destination())
                    || auxiliary
                        .iter()
                        .any(|p| p.destination() == file.destination())
                {
                    return Err("output, report and count files must be different".into());
                }
                args[index] = file.path().to_str().ok_or("output is not UTF-8")?.into();
                auxiliary.push(file);
            }
            if let (Some(index), Some(output)) = (output_index, &output) {
                args[index] = output.path().to_str().ok_or("output is not UTF-8")?.into();
            }
            tokenizer_cli::run(args, verbose)?;
            if let Some(output) = output {
                output.commit()?;
            }
            for file in auxiliary {
                file.commit()?;
            }
            Ok(())
        }
    }
}
fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("brepair: {e}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::scaled_fill_minimum;
    #[test]
    fn fill_floor_scales_with_bytes_without_rounding_large_counts() {
        assert_eq!(scaled_fill_minimum(0, 1280, 100_000_000).unwrap(), 128);
        assert_eq!(
            scaled_fill_minimum(0, 1280, 1_000_000_000_000).unwrap(),
            1_280_000
        );
        assert_eq!(scaled_fill_minimum(32, 1, 1).unwrap(), 32);
        assert_eq!(scaled_fill_minimum(0, 1, 1).unwrap(), 1);
        assert_eq!(
            scaled_fill_minimum(0, 1_000_000_000, (1 << 53) + 1).unwrap(),
            (1 << 53) + 1
        );
        assert_eq!(scaled_fill_minimum(u64::MAX, 0, 0).unwrap(), u64::MAX);
        assert!(scaled_fill_minimum(0, u64::MAX, u64::MAX).is_err());
    }
}
