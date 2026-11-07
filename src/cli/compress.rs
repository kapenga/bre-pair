//! Complete compression profile: minimum-two discovery always followed by
//! shared Huffman cleanup and selection by actual complete archive size.
use bre_pair::{
    archive_io, atomic_output::Output, engine::Config, field_engine, grammar_reparse,
    huffman_archive, memory, scoring::Model, shared_huffman,
};
use std::{
    fs,
    io::{BufReader, BufWriter, Write},
    path::{Path, PathBuf},
    time::Instant,
};

const USAGE: &str = "brepair compress INPUT OUTPUT.brh [--min-count 2 | --count-floors 8,4,2] [--cleanup-rounds 2] [--cleanup-arity-cost varint|huffman|both] [--rule-lengths auto|varint] [--fill-waves 2] [--reparse-refits 0 --max-reparse-field-bytes 134217728] [--score weighted-pmi|frequency] [--threads 4] [--fields lines|single] [--report FILE.json] [--keep-work NEW_DIRECTORY] [--memory SIZE]";

#[derive(Clone, Copy, PartialEq, Eq)]
enum ArityCost {
    Varint,
    Huffman,
    Both,
}
impl ArityCost {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "varint" => Ok(Self::Varint),
            "huffman" => Ok(Self::Huffman),
            "both" => Ok(Self::Both),
            _ => Err("cleanup arity cost must be varint, huffman or both".into()),
        }
    }
    fn name(self) -> &'static str {
        match self {
            Self::Varint => "varint",
            Self::Huffman => "huffman",
            Self::Both => "both",
        }
    }
}

fn count_floors(value: &str) -> Result<Vec<u64>, String> {
    let floors = value
        .split(',')
        .map(|s| {
            s.parse::<u64>()
                .map_err(|_| "invalid count floor".to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    if floors.iter().any(|&f| f < 2) || floors.windows(2).any(|w| w[0] <= w[1]) {
        return Err("count floors must strictly decrease and be at least two".into());
    }
    Ok(floors)
}

struct Stage {
    kind: &'static str,
    refit: usize,
    floor: u64,
    rules: usize,
    rounds: usize,
    seconds: f64,
    reparse_seconds: f64,
    reparse_bits: Option<(u128, u128)>,
    bytes: u64,
    fields: u64,
    archive_bytes: u64,
    cleanup: shared_huffman::Report,
    candidates: PathBuf,
}

fn write_packed(compressed: &bre_pair::engine::Compressed, path: &Path) -> Result<(), String> {
    let mut writer = BufWriter::new(fs::File::create_new(path).map_err(|e| e.to_string())?);
    archive_io::write(compressed, &mut writer)?;
    writer.flush().map_err(|e| e.to_string())
}

fn pack_and_clean(
    candidates: &Path,
    directory: &Path,
    options: shared_huffman::Options,
    arity_cost: ArityCost,
) -> Result<(shared_huffman::Report, PathBuf, u64), String> {
    let static_file = directory.join("static.brh");
    huffman_archive::pack_file(
        candidates,
        &static_file,
        huffman_archive::DEFAULT_BLOCK_SYMBOLS,
    )?;
    let cleanup_directory = directory.join("cleanup");
    let mut first = options;
    first.huffman_arity_cost = arity_cost == ArityCost::Huffman;
    let mut cleanup = shared_huffman::optimize(&static_file, &cleanup_directory, first)?;
    if arity_cost == ArityCost::Both {
        // Sequential optimizers release their full State before returning, so
        // the second estimate does not double the live dictionary workspace.
        let mut alternate = options;
        alternate.huffman_arity_cost = true;
        let other = shared_huffman::optimize(
            &static_file,
            &cleanup_directory.join("arity-huffman"),
            alternate,
        )?;
        let best_bytes = |r: &shared_huffman::Report| {
            r.variants
                .iter()
                .find(|v| v.name == r.selected)
                .map_or(r.original_bytes, |v| v.stats.archive_bytes)
        };
        if best_bytes(&other) < best_bytes(&cleanup) {
            cleanup.selected = format!("arity-huffman/{}", other.selected);
        }
        for mut variant in other.variants {
            variant.name = format!("arity-huffman/{}", variant.name);
            cleanup.variants.push(variant);
        }
    }
    let selected = if cleanup.selected == "original" {
        static_file
    } else {
        directory
            .join("cleanup")
            .join(format!("{}.brhv", cleanup.selected))
    };
    let bytes = fs::metadata(&selected).map_err(|e| e.to_string())?.len();
    Ok((cleanup, selected, bytes))
}

struct Workspace(PathBuf, bool);
impl Workspace {
    fn new(output: &Output) -> Result<Self, String> {
        // Output resolves the parent to an absolute directory. create_dir is
        // exclusive: Drop removes only this newly owned scratch directory.
        let path = output.path().with_extension("work");
        fs::create_dir(&path).map_err(|e| e.to_string())?;
        Ok(Self(path, false))
    }
    fn retained(reservation: &Output) -> Result<Self, String> {
        let path = reservation.destination().to_path_buf();
        fs::create_dir(&path).map_err(|e| e.to_string())?;
        Ok(Self(path, true))
    }
}
impl Drop for Workspace {
    fn drop(&mut self) {
        if !self.1 {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
}

pub fn run(args: &[String], limit: &memory::Limit, verbose: bool) -> Result<(), String> {
    if args.len() < 3 {
        return Err(USAGE.into());
    }
    let mut minimum = 2u64;
    let mut explicit_minimum = false;
    let mut floors = None;
    let mut cleanup_rounds = 2usize;
    let mut rule_lengths = shared_huffman::RuleLengths::Auto;
    let mut arity_cost = ArityCost::Varint;
    let mut fill_waves = 2u8;
    let mut reparse_refits = 0usize;
    let mut maximum_field = 134217728usize;
    let mut workers = 4usize;
    let mut score = Model::WeightedPmi;
    let mut single = false;
    let mut report_path = None;
    let mut keep_work = None;
    for pair in args[3..].chunks(2) {
        let value = pair.get(1).ok_or("missing compression option value")?;
        match pair[0].as_str() {
            "--min-count" => {
                minimum = value.parse().map_err(|_| "invalid minimum")?;
                explicit_minimum = true;
            }
            "--count-floors" => floors = Some(count_floors(value)?),
            "--cleanup-rounds" => {
                cleanup_rounds = value.parse().map_err(|_| "invalid cleanup rounds")?
            }
            "--fill-waves" => fill_waves = value.parse().map_err(|_| "invalid fill waves")?,
            "--rule-lengths" => rule_lengths = value.parse()?,
            "--cleanup-arity-cost" => arity_cost = ArityCost::parse(value)?,
            "--reparse-refits" => reparse_refits = value.parse().map_err(|_| "invalid refits")?,
            "--max-reparse-field-bytes" => {
                maximum_field = value.parse().map_err(|_| "invalid reparse field bound")?
            }
            "--threads" => workers = value.parse().map_err(|_| "invalid worker count")?,
            "--score" => {
                score = value.parse()?;
                if !matches!(score, Model::Frequency | Model::WeightedPmi) {
                    return Err("use frequency or weighted-pmi".into());
                }
            }
            "--fields" => match value.as_str() {
                "lines" => single = false,
                "single" => single = true,
                _ => return Err("compression fields must be lines or single".into()),
            },
            "--report" => report_path = Some(Path::new(value)),
            "--keep-work" => keep_work = Some(Path::new(value)),
            _ => return Err(format!("unknown compression option {}; {USAGE}", pair[0])),
        }
    }
    if minimum < 2 || !(1..=256).contains(&workers) {
        return Err("minimum count must be >=2 and threads must be 1..=256".into());
    }
    if cleanup_rounds == 0 || cleanup_rounds > 64 {
        return Err("cleanup rounds must be 1..=64".into());
    }
    if arity_cost != ArityCost::Varint && rule_lengths != shared_huffman::RuleLengths::Auto {
        return Err("Huffman arity costs require --rule-lengths auto".into());
    }
    if !(1..=8).contains(&fill_waves) {
        return Err("fill waves must be 1..=8".into());
    }
    if reparse_refits > 16 || maximum_field == 0 || maximum_field == usize::MAX {
        return Err("reparse refits must be 0..=16 with a positive bounded field size".into());
    }
    if reparse_refits != 0
        && single
        && fs::metadata(&args[1]).map_err(|e| e.to_string())?.len() > maximum_field as u64
    {
        return Err("single input exceeds --max-reparse-field-bytes; increase this explicit bound and the memory budget".into());
    }
    if explicit_minimum && floors.is_some() {
        return Err("use either --min-count or --count-floors".into());
    }
    let floors = floors.unwrap_or_else(|| vec![minimum]);
    minimum = *floors.last().unwrap();
    let output = Output::new(Path::new(&args[2]))?;
    let report_output = report_path.map(Output::new).transpose()?;
    if report_output
        .as_ref()
        .is_some_and(|r| r.destination() == output.destination())
    {
        return Err("compression report must differ from archive".into());
    }
    let work = if let Some(path) = keep_work {
        let reservation = Output::new(path)?;
        if reservation.destination() == output.destination()
            || report_output
                .as_ref()
                .is_some_and(|r| r.destination() == reservation.destination())
        {
            return Err("retained workspace must differ from archive/report".into());
        }
        Workspace::retained(&reservation)?
    } else {
        Workspace::new(&output)?
    };
    let started = Instant::now();
    let _progress = bre_pair::training_progress::enable();
    let cleanup_options = shared_huffman::Options {
        rounds: cleanup_rounds,
        rule_lengths,
        ..Default::default()
    };
    let mut stages = Vec::<Stage>::new();
    let mut previous = None;
    let mut winner = 0usize;
    for (index, &floor) in floors.iter().enumerate() {
        let directory = if index == 0 {
            work.0.clone()
        } else {
            let path = work.0.join(format!("stage-{index}-min{floor}"));
            fs::create_dir(&path).map_err(|e| e.to_string())?;
            path
        };
        let config = Config {
            minimum_count: floor,
            buffer_bytes: limit.heap,
            score,
            search: bre_pair::search::Options {
                certify: false,
                fill_waves,
                fill_min_count: floor,
                ..Default::default()
            },
            ..Default::default()
        };
        let options = field_engine::Options {
            frequency_cutoff: true,
            memory_saving: true,
            ..Default::default()
        };
        let mut created = 0usize;
        let mut progress = |round: &bre_pair::engine::Round| {
            created += round.new_rules;
            bre_pair::training_progress::completed(round, created);
            eprintln!(
                concat!(
                    "codec progress {{\"event\":\"round\",\"round\":{},\"count_floor\":{},",
                    "\"grammar_rules\":{},\"symbols\":{},\"packed_bytes\":{},",
                    "\"selected_pairs\":{},\"minimum_selected_count\":{},",
                    "\"maximum_selected_count\":{},\"seconds\":{:.3}}}"
                ),
                round.index,
                floor,
                round.total_rules,
                round.after_refill,
                round.packed_bytes,
                round.selected,
                round.minimum_selected_count,
                round.maximum_selected_count,
                started.elapsed().as_secs_f64()
            );
            if verbose || round.index % 10 == 0 {
                eprintln!(
                    "floor {floor}, batch {}: {} rules, {} symbols; selected {}; {:.3}s",
                    round.index,
                    round.total_rules,
                    round.rewritten,
                    round.selected,
                    started.elapsed().as_secs_f64()
                );
            }
        };
        eprintln!(
            "discovering pairs; stage {}; minimum {floor}; mandatory Huffman cleanup",
            index + 1
        );
        let result = if let Some(source) = &previous {
            let packed = archive_io::read_packed(
                &mut BufReader::new(fs::File::open(source).map_err(|e| e.to_string())?),
                1048576,
            )?;
            field_engine::compress_prepacked(
                packed,
                &config,
                workers,
                1048576,
                options,
                &mut progress,
            )?
        } else if single {
            field_engine::compress_file_single_field(
                Path::new(&args[1]),
                &config,
                workers,
                1048576,
                options,
                &mut progress,
            )?
        } else {
            field_engine::compress_file(
                Path::new(&args[1]),
                &config,
                workers,
                1048576,
                options,
                &mut progress,
            )?
        };
        let bytes = result.training.compressed.input_bytes;
        let fields = result.training.compressed.field_count;
        let rules = result.training.compressed.grammar.rules.len();
        let rounds = result.training.rounds.len();
        let seconds = result.training.seconds;
        let candidates = directory.join("discovery.brp");
        write_packed(&result.training.compressed, &candidates)?;
        // Do not retain the packed corpus/registry while optimizing Huffman.
        // Continuation reloads the unchanged grammar, never the pruned model.
        drop(result);
        let (cleanup, selected, archive_bytes) =
            pack_and_clean(&candidates, &directory, cleanup_options, arity_cost)?;
        if stages.is_empty() || archive_bytes < stages[winner].archive_bytes {
            fs::copy(&selected, output.path()).map_err(|e| e.to_string())?;
            winner = index;
        }
        eprintln!(
            "floor {floor} complete: {archive_bytes} archive bytes; best {}",
            if winner == index {
                archive_bytes
            } else {
                stages[winner].archive_bytes
            }
        );
        stages.push(Stage {
            kind: "discovery",
            refit: 0,
            floor,
            rules,
            rounds,
            seconds,
            reparse_seconds: 0.0,
            reparse_bits: None,
            bytes,
            fields,
            archive_bytes,
            cleanup,
            candidates: candidates.clone(),
        });
        previous = Some(candidates);
    }
    // Reparse only after discovery finishes. Start from the grammar belonging
    // to the smallest discovery archive, then retain every actual checkpoint.
    let reparse_floor = stages[winner].floor;
    let mut reparse_source = stages[winner].candidates.clone();
    for refit in 1..=reparse_refits {
        let directory = work.0.join(format!("reparse-{refit}"));
        fs::create_dir(&directory).map_err(|e| e.to_string())?;
        eprintln!("minimum-bit reparse {refit}; source floor {reparse_floor}");
        let parse_started = Instant::now();
        let source = archive_io::read_packed(
            &mut BufReader::new(fs::File::open(&reparse_source).map_err(|e| e.to_string())?),
            1048576,
        )?;
        let (parsed, before_bits, after_bits) = grammar_reparse::reparse_bounded(
            source,
            Path::new(&args[1]),
            single,
            true,
            maximum_field,
            &directory,
        )?;
        let reparse_seconds = parse_started.elapsed().as_secs_f64();
        let (bytes, fields, rules) = (
            parsed.input_bytes,
            parsed.field_count,
            parsed.grammar.rules.len(),
        );
        let candidates = directory.join("discovery.brp");
        write_packed(&parsed, &candidates)?;
        drop(parsed);
        let (cleanup, selected, archive_bytes) =
            pack_and_clean(&candidates, &directory, cleanup_options, arity_cost)?;
        if archive_bytes < stages[winner].archive_bytes {
            fs::copy(&selected, output.path()).map_err(|e| e.to_string())?;
            winner = stages.len();
        }
        stages.push(Stage {
            kind: "reparse",
            refit,
            floor: reparse_floor,
            rules,
            rounds: 0,
            seconds: 0.0,
            reparse_seconds,
            reparse_bits: Some((before_bits, after_bits)),
            bytes,
            fields,
            archive_bytes,
            cleanup,
            candidates: candidates.clone(),
        });
        eprintln!("reparse {refit}: {before_bits} -> {after_bits} fitted payload bits; {archive_bytes} archive bytes; best {}",
            stages[winner].archive_bytes);
        reparse_source = candidates;
    }
    let chosen = &stages[winner];
    let cleanup = &chosen.cleanup;
    let (bytes, fields, rules, archive_bytes) = (
        chosen.bytes,
        chosen.fields,
        chosen.rules,
        chosen.archive_bytes,
    );
    let rounds = stages.iter().map(|s| s.rounds).sum::<usize>();
    let discovery_seconds = stages.iter().map(|s| s.seconds).sum::<f64>();
    let reparse_seconds = stages.iter().map(|s| s.reparse_seconds).sum::<f64>();
    let schedule = floors
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join(",");
    let stage_reports = stages
        .iter()
        .map(|s| {
            let bits = s.reparse_bits.map_or("null".into(), |(before, after)| {
                format!("{{\"before\":{before},\"after\":{after}}}")
            });
            format!(
                concat!(
            "{{\"kind\":\"{}\",\"refit\":{},\"floor\":{},\"rules\":{},\"rounds\":{},",
            "\"archive_bytes\":{},\"static_archive_bytes\":{},\"selected\":\"{}\",",
            "\"discovery_seconds\":{:.6},\"reparse_seconds\":{:.6},\"fitted_payload_bits\":{}}}"),
                s.kind,
                s.refit,
                s.floor,
                s.rules,
                s.rounds,
                s.archive_bytes,
                s.cleanup.original_bytes,
                s.cleanup.selected,
                s.seconds,
                s.reparse_seconds,
                bits
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    let variants = cleanup.variants.iter().map(|v| format!(
        "{{\"name\":\"{}\",\"archive_bytes\":{},\"rules\":{},\"removed\":{},\"dictionary_bytes\":{},\"payload_bits\":{}}}",
        v.name, v.stats.archive_bytes, v.stats.rules, v.removed,
        v.stats.dictionary_bytes, v.stats.payload_bits
    )).collect::<Vec<_>>().join(",");
    let process = memory::process();
    let json = format!(
        concat!(
            "{{\"profile\":\"maximum-compression\",\"minimum_count\":{},",
            "\"count_floors\":[{}],\"selected_floor\":{},\"stages\":[{}],",
            "\"reparse_refits\":{},\"max_reparse_field_bytes\":{},\"selected_phase\":\"{}\",\"selected_refit\":{},",
            "\"score\":\"{}\",\"fields_policy\":\"{}\",\"fields\":{},\"input_bytes\":{},",
            "\"fill_waves\":{},\"fill_min_count\":{},\"rules\":{},\"rounds\":{},",
            "\"cleanup_ran\":true,\"cleanup_rounds\":{},\"rule_lengths\":\"{}\",\"cleanup_arity_cost\":\"{}\",\"selected\":\"{}\",",
            "\"static_archive_bytes\":{},\"archive_bytes\":{},\"variants\":[{}],",
            "\"discovery_seconds\":{:.6},\"reparse_seconds\":{:.6},\"workflow_seconds\":{:.6},",
            "\"peak_heap_bytes\":{},\"peak_private_bytes\":{}}}\n"
        ),
        minimum,
        schedule,
        chosen.floor,
        stage_reports,
        reparse_refits,
        maximum_field,
        chosen.kind,
        chosen.refit,
        score.name(),
        if single { "single" } else { "lines" },
        fields,
        bytes,
        fill_waves,
        minimum,
        rules,
        rounds,
        cleanup_options.rounds,
        rule_lengths.name(),
        arity_cost.name(),
        cleanup.selected,
        cleanup.original_bytes,
        archive_bytes,
        variants,
        discovery_seconds,
        reparse_seconds,
        started.elapsed().as_secs_f64(),
        memory::peak_heap_bytes(),
        process.peak_private
    );
    if let Some(report) = report_output {
        fs::write(report.path(), &json).map_err(|e| e.to_string())?;
        output.commit()?;
        report.commit()?;
    } else {
        output.commit()?;
    }
    print!("{json}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn both_arity_costs_keep_smallest_archive_across_stages_and_reparses() {
        let limit = memory::Limit::set(256_000_000, 1).unwrap();
        let reservation = Output::new(
            &std::env::temp_dir().join(format!("brepair-arity-cli-{}", std::process::id())),
        )
        .unwrap();
        let root = Workspace::new(&reservation).unwrap();
        let mut mixed = b"alpha beta alpha beta\n\n".repeat(24);
        mixed.extend(0..=255);
        for (index, bytes) in [mixed, Vec::new()].iter().enumerate() {
            let input = root.0.join(format!("input-{index}"));
            fs::write(&input, bytes).unwrap();
            let mut sizes = Vec::new();
            for policy in ["varint", "huffman", "both"] {
                let archive = root.0.join(format!("{index}-{policy}.brh"));
                let report = root.0.join(format!("{index}-{policy}.json"));
                let args = [
                    "compress",
                    input.to_str().unwrap(),
                    archive.to_str().unwrap(),
                    "--threads",
                    "1",
                    "--count-floors",
                    "4,2",
                    "--reparse-refits",
                    "1",
                    "--cleanup-rounds",
                    "4",
                    "--cleanup-arity-cost",
                    policy,
                    "--report",
                    report.to_str().unwrap(),
                ]
                .into_iter()
                .map(String::from)
                .collect::<Vec<_>>();
                run(&args, &limit, false).unwrap();
                let decoded = root.0.join(format!("{index}-{policy}.raw"));
                huffman_archive::decode_file(&archive, &decoded, false).unwrap();
                assert_eq!(&fs::read(decoded).unwrap(), bytes);
                sizes.push(fs::metadata(&archive).unwrap().len());
                let json = fs::read_to_string(report).unwrap();
                assert!(json.contains(&format!("\"cleanup_arity_cost\":\"{policy}\"")));
                if policy == "both" {
                    assert!(json.contains("arity-huffman/"));
                }
            }
            assert_eq!(sizes[2], sizes[0].min(sizes[1]));
            let failed = root.0.join(format!("{index}-invalid.brh"));
            let invalid = [
                "compress",
                input.to_str().unwrap(),
                failed.to_str().unwrap(),
                "--cleanup-arity-cost",
                "both",
                "--rule-lengths",
                "varint",
            ]
            .into_iter()
            .map(String::from)
            .collect::<Vec<_>>();
            assert!(run(&invalid, &limit, false)
                .unwrap_err()
                .contains("require --rule-lengths auto"));
            assert!(!failed.exists());
        }
    }

    #[test]
    fn count_schedules_are_explicit_and_strictly_descending() {
        assert_eq!(count_floors("8,4,2").unwrap(), vec![8, 4, 2]);
        assert_eq!(count_floors("4").unwrap(), vec![4]);
        for invalid in ["", "1", "4,4,2", "4,8,2", "4,", "-2", "8,x,2"] {
            assert!(count_floors(invalid).is_err(), "{invalid}");
        }
    }

    #[test]
    fn optional_reparse_preserves_fields_bounds_and_smallest_complete_checkpoint() {
        let limit = memory::Limit::set(256_000_000, 1).unwrap();
        let reservation = Output::new(
            &std::env::temp_dir().join(format!("brepair-reparse-cli-test-{}", std::process::id())),
        )
        .unwrap();
        let root = Workspace::new(&reservation).unwrap();
        let mut mixed = b"alpha beta alpha beta\n\n".repeat(40);
        mixed.extend(0..=255);
        for (index, single, bytes) in [
            (0, false, mixed.clone()),
            (1, true, mixed),
            (2, false, Vec::new()),
        ] {
            let input = root.0.join(format!("input-{index}"));
            let archive = root.0.join(format!("archive-{index}.brh"));
            let report = root.0.join(format!("report-{index}.json"));
            let work = root.0.join(format!("work-{index}"));
            fs::write(&input, &bytes).unwrap();
            let args = [
                "compress",
                input.to_str().unwrap(),
                archive.to_str().unwrap(),
                "--threads",
                "1",
                "--fields",
                if single { "single" } else { "lines" },
                "--reparse-refits",
                "3",
                "--cleanup-rounds",
                "4",
                "--max-reparse-field-bytes",
                "8192",
                "--keep-work",
                work.to_str().unwrap(),
                "--report",
                report.to_str().unwrap(),
            ]
            .into_iter()
            .map(String::from)
            .collect::<Vec<_>>();
            run(&args, &limit, false).unwrap();
            let decoded = root.0.join(format!("decoded-{index}"));
            let totals = huffman_archive::decode_file(&archive, &decoded, false).unwrap();
            assert_eq!(fs::read(decoded).unwrap(), bytes);
            let expected_fields = if bytes.is_empty() {
                0
            } else if single {
                1
            } else {
                bytes.iter().filter(|&&b| b == b'\n').count() as u64
                    + u64::from(bytes.last() != Some(&b'\n'))
            };
            assert_eq!(totals.fields, expected_fields);
            let mut smallest = u64::MAX;
            for directory in [
                work.clone(),
                work.join("reparse-1"),
                work.join("reparse-2"),
                work.join("reparse-3"),
            ] {
                smallest = smallest.min(fs::metadata(directory.join("static.brh")).unwrap().len());
                for entry in fs::read_dir(directory.join("cleanup")).unwrap() {
                    let path = entry.unwrap().path();
                    if path.extension().is_some_and(|e| e == "brhv") {
                        smallest = smallest.min(fs::metadata(path).unwrap().len());
                    }
                }
            }
            assert_eq!(fs::metadata(archive).unwrap().len(), smallest);
            let json = fs::read_to_string(report).unwrap();
            assert!(json.contains("\"reparse_refits\":3"));
            assert_eq!(json.matches("\"kind\":\"reparse\"").count(), 3);
            if index == 1 {
                let failed = root.0.join("bounded-failure.brh");
                let mut invalid = args.clone();
                invalid[2] = failed.to_str().unwrap().into();
                let position = invalid
                    .iter()
                    .position(|s| s == "--max-reparse-field-bytes")
                    .unwrap();
                invalid[position + 1] = "8".into();
                assert!(run(&invalid, &limit, false)
                    .unwrap_err()
                    .contains("single input exceeds"));
                assert!(!failed.exists());
            }
        }
    }

    #[test]
    fn progressive_profile_keeps_grammar_prefix_and_selects_smallest_archive() {
        let limit = memory::Limit::set(256_000_000, 1).unwrap();
        let output = Output::new(
            &std::env::temp_dir().join(format!("brepair-progressive-test-{}", std::process::id())),
        )
        .unwrap();
        let root = Workspace::new(&output).unwrap();
        let input = root.0.join("input");
        let bytes = b"alpha beta beta alpha\n\n\xff\x00xyxyxy\n".repeat(64);
        fs::write(&input, &bytes).unwrap();
        for (index, single) in [(0, false), (1, true)] {
            let archive = root.0.join(format!("output-{index}.brh"));
            let report = root.0.join(format!("report-{index}.json"));
            let work = root.0.join(format!("work-{index}"));
            let args = [
                "compress",
                input.to_str().unwrap(),
                archive.to_str().unwrap(),
                "--threads",
                "1",
                "--fields",
                if single { "single" } else { "lines" },
                "--count-floors",
                "8,4,2",
                "--cleanup-rounds",
                "4",
                "--fill-waves",
                if single { "1" } else { "2" },
                "--keep-work",
                work.to_str().unwrap(),
                "--report",
                report.to_str().unwrap(),
            ]
            .into_iter()
            .map(String::from)
            .collect::<Vec<_>>();
            run(&args, &limit, false).unwrap();
            let decoded = root.0.join(format!("decoded-{index}"));
            huffman_archive::decode_file(&archive, &decoded, false).unwrap();
            assert_eq!(fs::read(&decoded).unwrap(), bytes);
            let mut smallest = u64::MAX;
            let mut previous_rules = Vec::new();
            for directory in [
                work.clone(),
                work.join("stage-1-min4"),
                work.join("stage-2-min2"),
            ] {
                let packed = archive_io::read_packed(
                    &mut BufReader::new(fs::File::open(directory.join("discovery.brp")).unwrap()),
                    17,
                )
                .unwrap();
                let rules = packed
                    .grammar
                    .rules
                    .iter()
                    .map(|r| (r.left, r.right))
                    .collect::<Vec<_>>();
                assert_eq!(rules[..previous_rules.len()], previous_rules);
                previous_rules = rules;
                smallest = smallest.min(fs::metadata(directory.join("static.brh")).unwrap().len());
                for entry in fs::read_dir(directory.join("cleanup")).unwrap() {
                    let path = entry.unwrap().path();
                    if path.extension().is_some_and(|e| e == "brhv") {
                        smallest = smallest.min(fs::metadata(path).unwrap().len());
                    }
                }
            }
            assert_eq!(fs::metadata(archive).unwrap().len(), smallest);
            let json = fs::read_to_string(report).unwrap();
            assert!(json.contains("\"count_floors\":[8,4,2]"));
            assert!(json.contains("\"cleanup_rounds\":4"));
        }
    }

    #[test]
    fn minimum_two_profile_always_cleans_up_and_preserves_fields_and_bytes() {
        let _limit = memory::Limit::set(256_000_000, 1).unwrap();
        let output = Output::new(
            &std::env::temp_dir().join(format!("brepair-compression-test-{}", std::process::id())),
        )
        .unwrap();
        let root = Workspace::new(&output).unwrap();
        for (index, single, bytes) in [
            (0, false, (0..=255).cycle().take(4096).collect::<Vec<u8>>()),
            (1, true, b"alpha beta alpha\nalpha beta alpha\n".repeat(32)),
            (2, false, Vec::new()),
        ] {
            let input = root.0.join(format!("input-{index}"));
            let archive = root.0.join(format!("archive-{index}.brh"));
            let report = root.0.join(format!("report-{index}.json"));
            fs::write(&input, &bytes).unwrap();
            let args = vec![
                "compress".into(),
                input.to_str().unwrap().into(),
                archive.to_str().unwrap().into(),
                "--threads".into(),
                "1".into(),
                "--fields".into(),
                if single { "single" } else { "lines" }.into(),
                "--report".into(),
                report.to_str().unwrap().into(),
            ];
            run(&args, &_limit, false).unwrap();
            let decoded = root.0.join(format!("decoded-{index}"));
            let totals = huffman_archive::decode_file(&archive, &decoded, false).unwrap();
            assert_eq!(fs::read(decoded).unwrap(), bytes);
            let fields = if bytes.is_empty() {
                0
            } else if single {
                1
            } else {
                bytes.iter().filter(|&&b| b == b'\n').count() as u64
                    + u64::from(bytes.last() != Some(&b'\n'))
            };
            assert_eq!(totals.fields, fields);
            let json = fs::read_to_string(report).unwrap();
            assert!(json.contains("\"minimum_count\":2"));
            assert!(json.contains("\"fill_min_count\":2"));
            assert!(json.contains("\"cleanup_ran\":true"));
            assert!(json.contains("\"cleanup_rounds\":2"));
            assert!(json.contains("binary-shared"));
            let original = fs::read(&archive).unwrap();
            assert!(run(&args, &_limit, false).is_err());
            assert_eq!(fs::read(archive).unwrap(), original);
            assert_eq!(fs::read(input).unwrap(), bytes);
        }
        let input = root.0.join("input-0");
        let destination = root.0.join("invalid.brh");
        for extra in [
            ["--min-count", "1"],
            ["--cleanup-rounds", "0"],
            ["--fill-waves", "0"],
            ["--fill-waves", "9"],
            ["--fields", "words"],
            ["--cleanup-arity-cost", "unknown"],
        ] {
            let args = [
                "compress",
                input.to_str().unwrap(),
                destination.to_str().unwrap(),
                extra[0],
                extra[1],
            ]
            .into_iter()
            .map(String::from)
            .collect::<Vec<_>>();
            assert!(run(&args, &_limit, false).is_err());
            assert!(!destination.exists());
        }
        assert!(fs::read_dir(&root.0)
            .unwrap()
            .all(|e| !e.unwrap().path().is_dir()));
        let retained = root.0.join("retained");
        let kept_output = root.0.join("kept.brh");
        let args = [
            "compress",
            input.to_str().unwrap(),
            kept_output.to_str().unwrap(),
            "--threads",
            "1",
            "--keep-work",
            retained.to_str().unwrap(),
        ]
        .into_iter()
        .map(String::from)
        .collect::<Vec<_>>();
        run(&args, &_limit, false).unwrap();
        assert!(retained.join("discovery.brp").is_file());
        assert!(retained.join("static.brh").is_file());
        assert!(retained.join("cleanup/entropy-1.brhv").is_file());
        let prior = fs::read(retained.join("discovery.brp")).unwrap();
        let another = root.0.join("another.brh");
        let mut repeat = args.clone();
        repeat[2] = another.to_str().unwrap().into();
        assert!(run(&repeat, &_limit, false).is_err());
        assert_eq!(fs::read(retained.join("discovery.brp")).unwrap(), prior);
        assert!(!another.exists());
    }
}
