//! Rank learned entries using training-only whole-field removal experiments.
use bre_pair::{
    targeted,
    tokenizer::{fingerprint, Model, Tokenizer},
    vocabulary,
};
use std::{env, fs, io::Write, path::Path, process::ExitCode};

fn write(path: &Path, bytes: &[u8]) -> Result<(), String> {
    fs::File::create_new(path)
        .and_then(|mut f| f.write_all(bytes))
        .map_err(|e| e.to_string())
}

fn run() -> Result<(), String> {
    let args: Vec<_> = env::args().skip(1).collect();
    const HELP: &str = "rank-collapses TRAIN SEED.brt OUTPUT_DIRECTORY [--threads 4] [--sample-bytes 10000000] [--max-count 794]\n";
    if args.is_empty() || args[0] == "--help" {
        print!("{HELP}");
        return Ok(());
    }
    if args.len() < 3 {
        return Err(HELP.into());
    }
    let (mut workers, mut sample, mut maximum) = (4usize, 10000000u64, 794usize);
    let mut at = 3;
    while at < args.len() {
        let value = args.get(at + 1).ok_or("missing option value")?;
        match args[at].as_str() {
            "--threads" => workers = value.parse().map_err(|_| "invalid workers")?,
            "--sample-bytes" => sample = value.parse().map_err(|_| "invalid sample")?,
            "--max-count" => maximum = value.parse().map_err(|_| "invalid count")?,
            _ => return Err(format!("unknown option {}", args[at])),
        }
        at += 2;
    }
    let training = Path::new(&args[0]);
    let model = Model::from_bytes(&fs::read(&args[1]).map_err(|e| e.to_string())?)?;
    let fp = fingerprint(&model.to_bytes());
    let baseline = vocabulary::scan(
        training,
        &Tokenizer::new(model.clone())?,
        None,
        None,
        workers,
        1,
        0,
        true,
        true,
    )?;
    let ranking = targeted::rank(training, &model, &baseline.uses, maximum, workers, sample)?;
    let root = Path::new(&args[2]);
    fs::create_dir(root).map_err(|e| e.to_string())?;
    let entries = ranking.entries.iter().map(|r| format!(
        "{{\"id\":{},\"uses\":{},\"replacement_tokens\":{},\"upper_bound\":{},\"sample_uses\":{},\"sample_penalty\":{},\"estimated_penalty\":{:.6}}}",
        r.id,r.uses,r.replacement_tokens,r.upper_bound,r.sample_uses,r.sample_penalty,r.estimated_penalty)).collect::<Vec<_>>().join(",");
    let report = format!("{{\"model_fingerprint\":{fp},\"training_bytes\":{},\"training_tokens\":{},\"maximum_collapse\":{maximum},\"shortlist\":{},\"sample_bytes\":{},\"sample_fields\":{},\"sample_stride\":{},\"counterfactual_fields\":{},\"seconds\":{:.6},\"entries\":[{entries}]}}\n",
        baseline.all.bytes,baseline.all.tokens,ranking.entries.len(),ranking.sample_bytes,ranking.sample_fields,
        ranking.sample_stride,ranking.counterfactual_fields,ranking.seconds);
    write(&root.join("ranking.json"), report.as_bytes())?;
    let ids = format!(
        "BRCOLLAPSE01 {fp}\n{}\n",
        ranking
            .entries
            .iter()
            .map(|r| r.id.to_string())
            .collect::<Vec<_>>()
            .join("\n")
    );
    write(&root.join("ranking.collapse"), ids.as_bytes())?;
    println!(
        "ranked {} entries on {} sampled bytes; {} counterfactual field parses in {:.3}s",
        ranking.entries.len(),
        ranking.sample_bytes,
        ranking.counterfactual_fields,
        ranking.seconds
    );
    Ok(())
}
fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}
