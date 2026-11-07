//! Paper-method adaptation; the released PathPiece repository omits training.
use bre_pair::{
    grammar::BASE_SYMBOLS,
    memory, pathpiece,
    tokenizer::{Model, Tokenizer},
};
use std::{
    env,
    fs::{self, File},
    io::{BufRead, BufReader, Write},
    process::ExitCode,
    time::Instant,
};
#[global_allocator]
static ALLOCATOR: memory::Allocator = memory::Allocator;
fn run() -> Result<(), String> {
    let args: Vec<_> = env::args().skip(1).collect();
    if args.len() != 5 {
        return Err("paper-pathpiece-train SEED.brt RAW FINAL.brt VOCABULARY MEMORY_BYTES".into());
    }
    let vocabulary: usize = args[3].parse().map_err(|_| "invalid budget")?;
    let limit = memory::Limit::set(args[4].parse().map_err(|_| "invalid memory")?, 1)?;
    let mut model = Model::from_bytes(&fs::read(&args[0]).map_err(|e| e.to_string())?)?;
    // Published reference recommends width 16; EOF remains a standalone basis
    // entry, so its DP is equivalent to the authors' byte-only DAG plus EOF.
    model.tokens = model
        .tokens
        .into_iter()
        .enumerate()
        .filter(|(id, t)| *id < BASE_SYMBOLS || (!t.ends_field && t.bytes.len() <= 16))
        .map(|(_, t)| t)
        .collect();
    if model.tokens.len() < vocabulary {
        return Err("PathPiece seed has insufficient eligible entries".into());
    }
    let started = Instant::now();
    let mut round = 0;
    while model.tokens.len() > vocabulary {
        let n = model.tokens.len();
        let tokenizer = Tokenizer::new(model)?;
        let mut losses = vec![0; n];
        let mut uses = vec![0; n];
        let mut input = BufReader::new(File::open(&args[1]).map_err(|e| e.to_string())?);
        let mut field = Vec::new();
        let mut tokens = 0;
        let mut bytes = 0u64;
        loop {
            field.clear();
            if input
                .read_until(b'\n', &mut field)
                .map_err(|e| e.to_string())?
                == 0
            {
                break;
            }
            tokens += pathpiece::score_field(&tokenizer, &field, &mut losses, &mut uses)?;
            bytes += field.len() as u64;
        }
        let target = vocabulary.max(n * 9 / 10);
        model = pathpiece::remove_batch(tokenizer.model(), &losses, &uses, target)?;
        round += 1;
        eprintln!("{{\"round\":{round},\"before\":{n},\"after\":{target},\"bytes\":{bytes},\"tokens\":{tokens},\"seconds\":{:.3}}}",started.elapsed().as_secs_f64());
    }
    File::create_new(&args[2])
        .and_then(|mut file| file.write_all(&model.to_bytes()))
        .map_err(|e| e.to_string())?;
    println!("{{\"rounds\":{round},\"vocabulary\":{},\"seconds\":{:.3},\"heap_peak_bytes\":{},\"memory_limit_bytes\":{},\"maximum_token_bytes\":16,\"scope\":\"published occurrence-deletion approximation; 10 percent batches; exact corpus recount each round; local implementation, not authors trainer\"}}",model.tokens.len(),started.elapsed().as_secs_f64(),memory::peak_heap_bytes(),limit.total);
    Ok(())
}
fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}
