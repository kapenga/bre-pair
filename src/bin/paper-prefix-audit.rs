//! Inspect fixed grammar byte-prefix helpers; does not alter any vocabulary.
use bre_pair::{archive, grammar::BASE_SYMBOLS, memory, tokenizer::Model};
use std::{
    collections::HashMap,
    env,
    fs::{self, File},
    io::Write,
    process::ExitCode,
};
#[global_allocator]
static ALLOCATOR: memory::Allocator = memory::Allocator;
fn run() -> Result<(), String> {
    let args: Vec<_> = env::args().skip(1).collect();
    if args.len() != 3 && args.len() != 4 {
        return Err("paper-prefix-audit ARCHIVE PATTERNS.brt OUTPUT.json [RAW_SOURCE]".into());
    }
    let _limit = memory::Limit::set(4_000_000_000, 1)?;
    let source = archive::decode(&fs::read(&args[0]).map_err(|e| e.to_string())?)?;
    if let Some(raw) = args.get(3) {
        bre_pair::vocabulary::verify_training_source(&source, std::path::Path::new(raw))?;
    }
    let patterns = Model::from_bytes(&fs::read(&args[1]).map_err(|e| e.to_string())?)?;
    let patterns = &patterns.tokens[BASE_SYMBOLS..];
    if patterns.iter().any(|p| p.ends_field || p.bytes.len() > 4) {
        return Err("expected non-ending byte patterns of length2..4".into());
    }
    let lookup: HashMap<_, _> = patterns
        .iter()
        .enumerate()
        .map(|(i, p)| (p.bytes.as_slice(), i))
        .collect();
    let mut ids = vec![Vec::new(); patterns.len()];
    for (i, rule) in source.grammar.rules.iter().enumerate() {
        if rule.terminals <= 4 && !rule.ends_field {
            let id = (BASE_SYMBOLS + i) as u32;
            let mut raw = Vec::new();
            source.grammar.visit_terminals(id, |s| raw.push(s as u8));
            if let Some(&index) = lookup.get(raw.as_slice()) {
                ids[index].push(id as usize);
            }
        }
    }
    let mut own = vec![0u64; source.grammar.alphabet_size()];
    for block in &source.blocks {
        for id in block.iter() {
            own[id as usize] += 1;
        }
    }
    let mut derived = own.clone();
    for (i, rule) in source.grammar.rules.iter().enumerate().rev() {
        let count = derived[BASE_SYMBOLS + i];
        for child in [rule.left, rule.right] {
            derived[child as usize] = derived[child as usize]
                .checked_add(count)
                .ok_or("derivation count overflow")?;
        }
    }
    let rows:Vec<_>=ids.iter().enumerate().map(|(i,ids)|format!("{{\"pattern\":{i},\"grammar_ids\":[{}],\"direct_stream_uses\":{},\"derivation_node_uses\":{}}}",ids.iter().map(usize::to_string).collect::<Vec<_>>().join(","),ids.iter().map(|&id|own[id]).sum::<u64>(),ids.iter().map(|&id|derived[id]).sum::<u64>())).collect();
    let mut output = File::create_new(&args[2]).map_err(|e| e.to_string())?;
    write!(
        output,
        "{{\"input_bytes\":{},\"patterns\":[{}],\"source_verified\":{}}}",
        source.input_bytes,
        rows.join(","),
        args.len() == 4
    )
    .map_err(|e| e.to_string())?;
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
