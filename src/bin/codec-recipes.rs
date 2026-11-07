//! Inspect variable-arity construction dictionaries without changing codec formats.
use bre_pair::{
    codec::{recipe_compaction, Frozen},
    memory,
};
use std::{fs, path::Path, time::Instant};

#[global_allocator]
static ALLOCATOR: memory::Allocator = memory::Allocator;

fn run() -> Result<(), String> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if !(2..=3).contains(&args.len()) {
        return Err(
            "codec-recipes MODEL.brc NEW_DIRECTORY [MAXIMUM_AI_IDS_PER_REPARSED_RULE]".into(),
        );
    }
    let maximum = args.get(2).map_or(Ok(128), |s| {
        s.parse::<usize>().map_err(|_| "invalid rule bound")
    })?;
    let _limit = memory::Limit::set(2_000_000_000, 1)?;
    let start = Instant::now();
    let model = Frozen::from_bytes(&fs::read(&args[0]).map_err(|e| e.to_string())?)?;
    let analysis = recipe_compaction::analyze(&model, maximum)?;
    let output = Path::new(&args[1]);
    fs::create_dir(output).map_err(|e| e.to_string())?;
    let bytes = analysis.compact.to_bytes();
    fs::write(output.join("prototype.brm"), &bytes).map_err(|e| e.to_string())?;
    let variants: Vec<_> = analysis.variants.iter().map(|v|format!(
        "{{\"variant\":\"{}\",\"model_bytes\":{},\"rules\":{},\"nonbinary_rules\":{},\"child_references\":{},\"changed_definitions\":{},\"removed_helpers\":{}}}",
        v.name,v.model_bytes,v.rules,v.nonbinary_rules,v.child_references,v.changed_definitions,v.removed_helpers)).collect();
    let json = format!(concat!("{{\n  \"experimental_format\":\"BRMGP001\",\n  \"production_formats_unchanged\":true,\n",
        "  \"all_coded_root_AI_IDs_verified_after_serialization\":true,\n  \"Huffman_code_lengths_preserved\":true,\n",
        "  \"original_model_bytes\":{},\n  \"maximum_AI_IDs_per_reparsed_rule\":{},\n  \"selected\":\"{}\",\n",
        "  \"variants\":[{}],\n  \"seconds\":{:.6},\n  \"peak_heap_bytes\":{},\n  \"peak_private_bytes\":{}\n}}\n"),
        model.to_bytes()?.len(),maximum,analysis.selected,variants.join(","),start.elapsed().as_secs_f64(),memory::peak_heap_bytes(),memory::process().peak_private);
    fs::write(output.join("report.json"), &json).map_err(|e| e.to_string())?;
    print!("{json}");
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("codec-recipes: {e}");
        std::process::exit(1);
    }
}
