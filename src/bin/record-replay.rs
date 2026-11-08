//! Research-only portable replay of an already frozen static record model.
//! Whole LF-retaining rows are byte strings; this is not typed-table storage.
use bre_pair::{codec::MatchScratch, record_frame::Format, shared_huffman::StaticModel};
use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    path::Path,
    time::Instant,
};

fn mix(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
    value ^ (value >> 31)
}
fn run(
    model_path: &Path,
    test_path: &Path,
    output: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir(output)?;
    let start = Instant::now();
    let bytes = fs::read(model_path)?;
    let model = StaticModel::from_bytes(&bytes)?;
    let matcher = model.prepare()?;
    let encoder = model.codebook().encoding();
    let decoder = model.prepare_decoder(false, 64 << 20)?;
    fs::write(output.join("model.bin"), &bytes)?;
    let prepare_seconds = start.elapsed().as_secs_f64();
    let raw = fs::read(test_path)?;
    if raw.is_empty() || raw.last() != Some(&b'\n') {
        return Err("this control requires nonempty LF-retaining records".into());
    }
    let mut raw_offsets = vec![0usize];
    raw_offsets.extend(
        raw.iter()
            .enumerate()
            .filter_map(|(i, &b)| (b == b'\n').then_some(i + 1)),
    );
    let rows = raw_offsets.len() - 1;
    let header = Format::EofPlain.file_header();
    let mut offsets = Vec::with_capacity(rows + 1);
    offsets.push(header.len() as u64);
    let mut terminals = Vec::new();
    let mut scratch = MatchScratch::default();
    let mut data = BufWriter::new(File::create(output.join("payload.bin"))?);
    let start = Instant::now();
    data.write_all(&header)?;
    for ends in raw_offsets.windows(2) {
        terminals.clear();
        terminals.extend(raw[ends[0]..ends[1]].iter().map(|&b| u32::from(b)));
        terminals.push(256);
        let symbols = matcher.segment_with(&terminals, &mut scratch)?;
        let (payload, _) = encoder.encode(symbols)?;
        data.write_all(&payload)?;
        offsets.push(offsets.last().unwrap() + payload.len() as u64);
    }
    data.flush()?;
    drop(data);
    let mut index = BufWriter::new(File::create(output.join("offsets.u64"))?);
    for offset in &offsets {
        index.write_all(&offset.to_le_bytes())?;
    }
    index.flush()?;
    drop(index);
    let encode_store_seconds = start.elapsed().as_secs_f64();
    let coded = fs::read(output.join("payload.bin"))?;
    let stored_index = fs::read(output.join("offsets.u64"))?;
    if coded.len() as u64 != *offsets.last().unwrap()
        || coded[..header.len()] != header
        || stored_index.len() != offsets.len() * 8
        || stored_index
            .chunks_exact(8)
            .zip(&offsets)
            .any(|(chunk, &value)| u64::from_le_bytes(chunk.try_into().unwrap()) != value)
    {
        return Err("stored payload/index mismatch".into());
    }
    let maximum = raw_offsets.windows(2).map(|w| w[1] - w[0]).max().unwrap();
    let mut decoded = Vec::with_capacity(maximum + 32);
    let start = Instant::now();
    for (i, ends) in offsets.windows(2).enumerate() {
        model.decode_field(
            &decoder,
            &coded[ends[0] as usize..ends[1] as usize],
            1 << 20,
            &mut decoded,
        )?;
        if decoded != raw[raw_offsets[i]..raw_offsets[i + 1]] {
            return Err(format!("reconstruction mismatch at {i}").into());
        }
    }
    let decode_verify_seconds = start.elapsed().as_secs_f64();
    let mut full = vec![0u8; raw.len()];
    let mut written = 0;
    let start = Instant::now();
    for ends in offsets.windows(2) {
        model.decode_field(
            &decoder,
            &coded[ends[0] as usize..ends[1] as usize],
            1 << 20,
            &mut decoded,
        )?;
        full[written..written + decoded.len()].copy_from_slice(&decoded);
        written += decoded.len();
    }
    let decode_materialize_seconds = start.elapsed().as_secs_f64();
    if written != raw.len() || full != raw {
        return Err("full output mismatch".into());
    }
    let mut nanos = Vec::with_capacity(20000);
    let mut random = 20261108;
    for k in 0..21000 {
        random = mix(random);
        let i = (random % rows as u64) as usize;
        let start = Instant::now();
        model.decode_field(
            &decoder,
            &coded[offsets[i] as usize..offsets[i + 1] as usize],
            1 << 20,
            &mut decoded,
        )?;
        let elapsed = start.elapsed().as_nanos();
        if decoded != raw[raw_offsets[i]..raw_offsets[i + 1]] {
            return Err("random lookup mismatch".into());
        }
        if k >= 1000 {
            nanos.push(elapsed);
        }
    }
    nanos.sort_unstable();
    let model_bytes = bytes.len();
    let payload_bytes = coded.len();
    let index_bytes = stored_index.len();
    println!(
        "{{\"verified\":true,\"frozen_model_replay\":true,\"rows\":{rows},\"raw_bytes\":{},\"model_bytes\":{model_bytes},\"payload_bytes\":{payload_bytes},\"index_bytes\":{index_bytes},\"complete_bytes\":{},\"prepare_seconds\":{prepare_seconds},\"encode_store_seconds\":{encode_store_seconds},\"decode_verify_seconds\":{decode_verify_seconds},\"decode_materialize_seconds\":{decode_materialize_seconds},\"warm_lookup_p50_ns\":{},\"warm_lookup_p99_ns\":{}}}",
        raw.len(), model_bytes + payload_bytes + index_bytes, nanos[nanos.len()/2], nanos[nanos.len()*99/100]
    );
    Ok(())
}
fn main() {
    let args: Vec<_> = std::env::args_os().collect();
    if args.len() != 4 {
        eprintln!("usage: record-replay MODEL TEST OUTPUT_DIRECTORY");
        std::process::exit(2);
    }
    if let Err(error) = run(
        Path::new(&args[1]),
        Path::new(&args[2]),
        Path::new(&args[3]),
    ) {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
