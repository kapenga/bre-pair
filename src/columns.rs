//! Reusable lexical columns, with the historical sampled-model pilot adapter.
use crate::{
    codec::{EncodeOptions, Frozen},
    engine::Config,
    huffman_archive,
    huffman_dictionary::byte_model,
    memory,
    scoring::Model,
};
use std::{
    fs::{self, File},
    io::{BufRead, BufReader, BufWriter, Read, Write},
    path::Path,
    time::Instant,
};

fn err(e: std::io::Error) -> String {
    e.to_string()
}

const MAX_ROW_BYTES: usize = 64 << 20;

/// Lexical unquoted PSV: quotes/null spellings are literal; LF delimits rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ColumnLayout {
    pub columns: usize,
    pub rows: u64,
    pub input_bytes: u64,
    pub final_line_feed: bool,
}
impl ColumnLayout {
    pub fn validate(self) -> Result<Self, String> {
        if self.columns == 0
            || self.columns > 1024
            || (self.rows == 0 && (self.input_bytes != 0 || self.final_line_feed))
            || self.input_bytes < self.rows
        {
            return Err("invalid column layout".into());
        }
        Ok(self)
    }
    pub fn to_bytes(self) -> Result<[u8; 32], String> {
        self.validate()?;
        let mut out = [0; 32];
        out[..8].copy_from_slice(b"BRCOLS01");
        out[8..12].copy_from_slice(&(self.columns as u32).to_le_bytes());
        out[12..16].copy_from_slice(&u32::from(self.final_line_feed).to_le_bytes());
        out[16..24].copy_from_slice(&self.rows.to_le_bytes());
        out[24..].copy_from_slice(&self.input_bytes.to_le_bytes());
        Ok(out)
    }
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() != 32 || &bytes[..8] != b"BRCOLS01" {
            return Err("invalid column-layout header".into());
        }
        let columns = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        let flags = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
        let rows = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
        let input_bytes = u64::from_le_bytes(bytes[24..].try_into().unwrap());
        if columns == 0
            || columns > 1024
            || flags > 1
            || (rows == 0 && (input_bytes != 0 || flags != 0))
        {
            return Err("invalid column layout".into());
        }
        Self {
            columns,
            rows,
            input_bytes,
            final_line_feed: flags == 1,
        }
        .validate()
    }
}

/// Full columns in one pass, without the historical pilot's sampling copies.
pub fn split_unquoted_psv(
    input: &Path,
    output: &Path,
    columns: usize,
) -> Result<ColumnLayout, String> {
    if columns == 0 || columns > 1024 {
        return Err("columns must be 1..=1024".into());
    }
    fs::create_dir(output).map_err(err)?;
    let mut outputs = (0..columns)
        .map(|c| {
            File::create_new(output.join(format!("c{c:02}.txt")))
                .map(|f| BufWriter::with_capacity(64 << 10, f))
                .map_err(err)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut reader = BufReader::with_capacity(1 << 20, File::open(input).map_err(err)?);
    let mut raw = Vec::new();
    let mut layout = ColumnLayout {
        columns,
        rows: 0,
        input_bytes: 0,
        final_line_feed: false,
    };
    loop {
        raw.clear();
        let received = reader
            .by_ref()
            .take(MAX_ROW_BYTES as u64 + 1)
            .read_until(b'\n', &mut raw)
            .map_err(err)?;
        if received == 0 {
            break;
        }
        if raw.len() > MAX_ROW_BYTES {
            return Err("row exceeds 64 MiB limit".into());
        }
        layout.final_line_feed = raw.last() == Some(&b'\n');
        let bytes = if layout.final_line_feed {
            &raw[..raw.len() - 1]
        } else {
            &raw
        };
        let values = bytes
            .split(|&b| b == b'|')
            .take(columns + 1)
            .collect::<Vec<_>>();
        if values.len() != columns {
            return Err(format!("wrong column count at row {}", layout.rows));
        }
        write_values(&mut outputs, &values)?;
        layout.rows = layout.rows.checked_add(1).ok_or("row count overflow")?;
        layout.input_bytes = layout
            .input_bytes
            .checked_add(raw.len() as u64)
            .ok_or("input size overflow")?;
    }
    for writer in &mut outputs {
        writer.flush().map_err(err)?;
    }
    File::create_new(output.join("layout.brcol"))
        .map_err(err)?
        .write_all(&layout.to_bytes()?)
        .map_err(err)?;
    Ok(layout)
}

pub fn restore_unquoted_psv(
    input: &Path,
    output: &Path,
    layout: ColumnLayout,
) -> Result<(), String> {
    let layout = layout.validate()?;
    let mut inputs = (0..layout.columns)
        .map(|c| {
            File::open(input.join(format!("c{c:02}.txt")))
                .map(|f| BufReader::with_capacity(64 << 10, f))
                .map_err(err)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let pending = crate::atomic_output::Output::new(output)?;
    let mut writer =
        BufWriter::with_capacity(1 << 20, File::create_new(pending.path()).map_err(err)?);
    let mut value = Vec::new();
    let mut bytes = 0u64;
    for row in 0..layout.rows {
        for (c, reader) in inputs.iter_mut().enumerate() {
            value.clear();
            reader
                .by_ref()
                .take(MAX_ROW_BYTES as u64 + 2)
                .read_until(b'\n', &mut value)
                .map_err(err)?;
            if value.is_empty() || value.last() != Some(&b'\n') || value.len() > MAX_ROW_BYTES + 1 {
                return Err("invalid or truncated column value".into());
            }
            if value[..value.len() - 1].contains(&b'|') {
                return Err("pipe inside unquoted column value".into());
            }
            if c != 0 {
                writer.write_all(b"|").map_err(err)?;
                bytes = bytes.checked_add(1).ok_or("output size overflow")?;
            }
            writer.write_all(&value[..value.len() - 1]).map_err(err)?;
            bytes = bytes
                .checked_add(value.len() as u64 - 1)
                .ok_or("output size overflow")?;
        }
        if row + 1 < layout.rows || layout.final_line_feed {
            writer.write_all(b"\n").map_err(err)?;
            bytes = bytes.checked_add(1).ok_or("output size overflow")?;
        }
        if bytes > layout.input_bytes {
            return Err("column output exceeds declared size".into());
        }
    }
    for reader in &mut inputs {
        if !reader.fill_buf().map_err(err)?.is_empty() {
            return Err("extra column values".into());
        }
    }
    if bytes != layout.input_bytes {
        return Err("column output size mismatch".into());
    }
    writer.flush().map_err(err)?;
    drop(writer);
    pending.commit()
}

pub fn run_cli(args: &[String]) -> Result<(), String> {
    match args.first().map(String::as_str) {
        Some("split") if args.len()==4=>{
            let layout=split_unquoted_psv(Path::new(&args[1]),Path::new(&args[2]),args[3].parse().map_err(|_|"invalid columns")?)?;
            println!("{{\"columns\":{},\"rows\":{},\"input_bytes\":{},\"final_line_feed\":{}}}",layout.columns,layout.rows,layout.input_bytes,layout.final_line_feed);Ok(())
        }
        Some("restore") if args.len()==3=>{
            let input=Path::new(&args[1]);let layout=ColumnLayout::from_bytes(&fs::read(input.join("layout.brcol")).map_err(err)?)?;
            restore_unquoted_psv(input,Path::new(&args[2]),layout)
        }
        _=>Err("brepair columns split INPUT NEW_DIRECTORY COLUMNS | restore COLUMN_DIRECTORY NEW_OUTPUT (unquoted PSV; quotes are literal)".into()),
    }
}
fn mix(mut n: u64) -> u64 {
    n = n.wrapping_add(0x9e3779b97f4a7c15);
    n = (n ^ (n >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    n = (n ^ (n >> 27)).wrapping_mul(0x94d049bb133111eb);
    n ^ (n >> 31)
}
fn bucket(row: u64) -> u64 {
    mix(row ^ 0x425245434f4c3031) % 100
}
fn writers(root: &Path, name: &str, columns: usize) -> Result<Vec<BufWriter<File>>, String> {
    let dir = root.join(name);
    fs::create_dir(&dir).map_err(err)?;
    (0..columns)
        .map(|c| {
            File::create_new(dir.join(format!("c{c:02}.txt")))
                .map(|f| BufWriter::with_capacity(1 << 20, f))
                .map_err(err)
        })
        .collect()
}
fn write_values(out: &mut [BufWriter<File>], values: &[&[u8]]) -> Result<(), String> {
    for (writer, value) in out.iter_mut().zip(values) {
        writer.write_all(value).map_err(err)?;
        writer.write_all(b"\n").map_err(err)?;
    }
    Ok(())
}
fn split(input: &Path, output: &Path, columns: usize, expected: u64) -> Result<(), String> {
    if columns == 0 || columns > 1024 || expected == 0 {
        return Err("invalid table shape".into());
    }
    fs::create_dir(output).map_err(err)?;
    let mut full = writers(output, "full", columns)?;
    let mut one = writers(output, "train01", columns)?;
    let mut five = writers(output, "train05", columns)?;
    let mut test = writers(output, "holdout", columns)?;
    let mut future = writers(output, "future", columns)?;
    let mut test_csv = BufWriter::new(File::create_new(output.join("holdout.csv")).map_err(err)?);
    let mut future_csv = BufWriter::new(File::create_new(output.join("future.csv")).map_err(err)?);
    let mut reader = BufReader::with_capacity(1 << 20, File::open(input).map_err(err)?);
    let (mut row, mut bytes, mut n01, mut n05, mut nh, mut nf) =
        (0u64, 0u64, 0u64, 0u64, 0u64, 0u64);
    let cutoff = expected / 5 * 4 + (expected % 5) * 4 / 5;
    let mut raw = Vec::new();
    loop {
        raw.clear();
        if reader.read_until(b'\n', &mut raw).map_err(err)? == 0 {
            break;
        }
        if raw.len() > 64 << 20 || raw.last() != Some(&b'\n') {
            return Err("table needs complete bounded LF records".into());
        }
        let values: Vec<_> = raw[..raw.len() - 1]
            .split(|&b| b == b'|')
            .take(columns + 1)
            .collect();
        if values.len() != columns {
            return Err(format!("wrong field count at row {row}"));
        }
        write_values(&mut full, &values)?;
        let b = bucket(row);
        if row < cutoff && b < 5 {
            write_values(&mut five, &values)?;
            n05 += 1;
        }
        if row < cutoff && b < 1 {
            write_values(&mut one, &values)?;
            n01 += 1;
        }
        if row >= cutoff || b >= 5 {
            write_values(&mut test, &values)?;
            test_csv.write_all(&raw).map_err(err)?;
            nh += 1;
        }
        if row >= cutoff {
            write_values(&mut future, &values)?;
            future_csv.write_all(&raw).map_err(err)?;
            nf += 1;
        }
        row += 1;
        bytes = bytes
            .checked_add(raw.len() as u64)
            .ok_or("byte count overflow")?;
    }
    if row != expected || n05 + nh != row || n01 > n05 {
        return Err("table or sample counts differ".into());
    }
    for group in [&mut full, &mut one, &mut five, &mut test, &mut future] {
        for writer in group {
            writer.flush().map_err(err)?;
        }
    }
    test_csv.flush().map_err(err)?;
    future_csv.flush().map_err(err)?;
    let json=format!("{{\"rows\":{row},\"columns\":{columns},\"raw_bytes\":{bytes},\"training_prefix_rows\":{cutoff},\"train01_rows\":{n01},\"train05_rows\":{n05},\"holdout_rows\":{nh},\"future_rows\":{nf},\"seed_hex\":\"425245434f4c3031\",\"sampling\":\"nested 1/5 percent row-hash samples within first 80 percent; holdout excludes entire 5 percent sample; future is the final 20 percent and is a subset of holdout\"}}\n");
    fs::write(output.join("layout.json"), &json).map_err(err)?;
    print!("{json}");
    Ok(())
}
fn restore(input: &Path, output: &Path, columns: usize, rows: u64) -> Result<(), String> {
    if columns == 0 || columns > 1024 {
        return Err("invalid column count".into());
    }
    // Historical pilot columns always end every value with LF. Their combined
    // byte length equals the original PSV length, including row separators.
    let input_bytes = (0..columns).try_fold(0u64, |total, c| {
        total
            .checked_add(
                fs::metadata(input.join(format!("c{c:02}.txt")))
                    .map_err(err)?
                    .len(),
            )
            .ok_or_else(|| "column length overflow".to_string())
    })?;
    restore_unquoted_psv(
        input,
        output,
        ColumnLayout {
            columns,
            rows,
            input_bytes,
            final_line_feed: rows != 0,
        },
    )
}
fn run(args: &[String]) -> Result<(), String> {
    let usage="column-pilot split INPUT NEW_DIRECTORY COLUMNS ROWS | restore COLUMN_DIRECTORY NEW_OUTPUT COLUMNS ROWS | train SAMPLE NEW_MODEL frequency|weighted-pmi | apply MODEL INPUT NEW_ARCHIVE";
    let started = Instant::now();
    match args.first().map(String::as_str) {
        Some("split") if args.len() == 5 => split(
            Path::new(&args[1]),
            Path::new(&args[2]),
            args[3].parse().map_err(|_| "columns")?,
            args[4].parse().map_err(|_| "rows")?,
        ),
        Some("restore") if args.len() == 5 => restore(
            Path::new(&args[1]),
            Path::new(&args[2]),
            args[3].parse().map_err(|_| "columns")?,
            args[4].parse().map_err(|_| "rows")?,
        ),
        Some("train") if args.len() == 4 => {
            let mut config = Config {
                minimum_count: 2,
                buffer_bytes: 8_000_000_000,
                score: args[3].parse::<Model>()?,
                ..Default::default()
            };
            config.search.fill_waves = 2;
            config.search.fill_min_count = 2;
            let (model, report) =
                Frozen::train(&byte_model(), Path::new(&args[1]), &config, 4, None, |r| {
                    eprintln!(
                        "round {} rules {} symbols {}",
                        r.index, r.total_rules, r.after_refill
                    )
                })?;
            let bytes = model.to_bytes()?;
            let restored = Frozen::from_bytes(&bytes)?;
            if restored.book.lengths().collect::<Vec<_>>()
                != model.book.lengths().collect::<Vec<_>>()
            {
                return Err("frozen table changed".into());
            }
            File::create_new(&args[2])
                .map_err(err)?
                .write_all(&bytes)
                .map_err(err)?;
            println!("{{\"model_bytes\":{},\"rules\":{},\"training_bytes\":{},\"training_fields\":{},\"seconds\":{:.6},\"peak_heap_bytes\":{},\"training_rounds\":{},\"max_code_bits\":{}}}",bytes.len(),model.dictionary.rules.len(),model.training.bytes,model.training.fields,started.elapsed().as_secs_f64(),memory::peak_heap_bytes(),report.rounds,model.book.max_bits);
            Ok(())
        }
        Some("apply") if args.len() == 4 => {
            let bytes = fs::read(&args[1]).map_err(err)?;
            let model = Frozen::from_bytes(&bytes)?;
            let totals = model.encode_file_with_options(
                Path::new(&args[2]),
                Path::new(&args[3]),
                huffman_archive::DEFAULT_BLOCK_SYMBOLS,
                EncodeOptions {
                    workers: 4,
                    ..Default::default()
                },
            )?;
            println!("{{\"bytes\":{},\"fields\":{},\"symbols\":{},\"seconds\":{:.6},\"archive_bytes\":{},\"model_fingerprint\":{},\"model_refit\":false,\"peak_heap_bytes\":{}}}",totals.bytes,totals.fields,totals.symbols,started.elapsed().as_secs_f64(),fs::metadata(&args[3]).map_err(err)?.len(),crate::tokenizer::fingerprint(&bytes),memory::peak_heap_bytes());
            Ok(())
        }
        _ => Err(usage.into()),
    }
}
/// Historical sampled-model pilot CLI, retained for frozen benchmark recipes.
pub fn run_research_cli(args: &[String]) -> Result<(), String> {
    run(args)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn full_columns_preserve_crlf_unterminated_empty_and_binary_tables() {
        for raw in [
            b"".as_slice(),
            b"01|\"x\"|\\N\r\n|0|\xff\n7|8|final",
            b"||\n",
            b"||",
        ] {
            let r = root();
            let input = r.join("source");
            fs::write(&input, raw).unwrap();
            let columns = r.join("columns");
            let layout = split_unquoted_psv(&input, &columns, 3).unwrap();
            assert_eq!(
                ColumnLayout::from_bytes(&layout.to_bytes().unwrap()).unwrap(),
                layout
            );
            assert_eq!(layout.input_bytes, raw.len() as u64);
            restore_unquoted_psv(&columns, &r.join("restored"), layout).unwrap();
            assert_eq!(fs::read(r.join("restored")).unwrap(), raw);
        }
    }
    #[test]
    fn full_column_restoration_is_atomic_and_rejects_bad_layouts() {
        let r = root();
        let input = r.join("source");
        fs::write(&input, b"a|b\n").unwrap();
        let columns = r.join("columns");
        let layout = split_unquoted_psv(&input, &columns, 2).unwrap();
        fs::write(columns.join("c01.txt"), b"").unwrap();
        assert!(restore_unquoted_psv(&columns, &r.join("bad"), layout).is_err());
        assert!(!r.join("bad").exists());
        assert!(ColumnLayout {
            columns: 1usize << 32,
            ..layout
        }
        .to_bytes()
        .is_err());
        assert!(ColumnLayout::from_bytes(&[0; 32]).is_err());
        assert!(split_unquoted_psv(&input, &r.join("wrong"), 3).is_err());
    }
    #[test]
    fn malformed_delimiter_dense_rows_are_rejected_before_writing_values() {
        let r = root();
        let input = r.join("source");
        fs::write(&input, vec![b'|'; 1 << 20]).unwrap();
        let output = r.join("columns");
        assert!(split_unquoted_psv(&input, &output, 2)
            .unwrap_err()
            .contains("wrong column count"));
        assert_eq!(fs::metadata(output.join("c00.txt")).unwrap().len(), 0);
        assert!(!output.join("layout.brcol").exists());
    }
    fn root() -> std::path::PathBuf {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let p = std::env::temp_dir().join(format!(
            "bre-column-{}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        fs::create_dir(&p).unwrap();
        p
    }
    #[test]
    fn split_and_restore_preserve_empty_binary_and_numeric_spellings() {
        let r = root();
        let raw = b"001|1.00|\n02|-0|\x00\n03|0.10|x\n";
        fs::write(r.join("raw"), raw).unwrap();
        split(&r.join("raw"), &r.join("columns"), 3, 3).unwrap();
        restore(&r.join("columns/full"), &r.join("restored"), 3, 3).unwrap();
        assert_eq!(fs::read(r.join("restored")).unwrap(), raw);
    }
    #[test]
    fn sample_membership_is_nested_and_disjoint_from_holdout() {
        let mut selected = 0;
        for row in 0..10000 {
            let one = row < 8000 && bucket(row) < 1;
            let five = row < 8000 && bucket(row) < 5;
            let test = row >= 8000 || bucket(row) >= 5;
            assert!(!one || five);
            assert_ne!(five, test);
            selected += usize::from(one);
        }
        assert!(selected > 30 && selected < 160);
    }
    #[test]
    fn restoration_rejects_a_missing_column_value() {
        let r = root();
        fs::write(r.join("c00.txt"), b"1\n2\n").unwrap();
        fs::write(r.join("c01.txt"), b"x\n").unwrap();
        assert!(restore(&r, &r.join("bad"), 2, 2).is_err());
    }
    #[test]
    fn frozen_sample_model_covers_unseen_bytes_without_refitting() {
        let r = root();
        fs::write(r.join("train"), b"12\n12\n13\n12\n").unwrap();
        fs::write(r.join("test"), b"99\n\xff\n").unwrap();
        run(&[
            "train".into(),
            r.join("train").display().to_string(),
            r.join("model").display().to_string(),
            "weighted-pmi".into(),
        ])
        .unwrap();
        let before = fs::read(r.join("model")).unwrap();
        run(&[
            "apply".into(),
            r.join("model").display().to_string(),
            r.join("test").display().to_string(),
            r.join("archive").display().to_string(),
        ])
        .unwrap();
        huffman_archive::decode_file(&r.join("archive"), &r.join("restored"), false).unwrap();
        assert_eq!(fs::read(r.join("restored")).unwrap(), b"99\n\xff\n");
        assert_eq!(fs::read(r.join("model")).unwrap(), before);
    }
}
