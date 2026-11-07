//! Research-only cached language-weighted vocabulary reduction.
use bre_pair::{
    archive, document_support,
    engine::Compressed,
    grammar::BASE_SYMBOLS,
    memory,
    tokenizer::{fingerprint, Model, Tokenizer},
    vocabulary::{self, reverse, Totals},
};
use std::{
    env,
    fs::{self, File},
    io::{BufReader, Read, Write},
    path::Path,
    process::ExitCode,
    time::Instant,
};

#[global_allocator]
static ALLOCATOR: memory::Allocator = memory::Allocator;

fn u64_at(raw: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(raw[at..at + 8].try_into().unwrap())
}
fn write_new(path: &Path, raw: &[u8]) -> Result<(), String> {
    File::create_new(path)
        .and_then(|mut f| f.write_all(raw))
        .map_err(|e| e.to_string())
}
fn sum(a: Totals, b: Totals) -> Result<Totals, String> {
    let add = |x: u64, y: u64| {
        x.checked_add(y)
            .ok_or_else(|| "language total overflow".to_string())
    };
    Ok(Totals {
        bytes: add(a.bytes, b.bytes)?,
        tokens: add(a.tokens, b.tokens)?,
        fields: add(a.fields, b.fields)?,
        byte_tokens: add(a.byte_tokens, b.byte_tokens)?,
    })
}
fn totals_json(t: Totals) -> String {
    format!(
        "{{\"bytes\":{},\"tokens\":{},\"fields\":{},\"byte_tokens\":{}}}",
        t.bytes, t.tokens, t.fields, t.byte_tokens
    )
}

struct Documents<R> {
    reader: R,
    left: u64,
    end: u64,
    fields: u64,
    category: u16,
    offset: u64,
    chinese: Totals,
    turkish: Totals,
}
impl<R: Read> Documents<R> {
    fn new(mut reader: R) -> Result<Self, String> {
        let mut h = [0; 24];
        reader.read_exact(&mut h).map_err(|e| e.to_string())?;
        if &h[..8] != b"BREDOC01"
            || h[8..12] != 104u32.to_le_bytes()
            || h[12..16] != 0u32.to_le_bytes()
        {
            return Err("requires an indexed training corpus".into());
        }
        let mut result = Self {
            reader,
            left: u64_at(&h, 16),
            end: 0,
            fields: 0,
            category: 0,
            offset: 0,
            chinese: Totals::default(),
            turkish: Totals::default(),
        };
        result.next()?;
        Ok(result)
    }
    fn next(&mut self) -> Result<(), String> {
        if self.left == 0 {
            self.fields = 0;
            return Ok(());
        }
        let mut row = [0; 104];
        self.reader
            .read_exact(&mut row)
            .map_err(|e| e.to_string())?;
        self.category = u16::from_le_bytes(row[100..102].try_into().unwrap());
        let start = u64_at(&row, 76);
        let length = u64_at(&row, 84);
        self.fields = u64_at(&row, 92);
        if start != self.offset
            || length == 0
            || self.fields == 0
            || ![2, 4].contains(&self.category)
            || row[103] != 0
        {
            return Err("invalid language document index".into());
        }
        self.end = start.checked_add(length).ok_or("document byte overflow")?;
        self.left -= 1;
        Ok(())
    }
    fn symbol(&mut self, bytes: u64, ending: bool) -> Result<bool, String> {
        if self.fields == 0 {
            return Err("symbols exceed the document index".into());
        }
        let chinese = self.category == 4;
        self.offset = self
            .offset
            .checked_add(bytes)
            .ok_or("byte offset overflow")?;
        if self.offset > self.end {
            return Err("symbol crosses a document byte extent".into());
        }
        let target = if chinese {
            &mut self.chinese
        } else {
            &mut self.turkish
        };
        target.bytes += bytes;
        target.fields += u64::from(ending);
        if ending {
            self.fields -= 1;
            if self.fields == 0 {
                if self.offset != self.end {
                    return Err("document ending is misaligned".into());
                }
                self.next()?;
            }
        }
        Ok(chinese)
    }
    fn finish(mut self) -> Result<(Totals, Totals), String> {
        let mut byte = [0];
        if self.left != 0
            || self.fields != 0
            || self.reader.read(&mut byte).map_err(|e| e.to_string())? != 0
        {
            return Err("unfinished or trailing document index".into());
        }
        Ok((self.chinese, self.turkish))
    }
}

fn roots(source: &Compressed, index: &Path) -> Result<(Vec<u64>, Totals, Totals), String> {
    let mut documents = Documents::new(BufReader::new(
        File::open(index).map_err(|e| e.to_string())?,
    ))?;
    let mut counts = vec![0; source.grammar.alphabet_size()];
    for block in &source.blocks {
        for id in block.iter() {
            let ending = source.grammar.ends_field(id);
            let bytes = (source.grammar.terminals(id) - usize::from(ending)) as u64;
            if documents.symbol(bytes, ending)? {
                counts[id as usize] += 1;
            }
        }
    }
    let (chinese, turkish) = documents.finish()?;
    if chinese.bytes + turkish.bytes != source.input_bytes
        || chinese.fields + turkish.fields != source.field_count
    {
        return Err("indexed language totals differ from archive".into());
    }
    Ok((counts, chinese, turkish))
}

fn prepare(args: &[String]) -> Result<(), String> {
    if args.len() != 8 {
        return Err("prepare ARCHIVE MIXED INDEX CHINESE TURKISH CACHE MIN_DOCS WORKERS".into());
    }
    let started = Instant::now();
    let workers = args[7].parse().map_err(|_| "invalid workers")?;
    let minimum = args[6].parse().map_err(|_| "invalid document minimum")?;
    let root = Path::new(&args[5]);
    fs::create_dir(root).map_err(|e| e.to_string())?;
    let source = archive::decode(&fs::read(&args[0]).map_err(|e| e.to_string())?)?;
    vocabulary::verify_training_source(&source, Path::new(&args[1]))?;
    let (language_roots, expected_chinese, expected_turkish) = roots(&source, Path::new(&args[2]))?;
    let mut protected = vec![false; source.grammar.alphabet_size()];
    protected[..BASE_SYMBOLS].fill(true);
    let cleanup = document_support::Cleanup::plan(
        &source,
        &document_support::Policy {
            index: args[2].clone().into(),
            minimum_groups: minimum,
        },
        &protected,
    )?;
    let (pool, support, chinese_support) = Model::candidates_with_language_support(
        &source,
        128,
        Some(&cleanup.active),
        &language_roots,
    )?;
    drop(cleanup);
    drop(source);
    drop(language_roots);
    let tokenizer = Tokenizer::new(pool)?;
    let chinese = vocabulary::scan(
        Path::new(&args[3]),
        &tokenizer,
        None,
        None,
        workers,
        1,
        0,
        true,
        true,
    )?;
    let turkish = vocabulary::scan(
        Path::new(&args[4]),
        &tokenizer,
        None,
        None,
        workers,
        1,
        0,
        true,
        true,
    )?;
    for (actual, expected) in [
        (chinese.all, expected_chinese),
        (turkish.all, expected_turkish),
    ] {
        if (actual.bytes, actual.fields) != (expected.bytes, expected.fields) {
            return Err("language file does not match indexed byte/field scope".into());
        }
    }
    let total = sum(chinese.all, turkish.all)?;
    let uses: Vec<_> = chinese
        .uses
        .iter()
        .zip(&turkish.uses)
        .map(|(&a, &b)| a.checked_add(b).ok_or("usage overflow"))
        .collect::<Result<_, _>>()?;
    let model = tokenizer.model().to_bytes();
    write_new(&root.join("pool.brt"), &model)?;
    let mut cache = b"BRLANG01".to_vec();
    cache.extend(fingerprint(&model).to_le_bytes());
    cache.extend((uses.len() as u64).to_le_bytes());
    for t in [total, chinese.all] {
        for n in [t.bytes, t.tokens, t.fields, t.byte_tokens] {
            cache.extend(n.to_le_bytes());
        }
    }
    for values in [&uses, &chinese.uses, &support, &chinese_support] {
        for n in values {
            cache.extend(n.to_le_bytes());
        }
    }
    write_new(&root.join("counts.bin"), &cache)?;
    let report=format!("{{\"vocabulary\":{},\"total\":{},\"chinese\":{},\"turkish\":{},\"seconds\":{:.6},\"construction_support_partitioned\":true,\"round_trip_verified\":true}}\n",uses.len(),totals_json(total),totals_json(chinese.all),totals_json(turkish.all),started.elapsed().as_secs_f64());
    write_new(&root.join("preparation.json"), report.as_bytes())?;
    print!("{report}");
    Ok(())
}

fn select(args: &[String]) -> Result<(), String> {
    if args.len() != 6 {
        return Err("select CACHE OUTPUT NUMERATOR DENOMINATOR VOCABULARY WORKERS".into());
    }
    let started = Instant::now();
    let root = Path::new(&args[0]);
    let output = Path::new(&args[1]);
    let numerator = args[2]
        .parse()
        .map_err(|_| "invalid multiplier numerator")?;
    let denominator = args[3]
        .parse()
        .map_err(|_| "invalid multiplier denominator")?;
    let vocabulary = args[4].parse().map_err(|_| "invalid vocabulary")?;
    let workers = args[5].parse().map_err(|_| "invalid workers")?;
    fs::create_dir(output).map_err(|e| e.to_string())?;
    let bytes = fs::read(root.join("pool.brt")).map_err(|e| e.to_string())?;
    let pool = Model::from_bytes(&bytes)?;
    let cache = fs::read(root.join("counts.bin")).map_err(|e| e.to_string())?;
    let n = pool.tokens.len();
    if cache.len() != 88 + n * 32
        || &cache[..8] != b"BRLANG01"
        || u64_at(&cache, 8) != fingerprint(&bytes)
        || u64_at(&cache, 16) != n as u64
    {
        return Err("language cache length or fingerprint differs".into());
    }
    let total = Totals {
        bytes: u64_at(&cache, 24),
        tokens: u64_at(&cache, 32),
        fields: u64_at(&cache, 40),
        byte_tokens: u64_at(&cache, 48),
    };
    let vector = |part: usize| {
        (0..n)
            .map(|i| u64_at(&cache, 88 + part * n * 8 + i * 8))
            .collect::<Vec<_>>()
    };
    let uses = vector(0);
    let chinese = vector(1);
    let support = vector(2);
    let chinese_support = vector(3);
    let mut rounds = 0;
    let result = reverse::select_language_weighted(
        pool,
        &support,
        uses,
        total,
        reverse::LanguageWeight {
            uses: chinese,
            support: chinese_support,
            numerator,
            denominator,
        },
        reverse::Options {
            vocabulary,
            workers,
            schedule: vocabulary::PruningSchedule::default(),
            exact_refresh: false,
            coverage_discount_ppm: 0,
            coverage_settings: reverse::CoverageSettings::default(),
            batch_search: None,
        },
        |round| {
            rounds += 1;
            eprintln!(
                "round={rounds} before={} after={} tokens={}",
                round.before, round.after, round.after_totals.tokens
            );
        },
    )?;
    let mut chinese = Totals::default();
    for (id, (&count, token)) in result
        .maintained_language_uses
        .as_ref()
        .ok_or("missing language counts")?
        .iter()
        .zip(&result.model.tokens)
        .enumerate()
    {
        chinese.bytes += count * token.bytes.len() as u64;
        chinese.tokens += count;
        chinese.fields += count * u64::from(token.ends_field);
        if id < 256 {
            chinese.byte_tokens += count;
        }
    }
    write_new(&output.join("model.brt"), &result.model.to_bytes())?;
    let report=format!("{{\"numerator\":{numerator},\"denominator\":{denominator},\"vocabulary\":{vocabulary},\"rounds\":{rounds},\"maintained\":{},\"maintained_chinese\":{},\"seconds\":{:.6},\"peak_heap_bytes\":{}}}\n",totals_json(result.maintained),totals_json(chinese),started.elapsed().as_secs_f64(),memory::peak_heap_bytes());
    write_new(&output.join("selection.json"), report.as_bytes())?;
    print!("{report}");
    Ok(())
}

fn run() -> Result<(), String> {
    let args: Vec<_> = env::args().skip(1).collect();
    let _limit = memory::Limit::set(8_000_000_000, 16)?;
    let _progress = bre_pair::training_progress::enable();
    match args.first().map(String::as_str) {
        Some("prepare") => prepare(&args[1..]),
        Some("select") => select(&args[1..]),
        _ => Err("expected prepare or select".into()),
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use bre_pair::{
        grammar::{Grammar, EOF},
        packed::PackedSymbols,
    };
    #[test]
    fn shared_grammar_construction_counts_partition_before_pool_deduplication() {
        let mut grammar = Grammar::default();
        let ab = grammar.add((97, 98)).unwrap();
        let word = grammar.add((ab, EOF)).unwrap();
        let mut block = PackedSymbols::new(9);
        for _ in 0..5 {
            block.push(word);
        }
        let source = Compressed {
            grammar,
            blocks: vec![block],
            input_bytes: 10,
            field_count: 5,
        };
        let mut roots = vec![0; source.grammar.alphabet_size()];
        roots[word as usize] = 3;
        let (pool, support, chinese) =
            Model::candidates_with_language_support(&source, 128, None, &roots).unwrap();
        let (plain, expected) = Model::candidates(&source, 128).unwrap();
        assert_eq!(plain.to_bytes(), pool.to_bytes());
        assert_eq!(support, expected);
        for id in [97usize, 98, EOF as usize, ab as usize, word as usize] {
            assert_eq!(support[id], 5);
            assert_eq!(chinese[id], 3);
        }
        roots[word as usize] = 6;
        assert!(Model::candidates_with_language_support(&source, 128, None, &roots).is_err());
    }
    #[test]
    fn compressed_root_languages_follow_natural_eof_and_document_extents() {
        let mut bytes = b"BREDOC01".to_vec();
        bytes.extend(104u32.to_le_bytes());
        bytes.extend(0u32.to_le_bytes());
        bytes.extend(2u64.to_le_bytes());
        for (offset, category) in [(0u64, 2u16), (2, 4)] {
            let mut row = [0; 104];
            row[76..84].copy_from_slice(&offset.to_le_bytes());
            row[84..92].copy_from_slice(&2u64.to_le_bytes());
            row[92..100].copy_from_slice(&1u64.to_le_bytes());
            row[100..102].copy_from_slice(&category.to_le_bytes());
            bytes.extend(row);
        }
        let mut docs = Documents::new(bytes.as_slice()).unwrap();
        assert!(!docs.symbol(2, false).unwrap());
        assert!(!docs.symbol(0, true).unwrap());
        assert!(docs.symbol(2, true).unwrap());
        let (chinese, turkish) = docs.finish().unwrap();
        assert_eq!(
            (chinese.bytes, chinese.fields, turkish.bytes, turkish.fields),
            (2, 1, 2, 1)
        );
        bytes[12] = 1;
        assert!(Documents::new(bytes.as_slice()).is_err());
    }
}
