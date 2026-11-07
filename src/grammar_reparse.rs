//! Research refinement of a fixed raw-byte grammar using minimum-bit parsing.
//! The reference path retains a field-sized DP. The bounded path uses buffered
//! raw bytes and a choice spool for large fields, preserving exact costs/ties.
use crate::{
    codec::{Frozen, MatchScratch},
    engine::Compressed,
    grammar::{BASE_SYMBOLS, EOF},
    huffman::Codebook,
    huffman_archive::Totals,
    huffman_dictionary::Dictionary,
    packed::{symbol_width, PackedSymbols},
};
use std::{
    fs,
    io::{BufRead, BufReader, Read},
    path::Path,
};

fn histogram(source: &Compressed) -> Result<Vec<u64>, String> {
    let mut counts = vec![0u64; source.grammar.alphabet_size()];
    for block in &source.blocks {
        for id in block.iter() {
            counts[id as usize] = counts[id as usize].checked_add(1).ok_or("count overflow")?;
        }
    }
    Ok(counts)
}

fn weights(source: &Compressed, shared: bool) -> Result<Vec<u64>, String> {
    let mut counts = histogram(source)?;
    if shared {
        let mut live = counts.iter().map(|&n| n != 0).collect::<Vec<_>>();
        for (i, r) in source.grammar.rules.iter().enumerate().rev() {
            if live[BASE_SYMBOLS + i] {
                for id in [r.left, r.right] {
                    live[id as usize] = true;
                    counts[id as usize] = counts[id as usize]
                        .checked_add(1)
                        .ok_or("reference overflow")?;
                }
            }
        }
    }
    // Retain every existing grammar candidate, including currently unused
    // ones, and guarantee a coded byte/EOF fallback at every position.
    for count in &mut counts {
        *count = (*count).max(1);
    }
    Ok(counts)
}

pub fn reparse(
    source: Compressed,
    input: &Path,
    single: bool,
    shared: bool,
    maximum_field: usize,
) -> Result<(Compressed, u128, u128), String> {
    reparse_impl(source, input, single, shared, maximum_field, None)
}

/// Exact refinement with bounded field buffers and an owned scratch file for
/// large-field choices. Scratch lives beside the output, not on the OS temp drive.
pub fn reparse_bounded(
    source: Compressed,
    input: &Path,
    single: bool,
    shared: bool,
    maximum_field: usize,
    scratch_directory: &Path,
) -> Result<(Compressed, u128, u128), String> {
    reparse_impl(
        source,
        input,
        single,
        shared,
        maximum_field,
        Some(scratch_directory),
    )
}

fn reparse_impl(
    source: Compressed,
    input: &Path,
    single: bool,
    shared: bool,
    maximum_field: usize,
    scratch_directory: Option<&Path>,
) -> Result<(Compressed, u128, u128), String> {
    if maximum_field == 0 || maximum_field == usize::MAX {
        return Err("invalid reparse field-size bound".into());
    }
    let counts = weights(&source, shared)?;
    // Build the codebook before duplicating the grammar into a dictionary.
    let book = Codebook::from_counts(&counts)?;
    let mut lengths = vec![0u8; counts.len()];
    drop(counts);
    for (id, n) in book.lengths() {
        lengths[id as usize] = n;
    }
    let training = Totals {
        bytes: source.input_bytes,
        fields: source.field_count,
        tokens: source
            .input_bytes
            .checked_add(source.field_count)
            .ok_or("terminal overflow")?,
        symbols: source.token_count() as u64,
    };
    let before = source
        .blocks
        .iter()
        .flat_map(PackedSymbols::iter)
        .map(|id| u128::from(lengths[id as usize]))
        .sum();
    let Compressed {
        grammar,
        input_bytes,
        field_count,
        blocks: old_blocks,
    } = source;
    drop(old_blocks);
    crate::training_progress::phase("reparse/matcher", 0, 0, lengths.len() as u64, "symbols");
    let width = symbol_width(grammar.alphabet_size());
    if let Some(directory) = scratch_directory {
        let matcher = crate::codec::spooled::RawMatcher::new(&grammar, &book)?;
        drop(book);
        eprintln!(
            "reparse radix: {} retained bytes; current heap {}; peak heap {}",
            matcher.storage_bytes(),
            crate::memory::heap_bytes(),
            crate::memory::peak_heap_bytes()
        );
        crate::training_progress::phase("reparse/fields", 0, 0, input_bytes, "bytes");
        let (blocks, after) = bounded_fields(
            &matcher,
            &lengths,
            width,
            BoundedInput {
                path: input,
                bytes: input_bytes,
                fields: field_count,
                single,
                maximum_field,
                directory,
            },
        )?;
        return Ok((
            Compressed {
                grammar,
                input_bytes,
                field_count,
                blocks,
            },
            before,
            after,
        ));
    }
    let mut dictionary = Dictionary::from_grammar(&grammar)?;
    dictionary.ai_tokens = true;
    let model = Frozen {
        dictionary,
        book,
        training,
    };
    let matcher = model.prepare()?;
    drop(model);
    crate::training_progress::phase("reparse/fields", 0, 0, input_bytes, "bytes");
    let mut blocks = Vec::new();
    let mut block = PackedSymbols::with_capacity(width, 1 << 20);
    let mut reader = BufReader::new(fs::File::open(input).map_err(|e| e.to_string())?);
    let mut bytes = Vec::new();
    let mut terminals = Vec::new();
    let mut scratch = MatchScratch::default();
    let (mut actual_bytes, mut actual_fields, mut after) = (0u64, 0u64, 0u128);
    let mut next_progress = 16 * 1024 * 1024u64;
    loop {
        bytes.clear();
        if single {
            reader
                .by_ref()
                .take(maximum_field as u64 + 1)
                .read_to_end(&mut bytes)
                .map_err(|e| e.to_string())?;
        } else {
            // Bound allocation even for a malformed or unexpectedly long line.
            reader
                .by_ref()
                .take(maximum_field as u64 + 1)
                .read_until(b'\n', &mut bytes)
                .map_err(|e| e.to_string())?;
        }
        if bytes.is_empty() {
            break;
        }
        if bytes.len() > maximum_field {
            return Err("field exceeds explicit reparse bound".into());
        }
        actual_bytes = actual_bytes
            .checked_add(bytes.len() as u64)
            .ok_or("byte overflow")?;
        actual_fields += 1;
        terminals.clear();
        // Include EOF in the initial reservation. Appending it to an exactly
        // filled field buffer would otherwise trigger a doubling allocation.
        terminals.reserve(bytes.len().checked_add(1).ok_or("field length overflow")?);
        terminals.extend(bytes.iter().map(|&b| u32::from(b)));
        terminals.push(EOF);
        for &id in matcher.segment_with(&terminals, &mut scratch)? {
            after += u128::from(lengths[id as usize]);
            block.push(id);
        }
        if actual_bytes >= next_progress {
            crate::training_progress::phase(
                "reparse/fields",
                0,
                actual_bytes,
                input_bytes,
                "bytes",
            );
            next_progress = actual_bytes.saturating_add(16 * 1024 * 1024);
        }
        if block.len() >= 1 << 20 {
            block.shrink_to_fit();
            blocks.push(std::mem::replace(
                &mut block,
                PackedSymbols::with_capacity(width, 1 << 20),
            ));
        }
        if single {
            break;
        }
    }
    if (actual_bytes, actual_fields) != (input_bytes, field_count) {
        return Err("raw input or field policy differs from saved grammar totals".into());
    }
    crate::training_progress::phase("reparse/fields", 0, actual_bytes, input_bytes, "bytes");
    if !block.is_empty() {
        block.shrink_to_fit();
        blocks.push(block);
    }
    Ok((
        Compressed {
            grammar,
            blocks,
            input_bytes,
            field_count,
        },
        before,
        after,
    ))
}

struct BoundedInput<'a> {
    path: &'a Path,
    bytes: u64,
    fields: u64,
    single: bool,
    maximum_field: usize,
    directory: &'a Path,
}

fn bounded_fields(
    matcher: &crate::codec::spooled::RawMatcher,
    lengths: &[u8],
    width: u8,
    source: BoundedInput<'_>,
) -> Result<(Vec<PackedSymbols>, u128), String> {
    use std::io::{Seek, SeekFrom};
    const CHUNK: usize = crate::codec::spooled::CHUNK;
    let mut input = fs::File::open(source.path).map_err(|e| e.to_string())?;
    let mut scan = BufReader::new(fs::File::open(source.path).map_err(|e| e.to_string())?);
    if input.metadata().map_err(|e| e.to_string())?.len() != source.bytes {
        return Err("raw input byte count differs from saved grammar totals".into());
    }
    let mut blocks = Vec::new();
    let mut block = PackedSymbols::with_capacity(width, CHUNK);
    let mut bytes = Vec::new();
    let mut scratch = crate::codec::spooled::RawScratch::default();
    let (mut actual_bytes, mut actual_fields, mut after) = (0u64, 0u64, 0u128);
    let mut next_progress = 16 * 1024 * 1024u64;
    while actual_bytes < source.bytes {
        let mut size = 0usize;
        if source.single {
            size = usize::try_from(source.bytes).map_err(|_| "field exceeds platform size")?;
        } else {
            loop {
                let window = scan.fill_buf().map_err(|e| e.to_string())?;
                if window.is_empty() {
                    break;
                }
                let lf = window.iter().position(|&b| b == b'\n');
                let n = lf.map_or(window.len(), |at| at + 1);
                size = size.checked_add(n).ok_or("field length overflow")?;
                if size > source.maximum_field {
                    return Err("field exceeds explicit reparse bound".into());
                }
                scan.consume(n);
                if lf.is_some() {
                    break;
                }
            }
        }
        if size == 0 || size > source.maximum_field {
            return Err("field exceeds explicit reparse bound or raw input changed".into());
        }
        let emit = |id: u32| -> Result<(), String> {
            after += u128::from(lengths[id as usize]);
            block.push(id);
            // Physical packed chunks need not invent EOFs. BRPAIR01 writing
            // flattens them, so the exact saved stream is unchanged.
            if block.len() == CHUNK {
                blocks.push(std::mem::replace(
                    &mut block,
                    PackedSymbols::with_capacity(width, CHUNK),
                ));
            }
            Ok(())
        };
        if size < CHUNK {
            bytes.resize(size, 0);
            input
                .seek(SeekFrom::Start(actual_bytes))
                .map_err(|e| e.to_string())?;
            input.read_exact(&mut bytes).map_err(|e| e.to_string())?;
            matcher.segment_slice(&bytes, &mut scratch, emit)?;
        } else {
            matcher.segment_bytes_spooled(
                &mut input,
                actual_bytes,
                size,
                source.directory,
                emit,
            )?;
        }
        actual_bytes = actual_bytes
            .checked_add(size as u64)
            .ok_or("byte overflow")?;
        actual_fields += 1;
        if actual_bytes >= next_progress {
            crate::training_progress::phase(
                "reparse/fields",
                0,
                actual_bytes,
                source.bytes,
                "bytes",
            );
            next_progress = actual_bytes.saturating_add(16 * 1024 * 1024);
        }
    }
    if (actual_bytes, actual_fields) != (source.bytes, source.fields) {
        return Err("raw input or field policy differs from saved grammar totals".into());
    }
    if !block.is_empty() {
        block.shrink_to_fit();
        blocks.push(block);
    }
    crate::training_progress::phase("reparse/fields", 0, actual_bytes, source.bytes, "bytes");
    Ok((blocks, after))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bounded_reparse_matches_reference_archive_for_small_and_large_fields() {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "bounded-grammar-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&dir).unwrap();
        let raw = dir.join("raw");
        for (single, bytes) in [
            (false, b"abc\n\nabc\x00\xff".repeat(10)),
            (true, b"abc".repeat(400_000)),
            (true, vec![]),
        ] {
            fs::write(&raw, &bytes).unwrap();
            let fields = if bytes.is_empty() {
                vec![]
            } else if single {
                vec![bytes.clone()]
            } else {
                bytes
                    .split_inclusive(|&b| b == b'\n')
                    .map(<[u8]>::to_vec)
                    .collect()
            };
            for shared in [false, true] {
                let mut grammar = crate::grammar::Grammar::default();
                let ab = grammar.add((b'a' as u32, b'b' as u32)).unwrap();
                grammar.add((ab, b'c' as u32)).unwrap();
                let mut block = PackedSymbols::with_capacity(9, bytes.len() + fields.len());
                for f in &fields {
                    for &b in f {
                        block.push(u32::from(b));
                    }
                    block.push(EOF);
                }
                let source = Compressed {
                    grammar,
                    input_bytes: bytes.len() as u64,
                    field_count: fields.len() as u64,
                    blocks: if block.is_empty() {
                        vec![]
                    } else {
                        vec![block]
                    },
                };
                let copy = || Compressed {
                    grammar: source.grammar.clone(),
                    input_bytes: source.input_bytes,
                    field_count: source.field_count,
                    blocks: source.blocks.clone(),
                };
                let (expected, before, after) =
                    reparse(copy(), &raw, single, shared, 2_000_000).unwrap();
                let (actual, bb, ba) =
                    reparse_bounded(copy(), &raw, single, shared, 2_000_000, &dir).unwrap();
                assert_eq!((bb, ba), (before, after));
                let (mut a, mut b) = (Vec::new(), Vec::new());
                crate::archive_io::write(&expected, &mut a).unwrap();
                crate::archive_io::write(&actual, &mut b).unwrap();
                assert_eq!(a, b);
                assert_eq!(actual.decode().unwrap(), fields);
                assert_eq!(fs::read_dir(&dir).unwrap().count(), 1);
                if bytes.len() > 1 {
                    assert!(reparse_bounded(source, &raw, single, shared, 1, &dir).is_err());
                    assert_eq!(fs::read_dir(&dir).unwrap().count(), 1);
                }
            }
        }
        fs::remove_file(raw).unwrap();
        fs::remove_dir(dir).unwrap();
    }
    #[test]
    fn reparse_can_shift_a_former_merge_boundary_to_an_existing_cheaper_rule() {
        let dir = std::env::temp_dir().join(format!("reparse-boundary-{}", std::process::id()));
        fs::create_dir(&dir).unwrap();
        let raw = dir.join("raw");
        fs::write(&raw, b"abc".repeat(1100)).unwrap();
        for shared in [false, true] {
            let mut grammar = crate::grammar::Grammar::default();
            let ab = grammar.add((b'a' as u32, b'b' as u32)).unwrap();
            let bc = grammar.add((b'b' as u32, b'c' as u32)).unwrap();
            let mut block = PackedSymbols::with_capacity(9, 2201);
            for _ in 0..100 {
                block.push(ab);
                block.push(b'c' as u32);
            }
            for _ in 0..1000 {
                block.push(b'a' as u32);
                block.push(bc);
            }
            block.push(EOF);
            let source = Compressed {
                grammar,
                blocks: vec![block],
                input_bytes: 3300,
                field_count: 1,
            };
            let (next, before, after) = reparse(source, &raw, true, shared, 1 << 20).unwrap();
            assert!(after < before);
            assert_eq!(next.decode().unwrap().concat(), b"abc".repeat(1100));
            assert_eq!(next.grammar.rules.len(), 2);
            assert_eq!(
                (next.grammar.rules[0].left, next.grammar.rules[0].right),
                (b'a' as u32, b'b' as u32)
            );
        }
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn fixed_grammar_reparse_preserves_bytes_fields_and_reduces_fixed_code_cost() {
        let dir = std::env::temp_dir().join(format!("reparse-grammar-{}", std::process::id()));
        fs::create_dir(&dir).unwrap();
        let raw = dir.join("raw");
        let mut bytes = b"aabcabc\x00\xff\n".repeat(8);
        bytes.extend_from_slice(b"\ntail\x80");
        fs::write(&raw, &bytes).unwrap();
        for (shared, single) in [(false, false), (true, false), (false, true), (true, true)] {
            let source_fields = if single {
                vec![bytes.clone()]
            } else {
                bytes
                    .split_inclusive(|&b| b == b'\n')
                    .map(<[u8]>::to_vec)
                    .collect()
            };
            let source = crate::field_engine::compress(
                &source_fields,
                &crate::engine::Config {
                    buffer_bytes: 1 << 20,
                    minimum_count: 2,
                    ..Default::default()
                },
                1,
                64,
                17,
                |_| {},
            )
            .unwrap()
            .training
            .compressed;
            let (next, before, after) = reparse(source, &raw, single, shared, 1 << 20).unwrap();
            assert!(after <= before);
            assert_eq!(next.decode().unwrap(), source_fields);
        }
        fs::remove_dir_all(&dir).unwrap();
    }
}
