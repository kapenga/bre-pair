//! One streaming training-only pass for a small, explicit set of byte tokens.
//! Raw overlapping matches and document support are distinct from parser usage.
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};

const ALPHABET: usize = 257;
const GROUP_CAP: usize = 4096;

struct Matcher {
    next: Vec<[usize; ALPHABET]>,
    outputs: Vec<Vec<usize>>,
}

impl Matcher {
    fn new(patterns: &[Vec<usize>]) -> Self {
        let mut result = Self {
            next: vec![[0; ALPHABET]],
            outputs: vec![Vec::new()],
        };
        for (id, pattern) in patterns.iter().enumerate() {
            let mut state = 0;
            for &symbol in pattern {
                if result.next[state][symbol] == 0 {
                    result.next[state][symbol] = result.next.len();
                    result.next.push([0; ALPHABET]);
                    result.outputs.push(Vec::new());
                }
                state = result.next[state][symbol];
            }
            result.outputs[state].push(id);
        }
        let mut failure = vec![0; result.next.len()];
        let mut queue = VecDeque::new();
        for &child in &result.next[0] {
            if child != 0 {
                queue.push_back(child);
            }
        }
        while let Some(state) = queue.pop_front() {
            for symbol in 0..ALPHABET {
                let child = result.next[state][symbol];
                if child == 0 {
                    result.next[state][symbol] = result.next[failure[state]][symbol];
                } else {
                    failure[child] = result.next[failure[state]][symbol];
                    let inherited = result.outputs[failure[child]].clone();
                    result.outputs[child].extend(inherited);
                    queue.push_back(child);
                }
            }
        }
        result
    }
}

#[derive(Default)]
struct Stats {
    raw: u64,
    documents: u64,
    sources: BTreeSet<u32>,
    groups: BTreeMap<[u8; 32], u64>,
    groups_capped: bool,
    max_document: u64,
    top_source: u32,
    top_line: u64,
    top_offset: u64,
}

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

fn u32_at(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
}

fn u64_at(bytes: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap())
}

fn patterns(mut input: impl Read) -> io::Result<Vec<Vec<usize>>> {
    let mut header = [0; 12];
    input.read_exact(&mut header)?;
    if &header[..8] != b"BRTANOM1" || u32_at(&header, 8) > 4096 {
        return Err(invalid("invalid or oversized pattern set"));
    }
    let mut result = Vec::new();
    for _ in 0..u32_at(&header, 8) {
        let mut width = [0; 4];
        input.read_exact(&mut width)?;
        let width = u32_at(&width, 0) as usize;
        if !(1..=129).contains(&width) {
            return Err(invalid("invalid byte-token pattern length"));
        }
        let mut raw = vec![0; width * 2];
        input.read_exact(&mut raw)?;
        let labels: Vec<_> = raw
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes(b.try_into().unwrap()) as usize)
            .collect();
        if labels.iter().any(|&n| n > 256) || labels[..width - 1].contains(&256) {
            return Err(invalid("invalid EOF placement in pattern"));
        }
        result.push(labels);
    }
    Ok(result)
}

fn audit(
    patterns: &[Vec<usize>],
    mut source: impl Read,
    mut index: impl Read,
) -> io::Result<(Vec<Stats>, u64, u64)> {
    let matcher = Matcher::new(patterns);
    let lengths: Vec<_> = patterns
        .iter()
        .map(|p| p.iter().filter(|&&n| n != 256).count() as u64)
        .collect();
    let mut header = [0; 24];
    index.read_exact(&mut header)?;
    if &header[..8] != b"BREDOC01" || u32_at(&header, 8) != 104 || u32_at(&header, 12) != 0 {
        return Err(invalid("expected the training document index"));
    }
    let documents = u64_at(&header, 16);
    let mut stats: Vec<_> = (0..patterns.len()).map(|_| Stats::default()).collect();
    let mut doc_counts = vec![0u64; patterns.len()];
    let mut first_offsets = vec![0u64; patterns.len()];
    let mut touched = Vec::new();
    let mut buffer = vec![0; 65536];
    let mut consumed = 0;
    for _ in 0..documents {
        let mut record = [0; 104];
        index.read_exact(&mut record)?;
        let source_id = u32_at(&record, 64);
        let source_line = u64_at(&record, 68);
        let offset = u64_at(&record, 76);
        let length = u64_at(&record, 84);
        let expected_fields = u64_at(&record, 92);
        if offset != consumed || length == 0 {
            return Err(invalid("training document offset/length mismatch"));
        }
        let group: [u8; 32] = record[32..64].try_into().unwrap();
        let (mut state, mut fields, mut remaining) = (0usize, 0u64, length);
        let mut last = 0;
        while remaining != 0 {
            let size = remaining.min(buffer.len() as u64) as usize;
            source.read_exact(&mut buffer[..size])?;
            for &byte in &buffer[..size] {
                consumed += 1;
                last = byte;
                for symbol in
                    [byte as usize, 256]
                        .into_iter()
                        .take(if byte == b'\n' { 2 } else { 1 })
                {
                    state = matcher.next[state][symbol];
                    for &id in &matcher.outputs[state] {
                        if doc_counts[id] == 0 {
                            touched.push(id);
                            first_offsets[id] = consumed - lengths[id];
                        }
                        doc_counts[id] += 1;
                    }
                }
                if byte == b'\n' {
                    fields += 1;
                    state = 0;
                }
            }
            remaining -= size as u64;
        }
        if last != b'\n' {
            fields += 1;
            state = matcher.next[state][256];
            for &id in &matcher.outputs[state] {
                if doc_counts[id] == 0 {
                    touched.push(id);
                    first_offsets[id] = consumed - lengths[id];
                }
                doc_counts[id] += 1;
            }
        }
        if fields != expected_fields {
            return Err(invalid("document field count differs"));
        }
        for id in touched.drain(..) {
            let count = std::mem::take(&mut doc_counts[id]);
            let s = &mut stats[id];
            s.raw += count;
            s.documents += 1;
            s.sources.insert(source_id);
            if !s.groups_capped {
                if let Some(existing) = s.groups.get_mut(&group) {
                    *existing += count;
                } else if s.groups.len() < GROUP_CAP {
                    s.groups.insert(group, count);
                } else {
                    s.groups_capped = true;
                }
            }
            if count > s.max_document {
                s.max_document = count;
                s.top_source = source_id;
                s.top_line = source_line;
                s.top_offset = first_offsets[id];
            }
        }
    }
    if source.read(&mut [0])? != 0 || index.read(&mut [0])? != 0 {
        return Err(invalid("trailing corpus or document-index data"));
    }
    Ok((stats, consumed, documents))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args_os().collect();
    if args.len() != 5 {
        return Err("token-document-audit PATTERNS TRAINING.txt TRAINING.docs OUTPUT.json".into());
    }
    let patterns = patterns(BufReader::new(File::open(&args[1])?))?;
    let (stats, bytes, documents) = audit(
        &patterns,
        BufReader::with_capacity(4 * 1024 * 1024, File::open(&args[2])?),
        BufReader::new(File::open(&args[3])?),
    )?;
    let mut output = BufWriter::new(File::create_new(&args[4])?);
    write!(output,"{{\"training_bytes\":{bytes},\"training_documents\":{documents},\"normalized_group_cap\":{GROUP_CAP},\"patterns\":[")?;
    for (id, s) in stats.iter().enumerate() {
        if id != 0 {
            write!(output, ",")?;
        }
        write!(output,"{{\"pattern_id\":{id},\"raw_overlapping_matches\":{},\"training_documents\":{},\"training_sources\":{},\"normalized_groups_lower_bound\":{},\"normalized_group_count_exact\":{},\"max_document_raw_matches\":{},\"top_source_id\":{},\"top_source_line\":{},\"top_match_training_offset\":{}",
            s.raw,s.documents,s.sources.len(),s.groups.len(),!s.groups_capped,s.max_document,s.top_source,s.top_line,s.top_offset)?;
        if !s.groups_capped {
            let max_group = s.groups.values().max().copied().unwrap_or(0);
            write!(output, ",\"max_normalized_group_raw_matches\":{max_group}")?;
        }
        write!(output, "}}")?;
    }
    writeln!(output, "]}}")?;
    output.flush()?;
    println!(
        "{{\"training_bytes\":{bytes},\"training_documents\":{documents},\"patterns\":{}}}",
        stats.len()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn index(payloads: &[&[u8]], groups: &[u8]) -> Vec<u8> {
        let mut data = b"BREDOC01".to_vec();
        data.extend(104u32.to_le_bytes());
        data.extend(0u32.to_le_bytes());
        data.extend((payloads.len() as u64).to_le_bytes());
        let mut offset = 0u64;
        for (i, text) in payloads.iter().enumerate() {
            let mut record = [0; 104];
            record[32..64].fill(groups[i]);
            record[64..68].copy_from_slice(&(i as u32).to_le_bytes());
            record[68..76].copy_from_slice(&1u64.to_le_bytes());
            record[76..84].copy_from_slice(&offset.to_le_bytes());
            record[84..92].copy_from_slice(&(text.len() as u64).to_le_bytes());
            let fields = text.iter().filter(|&&b| b == b'\n').count() as u64
                + u64::from(!text.ends_with(b"\n"));
            record[92..100].copy_from_slice(&fields.to_le_bytes());
            data.extend(record);
            offset += text.len() as u64;
        }
        data
    }

    #[test]
    fn overlapping_matches_document_groups_eof_and_offsets_are_separate() {
        let docs = [&b"aaaaa\n"[..], &b"aa\naa\n"[..], &b"ab"[..]];
        let bytes = docs.concat();
        let patterns = vec![vec![97, 97], vec![97, 10, 256], vec![10, 97], vec![98, 256]];
        let (s, count, documents) =
            audit(&patterns, &bytes[..], &index(&docs, &[1, 1, 2])[..]).unwrap();
        assert_eq!((count, documents), (14, 3));
        assert_eq!((s[0].raw, s[0].documents, s[0].groups.len()), (6, 2, 1));
        assert_eq!(s[0].max_document, 4);
        assert_eq!(s[0].top_offset, 0);
        assert_eq!(s[0].sources.len(), 2);
        assert_eq!(s[1].raw, 3);
        assert_eq!(s[2].raw, 0); // Cannot cross a physical field boundary.
        assert_eq!(s[3].raw, 1); // EOF after the non-LF final field.
    }
}
