//! Self-contained, bounded byte-rank transport for file/column comparisons.
//! It retains the source's compressed dictionary and every original rank.
//! Chunk headers/indexes are per bounded IO chunk, never per field.
use super::{error, number, parse_metadata, MetadataSpec};
use crate::{
    atomic_output::Output,
    bytecode::{ByteCodes, Expansions},
    huffman::Codebook,
    huffman_archive::{self, Block, PreparedDecoder, Totals},
    huffman_dictionary::Dictionary,
    tokenizer::fingerprint,
};
use std::{
    fs::{File, OpenOptions},
    io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write},
    path::Path,
    sync::Arc,
    thread,
};

const MAGIC: &[u8; 8] = b"BRBYTEA1";
const INDEX: &[u8; 8] = b"BRBYIDX1";
const HEADER: usize = 80;
const CHUNK: usize = 8 << 20;
const BLOCK_HEADER: usize = 40;
const ARENA: usize = 1 << 30;

enum Source {
    Shared(super::Reader<BufReader<File>>),
    Ordinary(huffman_archive::Reader<BufReader<File>>),
}
impl Source {
    fn block(&mut self, i: usize) -> Result<Block, String> {
        match self {
            Self::Shared(r) => r.block(i),
            Self::Ordinary(r) => r.block(i),
        }
    }
}

fn prefix(path: &Path) -> Result<Vec<u8>, String> {
    let mut f = File::open(path).map_err(error)?;
    let mut h = [0; 128];
    f.read_exact(&mut h[..96]).map_err(error)?;
    let shared = super::is_format(h[..8].try_into().unwrap());
    let head = if shared { 128 } else { 96 };
    if shared {
        f.read_exact(&mut h[96..]).map_err(error)?;
    } else if &h[..8] != b"BRHUFF02" {
        return Err("byte transport requires a shared static Huffman table".into());
    }
    let file_size = f.metadata().map_err(error)?.len();
    let length = number(&h, 48)
        .checked_add(if shared {
            0
        } else {
            u32::from_le_bytes(h[84..88].try_into().unwrap()) as u64
        })
        .and_then(|n| n.checked_add(head as u64))
        .filter(|&n| n <= (1 << 30) && n <= file_size)
        .ok_or("invalid byte-transport model length")? as usize;
    let mut bytes = vec![0; length];
    bytes[..head].copy_from_slice(&h[..head]);
    f.read_exact(&mut bytes[head..]).map_err(error)?;
    Ok(bytes)
}

fn model(bytes: &[u8]) -> Result<(PreparedDecoder, Option<Arc<Codebook>>), String> {
    if bytes.len() < 96 {
        return Err("truncated byte-transport model".into());
    }
    let dlen = usize::try_from(number(bytes, 48)).map_err(|_| "model too large")?;
    if super::is_format(bytes[..8].try_into().unwrap()) {
        if bytes.len() < 128
            || number(bytes, 8) > 1
            || fingerprint(&bytes[..120]) != number(bytes, 120)
            || dlen.checked_add(128) != Some(bytes.len())
            || fingerprint(&bytes[128..]) != number(bytes, 112)
        {
            return Err("invalid shared byte-transport model".into());
        }
        let (graph, book, _, _) = parse_metadata(
            &bytes[128..],
            &MetadataSpec {
                ai: number(bytes, 8) != 0,
                huffman_rule_lengths: &bytes[..8] != super::MAGIC,
                run_rule_lengths: &bytes[..8] == super::RUN_LENGTHS_MAGIC,
                bits: number(bytes, 56),
                refs: number(bytes, 64),
                desc: number(bytes, 72),
                table: number(bytes, 80),
                leaves: number(bytes, 88),
            },
        )?;
        let book = book.map(Arc::new);
        let mut decoder = PreparedDecoder::from_graph(&graph, book.clone(), false, ARENA)?;
        decoder.prepare_field_references(&graph);
        Ok((decoder, book))
    } else {
        let table = u32::from_le_bytes(bytes[84..88].try_into().unwrap()) as usize;
        let end = 96usize.checked_add(dlen).ok_or("model length overflow")?;
        if &bytes[..8] != b"BRHUFF02"
            || bytes[8] > 1
            || bytes[9..12] != [0; 3]
            || u32::from_le_bytes(bytes[12..16].try_into().unwrap()) != 96
            || end.checked_add(table) != Some(bytes.len())
            || fingerprint(&bytes[96..end]) != number(bytes, 72)
            || fingerprint(&bytes[end..]) != number(bytes, 88)
        {
            return Err("invalid ordinary byte-transport model".into());
        }
        let dictionary = Dictionary::from_bytes(&bytes[96..end], bytes[8] != 0)?;
        let used = u32::from_le_bytes(bytes[80..84].try_into().unwrap()) as usize;
        let book = if used == 0 {
            None
        } else {
            Some(Arc::new(Codebook::from_bytes(
                &bytes[end..],
                used,
                dictionary.expansions.len(),
            )?))
        };
        let mut decoder = PreparedDecoder::from_graph(&dictionary, book.clone(), false, ARENA)?;
        decoder.prepare_field_references(&dictionary);
        Ok((decoder, book))
    }
}

struct Writer {
    file: BufWriter<File>,
    data: Vec<u8>,
    totals: Totals,
    current: Totals,
    index: Vec<(u64, u64)>,
}
impl Writer {
    fn flush(&mut self) -> Result<(), String> {
        if self.current.symbols == 0 {
            return Ok(());
        }
        self.index.push((
            self.file.stream_position().map_err(error)?,
            self.totals.bytes,
        ));
        for value in [
            self.data.len() as u64,
            self.current.bytes,
            self.current.fields,
            self.current.symbols,
            fingerprint(&self.data),
        ] {
            self.file.write_all(&value.to_le_bytes()).map_err(error)?;
        }
        self.file.write_all(&self.data).map_err(error)?;
        self.totals.add(self.current)?;
        self.data.clear();
        self.current = Totals::default();
        Ok(())
    }
    fn push(
        &mut self,
        rank: u32,
        codes: &ByteCodes,
        references: &[(u32, u32)],
    ) -> Result<(), String> {
        let flags = references
            .get(rank as usize)
            .ok_or("undefined transcode rank")?
            .1;
        let bytes = u64::from(flags & 0x7fff_ffff);
        if bytes > ARENA as u64 {
            return Err("one symbol exceeds the byte-transport cache bound".into());
        }
        if self.current.bytes + bytes > CHUNK as u64 || self.data.len() + 4 > CHUNK {
            self.flush()?;
        }
        codes.encode_rank(rank, &mut self.data)?;
        self.current.bytes += bytes;
        self.current.fields += u64::from(flags >> 31);
        self.current.symbols += 1;
        Ok(())
    }
}

/// Recode saved ranks; construction costs must also include discovery/cleanup.
pub fn pack(input: &Path, output: &Path) -> Result<Totals, String> {
    let metadata = prefix(input)?;
    let reader = BufReader::new(File::open(input).map_err(error)?);
    let (mut source, expected, blocks) = if super::is_format(metadata[..8].try_into().unwrap()) {
        let r = super::Reader::new(reader)?;
        let (totals, blocks) = (r.stats.totals, r.stats.blocks);
        (Source::Shared(r), totals, blocks)
    } else {
        let r = huffman_archive::Reader::new(reader)?;
        let (totals, blocks) = (r.totals, r.index.len());
        (Source::Ordinary(r), totals, blocks)
    };
    let (decoder, book) = model(&metadata)?;
    let codes = match &book {
        Some(b) => ByteCodes::from_book(b)?,
        None => ByteCodes::with_slots(1, [256, 0, 0, 0])?,
    };
    let pending = Output::new(output)?;
    let mut writer = Writer {
        file: BufWriter::with_capacity(1 << 20, File::create_new(pending.path()).map_err(error)?),
        data: Vec::with_capacity(CHUNK),
        totals: Totals::default(),
        current: Totals::default(),
        index: Vec::new(),
    };
    writer
        .file
        .write_all(&[0; HEADER])
        .and_then(|_| writer.file.write_all(&metadata))
        .and_then(|_| writer.file.write_all(&codes.metadata()))
        .map_err(error)?;
    if blocks != 0 {
        let (_, references) = decoder.bytecode_tables()?;
        for i in 0..blocks {
            let block = source.block(i)?;
            block
                .book
                .decode_ranks(&block.payload, block.bits, block.totals.symbols, |rank| {
                    writer.push(rank, &codes, references)
                })?;
        }
    }
    writer.flush()?;
    if (
        writer.totals.bytes,
        writer.totals.fields,
        writer.totals.symbols,
    ) != (expected.bytes, expected.fields, expected.symbols)
    {
        return Err("byte transcode totals differ".into());
    }
    let offset = writer.file.stream_position().map_err(error)?;
    let mut index = INDEX.to_vec();
    for &(input, output) in &writer.index {
        index.extend(input.to_le_bytes());
        index.extend(output.to_le_bytes());
    }
    index.extend(fingerprint(&index).to_le_bytes());
    writer.file.write_all(&index).map_err(error)?;
    let mut header = [0; HEADER];
    header[..8].copy_from_slice(MAGIC);
    for (at, value) in [
        (8, metadata.len() as u64),
        (16, writer.index.len() as u64),
        (24, offset),
        (32, expected.bytes),
        (40, expected.fields),
        (48, expected.symbols),
        (56, fingerprint(&metadata)),
    ] {
        header[at..at + 8].copy_from_slice(&value.to_le_bytes());
    }
    let check = fingerprint(&header[..72]);
    header[72..].copy_from_slice(&check.to_le_bytes());
    writer
        .file
        .seek(SeekFrom::Start(0))
        .and_then(|_| writer.file.write_all(&header))
        .and_then(|_| writer.file.flush())
        .map_err(error)?;
    drop(writer);
    pending.commit()?;
    Ok(expected)
}

#[derive(Clone, Copy)]
struct Entry {
    offset: u64,
    output: u64,
    payload: usize,
    bytes: usize,
    fields: u64,
    symbols: u64,
    checksum: u64,
}

/// Bounded input/output buffers per worker; dictionary and rank table are shared.
pub fn decode(input: &Path, output: &Path, workers: usize) -> Result<Totals, String> {
    if !matches!(workers, 1 | 4) {
        return Err("byte decoding supports one or four workers".into());
    }
    let mut file = BufReader::new(File::open(input).map_err(error)?);
    let size = file.get_ref().metadata().map_err(error)?.len();
    let mut h = [0; HEADER];
    file.read_exact(&mut h).map_err(error)?;
    let (model_len, blocks, index_offset) = (number(&h, 8), number(&h, 16), number(&h, 24));
    if &h[..8] != MAGIC
        || number(&h, 64) != 0
        || fingerprint(&h[..72]) != number(&h, 72)
        || model_len > 1 << 30
        || model_len
            .checked_add((HEADER + 16) as u64)
            .is_none_or(|n| n > index_offset)
        || blocks
            .checked_mul(16)
            .and_then(|n| n.checked_add(16))
            .and_then(|n| n.checked_add(index_offset))
            != Some(size)
    {
        return Err("invalid byte archive header or sizes".into());
    }
    let expected = Totals {
        bytes: number(&h, 32),
        fields: number(&h, 40),
        symbols: number(&h, 48),
        tokens: 0,
    };
    let mut metadata = vec![0; model_len as usize];
    file.read_exact(&mut metadata).map_err(error)?;
    if fingerprint(&metadata) != number(&h, 56) {
        return Err("byte model checksum differs".into());
    }
    let (decoder, book) = model(&metadata)?;
    let mut coding = [0; 16];
    file.read_exact(&mut coding).map_err(error)?;
    let codes = ByteCodes::from_metadata(book.as_ref().map_or(1, |b| b.symbols.len()), &coding)?;
    file.seek(SeekFrom::Start(index_offset)).map_err(error)?;
    let mut raw_index = vec![0; (size - index_offset) as usize];
    file.read_exact(&mut raw_index).map_err(error)?;
    if &raw_index[..8] != INDEX
        || fingerprint(&raw_index[..raw_index.len() - 8]) != number(&raw_index, raw_index.len() - 8)
    {
        return Err("byte index checksum differs".into());
    }
    let mut entries = Vec::with_capacity(blocks as usize);
    let mut totals = Totals::default();
    let mut next = (HEADER + 16) as u64 + model_len;
    for raw in raw_index[8..raw_index.len() - 8].chunks_exact(16) {
        if number(raw, 0) != next || number(raw, 8) != totals.bytes {
            return Err("nonmonotonic byte chunk index".into());
        }
        file.seek(SeekFrom::Start(next)).map_err(error)?;
        let mut h = [0; BLOCK_HEADER];
        file.read_exact(&mut h).map_err(error)?;
        let (payload, bytes, fields, symbols) =
            (number(&h, 0), number(&h, 8), number(&h, 16), number(&h, 24));
        if payload == 0
            || payload > CHUNK as u64
            || (bytes > CHUNK as u64 && (symbols != 1 || payload > 4 || bytes > ARENA as u64))
            || symbols == 0
            || symbols > payload
            || fields > symbols
        {
            return Err("invalid byte chunk bounds".into());
        }
        entries.push(Entry {
            offset: next + BLOCK_HEADER as u64,
            output: totals.bytes,
            payload: payload as usize,
            bytes: bytes as usize,
            fields,
            symbols,
            checksum: number(&h, 32),
        });
        next = next
            .checked_add(BLOCK_HEADER as u64)
            .and_then(|n| n.checked_add(payload))
            .ok_or("chunk overflow")?;
        if next > index_offset {
            return Err("byte chunk exceeds payload".into());
        }
        totals.add(Totals {
            bytes,
            fields,
            symbols,
            tokens: 0,
        })?;
    }
    if totals != expected || next != index_offset {
        return Err("byte chunk totals differ".into());
    }
    let pending = Output::new(output)?;
    let output_file = File::create_new(pending.path()).map_err(error)?;
    output_file.set_len(expected.bytes).map_err(error)?;
    drop(output_file);
    if !entries.is_empty() {
        let (arena, references) = decoder.bytecode_tables()?;
        let expansions = Expansions::new(arena, references)?;
        thread::scope(|scope| -> Result<(), String> {
            let mut jobs = Vec::new();
            for worker in 0..workers {
                let begin = entries.len() * worker / workers;
                let end = entries.len() * (worker + 1) / workers;
                let entries = &entries;
                let expansions = &expansions;
                let codes = &codes;
                let destination = pending.path();
                jobs.push(scope.spawn(move || -> Result<(), String> {
                    let mut source =
                        BufReader::with_capacity(1 << 20, File::open(input).map_err(error)?);
                    let mut target = BufWriter::with_capacity(
                        1 << 20,
                        OpenOptions::new()
                            .write(true)
                            .open(destination)
                            .map_err(error)?,
                    );
                    let mut input_buffer = Vec::new();
                    let mut output_buffer = Vec::new();
                    for (i, entry) in entries.iter().enumerate().take(end).skip(begin) {
                        input_buffer.resize(entry.payload + 3, 0);
                        source.seek(SeekFrom::Start(entry.offset)).map_err(error)?;
                        source
                            .read_exact(&mut input_buffer[..entry.payload])
                            .map_err(error)?;
                        if fingerprint(&input_buffer[..entry.payload]) != entry.checksum {
                            return Err("byte payload checksum differs".into());
                        }
                        if entry.bytes > CHUNK {
                            let (rank, consumed) =
                                codes.decode_rank(&input_buffer[..entry.payload])?;
                            let (bytes, ended) = expansions.rank_bytes(rank)?;
                            if consumed != entry.payload
                                || bytes.len() != entry.bytes
                                || u64::from(ended) != entry.fields
                                || (i + 1 == entries.len() && !ended)
                            {
                                return Err("oversized byte symbol totals differ".into());
                            }
                            target
                                .seek(SeekFrom::Start(entry.output))
                                .and_then(|_| target.write_all(bytes))
                                .map_err(error)?;
                            continue;
                        }
                        output_buffer.resize(entry.bytes + 16, 0);
                        let decoded = expansions.byte_block_checked(
                            codes,
                            &input_buffer,
                            0..entry.payload,
                            &mut output_buffer,
                            entry.bytes,
                        )?;
                        if decoded.bytes != entry.bytes
                            || decoded.fields != entry.fields
                            || decoded.symbols != entry.symbols
                            || (i + 1 == entries.len() && !decoded.ended)
                        {
                            return Err("decoded byte chunk totals differ".into());
                        }
                        target
                            .seek(SeekFrom::Start(entry.output))
                            .and_then(|_| target.write_all(&output_buffer[..entry.bytes]))
                            .map_err(error)?;
                    }
                    target.flush().map_err(error)
                }));
            }
            for job in jobs {
                job.join().map_err(|_| "byte decoder worker panicked")??;
            }
            Ok(())
        })?;
    }
    pending.commit()?;
    Ok(expected)
}
