//! BRHUFF02 uses one static canonical table for all indexed blocks. BRHUFF01
//! with block-local tables remains readable. Both have self-contained dictionaries
//! and a u64 index. Fixed-width BRTOKS01 and BRPAIR01 remain unchanged.
use crate::{
    huffman::{Codebook, Encoding},
    huffman_dictionary::{Arena, DecodeGraph, Dictionary},
    tokenizer::{fingerprint, Tokenizer},
};
use std::{
    fs::File,
    io::{BufRead, BufReader, BufWriter, Read, Seek, SeekFrom, Write},
    path::Path,
    sync::Arc,
};

pub const DEFAULT_BLOCK_SYMBOLS: usize = 1 << 20;
pub const MAX_BLOCK_SYMBOLS: usize = 1 << 24;
const HEADER: u64 = 96;
const BLOCK_HEADER: u64 = 56;
const ARENA_BUDGET: usize = 64 << 20;

fn error(e: std::io::Error) -> String {
    e.to_string()
}
fn read_array<const N: usize>(r: &mut impl Read) -> Result<[u8; N], String> {
    let mut b = [0; N];
    r.read_exact(&mut b).map_err(error)?;
    Ok(b)
}
fn number(b: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(b[at..at + 8].try_into().unwrap())
}
fn hash(parts: &[&[u8]]) -> u64 {
    parts
        .iter()
        .flat_map(|p| p.iter())
        .fold(0xcbf29ce484222325u64, |h, &b| {
            (h ^ u64::from(b)).wrapping_mul(0x100000001b3)
        })
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Totals {
    pub bytes: u64,
    pub fields: u64,
    pub tokens: u64,
    pub symbols: u64,
}
impl Totals {
    pub(crate) fn add_symbol<G: DecodeGraph>(&mut self, d: &G, id: u32) -> Result<bool, String> {
        let e = d
            .expansions()
            .get(id as usize)
            .ok_or("undefined codec symbol")?;
        self.bytes = self
            .bytes
            .checked_add(e.bytes)
            .ok_or("byte count overflow")?;
        self.tokens = self
            .tokens
            .checked_add(e.terminals)
            .ok_or("token count overflow")?;
        self.fields = self
            .fields
            .checked_add(u64::from(e.ends_field))
            .ok_or("field count overflow")?;
        self.symbols = self.symbols.checked_add(1).ok_or("symbol count overflow")?;
        Ok(e.ends_field)
    }
    pub(crate) fn add(&mut self, other: Self) -> Result<(), String> {
        self.bytes = self
            .bytes
            .checked_add(other.bytes)
            .ok_or("byte count overflow")?;
        self.tokens = self
            .tokens
            .checked_add(other.tokens)
            .ok_or("token count overflow")?;
        self.fields = self
            .fields
            .checked_add(other.fields)
            .ok_or("field count overflow")?;
        self.symbols = self
            .symbols
            .checked_add(other.symbols)
            .ok_or("symbol count overflow")?;
        Ok(())
    }
    pub(crate) fn exceeds(&self, other: &Self) -> bool {
        self.bytes > other.bytes
            || self.fields > other.fields
            || self.tokens > other.tokens
            || self.symbols > other.symbols
    }
}
#[derive(Clone, Copy, Debug)]
pub struct IndexEntry {
    pub offset: u64,
    pub before: Totals,
}
impl IndexEntry {
    fn bytes(self) -> [u8; 40] {
        let mut b = [0; 40];
        for (at, n) in [
            self.offset,
            self.before.symbols,
            self.before.bytes,
            self.before.tokens,
            self.before.fields,
        ]
        .into_iter()
        .enumerate()
        {
            b[at * 8..at * 8 + 8].copy_from_slice(&n.to_le_bytes());
        }
        b
    }
}

pub struct Encoder<'a, W> {
    out: W,
    dictionary: &'a Dictionary,
    dictionary_len: u64,
    dictionary_hash: u64,
    block_size: usize,
    ids: Vec<u32>,
    counts: Vec<u64>,
    index: Vec<IndexEntry>,
    pub totals: Totals,
    ended: bool,
    shared: Option<(u32, u32, u64)>,
    encoding: Option<Encoding>,
}
impl<'a, W: Write + Seek> Encoder<'a, W> {
    /// Legacy block-static writer, retained for compatibility experiments.
    pub fn new(out: W, dictionary: &'a Dictionary, block_size: usize) -> Result<Self, String> {
        Self::create(out, dictionary, block_size, None, None)
    }
    /// One fixed table for the complete stream. Counts are encoder-only input;
    /// the archive stores code lengths and IDs, never adaptive frequencies.
    pub fn with_counts(
        out: W,
        dictionary: &'a Dictionary,
        block_size: usize,
        counts: &[u64],
    ) -> Result<Self, String> {
        if counts.len() != dictionary.expansions.len() {
            return Err("global counts must match the dictionary".into());
        }
        Self::create(out, dictionary, block_size, Some(counts), None)
    }
    /// Reuse the exact training-derived canonical table, without recounting
    /// the new input or constructing a new Huffman tree.
    pub fn with_codebook(
        out: W,
        dictionary: &'a Dictionary,
        block_size: usize,
        book: &Codebook,
    ) -> Result<Self, String> {
        if book
            .symbols
            .iter()
            .any(|&id| id as usize >= dictionary.expansions.len())
        {
            return Err("static table exceeds dictionary".into());
        }
        Self::create(out, dictionary, block_size, None, Some(book))
    }
    fn create(
        mut out: W,
        dictionary: &'a Dictionary,
        block_size: usize,
        counts: Option<&[u64]>,
        prepared: Option<&Codebook>,
    ) -> Result<Self, String> {
        if !(1..=MAX_BLOCK_SYMBOLS).contains(&block_size) {
            return Err("block symbols must be 1..=16777216".into());
        }
        if out.stream_position().map_err(error)? != 0 {
            return Err("archive output must begin at offset zero".into());
        }
        let bytes = dictionary.to_bytes();
        out.write_all(&[0; HEADER as usize]).map_err(error)?;
        out.write_all(&bytes).map_err(error)?;
        let (shared, encoding) = if counts.is_some() || prepared.is_some() {
            let generated = counts
                .filter(|c| c.iter().any(|&n| n != 0))
                .map(Codebook::from_counts)
                .transpose()?;
            let book = prepared.or(generated.as_ref());
            let table = book.map_or_else(Vec::new, Codebook::to_bytes);
            let used = book.map_or(0, |b| b.symbols.len() as u32);
            let table_len =
                u32::try_from(table.len()).map_err(|_| "static table exceeds u32 length")?;
            out.write_all(&table).map_err(error)?;
            (
                Some((used, table_len, fingerprint(&table))),
                book.map(Codebook::encoding),
            )
        } else {
            (None, None)
        };
        Ok(Self {
            out,
            dictionary,
            dictionary_len: bytes.len() as u64,
            dictionary_hash: fingerprint(&bytes),
            block_size,
            ids: Vec::with_capacity(block_size),
            counts: if shared.is_some() {
                vec![]
            } else {
                vec![0; dictionary.expansions.len()]
            },
            index: Vec::new(),
            totals: Totals::default(),
            ended: true,
            shared,
            encoding,
        })
    }
    pub fn push(&mut self, id: u32) -> Result<(), String> {
        if id as usize >= self.dictionary.expansions.len() {
            return Err("undefined codec symbol".into());
        }
        self.ids.push(id);
        if self.shared.is_none() {
            self.counts[id as usize] += 1;
        }
        if self.ids.len() == self.block_size {
            self.flush_block()?;
        }
        Ok(())
    }
    /// Append an ordered batch without repeating buffer checks per symbol.
    pub fn extend(&mut self, mut ids: &[u32]) -> Result<(), String> {
        if ids
            .iter()
            .any(|&id| id as usize >= self.dictionary.expansions.len())
        {
            return Err("undefined codec symbol".into());
        }
        while !ids.is_empty() {
            let n = ids.len().min(self.block_size - self.ids.len());
            if self.shared.is_none() {
                for &id in &ids[..n] {
                    self.counts[id as usize] += 1;
                }
            }
            self.ids.extend_from_slice(&ids[..n]);
            ids = &ids[n..];
            if self.ids.len() == self.block_size {
                self.flush_block()?;
            }
        }
        Ok(())
    }
    fn flush_block(&mut self) -> Result<(), String> {
        if self.ids.is_empty() {
            return Ok(());
        }
        // At most MAX_BLOCK_SYMBOLS expansions, each with u64 lengths. Wide
        // sums fit u128 and let us check overflow once per block, preserving
        // full u64 byte/token totals without per-symbol checked arithmetic.
        let (mut bytes, mut tokens, mut fields) = (0u128, 0u128, 0u64);
        for &id in &self.ids {
            let e = self.dictionary.expansions[id as usize];
            bytes += u128::from(e.bytes);
            tokens += u128::from(e.terminals);
            fields += u64::from(e.ends_field);
            self.ended = e.ends_field;
        }
        let totals = Totals {
            bytes: u64::try_from(bytes).map_err(|_| "byte count overflow")?,
            tokens: u64::try_from(tokens).map_err(|_| "token count overflow")?,
            fields,
            symbols: self.ids.len() as u64,
        };
        let (table, used, payload, bits) = if self.shared.is_some() {
            let (payload, bits) = self
                .encoding
                .as_ref()
                .ok_or("empty static table cannot encode symbols")?
                .encode(&self.ids)?;
            (Vec::new(), 0u32, payload, bits)
        } else {
            let book = Codebook::from_counts(&self.counts)?;
            let (payload, bits) = book.encode(&self.ids)?;
            (book.to_bytes(), book.symbols.len() as u32, payload, bits)
        };
        let mut h = [0u8; BLOCK_HEADER as usize];
        h[..4].copy_from_slice(&(self.ids.len() as u32).to_le_bytes());
        h[4..8].copy_from_slice(&used.to_le_bytes());
        for (at, n) in [
            (8, bits),
            (16, totals.bytes),
            (24, totals.fields),
            (32, totals.tokens),
        ] {
            h[at..at + 8].copy_from_slice(&n.to_le_bytes());
        }
        h[40..44].copy_from_slice(&(table.len() as u32).to_le_bytes());
        let check = hash(&[&h[..48], &table, &payload]);
        h[48..56].copy_from_slice(&check.to_le_bytes());
        self.index.push(IndexEntry {
            offset: self.out.stream_position().map_err(error)?,
            before: self.totals,
        });
        self.out
            .write_all(&h)
            .and_then(|_| self.out.write_all(&table))
            .and_then(|_| self.out.write_all(&payload))
            .map_err(error)?;
        self.totals.add(totals)?;
        self.ids.clear();
        self.counts.fill(0);
        Ok(())
    }
    pub fn finish(mut self, expected_bytes: u64, expected_fields: u64) -> Result<Totals, String> {
        self.flush_block()?;
        if self.totals.bytes != expected_bytes
            || self.totals.fields != expected_fields
            || !self.ended
        {
            return Err("Huffman input totals or final EOF differ".into());
        }
        let index_offset = self.out.stream_position().map_err(error)?;
        self.out.write_all(b"BRHIDX01").map_err(error)?;
        let mut index_hash = 0xcbf29ce484222325u64;
        for entry in &self.index {
            let bytes = entry.bytes();
            for &b in &bytes {
                index_hash = (index_hash ^ u64::from(b)).wrapping_mul(0x100000001b3);
            }
            self.out.write_all(&bytes).map_err(error)?;
        }
        self.out
            .write_all(&index_hash.to_le_bytes())
            .map_err(error)?;
        self.out.seek(SeekFrom::Start(0)).map_err(error)?;
        let mut h = [0u8; HEADER as usize];
        h[..8].copy_from_slice(if self.shared.is_some() {
            b"BRHUFF02"
        } else {
            b"BRHUFF01"
        });
        h[8..12].copy_from_slice(&u32::from(self.dictionary.ai_tokens).to_le_bytes());
        h[12..16].copy_from_slice(&(HEADER as u32).to_le_bytes());
        for (at, n) in [
            (16, self.totals.bytes),
            (24, self.totals.fields),
            (32, self.totals.tokens),
            (40, self.totals.symbols),
            (48, self.dictionary_len),
            (56, self.index.len() as u64),
            (64, index_offset),
            (72, self.dictionary_hash),
        ] {
            h[at..at + 8].copy_from_slice(&n.to_le_bytes());
        }
        if let Some((used, len, hash)) = self.shared {
            h[80..84].copy_from_slice(&used.to_le_bytes());
            h[84..88].copy_from_slice(&len.to_le_bytes());
            h[88..96].copy_from_slice(&hash.to_le_bytes());
        }
        self.out
            .write_all(&h)
            .and_then(|_| self.out.flush())
            .map_err(error)?;
        Ok(self.totals)
    }
}

pub struct Block {
    pub book: Arc<Codebook>,
    pub payload: Vec<u8>,
    pub bits: u64,
    pub totals: Totals,
}
/// Reusable arena and static canonical references. Use only with the blocks
/// and dictionary from the Reader that prepared this decoder.
pub struct PreparedDecoder {
    arena: Arena,
    book: Option<Arc<Codebook>>,
    references: Option<Vec<(u32, u32)>>,
    token_ids: bool,
}
impl PreparedDecoder {
    /// Prepare the production decoder for independently framed records sharing
    /// an immutable dictionary/table. Use the same dictionary for all blocks;
    /// each block must hold an Arc clone of this exact codebook.
    pub fn with_codebook(
        dictionary: &Dictionary,
        book: Arc<Codebook>,
        token_ids: bool,
        arena_budget: usize,
    ) -> Result<Self, String> {
        if book
            .symbols
            .iter()
            .any(|&id| id as usize >= dictionary.expansions.len())
        {
            return Err("Huffman table refers to undefined codec symbols".into());
        }
        Self::from_graph(dictionary, Some(book), token_ids, arena_budget)
    }
    pub(crate) fn from_graph<G: DecodeGraph>(
        dictionary: &G,
        book: Option<Arc<Codebook>>,
        token_ids: bool,
        arena_budget: usize,
    ) -> Result<Self, String> {
        let arena = Arena::from_graph(dictionary, token_ids, arena_budget)?;
        let references = book
            .as_ref()
            .and_then(|b| arena.ranked_references(dictionary, &b.symbols));
        Ok(Self {
            arena,
            book,
            references,
            token_ids,
        })
    }
    pub fn arena_bytes(&self) -> usize {
        self.arena.allocated_bytes()
    }
    pub(crate) fn uses_codebook(&self, book: &Arc<Codebook>) -> bool {
        self.book.as_ref().is_some_and(|own| Arc::ptr_eq(own, book))
    }
    /// Decode a byte-bounded record until its EOF-ending symbol. No stored
    /// byte, bit or symbol counters are required. The dictionary must be the
    /// one used to prepare this decoder. Returns (actual bits, actual symbols).
    /// The caller supplies an output limit; on failure, discard partial output.
    pub fn decode_field(
        &self,
        dictionary: &Dictionary,
        payload: &[u8],
        maximum_output: usize,
        out: &mut Vec<u8>,
    ) -> Result<(u64, u64), String> {
        self.decode_graph_field(dictionary, payload, maximum_output, out)
    }
    pub(crate) fn decode_graph_field<G: DecodeGraph>(
        &self,
        dictionary: &G,
        payload: &[u8],
        maximum_output: usize,
        out: &mut Vec<u8>,
    ) -> Result<(u64, u64), String> {
        if self.token_ids {
            return Err("EOF field API requires byte output".into());
        }
        let book = self
            .book
            .as_ref()
            .ok_or("EOF field API requires a static table")?;
        out.clear();
        if let Some(references) = &self.references {
            book.decode_field_ranks(payload, |rank| {
                let (offset, flags) = references[rank as usize];
                let length = (flags & 0x7fff_ffff) as usize;
                if length > maximum_output - out.len() {
                    return Err("field exceeds output limit".into());
                }
                out.extend_from_slice(
                    &self.arena.data()[offset as usize..offset as usize + length],
                );
                Ok(flags >> 31 != 0)
            })
        } else {
            let mut stack = Vec::new();
            book.decode_field_ranks(payload, |rank| {
                let id = book.symbols[rank as usize];
                let expansion = dictionary.expansions()[id as usize];
                if expansion.bytes > (maximum_output - out.len()) as u64 {
                    return Err("field exceeds output limit".into());
                }
                self.arena.emit_graph(dictionary, id, &mut stack, |bytes| {
                    out.extend_from_slice(bytes);
                    Ok(())
                })?;
                Ok(expansion.ends_field)
            })
        }
    }
    pub fn decode_block(
        &self,
        dictionary: &Dictionary,
        block: &Block,
        out: &mut impl Write,
    ) -> Result<bool, String> {
        self.decode_graph_block(dictionary, block, out)
    }
    pub(crate) fn decode_graph_block<G: DecodeGraph>(
        &self,
        dictionary: &G,
        block: &Block,
        out: &mut impl Write,
    ) -> Result<bool, String> {
        type MemoryReader = Reader<std::io::Cursor<Vec<u8>>>;
        if let Some(book) = &self.book {
            if !Arc::ptr_eq(book, &block.book) {
                return Err("block uses a different static table".into());
            }
            MemoryReader::decode_prepared(
                dictionary,
                &self.arena,
                block,
                self.references.as_deref(),
                out,
            )
        } else {
            let references = self
                .arena
                .ranked_references(dictionary, &block.book.symbols);
            MemoryReader::decode_prepared(
                dictionary,
                &self.arena,
                block,
                references.as_deref(),
                out,
            )
        }
    }
}
/// Recycled encoded bytes. Reading does no Huffman table construction; the
/// decoder stage verifies the checksum and prepares the canonical table.
pub(crate) struct EncodedBlock {
    header: [u8; 56],
    table: Vec<u8>,
    pub payload: Vec<u8>,
    pub bits: u64,
    pub totals: Totals,
    shared_book: Option<Arc<Codebook>>,
    payload_checksum: Option<u64>,
}
impl Default for EncodedBlock {
    fn default() -> Self {
        Self {
            header: [0; 56],
            table: Vec::new(),
            payload: Vec::new(),
            bits: 0,
            totals: Totals::default(),
            shared_book: None,
            payload_checksum: None,
        }
    }
}
impl EncodedBlock {
    pub(crate) fn codebook(&self, alphabet: usize) -> Result<Arc<Codebook>, String> {
        {
            #[cfg(feature = "profile")]
            let _span = crate::profile::Span::new("huffman.block.checksum");
            let valid = if let Some(expected) = self.payload_checksum {
                fingerprint(&self.payload) == expected
            } else {
                hash(&[&self.header[..48], &self.table, &self.payload]) == number(&self.header, 48)
            };
            if !valid {
                return Err("Huffman block checksum mismatch".into());
            }
        }
        if let Some(book) = &self.shared_book {
            return Ok(Arc::clone(book));
        }
        #[cfg(feature = "profile")]
        let _table = crate::profile::Span::new("huffman.block.canonical_table");
        let used = u32::from_le_bytes(self.header[4..8].try_into().unwrap()) as usize;
        Codebook::from_bytes(&self.table, used, alphabet).map(Arc::new)
    }
}
pub(crate) struct BlockSource<R> {
    input: R,
    index: Vec<IndexEntry>,
    index_offset: u64,
    totals: Totals,
    alphabet: usize,
    shared_book: Option<Arc<Codebook>>,
    compact: bool,
}
impl<R: Read + Seek> BlockSource<R> {
    pub(crate) fn compact(
        input: R,
        index: Vec<IndexEntry>,
        index_offset: u64,
        totals: Totals,
        alphabet: usize,
        shared_book: Option<Arc<Codebook>>,
    ) -> Self {
        Self {
            input,
            index,
            index_offset,
            totals,
            alphabet,
            shared_book,
            compact: true,
        }
    }
    pub(crate) fn len(&self) -> usize {
        self.index.len()
    }
    pub(crate) fn shared_book(&self) -> Option<&Arc<Codebook>> {
        self.shared_book.as_ref()
    }
    pub(crate) fn read(&mut self, index: usize, raw: &mut EncodedBlock) -> Result<(), String> {
        let entry = *self.index.get(index).ok_or("block index out of range")?;
        let next = self.index.get(index + 1).copied().unwrap_or(IndexEntry {
            offset: self.index_offset,
            before: self.totals,
        });
        if self.compact {
            return read_compact_encoded(
                &mut self.input,
                entry,
                next,
                self.totals,
                self.shared_book.as_ref(),
                raw,
            );
        }
        read_encoded(
            &mut self.input,
            entry,
            next,
            self.alphabet,
            self.shared_book.as_ref(),
            raw,
        )
    }
}
pub(crate) fn read_compact_encoded<R: Read + Seek>(
    input: &mut R,
    entry: IndexEntry,
    next: IndexEntry,
    totals: Totals,
    book: Option<&Arc<Codebook>>,
    raw: &mut EncodedBlock,
) -> Result<(), String> {
    if entry.before.exceeds(&totals) || next.before.exceeds(&totals) {
        return Err("compact block exceeds archive totals".into());
    }
    input.seek(SeekFrom::Start(entry.offset)).map_err(error)?;
    let header = read_array::<24>(input)?;
    let symbols = number(&header, 0);
    let bits = number(&header, 8);
    let delta = |a: u64, b: u64| {
        a.checked_sub(b)
            .ok_or_else(|| "nonmonotonic compact block totals".to_string())
    };
    let totals = Totals {
        bytes: delta(next.before.bytes, entry.before.bytes)?,
        fields: delta(next.before.fields, entry.before.fields)?,
        tokens: delta(next.before.tokens, entry.before.tokens)?,
        symbols: delta(next.before.symbols, entry.before.symbols)?,
    };
    if symbols == 0
        || symbols > MAX_BLOCK_SYMBOLS as u64
        || symbols > bits
        || symbols != totals.symbols
        || entry
            .offset
            .checked_add(24)
            .and_then(|n| n.checked_add(bits.div_ceil(8)))
            != Some(next.offset)
    {
        return Err("invalid compact Huffman data block".into());
    }
    raw.payload.resize(
        usize::try_from(bits.div_ceil(8)).map_err(|_| "compact payload exceeds address space")?,
        0,
    );
    input.read_exact(&mut raw.payload).map_err(error)?;
    raw.table.clear();
    raw.bits = bits;
    raw.totals = totals;
    raw.shared_book = Some(Arc::clone(book.ok_or("missing compact Huffman table")?));
    raw.payload_checksum = Some(number(&header, 16));
    Ok(())
}
fn read_encoded(
    input: &mut (impl Read + Seek),
    entry: IndexEntry,
    next: IndexEntry,
    alphabet: usize,
    shared: Option<&Arc<Codebook>>,
    raw: &mut EncodedBlock,
) -> Result<(), String> {
    input.seek(SeekFrom::Start(entry.offset)).map_err(error)?;
    raw.payload_checksum = None;
    let h = read_array::<56>(input)?;
    let count = u32::from_le_bytes(h[..4].try_into().unwrap()) as usize;
    let used = u32::from_le_bytes(h[4..8].try_into().unwrap()) as usize;
    let bits = number(&h, 8);
    let totals = Totals {
        bytes: number(&h, 16),
        fields: number(&h, 24),
        tokens: number(&h, 32),
        symbols: count as u64,
    };
    let table_len = u32::from_le_bytes(h[40..44].try_into().unwrap()) as u64;
    let end = entry
        .offset
        .checked_add(BLOCK_HEADER)
        .and_then(|n| n.checked_add(table_len))
        .and_then(|n| n.checked_add(bits.div_ceil(8)))
        .ok_or("block size overflow")?;
    let mut after = entry.before;
    after.add(totals)?;
    if count == 0
        || count > MAX_BLOCK_SYMBOLS
        || if shared.is_some() {
            used != 0 || table_len != 0
        } else {
            used == 0 || used > count || used > alphabet
        }
        || bits < count as u64
        || bits > count as u64 * 56
        || table_len > used as u64 * 6 + 58
        || end != next.offset
        || after != next.before
        || h[44..48] != [0; 4]
        || totals.fields > totals.symbols
        || totals.tokens < totals.symbols
    {
        return Err("invalid Huffman block sizes or totals".into());
    }
    raw.header = h;
    raw.shared_book = shared.cloned();
    raw.bits = bits;
    raw.totals = totals;
    raw.table.resize(table_len as usize, 0);
    input.read_exact(&mut raw.table).map_err(error)?;
    raw.payload.resize(bits.div_ceil(8) as usize, 0);
    input.read_exact(&mut raw.payload).map_err(error)
}
pub struct Reader<R> {
    input: R,
    pub dictionary: Dictionary,
    pub totals: Totals,
    pub index: Vec<IndexEntry>,
    dictionary_end: u64,
    index_offset: u64,
    shared: bool,
    shared_book: Option<Arc<Codebook>>,
    shared_table_bytes: u32,
}
impl<R: Read + Seek> Reader<R> {
    pub fn new(mut input: R) -> Result<Self, String> {
        let size = input.seek(SeekFrom::End(0)).map_err(error)?;
        input.seek(SeekFrom::Start(0)).map_err(error)?;
        let h = read_array::<96>(&mut input)?;
        let shared = &h[..8] == b"BRHUFF02";
        if (!shared && &h[..8] != b"BRHUFF01")
            || u32::from_le_bytes(h[12..16].try_into().unwrap()) != HEADER as u32
            || h[9..12].iter().any(|&b| b != 0)
            || (!shared && h[80..].iter().any(|&b| b != 0))
            || h[8] > 1
        {
            return Err("invalid BRHUFF header".into());
        }
        let totals = Totals {
            bytes: number(&h, 16),
            fields: number(&h, 24),
            tokens: number(&h, 32),
            symbols: number(&h, 40),
        };
        let (dict_len, blocks, index_offset) = (number(&h, 48), number(&h, 56), number(&h, 64));
        let dictionary_end = HEADER
            .checked_add(dict_len)
            .ok_or("dictionary size overflow")?;
        let shared_used = u32::from_le_bytes(h[80..84].try_into().unwrap()) as usize;
        let shared_table_bytes = u32::from_le_bytes(h[84..88].try_into().unwrap());
        let data_start = dictionary_end
            .checked_add(u64::from(shared_table_bytes))
            .ok_or("static table size overflow")?;
        if data_start > index_offset
            || index_offset > size
            || blocks > size / (BLOCK_HEADER + 40)
            || blocks
                .checked_mul(40)
                .and_then(|n| index_offset.checked_add(n + 16))
                != Some(size)
            || (blocks == 0) != (totals.symbols == 0)
            || (shared
                && ((shared_used == 0 && totals.symbols != 0)
                    || (shared_table_bytes == 0) != (shared_used == 0)
                    || u64::from(shared_table_bytes) > shared_used as u64 * 6 + 58))
        {
            return Err("invalid Huffman archive sizes".into());
        }
        let mut bytes = vec![0; usize::try_from(dict_len).map_err(|_| "dictionary too large")?];
        input.read_exact(&mut bytes).map_err(error)?;
        if fingerprint(&bytes) != number(&h, 72) {
            return Err("dictionary checksum mismatch".into());
        }
        let dictionary = Dictionary::from_bytes(&bytes, h[8] != 0)?;
        drop(bytes);
        let shared_book = if shared {
            #[cfg(feature = "profile")]
            let _table = crate::profile::Span::new("huffman.startup.static_table");
            if shared_used > dictionary.expansions.len() {
                return Err("static table exceeds dictionary".into());
            }
            let mut table = vec![0; shared_table_bytes as usize];
            input.read_exact(&mut table).map_err(error)?;
            if fingerprint(&table) != number(&h, 88) {
                return Err("static table checksum mismatch".into());
            }
            if shared_used == 0 {
                None
            } else {
                Some(Arc::new(Codebook::from_bytes(
                    &table,
                    shared_used,
                    dictionary.expansions.len(),
                )?))
            }
        } else {
            None
        };
        input.seek(SeekFrom::Start(index_offset)).map_err(error)?;
        if &read_array::<8>(&mut input)? != b"BRHIDX01" {
            return Err("invalid Huffman index".into());
        }
        let mut index = Vec::with_capacity(blocks as usize);
        let mut index_hash = 0xcbf29ce484222325u64;
        for i in 0..blocks {
            let b = read_array::<40>(&mut input)?;
            for &v in &b {
                index_hash = (index_hash ^ u64::from(v)).wrapping_mul(0x100000001b3);
            }
            let entry = IndexEntry {
                offset: number(&b, 0),
                before: Totals {
                    symbols: number(&b, 8),
                    bytes: number(&b, 16),
                    tokens: number(&b, 24),
                    fields: number(&b, 32),
                },
            };
            if entry.offset < data_start
                || entry.offset >= index_offset
                || entry.before.exceeds(&totals)
            {
                return Err("invalid Huffman index entry".into());
            }
            if i == 0 {
                if entry.offset != data_start || entry.before != Totals::default() {
                    return Err("invalid initial index entry".into());
                }
            } else {
                let previous: &IndexEntry = &index[index.len() - 1];
                if entry.offset <= previous.offset
                    || entry.before.symbols <= previous.before.symbols
                    || entry.before.tokens <= previous.before.tokens
                    || entry.before.bytes < previous.before.bytes
                    || entry.before.fields < previous.before.fields
                {
                    return Err("nonmonotone Huffman index".into());
                }
            }
            index.push(entry);
        }
        if number(&read_array::<8>(&mut input)?, 0) != index_hash {
            return Err("index checksum mismatch".into());
        }
        if blocks == 0 && (data_start != index_offset || totals != Totals::default()) {
            return Err("invalid empty Huffman archive".into());
        }
        Ok(Self {
            input,
            dictionary,
            totals,
            index,
            dictionary_end,
            index_offset,
            shared,
            shared_book,
            shared_table_bytes,
        })
    }
    pub fn block(&mut self, index: usize) -> Result<Block, String> {
        let entry = *self.index.get(index).ok_or("block index out of range")?;
        let next = self.index.get(index + 1).copied().unwrap_or(IndexEntry {
            offset: self.index_offset,
            before: self.totals,
        });
        let mut raw = EncodedBlock::default();
        read_encoded(
            &mut self.input,
            entry,
            next,
            self.dictionary.expansions.len(),
            self.shared_book.as_ref(),
            &mut raw,
        )?;
        let book = raw.codebook(self.dictionary.expansions.len())?;
        Ok(Block {
            book,
            payload: raw.payload,
            bits: raw.bits,
            totals: raw.totals,
        })
    }
    pub(crate) fn into_source(self) -> (Dictionary, Totals, BlockSource<R>) {
        let source = BlockSource {
            input: self.input,
            index: self.index,
            index_offset: self.index_offset,
            totals: self.totals,
            alphabet: self.dictionary.expansions.len(),
            shared_book: self.shared_book,
            compact: false,
        };
        (self.dictionary, self.totals, source)
    }
    /// Decode one indexed block. Use an existing arena to separate model startup
    /// from steady-state throughput measurements, or to seek without scanning.
    fn decode_direct<const IDS: bool, G: DecodeGraph>(
        dictionary: &G,
        arena: &Arena,
        block: &Block,
        references: &[(u32, u32)],
        out: &mut impl Write,
    ) -> Result<bool, String> {
        let (mut bytes, mut fields, mut ended) = (0u64, 0u64, true);
        let output_bytes = if IDS {
            block
                .totals
                .tokens
                .checked_mul(4)
                .ok_or("ID output length overflow")?
        } else {
            block.totals.bytes
        };
        let capacity = output_bytes.clamp(16, 1 << 20) as usize;
        let mut buffer = vec![0u8; capacity + 16];
        let mut cursor = 0usize;
        block
            .book
            .decode_ranks(&block.payload, block.bits, block.totals.symbols, |rank| {
                let (offset, flags) = references[rank as usize];
                let len = (flags & 0x7fff_ffff) as usize;
                ended = flags >> 31 != 0;
                bytes = bytes
                    .checked_add(len as u64)
                    .ok_or("decoded byte count overflow")?;
                fields += u64::from(ended);
                if bytes > block.totals.bytes || fields > block.totals.fields {
                    return Err("decoded block exceeds declared totals".into());
                }
                let output_len = if IDS { 4 } else { len };
                if output_len > capacity - cursor {
                    out.write_all(&buffer[..cursor]).map_err(error)?;
                    cursor = 0;
                }
                if IDS {
                    buffer[cursor..cursor + 4].copy_from_slice(&offset.to_le_bytes());
                    cursor += 4;
                } else if len <= 16 {
                    buffer[cursor..cursor + 16]
                        .copy_from_slice(&arena.data()[offset as usize..offset as usize + 16]);
                    cursor += len;
                } else if len >= capacity {
                    out.write_all(&arena.data()[offset as usize..offset as usize + len])
                        .map_err(error)?;
                } else {
                    buffer[cursor..cursor + len]
                        .copy_from_slice(&arena.data()[offset as usize..offset as usize + len]);
                    cursor += len;
                }
                Ok(())
            })?;
        let tokens = if dictionary.ai_tokens() {
            block.totals.symbols
        } else {
            bytes
                .checked_add(fields)
                .ok_or("decoded terminal count overflow")?
        };
        if bytes != block.totals.bytes
            || fields != block.totals.fields
            || tokens != block.totals.tokens
        {
            return Err("decoded Huffman totals mismatch".into());
        }
        out.write_all(&buffer[..cursor]).map_err(error)?;
        Ok(ended)
    }
    pub fn decode_block(
        dictionary: &Dictionary,
        arena: &Arena,
        block: &Block,
        out: &mut impl Write,
    ) -> Result<bool, String> {
        let references = arena.ranked_references(dictionary, &block.book.symbols);
        Self::decode_prepared(dictionary, arena, block, references.as_deref(), out)
    }
    fn decode_prepared<G: DecodeGraph>(
        dictionary: &G,
        arena: &Arena,
        block: &Block,
        references: Option<&[(u32, u32)]>,
        out: &mut impl Write,
    ) -> Result<bool, String> {
        if let Some(references) = references {
            return if arena.token_ids {
                Self::decode_direct::<true, G>(dictionary, arena, block, references, out)
            } else {
                Self::decode_direct::<false, G>(dictionary, arena, block, references, out)
            };
        }
        let mut totals = Totals::default();
        let mut ended = true;
        let mut pending = Vec::new();
        let output_bytes = if arena.token_ids {
            block
                .totals
                .tokens
                .checked_mul(4)
                .ok_or("ID output length overflow")?
        } else {
            block.totals.bytes
        };
        let mut buffer = Vec::with_capacity(output_bytes.clamp(16, 1 << 20) as usize);
        block
            .book
            .decode(&block.payload, block.bits, block.totals.symbols, |id| {
                ended = totals.add_symbol(dictionary, id)?;
                if totals.exceeds(&block.totals) {
                    return Err("decoded block exceeds declared totals".into());
                }
                arena.emit_graph(dictionary, id, &mut pending, |bytes| {
                    if bytes.len() > buffer.capacity() - buffer.len() {
                        out.write_all(&buffer).map_err(error)?;
                        buffer.clear();
                    }
                    if bytes.len() >= buffer.capacity() {
                        out.write_all(bytes).map_err(error)?;
                    } else {
                        buffer.extend_from_slice(bytes);
                    }
                    Ok(())
                })
            })?;
        if totals != block.totals {
            return Err("decoded Huffman totals mismatch".into());
        }
        out.write_all(&buffer).map_err(error)?;
        Ok(ended)
    }
    pub fn decode_to(&mut self, token_ids: bool, out: &mut impl Write) -> Result<Totals, String> {
        let decoder = self.prepare_decoder(token_ids, ARENA_BUDGET)?;
        let mut ended = true;
        for index in 0..self.index.len() {
            let block = self.block(index)?;
            ended = decoder.decode_block(&self.dictionary, &block, out)?;
        }
        if !ended {
            return Err("Huffman stream lacks final EOF".into());
        }
        out.flush().map_err(error)?;
        Ok(self.totals)
    }
    /// Seek by stable AI token index; decode at most the intersecting blocks.
    /// Returned IDs preserve EOF tokens and the original tokenizer segmentation.
    pub fn token_range(&mut self, start: u64, count: usize) -> Result<Vec<u32>, String> {
        if !self.dictionary.ai_tokens {
            return Err("byte grammar has no frozen AI token IDs".into());
        }
        let end = start
            .checked_add(count as u64)
            .ok_or("token range overflow")?;
        if end > self.totals.tokens {
            return Err("token range exceeds stream".into());
        }
        let mut result = Vec::new();
        result
            .try_reserve_exact(count)
            .map_err(|_| "requested token range is too large")?;
        if count == 0 {
            return Ok(result);
        }
        let mut index = self
            .index
            .partition_point(|e| e.before.tokens <= start)
            .saturating_sub(1);
        let mut stack = Vec::new();
        while result.len() < count {
            let block = self.block(index)?;
            let mut position = self.index[index].before.tokens;
            let mut actual = Totals::default();
            block
                .book
                .decode(&block.payload, block.bits, block.totals.symbols, |id| {
                    actual.add_symbol(&self.dictionary, id)?;
                    if actual.exceeds(&block.totals) {
                        return Err("decoded block exceeds declared totals".into());
                    }
                    stack.push(id);
                    while let Some(id) = stack.pop() {
                        if (id as usize) < self.dictionary.model.tokens.len() {
                            if position >= start && position < end {
                                result.push(id);
                            }
                            position += 1;
                        } else {
                            let (a, b) = self.dictionary.rules
                                [id as usize - self.dictionary.model.tokens.len()];
                            // Skip complete expansions outside the requested range.
                            let n = self.dictionary.expansions[id as usize].terminals;
                            if position >= end || position + n <= start {
                                position += n;
                            } else {
                                stack.push(b);
                                stack.push(a);
                            }
                        }
                    }
                    Ok(())
                })?;
            if actual != block.totals {
                return Err("decoded Huffman totals mismatch".into());
            }
            index += 1;
        }
        Ok(result)
    }
    pub fn dictionary_bytes(&self) -> u64 {
        self.dictionary_end - HEADER
    }
    pub fn format(&self) -> &'static str {
        if self.shared {
            "BRHUFF02"
        } else {
            "BRHUFF01"
        }
    }
    pub fn static_codebook(&self) -> Option<&Codebook> {
        self.shared_book.as_deref()
    }
    pub fn static_table_bytes(&self) -> u32 {
        self.shared_table_bytes
    }
    pub fn prepare_decoder(
        &self,
        token_ids: bool,
        arena_budget: usize,
    ) -> Result<PreparedDecoder, String> {
        PreparedDecoder::from_graph(
            &self.dictionary,
            self.shared_book.clone(),
            token_ids,
            arena_budget,
        )
    }
}

pub fn encode_file(
    tokenizer: &Tokenizer,
    input: &Path,
    output: &Path,
    block_size: usize,
) -> Result<Totals, String> {
    let dictionary = Dictionary::new(tokenizer.model().clone(), vec![], true)?;
    let mut source = BufReader::with_capacity(1 << 20, File::open(input).map_err(error)?);
    let mut counts = vec![0u64; dictionary.expansions.len()];
    let (mut first_bytes, mut first_fields) = (0u64, 0u64);
    let mut field = Vec::new();
    loop {
        field.clear();
        let n = source.read_until(b'\n', &mut field).map_err(error)?;
        if n == 0 {
            break;
        }
        first_bytes = first_bytes
            .checked_add(n as u64)
            .ok_or("input length overflow")?;
        first_fields = first_fields.checked_add(1).ok_or("field count overflow")?;
        for id in tokenizer.encode(&field) {
            counts[id as usize] = counts[id as usize]
                .checked_add(1)
                .ok_or("symbol count overflow")?;
        }
    }
    source.seek(SeekFrom::Start(0)).map_err(error)?;
    let mut encoder = Encoder::with_counts(
        BufWriter::with_capacity(1 << 20, File::create_new(output).map_err(error)?),
        &dictionary,
        block_size,
        &counts,
    )?;
    drop(counts);
    let (mut bytes, mut fields) = (0u64, 0u64);
    loop {
        field.clear();
        let n = source.read_until(b'\n', &mut field).map_err(error)?;
        if n == 0 {
            break;
        }
        bytes = bytes.checked_add(n as u64).ok_or("input byte overflow")?;
        fields = fields.checked_add(1).ok_or("input field overflow")?;
        for id in tokenizer.encode(&field) {
            encoder.push(id)?;
        }
    }
    if bytes != first_bytes || fields != first_fields {
        return Err("input changed between Huffman passes".into());
    }
    encoder.finish(bytes, fields)
}

pub fn decode_file(input: &Path, output: &Path, token_ids: bool) -> Result<Totals, String> {
    let mut encoded = BufReader::with_capacity(1 << 20, File::open(input).map_err(error)?);
    let mut magic = [0; 8];
    encoded.read_exact(&mut magic).map_err(error)?;
    encoded.seek(SeekFrom::Start(0)).map_err(error)?;
    if crate::shared_huffman::is_format(&magic) {
        let mut reader = crate::shared_huffman::Reader::new(encoded)?;
        if token_ids && !reader.ai_tokens {
            return Err("byte grammar has no frozen AI token IDs".into());
        }
        let mut out = BufWriter::with_capacity(1 << 20, File::create_new(output).map_err(error)?);
        return reader.decode_to(&mut out, token_ids);
    }
    let mut reader = Reader::new(encoded)?;
    if token_ids && !reader.dictionary.ai_tokens {
        return Err("byte grammar has no frozen AI token IDs".into());
    }
    let mut out = BufWriter::with_capacity(1 << 20, File::create_new(output).map_err(error)?);
    reader.decode_to(token_ids, &mut out)
}

pub fn pack_file(input: &Path, output: &Path, block_size: usize) -> Result<Totals, String> {
    let mut source = crate::archive_io::SymbolReader::new(BufReader::with_capacity(
        1 << 20,
        File::open(input).map_err(error)?,
    ))?;
    let dictionary = Dictionary::from_grammar(&source.grammar)?;
    // Release the training hash table: only compact recipes are needed below.
    source.grammar = crate::grammar::Grammar::default();
    let mut counts = vec![0u64; dictionary.expansions.len()];
    while let Some(id) = source.next_symbol()? {
        counts[id as usize] = counts[id as usize]
            .checked_add(1)
            .ok_or("symbol count overflow")?;
    }
    source.rewind_symbols()?;
    let mut encoder = Encoder::with_counts(
        BufWriter::with_capacity(1 << 20, File::create_new(output).map_err(error)?),
        &dictionary,
        block_size,
        &counts,
    )?;
    drop(counts);
    // SymbolReader validates IDs against its grammar, so retain alphabet size
    // through the lightweight reader rather than retaining the pair hash map.
    while let Some(id) = source.next_symbol()? {
        encoder.push(id)?;
    }
    encoder.finish(source.expected_bytes, source.expected_fields)
}
