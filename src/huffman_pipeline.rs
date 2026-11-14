//! Four ordered stages: encoded IO -> Huffman to arena references -> assembly ->
//! output IO. Each edge owns a fixed recycled buffer pool. No dictionary copies,
//! polling, unbounded queues, or detached workers; all stages are joined.
use crate::{
    huffman_archive::{BlockSource, EncodedBlock, Reader, Totals},
    huffman_dictionary::{Arena, DecodeGraph, ExpansionReference},
};
use std::{
    fs::File,
    io::{BufReader, Read, Seek, SeekFrom, Write},
    path::Path,
    sync::mpsc::{sync_channel, Receiver, SyncSender},
    thread,
    time::{Duration, Instant},
};

const CANCELLED: &str = "Huffman pipeline cancelled";
const LOOKUP_BATCH: usize = 256;

#[derive(Clone, Copy, Debug)]
pub struct Options {
    pub buffers_per_stage: usize,
    pub references_per_buffer: usize,
    pub output_buffer_bytes: usize,
    pub arena_bytes: usize,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            buffers_per_stage: 2,
            references_per_buffer: 65536,
            output_buffer_bytes: 1 << 20,
            arena_bytes: 64 << 20,
        }
    }
}
impl Options {
    fn validate(self) -> Result<(), String> {
        if !(1..=8).contains(&self.buffers_per_stage)
            || !(1..=1 << 20).contains(&self.references_per_buffer)
            || !(1..=16 << 20).contains(&self.output_buffer_bytes)
            || self.arena_bytes > 1 << 30
        {
            return Err("invalid Huffman pipeline buffer limits".into());
        }
        Ok(())
    }
}
#[derive(Clone, Copy, Debug, Default)]
pub struct Stage {
    /// Elapsed time outside channel operations; includes OS scheduling delays.
    pub active_seconds: f64,
    pub wait_seconds: f64,
}
pub struct Report {
    pub totals: Totals,
    /// Input, Huffman decoding, assembly, output, in that order.
    pub stages: [Stage; 4],
    /// Excludes dictionary/arena setup; public command timing includes both.
    pub seconds: f64,
}
struct Clock {
    start: Instant,
    wait: Duration,
}
impl Clock {
    fn new() -> Self {
        Self {
            start: Instant::now(),
            wait: Duration::ZERO,
        }
    }
    fn recv<T>(&mut self, rx: &Receiver<T>) -> Result<T, String> {
        let start = Instant::now();
        let value = rx.recv();
        self.wait += start.elapsed();
        value.map_err(|_| CANCELLED.into())
    }
    fn send<T>(&mut self, tx: &SyncSender<T>, value: T) -> Result<(), String> {
        let start = Instant::now();
        let result = tx.send(value);
        self.wait += start.elapsed();
        result.map_err(|_| CANCELLED.into())
    }
    fn finish(&self) -> Stage {
        Stage {
            active_seconds: self.start.elapsed().saturating_sub(self.wait).as_secs_f64(),
            wait_seconds: self.wait.as_secs_f64(),
        }
    }
}
enum Message<T> {
    Data(T),
    End,
}
enum References {
    Block {
        cached: bool,
        prevalidated: bool,
        totals: Totals,
    },
    Data(Vec<(u32, u32)>),
    EndBlock(Option<bool>),
    End,
}
struct Bytes {
    data: Vec<u8>,
    used: usize,
}

fn read_stage<R: Read + Seek>(
    mut input: BlockSource<R>,
    free: Receiver<EncodedBlock>,
    ready: SyncSender<Message<EncodedBlock>>,
) -> Result<Stage, String> {
    #[cfg(feature = "profile")]
    let _stage = crate::profile::Span::new("huffman.stage.input");
    let mut clock = Clock::new();
    for index in 0..input.len() {
        let mut raw = {
            #[cfg(feature = "profile")]
            let _wait = crate::profile::Span::new("huffman.input.wait.free_encoded_buffer");
            clock.recv(&free)?
        };
        {
            #[cfg(feature = "profile")]
            let _read = crate::profile::Span::new("huffman.input.read_encoded_block");
            input.read(index, &mut raw)?;
        }
        clock.send(&ready, Message::Data(raw))?;
    }
    clock.send(&ready, Message::End)?;
    Ok(clock.finish())
}

struct ReferencesOut {
    clock: Clock,
    free: Receiver<Vec<(u32, u32)>>,
    ready: SyncSender<References>,
    current: Option<Vec<(u32, u32)>>,
    capacity: usize,
}
impl ReferencesOut {
    fn append(&mut self, mut references: &[(u32, u32)]) -> Result<(), String> {
        while !references.is_empty() {
            if self.current.is_none() {
                self.current = Some({
                    #[cfg(feature = "profile")]
                    let _wait =
                        crate::profile::Span::new("huffman.decoder.wait.free_reference_buffer");
                    self.clock.recv(&self.free)?
                });
            }
            let buffer = self.current.as_mut().unwrap();
            let n = references.len().min(self.capacity - buffer.len());
            buffer.extend_from_slice(&references[..n]);
            references = &references[n..];
            if buffer.len() == self.capacity {
                self.clock
                    .send(&self.ready, References::Data(self.current.take().unwrap()))?;
            }
        }
        Ok(())
    }
    fn resolve(&mut self, ranks: &[u32], references: &[(u32, u32)]) -> Result<(), String> {
        let mut batch = [(0, 0); LOOKUP_BATCH];
        for (to, &rank) in batch.iter_mut().zip(ranks) {
            *to = references[rank as usize];
        }
        self.append(&batch[..ranks.len()])
    }
    fn resolve_expansions<G: DecodeGraph>(
        &mut self,
        ranks: &[u32],
        references: &[ExpansionReference],
        dictionary: &G,
        actual: &mut Totals,
        expected: &Totals,
        ended: &mut bool,
    ) -> Result<(), String> {
        let mut batch = [(0, 0); LOOKUP_BATCH];
        let mut counts = Totals::default();
        let mut fallback_counts = Totals::default();
        for (to, &rank) in batch.iter_mut().zip(ranks) {
            let r = &references[rank as usize];
            *to = (r.offset, r.length);
            if r.length == u32::MAX {
                *ended = fallback_counts.add_symbol(dictionary, r.offset)?;
            } else {
                counts.symbols += 1;
                counts.bytes += u64::from(r.byte_flags & 0x7fff_ffff);
                counts.fields += u64::from(r.byte_flags >> 31);
                counts.tokens += u64::from(r.terminals);
                *ended = r.byte_flags >> 31 != 0;
            }
        }
        counts.add(fallback_counts)?;
        actual.add(counts)?;
        if actual.exceeds(expected) {
            return Err("decoded block exceeds declared totals".into());
        }
        self.append(&batch[..ranks.len()])
    }
    fn flush(&mut self) -> Result<(), String> {
        if let Some(buffer) = self.current.take() {
            if !buffer.is_empty() {
                self.clock.send(&self.ready, References::Data(buffer))?;
            }
        }
        Ok(())
    }
    fn finish(mut self) -> Result<Stage, String> {
        self.flush()?;
        self.clock.send(&self.ready, References::End)?;
        Ok(self.clock.finish())
    }
}
fn cached_assemble<const IDS: bool>(
    references: &[(u32, u32)],
    state: &mut AssemblyBlock,
    arena: &Arena,
    output: &mut BytesOut,
) -> Result<(), String> {
    let (bytes, fields) = references.iter().fold((0u64, 0u64), |(b, f), &(_, flags)| {
        (
            b + u64::from(flags & 0x7fff_ffff),
            f + u64::from(flags >> 31),
        )
    });
    // Nonnegative expansion lengths let us validate the whole batch before
    // any copy. Bounded block counts and 31-bit lengths make these sums fit.
    state.actual.bytes += bytes;
    state.actual.fields += fields;
    if state.actual.bytes > state.expected.bytes || state.actual.fields > state.expected.fields {
        return Err("decoded block exceeds declared totals".into());
    }
    if let Some(&(_, flags)) = references.last() {
        state.ended = flags >> 31 != 0;
    }
    if IDS {
        output.words(references)?;
    } else {
        output.copies(arena, references)?;
    }
    Ok(())
}
struct DecodeContext<'a, G: DecodeGraph> {
    dictionary: &'a G,
    arena: &'a Arena,
    references: Option<&'a [(u32, u32)]>,
    expansions: Option<&'a [ExpansionReference]>,
}
fn decode_stage<G: DecodeGraph>(
    context: DecodeContext<'_, G>,
    ready: Receiver<Message<EncodedBlock>>,
    free: SyncSender<EncodedBlock>,
    mut output: ReferencesOut,
) -> Result<Stage, String> {
    let DecodeContext {
        dictionary,
        arena,
        references: shared_references,
        expansions: shared_expansions,
    } = context;
    #[cfg(feature = "profile")]
    let _stage = crate::profile::Span::new("huffman.stage.decoder");
    loop {
        let message = {
            #[cfg(feature = "profile")]
            let _wait = crate::profile::Span::new("huffman.decoder.wait.encoded_input");
            output.clock.recv(&ready)?
        };
        match message {
            Message::End => break,
            Message::Data(raw) => {
                let book = raw.codebook(dictionary.expansions().len())?;
                let local_references = if shared_references.is_none() {
                    #[cfg(feature = "profile")]
                    let _refs = crate::profile::Span::new("huffman.decoder.prepare_references");
                    arena.ranked_references(dictionary, &book.symbols)
                } else {
                    None
                };
                let references = shared_references.or(local_references.as_deref());
                let local_expansions = if shared_expansions.is_none() && references.is_none() {
                    arena.ranked_expansions(dictionary, &book.symbols)
                } else {
                    None
                };
                let expansions = shared_expansions.or(local_expansions.as_deref());
                output.clock.send(
                    &output.ready,
                    References::Block {
                        cached: references.is_some() || expansions.is_some(),
                        prevalidated: expansions.is_some(),
                        totals: raw.totals,
                    },
                )?;
                #[cfg(feature = "profile")]
                let bits = crate::profile::Span::new("huffman.decoder.bits_validate_emit");
                let mut validated_end = None;
                if let Some(expansions) = expansions {
                    let mut ranks = [0; LOOKUP_BATCH];
                    let (mut used, mut actual, mut ended) = (0, Totals::default(), true);
                    book.decode_ranks(&raw.payload, raw.bits, raw.totals.symbols, |rank| {
                        ranks[used] = rank;
                        used += 1;
                        if used == ranks.len() {
                            output.resolve_expansions(
                                &ranks,
                                expansions,
                                dictionary,
                                &mut actual,
                                &raw.totals,
                                &mut ended,
                            )?;
                            used = 0;
                        }
                        Ok(())
                    })?;
                    output.resolve_expansions(
                        &ranks[..used],
                        expansions,
                        dictionary,
                        &mut actual,
                        &raw.totals,
                        &mut ended,
                    )?;
                    if actual != raw.totals {
                        return Err("decoded Huffman totals mismatch".into());
                    }
                    validated_end = Some(ended);
                } else if let Some(references) = references {
                    let mut ranks = [0; LOOKUP_BATCH];
                    let mut used = 0;
                    book.decode_ranks(&raw.payload, raw.bits, raw.totals.symbols, |rank| {
                        ranks[used] = rank;
                        used += 1;
                        if used == ranks.len() {
                            output.resolve(&ranks, references)?;
                            used = 0;
                        }
                        Ok(())
                    })?;
                    output.resolve(&ranks[..used], references)?;
                } else {
                    book.decode(&raw.payload, raw.bits, raw.totals.symbols, |id| {
                        output.append(&[(id, 0)])
                    })?;
                }
                output.flush()?;
                output
                    .clock
                    .send(&output.ready, References::EndBlock(validated_end))?;
                #[cfg(feature = "profile")]
                drop(bits);
                // The reader may already have sent End and dropped its free
                // receiver; returning the final pooled buffer can then fail.
                let _ = output.clock.send(&free, raw);
            }
        }
    }
    output.finish()
}

struct BytesOut {
    clock: Clock,
    free: Receiver<Bytes>,
    ready: SyncSender<Message<Bytes>>,
    current: Option<Bytes>,
    capacity: usize,
}
impl BytesOut {
    fn ensure(&mut self, len: usize) -> Result<(), String> {
        if self
            .current
            .as_ref()
            .is_some_and(|b| len > self.capacity - b.used)
        {
            self.flush()?;
        }
        if self.current.is_none() {
            self.current = Some({
                #[cfg(feature = "profile")]
                let _wait = crate::profile::Span::new("huffman.assembly.wait.free_output_buffer");
                self.clock.recv(&self.free)?
            });
        }
        Ok(())
    }
    fn flush(&mut self) -> Result<(), String> {
        if let Some(buffer) = self.current.take() {
            if buffer.used != 0 {
                self.clock.send(&self.ready, Message::Data(buffer))?;
            }
        }
        Ok(())
    }
    fn append(&mut self, mut bytes: &[u8]) -> Result<(), String> {
        while !bytes.is_empty() {
            self.ensure(1)?;
            let buffer = self.current.as_mut().unwrap();
            let n = bytes.len().min(self.capacity - buffer.used);
            buffer.data[buffer.used..buffer.used + n].copy_from_slice(&bytes[..n]);
            buffer.used += n;
            bytes = &bytes[n..];
        }
        Ok(())
    }
    fn copies(&mut self, arena: &Arena, mut references: &[(u32, u32)]) -> Result<(), String> {
        while let Some(&(offset, flags)) = references.first() {
            let len = (flags & 0x7fff_ffff) as usize;
            if len > self.capacity {
                let start = offset as usize;
                self.append(&arena.data()[start..start + len])?;
                references = &references[1..];
                continue;
            }
            self.ensure(len)?;
            let buffer = self.current.as_mut().unwrap();
            let mut used = buffer.used;
            let mut n = 0;
            for &(offset, flags) in references {
                let len = (flags & 0x7fff_ffff) as usize;
                if len > self.capacity - used {
                    break;
                }
                let start = offset as usize;
                if len != 0 {
                    if len <= 16 {
                        // Both buffers have real, initialized padding. The
                        // writer sees only `used`, never the copied padding.
                        buffer.data[used..used + 16]
                            .copy_from_slice(&arena.data()[start..start + 16]);
                    } else {
                        buffer.data[used..used + len]
                            .copy_from_slice(&arena.data()[start..start + len]);
                    }
                }
                used += len;
                n += 1;
            }
            buffer.used = used;
            references = &references[n..];
        }
        Ok(())
    }
    fn words(&mut self, mut references: &[(u32, u32)]) -> Result<(), String> {
        if self.capacity < 4 {
            for &(id, _) in references {
                self.append(&id.to_le_bytes())?;
            }
            return Ok(());
        }
        while !references.is_empty() {
            self.ensure(4)?;
            let buffer = self.current.as_mut().unwrap();
            let n = references.len().min((self.capacity - buffer.used) / 4);
            let end = buffer.used + n * 4;
            for (to, &(id, _)) in buffer.data[buffer.used..end]
                .chunks_exact_mut(4)
                .zip(&references[..n])
            {
                to.copy_from_slice(&id.to_le_bytes());
            }
            buffer.used = end;
            references = &references[n..];
        }
        Ok(())
    }
    fn finish(mut self) -> Result<Stage, String> {
        self.flush()?;
        self.clock.send(&self.ready, Message::End)?;
        Ok(self.clock.finish())
    }
}
struct AssemblyBlock {
    cached: bool,
    prevalidated: bool,
    expected: Totals,
    actual: Totals,
    ended: bool,
}
struct AssemblyContext<'a, G: DecodeGraph> {
    dictionary: &'a G,
    arena: &'a Arena,
    expected: Totals,
    blocks: usize,
}
fn assemble_stage<G: DecodeGraph>(
    context: AssemblyContext<'_, G>,
    ready: Receiver<References>,
    free: SyncSender<Vec<(u32, u32)>>,
    mut output: BytesOut,
) -> Result<Stage, String> {
    let AssemblyContext {
        dictionary,
        arena,
        expected,
        blocks,
    } = context;
    #[cfg(feature = "profile")]
    let _stage = crate::profile::Span::new("huffman.stage.assembly");
    let mut stack = Vec::new();
    let mut block: Option<AssemblyBlock> = None;
    let (mut total, mut count, mut ended) = (Totals::default(), 0usize, true);
    loop {
        let message = {
            #[cfg(feature = "profile")]
            let _wait = crate::profile::Span::new("huffman.assembly.wait.references");
            output.clock.recv(&ready)?
        };
        match message {
            References::End => break,
            References::Block {
                cached,
                prevalidated,
                totals,
            } => {
                if block.is_some() {
                    return Err("missing Huffman block boundary".into());
                }
                if cached && !prevalidated {
                    // Cached byte grammars have one terminal per byte plus
                    // EOF; flat AI grammars have one terminal per codec symbol.
                    // Validate this relation before emitting any expansion.
                    let tokens = if dictionary.ai_tokens() {
                        totals.symbols
                    } else {
                        totals
                            .bytes
                            .checked_add(totals.fields)
                            .ok_or("terminal count overflow")?
                    };
                    if tokens != totals.tokens {
                        return Err("decoded Huffman totals mismatch".into());
                    }
                }
                block = Some(AssemblyBlock {
                    cached,
                    prevalidated,
                    expected: totals,
                    actual: Totals::default(),
                    ended: true,
                });
            }
            References::EndBlock(validated_end) => {
                let mut state = block.take().ok_or("unexpected Huffman block boundary")?;
                if state.prevalidated {
                    let ended = validated_end.ok_or("missing validated expansion totals")?;
                    if state.actual.symbols != state.expected.symbols {
                        return Err("decoded Huffman totals mismatch".into());
                    }
                    state.actual = state.expected;
                    state.ended = ended;
                } else if validated_end.is_some() {
                    return Err("unexpected validated expansion totals".into());
                }
                if state.actual != state.expected {
                    return Err("decoded Huffman totals mismatch".into());
                }
                ended = state.ended;
                total.add(state.actual)?;
                count += 1;
            }
            References::Data(mut buffer) => {
                let state = block.as_mut().ok_or("missing Huffman block header")?;
                #[cfg(feature = "profile")]
                let copying = crate::profile::Span::new("huffman.assembly.copy_batch");
                if state.cached {
                    state.actual.symbols += buffer.len() as u64;
                    if state.actual.symbols > state.expected.symbols {
                        return Err("decoded block exceeds declared totals".into());
                    }
                    if state.prevalidated {
                        let mut references = buffer.as_slice();
                        while !references.is_empty() {
                            let n = references
                                .iter()
                                .position(|&(_, len)| len == u32::MAX)
                                .unwrap_or(references.len());
                            output.copies(arena, &references[..n])?;
                            references = &references[n..];
                            if let Some(&(id, _)) = references.first() {
                                arena
                                    .emit_graph(dictionary, id, &mut stack, |b| output.append(b))?;
                                references = &references[1..];
                            }
                        }
                    } else if arena.token_ids {
                        cached_assemble::<true>(&buffer, state, arena, &mut output)?;
                    } else {
                        cached_assemble::<false>(&buffer, state, arena, &mut output)?;
                    }
                    if !state.prevalidated {
                        state.actual.tokens = if dictionary.ai_tokens() {
                            state.actual.symbols
                        } else {
                            state.actual.bytes + state.actual.fields
                        };
                    }
                } else {
                    for &(id, _) in &buffer {
                        state.ended = state.actual.add_symbol(dictionary, id)?;
                        if state.actual.exceeds(&state.expected) {
                            return Err("decoded block exceeds declared totals".into());
                        }
                        arena.emit_graph(dictionary, id, &mut stack, |b| output.append(b))?;
                    }
                }
                #[cfg(feature = "profile")]
                drop(copying);
                buffer.clear();
                let _ = output.clock.send(&free, buffer);
            }
        }
    }
    if block.is_some() || count != blocks || total != expected || !ended {
        return Err("Huffman stream totals or final EOF differ".into());
    }
    output.finish()
}
fn write_stage(
    out: &mut impl Write,
    ready: Receiver<Message<Bytes>>,
    free: SyncSender<Bytes>,
    expected: u64,
) -> Result<Stage, String> {
    #[cfg(feature = "profile")]
    let _stage = crate::profile::Span::new("huffman.stage.output");
    let mut clock = Clock::new();
    let mut count = 0u64;
    loop {
        let message = {
            #[cfg(feature = "profile")]
            let _wait = crate::profile::Span::new("huffman.output.wait.bytes");
            clock.recv(&ready)?
        };
        match message {
            Message::End => break,
            Message::Data(mut buffer) => {
                count = count
                    .checked_add(buffer.used as u64)
                    .ok_or("pipeline output length overflow")?;
                if count > expected {
                    return Err("pipeline output exceeds declared length".into());
                }
                {
                    #[cfg(feature = "profile")]
                    let _write = crate::profile::Span::new("huffman.output.write_buffer");
                    out.write_all(&buffer.data[..buffer.used])
                        .map_err(|e| e.to_string())?;
                }
                buffer.used = 0;
                let _ = clock.send(&free, buffer);
            }
        }
    }
    if count != expected {
        return Err("pipeline output length mismatch".into());
    }
    out.flush().map_err(|e| e.to_string())?;
    Ok(clock.finish())
}

/// Consumes the reader to give IO exclusive ownership of the source. The
/// caller performs output IO while three scoped worker threads run upstream.
/// An error closes channels in both directions and every worker is joined.
pub fn decode<R: Read + Seek + Send>(
    reader: Reader<R>,
    token_ids: bool,
    out: &mut impl Write,
    options: Options,
) -> Result<Report, String> {
    let (dictionary, totals, source) = reader.into_source();
    decode_graph(dictionary, totals, source, token_ids, out, options)
}

pub(crate) fn decode_graph<G: DecodeGraph, R: Read + Seek + Send>(
    dictionary: G,
    totals: Totals,
    source: BlockSource<R>,
    token_ids: bool,
    out: &mut impl Write,
    options: Options,
) -> Result<Report, String> {
    options.validate()?;
    let arena = {
        #[cfg(feature = "profile")]
        let _arena = crate::profile::Span::new("huffman.startup.arena");
        Arena::from_graph(&dictionary, token_ids, options.arena_bytes)?
    };
    let shared_references = {
        #[cfg(feature = "profile")]
        let _references = crate::profile::Span::new("huffman.startup.static_references");
        source
            .shared_book()
            .and_then(|book| arena.ranked_references(&dictionary, &book.symbols))
    };
    let shared_expansions = if shared_references.is_none() {
        source
            .shared_book()
            .and_then(|book| arena.ranked_expansions(&dictionary, &book.symbols))
    } else {
        None
    };
    let expected_output = if token_ids {
        totals
            .tokens
            .checked_mul(4)
            .ok_or("ID output length overflow")?
    } else {
        totals.bytes
    };
    let blocks = source.len();
    let slots = options.buffers_per_stage;
    let (raw_free_tx, raw_free_rx) = sync_channel(slots);
    let (raw_tx, raw_rx) = sync_channel(slots);
    let (slice_free_tx, slice_free_rx) = sync_channel(slots);
    let (slice_tx, slice_rx) = sync_channel(slots);
    let (byte_free_tx, byte_free_rx) = sync_channel(slots);
    let (byte_tx, byte_rx) = sync_channel(slots);
    for _ in 0..slots {
        raw_free_tx
            .send(EncodedBlock::default())
            .map_err(|_| CANCELLED)?;
        slice_free_tx
            .send(Vec::with_capacity(options.references_per_buffer))
            .map_err(|_| CANCELLED)?;
        byte_free_tx
            .send(Bytes {
                data: vec![0; options.output_buffer_bytes + 16],
                used: 0,
            })
            .map_err(|_| CANCELLED)?;
    }
    let start = Instant::now();
    let stages = thread::scope(|scope| {
        let input = scope.spawn(move || read_stage(source, raw_free_rx, raw_tx));
        let d = &dictionary;
        let a = &arena;
        let references = shared_references.as_deref();
        let expansions = shared_expansions.as_deref();
        let decoding = scope.spawn(move || {
            decode_stage(
                DecodeContext {
                    dictionary: d,
                    arena: a,
                    references,
                    expansions,
                },
                raw_rx,
                raw_free_tx,
                ReferencesOut {
                    clock: Clock::new(),
                    free: slice_free_rx,
                    ready: slice_tx,
                    current: None,
                    capacity: options.references_per_buffer,
                },
            )
        });
        let assembly = scope.spawn(move || {
            assemble_stage(
                AssemblyContext {
                    dictionary: d,
                    arena: a,
                    expected: totals,
                    blocks,
                },
                slice_rx,
                slice_free_tx,
                BytesOut {
                    clock: Clock::new(),
                    free: byte_free_rx,
                    ready: byte_tx,
                    current: None,
                    capacity: options.output_buffer_bytes,
                },
            )
        });
        let writing = write_stage(out, byte_rx, byte_free_tx, expected_output);
        let results = [input.join(), decoding.join(), assembly.join()]
            .into_iter()
            .map(|r| r.unwrap_or_else(|_| Err("Huffman pipeline worker panicked".into())))
            .chain([writing])
            .collect::<Vec<_>>();
        let labels = ["input IO", "Huffman decoding", "assembly", "output IO"];
        // A failed stage cancels the others. Prefer its actual error over the
        // resulting channel-disconnected errors; never hide a failed write.
        for (i, result) in results.iter().enumerate() {
            if let Err(e) = result {
                if e != CANCELLED {
                    return Err(format!("{}: {e}", labels[i]));
                }
            }
        }
        let mut stages = [Stage::default(); 4];
        for (i, result) in results.into_iter().enumerate() {
            stages[i] = result?;
        }
        Ok(stages)
    })?;
    Ok(Report {
        totals,
        stages,
        seconds: start.elapsed().as_secs_f64(),
    })
}

pub fn decode_file(
    input: &Path,
    output: &Path,
    token_ids: bool,
    options: Options,
) -> Result<Report, String> {
    let mut encoded =
        BufReader::with_capacity(1 << 20, File::open(input).map_err(|e| e.to_string())?);
    let mut magic = [0; 8];
    encoded.read_exact(&mut magic).map_err(|e| e.to_string())?;
    encoded
        .seek(SeekFrom::Start(0))
        .map_err(|e| e.to_string())?;
    if crate::shared_huffman::is_format(&magic) {
        let reader = crate::shared_huffman::Reader::new(encoded)?;
        if token_ids && !reader.ai_tokens {
            return Err("byte grammar has no frozen AI token IDs".into());
        }
        let mut out = File::create_new(output).map_err(|e| e.to_string())?;
        return crate::shared_huffman::decode_pipeline(reader, token_ids, &mut out, options);
    }
    let reader = Reader::new(encoded)?;
    if token_ids && !reader.dictionary.ai_tokens {
        return Err("byte grammar has no frozen AI token IDs".into());
    }
    let mut out = File::create_new(output).map_err(|e| e.to_string())?;
    decode(reader, token_ids, &mut out, options)
}
