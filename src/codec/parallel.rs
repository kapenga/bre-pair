//! Ordered, bounded field parsing. One immutable model and one Huffman writer;
//! each worker owns scratch only. Exactly `workers` packets are in flight.
use super::{EncodeMode, Encoder, Frozen, MatchScratch, Matcher, ParseScratch, Parser, Totals};
use std::{
    io::{BufRead, Seek, Write},
    panic::{catch_unwind, AssertUnwindSafe},
    sync::{
        mpsc::{sync_channel, Receiver},
        Arc, Mutex,
    },
    thread,
};

#[derive(Clone, Copy, Debug)]
pub struct EncodeOptions {
    pub workers: usize,
    pub batch_bytes: usize,
    pub mode: EncodeMode,
}
impl Default for EncodeOptions {
    fn default() -> Self {
        Self {
            workers: 1,
            batch_bytes: 65536,
            mode: EncodeMode::Advanced,
        }
    }
}
impl EncodeOptions {
    pub(super) fn validate(self) -> Result<(), String> {
        if !(1..=256).contains(&self.workers) || !(1..=16 << 20).contains(&self.batch_bytes) {
            return Err("invalid codec encoding worker or batch limits".into());
        }
        Ok(())
    }
}

#[derive(Default)]
struct Packet {
    index: u64,
    bytes: Vec<u8>,
    ends: Vec<usize>,
    symbols: Vec<u32>,
    tokens: u64,
}
impl Packet {
    fn read(&mut self, input: &mut impl BufRead, index: u64, limit: usize) -> Result<bool, String> {
        #[cfg(feature = "profile")]
        let start = std::time::Instant::now();
        self.bytes.clear();
        self.ends.clear();
        self.symbols.clear();
        self.tokens = 0;
        self.index = index;
        while self.ends.len() < 512 && self.bytes.len() < limit {
            if input
                .read_until(b'\n', &mut self.bytes)
                .map_err(|e| e.to_string())?
                == 0
            {
                break;
            }
            self.ends.push(self.bytes.len());
        }
        #[cfg(feature = "profile")]
        crate::profile::elapsed(
            "codec.input",
            self.ends.len() as u64,
            start.elapsed().as_secs_f64(),
        );
        Ok(!self.bytes.is_empty())
    }
}

fn parse(
    mut packet: Packet,
    tokenizer: &Parser,
    matcher: &Matcher,
    bytes: &mut ParseScratch,
    grammar: &mut MatchScratch,
    mode: EncodeMode,
    #[cfg(feature = "profile")] times: &mut [std::time::Duration; 2],
) -> Result<Packet, String> {
    let mut begin = 0;
    for &end in &packet.ends {
        #[cfg(feature = "profile")]
        let start = std::time::Instant::now();
        let ids = tokenizer.encode(&packet.bytes[begin..end], bytes);
        #[cfg(feature = "profile")]
        {
            times[0] += start.elapsed();
        }
        packet.tokens = packet
            .tokens
            .checked_add(ids.len() as u64)
            .ok_or("AI count overflow")?;
        #[cfg(feature = "profile")]
        let start = std::time::Instant::now();
        packet
            .symbols
            .extend_from_slice(matcher.segment_using(ids, grammar, mode)?);
        #[cfg(feature = "profile")]
        {
            times[1] += start.elapsed();
        }
        begin = end;
    }
    Ok(packet)
}

fn receive<T>(receiver: &Mutex<Receiver<T>>) -> Result<T, String> {
    receiver
        .lock()
        .map_err(|_| "codec task queue poisoned")?
        .recv()
        .map_err(|_| "codec task queue closed".into())
}

pub(super) fn encode<R: BufRead, W: Write + Seek>(
    model: &Frozen,
    tokenizer: &Parser,
    matcher: &Matcher,
    mut source: R,
    output: W,
    block_size: usize,
    options: EncodeOptions,
) -> Result<Totals, String> {
    let mut encoder = Encoder::with_codebook(output, &model.dictionary, block_size, &model.book)?;
    let (mut bytes, mut fields, mut tokens) = (0u64, 0u64, 0u64);
    thread::scope(|scope| -> Result<(), String> {
        let (tasks, receiver) = sync_channel::<Packet>(options.workers);
        let receiver = Arc::new(Mutex::new(receiver));
        let (results, ready) = sync_channel::<Result<Packet, String>>(options.workers);
        let mut handles = Vec::new();
        for _ in 0..options.workers {
            let receiver = Arc::clone(&receiver);
            let results = results.clone();
            handles.push(scope.spawn(move || {
                let mut bytes = ParseScratch::default();
                let mut grammar = MatchScratch::default();
                #[cfg(feature = "profile")]
                let mut times = [std::time::Duration::ZERO; 2];
                #[cfg(feature = "profile")]
                let mut fields = 0u64;
                while let Ok(packet) = receive(&receiver) {
                    #[cfg(feature = "profile")]
                    {
                        fields += packet.ends.len() as u64;
                    }
                    let parsed = catch_unwind(AssertUnwindSafe(|| {
                        parse(
                            packet,
                            tokenizer,
                            matcher,
                            &mut bytes,
                            &mut grammar,
                            options.mode,
                            #[cfg(feature = "profile")]
                            &mut times,
                        )
                    }))
                    .unwrap_or_else(|_| Err("codec parsing worker panicked".into()));
                    let failed = parsed.is_err();
                    if results.send(parsed).is_err() || failed {
                        break;
                    }
                }
                #[cfg(feature = "profile")]
                {
                    crate::profile::elapsed("codec.tokenize", fields, times[0].as_secs_f64());
                    crate::profile::elapsed("codec.grammar_match", fields, times[1].as_secs_f64());
                }
            }));
        }
        drop(results);
        let result = catch_unwind(AssertUnwindSafe(|| -> Result<(), String> {
            let (mut issued, mut next) = (0u64, 0u64);
            let mut eof = false;
            let mut pending: Vec<Option<Packet>> = (0..options.workers).map(|_| None).collect();
            for _ in 0..options.workers {
                let mut packet = Packet::default();
                if !packet.read(&mut source, issued, options.batch_bytes)? {
                    eof = true;
                    break;
                }
                tasks.send(packet).map_err(|_| "codec workers closed")?;
                issued += 1;
            }
            while next < issued {
                let packet = ready.recv().map_err(|_| "codec results closed")??;
                if packet.index < next || packet.index >= issued {
                    return Err("invalid codec packet order".into());
                }
                let slot = (packet.index % options.workers as u64) as usize;
                if pending[slot].is_some() {
                    return Err("duplicate codec packet".into());
                }
                pending[slot] = Some(packet);
                while let Some(mut packet) =
                    pending[(next % options.workers as u64) as usize].take()
                {
                    if packet.index != next {
                        return Err("codec packet sequence differs".into());
                    }
                    #[cfg(feature = "profile")]
                    let start = std::time::Instant::now();
                    encoder.extend(&packet.symbols)?;
                    #[cfg(feature = "profile")]
                    crate::profile::elapsed(
                        "codec.huffman_write",
                        packet.ends.len() as u64,
                        start.elapsed().as_secs_f64(),
                    );
                    bytes = bytes
                        .checked_add(packet.bytes.len() as u64)
                        .ok_or("input size overflow")?;
                    fields = fields
                        .checked_add(packet.ends.len() as u64)
                        .ok_or("field count overflow")?;
                    tokens = tokens
                        .checked_add(packet.tokens)
                        .ok_or("AI count overflow")?;
                    next += 1;
                    if !eof {
                        if packet.read(&mut source, issued, options.batch_bytes)? {
                            tasks.send(packet).map_err(|_| "codec workers closed")?;
                            issued += 1;
                        } else {
                            eof = true;
                        }
                    }
                }
            }
            Ok(())
        }))
        .unwrap_or_else(|_| Err("codec encoding coordinator panicked".into()));
        // Close both directions before joining, including failed reads/writes.
        drop(tasks);
        drop(ready);
        let mut error = None;
        for handle in handles {
            if handle.join().is_err() {
                error = Some("codec parsing worker panicked".to_string());
            }
        }
        result?;
        if let Some(error) = error {
            return Err(error);
        }
        Ok(())
    })?;
    let totals = {
        #[cfg(feature = "profile")]
        let _span = crate::profile::Span::new("codec.huffman_finish");
        encoder.finish(bytes, fields)?
    };
    if totals.tokens != tokens {
        return Err("codec changed the frozen AI token count".into());
    }
    Ok(totals)
}
