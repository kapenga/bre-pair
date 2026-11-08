//! Research standalone frozen model using the shared canonical dictionary layout.
use super::{descriptors, number, parse_metadata, Graph, MetadataSpec, RuleLengths, HEADER, NONE};
use crate::{
    codec::{Frozen, Matcher},
    huffman::{put_var, Codebook},
    huffman_archive::{Block, PreparedDecoder, Totals},
    tokenizer::fingerprint,
};
use std::{io::Write, sync::Arc};

const MAGIC: &[u8; 8] = b"BRHMOD01";

pub struct StaticModel {
    graph: Graph,
    book: Arc<Codebook>,
    pub training: Totals,
}
impl StaticModel {
    /// Preserve the data codes exactly. Uncoded construction helpers are inlined.
    pub fn encode_frozen(source: &Frozen, coding: RuleLengths) -> Result<Vec<u8>, String> {
        let dictionary = &source.dictionary;
        let basis = dictionary.model.tokens.len();
        let mut coded = vec![false; dictionary.expansions.len()];
        for &id in &source.book.symbols {
            *coded.get_mut(id as usize).ok_or("undefined model code")? = true;
        }
        if !dictionary.ai_tokens || coded[..basis].contains(&false) {
            return Err("static record model requires coded frozen terminals".into());
        }
        let mut histogram = [0u64; 57];
        for (_, length) in source.book.lengths() {
            histogram[length as usize] += 1;
        }
        let min = (1..histogram.len())
            .find(|&i| histogram[i] != 0)
            .ok_or("empty model")?;
        let mut table = vec![min as u8, source.book.max_bits];
        for &n in &histogram[min..=source.book.max_bits as usize] {
            put_var(&mut table, n);
        }
        let mut raw = Vec::new();
        let mut references = Vec::new();
        let mut stack = Vec::new();
        for &id in &source.book.symbols {
            if (id as usize) < basis {
                put_var(&mut raw, 0);
                put_var(&mut raw, u64::from(id));
                continue;
            }
            let (a, b) = dictionary.rules[id as usize - basis];
            stack.extend([b, a]);
            let start = references.len();
            while let Some(child) = stack.pop() {
                if coded
                    .get(child as usize)
                    .copied()
                    .ok_or("undefined helper")?
                {
                    references.push(child);
                } else if (child as usize) >= basis {
                    let (a, b) = dictionary.rules[child as usize - basis];
                    stack.extend([b, a]);
                } else {
                    return Err("uncoded frozen terminal".into());
                }
            }
            put_var(&mut raw, (references.len() - start) as u64);
        }
        let compressed = if coding == RuleLengths::Auto {
            descriptors::compress(&raw, source.book.symbols.len() as u64)?
        } else {
            None
        };
        let huffman_lengths = compressed.is_some();
        let descriptor = compressed.unwrap_or(raw);
        let (encoded, bits) = source.book.encode(&references)?;
        let tokens = dictionary.model.to_bytes();
        let mut metadata = Vec::new();
        put_var(&mut metadata, tokens.len() as u64);
        metadata.extend(tokens);
        metadata.extend(&table);
        metadata.extend(&descriptor);
        metadata.extend(encoded);
        let mut header = [0u8; HEADER];
        header[..8].copy_from_slice(MAGIC);
        for (at, n) in [
            (8, 1),
            (16, source.training.bytes),
            (24, source.training.fields),
            (32, source.training.tokens),
            (40, source.training.symbols),
            (48, metadata.len() as u64),
            (56, bits),
            (64, references.len() as u64),
            (72, descriptor.len() as u64),
            (80, table.len() as u64),
            (88, source.book.symbols.len() as u64),
            (96, u64::from(huffman_lengths)),
            (112, fingerprint(&metadata)),
        ] {
            header[at..at + 8].copy_from_slice(&n.to_le_bytes());
        }
        let check = fingerprint(&header[..120]);
        header[120..].copy_from_slice(&check.to_le_bytes());
        let mut out = header.to_vec();
        out.extend(metadata);
        Ok(out)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < HEADER
            || &bytes[..8] != MAGIC
            || number(bytes, 8) != 1
            || number(bytes, 96) > 1
            || number(bytes, 104) != 0
            || fingerprint(&bytes[..120]) != number(bytes, 120)
            || number(bytes, 48) != (bytes.len() - HEADER) as u64
            || fingerprint(&bytes[HEADER..]) != number(bytes, 112)
        {
            return Err("invalid static record model".into());
        }
        let training = Totals {
            bytes: number(bytes, 16),
            fields: number(bytes, 24),
            tokens: number(bytes, 32),
            symbols: number(bytes, 40),
        };
        if training.fields > training.tokens || training.symbols > training.tokens {
            return Err("invalid static model totals".into());
        }
        let (old, book, _, _) = parse_metadata(
            &bytes[HEADER..],
            &MetadataSpec {
                ai: true,
                huffman_rule_lengths: number(bytes, 96) != 0,
                run_rule_lengths: false,
                bits: number(bytes, 56),
                refs: number(bytes, 64),
                desc: number(bytes, 72),
                table: number(bytes, 80),
                leaves: number(bytes, 88),
            },
        )?;
        let book = book.ok_or("empty static record model")?;
        let basis = old.model.tokens.len();
        let n = old.terminal.len();
        if basis > n {
            return Err("missing frozen terminals".into());
        }
        let mut mapping = vec![NONE; n];
        let mut inverse = vec![NONE; n];
        let mut next = basis;
        for (id, &terminal) in old.terminal.iter().enumerate() {
            let new = if terminal != NONE {
                terminal as usize
            } else {
                let new = next;
                next += 1;
                new
            };
            if new >= n || inverse[new] != NONE {
                return Err("invalid static model mapping".into());
            }
            mapping[id] = new as u32;
            inverse[new] = id as u32;
        }
        if inverse.contains(&NONE) {
            return Err("missing static model terminal".into());
        }
        let entries: Vec<_> = book
            .lengths()
            .map(|(id, len)| (mapping[id as usize], len))
            .collect();
        let book = Codebook::from_lengths(&entries)?;
        if !book.lengths().eq(entries.iter().copied()) {
            return Err("noncanonical static model terminal order".into());
        }
        let mut terminal = Vec::with_capacity(n);
        let mut offsets = vec![0];
        let mut children = Vec::with_capacity(old.children.len());
        let mut expansions = Vec::with_capacity(n);
        for old_id in inverse {
            let id = old_id as usize;
            terminal.push(old.terminal[id]);
            expansions.push(old.expansions[id]);
            children.extend(
                old.children[old.offsets[id]..old.offsets[id + 1]]
                    .iter()
                    .map(|&child| mapping[child as usize]),
            );
            offsets.push(children.len());
        }
        let graph = Graph {
            model: old.model,
            terminal,
            offsets,
            children,
            expansions,
            ai_tokens: true,
        };
        Ok(Self {
            graph,
            book: Arc::new(book),
            training,
        })
    }
    pub fn codebook(&self) -> Arc<Codebook> {
        Arc::clone(&self.book)
    }
    pub fn prepare(&self) -> Result<Matcher, String> {
        Matcher::from_graph(&self.graph, &self.book)
    }
    pub fn prepare_decoder(
        &self,
        token_ids: bool,
        budget: usize,
    ) -> Result<PreparedDecoder, String> {
        let mut decoder = PreparedDecoder::from_graph(
            &self.graph,
            Some(Arc::clone(&self.book)),
            token_ids,
            budget,
        )?;
        decoder.prepare_field_references(&self.graph);
        Ok(decoder)
    }
    pub fn decode_block(
        &self,
        decoder: &PreparedDecoder,
        block: &Block,
        out: &mut impl Write,
    ) -> Result<bool, String> {
        decoder.decode_graph_block(&self.graph, block, out)
    }
    /// Decode a byte-bounded field using EOF instead of stored counters.
    /// Returns (actual bits, actual symbols); discard partial output on error.
    pub fn decode_field(
        &self,
        decoder: &PreparedDecoder,
        payload: &[u8],
        maximum_output: usize,
        out: &mut Vec<u8>,
    ) -> Result<(u64, u64), String> {
        if !decoder.uses_codebook(&self.book) {
            return Err("field uses a different static table".into());
        }
        decoder.decode_graph_field(&self.graph, payload, maximum_output, out)
    }
    /// Decode directly into a bounded caller buffer, without a temporary field
    /// vector. Returns (bytes written, actual bits, actual symbols). No guard
    /// bytes are needed; discard partial output on error. Unused capacity may
    /// be overwritten by short copies and must not contain live data.
    pub fn decode_field_into(
        &self,
        decoder: &PreparedDecoder,
        payload: &[u8],
        out: &mut [u8],
    ) -> Result<(usize, u64, u64), String> {
        if !decoder.uses_codebook(&self.book) {
            return Err("field uses a different static table".into());
        }
        decoder.decode_graph_field_into(&self.graph, payload, out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        codec::MatchScratch,
        huffman_dictionary::{byte_model, Dictionary},
    };
    fn fixture() -> Frozen {
        let dictionary = Dictionary::new(
            byte_model(),
            vec![(97, 98), (257, 99), (98, 99), (97, 259), (258, 256)],
            true,
        )
        .unwrap();
        let mut counts = vec![1; 262];
        counts[258] = 100;
        counts[260] = 100;
        counts[261] = 37;
        Frozen {
            dictionary,
            book: Codebook::from_counts(&counts).unwrap(),
            training: Totals {
                bytes: 1000,
                fields: 10,
                tokens: 1010,
                symbols: 500,
            },
        }
    }
    #[test]
    fn direct_field_output_needs_no_guard_and_preserves_limits_and_model_identity() {
        let model = StaticModel::from_bytes(
            &StaticModel::encode_frozen(&fixture(), RuleLengths::Auto).unwrap(),
        )
        .unwrap();
        let matcher = model.prepare().unwrap();
        for budget in [0, 1, 65536] {
            let decoder = model.prepare_decoder(false, budget).unwrap();
            for raw in [
                vec![],
                b"abc".to_vec(),
                vec![0, 255],
                b"abcabcabcabcabcabcabc".to_vec(),
            ] {
                let mut terminals: Vec<u32> = raw.iter().map(|&b| u32::from(b)).collect();
                terminals.push(256);
                let ids = matcher.segment(&terminals).unwrap();
                let (payload, bits) = model.codebook().encode(&ids).unwrap();
                for extra in [0, 1, 15, 16, 31] {
                    // Sentinels outside the provided slice must remain intact,
                    // even when the logical output ends on a short copy.
                    let mut out = vec![0xa5; raw.len() + extra + 2];
                    let end = out.len() - 1;
                    assert_eq!(
                        model
                            .decode_field_into(&decoder, &payload, &mut out[1..end])
                            .unwrap(),
                        (raw.len(), bits, ids.len() as u64)
                    );
                    assert_eq!(&out[1..1 + raw.len()], raw);
                    assert_eq!(out[0], 0xa5);
                    assert_eq!(out[end], 0xa5);
                }
                if !raw.is_empty() {
                    assert!(model
                        .decode_field_into(&decoder, &payload, &mut vec![0; raw.len() - 1])
                        .is_err());
                }
                for end in 0..payload.len() {
                    assert!(model
                        .decode_field_into(&decoder, &payload[..end], &mut vec![0; raw.len()])
                        .is_err());
                }
                let mut trailing = payload;
                trailing.push(0);
                assert!(model
                    .decode_field_into(&decoder, &trailing, &mut vec![0; raw.len()])
                    .is_err());
            }
        }
        let other = StaticModel::from_bytes(
            &StaticModel::encode_frozen(&fixture(), RuleLengths::Auto).unwrap(),
        )
        .unwrap();
        assert!(model
            .decode_field_into(&other.prepare_decoder(false, 65536).unwrap(), &[], &mut [])
            .is_err());
        assert!(model
            .decode_field_into(&model.prepare_decoder(true, 65536).unwrap(), &[], &mut [])
            .is_err());
    }
    #[test]
    fn compact_model_eof_api_handles_cached_and_uncached_compound_endings() {
        let source = fixture();
        let model = StaticModel::from_bytes(
            &StaticModel::encode_frozen(&source, RuleLengths::Auto).unwrap(),
        )
        .unwrap();
        let matcher = model.prepare().unwrap();
        let field = [97, 98, 99, 256];
        let ids = matcher.segment(&field).unwrap();
        let (payload, bits) = model.codebook().encode(&ids).unwrap();
        for budget in [0, 65536] {
            let decoder = model.prepare_decoder(false, budget).unwrap();
            let mut out = Vec::new();
            assert_eq!(
                model.decode_field(&decoder, &payload, 3, &mut out).unwrap(),
                (bits, ids.len() as u64)
            );
            assert_eq!(out, b"abc");
        }
        let other = StaticModel::from_bytes(
            &StaticModel::encode_frozen(&source, RuleLengths::Auto).unwrap(),
        )
        .unwrap();
        assert!(model
            .decode_field(
                &other.prepare_decoder(false, 65536).unwrap(),
                &payload,
                3,
                &mut Vec::new()
            )
            .is_err());
    }
    #[test]
    fn compact_model_preserves_data_codes_both_outputs_and_duplicate_expansion_ties() {
        let model = fixture();
        let old_matcher = model.prepare().unwrap();
        for coding in [RuleLengths::Auto, RuleLengths::Varint] {
            let bytes = StaticModel::encode_frozen(&model, coding).unwrap();
            let restored = StaticModel::from_bytes(&bytes).unwrap();
            let matcher = restored.prepare().unwrap();
            let book = restored.codebook();
            let fields = [
                vec![97, 98, 99, 97, 98, 99, 256],
                (0..257).collect(),
                vec![256],
            ];
            for field in fields {
                let original = old_matcher.segment(&field).unwrap();
                let mut scratch = MatchScratch::default();
                let encoded = matcher.segment_with(&field, &mut scratch).unwrap();
                let (payload, bits) = model.book.encode(&original).unwrap();
                assert_eq!(book.encode(encoded).unwrap(), (payload.clone(), bits));
                let totals = Totals {
                    bytes: field.iter().filter(|&&v| v != 256).count() as u64,
                    fields: 1,
                    tokens: field.len() as u64,
                    symbols: encoded.len() as u64,
                };
                for (ids, budget) in [(false, 0), (false, 65536), (true, 0), (true, 65536)] {
                    let decoder = restored.prepare_decoder(ids, budget).unwrap();
                    let mut block = Block {
                        book: Arc::clone(&book),
                        payload: payload.clone(),
                        bits,
                        totals,
                    };
                    let mut out = Vec::new();
                    assert!(restored.decode_block(&decoder, &block, &mut out).unwrap());
                    let expected: Vec<_> = if ids {
                        field.iter().flat_map(|v| v.to_le_bytes()).collect()
                    } else {
                        field
                            .iter()
                            .filter(|&&v| v != 256)
                            .map(|&v| v as u8)
                            .collect()
                    };
                    assert_eq!(out, expected);
                    // A cached merged expansion can represent many terminal
                    // AI IDs. Declared tokens must never be inferred from the
                    // smaller number of coded symbols, even on this fast path.
                    block.totals.tokens += 1;
                    assert!(restored
                        .decode_block(&decoder, &block, &mut Vec::new())
                        .is_err());
                    block.totals.tokens -= 2;
                    assert!(restored
                        .decode_block(&decoder, &block, &mut Vec::new())
                        .is_err());
                }
            }
        }
    }
    #[test]
    fn compact_model_rejects_corruption_and_truncation() {
        let bytes = StaticModel::encode_frozen(&fixture(), RuleLengths::Auto).unwrap();
        for end in 0..bytes.len() {
            assert!(StaticModel::from_bytes(&bytes[..end]).is_err());
        }
        for i in 0..bytes.len() {
            let mut bad = bytes.clone();
            bad[i] ^= 1;
            assert!(StaticModel::from_bytes(&bad).is_err(), "byte {i}");
        }
    }
    #[test]
    fn uncoded_helpers_inline_to_variable_arity_without_changing_codes() {
        let dictionary =
            Dictionary::new(byte_model(), vec![(97, 98), (257, 99), (258, 256)], true).unwrap();
        let mut counts = vec![1; 260];
        counts[257] = 0;
        counts[258] = 0;
        counts[259] = 100;
        let model = Frozen {
            dictionary,
            book: Codebook::from_counts(&counts).unwrap(),
            training: Totals {
                bytes: 3,
                fields: 1,
                tokens: 4,
                symbols: 1,
            },
        };
        let bytes = StaticModel::encode_frozen(&model, RuleLengths::Auto).unwrap();
        let restored = StaticModel::from_bytes(&bytes).unwrap();
        let symbols = restored
            .prepare()
            .unwrap()
            .segment(&[97, 98, 99, 256])
            .unwrap();
        let (payload, bits) = restored.codebook().encode(&symbols).unwrap();
        assert_eq!(model.book.encode(&[259]).unwrap(), (payload.clone(), bits));
        let block = Block {
            book: restored.codebook(),
            payload,
            bits,
            totals: model.training,
        };
        for budget in [0, 65536] {
            let decoder = restored.prepare_decoder(false, budget).unwrap();
            let mut out = Vec::new();
            assert!(restored.decode_block(&decoder, &block, &mut out).unwrap());
            assert_eq!(out, b"abc");
        }
    }
}
