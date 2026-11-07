//! Stable document coordinates for support counting inside parallel rewrites.
//! No per-worker maps; normalized aliases are shared immutable metadata.
use super::{u64_at, Policy};
use crate::grammar::{Grammar, Symbol};
use std::{
    collections::{hash_map::Entry, HashMap},
    fs::{self, File},
    io::{BufReader, Read},
    path::PathBuf,
    time::SystemTime,
};

pub(crate) const DOCUMENT_MASK: u64 = (1u64 << 56) - 1;

struct Record {
    bytes_end: u64,
    fields_end: u64,
}

pub(crate) struct Documents {
    records: Vec<Record>,
    aliases: Vec<(u64, u64)>,
    path: PathBuf,
    length: u64,
    modified: Option<SystemTime>,
}

impl Documents {
    pub fn new(policy: &Policy, bytes: u64, fields: u64) -> Result<Self, String> {
        Self::new_at_round(policy, bytes, fields, 0)
    }

    pub fn new_at_round(
        policy: &Policy,
        bytes: u64,
        fields: u64,
        round: usize,
    ) -> Result<Self, String> {
        if !(1..=64).contains(&policy.minimum_groups) {
            return Err("document support must be 1..64".into());
        }
        let file = File::open(&policy.index).map_err(|e| e.to_string())?;
        let metadata = file.metadata().map_err(|e| e.to_string())?;
        let mut input = BufReader::with_capacity(65536, file);
        let mut header = [0; 24];
        input.read_exact(&mut header).map_err(|e| e.to_string())?;
        if &header[..8] != b"BREDOC01"
            || header[8..12] != 104u32.to_le_bytes()
            || header[12..16] != 0u32.to_le_bytes()
        {
            return Err(
                "last-seen counting requires a BREDOC01 training index, never holdout".into(),
            );
        }
        let count = u64_at(&header, 16);
        if count > DOCUMENT_MASK {
            return Err("last-seen document IDs exceed 56 bits".into());
        }
        if count.checked_mul(104).and_then(|n| n.checked_add(24)) != Some(metadata.len()) {
            return Err("document index length differs from record count".into());
        }
        let capacity = usize::try_from(count).map_err(|_| "document index too large")?;
        let mut records = Vec::with_capacity(capacity);
        let mut aliases = Vec::new();
        let mut first = HashMap::<[u8; 32], u64>::new();
        let (mut found_bytes, mut found_fields) = (0u64, 0u64);
        crate::training_progress::phase("document_index", round, 0, count, "documents");
        for ordinal in 1..=count {
            let mut raw = [0; 104];
            input.read_exact(&mut raw).map_err(|e| e.to_string())?;
            let (n, f) = (u64_at(&raw, 84), u64_at(&raw, 92));
            if u64_at(&raw, 76) != found_bytes || n == 0 || f == 0 {
                return Err("invalid training document offset/length/fields".into());
            }
            found_bytes = found_bytes
                .checked_add(n)
                .ok_or("document byte total overflow")?;
            found_fields = found_fields
                .checked_add(f)
                .ok_or("document field total overflow")?;
            records.push(Record {
                bytes_end: found_bytes,
                fields_end: found_fields,
            });
            let group: [u8; 32] = raw[32..64].try_into().unwrap();
            match first.entry(group) {
                Entry::Vacant(entry) => {
                    entry.insert(ordinal);
                }
                Entry::Occupied(entry) => aliases.push((ordinal, *entry.get())),
            }
            if ordinal % 65536 == 0 {
                crate::training_progress::phase(
                    "document_index",
                    round,
                    ordinal,
                    count,
                    "documents",
                );
            }
        }
        if found_bytes != bytes || found_fields != fields {
            return Err("document index differs from training byte/field totals".into());
        }
        crate::training_progress::phase("document_index", round, count, count, "documents");
        Ok(Self {
            records,
            aliases,
            path: policy.index.clone(),
            length: metadata.len(),
            modified: metadata.modified().ok(),
        })
    }

    pub fn validate_file(&self) -> Result<(), String> {
        let metadata = fs::metadata(&self.path).map_err(|e| e.to_string())?;
        if metadata.len() != self.length || metadata.modified().ok() != self.modified {
            return Err("training document index changed".into());
        }
        Ok(())
    }

    pub fn cursor(&self, field: u64) -> Cursor<'_> {
        let row = self
            .records
            .partition_point(|record| record.fields_end <= field);
        let mut cursor = Cursor {
            source: self,
            row,
            field,
            bytes: 0,
            document: 0,
        };
        cursor.refresh_document();
        cursor
    }
}

pub(crate) struct Cursor<'a> {
    source: &'a Documents,
    row: usize,
    pub field: u64,
    bytes: u64,
    document: u64,
}
impl Cursor<'_> {
    fn refresh_document(&mut self) {
        if self.row == self.source.records.len() {
            self.document = 0;
            return;
        }
        let ordinal = self.row as u64 + 1;
        self.document = self
            .source
            .aliases
            .binary_search_by_key(&ordinal, |&(record, _)| record)
            .map_or(ordinal, |i| self.source.aliases[i].1);
    }
    pub fn document(&self) -> u64 {
        self.document
    }
    pub fn advance(&mut self, ending: bool) -> Result<(), String> {
        if !ending {
            return Ok(());
        }
        self.field = self.field.checked_add(1).ok_or("field ordinal overflow")?;
        let record = self
            .source
            .records
            .get(self.row)
            .ok_or("packed fields exceed document index")?;
        if self.field > record.fields_end {
            return Err("packed fields cross indexed document boundary".into());
        }
        if self.field == record.fields_end {
            self.row += 1;
            self.refresh_document();
        }
        Ok(())
    }
    /// Fused with the existing initial symbol-statistics walk, not another pass.
    pub fn validate_symbol(
        &mut self,
        symbol: Symbol,
        ending: bool,
        grammar: &Grammar,
    ) -> Result<(), String> {
        let record = self
            .source
            .records
            .get(self.row)
            .ok_or("packed input exceeds document index")?;
        self.bytes = self
            .bytes
            .checked_add((grammar.terminals(symbol) - usize::from(ending)) as u64)
            .ok_or("document byte overflow")?;
        if self.bytes > record.bytes_end
            || (ending && self.field + 1 == record.fields_end && self.bytes != record.bytes_end)
        {
            return Err("document boundary differs from packed source fields".into());
        }
        self.advance(ending)
    }
    pub fn finish(&self, bytes: u64, fields: u64) -> Result<(), String> {
        if self.row != self.source.records.len() || self.bytes != bytes || self.field != fields {
            return Err("packed input ended inside document index".into());
        }
        Ok(())
    }
}
