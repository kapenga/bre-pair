//! Small reusable application API; CLI and language bindings share the engine.
//! Models are immutable and may be shared. Each encoding worker owns scratch.
use crate::{
    grammar::Symbol,
    tokenizer::{ByteParser, Model, ParseScratch},
};
use std::path::Path;

pub struct ByteTokenizer {
    parser: ByteParser,
}

impl ByteTokenizer {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        Ok(Self {
            parser: ByteParser::new(Model::from_bytes(bytes)?)?,
        })
    }
    pub fn load(path: &Path) -> Result<Self, String> {
        Self::from_bytes(&std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?)
    }
    pub fn model(&self) -> &Model {
        self.parser.model()
    }
    pub fn to_bytes(&self) -> Vec<u8> {
        self.model().to_bytes()
    }
    pub fn vocabulary_size(&self) -> usize {
        self.model().tokens.len()
    }
    /// Encode one complete field, including its EOF (possibly fused into a token).
    /// Returned IDs borrow the caller's workspace until its next encoding call.
    pub fn encode<'a>(&self, bytes: &[u8], scratch: &'a mut ParseScratch) -> &'a [Symbol] {
        self.parser.encode(bytes, scratch)
    }
    pub fn decode(&self, ids: &[Symbol], maximum_bytes: usize) -> Result<Vec<u8>, String> {
        let mut bytes = 0usize;
        for &id in ids {
            let token = self
                .model()
                .tokens
                .get(id as usize)
                .ok_or("undefined token ID")?;
            bytes = bytes
                .checked_add(token.bytes.len())
                .ok_or("decoded field length overflow")?;
            if bytes > maximum_bytes {
                return Err("decoded field exceeds output limit".into());
            }
        }
        self.model().decode_field(ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::{Token, Tokenizer};
    fn model() -> Model {
        let mut tokens = (0..=255)
            .map(|b| Token {
                bytes: vec![b],
                ends_field: false,
            })
            .collect::<Vec<_>>();
        tokens.push(Token {
            bytes: vec![],
            ends_field: true,
        });
        tokens.extend([
            Token {
                bytes: b"hello".to_vec(),
                ends_field: false,
            },
            Token {
                bytes: b" world".to_vec(),
                ends_field: true,
            },
        ]);
        Model::new(tokens).unwrap()
    }
    #[test]
    fn application_api_matches_exact_reference_and_preserves_arbitrary_bytes() {
        let model = model();
        let api = ByteTokenizer::from_bytes(&model.to_bytes()).unwrap();
        let reference = Tokenizer::new(model).unwrap();
        let mut scratch = ParseScratch::default();
        for raw in [b"".as_slice(), b"hello world", b"\0\xffhello\nworld\r\n"] {
            let ids = api.encode(raw, &mut scratch);
            assert_eq!(ids, reference.encode(raw));
            assert_eq!(api.decode(ids, raw.len()).unwrap(), raw);
            if !raw.is_empty() {
                assert!(api.decode(ids, raw.len() - 1).is_err());
            }
        }
        assert_eq!(api.vocabulary_size(), 259);
        assert_eq!(api.to_bytes(), api.model().to_bytes());
    }
    #[test]
    fn application_api_rejects_missing_eof_extra_tokens_and_bad_ids() {
        let api = ByteTokenizer::from_bytes(&model().to_bytes()).unwrap();
        for ids in [vec![], vec![97], vec![256, 97], vec![u32::MAX]] {
            assert!(api.decode(&ids, 100).is_err());
        }
        assert!(ByteTokenizer::from_bytes(b"invalid").is_err());
    }
}
