//! Research-only prior for sources validated as UTF-8 before training.
//! On such sources a join is inside a code point exactly when its right-hand
//! expansion starts with a continuation byte. No occurrence-sized state needed.
use crate::grammar::{Grammar, BASE_SYMBOLS, EOF};

#[derive(Default)]
pub(crate) struct Validator {
    tail: [u8; 4],
    used: usize,
}

impl Validator {
    pub(crate) fn push(&mut self, mut bytes: &[u8]) -> Result<(), String> {
        if self.used != 0 {
            let needed = match self.tail[0] {
                0xc2..=0xdf => 2,
                0xe0..=0xef => 3,
                0xf0..=0xf4 => 4,
                _ => return Err("invalid UTF-8 prefix".into()),
            };
            let take = (needed - self.used).min(bytes.len());
            self.tail[self.used..self.used + take].copy_from_slice(&bytes[..take]);
            self.used += take;
            bytes = &bytes[take..];
            if self.used < needed {
                return Ok(());
            }
            std::str::from_utf8(&self.tail[..needed])
                .map_err(|_| "UTF-8 prior requires valid UTF-8 input")?;
            self.used = 0;
        }
        match std::str::from_utf8(bytes) {
            Ok(_) => Ok(()),
            Err(error) if error.error_len().is_none() => {
                let suffix = &bytes[error.valid_up_to()..];
                self.tail[..suffix.len()].copy_from_slice(suffix);
                self.used = suffix.len();
                Ok(())
            }
            Err(_) => Err("UTF-8 prior requires valid UTF-8 input".into()),
        }
    }

    pub(crate) fn finish(self) -> Result<(), String> {
        if self.used == 0 {
            Ok(())
        } else {
            Err("UTF-8 prior rejects a truncated final character".into())
        }
    }
}

pub(crate) struct JoinPrior {
    continuation: Vec<bool>,
}

impl JoinPrior {
    pub(crate) fn new() -> Self {
        Self {
            continuation: (0..BASE_SYMBOLS)
                .map(|symbol| symbol < EOF as usize && symbol & 0xc0 == 0x80)
                .collect(),
        }
    }

    pub(crate) fn sync(&mut self, grammar: &Grammar) {
        for rule in &grammar.rules[self.continuation.len() - BASE_SYMBOLS..] {
            self.continuation
                .push(self.continuation[rule.left as usize]);
        }
    }

    pub(crate) fn score(&self, base: f64, right: u32, count: u64, bonus: f64) -> f64 {
        if self.continuation[right as usize] {
            base + count as f64 * bonus
        } else {
            base
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streaming_validation_matches_standard_validation_at_every_chunk_size() {
        let cases: &[&[u8]] = &[
            b"",
            b"abc\n",
            "a中é😀国\nİstanbul".as_bytes(),
            &[0xe4, 0xb8],
            &[0x80],
            &[0xc0, 0x80],
            &[0xed, 0xa0, 0x80],
            &[0xf4, 0x90, 0x80, 0x80],
            &[0xe4, b'\n', 0xad],
        ];
        for &bytes in cases {
            for width in 1..=bytes.len().max(1) {
                let mut validator = Validator::default();
                let accepted = bytes
                    .chunks(width)
                    .try_for_each(|b| validator.push(b))
                    .and_then(|()| validator.finish())
                    .is_ok();
                assert_eq!(accepted, std::str::from_utf8(bytes).is_ok());
            }
        }
    }

    #[test]
    fn grammar_metadata_distinguishes_internal_joins_from_character_joins() {
        let mut grammar = Grammar::default();
        let prefix = grammar.add((0xe4, 0xb8)).unwrap();
        let character = grammar.add((prefix, 0xad)).unwrap();
        let suffix = grammar.add((0xb8, 0xad)).unwrap();
        let end = grammar.add((character, EOF)).unwrap();
        let mut prior = JoinPrior::new();
        prior.sync(&grammar);
        for right in [0xb8, 0xad, suffix] {
            assert_eq!(prior.score(-20.0, right, 100, 0.5), 30.0);
        }
        for right in [prefix, character, end, EOF, b'a' as u32] {
            assert_eq!(prior.score(-20.0, right, 100, 0.5), -20.0);
        }
        let new_suffix = grammar.add((suffix, EOF)).unwrap();
        prior.sync(&grammar);
        assert_eq!(prior.score(7.0, new_suffix, 3, 1.0), 10.0);
        assert_eq!(prior.score(7.0, suffix, 3, 0.0), 7.0);
    }
}
