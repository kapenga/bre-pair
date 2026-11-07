//! Deliberately slow independent recounts; absent from ordinary builds.
use std::collections::BTreeMap;

use crate::{
    grammar::{Grammar, Pair, BASE_SYMBOLS, EOF},
    packed::PackedSymbols,
    registry::Frozen,
    symbol_counts::SymbolCounts,
};

/// Reconstruct runs and raw adjacencies without Position, PairCounter, delta
/// maintenance, the optimized pair map, or the eligibility cache.
pub(crate) fn check(
    blocks: &[PackedSymbols],
    grammar: &Grammar,
    registry: &Frozen<'_>,
    dense: Option<&SymbolCounts>,
    minimum: u64,
    track: bool,
) -> Result<(), String> {
    check_inner(blocks, grammar, registry, dense, minimum, track, true)
}

pub(crate) fn check_segments(
    blocks: &[PackedSymbols],
    grammar: &Grammar,
    registry: &Frozen<'_>,
    dense: Option<&SymbolCounts>,
    minimum: u64,
    track: bool,
) -> Result<(), String> {
    check_inner(blocks, grammar, registry, dense, minimum, track, false)
}

fn check_inner(
    blocks: &[PackedSymbols],
    grammar: &Grammar,
    registry: &Frozen<'_>,
    dense: Option<&SymbolCounts>,
    minimum: u64,
    track: bool,
    complete_chunks: bool,
) -> Result<(), String> {
    let alphabet = grammar.alphabet_size();
    let mut uses = vec![0u64; alphabet];
    let mut starts = vec![0u64; alphabet];
    let mut overlaps = vec![0u64; alphabet];
    let mut left = vec![0u64; alphabet];
    let mut right = vec![0u64; alphabet];
    let mut expected = BTreeMap::<Pair, u64>::new();
    for (i, rule) in grammar.rules.iter().enumerate() {
        let id = (BASE_SYMBOLS + i) as u32;
        if rule.left >= id
            || rule.right >= id
            || grammar.ends_field(rule.left)
            || rule.ends_field != grammar.ends_field(rule.right)
            || rule.terminals != grammar.terminals(rule.left) + grammar.terminals(rule.right)
        {
            return Err(format!("invalid rule {id}"));
        }
    }
    for block in blocks {
        let symbols: Vec<_> = block.iter().collect();
        if complete_chunks && symbols.last().is_some_and(|&s| !grammar.ends_field(s)) {
            return Err("chunk ends inside a field".into());
        }
        for (at, &symbol) in symbols.iter().enumerate() {
            if symbol as usize >= alphabet {
                return Err("undefined stream symbol".into());
            }
            uses[symbol as usize] += 1;
            if at == 0 || grammar.ends_field(symbols[at - 1]) {
                starts[symbol as usize] += 1;
            }
            if at + 1 < symbols.len() && !grammar.ends_field(symbol) {
                left[symbol as usize] += 1;
                right[symbols[at + 1] as usize] += 1;
            }
        }
        let mut at = 0;
        while at < symbols.len() {
            let symbol = symbols[at];
            let mut end = at + 1;
            while end < symbols.len() && symbols[end] == symbol {
                end += 1;
            }
            if !grammar.ends_field(symbol) {
                let length = (end - at) as u64;
                if length >= 2 {
                    *expected.entry((symbol, symbol)).or_default() += length / 2;
                    overlaps[symbol as usize] += length - 1 - length / 2;
                }
                if end < symbols.len() {
                    *expected.entry((symbol, symbols[end])).or_default() += 1;
                }
            }
            at = end;
        }
    }
    if let Some(dense) = dense {
        if uses != dense.uses {
            return Err("dense symbol frequencies differ from the stream".into());
        }
        for symbol in 0..alphabet as u32 {
            if starts[symbol as usize] != dense.start_count(symbol)
                || (track && overlaps[symbol as usize] != dense.overlap_count(symbol))
            {
                return Err(format!("dense start/overlap mismatch for {symbol}"));
            }
        }
        expected.retain(|&(a, b), &mut count| {
            uses[a as usize] >= minimum
                && uses[b as usize] >= minimum
                && (a != b || uses[a as usize] / 2 >= minimum)
                && (!registry.retires_low_counts() || count >= minimum)
        });
    } else if track {
        for (symbol, &count) in overlaps.iter().enumerate() {
            if count != 0 {
                expected.insert((EOF, symbol as u32), count);
            }
        }
    }
    let mut actual = BTreeMap::new();
    registry.visit(|pair, count| {
        actual.insert(pair, count);
    });
    if expected != actual {
        let mismatch = expected
            .iter()
            .find(|(pair, count)| actual.get(pair) != Some(count));
        return Err(format!(
            "pair registry differs: expected {mismatch:?}; entries expected {}, actual {}",
            expected.len(),
            actual.len()
        ));
    }
    // Raw marginals are independently reconstructed even for omitted edges.
    let stats = if track {
        Some(match dense {
            Some(dense) => crate::scoring::Statistics::from_symbols(registry, dense, grammar),
            None => crate::scoring::Statistics::from_registry(registry, alphabet),
        })
    } else {
        None
    };
    if let Some(stats) = stats {
        if stats.left != left || stats.right != right || stats.edges != left.iter().sum() {
            return Err("raw adjacency marginals differ from the stream".into());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::Registry;

    #[test]
    fn audit_rejects_phantom_counts_even_when_the_stream_is_unchanged() {
        let grammar = Grammar::default();
        let mut block = PackedSymbols::new(9);
        for s in [97, 97, 97, 97, 97, 98, EOF, EOF] {
            block.push(s);
        }
        let mut registry = Registry::new(2).unwrap();
        let mut writer = registry.writer();
        writer.pair((97, 97), 2).unwrap();
        writer.pair((97, 98), 1).unwrap();
        writer.pair((98, EOF), 1).unwrap();
        writer.overlap(97, 2).unwrap();
        writer.finish().unwrap();
        check(
            std::slice::from_ref(&block),
            &grammar,
            &registry.freeze(),
            None,
            4,
            true,
        )
        .unwrap();
        let mut writer = registry.writer();
        writer.pair((97, 97), 1).unwrap();
        writer.finish().unwrap();
        assert!(check(&[block], &grammar, &registry.freeze(), None, 4, true).is_err());
    }
}
