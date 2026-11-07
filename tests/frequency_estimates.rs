use bre_pair::{
    engine::Compressed,
    grammar::{Grammar, BASE_SYMBOLS, EOF},
    packed::{symbol_width, PackedSymbols},
    tokenizer::{Model, Token, Tokenizer},
    vocabulary::frequency_estimates,
};
use std::collections::HashMap;

fn model(extra: &[(&[u8], bool)]) -> Model {
    let mut tokens: Vec<_> = (0..=255)
        .map(|b| Token {
            bytes: vec![b],
            ends_field: false,
        })
        .chain([Token {
            bytes: vec![],
            ends_field: true,
        }])
        .collect();
    tokens.extend(extra.iter().map(|&(bytes, ends_field)| Token {
        bytes: bytes.to_vec(),
        ends_field,
    }));
    Model::new(tokens).unwrap()
}
fn source(grammar: Grammar, roots: &[u32]) -> Compressed {
    let mut block = PackedSymbols::new(symbol_width(grammar.alphabet_size()));
    let (mut bytes, mut fields) = (0, 0);
    for &id in roots {
        block.push(id);
        grammar.visit_terminals(id, |t| if t == EOF { fields += 1 } else { bytes += 1 });
    }
    Compressed {
        grammar,
        blocks: vec![block],
        input_bytes: bytes,
        field_count: fields,
    }
}
fn expansion(grammar: &Grammar, id: u32) -> Token {
    let mut bytes = vec![];
    grammar.visit_terminals(id, |t| {
        if t < EOF {
            bytes.push(t as u8)
        }
    });
    Token {
        bytes,
        ends_field: grammar.ends_field(id),
    }
}
fn walk(source: &Compressed, model: &Model, stop: bool) -> Vec<u64> {
    let ids: HashMap<_, _> = model
        .tokens
        .iter()
        .enumerate()
        .map(|(id, t)| (t, id))
        .collect();
    let mut counts = vec![0; model.tokens.len()];
    for block in &source.blocks {
        for root in block.iter() {
            let mut stack = vec![root];
            while let Some(id) = stack.pop() {
                if let Some(&target) = ids.get(&expansion(&source.grammar, id)) {
                    counts[target] += 1;
                    if stop {
                        continue;
                    }
                }
                if id as usize >= BASE_SYMBOLS {
                    let r = &source.grammar.rules[id as usize - BASE_SYMBOLS];
                    stack.extend([r.right, r.left]);
                }
            }
        }
    }
    counts
}

#[test]
fn estimator_agrees_with_independent_visits_for_aliases_nested_rules_and_eof() {
    let mut g = Grammar::default();
    let ab = g.add((97, 98)).unwrap();
    let bc = g.add((98, 99)).unwrap();
    let abc = g.add((ab, 99)).unwrap();
    let alias = g.add((97, bc)).unwrap();
    let ending = g.add((alias, EOF)).unwrap();
    let aa = g.add((97, 97)).unwrap();
    let aaaa = g.add((aa, aa)).unwrap();
    let raw = source(g, &[abc, EOF, ending, aaaa, EOF]);
    for chosen in [
        model(&[]),
        model(&[(b"abc", false), (b"abc", true), (b"aa", false)]),
        model(&[
            (b"ab", false),
            (b"bc", false),
            (b"aaaa", false),
            (b"missing", false),
        ]),
    ] {
        let e = frequency_estimates::from_grammar(&raw, &chosen).unwrap();
        assert_eq!(e.inclusive_uses, walk(&raw, &chosen, false));
        assert_eq!(e.collapsed_uses, walk(&raw, &chosen, true));
        assert_eq!(e.collapsed.bytes, raw.input_bytes);
        assert_eq!(e.collapsed.fields, raw.field_count);
    }
}

#[test]
fn cross_boundary_resegmentation_changes_counts_with_no_accounting_bug() {
    let mut g = Grammar::default();
    let ab = g.add((97, 98)).unwrap();
    let bc = g.add((98, 99)).unwrap();
    let raw = source(g, &[ab, 99, EOF, 97, bc, EOF]);
    let chosen = model(&[(b"bc", false)]);
    let predicted = frequency_estimates::from_grammar(&raw, &chosen).unwrap();
    assert_eq!(predicted.collapsed.tokens, 7);
    assert_eq!(predicted.collapsed_uses[BASE_SYMBOLS], 1);
    let tokenizer = Tokenizer::new(chosen).unwrap();
    let mut measured = vec![0; tokenizer.model().tokens.len()];
    for field in raw.decode().unwrap() {
        for id in tokenizer.encode(&field) {
            measured[id as usize] += 1;
        }
    }
    assert_eq!(measured.iter().sum::<u64>(), 6);
    assert_eq!(measured[BASE_SYMBOLS], 2);
    assert_ne!(predicted.collapsed_uses, measured);
}

#[test]
fn equal_token_totals_can_still_hide_large_per_token_errors() {
    let mut g = Grammar::default();
    let bc = g.add((98, 99)).unwrap();
    let raw = source(g, &[97, bc, EOF]);
    let chosen = model(&[(b"ab", false), (b"bc", false)]);
    let predicted = frequency_estimates::from_grammar(&raw, &chosen).unwrap();
    let tokenizer = Tokenizer::new(chosen).unwrap();
    let mut measured = vec![0; tokenizer.model().tokens.len()];
    for id in tokenizer.encode(b"abc") {
        measured[id as usize] += 1;
    }
    assert_eq!(predicted.collapsed.tokens, measured.iter().sum::<u64>());
    assert_ne!(predicted.collapsed_uses, measured);
    assert_eq!(predicted.collapsed_uses[BASE_SYMBOLS + 1], 1);
    assert_eq!(measured[BASE_SYMBOLS], 1);
}

#[test]
fn random_grammar_subsets_preserve_bytes_fields_and_match_the_walk_oracle() {
    let mut seed = 9182u64;
    let mut random = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed as usize
    };
    for _ in 0..30 {
        let mut g = Grammar::default();
        for _ in 0..24 {
            let mut left = (random() % g.alphabet_size()) as u32;
            if g.ends_field(left) {
                left = b'a' as u32;
            }
            let right = (random() % g.alphabet_size()) as u32;
            g.add((left, right)).unwrap();
        }
        let mut roots = vec![];
        for _ in 0..16 {
            let id = (random() % g.alphabet_size()) as u32;
            roots.push(id);
            if !g.ends_field(id) {
                roots.push(EOF);
            }
        }
        let raw = source(g, &roots);
        let mut chosen = model(&[]);
        for id in BASE_SYMBOLS..raw.grammar.alphabet_size() {
            let t = expansion(&raw.grammar, id as u32);
            if random() % 2 == 0 && !chosen.tokens.contains(&t) {
                chosen.tokens.push(t);
            }
        }
        let e = frequency_estimates::from_grammar(&raw, &chosen).unwrap();
        assert_eq!(e.collapsed_uses, walk(&raw, &chosen, true));
        assert_eq!(e.inclusive_uses, walk(&raw, &chosen, false));
        let tokenizer = Tokenizer::new(chosen).unwrap();
        let exact = raw
            .decode()
            .unwrap()
            .iter()
            .map(|f| tokenizer.encode(f).len() as u64)
            .sum::<u64>();
        assert!(e.collapsed.tokens >= exact);
    }
}

#[test]
fn estimate_matches_exact_counts_when_retained_pieces_preserve_the_partition() {
    let mut g = Grammar::default();
    let ab = g.add((0, 255)).unwrap();
    let ending = g.add((ab, EOF)).unwrap();
    let raw = source(g, &[ending, ending]);
    let chosen = model(&[(&[0, 255], true)]);
    let e = frequency_estimates::from_grammar(&raw, &chosen).unwrap();
    let tokenizer = Tokenizer::new(chosen).unwrap();
    let mut exact = vec![0; tokenizer.model().tokens.len()];
    for f in raw.decode().unwrap() {
        for id in tokenizer.encode(&f) {
            exact[id as usize] += 1;
        }
    }
    assert_eq!(e.collapsed_uses, exact);
    assert_eq!(e.collapsed.byte_tokens, 0);
    assert_eq!(e.collapsed.fields, 2);
}

#[test]
fn cached_counts_match_independent_collapses_after_multiple_vocabulary_changes() {
    let mut g = Grammar::default();
    let ab = g.add((97, 98)).unwrap();
    let bc = g.add((98, 99)).unwrap();
    let abc = g.add((ab, 99)).unwrap();
    let alias = g.add((97, bc)).unwrap();
    let ending = g.add((alias, EOF)).unwrap();
    let raw = source(g, &[abc, EOF, ending, 97, bc, EOF]);
    let pool = model(&[
        (b"ab", false),
        (b"bc", false),
        (b"abc", false),
        (b"abc", true),
    ]);
    let counter = frequency_estimates::Counter::new(&raw, &pool).unwrap();
    for mask in (0..16).chain((0..16).rev()) {
        let mut active = vec![1; pool.tokens.len()];
        for i in 0..4 {
            active[BASE_SYMBOLS + i] = u8::from(mask & (1 << i) != 0);
        }
        let subset = pool.subset(&active).unwrap();
        let expected = walk(&raw, &subset, true);
        let (actual, totals) = counter.count(&active).unwrap();
        for (id, token) in pool.tokens.iter().enumerate() {
            let expected = subset
                .tokens
                .iter()
                .position(|t| t == token)
                .map_or(0, |i| expected[i]);
            assert_eq!(actual[id], expected);
        }
        assert_eq!(totals.bytes, raw.input_bytes);
        assert_eq!(totals.fields, raw.field_count);
    }
    assert!(counter.count(&[]).is_err());
    assert!(counter.count(&vec![0; pool.tokens.len()]).is_err());
}

#[test]
fn estimated_selection_uses_cached_statistics_without_a_raw_corpus_path() {
    let mut g = Grammar::default();
    let ab = g.add((97, 98)).unwrap();
    let abab = g.add((ab, ab)).unwrap();
    let ending = g.add((abab, EOF)).unwrap();
    let raw = source(g, &vec![ending; 100]);
    let (pool, support) = Model::candidates(&raw, 128).unwrap();
    let mut rounds = 0;
    let selected = bre_pair::vocabulary::select_estimated(pool, &support, &raw, 258, |r| {
        rounds += 1;
        assert_eq!(r.estimated.bytes, 400);
        assert_eq!(r.estimated.fields, 100);
    })
    .unwrap();
    assert!(rounds > 0);
    assert_eq!(selected.tokens.len(), 258);
    assert_eq!(
        selected.tokens.last().unwrap(),
        &Token {
            bytes: b"abab".to_vec(),
            ends_field: true
        }
    );
}
