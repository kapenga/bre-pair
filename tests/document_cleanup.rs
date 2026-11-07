use bre_pair::{
    document_support::{CandidateSupport, Cleanup, Policy},
    engine::Compressed,
    grammar::{Grammar, BASE_SYMBOLS, EOF},
    packed::{symbol_width, PackedSymbols},
};
use std::{
    collections::HashSet,
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};
static NEXT: AtomicU64 = AtomicU64::new(0);

struct Fixture {
    root: PathBuf,
    source: Compressed,
    docs: Vec<(Vec<u32>, u8)>,
}
impl Fixture {
    fn new(grammar: Grammar, docs: Vec<(Vec<u32>, u8)>) -> Self {
        let root = std::env::temp_dir().join(format!(
            "brepair-cleanup-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&root).unwrap();
        let mut raw = Vec::new();
        let mut index = b"BREDOC01".to_vec();
        index.extend(104u32.to_le_bytes());
        index.extend(0u32.to_le_bytes());
        index.extend((docs.len() as u64).to_le_bytes());
        let mut blocks = Vec::new();
        let mut block = PackedSymbols::new(symbol_width(grammar.alphabet_size()));
        let mut field_count = 0;
        for (symbols, group) in &docs {
            let start = raw.len();
            let mut fields = 0u64;
            for &symbol in symbols {
                grammar.visit_terminals(symbol, |byte| {
                    if byte == EOF {
                        fields += 1;
                    } else {
                        raw.push(byte as u8);
                    }
                });
                block.push(symbol);
                if block.len() == 5 {
                    blocks.push(block);
                    block = PackedSymbols::new(symbol_width(grammar.alphabet_size()));
                }
            }
            field_count += fields;
            let mut record = [0u8; 104];
            record[32..64].fill(*group);
            record[76..84].copy_from_slice(&(start as u64).to_le_bytes());
            record[84..92].copy_from_slice(&((raw.len() - start) as u64).to_le_bytes());
            record[92..100].copy_from_slice(&fields.to_le_bytes());
            index.extend(record);
        }
        if !block.is_empty() {
            blocks.push(block);
        }
        fs::write(root.join("index"), index).unwrap();
        fs::write(root.join("train"), &raw).unwrap();
        Self {
            root,
            source: Compressed {
                grammar,
                blocks,
                input_bytes: raw.len() as u64,
                field_count,
            },
            docs,
        }
    }
    fn policy(&self, n: usize) -> Policy {
        Policy {
            index: self.root.join("index"),
            minimum_groups: n,
        }
    }
    fn protected(&self) -> Vec<bool> {
        let mut p = vec![false; self.source.grammar.alphabet_size()];
        p[..BASE_SYMBOLS].fill(true);
        p
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.root).unwrap();
    }
}

fn oracle(f: &Fixture, n: usize, protected: &[bool]) -> (Vec<bool>, Vec<u64>, Vec<u8>) {
    let mut docs = f.docs.clone();
    let mut active = protected.to_vec();
    let mut support = vec![0; active.len()];
    for id in (BASE_SYMBOLS..active.len()).rev() {
        if protected[id] {
            continue;
        }
        let groups: HashSet<_> = docs
            .iter()
            .filter(|(s, _)| s.contains(&(id as u32)))
            .map(|(_, g)| g)
            .collect();
        support[id] = groups.len().min(n) as u8;
        if groups.len() >= n {
            active[id] = true;
        } else {
            let r = &f.source.grammar.rules[id - BASE_SYMBOLS];
            for (symbols, _) in &mut docs {
                *symbols = symbols
                    .iter()
                    .flat_map(|&s| {
                        if s == id as u32 {
                            vec![r.left, r.right]
                        } else {
                            vec![s]
                        }
                    })
                    .collect();
            }
        }
    }
    let mut uses = vec![0; active.len()];
    for (symbols, _) in docs {
        for s in symbols {
            uses[s as usize] += 1;
        }
    }
    (active, uses, support)
}

#[test]
fn parent_first_collapse_rescues_children_and_deduplicates_aliases() {
    let mut grammar = Grammar::default();
    let ab = grammar.add((97, 98)).unwrap();
    let abc = grammar.add((ab, 99)).unwrap();
    let abx = grammar.add((ab, 120)).unwrap();
    let end = grammar.add((10, EOF)).unwrap();
    let a = grammar.add((abc, end)).unwrap();
    let b = grammar.add((abx, end)).unwrap();
    for groups in [[1, 2], [1, 1]] {
        let fixture = Fixture::new(
            grammar.clone(),
            vec![(vec![a], groups[0]), (vec![b], groups[1])],
        );
        let protected = fixture.protected();
        let plan = Cleanup::plan(&fixture.source, &fixture.policy(2), &protected).unwrap();
        let expected = oracle(&fixture, 2, &protected);
        assert_eq!(
            (plan.active.clone(), plan.uses.clone(), plan.groups.clone()),
            expected
        );
        assert!(!plan.active[a as usize]);
        assert!(!plan.active[b as usize]);
        assert_eq!(plan.active[ab as usize], groups[0] != groups[1]);
        if groups[0] != groups[1] {
            assert_eq!(plan.uses[ab as usize], 2);
            assert_eq!((plan.before_symbols, plan.after_symbols), (2, 6));
        }
        let mut protected = fixture.protected();
        protected[a as usize] = true;
        let plan = Cleanup::plan(&fixture.source, &fixture.policy(2), &protected).unwrap();
        assert!(plan.active[a as usize]);
        assert!(!plan.active[ab as usize]);
        assert_eq!(plan.uses[a as usize], 1);
    }
}

#[test]
fn capped_set_propagation_matches_independent_document_stream_expansion() {
    let mut state = 123456789u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    for _ in 0..24 {
        let mut grammar = Grammar::default();
        let mut choices = vec![97, 98, 99, 100];
        for _ in 0..24 {
            let a = choices[next() as usize % choices.len()];
            let b = choices[next() as usize % choices.len()];
            let s = grammar.add((a, b)).unwrap();
            choices.push(s);
        }
        let mut docs = Vec::new();
        for i in 0..18 {
            let mut roots = Vec::new();
            for _ in 0..2 {
                for _ in 0..3 {
                    roots.push(choices[next() as usize % choices.len()]);
                }
                roots.extend([10, EOF]);
            }
            docs.push((roots, (i % 9) as u8));
        }
        let fixture = Fixture::new(grammar, docs);
        for n in [1, 2, 4, 8, 64] {
            let candidates = CandidateSupport::plan(&fixture.source, &fixture.policy(n)).unwrap();
            let mut groups = vec![HashSet::new(); fixture.source.grammar.alphabet_size()];
            for (roots, group) in &fixture.docs {
                let mut seen = HashSet::new();
                let mut pending = roots.clone();
                while let Some(id) = pending.pop() {
                    if !seen.insert(id) {
                        continue;
                    }
                    groups[id as usize].insert(*group);
                    if id as usize >= BASE_SYMBOLS {
                        let rule = &fixture.source.grammar.rules[id as usize - BASE_SYMBOLS];
                        pending.extend([rule.left, rule.right]);
                    }
                }
            }
            for (id, observed) in groups.iter().enumerate().skip(BASE_SYMBOLS) {
                assert_eq!(candidates.groups[id], observed.len().min(n) as u8);
                assert_eq!(candidates.included[id], observed.len() >= n);
            }
            let mut protected = fixture.protected();
            for id in (BASE_SYMBOLS..protected.len()).step_by(7) {
                protected[id] = true;
            }
            let plan = Cleanup::plan(&fixture.source, &fixture.policy(n), &protected).unwrap();
            let expected = oracle(&fixture, n, &protected);
            assert_eq!(
                (plan.active, plan.uses, plan.groups),
                expected,
                "document floor {n}"
            );
        }
    }
}

#[test]
fn vocabulary_candidates_include_children_swallowed_by_saturated_parents() {
    let mut grammar = Grammar::default();
    let ab = grammar.add((97, 98)).unwrap();
    let abc = grammar.add((ab, 99)).unwrap();
    let end = grammar.add((10, EOF)).unwrap();
    let line = grammar.add((abc, end)).unwrap();
    let fixture = Fixture::new(
        grammar,
        (0..130).map(|i| (vec![line], (i % 65) as u8)).collect(),
    );
    let stream = Cleanup::plan(&fixture.source, &fixture.policy(64), &fixture.protected()).unwrap();
    assert!(stream.active[line as usize]);
    assert!(!stream.active[ab as usize]);
    let candidates = CandidateSupport::plan(&fixture.source, &fixture.policy(64)).unwrap();
    for id in [ab, abc, end, line] {
        assert!(candidates.included[id as usize]);
        assert_eq!(candidates.groups[id as usize], 64);
    }
    let aliases = Fixture::new(
        fixture.source.grammar.clone(),
        (0..130).map(|_| (vec![line], 1)).collect(),
    );
    let candidates = CandidateSupport::plan(&aliases.source, &aliases.policy(64)).unwrap();
    assert!(!candidates.included[ab as usize]);
    assert_eq!(candidates.groups[ab as usize], 1);
}

#[test]
fn cleanup_rejects_holdout_bad_boundaries_and_missing_mandatory_symbols() {
    let fixture = Fixture::new(
        Grammar::default(),
        vec![(vec![97, 10, EOF], 1), (vec![98, 10, EOF], 2)],
    );
    let mut protected = fixture.protected();
    protected[EOF as usize] = false;
    assert!(Cleanup::plan(&fixture.source, &fixture.policy(2), &protected).is_err());
    let mut index = fs::read(fixture.root.join("index")).unwrap();
    index[12] = 1;
    fs::write(fixture.root.join("index"), &index).unwrap();
    assert!(Cleanup::plan(&fixture.source, &fixture.policy(2), &fixture.protected()).is_err());
    index[12] = 0;
    index[24 + 84..24 + 92].copy_from_slice(&1u64.to_le_bytes());
    index[24 + 104 + 76..24 + 104 + 84].copy_from_slice(&1u64.to_le_bytes());
    index[24 + 104 + 84..24 + 104 + 92].copy_from_slice(&3u64.to_le_bytes());
    fs::write(fixture.root.join("index"), index).unwrap();
    assert!(Cleanup::plan(&fixture.source, &fixture.policy(2), &fixture.protected()).is_err());
}

#[test]
fn public_vocabulary_stage_filters_before_selection_and_freezes_qualified_tokens() {
    let mut grammar = Grammar::default();
    let ab = grammar.add((97, 98)).unwrap();
    let abc = grammar.add((ab, 99)).unwrap();
    let abx = grammar.add((ab, 120)).unwrap();
    let end = grammar.add((10, EOF)).unwrap();
    let a = grammar.add((abc, end)).unwrap();
    let b = grammar.add((abx, end)).unwrap();
    let fixture = Fixture::new(grammar, vec![(vec![a], 1), (vec![b], 2)]);
    fs::write(
        fixture.root.join("candidates.brp"),
        bre_pair::archive::encode(&fixture.source).unwrap(),
    )
    .unwrap();
    let result = std::process::Command::new(env!("CARGO_BIN_EXE_brepair"))
        .current_dir(&fixture.root)
        .args([
            "vocab",
            "candidates.brp",
            "train",
            "tokens.brt",
            "--vocab-size",
            "259",
            "--min-token-uses",
            "1",
            "--document-index",
            "index",
            "--min-token-documents",
            "2",
            "--threads",
            "2",
            "--report",
            "vocabulary.json",
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let model =
        bre_pair::tokenizer::Model::from_bytes(&fs::read(fixture.root.join("tokens.brt")).unwrap())
            .unwrap();
    assert_eq!(model.tokens.len(), 259);
    assert!(model
        .tokens
        .iter()
        .any(|t| t.bytes == b"ab" && !t.ends_field));
    assert!(model
        .tokens
        .iter()
        .any(|t| t.bytes == b"\n" && t.ends_field));
    assert!(model
        .tokens
        .iter()
        .all(|t| t.bytes != b"abc\n" && t.bytes != b"abx\n"));
    let report = fs::read_to_string(fixture.root.join("vocabulary.json")).unwrap();
    assert!(report.contains("\"minimum_training_document_groups\":2"));
    assert!(report.contains("\"document_cleanup\":{\"removed_rules\":4,"));
    let result = std::process::Command::new(env!("CARGO_BIN_EXE_brepair"))
        .current_dir(&fixture.root)
        .args([
            "vocab",
            "candidates.brp",
            "train",
            "grammar.brt",
            "--vocab-size",
            "259",
            "--min-token-uses",
            "1",
            "--document-index",
            "index",
            "--min-token-documents",
            "2",
            "--candidate-document-support",
            "grammar",
            "--threads",
            "2",
            "--report",
            "grammar.json",
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(
        fs::read(fixture.root.join("grammar.brt")).unwrap(),
        model.to_bytes()
    );
    let report = fs::read_to_string(fixture.root.join("grammar.json")).unwrap();
    assert!(report.contains("\"candidate_document_support\":\"grammar\""));
    assert!(report.contains("\"minimum_learned_document_groups_lower_bound\":2"));
    assert!(report.contains("\"document_cleanup\":null"));
    for mode in ["grammar", "invalid"] {
        let rejected = std::process::Command::new(env!("CARGO_BIN_EXE_brepair"))
            .current_dir(&fixture.root)
            .args([
                "vocab",
                "candidates.brp",
                "train",
                "unqualified.brt",
                "--candidate-document-support",
                mode,
            ])
            .output()
            .unwrap();
        assert!(!rejected.status.success());
        assert!(!fixture.root.join("unqualified.brt").exists());
    }
    let failed = std::process::Command::new(env!("CARGO_BIN_EXE_brepair"))
        .current_dir(&fixture.root)
        .args([
            "vocab",
            "candidates.brp",
            "train",
            "impossible.brt",
            "--vocab-size",
            "260",
            "--min-token-uses",
            "1",
            "--document-index",
            "index",
            "--min-token-documents",
            "2",
        ])
        .output()
        .unwrap();
    assert!(!failed.status.success());
    assert!(!fixture.root.join("impossible.brt").exists());
}

#[test]
fn raw_discovery_reports_exact_rounds_buffer_and_selected_frequencies() {
    let fixture = Fixture::new(
        Grammar::default(),
        vec![
            (vec![97, 98, 99, 10, EOF], 1),
            (vec![97, 98, 120, 10, EOF], 2),
        ],
    );
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_brepair"))
        .current_dir(&fixture.root)
        .args([
            "train",
            "train",
            "raw.brp",
            "--min-count",
            "2",
            "--threads",
            "2",
            "--memory",
            "64MB",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("\"phase\":\"packing\",\"round\":0,\"done\":0,\"total\":8,"));
    assert!(stderr.contains("\"phase\":\"packing\",\"round\":0,\"done\":8,\"total\":8,"));
    let row = stderr
        .lines()
        .find(|line| line.starts_with("codec progress "))
        .unwrap();
    for field in [
        "\"round\":1,",
        "\"symbols\":6,",
        "\"rules_created\":2,",
        "\"packed_bytes\":8,",
        "\"minimum_selected_count\":2,",
        "\"maximum_selected_count\":2,",
    ] {
        assert!(row.contains(field), "missing {field}: {row}");
    }
    let decoded =
        bre_pair::archive::decode(&fs::read(fixture.root.join("raw.brp")).unwrap()).unwrap();
    assert_eq!(decoded.decode().unwrap(), fixture.source.decode().unwrap());
}
