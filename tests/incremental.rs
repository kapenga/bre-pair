use bre_pair::archive;
use bre_pair::batch::{self, Counting, PairCounter};
use bre_pair::engine::{self, Config};
use bre_pair::grammar::{Grammar, EOF};
use bre_pair::packed::{symbol_width, PackedSymbols};
use bre_pair::pair_map::PairMap;
use bre_pair::scoring::Model;

#[test]
fn delta_counts_match_full_recounts_with_run_parity_boundaries_and_reused_rules() {
    let mut grammar = Grammar::default();
    let double = grammar.add((0, 0)).unwrap();
    let ending = grammar.add((1, EOF)).unwrap();
    let alphabet = [0, 1, double, ending];
    for code in 0..4usize.pow(6) {
        let mut value = code;
        let mut input = PackedSymbols::new(10);
        for _ in 0..6 {
            input.push(alphabet[value % 4]);
            value /= 4;
        }
        let pairs: Vec<_> = PairCounter::from_buffer(&input, &grammar)
            .counts
            .keys()
            .copied()
            .collect();
        // Individual pairs plus simultaneous neighboring replacements. The
        // counter is also correct if a caller supplies overlapping candidates.
        for choice in 0..=pairs.len() {
            let mut replacements = PairMap::default();
            for (i, &pair) in pairs.iter().enumerate() {
                if choice == pairs.len() || choice == i {
                    replacements.insert(pair, grammar.add(pair).unwrap());
                }
            }
            for track in [false, true] {
                let mut initial = PairCounter::new(track);
                for symbol in input.iter() {
                    initial.push(symbol, &grammar);
                }
                let width = symbol_width(grammar.alphabet_size());
                let (output, mut actual) = batch::rewrite_counted(
                    &input,
                    &replacements,
                    &grammar,
                    width,
                    initial,
                    Counting::Incremental,
                );
                // Appending exercises the preserved counter state used by refill.
                let mut appended = output.clone();
                for symbol in [double, double, double, ending, 0, 0, EOF] {
                    actual.push(symbol, &grammar);
                    appended.push(symbol);
                }
                let mut expected = PairCounter::new(track);
                for symbol in appended.iter() {
                    expected.push(symbol, &grammar);
                }
                assert_eq!(
                    actual.counts, expected.counts,
                    "input {code}, choice {choice}"
                );
                assert_eq!(actual.self_overlaps, expected.self_overlaps);
                assert!(actual.counts.values().all(|&n| n > 0));
            }
        }
    }
}

#[test]
fn maintenance_modes_preserve_archives_across_models_refills_and_pruning() {
    let fields = vec![
        b"aaaaaabacabad ababcabc xyz xyz xyz\n".repeat(12),
        vec![],
        b"abcabca aaaaab aabaa ababab\n".repeat(8),
    ];
    for model in [
        Model::Frequency,
        Model::Conditional,
        Model::Pmi,
        Model::WeightedPmi,
        Model::Likelihood,
    ] {
        for budget in [64, 4096] {
            let mut config = Config {
                score: model,
                pair_counting: Counting::Full,
                buffer_bytes: budget,
                minimum_count: 2,
                ..Config::default()
            };
            let reference = engine::compress(&fields, &config, |_| {}).unwrap();
            let reference = archive::encode(&reference.compressed).unwrap();
            for mode in [Counting::Reuse, Counting::Incremental, Counting::Adaptive] {
                config.pair_counting = mode;
                let output = engine::compress(&fields, &config, |_| {}).unwrap();
                assert_eq!(archive::encode(&output.compressed).unwrap(), reference);
                assert_eq!(output.compressed.decode().unwrap(), fields);
            }
        }
    }
    for mode in [
        Counting::Full,
        Counting::Reuse,
        Counting::Incremental,
        Counting::Adaptive,
    ] {
        let config = Config {
            pair_counting: mode,
            minimum_count: 2,
            score: Model::WeightedPmi,
            pruning: Some(bre_pair::prune::Checkpoint {
                after_round: 2,
                max_uses: 1,
            }),
            ..Config::default()
        };
        let result = engine::compress(&fields, &config, |_| {}).unwrap();
        assert_eq!(result.compressed.decode().unwrap(), fields);
    }
}
