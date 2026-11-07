//! Two complementary, global pair-type masks, followed by an unrestricted
//! second batch. Experimental probes never alter the ordinary trainer.
use crate::{
    batch::{self, Candidate, PairCounter},
    engine::Compressed,
    grammar::{Grammar, Pair, BASE_SYMBOLS, EOF},
    packed::{symbol_width, PackedSymbols},
    pair_map::PairMap,
    scoring::{Model, Statistics},
    search,
};

#[derive(Clone)]
pub struct State {
    pub grammar: Grammar,
    pub symbols: PackedSymbols,
    pub input_bytes: u64,
    pub fields: u64,
}

impl State {
    pub fn from_fields(fields: &[Vec<u8>]) -> Result<Self, String> {
        let mut size = 0usize;
        let mut input_bytes = 0u64;
        for field in fields {
            size = size
                .checked_add(field.len())
                .and_then(|n| n.checked_add(1))
                .ok_or("input size overflow")?;
            input_bytes = input_bytes
                .checked_add(field.len() as u64)
                .ok_or("input bytes overflow")?;
        }
        let mut symbols = PackedSymbols::with_capacity(9, size);
        for field in fields {
            for &byte in field {
                symbols.push(u32::from(byte));
            }
            symbols.push(EOF);
        }
        Ok(Self {
            grammar: Grammar::default(),
            symbols,
            input_bytes,
            fields: fields.len() as u64,
        })
    }
    pub fn compressed(&self) -> Compressed {
        Compressed {
            grammar: self.grammar.clone(),
            blocks: vec![self.symbols.clone()],
            input_bytes: self.input_bytes,
            field_count: self.fields,
        }
    }
}

#[derive(Clone, Copy)]
pub struct Options {
    pub minimum_count: u64,
    pub score: Model,
    /// Upper bound on first-layer new definitions. Total horizon <= twice this.
    pub layer_rules: usize,
}

#[derive(Debug)]
pub struct Stage {
    pub before_symbols: u64,
    pub after_symbols: u64,
    pub new_rules: usize,
    /// Whole types, with exact non-overlapping replacement counts.
    pub pairs: Vec<(Pair, u64)>,
}

pub struct World {
    pub name: &'static str,
    pub first: Stage,
    pub second: Stage,
    pub state: State,
}

pub struct Probe {
    pub worlds: Vec<World>,
    pub additional_rules: usize,
    pub effective_first_cap: usize,
    pub budget_retries: usize,
    pub eligible_types: usize,
    pub enabled_types: [usize; 2],
    /// Index 0 is the control; ties retain it.
    pub winner: usize,
}

fn candidates(state: &State, options: Options) -> Vec<Candidate> {
    // Exactly one counting namespace exists at a time. Association marginals
    // use ALL observations, before any on/off mask is applied.
    let counter = PairCounter::from_buffer_with_tracking(
        &state.symbols,
        &state.grammar,
        options.score != Model::Frequency,
    );
    let stats = (options.score != Model::Frequency)
        .then(|| Statistics::from_counter(&counter, state.grammar.alphabet_size()));
    let mut result: Vec<_> = counter
        .counts
        .iter()
        .filter(|&(_, &n)| n >= options.minimum_count)
        .map(|(&pair, &count)| Candidate {
            pair,
            count,
            priority: stats
                .as_ref()
                .map_or(count as f64, |s| s.score(options.score, pair, count)),
        })
        .collect();
    result.sort_unstable_by(|a, b| {
        b.priority
            .total_cmp(&a.priority)
            .then_with(|| b.count.cmp(&a.count))
            .then_with(|| a.pair.cmp(&b.pair))
    });
    result
}

fn choose(state: &State, candidates: Vec<Candidate>) -> Vec<Candidate> {
    batch::select_candidates(
        state.symbols.iter(),
        candidates,
        &search::Options {
            certify: false,
            ..search::Options::default()
        },
    )
    .pairs
}

fn fresh(grammar: &Grammar, selected: &[Candidate]) -> usize {
    selected
        .iter()
        .filter(|c| grammar.find(c.pair).is_none())
        .count()
}

fn apply(
    state: &State,
    selected: &[Candidate],
    maximum_new: usize,
) -> Result<(State, Stage), String> {
    let mut grammar = state.grammar.clone();
    let mut replacements = PairMap::default();
    let mut pairs = Vec::new();
    let mut added = 0;
    let mut removed = 0u64;
    for candidate in selected {
        if grammar.find(candidate.pair).is_none() {
            if added == maximum_new {
                break;
            }
            added += 1;
        }
        let id = grammar.add(candidate.pair)?;
        replacements.insert(candidate.pair, id);
        pairs.push((candidate.pair, candidate.count));
        removed = removed
            .checked_add(candidate.count)
            .ok_or("replacement count overflow")?;
    }
    let length = (state.symbols.len() as u64)
        .checked_sub(removed)
        .ok_or("invalid replacement prediction")?;
    let length = usize::try_from(length).map_err(|_| "rewritten length exceeds address space")?;
    let mut output = PackedSymbols::with_capacity(symbol_width(grammar.alphabet_size()), length);
    let mut symbols = state.symbols.iter();
    let mut pending = symbols.next();
    while let Some(a) = pending {
        let next = symbols.next();
        if let Some(id) = next.and_then(|b| replacements.get(&(a, b))) {
            output.push(*id);
            pending = symbols.next();
        } else {
            output.push(a);
            pending = next;
        }
    }
    if output.len() != length {
        return Err("actual world rewrite differs from pair counts".into());
    }
    let stage = Stage {
        before_symbols: state.symbols.len() as u64,
        after_symbols: output.len() as u64,
        new_rules: added,
        pairs,
    };
    Ok((
        State {
            grammar,
            symbols: output,
            input_bytes: state.input_bytes,
            fields: state.fields,
        },
        stage,
    ))
}

/// Advance the unchanged all-enabled reference trajectory by one complete batch.
pub fn advance(state: &State, options: Options) -> Result<Option<State>, String> {
    validate(options)?;
    let selected = choose(state, candidates(state, options));
    if selected.is_empty() {
        return Ok(None);
    }
    Ok(Some(apply(state, &selected, usize::MAX)?.0))
}

fn validate(options: Options) -> Result<(), String> {
    if options.minimum_count < 2
        || options.layer_rules == 0
        || options.layer_rules.checked_mul(2).is_none()
    {
        return Err(
            "minimum count must be >=2 and the positive layer budget must fit usize".into(),
        );
    }
    Ok(())
}

/// Mask BEFORE conflict selection. This lets a deferred strong pair stop
/// blocking alternatives in that world. Layer two admits every eligible type.
/// No occurrence gets its own independent merge/no-merge switch.
pub fn probe(state: &State, options: Options) -> Result<Probe, String> {
    probe_with_budget(state, options, usize::MAX)
}

/// Limit the complete two-layer horizon, including a partial final decision.
pub fn probe_with_budget(
    state: &State,
    options: Options,
    maximum_additional: usize,
) -> Result<Probe, String> {
    validate(options)?;
    let all = candidates(state, options);
    let types = all.len();
    let names = ["all-enabled", "alternating-A", "alternating-B"];
    let selections: Vec<_> = (0..3)
        .map(|world| {
            let enabled = all
                .iter()
                .enumerate()
                .filter(|(rank, _)| world == 0 || rank % 2 == world - 1)
                .map(|(_, &c)| c)
                .collect();
            choose(state, enabled)
        })
        .collect();
    drop(all);
    let mut cap = options.layer_rules.min(maximum_additional);
    let mut retries = 0;
    loop {
        // Keep the three first-layer streams, but recount worlds sequentially.
        // No per-world counting tables or worker-private maps are retained.
        let mut prepared = Vec::new();
        let mut target = (2 * options.layer_rules).min(maximum_additional);
        let mut largest_first = 0;
        for selected in &selections {
            // At zero first-layer budget defer the whole first layer. This
            // avoids a free reuse destroying another world's last fresh rule
            // and forcing an unproductive zero-rule horizon in the control.
            let selected = if cap == 0 {
                &[][..]
            } else {
                selected.as_slice()
            };
            let (first_state, first) = apply(state, selected, cap)?;
            let second = choose(&first_state, candidates(&first_state, options));
            target = target.min(
                first
                    .new_rules
                    .checked_add(fresh(&first_state.grammar, &second))
                    .ok_or("rule horizon overflow")?,
            );
            largest_first = largest_first.max(first.new_rules);
            prepared.push((first_state, first, second));
        }
        if target < largest_first {
            // The initial cap committed more rules than another world can
            // reach in two layers. Retry every world with a smaller shared cap.
            cap = target;
            retries += 1;
            continue;
        }
        let mut worlds = Vec::new();
        for ((first_state, first, selected), name) in prepared.into_iter().zip(names) {
            let (output, second) = apply(&first_state, &selected, target - first.new_rules)?;
            if first.new_rules + second.new_rules != target {
                return Err("worlds did not reach the same rule budget".into());
            }
            worlds.push(World {
                name,
                first,
                second,
                state: output,
            });
        }
        let winner = worlds
            .iter()
            .enumerate()
            .min_by_key(|(i, w)| (w.state.symbols.len(), *i))
            .unwrap()
            .0;
        debug_assert!(worlds.iter().all(|w| w.state.grammar.alphabet_size()
            == BASE_SYMBOLS + state.grammar.rules.len() + target));
        return Ok(Probe {
            worlds,
            additional_rules: target,
            effective_first_cap: cap,
            budget_retries: retries,
            eligible_types: types,
            enabled_types: [types.div_ceil(2), types / 2],
            winner,
        });
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Policy {
    Standard,
    Control,
    Lookahead,
}

impl Policy {
    pub fn name(self) -> &'static str {
        match self {
            Self::Standard => "standard",
            Self::Control => "control",
            Self::Lookahead => "lookahead",
        }
    }
}

#[derive(Debug)]
pub struct Decision {
    pub index: usize,
    pub before_symbols: u64,
    pub after_symbols: u64,
    pub total_rules: usize,
    pub new_rules: usize,
    pub winner: usize,
    pub world_symbols: Vec<u64>,
    pub first_rules: usize,
    pub second_rules: usize,
    pub budget_retries: usize,
}

pub struct Training {
    pub state: State,
    pub decisions: usize,
    pub chosen_worlds: [usize; 3],
    pub converged: bool,
    pub stopped_at_rule_limit: bool,
}

/// Commit both layers of the selected world, then construct fresh masks from
/// that committed state. A decision's local saving is not a full-run bound.
pub fn train(
    mut state: State,
    options: Options,
    policy: Policy,
    maximum_rules: Option<usize>,
    mut on_commit: impl FnMut(&Decision, &State),
) -> Result<Training, String> {
    validate(options)?;
    if maximum_rules.is_some_and(|n| n < state.grammar.rules.len()) {
        return Err("rule limit is below the starting grammar size".into());
    }
    let mut decisions = 0;
    let mut chosen_worlds = [0; 3];
    loop {
        let remaining = maximum_rules.map_or(usize::MAX, |n| n - state.grammar.rules.len());
        if remaining == 0 {
            return Ok(Training {
                state,
                decisions,
                chosen_worlds,
                converged: false,
                stopped_at_rule_limit: true,
            });
        }
        let before = state.symbols.len() as u64;
        let old_rules = state.grammar.rules.len();
        let (next, winner, world_symbols, first_rules, second_rules, budget_retries) =
            if policy == Policy::Standard {
                let selected = choose(&state, candidates(&state, options));
                if selected.is_empty() {
                    return Ok(Training {
                        state,
                        decisions,
                        chosen_worlds,
                        converged: true,
                        stopped_at_rule_limit: false,
                    });
                }
                let (next, stage) = apply(&state, &selected, remaining)?;
                let count = next.symbols.len() as u64;
                (next, 0, vec![count], stage.new_rules, 0, 0)
            } else {
                let mut preview = probe_with_budget(&state, options, remaining)?;
                let winner = if policy == Policy::Lookahead {
                    preview.winner
                } else {
                    0
                };
                let counts = preview
                    .worlds
                    .iter()
                    .map(|w| w.state.symbols.len() as u64)
                    .collect();
                if preview.worlds[winner].state.symbols.len() == state.symbols.len() {
                    if preview.eligible_types != 0 {
                        return Err("two-layer training stalled with eligible pair types".into());
                    }
                    return Ok(Training {
                        state,
                        decisions,
                        chosen_worlds,
                        converged: true,
                        stopped_at_rule_limit: false,
                    });
                }
                let world = preview.worlds.swap_remove(winner);
                (
                    world.state,
                    winner,
                    counts,
                    world.first.new_rules,
                    world.second.new_rules,
                    preview.budget_retries,
                )
            };
        if next.symbols.len() as u64 >= before || next.grammar.rules.len() < old_rules {
            return Err("committed decision failed to reduce the symbol stream".into());
        }
        decisions += 1;
        chosen_worlds[winner] += 1;
        let decision = Decision {
            index: decisions,
            before_symbols: before,
            after_symbols: next.symbols.len() as u64,
            total_rules: next.grammar.rules.len(),
            new_rules: next.grammar.rules.len() - old_rules,
            winner,
            world_symbols,
            first_rules,
            second_rules,
            budget_retries,
        };
        on_commit(&decision, &next);
        state = next;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{archive, engine};
    use std::collections::HashSet;

    fn options() -> Options {
        Options {
            minimum_count: 2,
            score: Model::Frequency,
            layer_rules: 2,
        }
    }

    fn verify(fields: &[Vec<u8>], probe: &Probe, source_rules: usize) {
        for world in &probe.worlds {
            assert_eq!(
                world.state.grammar.rules.len(),
                source_rules + probe.additional_rules
            );
            assert_eq!(
                world.first.new_rules + world.second.new_rules,
                probe.additional_rules
            );
            let archive = archive::encode(&world.state.compressed()).unwrap();
            assert_eq!(archive::decode(&archive).unwrap().decode().unwrap(), fields);
            assert_eq!(
                world.first.before_symbols - world.first.after_symbols,
                world.first.pairs.iter().map(|x| x.1).sum::<u64>()
            );
            assert_eq!(
                world.second.before_symbols - world.second.after_symbols,
                world.second.pairs.iter().map(|x| x.1).sum::<u64>()
            );
        }
    }

    #[test]
    fn masks_are_complementary_and_apply_to_whole_types_before_conflicts() {
        let mut fields = vec![b"abc".to_vec(); 8];
        fields.extend(vec![b"ab".to_vec(); 2]);
        let source = State::from_fields(&fields).unwrap();
        let options = Options {
            minimum_count: 4,
            layer_rules: 1,
            ..options()
        };
        let all = candidates(&source, options);
        let a: HashSet<_> = all.iter().step_by(2).map(|c| c.pair).collect();
        let b: HashSet<_> = all.iter().skip(1).step_by(2).map(|c| c.pair).collect();
        assert!(a.is_disjoint(&b));
        assert_eq!(a.len() + b.len(), all.len());
        let result = probe(&source, options).unwrap();
        for &(p, n) in &result.worlds[1].first.pairs {
            assert!(a.contains(&p));
            assert_eq!(all.iter().find(|c| c.pair == p).unwrap().count, n);
        }
        for &(p, n) in &result.worlds[2].first.pairs {
            assert!(b.contains(&p));
            assert_eq!(all.iter().find(|c| c.pair == p).unwrap().count, n);
        }
        // AB is stronger but deferred in B, so BC may merge at all eight
        // occurrences. The disabled AB type must not block it in this world.
        assert!(result.worlds[2].first.pairs.contains(&((98, 99), 8)));
        assert!(!result.worlds[2]
            .first
            .pairs
            .iter()
            .any(|(p, _)| *p == (97, 98)));
        verify(&fields, &result, 0);
    }

    #[test]
    fn self_runs_use_nonoverlapping_counts_and_fields_never_join() {
        let fields = vec![b"aaaaa".to_vec(); 8];
        let source = State::from_fields(&fields).unwrap();
        let result = probe(
            &source,
            Options {
                minimum_count: 8,
                layer_rules: 1,
                ..options()
            },
        )
        .unwrap();
        assert!(result.worlds[1].first.pairs.contains(&((97, 97), 16)));
        assert_eq!(
            result.worlds[1].first.before_symbols - result.worlds[1].first.after_symbols,
            16
        );
        verify(&fields, &result, 0);
    }

    #[test]
    fn exhaustive_short_worlds_match_budgets_and_round_trip_with_binary_eof() {
        for len in 0..=7 {
            for pattern in 0..(1usize << len) {
                let raw: Vec<_> = (0..len)
                    .map(|i| if pattern >> i & 1 == 0 { 0 } else { 255 })
                    .collect();
                let fields = vec![raw.clone(), vec![], raw];
                let source = State::from_fields(&fields).unwrap();
                for cap in [1, 2] {
                    let result = probe(
                        &source,
                        Options {
                            layer_rules: cap,
                            ..options()
                        },
                    )
                    .unwrap();
                    verify(&fields, &result, 0);
                    if let Some(next) = advance(&source, options()).unwrap() {
                        let result = probe(
                            &next,
                            Options {
                                layer_rules: cap,
                                ..options()
                            },
                        )
                        .unwrap();
                        verify(&fields, &result, next.grammar.rules.len());
                    }
                }
            }
        }
    }

    #[test]
    fn unmasked_trajectory_matches_the_existing_engine() {
        let fields = vec![
            b"abcabca\n".repeat(8),
            b"aaaaa".repeat(8),
            vec![],
            vec![255, 0, 255, 0],
        ];
        for score in [Model::Frequency, Model::WeightedPmi] {
            let options = Options { score, ..options() };
            let mut state = State::from_fields(&fields).unwrap();
            while let Some(next) = advance(&state, options).unwrap() {
                state = next;
            }
            let config = engine::Config {
                buffer_bytes: 1024 * 1024,
                minimum_count: 2,
                score,
                search: search::Options {
                    certify: false,
                    ..search::Options::default()
                },
                ..engine::Config::default()
            };
            let reference = engine::compress(&fields, &config, |_| {})
                .unwrap()
                .compressed;
            assert_eq!(
                archive::encode(&state.compressed()).unwrap(),
                archive::encode(&reference).unwrap()
            );
        }
    }

    #[test]
    fn wide_minimum_and_exhausted_inputs_are_explicit_and_ties_keep_control() {
        let fields = vec![b"aaaaa".to_vec()];
        let source = State::from_fields(&fields).unwrap();
        let result = probe(
            &source,
            Options {
                minimum_count: 1u64 << 32,
                ..options()
            },
        )
        .unwrap();
        assert_eq!(result.eligible_types, 0);
        assert_eq!(result.additional_rules, 0);
        assert_eq!(result.winner, 0);
        verify(&fields, &result, 0);
        assert!(probe(
            &source,
            Options {
                minimum_count: 1,
                ..options()
            }
        )
        .is_err());
        assert!(probe(
            &source,
            Options {
                layer_rules: 0,
                ..options()
            }
        )
        .is_err());
    }

    #[test]
    fn rolling_training_reaches_convergence_and_preserves_each_commit() {
        for len in 0..=6 {
            for pattern in 0..(1usize << len) {
                let raw: Vec<_> = (0..len)
                    .map(|i| if pattern >> i & 1 == 0 { 0 } else { 255 })
                    .collect();
                let fields = vec![raw.clone(), raw, vec![]];
                for policy in [Policy::Standard, Policy::Control, Policy::Lookahead] {
                    let mut previous =
                        (fields.iter().map(|f| f.len() + 1).sum::<usize>() as u64, 0);
                    let trained = train(
                        State::from_fields(&fields).unwrap(),
                        Options {
                            layer_rules: 1,
                            ..options()
                        },
                        policy,
                        None,
                        |d, state| {
                            assert_eq!(d.before_symbols, previous.0);
                            assert!(d.after_symbols < previous.0);
                            assert!(d.total_rules >= previous.1);
                            if policy == Policy::Lookahead {
                                assert_eq!(d.after_symbols, *d.world_symbols.iter().min().unwrap());
                            }
                            assert_eq!(state.compressed().decode().unwrap(), fields);
                            previous = (d.after_symbols, d.total_rules);
                        },
                    )
                    .unwrap();
                    assert!(trained.converged);
                    assert!(!trained.stopped_at_rule_limit);
                    assert!(advance(&trained.state, options()).unwrap().is_none());
                }
            }
        }
    }

    #[test]
    fn rolling_partial_horizons_obey_odd_rule_limits_and_can_continue() {
        let fields = vec![
            b"abcabacababacdabcabc\n".repeat(8),
            b"aaaaabaaaba".repeat(8),
            vec![],
        ];
        for policy in [Policy::Standard, Policy::Control, Policy::Lookahead] {
            for cap in [0, 1, 3, 7, 13] {
                let first = train(
                    State::from_fields(&fields).unwrap(),
                    Options {
                        layer_rules: 2,
                        ..options()
                    },
                    policy,
                    Some(cap),
                    |_, s| assert_eq!(s.compressed().decode().unwrap(), fields),
                )
                .unwrap();
                assert!(first.state.grammar.rules.len() <= cap);
                assert!(first.converged || first.stopped_at_rule_limit);
                if first.stopped_at_rule_limit {
                    assert_eq!(first.state.grammar.rules.len(), cap);
                }
                let continued = train(first.state, options(), policy, None, |_, _| {}).unwrap();
                assert!(continued.converged);
                assert!(advance(&continued.state, options()).unwrap().is_none());
                assert_eq!(continued.state.compressed().decode().unwrap(), fields);
            }
        }
    }
}
