//! Experimental pruning. Larger rules survive by inlining retired
//! children. Eligibility is evaluated once on a snapshot, never recursively
//! reclassified while expanding its data.
use crate::batch::PairCounter;
use crate::engine::Compressed;
use crate::grammar::{Grammar, Symbol, BASE_SYMBOLS};
use crate::maximal::NativeCompressed;
use crate::packed::{symbol_width, PackedSymbols};

#[derive(Clone, Copy, Debug)]
pub struct Checkpoint {
    pub after_round: usize,
    /// Retire a learned symbol with at most this many direct data occurrences.
    pub max_uses: u64,
}

#[derive(Debug)]
pub struct Event {
    pub after_round: usize,
    pub max_uses: u64,
    pub selected_rules: usize,
    pub before_tokens: usize,
    pub after_tokens: usize,
    pub seconds: f64,
}

pub struct State {
    /// Indexed by historical rule offset, not by full symbol ID.
    pub retired: Vec<bool>,
    pub event: Event,
    pub reactivated: usize,
}

pub struct Plan {
    pub retired: Vec<bool>,
    pub selected_rules: usize,
    pub before_tokens: usize,
    pub added_tokens: usize,
}

pub struct CostPlan {
    pub plan: Plan,
    pub baseline_bytes: usize,
    pub predicted_bytes: usize,
    pub choice: &'static str,
}

struct Profile {
    bytes: usize,
    tokens: usize,
}

/// Exact size of the native packed archive, without constructing its payload.
/// This objective includes progressive dictionary widths, gamma arities,
/// global data width, and per-block padding. It does not model XZ or Huffman.
fn profile(
    source: &Compressed,
    direct: &[u64],
    reachable: &[bool],
    retired: &[bool],
) -> Result<Profile, String> {
    let mut lengths = vec![1usize; BASE_SYMBOLS];
    let mut rules = 0usize;
    let mut dictionary_bits = 0u128;
    let mut tokens = source.token_count() as u128;
    for (i, rule) in source.grammar.rules.iter().enumerate() {
        let length = lengths[rule.left as usize]
            .checked_add(lengths[rule.right as usize])
            .ok_or("pruning length overflow")?;
        if reachable[i] && !retired[i] {
            let gamma = 2 * (usize::BITS - (length - 1).leading_zeros()) - 1;
            dictionary_bits +=
                gamma as u128 + length as u128 * symbol_width(BASE_SYMBOLS + rules) as u128;
            rules += 1;
            lengths.push(1);
        } else {
            lengths.push(length);
            tokens += direct[i] as u128 * (length - 1) as u128;
        }
    }
    let width = symbol_width(BASE_SYMBOLS + rules) as u128;
    let data_bytes = if source.blocks.len() == 1 {
        (tokens * width).div_ceil(8)
    } else {
        source
            .blocks
            .iter()
            .map(|block| {
                let count: u128 = block.iter().map(|s| lengths[s as usize] as u128).sum();
                (count * width).div_ceil(8)
            })
            .sum()
    };
    Ok(Profile {
        tokens: usize::try_from(tokens).map_err(|_| "pruning token overflow")?,
        bytes: usize::try_from(
            40 + 9 * source.blocks.len() as u128 + dictionary_bits.div_ceil(8) + data_bytes,
        )
        .map_err(|_| "pruning archive overflow")?,
    })
}

/// Cost-aware post-pruning. A reverse DAG pass propagates both direct data
/// occurrences and references from surviving parents. Inlining is locally
/// allowed only when its estimated saving is positive; arity growth is charged
/// conservatively (at most two gamma bits per additional RHS symbol).
/// Whole threshold groups are also tried where they cross a data-width boundary.
/// An exact packed-size comparison rejects regressions against native control.
pub fn cost_plan(source: &Compressed, max_uses: u64) -> Result<CostPlan, String> {
    let grammar = &source.grammar;
    let mut direct = vec![0u64; grammar.rules.len()];
    for block in &source.blocks {
        for symbol in block.iter() {
            if symbol as usize >= BASE_SYMBOLS {
                *direct
                    .get_mut(symbol as usize - BASE_SYMBOLS)
                    .ok_or("undefined pruning symbol")? += 1;
            }
        }
    }
    let mut reachable: Vec<_> = direct.iter().map(|&n| n > 0).collect();
    for i in (0..grammar.rules.len()).rev() {
        if reachable[i] {
            let rule = &grammar.rules[i];
            for child in [rule.left, rule.right] {
                if child as usize >= BASE_SYMBOLS {
                    reachable[child as usize - BASE_SYMBOLS] = true;
                }
            }
        }
    }
    let mut retired: Vec<_> = reachable.iter().map(|&r| !r).collect();
    let baseline = profile(source, &direct, &reachable, &retired)?;
    let mut best = Profile {
        bytes: baseline.bytes,
        tokens: baseline.tokens,
    };
    let mut best_retired = retired.clone();
    let mut choice = "control";
    let data_width = symbol_width(BASE_SYMBOLS + reachable.iter().filter(|&&r| r).count());
    let mut data_uses = direct.clone();
    let mut parent_cost = vec![0u128; grammar.rules.len()];
    for i in (0..grammar.rules.len()).rev() {
        if !reachable[i] {
            continue;
        }
        let definition = 2 * symbol_width(BASE_SYMBOLS + i) as u128 + 1;
        let expansion = data_uses[i] as u128 * data_width as u128 + parent_cost[i];
        retired[i] = direct[i] <= max_uses && expansion < definition;
        let rule = &grammar.rules[i];
        for child in [rule.left, rule.right] {
            if child as usize >= BASE_SYMBOLS {
                let child = child as usize - BASE_SYMBOLS;
                if retired[i] {
                    data_uses[child] = data_uses[child]
                        .checked_add(data_uses[i])
                        .ok_or("pruning use count overflow")?;
                    parent_cost[child] += parent_cost[i];
                } else {
                    parent_cost[child] += symbol_width(BASE_SYMBOLS + i) as u128 + 2;
                }
            }
        }
    }
    let candidate = profile(source, &direct, &reachable, &retired)?;
    if candidate.bytes < best.bytes {
        best = candidate;
        best_retired.clone_from(&retired);
        choice = "local-cost";
    }
    // Threshold groups capture a nonlocal effect: deleting several individually
    // unprofitable rules may save one bit on every data token.
    let mut frequencies = std::collections::BTreeMap::<u64, usize>::new();
    for (i, &uses) in direct.iter().enumerate() {
        if reachable[i] && uses <= max_uses {
            *frequencies.entry(uses).or_default() += 1;
        }
    }
    frequencies.entry(max_uses).or_default();
    let mut remaining = reachable.iter().filter(|&&r| r).count();
    let mut width = symbol_width(BASE_SYMBOLS + remaining);
    for (cutoff, count) in frequencies {
        remaining -= count;
        let next_width = symbol_width(BASE_SYMBOLS + remaining);
        if next_width < width || cutoff == 0 || cutoff == max_uses {
            for (i, value) in retired.iter_mut().enumerate() {
                *value = !reachable[i] || direct[i] <= cutoff;
            }
            let candidate = profile(source, &direct, &reachable, &retired)?;
            if candidate.bytes < best.bytes {
                best = candidate;
                best_retired.clone_from(&retired);
                choice = "threshold-group";
            }
        }
        width = next_width;
    }
    Ok(CostPlan {
        plan: Plan {
            selected_rules: best_retired
                .iter()
                .zip(&reachable)
                .filter(|&(r, live)| *r && *live)
                .count(),
            retired: best_retired,
            before_tokens: source.token_count(),
            added_tokens: best
                .tokens
                .checked_sub(source.token_count())
                .ok_or("pruning shortened data")?,
        },
        baseline_bytes: baseline.bytes,
        predicted_bytes: best.bytes,
        choice,
    })
}

fn is_retired(symbol: Symbol, retired: &[bool]) -> bool {
    (symbol as usize >= BASE_SYMBOLS)
        && retired
            .get(symbol as usize - BASE_SYMBOLS)
            .copied()
            .unwrap_or(false)
}

pub fn plan(grammar: &Grammar, blocks: &[PackedSymbols], max_uses: u64) -> Result<Plan, String> {
    let mut uses = vec![0u64; grammar.rules.len()];
    let mut before_tokens = 0usize;
    for block in blocks {
        before_tokens = before_tokens
            .checked_add(block.len())
            .ok_or("token count overflow")?;
        for symbol in block.iter() {
            if symbol as usize >= BASE_SYMBOLS {
                *uses
                    .get_mut(symbol as usize - BASE_SYMBOLS)
                    .ok_or("undefined pruning symbol")? += 1;
            }
        }
    }
    let retired: Vec<_> = uses.iter().map(|&n| n <= max_uses).collect();
    // Replacement length in *surviving symbols*, not full terminal length.
    // Topological grammar order avoids repeated traversal for this prediction.
    let mut lengths = vec![1usize; BASE_SYMBOLS];
    let mut added_tokens = 0usize;
    for (i, rule) in grammar.rules.iter().enumerate() {
        let length = if retired[i] {
            lengths[rule.left as usize]
                .checked_add(lengths[rule.right as usize])
                .ok_or("pruning expansion overflow")?
        } else {
            1
        };
        lengths.push(length);
        added_tokens = added_tokens
            .checked_add(
                usize::try_from(uses[i])
                    .map_err(|_| "pruning frequency overflow")?
                    .checked_mul(length - 1)
                    .ok_or("pruning token count overflow")?,
            )
            .ok_or("pruning token count overflow")?;
    }
    Ok(Plan {
        selected_rules: retired.iter().filter(|&&v| v).count(),
        retired,
        before_tokens,
        added_tokens,
    })
}

fn visit_survivors(
    symbol: Symbol,
    grammar: &Grammar,
    retired: &[bool],
    mut emit: impl FnMut(Symbol),
) {
    if !is_retired(symbol, retired) {
        emit(symbol);
        return;
    }
    let mut stack = vec![symbol];
    while let Some(symbol) = stack.pop() {
        if is_retired(symbol, retired) {
            let rule = &grammar.rules[symbol as usize - BASE_SYMBOLS];
            stack.push(rule.right);
            stack.push(rule.left);
        } else {
            emit(symbol);
        }
    }
}

pub(crate) fn rewrite_active(
    buffer: &PackedSymbols,
    grammar: &Grammar,
    retired: &[bool],
    track_overlaps: bool,
) -> (PackedSymbols, PairCounter) {
    let mut output = PackedSymbols::new(buffer.width());
    let mut counter = PairCounter::new(track_overlaps);
    for symbol in buffer.iter() {
        visit_survivors(symbol, grammar, retired, |symbol| {
            output.push(symbol);
            counter.push(symbol, grammar);
        });
    }
    (output, counter)
}

/// Compact only the exported vocabulary. Historical binary construction nodes
/// can remain inside the experimental trainer without being serialized.
pub fn export(source: &Compressed, retired: &[bool]) -> Result<NativeCompressed, String> {
    let grammar = &source.grammar;
    let mut reachable = vec![false; grammar.rules.len()];
    let mut stack = Vec::new();
    for block in &source.blocks {
        for symbol in block.iter() {
            stack.push(symbol);
            while let Some(symbol) = stack.pop() {
                if symbol as usize >= BASE_SYMBOLS {
                    let index = symbol as usize - BASE_SYMBOLS;
                    let seen = reachable.get_mut(index).ok_or("undefined export symbol")?;
                    if !*seen {
                        *seen = true;
                        let rule = &grammar.rules[index];
                        stack.push(rule.right);
                        stack.push(rule.left);
                    }
                }
            }
        }
    }
    let mut mapping: Vec<_> = (0..BASE_SYMBOLS as Symbol).collect();
    mapping.resize(grammar.alphabet_size(), u32::MAX);
    let mut count = 0usize;
    for (i, &used) in reachable.iter().enumerate() {
        let symbol = (BASE_SYMBOLS + i) as Symbol;
        if used && !is_retired(symbol, retired) {
            mapping[symbol as usize] =
                u32::try_from(BASE_SYMBOLS + count).map_err(|_| "export IDs exhausted")?;
            count += 1;
        }
    }
    let mut rules = Vec::with_capacity(count);
    for (i, rule) in grammar.rules.iter().enumerate() {
        let symbol = (BASE_SYMBOLS + i) as Symbol;
        if mapping[symbol as usize] != u32::MAX {
            let mut rhs = Vec::new();
            for child in [rule.left, rule.right] {
                visit_survivors(child, grammar, retired, |symbol| {
                    rhs.push(mapping[symbol as usize])
                });
            }
            rules.push(rhs);
        }
    }
    let width = symbol_width(BASE_SYMBOLS + rules.len());
    let mut blocks = Vec::with_capacity(source.blocks.len());
    for block in &source.blocks {
        let mut output = PackedSymbols::new(width);
        for symbol in block.iter() {
            visit_survivors(symbol, grammar, retired, |symbol| {
                output.push(mapping[symbol as usize])
            });
        }
        blocks.push(output);
    }
    Ok(NativeCompressed {
        rules,
        blocks,
        input_bytes: source.input_bytes,
        field_count: source.field_count,
    })
}
