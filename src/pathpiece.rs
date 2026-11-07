//! Research adaptation of Schmidt et al. (EMNLP 2024), section 3.2.
//! Score deletion of each selected edge occurrence, then sum by token type.
//! This is not exact global deletion cost and is not the authors' trainer.
use crate::{
    grammar::BASE_SYMBOLS,
    tokenizer::{Model, Tokenizer},
};

#[derive(Clone, Copy, Debug)]
struct Edge {
    start: usize,
    end: usize,
    id: usize,
}

fn distances(
    edges: &[Edge],
    n: usize,
    exclude: Option<(usize, usize, usize)>,
) -> (Vec<u32>, Vec<usize>, Vec<usize>) {
    let mut distance = vec![u32::MAX; n + 1];
    distance[0] = 0;
    let mut ids = vec![0; n + 1];
    let mut widths = vec![0; n + 1];
    for edge in edges {
        if exclude == Some((edge.start, edge.end, edge.id)) || distance[edge.start] == u32::MAX {
            continue;
        }
        let value = distance[edge.start] + 1;
        let width = edge.end - edge.start;
        if value < distance[edge.end] || (value == distance[edge.end] && width > widths[edge.end]) {
            distance[edge.end] = value;
            ids[edge.end] = edge.id;
            widths[edge.end] = width;
        }
    }
    (distance, ids, widths)
}

fn occurrence_losses(tokenizer: &Tokenizer, bytes: &[u8]) -> Result<Vec<(Edge, u32)>, String> {
    let n = bytes.len().checked_add(1).ok_or("field too large")?;
    if n >= u32::MAX as usize {
        return Err("PathPiece research field exceeds 32-bit local DP extent".into());
    }
    let mut edges = Vec::new();
    tokenizer.matching_edges(bytes, |start, end, id| {
        edges.push(Edge {
            start,
            end,
            id: id as usize,
        });
        Ok(())
    })?;
    let (forward, ids, widths) = distances(&edges, n, None);
    let total = forward[n];
    let mut backward = vec![u32::MAX; n + 1];
    backward[n] = 0;
    for edge in edges.iter().rev() {
        if backward[edge.end] != u32::MAX {
            backward[edge.start] = backward[edge.start].min(1 + backward[edge.end]);
        }
    }
    let mut chosen = Vec::new();
    let mut end = n;
    while end > 0 {
        let width = widths[end];
        if width == 0 {
            return Err("unreachable byte basis".into());
        }
        chosen.push(Edge {
            start: end - width,
            end,
            id: ids[end],
        });
        end -= width;
    }
    chosen.reverse();
    let mut loss = vec![u32::MAX; chosen.len()];
    for (i, edge) in chosen.iter().enumerate() {
        if edge.id < BASE_SYMBOLS {
            continue;
        }
        for cut in edge.start + 1..edge.end {
            loss[i] = loss[i].min(forward[cut] + backward[cut] - total);
        }
    }
    // A detour either has a boundary strictly inside the removed edge, or
    // crosses its entire span in one alternative edge. This enumerates both.
    for edge in edges {
        let mut at = chosen.partition_point(|selected| selected.start < edge.start);
        while at < chosen.len() && chosen[at].end <= edge.end {
            let selected = chosen[at];
            if selected.id >= BASE_SYMBOLS
                && (edge.start, edge.end, edge.id) != (selected.start, selected.end, selected.id)
            {
                loss[at] = loss[at].min(forward[edge.start] + 1 + backward[edge.end] - total);
            }
            at += 1;
        }
    }
    Ok(chosen.into_iter().zip(loss).collect())
}

pub fn score_field(
    tokenizer: &Tokenizer,
    bytes: &[u8],
    losses: &mut [u64],
    uses: &mut [u64],
) -> Result<u64, String> {
    let occurrences = occurrence_losses(tokenizer, bytes)?;
    for (edge, loss) in &occurrences {
        uses[edge.id] = uses[edge.id].checked_add(1).ok_or("usage overflow")?;
        if edge.id >= BASE_SYMBOLS {
            if *loss == u32::MAX {
                return Err("learned edge has no byte-basis replacement".into());
            }
            losses[edge.id] = losses[edge.id]
                .checked_add(u64::from(*loss))
                .ok_or("loss overflow")?;
        }
    }
    Ok(occurrences.len() as u64)
}

pub fn remove_batch(
    model: &Model,
    losses: &[u64],
    uses: &[u64],
    target: usize,
) -> Result<Model, String> {
    if target < BASE_SYMBOLS
        || target >= model.tokens.len()
        || losses.len() != model.tokens.len()
        || uses.len() != losses.len()
    {
        return Err("invalid PathPiece reduction budget".into());
    }
    let mut ranked: Vec<_> = (BASE_SYMBOLS..model.tokens.len()).collect();
    ranked.sort_unstable_by_key(|&id| (losses[id], uses[id], model.tokens[id].bytes.len(), id));
    let mut removed = vec![false; model.tokens.len()];
    for id in ranked.into_iter().take(model.tokens.len() - target) {
        removed[id] = true;
    }
    Ok(Model {
        tokens: model
            .tokens
            .iter()
            .enumerate()
            .filter(|(id, _)| !removed[*id])
            .map(|(_, t)| t.clone())
            .collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::Token;
    fn model() -> Model {
        let mut tokens: Vec<_> = (0..256)
            .map(|b| Token {
                bytes: vec![b as u8],
                ends_field: false,
            })
            .collect();
        tokens.push(Token {
            bytes: Vec::new(),
            ends_field: true,
        });
        for bytes in [
            b"ab".as_slice(),
            b"bc",
            b"abc",
            b"bcd",
            b"cde",
            b"abcde",
            b"bcdef",
            b"aa",
        ] {
            tokens.push(Token {
                bytes: bytes.to_vec(),
                ends_field: false,
            });
        }
        Model { tokens }
    }
    #[test]
    fn occurrence_detours_match_brute_force_edge_deletion_including_middle_and_crossing() {
        let tokenizer = Tokenizer::new(model()).unwrap();
        for bytes in [
            b"abcdefg".as_slice(),
            b"aaaaa",
            b"abcbcde",
            b"",
            b"\0abc\xff",
        ] {
            let n = bytes.len() + 1;
            let mut edges = Vec::new();
            tokenizer
                .matching_edges(bytes, |start, end, id| {
                    edges.push(Edge {
                        start,
                        end,
                        id: id as usize,
                    });
                    Ok(())
                })
                .unwrap();
            let total = distances(&edges, n, None).0[n];
            for (edge, loss) in occurrence_losses(&tokenizer, bytes).unwrap() {
                if edge.id >= BASE_SYMBOLS {
                    assert_eq!(
                        loss,
                        distances(&edges, n, Some((edge.start, edge.end, edge.id))).0[n] - total
                    );
                }
            }
        }
    }
    #[test]
    fn pruning_keeps_the_entire_byte_and_eof_basis() {
        let model = model();
        let n = model.tokens.len();
        let reduced = remove_batch(&model, &vec![0; n], &vec![0; n], BASE_SYMBOLS).unwrap();
        assert_eq!(reduced.tokens.len(), BASE_SYMBOLS);
        let tokenizer = Tokenizer::new(reduced).unwrap();
        let ids = tokenizer.encode(b"\0abcdef\xff");
        assert_eq!(tokenizer.model.decode_field(&ids).unwrap(), b"\0abcdef\xff");
    }
}
