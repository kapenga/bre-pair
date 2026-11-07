//! Exact byte/EOF radix matching and disk-backed large-field reconstruction.
//! IO chunk boundaries never become parsing boundaries.
use super::{Edge, NONE};
use crate::{
    atomic_output::Output,
    grammar::{Grammar, BASE_SYMBOLS, EOF},
    huffman::Codebook,
};
use std::{
    fs::{File, OpenOptions},
    io::{Read, Seek, SeekFrom, Write},
    path::Path,
};

pub(crate) const CHUNK: usize = 1 << 20;
const NODE_PAGE: usize = 1 << 16;
const LABEL_PAGE: usize = 1 << 20;
#[derive(Clone, Copy)]
struct Node {
    first: u32,
    next: u32,
    token: u32,
    start: usize,
    length: usize,
}
#[derive(Default)]
struct Nodes {
    pages: Vec<Vec<Node>>,
    len: usize,
}
impl Nodes {
    fn get(&self, id: u32) -> &Node {
        &self.pages[id as usize / NODE_PAGE][id as usize % NODE_PAGE]
    }
    fn get_mut(&mut self, id: u32) -> &mut Node {
        &mut self.pages[id as usize / NODE_PAGE][id as usize % NODE_PAGE]
    }
    fn push(&mut self, node: Node) -> Result<u32, String> {
        let id = u32::try_from(self.len).map_err(|_| "radix node IDs exhausted")?;
        if id == NONE {
            return Err("radix node IDs exhausted".into());
        }
        if self.len % NODE_PAGE == 0 {
            self.pages.push(Vec::with_capacity(NODE_PAGE));
        }
        self.pages.last_mut().unwrap().push(node);
        self.len += 1;
        Ok(id)
    }
    fn allocated(&self) -> usize {
        self.pages
            .iter()
            .map(|p| p.capacity() * std::mem::size_of::<Node>())
            .sum()
    }
}
#[derive(Default)]
struct Labels {
    pages: Vec<Vec<u16>>,
    len: usize,
}
impl Labels {
    fn get(&self, at: usize) -> u16 {
        self.pages[at / LABEL_PAGE][at % LABEL_PAGE]
    }
    fn extend(&mut self, labels: &[u16]) -> Result<usize, String> {
        let start = self.len;
        self.len
            .checked_add(labels.len())
            .ok_or("radix label positions overflow")?;
        for &label in labels {
            if self.len % LABEL_PAGE == 0 {
                self.pages.push(Vec::with_capacity(LABEL_PAGE));
            }
            self.pages.last_mut().unwrap().push(label);
            self.len += 1;
        }
        Ok(start)
    }
    fn allocated(&self) -> usize {
        self.pages.iter().map(|p| p.capacity() * 2).sum()
    }
}
struct Choices {
    file: File,
    _cleanup: Output,
}
impl Choices {
    fn new(directory: &Path) -> Result<Self, String> {
        let cleanup = Output::new(&directory.join("reparse-choices"))?;
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(cleanup.path())
            .map_err(|e| e.to_string())?;
        Ok(Self {
            file,
            _cleanup: cleanup,
        })
    }
}
#[derive(Default)]
pub(crate) struct RawScratch {
    costs: Vec<u64>,
    choices: Vec<u32>,
}
pub(crate) struct RawMatcher {
    nodes: Nodes,
    edges: Vec<Edge>,
    labels: Labels,
    roots: [u32; BASE_SYMBOLS],
    second: Vec<u32>,
    lengths: Vec<usize>,
    bits: Vec<u8>,
    longest: usize,
}
impl RawMatcher {
    pub(crate) fn new(grammar: &Grammar, book: &Codebook) -> Result<Self, String> {
        let n = grammar.alphabet_size();
        let mut bits = vec![0; n];
        for (id, b) in book.lengths() {
            bits[id as usize] = b;
        }
        if bits[..BASE_SYMBOLS].contains(&0) {
            return Err("radix parser requires the coded byte/EOF basis".into());
        }
        let lengths: Vec<_> = (0..n).map(|id| grammar.terminals(id as u32)).collect();
        let longest = lengths
            .iter()
            .zip(&bits)
            .filter(|(_, b)| **b != 0)
            .map(|(&l, _)| l)
            .max()
            .unwrap_or(1);
        let mut result = Self {
            nodes: Nodes::default(),
            edges: Vec::new(),
            labels: Labels::default(),
            roots: [NONE; BASE_SYMBOLS],
            second: Vec::new(),
            lengths,
            bits,
            longest,
        };
        result.nodes.push(Node {
            first: NONE,
            next: NONE,
            token: NONE,
            start: 0,
            length: 0,
        })?;
        let (mut word, mut stack) = (Vec::new(), Vec::new());
        for id in 0..n {
            if result.bits[id] == 0 {
                continue;
            }
            word.clear();
            stack.push(id as u32);
            while let Some(s) = stack.pop() {
                if s < BASE_SYMBOLS as u32 {
                    word.push(s as u16);
                } else {
                    let r = &grammar.rules[s as usize - BASE_SYMBOLS];
                    stack.push(r.right);
                    stack.push(r.left);
                }
            }
            result.insert(&word, id as u32)?;
        }
        // Radix splitting can create a parent after its child. First capture
        // all sibling lists, then reuse Node.next as a child count. Sorted
        // primitive edges avoid scanning cold/rare siblings in the hot loop.
        let mut offsets = Vec::with_capacity(result.nodes.len + 1);
        result.edges = Vec::with_capacity(result.nodes.len - 1);
        offsets.push(0usize);
        for parent in 0..result.nodes.len {
            let begin = result.edges.len();
            let mut child = result.nodes.get(parent as u32).first;
            while child != NONE {
                let node = *result.nodes.get(child);
                result.edges.push(Edge {
                    label: u32::from(result.labels.get(node.start)),
                    child,
                });
                child = node.next;
            }
            result.edges[begin..].sort_unstable_by_key(|e| e.label);
            offsets.push(result.edges.len());
        }
        for parent in 0..result.nodes.len {
            let node = result.nodes.get_mut(parent as u32);
            node.first = offsets[parent] as u32;
            node.next = (offsets[parent + 1] - offsets[parent]) as u32;
        }
        drop(offsets);
        result.second = vec![NONE; BASE_SYMBOLS * BASE_SYMBOLS];
        for base in 0..BASE_SYMBOLS {
            let node = result.nodes.get(result.roots[base]);
            for edge in &result.edges[node.first as usize..node.first as usize + node.next as usize]
            {
                result.second[base * BASE_SYMBOLS + edge.label as usize] = edge.child;
            }
        }
        eprintln!(
            "radix matcher: {} nodes, {} label terminals, longest {}",
            result.nodes.len, result.labels.len, longest
        );
        Ok(result)
    }
    fn insert(&mut self, word: &[u16], id: u32) -> Result<(), String> {
        let (mut parent, mut at) = (0u32, 0usize);
        loop {
            if at == word.len() {
                let old = self.nodes.get(parent).token;
                if old == NONE || (self.bits[id as usize], id) < (self.bits[old as usize], old) {
                    self.nodes.get_mut(parent).token = id;
                }
                return Ok(());
            }
            let (mut previous, mut child) = (NONE, self.nodes.get(parent).first);
            if parent == 0 {
                child = self.roots[word[at] as usize];
            }
            while child != NONE && self.labels.get(self.nodes.get(child).start) != word[at] {
                previous = child;
                child = self.nodes.get(child).next;
            }
            if child == NONE {
                let start = self.labels.extend(&word[at..])?;
                let next = self.nodes.get(parent).first;
                let leaf = self.nodes.push(Node {
                    first: NONE,
                    next,
                    token: id,
                    start,
                    length: word.len() - at,
                })?;
                self.nodes.get_mut(parent).first = leaf;
                if parent == 0 {
                    self.roots[word[at] as usize] = leaf;
                }
                return Ok(());
            }
            let old = *self.nodes.get(child);
            let mut common = 1;
            while common < old.length.min(word.len() - at)
                && self.labels.get(old.start + common) == word[at + common]
            {
                common += 1;
            }
            if common == old.length {
                parent = child;
                at += common;
                continue;
            }
            let prefix = self.nodes.push(Node {
                first: child,
                next: old.next,
                token: NONE,
                start: old.start,
                length: common,
            })?;
            if previous == NONE {
                self.nodes.get_mut(parent).first = prefix;
            } else {
                self.nodes.get_mut(previous).next = prefix;
            }
            if parent == 0 {
                self.roots[word[at] as usize] = prefix;
            }
            let tail = self.nodes.get_mut(child);
            tail.start += common;
            tail.length -= common;
            tail.next = NONE;
            parent = prefix;
            at += common;
        }
    }
    pub(crate) fn storage_bytes(&self) -> usize {
        self.nodes.allocated()
            + self.edges.capacity() * std::mem::size_of::<Edge>()
            + self.labels.allocated()
            + self.second.capacity() * 4
            + self.lengths.capacity() * std::mem::size_of::<usize>()
            + self.bits.capacity()
    }
    fn choose(
        &self,
        start: usize,
        n: usize,
        raw: &[u8],
        offset: usize,
        costs: &[u64],
        mask: usize,
    ) -> Result<(u32, u64), String> {
        let terminal = |at| {
            if at == n - 1 {
                EOF as u16
            } else {
                u16::from(raw[at - offset])
            }
        };
        let base = u32::from(terminal(start));
        let (mut choice, mut best_len) = (base, 1usize);
        let mut best = costs[(start + 1) & mask]
            .checked_add(u64::from(self.bits[base as usize]))
            .ok_or("encoded field bit count overflow")?;
        let (mut parent, mut at) = (self.roots[base as usize], start + 1);
        while at < n {
            let p = self.nodes.get(parent);
            if p.next == 0 {
                break;
            }
            let label = terminal(at);
            let child = if p.token < BASE_SYMBOLS as u32 {
                self.second[p.token as usize * BASE_SYMBOLS + label as usize]
            } else {
                let edges = &self.edges[p.first as usize..p.first as usize + p.next as usize];
                match edges.binary_search_by_key(&u32::from(label), |e| e.label) {
                    Ok(at) => edges[at].child,
                    Err(_) => NONE,
                }
            };
            if child == NONE {
                break;
            }
            let edge = self.nodes.get(child);
            if edge.length > n - at
                || (1..edge.length).any(|j| self.labels.get(edge.start + j) != terminal(at + j))
            {
                break;
            }
            at += edge.length;
            let id = edge.token;
            if id != NONE {
                let cost = costs[at & mask]
                    .checked_add(u64::from(self.bits[id as usize]))
                    .ok_or("encoded field bit count overflow")?;
                let length = at - start;
                if cost < best
                    || (cost == best && (length > best_len || (length == best_len && id < choice)))
                {
                    best = cost;
                    best_len = length;
                    choice = id;
                }
            }
            parent = child;
        }
        Ok((choice, best))
    }
    pub(crate) fn segment_slice(
        &self,
        bytes: &[u8],
        scratch: &mut RawScratch,
        mut emit: impl FnMut(u32) -> Result<(), String>,
    ) -> Result<(), String> {
        let n = bytes.len().checked_add(1).ok_or("field length overflow")?;
        let window = self
            .longest
            .min(n)
            .checked_add(1)
            .and_then(usize::checked_next_power_of_two)
            .ok_or("frontier overflow")?;
        scratch.costs.resize(window, 0);
        scratch.choices.resize(n, 0);
        let mask = window - 1;
        scratch.costs[n & mask] = 0;
        for start in (0..n).rev() {
            let (id, cost) = self.choose(start, n, bytes, 0, &scratch.costs, mask)?;
            scratch.choices[start] = id;
            scratch.costs[start & mask] = cost;
        }
        let mut at = 0;
        while at < n {
            let id = scratch.choices[at];
            emit(id)?;
            at += self.lengths[id as usize];
        }
        Ok(())
    }
    /// Parse one raw byte range plus its virtual EOF. Exactly the costs and
    /// tie order of segment_with; only storage and input representation differ.
    pub(crate) fn segment_bytes_spooled(
        &self,
        input: &mut (impl Read + Seek),
        begin: u64,
        bytes: usize,
        directory: &Path,
        mut emit: impl FnMut(u32) -> Result<(), String>,
    ) -> Result<(), String> {
        self.segment_bytes_chunks(input, begin, bytes, directory, CHUNK, &mut emit)
    }

    fn segment_bytes_chunks(
        &self,
        input: &mut (impl Read + Seek),
        begin: u64,
        bytes: usize,
        directory: &Path,
        chunk: usize,
        emit: &mut impl FnMut(u32) -> Result<(), String>,
    ) -> Result<(), String> {
        if self.roots.len() != BASE_SYMBOLS
            || self.bits[..BASE_SYMBOLS].contains(&0)
            || self.roots.contains(&NONE)
            || chunk == 0
        {
            return Err(
                "spooled reparse requires the coded byte/EOF basis and a positive chunk".into(),
            );
        }
        let n = bytes.checked_add(1).ok_or("field length overflow")?;
        let total = u64::try_from(n).map_err(|_| "field exceeds u64 positions")?;
        total.checked_mul(4).ok_or("choice file offset overflow")?;
        begin
            .checked_add(bytes as u64)
            .ok_or("raw offset overflow")?;
        let window = self
            .longest
            .min(n)
            .checked_add(1)
            .ok_or("reparse frontier overflow")?
            .checked_next_power_of_two()
            .ok_or("reparse frontier overflow")?;
        let mask = window - 1;
        let mut costs = vec![0u64; window];
        let mut raw = Vec::new();
        let mut coded = Vec::with_capacity(chunk.min(n).checked_mul(4).ok_or("chunk overflow")?);
        let mut spool = Choices::new(directory)?;
        let mut high = n;
        let mut next_progress = 16 * 1024 * 1024usize;
        while high != 0 {
            let low = high.saturating_sub(chunk);
            let raw_end = bytes.min(high.saturating_add(self.longest.saturating_sub(1)));
            raw.resize(raw_end - low, 0);
            input
                .seek(SeekFrom::Start(begin + low as u64))
                .map_err(|e| e.to_string())?;
            input.read_exact(&mut raw).map_err(|e| e.to_string())?;
            coded.clear();
            for start in (low..high).rev() {
                let (choice, best) = self.choose(start, n, &raw, low, &costs, mask)?;
                costs[start & mask] = best;
                // Descending positions are written sequentially. This avoids
                // growing/zero-filling the entire scratch file by an initial
                // far seek. Forward reconstruction reads buffered reverse pages.
                coded.extend_from_slice(&choice.to_le_bytes());
            }
            spool.file.write_all(&coded).map_err(|e| e.to_string())?;
            high = low;
            if n - high >= next_progress {
                crate::training_progress::phase(
                    "reparse/backward",
                    0,
                    (n - high) as u64,
                    total,
                    "terminals",
                );
                next_progress = (n - high).saturating_add(16 * 1024 * 1024);
            }
        }
        // Release lookahead/cost storage before constructing the packed roots.
        drop(raw);
        drop(costs);
        coded.clear();
        let mut cache = None;
        let mut at = 0usize;
        next_progress = 16 * 1024 * 1024;
        while at < n {
            let index = n - 1 - at;
            let page = index / chunk * chunk;
            if cache != Some(page) {
                coded.resize((n - page).min(chunk) * 4, 0);
                spool
                    .file
                    .seek(SeekFrom::Start(page as u64 * 4))
                    .map_err(|e| e.to_string())?;
                spool
                    .file
                    .read_exact(&mut coded)
                    .map_err(|e| e.to_string())?;
                cache = Some(page);
            }
            let offset = (index - page) * 4;
            let id = u32::from_le_bytes(coded[offset..offset + 4].try_into().unwrap());
            let length = *self
                .lengths
                .get(id as usize)
                .ok_or("invalid reparse choice")?;
            if length == 0 || length > n - at {
                return Err("invalid reparse step".into());
            }
            emit(id)?;
            at += length;
            if at >= next_progress {
                crate::training_progress::phase(
                    "reparse/reconstruct",
                    0,
                    at as u64,
                    total,
                    "terminals",
                );
                next_progress = at.saturating_add(16 * 1024 * 1024);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::Matcher;
    use crate::{grammar::Grammar, huffman_dictionary::Dictionary};
    use std::{
        io::Cursor,
        path::PathBuf,
        sync::atomic::{AtomicU64, Ordering},
    };
    static NEXT: AtomicU64 = AtomicU64::new(0);
    struct Scratch(PathBuf);
    impl Scratch {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "spooled-reparse-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir(&path).unwrap();
            Self(path)
        }
    }
    impl Drop for Scratch {
        fn drop(&mut self) {
            std::fs::remove_dir(&self.0).unwrap();
        }
    }
    #[test]
    fn paged_nodes_and_wide_label_positions_preserve_boundaries() {
        let mut nodes = Nodes::default();
        for i in 0..NODE_PAGE + 3 {
            assert_eq!(
                nodes
                    .push(Node {
                        first: NONE,
                        next: NONE,
                        token: i as u32,
                        start: i,
                        length: 1
                    })
                    .unwrap(),
                i as u32
            );
        }
        for i in [0, NODE_PAGE - 1, NODE_PAGE, NODE_PAGE + 2] {
            assert_eq!(nodes.get(i as u32).start, i);
        }
        nodes.get_mut(NODE_PAGE as u32).length = 7;
        assert_eq!(nodes.get(NODE_PAGE as u32).length, 7);
        let data: Vec<_> = (0..LABEL_PAGE + 3).map(|i| (i % 257) as u16).collect();
        let mut labels = Labels::default();
        assert_eq!(labels.extend(&data).unwrap(), 0);
        for i in [0, LABEL_PAGE - 1, LABEL_PAGE, LABEL_PAGE + 2] {
            assert_eq!(labels.get(i), data[i]);
        }
    }
    #[test]
    fn radix_splits_duplicate_paths_and_uncoded_helpers_match_reference() {
        let mut rng = 9413u64;
        let mut next = || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            (rng >> 32) as usize
        };
        for _ in 0..3 {
            let mut grammar = Grammar::default();
            let mut pool = vec![0, 255, b'a' as u32, b'b' as u32, b'c' as u32, EOF];
            for _ in 0..3000 {
                let (a, b) = (pool[next() % pool.len()], pool[next() % pool.len()]);
                if !grammar.ends_field(a) && grammar.terminals(a) + grammar.terminals(b) <= 24 {
                    pool.push(grammar.add((a, b)).unwrap());
                    if grammar.rules.len() >= 200 {
                        break;
                    }
                }
            }
            assert!(grammar.rules.len() >= 100);
            let counts: Vec<_> = (0..grammar.alphabet_size())
                .map(|id| {
                    if id < BASE_SYMBOLS {
                        1
                    } else {
                        (next() % 8) as u64
                    }
                })
                .collect();
            let book = Codebook::from_counts(&counts).unwrap();
            let dictionary = Dictionary::from_grammar(&grammar).unwrap();
            let reference = Matcher::from_graph(&dictionary, &book).unwrap();
            let radix = RawMatcher::new(&grammar, &book).unwrap();
            let mut scratch = RawScratch::default();
            for n in 0..100 {
                let bytes: Vec<_> = (0..n * 2).map(|_| b"abc\0\xff"[next() % 5]).collect();
                let mut terminals: Vec<_> = bytes.iter().map(|&b| u32::from(b)).collect();
                terminals.push(EOF);
                let expected = reference.segment(&terminals).unwrap();
                let mut actual = Vec::new();
                radix
                    .segment_slice(&bytes, &mut scratch, |id| {
                        actual.push(id);
                        Ok(())
                    })
                    .unwrap();
                assert_eq!(actual, expected);
            }
        }
    }
    fn model() -> (Grammar, Dictionary, Codebook) {
        let mut g = Grammar::default();
        let ab = g.add((b'a' as u32, b'b' as u32)).unwrap();
        let bc = g.add((b'b' as u32, b'c' as u32)).unwrap();
        g.add((ab, b'c' as u32)).unwrap();
        g.add((b'a' as u32, bc)).unwrap(); // identical bytes, different recipes
        let ce = g.add((b'c' as u32, EOF)).unwrap();
        g.add((ab, ce)).unwrap();
        g.add((0, 255)).unwrap();
        let mut power = b'a' as u32;
        for _ in 0..10 {
            power = g.add((power, power)).unwrap();
        }
        let mut counts: Vec<_> = (0..g.alphabet_size()).map(|i| 1 + i as u64 % 7).collect();
        counts[ab as usize] = 0; // uncoded helper remains expandable
        let book = Codebook::from_counts(&counts).unwrap();
        (g.clone(), Dictionary::from_grammar(&g).unwrap(), book)
    }
    #[test]
    fn compact_trie_and_spooled_boundaries_preserve_all_exact_ids_and_ties() {
        let dir = Scratch::new();
        let (grammar, dictionary, book) = model();
        let reference = Matcher::from_graph(&dictionary, &book).unwrap();
        let compact = RawMatcher::new(&grammar, &book).unwrap();
        let mut samples = vec![
            vec![],
            b"abcabc".to_vec(),
            b"aaaaaaa".to_vec(),
            vec![0, 255, b'\n', 0, 255],
            vec![b'a'; 4097],
        ];
        let mut rng = 713u64;
        for size in 0..40 {
            samples.push(
                (0..size * 3)
                    .map(|_| {
                        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                        b"abc\0\xff\n"[(rng >> 32) as usize % 6]
                    })
                    .collect(),
            );
        }
        for bytes in samples {
            let mut terminals: Vec<_> = bytes.iter().map(|&b| u32::from(b)).collect();
            terminals.push(EOF);
            let expected = reference.segment(&terminals).unwrap();
            let mut small = Vec::new();
            compact
                .segment_slice(&bytes, &mut RawScratch::default(), |id| {
                    small.push(id);
                    Ok(())
                })
                .unwrap();
            assert_eq!(small, expected);
            for chunk in [1, 2, 3, 7, 16, 63] {
                let mut file = b"unrelated prefix".to_vec();
                let begin = file.len() as u64;
                file.extend_from_slice(&bytes);
                file.extend_from_slice(b"abc suffix must not be matched");
                let mut actual = Vec::new();
                compact
                    .segment_bytes_chunks(
                        &mut Cursor::new(file),
                        begin,
                        bytes.len(),
                        &dir.0,
                        chunk,
                        &mut |id| {
                            actual.push(id);
                            Ok(())
                        },
                    )
                    .unwrap();
                assert_eq!(actual, expected, "chunk {chunk}; bytes {}", bytes.len());
                assert_eq!(std::fs::read_dir(&dir.0).unwrap().count(), 0);
            }
        }
    }
    #[test]
    fn spool_is_removed_on_input_and_output_errors() {
        let dir = Scratch::new();
        let (grammar, _, book) = model();
        let compact = RawMatcher::new(&grammar, &book).unwrap();
        assert!(compact
            .segment_bytes_chunks(&mut Cursor::new(b"abc"), 0, 4, &dir.0, 2, &mut |_| Ok(()))
            .is_err());
        #[cfg(target_pointer_width = "64")]
        {
            // A field beyond u32 positions reaches a wide raw seek without
            // allocating field-sized arrays. This deliberately truncated source
            // errors after the EOF choice; its tiny scratch still gets removed.
            struct TrackSeek {
                cursor: Cursor<Vec<u8>>,
                maximum: u64,
            }
            impl Read for TrackSeek {
                fn read(&mut self, bytes: &mut [u8]) -> std::io::Result<usize> {
                    self.cursor.read(bytes)
                }
            }
            impl Seek for TrackSeek {
                fn seek(&mut self, from: SeekFrom) -> std::io::Result<u64> {
                    let at = self.cursor.seek(from)?;
                    self.maximum = self.maximum.max(at);
                    Ok(at)
                }
            }
            let mut truncated = TrackSeek {
                cursor: Cursor::new(Vec::new()),
                maximum: 0,
            };
            let size = u32::MAX as usize + 17;
            assert!(compact
                .segment_bytes_chunks(&mut truncated, 0, size, &dir.0, 1, &mut |_| Ok(()))
                .is_err());
            assert!(truncated.maximum > u64::from(u32::MAX));
            assert_eq!(std::fs::read_dir(&dir.0).unwrap().count(), 0);
        }
        assert_eq!(std::fs::read_dir(&dir.0).unwrap().count(), 0);
        assert!(compact
            .segment_bytes_chunks(&mut Cursor::new(b"abcabc"), 0, 6, &dir.0, 2, &mut |_| Err(
                "intentional output error".into()
            ))
            .is_err());
        assert_eq!(std::fs::read_dir(&dir.0).unwrap().count(), 0);
        assert!(compact
            .segment_bytes_chunks(&mut Cursor::new(b"abc"), 0, 3, &dir.0, 0, &mut |_| Ok(()))
            .is_err());
        assert!(compact
            .segment_bytes_chunks(
                &mut Cursor::new(b"abc"),
                u64::MAX,
                3,
                &dir.0,
                1,
                &mut |_| Ok(())
            )
            .is_err());
    }
}
