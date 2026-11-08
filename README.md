# BRe-Pair

BRe-Pair is a lossless grammar compressor and byte-level tokenizer trainer
built around **batched Re-Pair**. It replaces compatible symbol pairs together,
keeps the working stream packed at the current symbol width, and supports
reusable vocabularies and static canonical Huffman coding.

The goal is practical offline construction with controlled memory use for
collections that can be encoded or read repeatedly. The public command-line
executable is **`brepair`**.

## Features

- Batched pair replacement with frequency or frequency-weighted pointwise
  mutual information (PMI) scoring.
- Packed symbols, one central pair registry, bounded worker update logs,
  32-bit symbol IDs, and 64-bit frequencies and stream positions.
- Fixed-budget byte vocabularies with exact minimum-token segmentation.
- Shared static Huffman coding for dictionary references and payload symbols,
  with helper inlining and complete-size checkpoint selection.
- Byte reconstruction or preserved AI-token-ID output from token-compatible
  compressed streams, using a single-thread decoder or four-stage pipeline.
- Byte-exact input handling, including invalid UTF-8; no normalization.

This is an early research implementation. Pin a commit when reproducing a
result. Additional experimental executables live in `src/bin/`; ordinary use
requires only `brepair`.

## Build

Use **Rust 1.87 or newer** and a **64-bit target**. The core executable has
been tested on Windows x86-64 and Linux x86-64. There are no external Cargo
crate dependencies, and the executable requires no Python runtime.

```sh
cargo build --locked --profile dist --bin brepair
```

Run `./target/dist/brepair --help` on Linux, or
`.\target\dist\brepair.exe --help` in Windows PowerShell. The examples below
use `brepair`; substitute the appropriate executable path or add its directory
to `PATH`.

## Compress and restore a file

```sh
brepair compress input.txt archive.brh --min-count 8 --threads 4 --memory 2GB --report compression.json
brepair huffman-decode archive.brh restored-input.txt --threads 4 --memory 2GB
```

The example uses minimum count eight. The command defaults to minimum count
two, weighted PMI and two compatible waves. `--score frequency` selects pure
frequency. Dictionary cleanup always runs, and the smallest complete archive
among the produced checkpoints is retained.

By default, each raw line is a field: its LF byte is retained and an internal
end-of-field symbol prevents merges across fields. Use `--fields single` for
one field containing the entire file, including binary data.

Optional quality settings include a descending floor schedule
`--count-floors 8,4,2`, more cleanup rounds with `--cleanup-rounds 16`, and exact
reparsing with `--reparse-refits 2`. A schedule replaces `--min-count`.
Reparsing is off by default; large fields use temporary disk choices of four
bytes per input position and have an explicit `--max-reparse-field-bytes`
bound. Consult `brepair --help` before enabling it on large single fields.

Existing shared-Huffman archives can also use run-coded rule-length metadata
without retraining:

```sh
brepair huffman-repack archive.brh compact.brh --rule-lengths runs --memory 2GB
```

The repacker retains the earlier representation when it is smaller. All
supported representations use the same reconstruction engine.

## Decode independent fields through the library

`shared_huffman::StaticModel` can prepare one immutable decoder for many
EOF-terminated fields. `decode_field_into` writes directly into a caller-owned
slice and returns `(bytes_written, bits_consumed, coded_symbols)`. The slice
length bounds output; caller padding is unnecessary. Its unused capacity is
scratch and may be overwritten. Consume only `output[..bytes_written]` and
discard partial output if decoding fails. Share the model/prepared decoder
between workers, with separate output buffers.

Block decoding handles mixed cached/uncached expansions and checks exact byte,
field and terminal-ID totals. The richer shared lookup table is initialized
on first block use; it adds 16 bytes per coded leaf when needed. The expansion
arena remains bounded, and uncached rules use iterative expansion. All supported
archive layouts retain the same canonical Huffman bit decoder.

The `bytecode-experiments` Cargo feature exposes research-only byte-aligned
and fixed-rank payload helpers; it is disabled by default. The `bytecode-speed`,
`record-decode` and `record-replay` binaries are benchmark harnesses with explicit
input contracts, rather than normal file-compression commands.

## Train and apply a tokenizer

```sh
brepair train train.txt candidates.brp --min-count 8 --max-rules 130815 --threads 4 --memory 2GB
brepair vocab candidates.brp train.txt tokenizer.brt --vocab-size 8192 --memory 2GB
brepair evaluate tokenizer.brt holdout.txt --verify --memory 2GB
brepair encode tokenizer.brt holdout.txt tokens.bts --memory 2GB
brepair decode tokenizer.brt tokens.bts restored-holdout.txt --memory 2GB
```

Vocabulary selection uses training data; evaluation and encoding use the frozen
model. Vocabulary budgets include all 256 byte values and the end-of-field
symbol. The selected vocabulary is a heuristic; segmentation minimizes emitted
tokens for that vocabulary. The candidate-rule ceiling in this example is
optional: omit `--max-rules` to discover until exhaustion.

For JSONL, extract the intended text values before training. Passing JSONL
directly trains on the literal JSON syntax. Newline-delimited fields retain
their original bytes, including CRLF when present.

A fixed tokenizer can become the terminal alphabet of a reusable compression
model:

```sh
brepair codec-train tokenizer.brt train.txt frozen.brc --min-count 8 --threads 4 --memory 2GB
brepair codec-encode frozen.brc holdout.txt compressed.brh --threads 4 --memory 2GB
brepair huffman-ids compressed.brh tokens.ids --threads 4 --memory 2GB
```

Use `huffman-decode` to reconstruct bytes from the same archive. The default
advanced encoding mode preserves the tokenizer's exact segmentation; the
optional greedy mode can emit different token IDs.

## Memory and output handling

Every command accepts `--memory SIZE`; `MB`/`GB` are decimal and `MiB`/`GiB`
are binary. This bounds tracked allocations, with runtime headroom reserved
inside the budget. Windows also applies a process private-commit backstop.
Linux RSS, filesystem cache and temporary disk space are separate quantities.
A particular input or model can require a larger budget.

Output paths must be new. Final names appear after successful completion;
allocation failures stop with an error. Keep source data until restoration
has been verified. `--report FILE.json` is available on the principal training
and compression commands; `brepair --help` lists each command's options.

## Development

`src/` contains the library and CLI, and `tests/` contains Rust integration
tests. The public entry point is `src/bin/brepair.rs`.

```sh
cargo fmt --check
cargo test --locked --lib --bin brepair --test codec --test huffman
cargo clippy --locked --lib --bin brepair --test codec --test huffman -- -D warnings
cargo check --locked --all-targets --all-features
```

The focused suite covers library behavior, the public CLI and codec/Huffman
round trips. Further integration tests are available through `cargo test`.
Local-corpus audits are explicitly ignored unless requested. Some research
executables require separate datasets or optional native libraries; these are
not needed for the public CLI.

## Attribution and licence

Re-Pair and canonical Huffman coding are established techniques. BRe-Pair is
developed by **Wybren Kapenga**, with implementation, testing and documentation
assistance from **OpenAI Codex**. The release decoder follows Kapenga's
Decoder2 design with the project's prefix-length and refill optimizations.
External comparison implementations are not bundled.

Project source is licensed under **Apache-2.0**. See [LICENSE](LICENSE),
[NOTICE](NOTICE) and [third-party notices](THIRD-PARTY-NOTICES.md).
