# Huffman and byte-aligned payloads

The default archive uses shared canonical Huffman coding for dictionary
references and roots. Helper inlining and arity coding select the smallest
complete checkpoint. `huffman-repack` changes arity metadata without training
a new grammar or changing the root sequence.

The optional byte-aligned archive retains the frozen grammar and rank sequence
and assigns one-to-four-byte codes through a lead-byte bracket lookup:

```sh
cargo build --profile dist --features bytecode-experiments --bin byte-archive
byte-archive pack saved.brh column.brb --memory 2GB
byte-archive decode column.brb restored.txt --threads 4 --memory 3GB
```

BRBYTEA1 stores an 80-byte outer header, original compressed model prefix,
16-byte bracket descriptor, chunk headers/payloads, and an input/output-offset
index. Checksums and byte/field/symbol totals are checked. Decode needs only
the new archive. One/four workers share immutable expansions and own bounded
IO buffers. Fully cached expansions must fit the supplied arena/process cap.
This transport uses scalar rank extraction and native copies; it does not
claim an AVX2 byte-symbol hot loop.

EOF terminates fields, including when fused into a learned expansion. Ordinary
byte-aligned coding has no universally scannable EOF byte; the separate
`bytecode::scannable` prototype offers an opt-in delimiter contract. Neither
mode locates arbitrary source positions without suitable indexing. The
research `record-byte` adapter uses an N+1 offset index for independent fields.

Byte alignment changes storage and rank decoding, not discovery or raw-byte
parsing. Benchmark conversions charge model, descriptor, index and payload,
and add conversion CPU to the preceding encoding workflow.
