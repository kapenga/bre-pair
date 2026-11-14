# Static decoder

One immutable canonical table maps codes to ranks; prepared expansions map
ranks to byte slices or iterative grammar traversal when uncached. The arena
is bounded and shared. Each reader owns its input cursor and output scratch.

The scalar range/rank path derives from Wybren Kapenga's PCMA Decoder2, with
prefix-length lookup, wide refill and checked tails. Block decoding can select
the attributed grouped AVX2 kernel adapted from Zavadskyi and Kovalchuk's
Fast-Large-Huffman at revision `ada4c1cb10b45b4b9aecd3e88b50756facd4d89e`.
Automatic dispatch requires AVX2, at least 1,048,576 codes and maximum code
length at most 31 bits. The scalar fallback supports lengths through 56 bits.
The EOF field API retains the scalar path. Source attribution and licensing
are in `THIRD-PARTY-NOTICES.md` and `third-party/`.

The four-thread file decoder overlaps input IO, rank decoding, expansion and
output IO using recycled buffers. This differs from four independent record
workers or byte-archive block workers. One-thread decoding is also available.

`StaticModel::decode_field_into` checks model identity, EOF, padding and the
caller slice limit. Caller padding is unnecessary. Unused capacity is scratch;
discard partial output on failure. Mixed-cache block references use a lazy
16-byte entry per coded leaf alongside the bounded expansion arena.

For matched measurements use `huffman-decode --decoder auto|scalar|avx2`,
`--threads 1|4` and an explicit `--arena-mib` cap. End-to-end file measurements
include preparation and IO; prepared-RAM record loops exclude setup.
