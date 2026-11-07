# Third-party notices

BRe-Pair has no external Cargo crate dependencies. Project source is licensed
under Apache-2.0; see `LICENSE` and `NOTICE`.

Compiled executables include Rust standard-library components available under
MIT or Apache-2.0. Copies of those upstream texts from Rust 1.87.0 are included
in `third-party/`, with source URLs and SHA-256 values in `third-party/sources.json`.
Their original terms continue to apply to those components.

Windows builds use the installed MSVC toolchain and statically link the
Microsoft C runtime. Redistribution of toolchain runtime components remains
subject to Microsoft's applicable terms. Operating-system libraries are
provided by the operating system.

The release Huffman decoder follows Wybren Kapenga's Decoder2 canonical
range/rank design, with the project's prefix-length and refill optimizations.
Previously evaluated source-derived grouped/SIMD ports are excluded from this
tree. External decoder and Re-Pair comparison implementations, their binaries,
and corpus datasets are not included.
