# Third-party notices

The executable includes components of the Rust standard library, available
under MIT or Apache License 2.0. Copies of those upstream license texts from
the Rust 1.87.0 source release are included in `third-party/`, together with
their source URLs and SHA-256 hashes. Upstream licenses continue to apply to
their respective components; the BRe-Pair license does not replace
them.

Windows x86-64 builds link the Microsoft C runtime statically through the
installed MSVC toolchain. The toolchain's redistribution terms apply to those
components. Imported operating-system libraries remain provided by Windows;
the package manifest records their names.

BRe-Pair has no external Cargo crate dependencies. The project source is
licensed under Apache-2.0; see `LICENSE` and `NOTICE`.

The large-alphabet grouped/SIMD Huffman backend in `src/huffman_simd.rs` is
adapted from Igor Zavadskyi and Maksym Kovalchuk's Fast-Large-Huffman code,
licensed under Apache-2.0 at revision
`ada4c1cb10b45b4b9aecd3e88b50756facd4d89e`. Its license is included as
`third-party/FAST-LARGE-HUFFMAN-LICENSE-APACHE.txt`; source URLs, hashes and
local modifications are recorded in `third-party/fast-large-huffman.json`.
The portable and long-code fallback retains the owner's Decoder2-style
canonical range path with the project's prefix-length/refill optimizations.

Credit: I. Zavadskyi and M. Kovalchuk, “Engineering a Faster Huffman Decoder,”
DCC 2026, pp. 213–222, DOI
[10.1109/DCC66757.2026.00029](https://doi.org/10.1109/DCC66757.2026.00029),
and the preceding 2025 work, DOI
[10.17721/1812-5409.2025/2.25](https://doi.org/10.17721/1812-5409.2025/2.25).
The historical removal before the license was identified remains documented;
this restoration uses the licensed revision and retains attribution.
External reference implementations used by research scripts retain their own
terms and are not bundled here.
