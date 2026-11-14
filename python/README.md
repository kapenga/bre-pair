# BRe-Pair for Python

Native, byte-exact tokenization using a frozen BRe-Pair vocabulary, plus the
existing bounded CLI workflows for discovery and static Huffman compression.
Requires Python 3.10+ on a 64-bit platform. Building from source needs Rust
1.87+ and the platform's Rust linker/toolchain.

From the repository root:

```console
python -m pip install .
```

A built platform wheel contains the native library, `brepair` executable and
license notices; using that wheel requires no Rust installation. Python runtime
dependencies are limited to its standard library.

```python
from brepair import Tokenizer

with Tokenizer("vocabulary.brt") as tokenizer:
    original = "Hello, 世界!".encode("utf-8")
    ids = tokenizer.encode(original)
    assert tokenizer.decode(ids) == original
    print(tokenizer.vocabulary_size, ids)
```

Each call encodes **one complete field**, including EOF. EOF may be fused into
a learned token, so removing ID 256 is not a general way to strip it. The model
keeps IDs 0–255 for bytes and 256 for the empty EOF. No Unicode normalization,
replacement or implicit newline is performed. Decode returns bytes; explicitly
call `.decode("utf-8")` on that result when text is required.

`encode_fields(iterable)` yields one ID list per field. Parser scratch is reused
and grows with the largest field. Native calls release the GIL; one handle
serializes its workspace. `fork()` shares the immutable model while giving a
worker its own workspace and lifetime. Close handles or use context managers.
`from_bytes()`, `to_bytes()` and exclusive `save()` support model persistence.
`decode(ids, maximum_bytes=...)` checks a field's output limit.

```python
from brepair import train_file, select_vocabulary, compress_file, decompress_file

train_file("train.txt", "candidates.brp", minimum_count=256,
           threads=4, memory="2GB")
select_vocabulary("candidates.brp", "train.txt", "vocabulary.brt",
                  vocabulary_size=32768, memory="2GB")
compress_file("input.txt", "archive.brh", memory="2GB")
decompress_file("archive.brh", "restored.txt", threads=4, memory="2GB")
```

These file operations run the bundled executable without loading the entire
corpus into Python. They return a `RunResult` with command, output and elapsed
time, and raise on failure. Outputs must be new. `select_vocabulary` uses the
supported CLI statistics selector; research workbench selection and word-PMI
experiments remain separate. Training defaults to natural exhaustion of the
declared count floor, with no implicit twice-the-floor stopping rule.

Set `BREPAIR_EXECUTABLE` or pass `executable=...` to select another CLI build.
`BREPAIR_NATIVE_LIBRARY` selects a development native library. Binary model and
archive formats match the Rust executable. This package is an initial API;
the private C ABI is version-checked and not a general untrusted-pointer API.

For unquoted pipe-separated tables, column models are independent and all
stored model/layout/manifest bytes are included in the returned complete size:

```python
from brepair import compress_columns, decompress_columns, decompress_column

info = compress_columns("table.psv", "table-columns", columns=12,
                        threads=4, memory="2GB")
decompress_column("table-columns", 3, "column-3.txt")
decompress_columns("table-columns", "restored.psv")
```

Columns train serially under the supplied per-instance budget. Values preserve
their lexical bytes, including empty values, leading zeros, null spellings,
literal quotes and binary bytes. LF separates rows; CRLF and a missing final
LF reconstruct exactly. Embedded separators/newlines in quoted CSV are outside
this unquoted-PSV contract. The default `fields="lines"` terminates each value
with EOF; `fields="single"` is a continuous-column compression experiment.
Full restoration verifies SHA-256 before publishing the output. Column
reconstruction uses temporary disk files beside the requested output.
Projected-column decoding is exposed now; per-cell lookup and its access index
remain separate from these file APIs.
