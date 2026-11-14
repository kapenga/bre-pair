# Contributing

Use Rust 1.87+ on a 64-bit target. `src/` contains the shared library,
`src/cli/` command handling and `src/bin/brepair.rs` the public CLI.
Rust integration tests live in `tests/`; Python/native integration tests
live in `python/tests/`. Keep research entry points separate from defaults.

```sh
cargo fmt --check
cargo test --locked --all-targets --features bytecode-experiments
cargo clippy --locked --all-targets --features bytecode-experiments -- -D warnings
cargo build --locked --profile dist --bin brepair
```

For Python changes, build/install the package and test the installed native
code, including columns and file round trips:

```sh
python -m pip install .
python -m unittest discover -s python/tests -v
cargo fmt --manifest-path bindings/python-native/Cargo.toml --check
```

Use rustfmt, four-space indentation, `snake_case` functions/modules and
`CamelCase` types. Preserve u64 frequencies and positions; u32 symbol IDs do
not justify narrowing counters. Algorithm changes need independent count,
conflict and reconstruction oracles. Corpus-specific audits are explicitly
ignored until their local data is supplied.

Changes should preserve existing archive compatibility or document a format
change. Record CPU, wall time, allocated heap and process RSS separately.
Benchmark complete workflows with explicit limits and IO scopes. Exact
measured-source snapshots in the benchmark companion remain immutable.

Use concise imperative commits. Pull requests should explain the behavior,
report validation and identify checks not run. Preserve Apache-2.0 and all
third-party attribution. No coverage threshold is configured.
