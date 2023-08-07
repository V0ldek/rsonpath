# `rsonpath` &ndash; SIMD-powered JSONPath 🚀

[![Rust](https://github.com/V0ldek/rsonpath/actions/workflows/rust.yml/badge.svg)](https://github.com/V0ldek/rsonpath/actions/workflows/rust.yml)
[![docs.rs](https://img.shields.io/docsrs/rsonpath-lib?logo=docs.rs)](https://docs.rs/crate/rsonpath-lib/latest)
[![Book](https://img.shields.io/badge/book-available-4DC720?logo=mdbook
)](https://v0ldek.github.io/rsonpath/)

[![Crates.io](https://img.shields.io/crates/v/rsonpath?logo=docs.rs)](https://crates.io/crates/rsonpath)
[![GitHub Release Date](https://img.shields.io/github/release-date/v0ldek/rsonpath?logo=github)](https://github.com/V0ldek/rsonpath/releases)
[![GitHub last commit](https://img.shields.io/github/last-commit/v0ldek/rsonpath?logo=github)](https://github.com/V0ldek/rsonpath/commits/main)

![MSRV](https://img.shields.io/badge/msrv-v1.70.0-orange?logo=rust "Minimum Supported Rust Version for `rq`")
[![License](https://img.shields.io/crates/l/rsonpath)](https://choosealicense.com/licenses/mit/)

Experimental JSONPath engine for querying massive streamed datasets.

The `rsonpath` crate provides a JSONPath parser and a query execution engine `rq`,
which utilizes SIMD instructions to provide massive throughput improvements over conventional engines.

Benchmarks of `rsonpath` against a reference no-SIMD engine on the
[Pison dataset](https://github.com/AutomataLab/Pison). **NOTE: Scale is logarithmic!**
![Main throughput plot](/img/main-plot.svg)

## Usage

To run a JSONPath query on a file execute:

```console,ignore
$ rq '$..a.b' ./file.json
```

If the file is omitted, the engine reads standard input. JSON can also be passed inline:

```console
$ rq '$..a.b' --json '{"c":{"a":{"b":42}}}'
42

```

For details, consult `rq --help` or [the rsonbook](https://v0ldek.github.io/rsonpath/).

### Results

The result of running a query is a sequence of matched values, delimited by newlines.
Alternatively, passing `--result count` returns only the number of matches, which might be much faster.
For other result modes consult the `--help` usage page.

## Installation

See [Releases](https://github.com/V0ldek/rsonpath/releases/latest) for precompiled binaries for
all first-class support targets.

Easiest way to install is via [`cargo`](https://doc.rust-lang.org/cargo/getting-started/installation.html).

```console,ignore
$ cargo install rsonpath
...
```

This might fail with the following error:

```console,ignore
Target architecture is not supported by SIMD features of this crate. Disable the default `simd` feature.
```

This means the SIMD features of the engine are not implemented for the machine's CPU.
You can still use `rsonpath`, but the speed will be limited (see the reference engine in the chart above). To install without simd, run `cargo install --no-default-features -F default-optimizations`.

Alternatively, you can download the source code and manually run `just install` (requires [`just`](https://github.com/casey/just))
or `cargo install --path ./crates/rsonpath`.

### Native CPU optimizations

If maximum speed is paramount, you should install `rsonpath` with native CPU instructions support.
This will result in a binary that is _not_ portable and might work incorrectly on any other machine,
but will squeeze out every last bit of throughput.

To do this, run the following `cargo install` variant:

```console,ignore
$ RUSTFLAGS="-C target-cpu=native" cargo install rsonpath
...
```

## Query language

The project is actively developed and currently supports only a subset of the JSONPath query language.
A query is a sequence of segments, each containing one or more selectors.

### Supported segments

| Segment                        | Syntax                           | Supported | Since  | Tracking Issue |
|--------------------------------|----------------------------------|-----------|--------|---------------:|
| Child segment (single)         | `[<selector>]`                   | ✔️        | v0.1.0 |                |
| Child segment (multiple)       | `[<selector1>,...,<selectorN>]`  | ❌        |        |                |
| Descendant segment (single)    | `..[<selector>]`                 | ✔️        | v0.1.0 |                |
| Descendant segment (multiple)  | `..[<selector1>,...,<selectorN>]`| ❌        |        |                |

### Supported selectors

| Selector                                 | Syntax                           | Supported | Since  | Tracking Issue |
|------------------------------------------|----------------------------------|-----------|--------|---------------:|
| Root                                     | `$`                              | ✔️        | v0.1.0 |                |
| Name                                     | `.<member>`, `[<member>]`        | ✔️        | v0.1.0 |                |
| Wildcard                                 | `.*`, `..*`, `[*]`               | ✔️        | v0.4.0 |                |
| Index (array index)                      | `[<index>]`                      | ✔️        | v0.5.0 |                |
| Index (array index from end)             | `[-<index>]`                     | ❌        |        |                |
| Array slice (forward, positive bounds)   | `[<start>:<end>:<step>]`         | ❌        |        | [#152](https://github.com/V0ldek/rsonpath/issues/152) |
| Array slice (forward, arbitrary bounds)  | `[<start>:<end>:<step>]`         | ❌        |        |                |
| Array slice (backward, arbitrary bounds) | `[<start>:<end>:-<step>]`        | ❌        |        |                |
| Filters &ndash; existential tests        | `[?<path>]`                      | ❌        |        | [#154](https://github.com/V0ldek/rsonpath/issues/154) |
| Filters &ndash; const atom comparisons   | `[?<path> <binop> <atom>]`       | ❌        |        | [#156](https://github.com/V0ldek/rsonpath/issues/156) |
| Filters &ndash; logical expressions      | `&&`, `\|\|`, `!`                | ❌        |        |                |
| Filters &ndash; nesting                  | `[?<expr>[?<expr>]...]`          | ❌        |        |                |
| Filters &ndash; arbitrary comparisons    | `[?<path> <binop> <path>]`       | ❌        |        |                |
| Filters &ndash; function extensions      | `[?func(<path>)]`                | ❌        |        |                |

## Supported platforms

The crate is continuously built for all Tier 1 Rust targets, and tests are continuously ran for targets that can be ran with GitHub action images. SIMD is supported only on x86-64 platforms for AVX2, while nosimd builds are always available for all targets.

| Target triple             | nosimd build | SIMD support        | Continuous testing | Tracking issues |
|:--------------------------|:-------------|:--------------------|:-------------------|----------------:|
| aarch64-unknown-linux-gnu | ✔️          | ❌                  | ❌                | [#21](https://github.com/V0ldek/rsonpath/issues/21), [#115](https://github.com/V0ldek/rsonpath/issues/115) |
| i686-unknown-linux-gnu    | ✔️          | ❌                  | ✔️                | [#14](https://github.com/V0ldek/rsonpath/issues/14) |
| x86_64-unknown-linux-gnu  | ✔️          | ✔️ avx2+pclmulqdq   | ✔️                | |
| x86_64-apple-darwin       | ✔️          | ❌                  | ✔️                | |
| i686-pc-windows-gnu       | ✔️          | ❌                  | ✔️                | [#14](https://github.com/V0ldek/rsonpath/issues/14) |
| i686-pc-windows-msvc      | ✔️          | ❌                  | ✔️                | [#14](https://github.com/V0ldek/rsonpath/issues/14) |
| x86_64-pc-windows-gnu     | ✔️          | ✔️ avx2+pclmulqdq   | ✔️                | |
| x86_64-pc-windows-msvc    | ✔️          | ✔️ avx2+pclmulqdq   | ✔️                | |

## Caveats and limitations

### JSONPath

Not all selectors are supported, see the support table above.

### Duplicate keys

The engine assumes that every object in the input JSON has no duplicate keys.
Behavior on duplicate keys is not guaranteed to be stable, but currently
the engine will simply match the _first_ such key.

```console
$ rq '$.key' --json '{"key":"value","key":"other value"}'
"value"

```

### Unicode

The engine does _not_ parse unicode escape sequences in member names.
This means that a key `"a"` is different from a key `"\u0041"`, even though semantically they represent the same string.
This is actually as-designed with respect to the current JSONPath spec.
Parsing unicode sequences is costly, so the support for this was postponed
in favour of high performance. This is tracked as [#117](https://github.com/v0ldek/rsonpath/issues/117).

## Contributing

The gist is: fork, implement, make a PR back here. More details are in the [CONTRIBUTING](/CONTRIBUTING.md) doc.

### Build & test

The dev workflow utilizes [`just`](https://github.com/casey/just).
Use the included `Justfile`. It will automatically install Rust for you using the `rustup` tool if it detects there is no Cargo in your environment.

```console,ignore
$ just build
...
$ just test
...
```

## Benchmarks

Benchmarks for `rsonpath` are located in a [separate repository](https://github.com/v0ldek/rsonpath-benchmarks),
included as a [git submodule](https://git-scm.com/book/en/v2/Git-Tools-Submodules) in this main repository.

Easiest way to run all the benchmarks is `just bench`. For details, look at the README in the submodule.

## Background

This project is the result of [my thesis](/pdf/Fast_execution_of_JSONPath_queries.pdf). You can read it for details on the theoretical
background on the engine and details of its implementation.

## Dependencies

Showing direct dependencies, for full graph see below.

```bash
cargo tree --package rsonpath --edges normal --depth 1
```

<!-- rsonpath dependencies start -->
```ini
rsonpath v0.6.0 (/home/mat/rsonpath/crates/rsonpath)
├── clap v4.3.19
├── color-eyre v0.6.2
├── eyre v0.6.8
├── log v0.4.19
├── rsonpath-lib v0.6.0 (/home/mat/rsonpath/crates/rsonpath-lib)
└── simple_logger v4.2.0
[build-dependencies]
├── rustflags v0.1.4
└── vergen v8.2.4
    [build-dependencies]
```
<!-- rsonpath dependencies end -->

```bash
cargo tree --package rsonpath-lib --edges normal --depth 1
```

<!-- rsonpath-lib dependencies start -->
```ini
rsonpath-lib v0.6.0 (/home/mat/rsonpath/crates/rsonpath-lib)
├── cfg-if v1.0.0
├── log v0.4.19
├── memchr v2.5.0
├── memmap2 v0.7.1
├── nom v7.1.3
├── replace_with v0.1.7
├── smallvec v1.11.0
├── static_assertions v1.1.0
├── thiserror v1.0.44
└── vector-map v1.0.1
```
<!-- rsonpath-lib dependencies end -->

### Justification

- `clap` &ndash; standard crate to provide the CLI.
- `color-eyre`, `eyre` &ndash; more accessible error messages for the parser.
- `log`, `simple-logger` &ndash; diagnostic logs during compilation and execution.

- `cfg-if` &ndash; used to support SIMD and no-SIMD versions.
- `memchr` &ndash; rapid, SIMDified substring search for fast-forwarding to labels.
- `memmap2` &ndash; for fast reading of source files via a memory map instead of buffered copies.
- `nom` &ndash; for parser implementation.
- `replace_with` &ndash; for safe handling of internal classifier state when switching classifiers.
- `smallvec` &ndash; crucial for small-stack performance.
- `static_assertions` &ndash; additional reliability by some constant assumptions validated at compile time.
- `thiserror` &ndash; idiomatic `Error` implementations.
- `vector_map` &ndash; used in the query compiler for measurably better performance.

## Full dependency tree

```bash
cargo tree --package rsonpath --edges normal
```

<!-- rsonpath-full dependencies start -->
```ini
rsonpath v0.6.0 (/home/mat/rsonpath/crates/rsonpath)
├── clap v4.3.19
│   ├── clap_builder v4.3.19
│   │   ├── anstream v0.3.2
│   │   │   ├── anstyle v1.0.1
│   │   │   ├── anstyle-parse v0.2.1
│   │   │   │   └── utf8parse v0.2.1
│   │   │   ├── anstyle-query v1.0.0
│   │   │   ├── colorchoice v1.0.0
│   │   │   ├── is-terminal v0.4.9
│   │   │   │   └── rustix v0.38.6
│   │   │   │       ├── bitflags v2.3.3
│   │   │   │       └── linux-raw-sys v0.4.5
│   │   │   └── utf8parse v0.2.1
│   │   ├── anstyle v1.0.1
│   │   ├── clap_lex v0.5.0
│   │   ├── strsim v0.10.0
│   │   └── terminal_size v0.2.6
│   │       └── rustix v0.37.23
│   │           ├── bitflags v1.3.2
│   │           ├── io-lifetimes v1.0.11
│   │           │   └── libc v0.2.147
│   │           ├── libc v0.2.147
│   │           └── linux-raw-sys v0.3.8
│   ├── clap_derive v4.3.12 (proc-macro)
│   │   ├── heck v0.4.1
│   │   ├── proc-macro2 v1.0.66
│   │   │   └── unicode-ident v1.0.11
│   │   ├── quote v1.0.32
│   │   │   └── proc-macro2 v1.0.66 (*)
│   │   └── syn v2.0.28
│   │       ├── proc-macro2 v1.0.66 (*)
│   │       ├── quote v1.0.32 (*)
│   │       └── unicode-ident v1.0.11
│   └── once_cell v1.18.0
├── color-eyre v0.6.2
│   ├── backtrace v0.3.68
│   │   ├── addr2line v0.20.0
│   │   │   └── gimli v0.27.3
│   │   ├── cfg-if v1.0.0
│   │   ├── libc v0.2.147
│   │   ├── miniz_oxide v0.7.1
│   │   │   └── adler v1.0.2
│   │   ├── object v0.31.1
│   │   │   └── memchr v2.5.0
│   │   └── rustc-demangle v0.1.23
│   │   [build-dependencies]
│   │   └── cc v1.0.81
│   │       └── libc v0.2.147
│   ├── eyre v0.6.8
│   │   ├── indenter v0.3.3
│   │   └── once_cell v1.18.0
│   ├── indenter v0.3.3
│   ├── once_cell v1.18.0
│   └── owo-colors v3.5.0
├── eyre v0.6.8 (*)
├── log v0.4.19
├── rsonpath-lib v0.6.0 (/home/mat/rsonpath/crates/rsonpath-lib)
│   ├── cfg-if v1.0.0
│   ├── log v0.4.19
│   ├── memchr v2.5.0
│   ├── memmap2 v0.7.1
│   │   └── libc v0.2.147
│   ├── nom v7.1.3
│   │   ├── memchr v2.5.0
│   │   └── minimal-lexical v0.2.1
│   ├── replace_with v0.1.7
│   ├── smallvec v1.11.0
│   ├── static_assertions v1.1.0
│   ├── thiserror v1.0.44
│   │   └── thiserror-impl v1.0.44 (proc-macro)
│   │       ├── proc-macro2 v1.0.66 (*)
│   │       ├── quote v1.0.32 (*)
│   │       └── syn v2.0.28 (*)
│   └── vector-map v1.0.1
│       ├── contracts v0.4.0 (proc-macro)
│       │   ├── proc-macro2 v1.0.66 (*)
│       │   ├── quote v1.0.32 (*)
│       │   └── syn v1.0.109
│       │       ├── proc-macro2 v1.0.66 (*)
│       │       ├── quote v1.0.32 (*)
│       │       └── unicode-ident v1.0.11
│       └── rand v0.7.3
│           ├── getrandom v0.1.16
│           │   ├── cfg-if v1.0.0
│           │   └── libc v0.2.147
│           ├── libc v0.2.147
│           ├── rand_chacha v0.2.2
│           │   ├── ppv-lite86 v0.2.17
│           │   └── rand_core v0.5.1
│           │       └── getrandom v0.1.16 (*)
│           └── rand_core v0.5.1 (*)
└── simple_logger v4.2.0
    ├── colored v2.0.4
    │   ├── is-terminal v0.4.9 (*)
    │   └── lazy_static v1.4.0
    ├── log v0.4.19
    └── time v0.3.25
        ├── deranged v0.3.7
        ├── itoa v1.0.9
        ├── libc v0.2.147
        ├── num_threads v0.1.6
        ├── time-core v0.1.1
        └── time-macros v0.2.11 (proc-macro)
            └── time-core v0.1.1
[build-dependencies]
├── rustflags v0.1.4
└── vergen v8.2.4
    ├── anyhow v1.0.72
    ├── rustc_version v0.4.0
    │   └── semver v1.0.18
    └── time v0.3.25
        ├── deranged v0.3.7
        ├── itoa v1.0.9
        ├── libc v0.2.147
        ├── num_threads v0.1.6
        └── time-core v0.1.1
    [build-dependencies]
    └── rustversion v1.0.14 (proc-macro)
```
<!-- rsonpath-full dependencies end -->
