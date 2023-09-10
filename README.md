# `rsonpath` &ndash; SIMD-powered JSONPath 🚀

[![Rust](https://github.com/V0ldek/rsonpath/actions/workflows/rust.yml/badge.svg)](https://github.com/V0ldek/rsonpath/actions/workflows/rust.yml)
[![docs.rs](https://img.shields.io/docsrs/rsonpath-lib?logo=docs.rs)](https://docs.rs/crate/rsonpath-lib/latest)
[![Book](https://img.shields.io/badge/book-available-4DC720?logo=mdbook)](https://v0ldek.github.io/rsonpath/)

[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/7790/badge)](https://www.bestpractices.dev/projects/7790)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/V0ldek/rsonpath/badge)](https://securityscorecards.dev/viewer/?uri=github.com/V0ldek/rsonpath)

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
rq '$..a.b' ./file.json
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

### `cargo`

Easiest way to install is via [`cargo`](https://doc.rust-lang.org/cargo/getting-started/installation.html).

```console,ignore
$ cargo install rsonpath
...
```

### Native CPU optimizations

If maximum speed is paramount, you should install `rsonpath` with native CPU instructions support.
This will result in a binary that is _not_ portable and might work incorrectly on any other machine,
but will squeeze out every last bit of throughput.

To do this, run the following `cargo install` variant:

```console,ignore
$ RUSTFLAGS="-C target-cpu=native" cargo install rsonpath
...
```

Check out [the relevant chapter in the rsonbook](https://v0ldek.github.io/rsonpath/user/installation/manual.html).

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

The crate is continuously built for all Tier 1 Rust targets, and tests are continuously ran for targets that can be ran with GitHub action images.
SIMD is supported only on x86/x86_64 platforms.

| Target triple             | nosimd build | SIMD support        | Continuous testing | Tracking issues |
|:--------------------------|:-------------|:--------------------|:-------------------|----------------:|
| aarch64-unknown-linux-gnu | ✔️          | ❌                  | ❌                | [#21](https://github.com/V0ldek/rsonpath/issues/21), [#115](https://github.com/V0ldek/rsonpath/issues/115) |
| i686-unknown-linux-gnu    | ✔️          | ✔️                  | ✔️                | |
| x86_64-unknown-linux-gnu  | ✔️          | ✔️                  | ✔️                | |
| x86_64-apple-darwin       | ✔️          | ❌                  | ✔️                | |
| i686-pc-windows-gnu       | ✔️          | ✔️                  | ✔️                | |
| i686-pc-windows-msvc      | ✔️          | ✔️                  | ✔️                | |
| x86_64-pc-windows-gnu     | ✔️          | ✔️                  | ✔️                | |
| x86_64-pc-windows-msvc    | ✔️          | ✔️                  | ✔️                | |

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

We have a paper on `rsonpath` to be published at [ASPLOS '24](https://www.asplos-conference.org/asplos2024/)! You can read it
[here]([my thesis](/pdf/fast-execution-of-jsonpath-queries.pdf)).

This project was conceived as [my thesis](/pdf/fast-execution-of-jsonpath-queries.pdf). You can read it for details on the theoretical
background on the engine and details of its implementation.

## Dependencies

Showing direct dependencies, for full graph see below.

```bash
cargo tree --package rsonpath --edges normal --depth 1
```

<!-- rsonpath dependencies start -->
```ini
rsonpath v0.7.1 (/home/mat/rsonpath/crates/rsonpath)
├── clap v4.4.2
├── color-eyre v0.6.2
├── eyre v0.6.8
├── log v0.4.20
├── rsonpath-lib v0.7.1 (/home/mat/rsonpath/crates/rsonpath-lib)
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
rsonpath-lib v0.7.1 (/home/mat/rsonpath/crates/rsonpath-lib)
├── cfg-if v1.0.0
├── log v0.4.20
├── memmap2 v0.7.1
├── nom v7.1.3
├── smallvec v1.11.0
├── static_assertions v1.1.0
├── thiserror v1.0.48
└── vector-map v1.0.1
```
<!-- rsonpath-lib dependencies end -->

### Justification

- `clap` &ndash; standard crate to provide the CLI.
- `color-eyre`, `eyre` &ndash; more accessible error messages for the parser.
- `log`, `simple-logger` &ndash; diagnostic logs during compilation and execution.
- `cfg-if` &ndash; used to support SIMD and no-SIMD versions.
- `memmap2` &ndash; for fast reading of source files via a memory map instead of buffered copies.
- `nom` &ndash; for parser implementation.
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
rsonpath v0.7.1 (/home/mat/rsonpath/crates/rsonpath)
├── clap v4.4.2
│   ├── clap_builder v4.4.2
│   │   ├── anstream v0.5.0
│   │   │   ├── anstyle v1.0.2
│   │   │   ├── anstyle-parse v0.2.1
│   │   │   │   └── utf8parse v0.2.1
│   │   │   ├── anstyle-query v1.0.0
│   │   │   │   └── windows-sys v0.48.0
│   │   │   │       └── windows-targets v0.48.5
│   │   │   │           ├── windows_aarch64_gnullvm v0.48.5
│   │   │   │           ├── windows_aarch64_msvc v0.48.5
│   │   │   │           ├── windows_i686_gnu v0.48.5
│   │   │   │           ├── windows_i686_msvc v0.48.5
│   │   │   │           ├── windows_x86_64_gnu v0.48.5
│   │   │   │           ├── windows_x86_64_gnullvm v0.48.5
│   │   │   │           └── windows_x86_64_msvc v0.48.5
│   │   │   ├── anstyle-wincon v2.1.0
│   │   │   │   ├── anstyle v1.0.2
│   │   │   │   └── windows-sys v0.48.0 (*)
│   │   │   ├── colorchoice v1.0.0
│   │   │   └── utf8parse v0.2.1
│   │   ├── anstyle v1.0.2
│   │   ├── clap_lex v0.5.1
│   │   ├── strsim v0.10.0
│   │   └── terminal_size v0.2.6
│   │       ├── rustix v0.37.23
│   │       │   ├── bitflags v1.3.2
│   │       │   ├── errno v0.3.3
│   │       │   │   ├── errno-dragonfly v0.1.2
│   │       │   │   │   └── libc v0.2.147
│   │       │   │   │   [build-dependencies]
│   │       │   │   │   └── cc v1.0.83
│   │       │   │   │       └── libc v0.2.147
│   │       │   │   ├── libc v0.2.147
│   │       │   │   └── windows-sys v0.48.0 (*)
│   │       │   ├── io-lifetimes v1.0.11
│   │       │   │   ├── hermit-abi v0.3.2
│   │       │   │   ├── libc v0.2.147
│   │       │   │   └── windows-sys v0.48.0 (*)
│   │       │   ├── libc v0.2.147
│   │       │   ├── linux-raw-sys v0.3.8
│   │       │   └── windows-sys v0.48.0 (*)
│   │       └── windows-sys v0.48.0 (*)
│   └── clap_derive v4.4.2 (proc-macro)
│       ├── heck v0.4.1
│       ├── proc-macro2 v1.0.66
│       │   └── unicode-ident v1.0.11
│       ├── quote v1.0.33
│       │   └── proc-macro2 v1.0.66 (*)
│       └── syn v2.0.31
│           ├── proc-macro2 v1.0.66 (*)
│           ├── quote v1.0.33 (*)
│           └── unicode-ident v1.0.11
├── color-eyre v0.6.2
│   ├── backtrace v0.3.69
│   │   ├── addr2line v0.21.0
│   │   │   └── gimli v0.28.0
│   │   ├── cfg-if v1.0.0
│   │   ├── libc v0.2.147
│   │   ├── miniz_oxide v0.7.1
│   │   │   └── adler v1.0.2
│   │   ├── object v0.32.1
│   │   │   └── memchr v2.6.3
│   │   └── rustc-demangle v0.1.23
│   │   [build-dependencies]
│   │   └── cc v1.0.83 (*)
│   ├── eyre v0.6.8
│   │   ├── indenter v0.3.3
│   │   └── once_cell v1.18.0
│   ├── indenter v0.3.3
│   ├── once_cell v1.18.0
│   └── owo-colors v3.5.0
├── eyre v0.6.8 (*)
├── log v0.4.20
├── rsonpath-lib v0.7.1 (/home/mat/rsonpath/crates/rsonpath-lib)
│   ├── cfg-if v1.0.0
│   ├── log v0.4.20
│   ├── memmap2 v0.7.1
│   │   └── libc v0.2.147
│   ├── nom v7.1.3
│   │   ├── memchr v2.6.3
│   │   └── minimal-lexical v0.2.1
│   ├── smallvec v1.11.0
│   ├── static_assertions v1.1.0
│   ├── thiserror v1.0.48
│   │   └── thiserror-impl v1.0.48 (proc-macro)
│   │       ├── proc-macro2 v1.0.66 (*)
│   │       ├── quote v1.0.33 (*)
│   │       └── syn v2.0.31 (*)
│   └── vector-map v1.0.1
│       ├── contracts v0.4.0 (proc-macro)
│       │   ├── proc-macro2 v1.0.66 (*)
│       │   ├── quote v1.0.33 (*)
│       │   └── syn v1.0.109
│       │       ├── proc-macro2 v1.0.66 (*)
│       │       ├── quote v1.0.33 (*)
│       │       └── unicode-ident v1.0.11
│       └── rand v0.7.3
│           ├── getrandom v0.1.16
│           │   ├── cfg-if v1.0.0
│           │   ├── libc v0.2.147
│           │   └── wasi v0.9.0+wasi-snapshot-preview1
│           ├── libc v0.2.147
│           ├── rand_chacha v0.2.2
│           │   ├── ppv-lite86 v0.2.17
│           │   └── rand_core v0.5.1
│           │       └── getrandom v0.1.16 (*)
│           ├── rand_core v0.5.1 (*)
│           └── rand_hc v0.2.0
│               └── rand_core v0.5.1 (*)
└── simple_logger v4.2.0
    ├── colored v2.0.4
    │   ├── is-terminal v0.4.9
    │   │   ├── hermit-abi v0.3.2
    │   │   ├── rustix v0.38.12
    │   │   │   ├── bitflags v2.4.0
    │   │   │   ├── errno v0.3.3 (*)
    │   │   │   ├── libc v0.2.147
    │   │   │   ├── linux-raw-sys v0.4.5
    │   │   │   └── windows-sys v0.48.0 (*)
    │   │   └── windows-sys v0.48.0 (*)
    │   ├── lazy_static v1.4.0
    │   └── windows-sys v0.48.0 (*)
    ├── log v0.4.20
    ├── time v0.3.28
    │   ├── deranged v0.3.8
    │   ├── itoa v1.0.9
    │   ├── libc v0.2.147
    │   ├── num_threads v0.1.6
    │   │   └── libc v0.2.147
    │   ├── time-core v0.1.1
    │   └── time-macros v0.2.14 (proc-macro)
    │       └── time-core v0.1.1
    └── windows-sys v0.42.0
        ├── windows_aarch64_gnullvm v0.42.2
        ├── windows_aarch64_msvc v0.42.2
        ├── windows_i686_gnu v0.42.2
        ├── windows_i686_msvc v0.42.2
        ├── windows_x86_64_gnu v0.42.2
        ├── windows_x86_64_gnullvm v0.42.2
        └── windows_x86_64_msvc v0.42.2
[build-dependencies]
├── rustflags v0.1.4
└── vergen v8.2.4
    ├── anyhow v1.0.75
    ├── rustc_version v0.4.0
    │   └── semver v1.0.18
    └── time v0.3.28
        ├── deranged v0.3.8
        ├── itoa v1.0.9
        ├── libc v0.2.147
        ├── num_threads v0.1.6 (*)
        └── time-core v0.1.1
    [build-dependencies]
    └── rustversion v1.0.14 (proc-macro)
```
<!-- rsonpath-full dependencies end -->
