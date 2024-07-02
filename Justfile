set windows-shell := ["pwsh.exe", "-Command"]

[private]
default: (build-all "release")

# === INIT ===

# Initialize the repository for development.
init: check-cargo hooks-init checkout-benchmarks

# Check if cargo is installed and install it from rustup if not.
[private]
@check-cargo:
    cargo --version || \
      (echo "Installing rustup from https://sh.rustup.rs" && \
       curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y)
    . ${HOME}/.cargo/env
    rustup install stable
    rustup install nightly

# Initialize git hooks.
[private]
@hooks-init:
    echo "#!/bin/sh\n\njust hook-pre-commit" > ./.git/hooks/pre-commit
    echo "#!/bin/sh\n\njust hook-post-checkout" > ./.git/hooks/post-checkout
    chmod u+x ./.git/hooks/pre-commit
    chmod u+x ./.git/hooks/post-checkout

# Checkout and populate the benchmarks repository, excluding datasets.
[private]
checkout-benchmarks:
    git submodule init
    git submodule update

# === BUILD ===

alias b := build-bin

# alias for build-all release
build profile="release": (build-all profile)

# Build the rq binary.
build-bin profile="dev": (build-lib profile)
    cargo build --bin rq --profile {{profile}}

# Build the rsonpath-lib library.
build-lib profile="dev":
    cargo build --package rsonpath-lib --profile {{profile}}

build-avx512 profile="dev":
    rustup run nightly cargo build  --package rsonpath-lib --profile dev

# Build all rsonpath parts, the binary and library.
build-all profile="dev": (build-lib profile) (build-bin profile) (gen-tests)

# Build and open the library documentation.
doc $RUSTDOCFLAGS="--cfg docsrs":
    cargo +nightly doc --open --all-features -Z rustdoc-scrape-examples

# Run the codegen for rsonpath-test, generating the E2E tests and JSONs.
gen-tests:
    RSONPATH_ENABLE_TEST_CODEGEN=1 cargo build --package rsonpath-test

# === RUN ===

alias r := run-debug

# Run the CLI in debug profile. ARGS are passed to the rsonpath executable.
[no-exit-message]
run-debug *ARGS: (build-bin "dev")
    ./target/debug/rq {{ARGS}}

# Run the CLI in release profile. ARGS are passed to the rsonpath executable.
[no-exit-message]
run *ARGS: (build-bin "release")
    ./target/release/rq {{ARGS}}

# === WATCH ===
watch *ARGS:
    cargo watch -x "check" -x "test --lib -q" -x "test --doc -q" {{ARGS}}


# === TEST ===

# Run all tests.

alias t := test-quick
alias test := test-full
alias doctest := test-doc

# Run the quick unit and doc tests of the library with all features.
test-quick:
    cargo test --lib -q
    cargo test --doc -q

# Run the quick unit tests of the library on feature powerset.
test-unit:
    -cargo install cargo-hack
    cargo rsontest --lib

# Run the classifier tests on default features.
test-classifier:
    cargo test -p rsonpath-lib classifier_correctness -q

# Run the main engine end-to-end tests on default features.
test-engine: (gen-tests)
    cargo test -p rsonpath-test --tests -q

# Run the input tests on default features.
test-input:
    cargo test --test input_implementation_tests -q

# Run the query tests on default features.
test-parser:
    cargo test --test query_parser_tests -q

# Run all tests, including real dataset tests, on the feature powerset of the project.
test-full: (gen-tests)
    -cargo install cargo-hack
    cargo rsontest
    just test-book

# Run E2E engine tests on all combinations of SIMD features for x86 platforms.
test-x86-simd:
    RSONPATH_UNSAFE_FORCE_SIMD="avx2;fast_quotes;fast_popcnt" cargo test -p rsonpath-test --tests -q
    RSONPATH_UNSAFE_FORCE_SIMD="ssse3;fast_quotes;fast_popcnt" cargo test -p rsonpath-test --tests -q
    RSONPATH_UNSAFE_FORCE_SIMD="ssse3;fast_quotes;slow_popcnt" cargo test -p rsonpath-test --tests -q
    RSONPATH_UNSAFE_FORCE_SIMD="ssse3;slow_quotes;fast_popcnt" cargo test -p rsonpath-test --tests -q
    RSONPATH_UNSAFE_FORCE_SIMD="ssse3;slow_quotes;slow_popcnt" cargo test -p rsonpath-test --tests -q
    RSONPATH_UNSAFE_FORCE_SIMD="sse2;fast_quotes;fast_popcnt" cargo test -p rsonpath-test --tests -q
    RSONPATH_UNSAFE_FORCE_SIMD="sse2;fast_quotes;slow_popcnt" cargo test -p rsonpath-test --tests -q
    RSONPATH_UNSAFE_FORCE_SIMD="sse2;slow_quotes;fast_popcnt" cargo test -p rsonpath-test --tests -q
    RSONPATH_UNSAFE_FORCE_SIMD="sse2;slow_quotes;slow_popcnt" cargo test -p rsonpath-test --tests -q
    RSONPATH_UNSAFE_FORCE_SIMD="nosimd;slow_quotes;slow_popcnt" cargo test -p rsonpath-test --tests -q

# Run doctests on the library.
test-doc:
    cargo test --doc

# Run cmd tests
test-cmd:
    cargo test --test cli_tests

# Run doctests on the book.
test-book:
    rm -f ./target/debug/deps/librsonpath-*
    cargo build -p rsonpath-lib
    cargo build -p rsonpath-syntax
    mdbook test ./book -L ./target/debug/deps

@add-test name:
    f=`echo {{name}} | sed s/-/_/g` && \
        cp ./crates/rsonpath-test/documents/toml/test_template_inline.toml ./crates/rsonpath-test/documents/toml/$f.toml && \
        echo "Test template initialised at crates/rsonpath-test/documents/toml/$f.toml"

@add-test-large name:
    f=`echo {{name}} | sed s/-/_/g` && \
        cp ./crates/rsonpath-test/documents/toml/test_template_large.toml ./crates/rsonpath-test/documents/toml/$f.toml && \
        echo "Test template initialised at crates/rsonpath-test/documents/toml/$f.toml" && \
        echo "{}" > ./crates/rsonpath-test/documents/json/large/$f.json && \
        echo "Put your large JSON document as contents of crates/rsonpath-test/documents/json/large/$f.json"

# === INSTALL ===

# Install the rsonpath binary from current source.
install: (build-bin "release")
    cargo install --path ./crates/rsonpath

# Uninstall the rsonpath binary.
uninstall:
    cargo uninstall rsonpath

# === VERIFICATION/LINTING ===

alias v := verify-quick
alias verify := verify-full

# Run all lints and checks required.
verify-full: verify-quick test-full (build-bin "release")

# Run a quick formatting and compilation check.
verify-quick: verify-fmt verify-check verify-doc verify-deny verify-bench verify-clippy

# Run cargo check on non-benchmark packages.
verify-check:
    cargo check --workspace --all-features

# Run cargo deny check.
verify-deny:
    cargo deny check

# Run cargo check on the benchmark package
verify-bench:
    cargo check --manifest-path ./crates/rsonpath-benchmarks/Cargo.toml --all-features

# Run clippy lints on all packages.
verify-clippy: (build-all "release")
    cargo +nightly clippy --workspace --no-default-features --release -- --deny warnings
    cargo +nightly clippy --workspace --all-features --release -- --deny warnings

# Verify that documentation successfully builds for rsonpath-lib.
verify-doc $RUSTDOCFLAGS="--cfg docsrs -D warnings":
    cargo +nightly doc --package rsonpath-lib --no-default-features --no-deps --release
    cargo +nightly doc --package rsonpath-lib --all-features --no-deps --release
    cargo +nightly doc --package rsonpath-syntax --all-features --no-deps --release

# Verify formatting rules are not violated.
verify-fmt:
    cargo fmt -- --check

# === CLEAN ===

tmpdir := if os() == "windows" {
    `New-TemporaryFile`
} else {
    `mktemp -d -t criterion-reports-tmp-XXXXXXXX`
}

# Clean all build artifacts without deleting benchmark results.
clean:
    -cp -r ./target/criterion/* {{tmpdir}}/
    cargo clean
    mkdir -p ./target/criterion
    -cp -r {{tmpdir}}/* ./target/criterion
    rm -rf {{tmpdir}}

# Delete benchmark results.
clean-benches:
    -rm -rf ./target/criterion/*

# Clean all artifacts, including benchmark results.
clean-all:
    cargo clean

# === GIT ===

# Commit (add all first) both rsonpath and the benchmarks with a given message.
[no-exit-message]
commit msg:
    -cd ./crates/rsonpath-benchmarks && git add --all && git commit -m '{{msg}}' && git push
    -git add --all && git commit -am '{{msg}}'

# === HOOKS ===

[private]
hook-pre-commit:
    #!/bin/sh
    just assert-benchmarks-committed
    (just verify-fmt && just verify-check);

[private]
@hook-post-checkout: checkout-benchmarks

[private]
assert-benchmarks-committed:
    #!/bin/sh
    count=$(git diff HEAD ./crates/rsonpath-benchmarks | grep "^+Subproject commit [a-f0-9]*-dirty$" --count)
    if [ $count -ne 0 ]
    then
        echo "\033[31;1mCannot commit when rsonpath-benchmarks submodule is dirty, as this can lead to unexpected behaviour.
    First commit the changes in rsonpath-benchmarks by cd-ing into ./crates/rsonpath-benchmarks, or use just commit.\033[0"
        exit 1
    fi

# === RELEASE ===

# Execute prerequisits for a release for the given version.
release ver:
    cargo update
    just release-patch {{ver}}
    just release-readme
    just release-bug-template {{ver}}
    cargo build
    cargo +nightly fuzz build

[private]
release-patch ver:
    #!/usr/bin/env nu
    let ver = "{{ver}}";
    let crates = ["rsonpath", "rsonpath-lib", "rsonpath-benchmarks", "rsonpath-test", "rsonpath-test-codegen"];
    $crates | each { |cr|
        let path = $"./crates/($cr)/Cargo.toml";
        sed -i $'s/^version = "[^"]*"/version = "($ver)"/;s/^rsonpath-lib = { version = "[^"]*"/rsonpath-lib = { version = "($ver)"/;s/rsonpath-test-codegen = { version = "[^"]*"/rsonpath-test-codegen = { version = "($ver)"/' $path;
    };

[private]
release-readme:
    #!/usr/bin/env nu
    let rsonpath_deps = (cargo tree --package rsonpath --edges normal --edges build --depth 1 --target=all --all-features);
    let rsonpath_lib_deps = (cargo tree --package rsonpath-lib --edges normal --edges build --depth 1 --target=all --all-features);
    let rsonpath_full_deps = (cargo tree --package rsonpath --edges normal --edges build --target=all --all-features);
    let params = [
        [$rsonpath_deps, "rsonpath", "./README.md"],
        [$rsonpath_lib_deps, "rsonpath-lib", "./README.md"],
        [$rsonpath_lib_deps, "rsonpath-lib", "./crates/rsonpath-lib/README.md"],
        [$rsonpath_full_deps, "rsonpath-full", "./README.md"]
    ];
    $params | each {|x|
        let deps = ($x.0 | str replace "\n" '\n' --all | str replace '/' '\/' --all);
        sed -z -i $'s/<!-- ($x.1) dependencies start -->\n```ini\n.*```\n<!-- ($x.1) dependencies end -->/<!-- ($x.1) dependencies start -->\n```ini\n($deps)\n```\n<!-- ($x.1) dependencies end -->/' $x.2
    };

[private]
release-bug-template ver:
    #!/usr/bin/env nu
    let path = './.github/ISSUE_TEMPLATE/bug_report.yml';
    let idx = (cat $path | str index-of '# <newest-release=v{{ver}}>');
    if ($idx == -1) {
        sed -z -i 's/# <newest-release=v[^>]*>/# <newest-release=v{{ver}}>\n      - v{{ver}}/' $path;
    }