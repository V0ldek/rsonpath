CARGO=cargo

make: rsonpath

rsonpath: check_cargo
	$(CARGO) build --bin rsonpath --release

.PHONY: bench check_cargo clean clean_benches doc install uninstall test

bench: rsonpath
	$(CARGO) bench --bench rsonpath_wikidata

# Check if cargo is present, if not, use rustup to setup.
check_cargo:
	$(CARGO) --version || (curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y)
	source ${HOME}/.cargo/env

# Handle the criterion reports directory separately to avoid losing previous benches.
clean:
	$(eval TMPDIR := $(shell mktemp -d -t criterion-reports-tmp-XXXXXXXX))
	-cp -r ./target/criterion/* $(TMPDIR)/
	$(CARGO) clean
	mkdir -p ./target/criterion
	-cp -r $(TMPDIR)/* ./target/criterion
	rm -rf $(TMPDIR)

clean_benches:
	rm -rf ./target/criterion/*

doc: rsonpath
	RUSTDOCFLAGS="--cfg docsrs" cargo +nightly doc --open

install: rsonpath
	$(CARGO) install --path ./rsonpath

test: rsonpath
	cargo install cargo-hack
	$(CARGO) rsontest --package rsonpath

uninstall:
	$(CARGO) uninstall rsonpath
