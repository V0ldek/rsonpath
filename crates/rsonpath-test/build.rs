use eyre::WrapErr;
use std::{
    fs,
    io::{self, BufRead, BufReader, ErrorKind},
    path::Path,
    process::Command,
};

const TOML_DIRECTORY_PATH: &str = "../rsonpath-lib/tests/documents/toml";
const JSON_DIRECTORY_PATH: &str = "../rsonpath-lib/tests/documents/json";
const OUTPUT_FILE_PATH: &str = "../rsonpath-lib/tests/end_to_end.rs";
const CONTROL_ENV_VAR: &str = "RSONPATH_ENABLE_TEST_CODEGEN";

fn main() -> eyre::Result<()> {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=../rsonpath-lib/tests/documents/toml/*");
    println!("cargo:rerun-if-changed=../rsonpath-lib/tests/documents/json/large/*");
    println!("cargo:rerun-if-changed=../rsonpath-lib/tests/end_to_end.rs");
    println!("cargo:rerun-if-changed=../rsonpath-test-codegen/**/*");
    println!("cargo:rerun-if-env-changed={CONTROL_ENV_VAR}");

    if std::env::var_os(CONTROL_ENV_VAR).is_some_and(|x| x == "1") {
        generate()
    } else {
        Ok(())
    }
}

pub(crate) fn generate() -> eyre::Result<()> {
    let tokens = rsonpath_test_codegen::generate_tests(TOML_DIRECTORY_PATH, JSON_DIRECTORY_PATH)
        .wrap_err("error generating end-to-end tests")?;
    // Format and normalize line endings, so that MD5 sums agree between platforms.
    let source = format!("{}", tokens).replace("\r\n", "\n");

    // We store the MD5 checksum of the file in a comment on the first line to avoid needless regeneration.
    // The write of the source itself is not really costly, but the latter rustfmt run is.
    // It also helps with CI, since we don't have to install the rustfmt component in each job.
    let new_md5 = md5::compute(&source);
    let old_comment = read_md5_from_comment(OUTPUT_FILE_PATH)?;
    let new_comment = format!("// {:x}\n", new_md5);

    if old_comment.is_some_and(|x| x == new_comment) {
        eprintln!("MD5 digest up to date, not regenerating");
        return Ok(());
    }
    let contents = new_comment + &source;
    fs::write(OUTPUT_FILE_PATH, contents).wrap_err("error writing to test file")?;

    // By default the output is a single line of tokens, which is completely unreadable.
    // Note that the MD5 hash is computed beforehand on the raw token stream, so it stays up-to-date regardless
    // of rustfmt or any updates to it.
    let rustfmt_status = Command::new("rustfmt").arg(OUTPUT_FILE_PATH).status()?;

    assert!(
        rustfmt_status.success(),
        "'rustfmt {}' excited with code {}",
        OUTPUT_FILE_PATH,
        rustfmt_status
    );

    Ok(())
}

fn read_md5_from_comment<P: AsRef<Path>>(path: P) -> Result<Option<String>, io::Error> {
    match fs::File::open(&path) {
        Ok(f) => {
            let mut buffer = BufReader::new(f);
            let mut first_line = String::new();

            buffer.read_line(&mut first_line)?;

            Ok(Some(first_line))
        }
        Err(err) if err.kind() == ErrorKind::NotFound => Ok(None),
        Err(err) => Err(err),
    }
}
