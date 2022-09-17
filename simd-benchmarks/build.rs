use proc_macro2::TokenStream;
use std::env;
use std::error::Error;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::process::Command;

struct Codegen<'a> {
    pub md5_digest: md5::Digest,
    pub files: Vec<CodegenFile<'a>>,
}

struct CodegenFile<'a> {
    pub source_token_stream: TokenStream,
    pub file_name: &'a str,
}

fn main() -> Result<(), Box<dyn Error>> {
    let codegen = Codegen {
        md5_digest: simd_codegen::calculate_codegen_md5(),
        files: vec![
            CodegenFile {
                source_token_stream: simd_codegen::sequences::sse2::get_sse2_source(),
                file_name: "bytes/sequences/sse2.rs",
            },
            CodegenFile {
                source_token_stream: simd_codegen::sequences::avx2::get_avx2_source(),
                file_name: "bytes/sequences/avx2.rs",
            },
            CodegenFile {
                source_token_stream: simd_codegen::sequences::nosimd::get_nosimd_source(),
                file_name: "bytes/sequences/nosimd.rs",
            },
        ],
    };

    generate(codegen)?;

    Ok(())
}

fn generate(codegen: Codegen) -> Result<(), Box<dyn Error>> {
    let comment = format!("// {:x}\n", codegen.md5_digest);
    eprintln!("Current MD5 digest: {:x}", codegen.md5_digest);

    for codegen_file in codegen.files {
        let out_dir_root = env::var_os("OUT_DIR").ok_or("OUT_DIR env variable not found")?;
        let dest_dir = Path::new(&out_dir_root).join("simd-codegen");
        let dest = Path::new(&dest_dir).join(codegen_file.file_name);

        eprintln!("Reading MD5 digest of '{}'", dest.display());

        if dest.exists() && comment == read_file_comment(&dest)? {
            eprintln!(
                "MD5 digest up to date, skipping '{}'",
                codegen_file.file_name
            );
            continue;
        }

        eprintln!(
            "MD5 digest outdated, generating '{}'",
            codegen_file.file_name
        );

        let source = format!("{}\n{}", comment, codegen_file.source_token_stream);

        fs::create_dir_all(dest.parent().unwrap())?;
        fs::write(&dest, source)?;

        let rustfmt_status = Command::new("rustfmt")
            .arg(dest.display().to_string())
            .status()?;

        assert!(
            rustfmt_status.success(),
            "'rustfmt {}' excited with code {}",
            dest.display(),
            rustfmt_status
        );
    }

    println!("cargo:rerun-if-changed=../simd-codegen");

    Ok(())
}

fn read_file_comment(path: &Path) -> Result<String, Box<dyn Error>> {
    let file = fs::File::open(path)?;
    let mut buffer = BufReader::new(file);
    let mut first_line = String::new();

    buffer.read_line(&mut first_line)?;

    Ok(first_line)
}
