//! Code generation for other modules.
//!
//! Used by the `build.rs` script to generate some of the modules that would be tedious
//! to write manually. Autogenerated modules:
//! - `simd_benchmarks::sequences` generated from [`sequences`]
//! - `simd_benchmarks::sse2` generated from [`sequences::sse2`]
//! - `simd_benchmarks::avx2` generated from [`sequences::avx2`]
//! - `simd_benchmarks::nosimd` generated from [`sequences::nosimd`]

#![warn(missing_docs)]
#![warn(rustdoc::missing_crate_level_docs)]

pub mod sequences;

/// Calculate the MD5 hash of the source of the crate.
///
/// All autogenerated files are prefixed with a comment of the hex representation
/// of MD5 digest of the codegen source to ensure it is generated from the up-to-date
/// version.
pub fn calculate_codegen_md5() -> md5::Digest {
    let sse2_mod = include_str!("sequences/sse2.rs");
    let avx2_mod = include_str!("sequences/avx2.rs");
    let cmp_and_tree_mod = include_str!("sequences/cmp_and_tree.rs");
    let nosimd_mod = include_str!("sequences/nosimd.rs");
    let sequences_mod = include_str!("sequences.rs");
    let lib = include_str!("lib.rs");

    let contents =
        format!("{lib}{avx2_mod}{cmp_and_tree_mod}{nosimd_mod}{sse2_mod}{sequences_mod}");
    md5::compute(contents)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsStr;
    use std::fs;
    use std::path::Path;
    use walkdir::WalkDir;

    #[test]
    fn calculate_codegen_md5_calculates_md5_of_all_source_files() {
        let current_dir = std::env::current_dir().unwrap();

        let mut sources: Vec<_> = WalkDir::new(current_dir)
            .into_iter()
            .filter_map(Result::ok)
            .filter(|e| !e.file_type().is_dir())
            .filter(|e| Path::new(e.file_name()).extension().and_then(OsStr::to_str) == Some("rs"))
            .collect();
        sources.sort_by(|a, b| a.path().cmp(b.path()));
        let mut combined_source = String::new();

        for entry in sources.iter() {
            let contents = fs::read_to_string(entry.path())
                .unwrap_or_else(|_| panic!("cannot read file {}", entry.path().display()));
            combined_source += &contents;
        }

        let digest = md5::compute(combined_source);
        let actual = calculate_codegen_md5();

        let all_source_paths: String = sources
            .iter()
            .map(|x| format!("{}\n", x.path().display()))
            .collect();

        assert_eq!(
            digest, actual,
            "expected to calculate based on: {}",
            all_source_paths
        );
    }
}
