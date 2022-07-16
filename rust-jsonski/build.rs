use std::{error::Error, fs};

fn main() -> Result<(), Box<dyn Error>> {
    let dir = fs::read_dir("jsonski")?;
    let cpps = dir
        .filter_map(Result::ok)
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "cpp"))
        .map(|e| e.path());

    cc::Build::new()
        .cpp(true)
        .files(cpps)
        .flag("-std=c++11")
        .flag("-mavx")
        .flag("-mavx2")
        .flag("-mpclmul")
        .flag("-lpthread")
        .flag("-mcmodel=medium")
        .flag("-static-libstdc++")
        .pic(false)
        .compile("jsonski");

    Ok(())
}
