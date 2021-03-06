use clap::{ArgEnum, Parser};
use color_eyre::eyre::{eyre, Result, WrapErr};
use log::*;
use rsonpath::engine::result::{CountResult, IndexResult, QueryResult};
use rsonpath::engine::{Input, Runner};
use rsonpath::query::JsonPathQuery;
use rsonpath::stack_based::StackBasedRunner;
use rsonpath::stackless::StacklessRunner;
use simple_logger::SimpleLogger;

fn main() -> Result<()> {
    color_eyre::install()?;
    let args = Args::parse();

    configure_logger(args.verbose)?;

    let query = parse_query(&args.query)?;
    info!("Preparing query: `{}`\n", query);

    let contents = get_contents(args.file_path.as_deref())?;
    let input = Input::new(contents);

    match args.result {
        ResultArg::Bytes => run::<IndexResult>(&query, &input, args.engine),
        ResultArg::Count => run::<CountResult>(&query, &input, args.engine),
    }
}

fn run<R: QueryResult>(query: &JsonPathQuery, input: &Input, engine: EngineArg) -> Result<()> {
    match engine {
        EngineArg::Main => {
            let stackless_runner = StacklessRunner::compile_query(query);
            info!("Compilation finished, running...");

            let stackless_result = stackless_runner.run::<R>(input);
            info!("Stackless: {}", stackless_result);

            println!("{}", stackless_result);
        }
        EngineArg::Recursive => {
            let stack_based_runner = StackBasedRunner::compile_query(query);
            info!("Compilation finished, running...");

            let stack_based_result = stack_based_runner.run::<R>(input);
            info!("Stack based: {}", stack_based_result);

            println!("{}", stack_based_result);
        }
        EngineArg::VerifyBoth => {
            let stackless_runner = StacklessRunner::compile_query(query);
            let stack_based_runner = StackBasedRunner::compile_query(query);
            info!("Compilation finished, running...");

            let stackless_result = stackless_runner.run::<R>(input);
            info!("Stackless: {}", stackless_result);

            let stack_based_result = stack_based_runner.run::<R>(input);
            info!("Stack based: {}", stack_based_result);

            if stack_based_result != stackless_result {
                return Err(eyre!("Result mismatch!"));
            }

            println!("{}", stack_based_result);
        }
    }

    Ok(())
}

fn parse_query(query_string: &str) -> Result<JsonPathQuery> {
    JsonPathQuery::parse(query_string).wrap_err("Could not parse JSONPath query.")
}

#[derive(Parser, Debug)]
#[clap(author, version, about)]
struct Args {
    /// JSONPath query to run against the input JSON.
    query: String,
    /// Input JSON file to query.
    ///
    /// If not specified uses the standard input stream.
    file_path: Option<String>,
    /// Include verbose debug information.
    #[clap(short, long)]
    verbose: bool,
    /// Engine to use for evaluating the query.
    #[clap(short, long, arg_enum, default_value_t = EngineArg::Main)]
    engine: EngineArg,
    ///
    #[clap(short, long, arg_enum, default_value_t = ResultArg::Bytes)]
    result: ResultArg,
}

#[derive(ArgEnum, Debug, Clone, Copy, PartialEq, Eq)]
enum EngineArg {
    /// Main SIMD-optimized iterative engine.
    Main,
    /// Alternative recursive engine.
    Recursive,
    /// Use both engines and verify that their outputs match.
    ///
    /// This is for testing purposes only.
    VerifyBoth,
}

#[derive(ArgEnum, Debug, Clone, Copy, PartialEq, Eq)]
enum ResultArg {
    /// Return a list of all bytes at which a match occurred.
    Bytes,
    /// Return only the number of matches.
    Count,
}

fn configure_logger(verbose: bool) -> Result<()> {
    SimpleLogger::new()
        .with_level(if verbose {
            LevelFilter::Debug
        } else {
            LevelFilter::Warn
        })
        .init()
        .wrap_err("Logger configuration error.")
}

fn get_contents(file_path: Option<&str>) -> Result<String> {
    use std::fs;
    use std::io::{self, Read};
    match file_path {
        Some(path) => fs::read_to_string(path).wrap_err("Reading from file failed."),
        None => {
            let mut result = String::new();
            io::stdin()
                .read_to_string(&mut result)
                .wrap_err("Reading from stdin failed.")?;
            Ok(result)
        }
    }
}
