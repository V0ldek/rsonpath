use clap::{Parser, ValueEnum};
use color_eyre::eyre::{eyre, Result, WrapErr};
use color_eyre::{Help, SectionExt};
use log::*;
use rsonpath_lib::engine::result::{CountResult, IndexResult, QueryResult};
use rsonpath_lib::engine::{Input, Runner};
use rsonpath_lib::query::JsonPathQuery;
use rsonpath_lib::stack_based::StackBasedRunner;
use rsonpath_lib::stackless::StacklessRunner;
use simple_logger::SimpleLogger;

fn main() -> Result<()> {
    color_eyre::install()?;
    let args = Args::parse();

    configure_logger(args.verbose)?;

    let query = parse_query(&args.query)?;
    info!("Preparing query: `{}`\n", query);

    let mut contents = get_contents(args.file_path.as_deref())?;
    let input = Input::new(&mut contents);

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
    use rsonpath_lib::query::errors::QueryError;
    match JsonPathQuery::parse(query_string) {
        Ok(query) => Ok(query),
        Err(e) => {
            if let QueryError::ParseError { report } = e {
                let mut eyre = Err(eyre!("Could not parse JSONPath query."));
                eyre = eyre.note(format!("for query string {}", query_string));

                for error in report.errors() {
                    use color_eyre::owo_colors::OwoColorize;
                    use std::cmp;
                    const MAX_DISPLAY_LENGTH: usize = 80;

                    let display_start_idx = if error.start_idx > MAX_DISPLAY_LENGTH {
                        error.start_idx - MAX_DISPLAY_LENGTH
                    } else {
                        0
                    };
                    let display_length = cmp::min(
                        error.len + MAX_DISPLAY_LENGTH,
                        query_string.len() - display_start_idx,
                    );
                    let error_slice = &query_string[error.start_idx..error.start_idx + error.len];
                    let slice =
                        &query_string[display_start_idx..display_start_idx + display_length];
                    let error_idx = error.start_idx - display_start_idx;

                    let underline: String = std::iter::repeat(' ')
                        .take(error_idx)
                        .chain(std::iter::repeat('^').take(error.len))
                        .collect();
                    let display_string = format!(
                        "{}\n{}",
                        slice,
                        (underline + " invalid tokens").bright_red()
                    );

                    eyre = eyre.section(display_string.header("Parse error:"));

                    if error.start_idx == 0 {
                        eyre = eyre.suggestion("Queries should start with the root selector `$`.");
                    }

                    if error_slice.contains('$') {
                        eyre = eyre.suggestion("The `$` character is reserved for the root selector and may appear only at the start.");
                    }
                }

                eyre
            } else {
                Err(e).wrap_err("Could not parse JSONPath query.")
            }
        }
    }
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
    #[clap(short, long, value_enum, default_value_t = EngineArg::Main)]
    engine: EngineArg,
    ///
    #[clap(short, long, value_enum, default_value_t = ResultArg::Bytes)]
    result: ResultArg,
}

#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
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

#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
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
