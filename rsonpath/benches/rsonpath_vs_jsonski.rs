use core::time::Duration;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use decimal_byte_measurement::DecimalByteMeasurement;
use rsonpath::engine::result::CountResult;
use rsonpath::engine::{Input, Runner};
use rsonpath::query::JsonPathQuery;
use rsonpath::stackless::StacklessRunner;
use std::fs;

const ROOT_TEST_DIRECTORY: &str = "./data";

type CriterionCtx = Criterion<DecimalByteMeasurement>;

struct BenchmarkOptions<'a> {
    pub path: &'a str,
    pub query_string: &'a str,
    pub jsonski_query_string: &'a str,
    pub id: &'a str,
    pub warm_up_time: Duration,
    pub measurement_time: Duration,
}

fn get_contents(test_path: &str) -> Input {
    let path = format!("{}/{}", ROOT_TEST_DIRECTORY, test_path);
    let raw = fs::read_to_string(path).unwrap();
    Input::new(raw)
}

fn get_jsonski_record(test_path: &str) -> rust_jsonski::JsonSkiRecord {
    let path = format!("{}/{}", ROOT_TEST_DIRECTORY, test_path);
    rust_jsonski::load_jsonski_record(&path)
}

fn rsonpath_stack_based_vs_stackless(c: &mut CriterionCtx, options: BenchmarkOptions<'_>) {
    let contents = get_contents(options.path);
    let jsonski_record = get_jsonski_record(options.path);
    let query = JsonPathQuery::parse(options.query_string).unwrap();

    let mut group = c.benchmark_group(format! {"rsonpath_vs_jsonski_{}", options.id});
    group.warm_up_time(options.warm_up_time);
    group.measurement_time(options.measurement_time);
    group.throughput(criterion::Throughput::Bytes(contents.len() as u64));

    let rsonpath = StacklessRunner::compile_query(&query);

    group.bench_with_input(
        BenchmarkId::new("rsonpath", options.id),
        &contents,
        |b, c| b.iter(|| rsonpath.run::<CountResult>(c)),
    );
    group.bench_with_input(
        BenchmarkId::new("jsonski", options.id),
        &(&jsonski_record, options.jsonski_query_string),
        |b, &(r, q)| b.iter(|| rust_jsonski::call_jsonski(q, *r)),
    );

    group.finish();
}

fn decimal_byte_measurement() -> CriterionCtx {
    Criterion::default().with_measurement(DecimalByteMeasurement::new())
}

pub fn wikidata_combined(c: &mut CriterionCtx) {
    rsonpath_stack_based_vs_stackless(
        c,
        BenchmarkOptions {
            path: "wikidata_compressed/wikidata_combined.json",
            query_string: "$.key..P7103.claims.P31..references..snaks.P4656..hash",
            jsonski_query_string: "$.key[*].P7103.claims.P31[*].references[*].snaks.P4656[*].hash",
            id: "wikidata_combined",
            warm_up_time: Duration::from_secs(10),
            measurement_time: Duration::from_secs(40),
        },
    );
}

pub fn wikidata_combined_with_whitespace(c: &mut CriterionCtx) {
    rsonpath_stack_based_vs_stackless(
        c,
        BenchmarkOptions {
            path: "wikidata_prettified/wikidata_combined.json",
            query_string: "$.key..P7103.claims.P31..references..snaks.P4656..hash",
            jsonski_query_string: "$.key[*].P7103.claims.P31[*].references[*].snaks.P4656[*].hash",
            id: "wikidata_combined_with_whitespace",
            warm_up_time: Duration::from_secs(10),
            measurement_time: Duration::from_secs(40),
        },
    );
}

criterion_group!(
    name = jsonski_benches;
    config = decimal_byte_measurement();
    targets =
        wikidata_combined,
        wikidata_combined_with_whitespace
);

criterion_main!(jsonski_benches);
