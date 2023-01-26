use rsonpath_lib::engine::main::MainEngine;
use rsonpath_lib::engine::recursive::RecursiveEngine;
use rsonpath_lib::engine::{
    result::{CountResult, IndexResult},
    Compiler, Engine, Input,
};
use rsonpath_lib::query::JsonPathQuery;
use std::fs;
use test_case::test_case;

const ROOT_TEST_DIRECTORY: &str = "./tests/data";

fn get_contents(test_path: &str) -> Input {
    let path = format!("{ROOT_TEST_DIRECTORY}/{test_path}");
    let mut raw = fs::read_to_string(path).unwrap();
    Input::new(&mut raw)
}

macro_rules! count_test_cases {
    ($test_name:ident, $impl:ident) => {
        #[test_case("basic/child.json", "$..a..b.c..d" => 3; "child.json $..a..b.c..d")]
        #[test_case("basic/child_hell.json", "$..x..a.b.a.b.c" => 6; "child_hell.json $..x..a.b.a.b.c")]
        #[test_case("basic/empty.json", "" => 0; "empty.json")]
        #[test_case("basic/empty.json", "$" => 0; "empty.json $")]
        #[test_case("basic/escapes.json", r#"$..a..b..['label\\']"# => 1; "escapes.json existing label")]
        #[test_case("basic/escapes.json", r#"$..a..b..['label\']"# => 0; "escapes.json nonexistent label")]
        #[test_case("basic/quote_escape.json", r#"$['x']"# => 1; "quote_escape.json without quote")]
        #[test_case("basic/quote_escape.json", r#"$['\"x']"# => 1; "quote_escape.json with quote")]
        #[test_case("basic/root.json", "$" => 1; "root.json $")]
        #[test_case("basic/root.json", "" => 1; "root.json")]
        #[test_case("basic/skipping.json", r#"$.a.b"# => 1; "skipping")]
        #[test_case("basic/small_no_list.json", "$..person..phoneNumber..number" => 2; "small_no_list.json $..person..phoneNumber..number")]
        #[test_case("basic/small.json", "$..person..phoneNumber..number" => 4; "small.json $..person..phoneNumber..number")]
        #[test_case("basic/spaced_colon.json", r#"$..a..b..label"# => 2; "spaced colon")]
        #[test_case("basic/wildcard_list.json", r#"$..a.*"# => 6; "wildcard_list.json $..a.*")]
        #[test_case("basic/wildcard_list2.json", r#"$..a.*..b.*"# => 8; "wildcard_list2.json $..a.*..b.*")]
        #[test_case("basic/wildcard_object.json", r#"$..a.*"# => 7; "wildcard_object.json $..a.*")]
        #[test_case("basic/wildcard_object2.json", r#"$..a.*.*..b.*.*"# => 9; "wildcard_object2.json $..a.*.*..b.*.*")]
        #[test_case("twitter/twitter.json", "$..user..entities..url" => 44; "twitter.json $..user..entities..url (recursive)")]
        #[test_case("twitter/twitter.json", "$..user..entities.url" => 18; "twitter.json $..user..entities.url (child)")]
        #[test_case("twitter/twitter.json", "$.search_metadata.count" => 1; "twitter.json $.search_metadata.count (child-child)")]
        #[test_case("twitter/twitter.json", "$..search_metadata.count" => 1; "twitter.json $..search_metadata.count (recursive-child)")]
        #[test_case("twitter/twitter.json", "$.search_metadata..count" => 1; "twitter.json $.search_metadata..count (child-recursive)")]
        #[test_case("twitter/twitter.json", "$..search_metadata..count" => 1; "twitter.json $..search_metadata..count (recursive-recursive)")]
        #[test_case("twitter/twitter.json", "$..count" => 1; "twitter.json $..count")]
        #[test_case("twitter/twitter_urls.json", "$..entities..urls..url" => 2; "twitter_urls.json $..entities..urls..url")]
        #[test_case("twitter/twitter_urls.json", "$..entities.urls..url" => 2; "twitter_urls.json $..entities.urls..url (child)")]
        #[test_case("basic/compressed/child.json", "$..a..b.c..d" => 3; "compressed child.json $..a..b.c..d")]
        #[test_case("basic/compressed/child_hell.json", "$..x..a.b.a.b.c" => 6; "compressed child_hell.json $..x..a.b.a.b.c")]
        #[test_case("basic/compressed/escapes.json", r#"$..a..b..['label\\']"# => 1; "compressed escapes.json existing label")]
        #[test_case("basic/compressed/escapes.json", r#"$..a..b..['label\']"# => 0; "compressed escapes.json nonexistent label")]
        #[test_case("basic/compressed/fake2.json", r#"$.a999999.b.c.d"# => 1; "compressed fake2.json")]
        #[test_case("basic/compressed/quote_escape.json", r#"$['x']"# => 1; "compressed quote_escape.json without quote")]
        #[test_case("basic/compressed/quote_escape.json", r#"$['\"x']"# => 1; "compressed quote_escape.json with quote")]
        #[test_case("basic/compressed/skipping.json", r#"$.a.b"# => 1; "compressed skipping")]
        #[test_case("basic/compressed/small_no_list.json", "$..person..phoneNumber..number" => 2; "compressed small_no_list.json $..person..phoneNumber..number")]
        #[test_case("basic/compressed/small.json", "$..person..phoneNumber..number" => 4; "compressed small.json $..person..phoneNumber..number")]
        #[test_case("twitter/compressed/twitter.json", "$..user..entities..url" => 44; "compressed twitter.json $..user..entities..url (recursive)")]
        #[test_case("twitter/compressed/twitter.json", "$..user..entities.url" => 18; "compressed twitter.json $..user..entities.url (child)")]
        #[test_case("twitter/compressed/twitter_urls.json", "$..entities..urls..url" => 2; "compressed twitter_urls.json $..entities..urls..url")]
        #[test_case("twitter/compressed/twitter_urls.json", "$..entities.urls..url" => 2; "compressed twitter_urls.json $..entities.urls..url (child)")]
        #[test_case(
            "wikidata/wikidata_person.json", "$..claims..references..hash" => 37736;
            "wikidata_person.json $..claims..references..hash"
        )]
        #[test_case(
            "wikidata/wikidata_person.json", "$..references..snaks..datavalue" => 25118;
            "wikidata_person.json $..references..snaks..datavalue"
        )]
        #[test_case(
            "wikidata/wikidata_person.json", "$..references..snaks..datavalue..value" => 25118;
            "wikidata_person.json $..references..snaks..datavalue..value"
        )]
        #[test_case(
            "wikidata/wikidata_person.json", "$..references..snaks..datavalue..value..id" => 11113;
            "wikidata_person.json $..references..snaks..datavalue..value..id"
        )]
        #[test_case(
            "wikidata/wikidata_person.json", "$..snaks..datavalue..value" => 25118;
            "wikidata_person.json $..snaks..datavalue..value"
        )]
        #[test_case(
            "wikidata/wikidata_person.json", "$..datavalue..value..id" => 25093;
            "wikidata_person.json $..datavalue..value..id"
        )]
        #[test_case(
            "wikidata/wikidata_person.json", "$..mainsnak..datavalue..value" => 26115;
            "wikidata_person.json $..mainsnak..datavalue..value"
        )]
        #[test_case(
            "wikidata/wikidata_person.json", "$..mainsnak..datavalue..value..id" => 12958;
            "wikidata_person.json $..mainsnak..datavalue..value..id"
        )]
        #[test_case(
            "wikidata/wikidata_person.json", "$..en..value" => 2360;
            "wikidata_person.json $..en..value (recursive)"
        )]
        #[test_case(
            "wikidata/wikidata_person.json", "$..en.value" => 1753;
            "wikidata_person.json $..en.value (child)"
        )]
        #[test_case(
            "wikidata/wikidata_profession.json", "$..claims..mainsnak..value" => 59112;
            "wikidata_profession.json $..claims..mainsnak..value"
        )]
        #[test_case(
            "wikidata/wikidata_profession.json", "$..en..value" => 13634;
            "wikidata_profession.json $..en..value (recursive)"
        )]
        #[test_case(
            "wikidata/wikidata_profession.json", "$..en.value" => 9452;
            "wikidata_profession.json $..en.value (child)"
        )]
        #[test_case(
            "wikidata/wikidata_properties.json", "$..qualifiers..datavalue..id" => 18219;
            "wikidata_properties.json $..qualifiers..datavalue..id"
        )]
        #[test_case(
            "wikidata/wikidata_properties.json", "$..en..value" => 4504;
            "wikidata_properties.json $..en..value (recursive)"
        )]
        #[test_case(
            "wikidata/wikidata_properties.json", "$..en.value" => 1760;
            "wikidata_properties.json $..en.value (child)"
        )]
        #[test_case(
            "wikidata/wikidata_properties.json", "$..P7103.claims.P31..references..snaks.P4656..hash" => 1;
            "wikidata_properties.json $..P7103.claims.P31..references..snaks.P4656..hash"
        )]
        fn $test_name(test_path: &str, query_string: &str) -> usize {
            let contents = get_contents(test_path);
            let query = JsonPathQuery::parse(query_string).unwrap();
            let result = $impl::compile_query(&query).unwrap().run::<CountResult>(&contents).unwrap();

            result.get()
        }
    };
}

macro_rules! indices_test_cases {
    ($test_name:ident, $impl:ident) => {
        #[test_case("basic/child.json", "$..a..b.c..d" => vec![984, 1297, 1545]; "child.json $..a..b.c..d")]
        #[test_case("basic/child_hell.json", "$..x..a.b.a.b.c" => vec![198, 756, 1227, 1903, 2040, 2207]; "child_hell.json $..x..a.b.a.b.c")]
        #[test_case("basic/empty.json", "" => Vec::<usize>::new(); "empty.json")]
        #[test_case("basic/empty.json", "$" => Vec::<usize>::new(); "empty.json $")]
        #[test_case("basic/escapes.json", r#"$..a..b..['label\\']"# => vec![609]; "escapes.json existing label")]
        #[test_case("basic/escapes.json", r#"$..a..b..['label\']"# => Vec::<usize>::new(); "escapes.json nonexistent label")]
        #[test_case("basic/quote_escape.json", r#"$['\"x']"# => vec![11]; "quote_escape.json with quote")]
        #[test_case("basic/quote_escape.json", r#"$['x']"# => vec![24]; "quote_escape.json without quote")]
        #[test_case("basic/root.json", "$" => vec![0]; "root.json $")]
        #[test_case("basic/root.json", "" => vec![0]; "root.json")]
        #[test_case("basic/skipping.json", r#"$.a.b"# => vec![808]; "skipping")]
        #[test_case("basic/small_no_list.json", "$..person..phoneNumber..number" => vec![310, 764]; "small_no_list.json $..person..phoneNumber..number")]
        #[test_case("basic/small.json", "$..person..phoneNumber..number" => vec![332, 436, 934, 1070]; "small.json $..person..phoneNumber..number")]
        #[test_case("basic/spaced_colon.json", r#"$..a..b..label"# => vec![106, 213]; "spaced colon")]
        #[test_case("basic/wildcard_list.json", r#"$..a.*"# => vec![46, 64, 83, 103, 123, 287]; "wildcard_list.json $..a.*")]
        #[test_case("basic/wildcard_list2.json", r#"$..a.*..b.*"# => vec![226, 364, 402, 441, 481, 521, 641, 881]; "wildcard_list2.json $..a.*..b.*")]
        #[test_case("basic/wildcard_object.json", r#"$..a.*"# => vec![66, 91, 116, 141, 211, 238, 267]; "wildcard_object.json $..a.*")]
        #[test_case("basic/wildcard_object2.json", r#"$..a.*.*..b.*.*"# => vec![652, 709, 749, 791, 855, 899, 1713, 1811, 1876]; "wildcard_object2.json $..a.*.*..b.*.*")]
        #[test_case(
            "twitter/twitter.json",
            "$..user..entities..url"
                => vec![5463, 5568, 9494, 9983, 18494, 18599, 23336, 23441, 24015, 89783, 89900, 112196, 112313, 134218, 134335, 134934, 201053, 201158, 205279, 205396, 206006, 333128, 333233, 352430, 352535, 353094, 356998, 357115, 399783, 399900, 451852, 475582, 475687, 511440, 511545, 516536, 516641, 728250, 728355, 743600, 743717, 762795, 762900, 763472];
            "twitter.json $..user..entities..url (recursive)")]
        #[test_case(
            "twitter/twitter.json",
            "$..user..entities.url"
                => vec![5463, 18494, 23336, 89783, 112196, 134218, 201053, 205279, 333128, 352430, 356998, 399783, 475582, 511440, 516536, 728250, 743600, 762795];
            "twitter.json $..user..entities.url (child)")]
        #[test_case("twitter/twitter_urls.json", "$..entities..urls..url" => vec![321, 881]; "twitter_urls.json $..entities..urls..url")]
        #[test_case("twitter/twitter_urls.json", "$..entities.urls..url" => vec![321, 881]; "twitter_urls.json $..entities.urls..url (child)")]
        #[test_case("basic/compressed/child.json", "$..a..b.c..d" => vec![99, 132, 152]; "compressed child.json $..a..b.c..d")]
        #[test_case("basic/compressed/child_hell.json", "$..x..a.b.a.b.c" => vec![39, 108, 189, 240, 263, 280]; "compressed child_hell.json $..x..a.b.a.b.c")]
        #[test_case("basic/compressed/escapes.json", r#"$..a..b..['label\\']"# => vec![524]; "compressed escapes.json existing label")]
        #[test_case("basic/compressed/escapes.json", r#"$..a..b..['label\']"# => Vec::<usize>::new(); "compressed escapes.json nonexistent label")]
        #[test_case("basic/compressed/quote_escape.json", r#"$['\"x']"# => vec![6]; "compressed quote_escape.json with quote")]
        #[test_case("basic/compressed/quote_escape.json", r#"$['x']"# => vec![13]; "compressed quote_escape.json without quote")]
        #[test_case("basic/compressed/skipping.json", r#"$.a.b"# => vec![452]; "compressed skipping")]
        #[test_case("basic/compressed/small_no_list.json", "$..person..phoneNumber..number" => vec![176, 380]; "compressed small_no_list.json $..person..phoneNumber..number")]
        #[test_case("basic/compressed/small.json", "$..person..phoneNumber..number" => vec![177, 219, 425, 467]; "compressed small.json $..person..phoneNumber..number")]
        #[test_case(
            "twitter/compressed/twitter.json",
            "$..user..entities..url"
                => vec![3487, 3503, 5802, 5954, 9835, 9851, 12717, 12733, 12912, 52573, 52589, 64602, 64618, 77996, 78012, 78164, 119306, 119322, 121917, 121933, 122096, 201072, 201088, 212697, 212713, 212877, 215342, 215358, 241825, 241841, 274277, 288268, 288284, 310029, 310045, 312971, 312987, 445430, 445446, 454459, 454475, 464575, 464591, 464768];
            "compressed twitter.json $..user..entities..url (recursive)")]
        #[test_case(
            "twitter/compressed/twitter.json",
            "$..user..entities.url"
                => vec![3487, 9835, 12717, 52573, 64602, 77996, 119306, 121917, 201072, 212697, 215342, 241825, 288268, 310029, 312971, 445430, 454459, 464575];
            "compressed twitter.json $..user..entities.url (child)")]
        #[test_case("twitter/compressed/twitter_urls.json", "$..entities..urls..url" => vec![145, 326]; "compressed twitter_urls.json $..entities..urls..url")]
        #[test_case("twitter/compressed/twitter_urls.json", "$..entities.urls..url" => vec![145, 326]; "compressed twitter_urls.json $..entities.urls..url (child)")]
        #[test_case(
            "wikidata/wikidata_properties.json", "$..P7103.claims.P31..references..snaks.P4656..hash" => vec![22639033];
            "wikidata_properties.json $..P7103.claims.P31..references..snaks.P4656..hash"
        )]
        fn $test_name(test_path: &str, query_string: &str) -> Vec<usize> {
            let contents = get_contents(test_path);
            let query = JsonPathQuery::parse(query_string).unwrap();
            let result = $impl::compile_query(&query).unwrap().run::<IndexResult>(&contents).unwrap();

            result.into()
        }
    };
}

count_test_cases!(rsonpath_count_main, MainEngine);
count_test_cases!(rsonpath_count_recursive, RecursiveEngine);
indices_test_cases!(rsonpath_indices_main, MainEngine);
indices_test_cases!(rsonpath_indices_recursive, RecursiveEngine);
