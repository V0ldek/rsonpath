use rsonpath_syntax::builder::JsonPathQueryBuilder;
use std::{error::Error, process::ExitCode};

fn main() -> Result<ExitCode, Box<dyn Error>> {
    // === Build each example from the JSONPath spec. ===

    // $.store.book[*].author
    let query = JsonPathQueryBuilder::new()
        .child_name("store")
        .child_name("book")
        .child_wildcard()
        .child_name("author")
        .to_query();
    println!("{query}");

    // $..author
    let query = JsonPathQueryBuilder::new().descendant_name("author").to_query();
    println!("{query}");

    // $.store.*
    let query = JsonPathQueryBuilder::new()
        .child_name("store")
        .child_wildcard()
        .to_query();
    println!("{query}");

    // $.store..price
    let query = JsonPathQueryBuilder::new()
        .child_name("store")
        .descendant_name("price")
        .to_query();
    println!("{query}");

    // $..book[2]
    let query = JsonPathQueryBuilder::new()
        .descendant_name("book")
        .child_index(2)
        .to_query();
    println!("{query}");

    // $..book[2].author
    let query = JsonPathQueryBuilder::new()
        .descendant_name("book")
        .child_index(2)
        .child_name("author")
        .to_query();
    println!("{query}");

    // $..book[2].publisher
    let query = JsonPathQueryBuilder::new()
        .descendant_name("book")
        .child_index(2)
        .child_name("publisher")
        .to_query();
    println!("{query}");

    // $..book[-1]
    let query = JsonPathQueryBuilder::new()
        .descendant_name("book")
        .child_index(-1)
        .to_query();
    println!("{query}");

    // $..book[0, 1]
    let query = JsonPathQueryBuilder::new()
        .descendant_name("book")
        .child(|x| x.index(0).index(1))
        .to_query();
    println!("{query}");

    // $..book[:2]

    // $..book[?@.isbn]

    // $..book[?@.price<10]

    // $..*
    let query = JsonPathQueryBuilder::new().descendant_wildcard().to_query();
    println!("{query}");

    // === Build a query showcasing all syntactic elements. ===
    // $.['store\t1']
    //  ..[3, -5, ::3, ::-5, :7:2, 3::2, 3:7:, 3:7:2, -3:-7:-2]
    //  .*
    //  .

    println!("{query}");

    Ok(ExitCode::SUCCESS)
}
