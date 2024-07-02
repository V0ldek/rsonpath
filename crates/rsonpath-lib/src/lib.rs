//! Blazing fast execution of JSONPath queries.
//!
//! JSONPath parser, execution engines and byte stream utilities useful when parsing
//! JSON structures.
//!
//! # Examples
//! ```rust
//! use rsonpath::engine::{Compiler, Engine, RsonpathEngine};
//! use rsonpath::input::BorrowedBytes;
//! use rsonpath::result::count::CountRecorder;
//! # use std::error::Error;
//!
//! # fn main() -> Result<(), Box<dyn Error>> {
//! // Parse a JSONPath query from string.
//! let query = rsonpath_syntax::parse("$..phoneNumbers[*].number")?;
//! // Convert the contents to the Input type required by the Engines.
//! let mut contents = r#"
//! {
//!   "person": {
//!     "name": "John",
//!     "surname": "Doe",
//!     "phoneNumbers": [
//!       {
//!         "type": "Home",
//!         "number": "111-222-333"
//!       },
//!       {
//!         "type": "Work",
//!         "number": "123-456-789"
//!       }
//!     ]
//!   }
//! }
//! "#;
//! let input = BorrowedBytes::new(contents.as_bytes());
//! // Compile the query. The engine can be reused to run the same query on different contents.
//! let engine = RsonpathEngine::compile_query(&query)?;
//! // Count the number of occurrences of elements satisfying the query.
//! let count = engine.count(&input)?;
//!
//! assert_eq!(2, count);
//! # Ok(())
//! # }
//! ```
//! # Input JSON assumptions
//!
//! The JSON must be a syntactically valid JSON encoded in UTF-8 as defined by
//! [RFC4627](https://datatracker.ietf.org/doc/html/rfc4627).
//!
//! If the assumptions are violated the algorithm's behavior is undefined. It might return nonsensical results,
//! not process the whole document, or stop with an error.
//! It should not panic &ndash; if you encounter a panic, you may report it as a bug.
//! Some simple mistakes are caught, for example missing closing brackets or braces, but robust validation is
//! sacrificed for performance. Asserting the assumptions falls on the user of this library.
//! If you need a high-throughput parser that validates the document, take a look at
//! [simdjson](https://lib.rs/crates/simd-json).
//!
//! # JSONPath language
//!
//! The library implements the JSONPath syntax as established by Stefan Goessner in
//! <https://goessner.net/articles/JsonPath/>.
//! That implementation does not describe its semantics. There is no guarantee that this library has the same semantics
//! as Goessner's implementation. The semantics used by rsonpath are described below.
//!
//! ## Grammar
//!
//! ```ebnf
//! query = [root] , { selector }
//! root = "$"
//! selector = child | descendant | wildcard child | wildcard descendant
//! wildcard child = ".*" | "[*]"
//! wildcard descendant = "..*" | "..[*]"
//! child = dot | index
//! dot = "." , member
//! descendant = ".." , ( member | index )
//! index = "[" , quoted member , "]"
//! member = member first , { member character }
//! member first = ALPHA | "_" | NONASCII
//! member character = ALPHANUMERIC | "_" | NONASCII
//! quoted member = ("'" , single quoted member , "'") | ('"' , double quoted member , '"')
//! single quoted member = { UNESCAPED | ESCAPED | '"' | "\'" }
//! double quoted member = { UNESCAPED | ESCAPED | "'" | '\"' }
//!
//! ALPHA = ? [A-Za-z] ?
//! ALPHANUMERIC = ? [A-Za-z0-9] ?
//! NONASCII = ? [\u0080-\u10FFFF] ?
//! UNESCAPED = ? [^'"\u0000-\u001F] ?
//! ESCAPED = ? \\[btnfr/\\] ?
//! ```
//!
//! ## Semantics
//!
//! The query is executed from left to right, selector by selector. When a value is found that matches
//! the current selector, the execution advances to the next selector and evaluates it recursively within
//! the context of that value.
//!
//! ### Root selector (`$`)
//! The root selector may only appear at the beginning of the query and is implicit if not specified.
//! It matches the root object or array. Thus the query "$" gives either 1 or 0 results, if the JSON
//! is empty or non-empty, respectively.
//!
//! ### Child selector (`.<member>`, `[<member>]`)
//! Matches any value under a specified key in the current object
//! and then executes the rest of the query on that value.
//!
//! ### Child wildcard selector (`.*`, `[*]`)
//! Matches any value regardless of key in the current object, or any value within the current array,
//! and then executes the rest of the query on that value.
//!
//! ### Descendant selector (`..<member>`, `..[<member>]`)
//! Switches the engine into a recursive descent mode.
//! Looks for the specified key in every value nested in the current object or array,
//! recursively, and then executes the rest of the query on that value..
//!
//! ### Descendant wildcard selector (`..*`, `..[*]`)
//! Switches the engine into a recursive descent mode.
//! Matches any value regardless of key in any object, or any value within any array nested
//! within the current object or array, recursively, and then executes the rest of the query on that value.
//!
//! ## Active development
//!
//! Only the aforementioned selectors are supported at this moment.
//! This library is under active development.

#![doc(html_logo_url = "https://raw.githubusercontent.com/V0ldek/rsonpath/main/img/rsonquery-logo.svg")]
// Documentation lints, enabled only on --release.
#![cfg_attr(
    not(debug_assertions),
    warn(missing_docs, clippy::missing_errors_doc, clippy::missing_panics_doc,)
)]
#![cfg_attr(not(debug_assertions), warn(rustdoc::missing_crate_level_docs))]
// Generic pedantic lints.
#![warn(
    explicit_outlives_requirements,
    semicolon_in_expressions_from_macros,
    unreachable_pub,
    unused_import_braces,
    unused_lifetimes
)]
// Clippy pedantic lints.
#![warn(
    clippy::allow_attributes_without_reason,
    clippy::cargo_common_metadata,
    clippy::cast_lossless,
    clippy::cloned_instead_of_copied,
    clippy::empty_drop,
    clippy::empty_line_after_outer_attr,
    clippy::equatable_if_let,
    clippy::expl_impl_clone_on_copy,
    clippy::explicit_deref_methods,
    clippy::explicit_into_iter_loop,
    clippy::explicit_iter_loop,
    clippy::fallible_impl_from,
    clippy::flat_map_option,
    clippy::if_then_some_else_none,
    clippy::inconsistent_struct_constructor,
    clippy::large_digit_groups,
    clippy::let_underscore_must_use,
    clippy::manual_ok_or,
    clippy::map_err_ignore,
    clippy::map_unwrap_or,
    clippy::match_same_arms,
    clippy::match_wildcard_for_single_variants,
    clippy::missing_inline_in_public_items,
    clippy::mod_module_files,
    clippy::must_use_candidate,
    clippy::needless_continue,
    clippy::needless_for_each,
    clippy::needless_pass_by_value,
    clippy::ptr_as_ptr,
    clippy::redundant_closure_for_method_calls,
    clippy::ref_binding_to_reference,
    clippy::ref_option_ref,
    clippy::rest_pat_in_fully_bound_structs,
    clippy::undocumented_unsafe_blocks,
    clippy::unneeded_field_pattern,
    clippy::unseparated_literal_suffix,
    clippy::unreadable_literal,
    clippy::unused_self,
    clippy::use_self
)]
// Panic-free lint.
#![warn(clippy::exit)]
// Panic-free lints (disabled for tests).
#![cfg_attr(not(test), warn(clippy::panic, clippy::panic_in_result_fn, clippy::unwrap_used))]
// IO hygiene, only on --release.
#![cfg_attr(
    not(debug_assertions),
    warn(clippy::print_stderr, clippy::print_stdout, clippy::todo)
)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![feature(stdarch_x86_avx512)]
#![feature(avx512_target_feature)]

pub mod automaton;
pub mod classification;
mod depth;
pub mod engine;
pub mod error;
pub mod input;
pub mod result;

cfg_if::cfg_if! {
    if #[cfg(target_pointer_width = "32")] {
        pub(crate) const BLOCK_SIZE: usize = 32;
        pub(crate) type MaskType = u32;
    }
    else if #[cfg(target_pointer_width = "64")] {
        pub(crate) const BLOCK_SIZE: usize = 64;
        pub(crate) type MaskType = u64;
    } 
}

/// Macro for debug logging. Evaluates to [`log::debug`], if debug assertions are enabled.
/// Otherwise it's an empty statement.
///
/// Use this instead of plain [`log::debug`], since this is automatically removed in
/// release mode and incurs no performance penalties.
#[cfg(debug_assertions)]
#[allow(unused_macros)]
macro_rules! debug {
    (target: $target:expr, $($arg:tt)+) => (log::debug!(target: $target, $($arg)+));
    ($($arg:tt)+) => (log::debug!($($arg)+))
}

#[allow(unused_macros)]
macro_rules! block {
    ($b:expr) => {
        crate::debug!(
            "{: >24}: {}",
            "block",
            std::str::from_utf8(
                &$b.iter()
                    .map(|x| if x.is_ascii_whitespace() { b' ' } else { *x })
                    .collect::<Vec<_>>()
            )
            .unwrap_or("[INVALID UTF8]")
        );
    };
}

/// Macro for debug logging. Evaluates to [`log::debug`], if debug assertions are enabled.
/// Otherwise it's an empty statement.
///
/// Use this instead of plain [`log::debug`], since this is automatically removed in
/// release mode and incurs no performance penalties.
#[cfg(not(debug_assertions))]
#[allow(unused_macros)]
macro_rules! debug {
    (target: $target:expr, $($arg:tt)+) => {};
    ($($arg:tt)+) => {};
}

/// Debug log the given u64 expression by its full 64-bit binary string representation.
#[allow(unused_macros)]
macro_rules! bin_u64 {
    ($name:expr, $e:expr) => {
        $crate::debug!(
            "{: >24}: {:064b} ({})",
            $name,
            {
                let mut res = 0_u64;
                for i in 0..64 {
                    let bit = (($e) & (1 << i)) >> i;
                    res |= bit << (63 - i);
                }
                res
            },
            $e
        );
    };
}

/// Debug log the given u32 expression by its full 32-bit binary string representation.
#[allow(unused_macros)]
macro_rules! bin_u32 {
    ($name:expr, $e:expr) => {
        $crate::debug!(
            "{: >24}: {:032b} ({})",
            $name,
            {
                let mut res = 0_u32;
                for i in 0..32 {
                    let bit = (($e) & (1 << i)) >> i;
                    res |= bit << (31 - i);
                }
                res
            },
            $e
        );
    };
}

#[allow(unused_imports)]
pub(crate) use bin_u32;
#[allow(unused_imports)]
pub(crate) use bin_u64;
#[allow(unused_imports)]
pub(crate) use block;
#[allow(unused_imports)]
pub(crate) use debug;

/// Variation of the [`Iterator`] trait where each read can fail.
pub trait FallibleIterator {
    /// Type of items returned by this iterator.
    type Item;
    /// Type of errors that can occur when reading from this iterator.
    type Error: std::error::Error;

    /// Advances the iterator and returns the next value.
    ///
    /// # Errors
    /// May fail depending on the implementation.
    fn next(&mut self) -> Result<Option<Self::Item>, Self::Error>;

    /// Transforms an iterator into a collection.
    ///
    /// # Errors
    /// This consumes the iterator and reads from it. If any read fails,
    /// the result is the first error encountered.
    #[inline]
    fn collect<B>(self) -> Result<B, Self::Error>
    where
        B: FromIterator<Self::Item>,
        Self: Sized,
    {
        let iter = FallibleIntoIter { src: self };
        iter.collect()
    }
}

struct FallibleIntoIter<F> {
    src: F,
}

impl<F: FallibleIterator> Iterator for FallibleIntoIter<F> {
    type Item = Result<F::Item, F::Error>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.src.next() {
            Ok(item) => item.map(Ok),
            Err(e) => Some(Err(e)),
        }
    }
}

pub(crate) const JSON_SPACE_BYTE: u8 = b' ';

pub(crate) const JSON_WHITESPACE_BYTES: [u8; 4] = [b' ', b'\t', b'\n', b'\r'];

#[inline(always)]
#[must_use]
pub(crate) fn is_json_whitespace(x: u8) -> bool {
    JSON_WHITESPACE_BYTES.contains(&x)
}
