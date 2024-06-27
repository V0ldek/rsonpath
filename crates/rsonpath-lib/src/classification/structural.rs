//! Classification of structurally significant JSON bytes.
//!
//! Provides the [`Structural`] struct and [`StructuralIterator`] trait
//! that allow effectively iterating over structural characters in a JSON document.
//!
//! Classifying [`Commas`](`Structural::Comma`) and [`Colons`](`Structural::Colon`) is disabled by default.
//! It can be enabled on demand by calling
//! [`StructuralIterator::turn_commas_on`]/[`StructuralIterator::turn_colons_on`].
//! This configuration is persisted across [`stop`](StructuralIterator::stop) and
//! [`resume`](StructuralIterator::resume) calls.
//!
//! A structural classifier needs ownership over a base
//! [`QuoteClassifiedIterator`](`crate::classification::quotes::QuoteClassifiedIterator`).
use crate::{
    classification::{quotes::QuoteClassifiedIterator, ResumeClassifierState},
    input::{error::InputError, InputBlockIterator},
    FallibleIterator, MaskType, BLOCK_SIZE,
};

/// Defines the kinds of brackets that can be identified as structural.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
#[repr(u8)]
pub enum BracketType {
    /// Square brackets, '[' and ']'.
    Square,
    /// Curly braces, '{' and '}'.
    Curly,
}

/// Defines structural characters in JSON documents.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub enum Structural {
    /// Represents the closing square or curly brace, ']' or '}'.
    Closing(BracketType, usize),
    /// Represents the colon ':' character.
    Colon(usize),
    /// Represents the opening square or curly brace, '[' or '{'.
    Opening(BracketType, usize),
    /// Represents the comma ',' character.
    Comma(usize),
}
use Structural::*;

impl Structural {
    /// Returns the index of the character in the document,
    /// i.e. which byte it is counting from 0.
    #[inline(always)]
    #[must_use]
    pub fn idx(self) -> usize {
        match self {
            Closing(_, idx) | Colon(idx) | Opening(_, idx) | Comma(idx) => idx,
        }
    }

    /// Add a given amount to the structural's index.
    ///
    /// # Examples
    /// ```rust
    /// # use rsonpath::classification::structural::Structural;
    ///
    /// let structural = Structural::Colon(42);
    /// let offset_structural = structural.offset(10);
    ///
    /// assert_eq!(structural.idx(), 42);
    /// assert_eq!(offset_structural.idx(), 52);
    /// ```
    #[inline(always)]
    #[must_use]
    pub fn offset(self, amount: usize) -> Self {
        match self {
            Closing(b, idx) => Closing(b, idx + amount),
            Colon(idx) => Colon(idx + amount),
            Opening(b, idx) => Opening(b, idx + amount),
            Comma(idx) => Comma(idx + amount),
        }
    }

    /// Check if the structural represents a closing character,
    /// i.e. a [`Closing`] with either of the [`BracketType`] variants.
    ///
    /// # Examples
    /// ```rust
    /// # use rsonpath::classification::structural::{BracketType, Structural};
    ///
    /// let brace = Structural::Closing(BracketType::Curly, 42);
    /// let bracket = Structural::Closing(BracketType::Square, 43);
    /// let neither = Structural::Comma(44);
    ///
    /// assert!(brace.is_closing());
    /// assert!(bracket.is_closing());
    /// assert!(!neither.is_closing());
    /// ```
    #[inline(always)]
    #[must_use]
    pub fn is_closing(&self) -> bool {
        matches!(self, Closing(_, _))
    }

    /// Check if the structural represents an opening character,
    /// i.e. an [`Opening`] with either of the [`BracketType`] variants.
    ///
    /// # Examples
    /// ```rust
    /// # use rsonpath::classification::structural::{BracketType, Structural};
    ///
    /// let brace = Structural::Opening(BracketType::Curly, 42);
    /// let bracket = Structural::Opening(BracketType::Square, 43);
    /// let neither = Structural::Comma(44);
    ///
    /// assert!(brace.is_opening());
    /// assert!(bracket.is_opening());
    /// assert!(!neither.is_opening());
    /// ```
    #[inline(always)]
    #[must_use]
    pub fn is_opening(&self) -> bool {
        matches!(self, Opening(_, _))
    }
}

/// Trait for classifier iterators, i.e. finite iterators of [`Structural`] characters
/// that hold a reference to the JSON document valid for `'a`.
pub trait StructuralIterator<'i, I, Q, M, const N: usize>:
    FallibleIterator<Item = Structural, Error = InputError>
where
    I: InputBlockIterator<'i, N>,
{
    /// Stop classification and return a state object that can be used to resume
    /// a classifier from the place in which the current one was stopped.
    fn stop(self) -> ResumeClassifierState<'i, I, Q, M, N>;

    /// Resume classification from a state retrieved by stopping a classifier.
    fn resume(state: ResumeClassifierState<'i, I, Q, M, N>) -> Self;

    /// Turn classification of [`Structural::Colon`] characters off.
    fn turn_colons_off(&mut self);

    /// Turn classification of [`Structural::Colon`] characters on.
    ///
    /// The `idx` passed should be the index of the byte in the input
    /// from which commas are to be classified. Passing an `idx` that
    /// does not match the index which the internal [`QuoteClassifiedIterator`]
    /// reached may result in incorrect results.
    fn turn_colons_on(&mut self, idx: usize);

    /// Turn classification of [`Structural::Comma`] characters off.
    fn turn_commas_off(&mut self);

    /// Turn classification of [`Structural::Comma`] characters on.
    ///
    /// The `idx` passed should be the index of the byte in the input
    /// from which commas are to be classified. Passing an `idx` that
    /// does not match the index which the internal [`QuoteClassifiedIterator`]
    /// reached may result in incorrect results.
    fn turn_commas_on(&mut self, idx: usize);

    /// Turn classification of both [`Structural::Comma`] and [`Structural::Colon`]
    /// characters on. This is generally faster than calling
    /// [`turn_colons_on`](`StructuralIterator::turn_colons_on`) and
    /// [`turn_commas_on`](`StructuralIterator::turn_commas_on`)
    /// in sequence.
    fn turn_colons_and_commas_on(&mut self, idx: usize);

    /// Turn classification of both [`Structural::Comma`] and [`Structural::Colon`]
    /// characters off. This is generally faster than calling
    /// [`turn_colons_on`](`StructuralIterator::turn_colons_off`) and
    /// [`turn_commas_on`](`StructuralIterator::turn_commas_off`)
    /// in sequence.
    fn turn_colons_and_commas_off(&mut self);
}

pub(crate) mod nosimd;
pub(crate) mod shared;

#[cfg(target_arch = "x86_64")]
pub(crate) mod avx512_64;
#[cfg(target_arch = "x86")]
pub(crate) mod avx2_32;
#[cfg(target_arch = "x86_64")]
pub(crate) mod avx2_64;
#[cfg(target_arch = "x86")]
pub(crate) mod ssse3_32;
#[cfg(target_arch = "x86_64")]
pub(crate) mod ssse3_64;

pub(crate) trait StructuralImpl {
    type Classifier<'i, I, Q>: StructuralIterator<'i, I, Q, MaskType, BLOCK_SIZE>
    where
        I: InputBlockIterator<'i, BLOCK_SIZE>,
        Q: QuoteClassifiedIterator<'i, I, MaskType, BLOCK_SIZE>;

    fn new<'i, I, Q>(iter: Q) -> Self::Classifier<'i, I, Q>
    where
        I: InputBlockIterator<'i, BLOCK_SIZE>,
        Q: QuoteClassifiedIterator<'i, I, MaskType, BLOCK_SIZE>;

    fn resume<'i, I, Q>(state: ResumeClassifierState<'i, I, Q, MaskType, BLOCK_SIZE>) -> Self::Classifier<'i, I, Q>
    where
        I: InputBlockIterator<'i, BLOCK_SIZE>,
        Q: QuoteClassifiedIterator<'i, I, MaskType, BLOCK_SIZE>,
    {
        <Self::Classifier<'i, I, Q> as StructuralIterator<'i, I, Q, MaskType, BLOCK_SIZE>>::resume(state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        classification::simd::{self, config_simd, Simd},
        input::{BorrowedBytes, Input},
        result::empty::EmptyRecorder,
    };

    #[test]
    fn resumption_without_commas_or_colons() {
        use BracketType::*;
        use Structural::*;

        let simd = simd::configure();
        config_simd!(simd => |simd| {
            let json = r#"{"a": [42, 36, { "b": { "c": 1, "d": 2 } }]}"#;
            let json_string = json.to_owned();
            let input = BorrowedBytes::new(json_string.as_bytes());
            let iter = input.iter_blocks(&EmptyRecorder);
            let quotes = simd.classify_quoted_sequences(iter);
            let offset = input.leading_padding_len();

            let mut classifier = simd.classify_structural_characters(quotes);

            assert_eq!(Some(Opening(Curly, offset)), classifier.next().unwrap());
            assert_eq!(Some(Opening(Square, 6 + offset)), classifier.next().unwrap());

            let resume_state = classifier.stop();

            let mut resumed_classifier = simd.resume_structural_classification(resume_state);

            assert_eq!(Some(Opening(Curly, 15 + offset)), resumed_classifier.next().unwrap());
            assert_eq!(Some(Opening(Curly, 22 + offset)), resumed_classifier.next().unwrap());
        });
    }

    #[test]
    fn resumption_with_commas_but_no_colons() {
        use BracketType::*;
        use Structural::*;

        let simd = simd::configure();
        config_simd!(simd => |simd| {
            let json = r#"{"a": [42, 36, { "b": { "c": 1, "d": 2 } }]}"#;
            let json_string = json.to_owned();
            let input = BorrowedBytes::new(json_string.as_bytes());
            let iter = input.iter_blocks(&EmptyRecorder);
            let quotes = simd.classify_quoted_sequences(iter);
            let offset = input.leading_padding_len();

            let mut classifier = simd.classify_structural_characters(quotes);
            classifier.turn_commas_on(0);

            assert_eq!(Some(Opening(Curly, offset)), classifier.next().unwrap());
            assert_eq!(Some(Opening(Square, 6 + offset)), classifier.next().unwrap());
            assert_eq!(Some(Comma(9 + offset)), classifier.next().unwrap());
            assert_eq!(Some(Comma(13 + offset)), classifier.next().unwrap());

            let resume_state = classifier.stop();

            let mut resumed_classifier = simd.resume_structural_classification(resume_state);

            assert_eq!(Some(Opening(Curly, 15 + offset)), resumed_classifier.next().unwrap());
            assert_eq!(Some(Opening(Curly, 22 + offset)), resumed_classifier.next().unwrap());
            assert_eq!(Some(Comma(30 + offset)), resumed_classifier.next().unwrap());
        });
    }

    #[test]
    fn resumption_with_colons_but_no_commas() {
        use BracketType::*;
        use Structural::*;

        let simd = simd::configure();
        config_simd!(simd => |simd| {
            let json = r#"{"a": [42, 36, { "b": { "c": 1, "d": 2 } }]}"#;
            let json_string = json.to_owned();
            let input = BorrowedBytes::new(json_string.as_bytes());
            let iter = input.iter_blocks(&EmptyRecorder);
            let quotes = simd.classify_quoted_sequences(iter);
            let offset = input.leading_padding_len();

            let mut classifier = simd.classify_structural_characters(quotes);
            classifier.turn_colons_on(0);

            assert_eq!(Some(Opening(Curly, offset)), classifier.next().unwrap());
            assert_eq!(Some(Colon(4 + offset)), classifier.next().unwrap());
            assert_eq!(Some(Opening(Square, 6 + offset)), classifier.next().unwrap());

            let resume_state = classifier.stop();

            let mut resumed_classifier = simd.resume_structural_classification(resume_state);

            assert_eq!(Some(Opening(Curly, 15 + offset)), resumed_classifier.next().unwrap());
            assert_eq!(Some(Colon(20 + offset)), resumed_classifier.next().unwrap());
            assert_eq!(Some(Opening(Curly, 22 + offset)), resumed_classifier.next().unwrap());
            assert_eq!(Some(Colon(27 + offset)), resumed_classifier.next().unwrap());
        });
    }

    #[test]
    fn resumption_with_commas_and_colons() {
        use BracketType::*;
        use Structural::*;

        let simd = simd::configure();
        config_simd!(simd => |simd| {
            let json = r#"{"a": [42, 36, { "b": { "c": 1, "d": 2 } }]}"#;
            let json_string = json.to_owned();
            let input = BorrowedBytes::new(json_string.as_bytes());
            let iter = input.iter_blocks(&EmptyRecorder);
            let quotes = simd.classify_quoted_sequences(iter);
            let offset = input.leading_padding_len();

            let mut classifier = simd.classify_structural_characters(quotes);
            classifier.turn_commas_on(0);
            classifier.turn_colons_on(0);

            assert_eq!(Some(Opening(Curly, offset)), classifier.next().unwrap());
            assert_eq!(Some(Colon(4 + offset)), classifier.next().unwrap());
            assert_eq!(Some(Opening(Square, 6 + offset)), classifier.next().unwrap());
            assert_eq!(Some(Comma(9 + offset)), classifier.next().unwrap());
            assert_eq!(Some(Comma(13 + offset)), classifier.next().unwrap());

            let resume_state = classifier.stop();

            let mut resumed_classifier = simd.resume_structural_classification(resume_state);

            assert_eq!(Some(Opening(Curly, 15 + offset)), resumed_classifier.next().unwrap());
            assert_eq!(Some(Colon(20 + offset)), resumed_classifier.next().unwrap());
            assert_eq!(Some(Opening(Curly, 22 + offset)), resumed_classifier.next().unwrap());
            assert_eq!(Some(Colon(27 + offset)), resumed_classifier.next().unwrap());
            assert_eq!(Some(Comma(30 + offset)), resumed_classifier.next().unwrap());
        });
    }

    #[test]
    fn resumption_at_block_boundary() {
        use BracketType::*;
        use Structural::*;

        let simd = simd::configure();
        config_simd!(simd => |simd| {
            let mut json_string = "{".to_owned();
            json_string += &" ".repeat(128);
            json_string += "}";
            let input = BorrowedBytes::new(json_string.as_bytes());
            let iter = input.iter_blocks(&EmptyRecorder);
            let quotes = simd.classify_quoted_sequences(iter);
            let offset = input.leading_padding_len();

            let mut classifier = simd.classify_structural_characters(quotes);

            assert_eq!(Some(Opening(Curly, offset)), classifier.next().unwrap());

            let resume_state = classifier.stop();
            let mut resumed_classifier = simd.resume_structural_classification(resume_state);

            assert_eq!(Some(Closing(Curly, 129 + offset)), resumed_classifier.next().unwrap());
        });
    }
}
