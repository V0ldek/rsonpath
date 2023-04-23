//! Classification of bytes withing JSON quote sequences.
//!
//! Provides the [`QuoteClassifiedBlock`] struct and [`QuoteClassifiedIterator`] trait
//! that allow effectively enriching JSON inputs with quote sequence information.
//!
//! The output of quote classification is an iterator of [`QuoteClassifiedBlock`]
//! which contain bitmasks whose lit bits signify characters that are within quotes
//! in the source document. These characters need to be ignored.
//!
//! Note that the actual quote characters are not guaranteed to be classified
//! as "within themselves" or otherwise. In particular the current implementation
//! marks _opening_ quotes with lit bits, but _closing_ quotes are always unmarked.
//! This behavior should not be presumed to be stable, though, and can change
//! without a major semver bump.
//!
//! # Examples
//! ```
//! use rsonpath_lib::classification::quotes::{classify_quoted_sequences, QuoteClassifiedIterator};
//! use aligners::AlignedBytes;
//!
//! let json = r#"{"x": "string", "y": {"z": "\"escaped\""}}"#;
//! //            011000111111100011000011000111111111111000
//! // The mask below appears reversed due to endianness.
//! let expd = 0b000111111111111000110000110001111111000110;
//! let aligned = AlignedBytes::new_padded(json.as_bytes());
//! let mut quote_classifier = classify_quoted_sequences(&aligned);
//!
//! let block = quote_classifier.next().unwrap();
//! assert_eq!(expd, block.within_quotes_mask);
//! ```
use std::marker::PhantomData;

use crate::classification::BLOCK_SIZE;
use crate::input::{IBlock, Input, InputBlock};
use cfg_if::cfg_if;

/// Input block with a bitmask signifying which characters are within quotes.
///
/// Characters within quotes in the input are guaranteed to have their corresponding
/// bit in `within_quotes_mask` set. The $0$-th bit of the mask corresponds to the
/// last character in `block`, the $1$-st bit to the second-to-last character, etc.
///
/// There is no guarantee on how the boundary quote characters are classified,
/// their bits might be lit or not lit depending on the implementation.
pub struct QuoteClassifiedBlock<'a, B: InputBlock<'a, N>, const N: usize> {
    /// The block that was classified.
    pub block: B,
    /// Mask marking characters within a quoted sequence.
    pub within_quotes_mask: u64,
    pub phantom: PhantomData<&'a B>,
}

impl<'a, B: InputBlock<'a, N>, const N: usize> QuoteClassifiedBlock<'a, B, N> {
    /// Returns the length of the classified block.
    #[must_use]
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.block.len()
    }

    /// Whether the classified block is empty.
    #[must_use]
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.block.is_empty()
    }
}

/// Trait for quote classifier iterators, i.e. finite iterators
/// enriching blocks of input with quote bitmasks.
/// Iterator is allowed to hold a reference to the JSON document valid for `'a`.
pub trait QuoteClassifiedIterator<'a, I: Input + 'a, const N: usize>:
    Iterator<Item = QuoteClassifiedBlock<'a, IBlock<'a, I, N>, N>> + 'a
{
    /// Get the total offset in bytes from the beginning of input.
    fn get_offset(&self) -> usize;

    /// Move the iterator `count` blocks forward.
    /// Effectively skips `count * Twice<BlockAlignment>::size()` bytes.
    fn offset(&mut self, count: isize);

    /// Flip the bit representing whether the last block ended with a nonescaped quote.
    ///
    /// This should be done only in very specific circumstances where the previous-block
    /// state could have been damaged due to stopping and resuming the classification at a later point.
    fn flip_quotes_bit(&mut self);
}
cfg_if! {
    if #[cfg(any(doc, not(feature = "simd")))] {
        mod nosimd;
        use nosimd::SequentialQuoteClassifier;

        /// Walk through the JSON document represented by `bytes`
        /// and classify quoted sequences.
        #[must_use]
        #[inline(always)]
        pub fn classify_quoted_sequences<I: Input>(
            bytes: &I,
        ) -> impl QuoteClassifiedIterator<I, BLOCK_SIZE> {
            SequentialQuoteClassifier::new(bytes)
        }
    }
    else if #[cfg(simd = "avx2")] {
        mod avx2;
        use avx2::Avx2QuoteClassifier;

        /// Walk through the JSON document represented by `bytes`
        /// and classify quoted sequences.
        #[must_use]
        #[inline(always)]
        pub fn classify_quoted_sequences<I: Input>(
            bytes: &I,
        ) -> impl QuoteClassifiedIterator<I, BLOCK_SIZE> {
            Avx2QuoteClassifier::new(bytes)
        }
    }
    else {
        compile_error!("Target architecture is not supported by SIMD features of this crate. Disable the default `simd` feature.");
    }
}
