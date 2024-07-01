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
use crate::{
    input::{error::InputError, InputBlock, InputBlockIterator},
    FallibleIterator, MaskType, BLOCK_SIZE,
};

/// Result of the [`FallibleIterator`] for quote classification,
/// and of the [`offset`](`QuoteClassifiedIterator::offset`) function.
pub type QuoteIterResult<I, M, const N: usize> = Result<Option<QuoteClassifiedBlock<I, M, N>>, InputError>;

/// Input block with a bitmask signifying which characters are within quotes.
///
/// Characters within quotes in the input are guaranteed to have their corresponding
/// bit in `within_quotes_mask` set. The $0$-th bit of the mask corresponds to the
/// last character in `block`, the $1$-st bit to the second-to-last character, etc.
///
/// There is no guarantee on how the boundary quote characters are classified,
/// their bits might be lit or not lit depending on the implementation.
pub struct QuoteClassifiedBlock<B, M, const N: usize> {
    /// The block that was classified.
    pub block: B,
    /// Mask marking characters within a quoted sequence.
    pub within_quotes_mask: M,
}

/// Result of resuming quote classification, the resulting iterator
/// and optionally the first block (already quote classified).
pub struct ResumedQuoteClassifier<Q, B, M, const N: usize> {
    /// Resumed iterator.
    pub classifier: Q,
    /// Optional first quote classified block.
    pub first_block: Option<QuoteClassifiedBlock<B, M, N>>,
}

/// Trait for quote classifier iterators, i.e. finite iterators
/// enriching blocks of input with quote bitmasks.
/// Iterator is allowed to hold a reference to the JSON document valid for `'a`.
pub trait QuoteClassifiedIterator<'i, I: InputBlockIterator<'i, N>, M, const N: usize>:
    FallibleIterator<Item = QuoteClassifiedBlock<I::Block, M, N>, Error = InputError>
{
    /// Get the total offset in bytes from the beginning of input.
    fn get_offset(&self) -> usize;

    /// Move the iterator `count` blocks forward.
    ///
    /// # Errors
    /// At least one new block is read from the underlying
    /// [`InputBlockIterator`] implementation, which can fail.
    fn offset(&mut self, count: isize) -> QuoteIterResult<I::Block, M, N>;

    /// Flip the bit representing whether the last block ended with a nonescaped quote.
    ///
    /// This should be done only in very specific circumstances where the previous-block
    /// state could have been damaged due to stopping and resuming the classification at a later point.
    fn flip_quotes_bit(&mut self);
}

/// Higher-level classifier that can be consumed to retrieve the inner
/// [`Input::BlockIterator`](crate::input::Input::BlockIterator).
pub trait InnerIter<I> {
    /// Consume `self` and return the wrapped [`Input::BlockIterator`](crate::input::Input::BlockIterator).
    fn into_inner(self) -> I;
}

impl<'i, B, M, const N: usize> QuoteClassifiedBlock<B, M, N>
where
    B: InputBlock<'i, N>,
{
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

pub(crate) mod nosimd;
pub(crate) mod shared;

#[cfg(target_arch = "x86_64")]
pub(crate) mod avx512_64;
#[cfg(target_arch = "x86")]
pub(crate) mod avx2_32;
#[cfg(target_arch = "x86_64")]
pub(crate) mod avx2_64;
#[cfg(target_arch = "x86")]
pub(crate) mod sse2_32;
#[cfg(target_arch = "x86_64")]
pub(crate) mod sse2_64;

pub(crate) trait QuotesImpl {
    type Classifier<'i, I>: QuoteClassifiedIterator<'i, I, MaskType, BLOCK_SIZE> + InnerIter<I>
    where
        I: InputBlockIterator<'i, BLOCK_SIZE>;

    fn new<'i, I>(iter: I) -> Self::Classifier<'i, I>
    where
        I: InputBlockIterator<'i, BLOCK_SIZE>;

    fn resume<'i, I>(
        iter: I,
        first_block: Option<I::Block>,
    ) -> ResumedQuoteClassifier<Self::Classifier<'i, I>, I::Block, MaskType, BLOCK_SIZE>
    where
        I: InputBlockIterator<'i, BLOCK_SIZE>;
}
