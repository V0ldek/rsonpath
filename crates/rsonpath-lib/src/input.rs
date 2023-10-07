//! Input structures that can be fed into an [`Engine`](crate::engine::Engine).
//!
//! The engine itself is generic in the [`Input`] trait declared here.
//! There are a couple of different built-in implementations, each
//! suitable for a different scenario. Consult the module-level
//! documentation of each type to determine which to use. Here's a quick
//! cheat-sheet:
//!
//! | Input scenario | Type to use    |
//! |:---------------|:---------------|
//! | file based     | [`MmapInput`]  |
//! | memory based | [`BorrowedBytes`] |
//! | [`Read`](std::io::Read) based | [`BufferedInput`] |
//!
pub mod borrowed;
pub mod buffered;
pub mod error;
pub mod mmap;
mod padding;
mod slice;
pub use borrowed::BorrowedBytes;
pub use buffered::BufferedInput;
pub use mmap::MmapInput;

use self::error::InputError;
use crate::{query::JsonString, result::InputRecorder, FallibleIterator};
use std::ops::Deref;

/// Make the struct repr(C) with alignment equal to [`MAX_BLOCK_SIZE`].
macro_rules! repr_align_block_size {
    ($it:item) => {
        #[repr(C, align(128))]
        $it
    };
}
pub(crate) use repr_align_block_size;

/// Global padding guarantee for all [`Input`] implementations.
/// Iterating over blocks of at most this size is guaranteed
/// to produce only full blocks.
///
/// # Remarks
/// This is set to `128` and unlikely to change.
/// Widest available SIMD is AVX512, which has 64-byte blocks.
/// The engine processes blocks in pairs, thus 128 is the highest possible request made to a block iterator.
/// For this value to change a new, wider SIMD implementation would have to appear.
pub const MAX_BLOCK_SIZE: usize = 128;

/// UTF-8 encoded bytes representing a JSON document that support
/// block-by-block iteration and basic seeking procedures.
pub trait Input: Sized {
    /// Type of the iterator used by [`iter_blocks`](Input::iter_blocks), parameterized
    /// by the lifetime of source input and the size of the block.
    type BlockIterator<'i, 'r, R, const N: usize>: InputBlockIterator<
        'i,
        N,
        Block = Self::Block<'i, N>,
        Error = Self::Error,
    >
    where
        Self: 'i,
        R: InputRecorder<Self::Block<'i, N>> + 'r;

    type Error: Into<InputError>;

    /// Type of the blocks returned by the `BlockIterator`.
    type Block<'i, const N: usize>: InputBlock<'i, N>
    where
        Self: 'i;

    /// Return the length of the entire input, if known.
    ///
    /// This is meant to be used merely as a hint.
    /// There are [`Input`] implementations that may not be able to know the entire
    /// length a priori, and they should return [`None`].
    #[inline(always)]
    #[must_use]
    fn len_hint(&self) -> Option<usize> {
        None
    }

    #[must_use]
    fn leading_padding_len(&self) -> usize;

    #[must_use]
    fn trailing_padding_len(&self) -> usize;

    /// Iterate over blocks of size `N` of the input.
    /// `N` has to be a power of two larger than 1.
    #[must_use]
    fn iter_blocks<'i, 'r, R, const N: usize>(&'i self, recorder: &'r R) -> Self::BlockIterator<'i, 'r, R, N>
    where
        R: InputRecorder<Self::Block<'i, N>>;

    /// Search for an occurrence of `needle` in the input,
    /// starting from `from` and looking back. Returns the index
    /// of the first occurrence or `None` if the `needle` was not found.
    #[must_use]
    fn seek_backward(&self, from: usize, needle: u8) -> Option<usize>;

    /// Search for an occurrence of any of the `needles` in the input,
    /// starting from `from` and looking forward. Returns the index
    /// of the first occurrence and the needle found, or `None` if none of the `needles` were not found.
    ///
    /// # Errors
    /// This function can read more data from the input if no relevant characters are found
    /// in the current buffer, which can fail.
    fn seek_forward<const N: usize>(&self, from: usize, needles: [u8; N]) -> Result<Option<(usize, u8)>, Self::Error>;

    /// Search for the first byte in the input that is not ASCII whitespace
    /// starting from `from`. Returns a pair: the index of first such byte,
    /// and the byte itself; or `None` if no non-whitespace characters
    /// were found.
    ///
    /// # Errors
    /// This function can read more data from the input if no relevant characters are found
    /// in the current buffer, which can fail.
    fn seek_non_whitespace_forward(&self, from: usize) -> Result<Option<(usize, u8)>, Self::Error>;

    /// Search for the first byte in the input that is not ASCII whitespace
    /// starting from `from` and looking back. Returns a pair:
    /// the index of first such byte, and the byte itself;
    /// or `None` if no non-whitespace characters were found.
    #[must_use]
    fn seek_non_whitespace_backward(&self, from: usize) -> Option<(usize, u8)>;

    /// Decide whether the slice of input between `from` (inclusive)
    /// and `to` (exclusive) matches the `member` (comparing bitwise,
    /// including double quotes delimiters).
    ///
    /// This will also check if the leading double quote is not
    /// escaped by a backslash character.
    fn is_member_match(&self, from: usize, to: usize, member: &JsonString) -> bool;
}

/// An iterator over blocks of input of size `N`.
/// Implementations MUST guarantee that the blocks returned from `next`
/// are *exactly* of size `N`.
pub trait InputBlockIterator<'i, const N: usize> {
    /// The type of blocks returned.
    type Block: InputBlock<'i, N>;

    type Error: Into<InputError>;

    fn next(&mut self) -> Result<Option<Self::Block>, Self::Error>;

    /// Get the offset of the iterator in the input.
    ///
    /// The offset is the starting point of the block that will be returned next
    /// from this iterator, if any. It starts as 0 and increases by `N` on every
    /// block retrieved.
    fn get_offset(&self) -> usize;

    /// Offset the iterator by `count` full blocks forward.
    ///
    /// The `count` parameter must be greater than 0.
    fn offset(&mut self, count: isize);
}

/// A block of bytes of size `N` returned from [`InputBlockIterator`].
pub trait InputBlock<'i, const N: usize>: Deref<Target = [u8]> {
    /// Split the block in half, giving two slices of size `N`/2.
    #[must_use]
    fn halves(&self) -> (&[u8], &[u8]);

    /// Split the block in four, giving four slices of size `N`/4.
    #[inline]
    #[must_use]
    fn quarters(&self) -> (&[u8], &[u8], &[u8], &[u8]) {
        assert_eq!(N % 4, 0);
        let (half1, half2) = self.halves();
        let (q1, q2) = (&half1[..N / 4], &half1[N / 4..]);
        let (q3, q4) = (&half2[..N / 4], &half2[N / 4..]);

        (q1, q2, q3, q4)
    }
}

impl<'i, const N: usize> InputBlock<'i, N> for &'i [u8] {
    #[inline(always)]
    fn halves(&self) -> (&[u8], &[u8]) {
        assert_eq!(N % 2, 0);
        (&self[..N / 2], &self[N / 2..])
    }
}

pub(super) trait SliceSeekable {
    fn is_member_match(&self, from: usize, to: usize, member: &JsonString) -> bool;

    fn seek_backward(&self, from: usize, needle: u8) -> Option<usize>;

    fn seek_forward<const N: usize>(&self, from: usize, needles: [u8; N]) -> Option<(usize, u8)>;

    fn seek_non_whitespace_forward(&self, from: usize) -> Option<(usize, u8)>;

    fn seek_non_whitespace_backward(&self, from: usize) -> Option<(usize, u8)>;
}
