//! Input structures that can be fed into an [`Engine`](crate::engine::Engine).
//!
//! The engine itself is generic in the [`Input`] trait declared here.
//! There are a couple of different built-in implementations, each
//! suitable for a different scenario. Consult the module-level
//! documentation of each type to determine which to use. Here's a quick
//! cheat-sheet:
//!
//! |:Input scenario|Type to use|
//! ----------------
//! |file based | [`MmapInput`] |
//! |memory based | [`OwnedBytes`] |
//! |memory based, already aligned | [`BorrowedBytes`] |
//! |[`Read`](std::io::Read) based | [`BufferedInput`] |
//!
pub mod borrowed;
pub mod buffered;
pub mod error;
pub mod owned;
pub use borrowed::BorrowedBytes;
pub use buffered::BufferedInput;
pub use owned::OwnedBytes;
pub mod mmap;
pub use mmap::MmapInput;

use self::error::InputError;
use crate::{query::JsonString, FallibleIterator};
use std::ops::Deref;

/// Shorthand for the associated [`InputBlock`] type for given
/// [`Input`]'s iterator.
///
/// Typing `IBlock<'a, I, N>` is a bit more ergonomic than
/// `<<I as Input>::BlockIterator<'a, N> as InputBlockIterator<'a, N>>::Block`.
pub type IBlock<'a, I, const N: usize> = <<I as Input>::BlockIterator<'a, N> as InputBlockIterator<'a, N>>::Block;

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
    type BlockIterator<'a, const N: usize>: InputBlockIterator<'a, N>
    where
        Self: 'a;

    /// Iterate over blocks of size `N` of the input.
    /// `N` has to be a power of two larger than 1.
    #[must_use]
    fn iter_blocks<const N: usize>(&self) -> Self::BlockIterator<'_, N>;

    /// Search for an occurrence of `needle` in the input,
    /// starting from `from` and looking back. Returns the index
    /// of the first occurrence or `None` if the `needle` was not found.
    #[must_use]
    fn seek_backward(&self, from: usize, needle: u8) -> Option<usize>;

    /// Search for the first byte in the input that is not ASCII whitespace
    /// starting from `from`. Returns a pair: the index of first such byte,
    /// and the byte itself; or `None` if no non-whitespace characters
    /// were found.
    ///
    /// # Errors
    /// This function can read more data from the input if no relevant characters are found
    /// in the current buffer, which can fail.
    fn seek_non_whitespace_forward(&self, from: usize) -> Result<Option<(usize, u8)>, InputError>;

    /// Search for the first byte in the input that is not ASCII whitespace
    /// starting from `from` and looking back. Returns a pair:
    /// the index of first such byte, and the byte itself;
    /// or `None` if no non-whitespace characters were found.
    #[must_use]
    fn seek_non_whitespace_backward(&self, from: usize) -> Option<(usize, u8)>;

    /// Search the input for the first occurrence of member name `member`
    /// (comparing bitwise, including double quotes delimiters)
    /// starting from `from`. Returns the index of the first occurrence,
    /// or `None` if no occurrence was found.
    ///
    /// This will also check if the leading double quote is not
    /// escaped by a backslash character, but will ignore any other
    /// structural properties of the input. In particular, the member
    /// might be found at an arbitrary depth.
    ///
    /// # Errors
    /// This function can read more data from the input if no relevant characters are found
    /// in the current buffer, which can fail.
    #[cfg(feature = "head-skip")]
    fn find_member(&self, from: usize, member: &JsonString) -> Result<Option<usize>, InputError>;

    /// Decide whether the slice of input between `from` (inclusive)
    /// and `to` (exclusive) matches the `member` (comparing bitwise,
    /// including double quotes delimiters).
    ///
    /// This will also check if the leading double quote is not
    /// escaped by a backslash character.
    #[must_use]
    fn is_member_match(&self, from: usize, to: usize, member: &JsonString) -> bool;
}

/// An iterator over blocks of input of size `N`.
/// Implementations MUST guarantee that the blocks returned from `next`
/// are *exactly* of size `N`.
pub trait InputBlockIterator<'a, const N: usize>: FallibleIterator<Item = Self::Block, Error = InputError> {
    /// The type of blocks returned.
    type Block: InputBlock<'a, N>;

    /// Offset the iterator by `count` full blocks forward.
    ///
    /// The `count` parameter must be greater than 0.
    fn offset(&mut self, count: isize);
}

/// A block of bytes of size `N` returned from [`InputBlockIterator`].
pub trait InputBlock<'a, const N: usize>: Deref<Target = [u8]> {
    /// Split the block in half, giving two slices of size `N`/2.
    fn halves(&self) -> (&[u8], &[u8]);
}

impl<'a, const N: usize> InputBlock<'a, N> for &'a [u8] {
    #[inline(always)]
    fn halves(&self) -> (&[u8], &[u8]) {
        assert_eq!(N % 2, 0);
        (&self[..N / 2], &self[N / 2..])
    }
}

pub(super) mod in_slice {
    use super::MAX_BLOCK_SIZE;
    use crate::query::JsonString;

    #[inline]
    pub(super) fn pad_last_block(bytes: &[u8]) -> [u8; MAX_BLOCK_SIZE] {
        let mut last_block_buf = [0; MAX_BLOCK_SIZE];
        let last_block_start = (bytes.len() / MAX_BLOCK_SIZE) * MAX_BLOCK_SIZE;
        let last_block_slice = &bytes[last_block_start..];

        last_block_buf[..last_block_slice.len()].copy_from_slice(last_block_slice);

        last_block_buf
    }

    #[inline]
    pub(super) fn seek_backward(bytes: &[u8], from: usize, needle: u8) -> Option<usize> {
        let mut idx = from;

        if idx >= bytes.len() {
            return None;
        }

        loop {
            if bytes[idx] == needle {
                return Some(idx);
            }
            if idx == 0 {
                return None;
            }
            idx -= 1;
        }
    }

    #[inline]
    pub(super) fn seek_non_whitespace_forward(bytes: &[u8], from: usize) -> Option<(usize, u8)> {
        let mut idx = from;

        if idx >= bytes.len() {
            return None;
        }

        loop {
            let b = bytes[idx];
            if !b.is_ascii_whitespace() {
                return Some((idx, b));
            }
            idx += 1;
            if idx == bytes.len() {
                return None;
            }
        }
    }

    #[inline]
    pub(super) fn seek_non_whitespace_backward(bytes: &[u8], from: usize) -> Option<(usize, u8)> {
        let mut idx = from;

        if idx >= bytes.len() {
            return None;
        }

        loop {
            let b = bytes[idx];
            if !b.is_ascii_whitespace() {
                return Some((idx, b));
            }
            if idx == 0 {
                return None;
            }
            idx -= 1;
        }
    }

    #[inline]
    #[cfg(feature = "head-skip")]
    pub(super) fn find_member(bytes: &[u8], from: usize, member: &JsonString) -> Option<usize> {
        use memchr::memmem;

        let finder = memmem::Finder::new(member.bytes_with_quotes());
        let mut idx = from;
        let brr = bytes.len();

        if brr <= idx {
            return None;
        }

        loop {
            match finder.find(&bytes[idx..brr]) {
                Some(offset) => {
                    let starting_quote_idx = offset + idx;
                    if bytes[starting_quote_idx - 1] != b'\\' {
                        return Some(starting_quote_idx);
                    } else {
                        idx = starting_quote_idx + member.bytes_with_quotes().len() + 1;
                    }
                }
                None => return None,
            }
        }
    }

    #[inline]
    pub(super) fn is_member_match(bytes: &[u8], from: usize, to: usize, member: &JsonString) -> bool {
        let slice = &bytes[from..to];
        member.bytes_with_quotes() == slice && (from == 0 || bytes[from - 1] != b'\\')
    }
}
