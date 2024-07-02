//! JSON depth calculations on byte streams.
//!
//! Used for iterating over the document while keeping track of depth.
//! This is heavily optimized for skipping irrelevant parts of a JSON document.
//! For example, to quickly skip to the end of the currently opened object one can set the
//! depth to `1` and then advance until it reaches `0`.
//!
//! It also supports stopping and resuming with [`ResumeClassifierState`].
use super::structural::BracketType;
use crate::{
    classification::{quotes::QuoteClassifiedIterator, ResumeClassifierState},
    input::{error::InputError, InputBlockIterator},
    FallibleIterator, MaskType, BLOCK_SIZE,
};

/// Common trait for structs that enrich a byte block with JSON depth information.
#[allow(clippy::len_without_is_empty)]
pub trait DepthBlock<'a>: Sized {
    /// Add depth to the block.
    /// This is usually done at the start of a block to carry any accumulated
    /// depth over.
    fn add_depth(&mut self, depth: isize);

    /// Returns depth at the current position.
    fn get_depth(&self) -> isize;

    /// A lower bound on the depth that can be reached when advancing.
    ///
    /// It is guaranteed that [`get_depth`](`DepthBlock::get_depth`)
    /// will always return something greater or equal to this return value, but it is not guaranteed to be a depth that
    /// is actually achievable within the block. In particular, an implementation always returning
    /// [`isize::MIN`] is a correct implementation. This is meant to be a tool for performance improvements,
    /// not reliably checking the actual minimal depth within the block.
    fn estimate_lowest_possible_depth(&self) -> isize;

    /// Returns exact depth at the end of the decorated slice.
    fn depth_at_end(&self) -> isize;

    /// Advance to the next position at which depth may decrease.
    ///
    /// # Returns
    /// `false` if the end of the block was reached without any depth decrease,
    /// `true` otherwise.
    fn advance_to_next_depth_decrease(&mut self) -> bool;
}

/// Trait for depth iterators, i.e. finite iterators returning depth information
/// about JSON documents.
pub trait DepthIterator<'i, I, Q, M, const N: usize>: FallibleIterator<Item = Self::Block, Error = InputError>
where
    I: InputBlockIterator<'i, N>,
{
    /// Type of the [`DepthBlock`] implementation used by this iterator.
    type Block: DepthBlock<'i>;

    /// Resume classification from a state retrieved by a previous
    /// [`DepthIterator::stop`] or [`StructuralIterator::stop`](`crate::classification::structural::StructuralIterator::stop`) invocation.
    fn resume(state: ResumeClassifierState<'i, I, Q, M, N>, opening: BracketType) -> (Option<Self::Block>, Self);

    /// Stop classification and return a state object that can be used to resume
    /// a classifier from the place in which the current one was stopped.
    fn stop(self, block: Option<Self::Block>) -> ResumeClassifierState<'i, I, Q, M, N>;
}

/// The result of resuming a [`DepthIterator`] &ndash; the first block and the rest of the iterator.
pub struct DepthIteratorResumeOutcome<'i, I, Q, D, M, const N: usize>(pub Option<D::Block>, pub D)
where
    I: InputBlockIterator<'i, N>,
    D: DepthIterator<'i, I, Q, M, N>;

pub(crate) mod nosimd;
pub(crate) mod shared;

#[cfg(target_arch = "x86")]
pub(crate) mod avx2_32;
#[cfg(target_arch = "x86_64")]
pub(crate) mod avx2_64;
#[cfg(target_arch = "x86_64")]
pub(crate) mod avx512_64;
#[cfg(target_arch = "x86")]
pub(crate) mod sse2_32;
#[cfg(target_arch = "x86_64")]
pub(crate) mod sse2_64;

pub(crate) trait DepthImpl {
    type Classifier<'i, I, Q>: DepthIterator<'i, I, Q, MaskType, BLOCK_SIZE>
    where
        I: InputBlockIterator<'i, BLOCK_SIZE>,
        Q: QuoteClassifiedIterator<'i, I, MaskType, BLOCK_SIZE>;

    fn resume<'i, I, Q>(
        state: ResumeClassifierState<'i, I, Q, MaskType, BLOCK_SIZE>,
        opening: BracketType,
    ) -> DepthIteratorResumeOutcome<'i, I, Q, Self::Classifier<'i, I, Q>, MaskType, BLOCK_SIZE>
    where
        I: InputBlockIterator<'i, BLOCK_SIZE>,
        Q: QuoteClassifiedIterator<'i, I, MaskType, BLOCK_SIZE>,
    {
        let (first_block, iter) =
            <Self::Classifier<'i, I, Q> as DepthIterator<'i, I, Q, MaskType, BLOCK_SIZE>>::resume(state, opening);
        DepthIteratorResumeOutcome(first_block, iter)
    }
}
