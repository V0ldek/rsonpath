use crate::classification::{
    structural::{BracketType, Structural},
    QuoteClassifiedBlock,
};
use std::ops::Deref;

const SIZE: usize = 128;

pub(crate) struct StructuralsBlock<B> {
    pub(crate) quote_classified: QuoteClassifiedBlock<B, u128, SIZE>,
    pub(crate) structural_mask: u128,
}

impl<B> StructuralsBlock<B> {
    #[inline(always)]
    pub(crate) fn new(block: QuoteClassifiedBlock<B, u128, SIZE>, structural_mask: u128) -> Self {
        Self {
            quote_classified: block,
            structural_mask,
        }
    }

    #[inline(always)]
    pub(crate) fn is_empty(&self) -> bool {
        self.structural_mask == 0
    }

    #[inline(always)]
    pub(crate) fn get_idx(&self) -> u32 {
        self.structural_mask.trailing_zeros()
    }
}

impl<B: Deref<Target = [u8]>> Iterator for StructuralsBlock<B> {
    type Item = Structural;

    #[inline(always)]
    fn next(&mut self) -> Option<Structural> {
        let idx = self.get_idx() as usize;
        (idx < SIZE).then(|| {
            let bit_mask = 1 << idx;

            self.structural_mask ^= bit_mask;

            // The last match being a catch-all *is important*.
            // It has major performance implications, since the jump table generated here is a hot path for the engine.
            // Changing this match must be accompanied with benchmark runs to make sure perf does not regress.
            match self.quote_classified.block[idx] {
                b':' => Structural::Colon(idx),
                b'{' => Structural::Opening(BracketType::Curly, idx),
                b'[' => Structural::Opening(BracketType::Square, idx),
                b',' => Structural::Comma(idx),
                b'}' => Structural::Closing(BracketType::Curly, idx),
                _ => Structural::Closing(BracketType::Square, idx),
            }
        })
    }
}

impl<B: Deref<Target = [u8]>> std::iter::FusedIterator for StructuralsBlock<B> {}
