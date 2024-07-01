use super::{
    shared::{mask_64::DepthVector64, vector_512::DelimiterClassifierImpl512},
    *,
};
use crate::{
    classification::{QuoteClassifiedBlock, ResumeClassifierBlockState},
    debug,
    input::InputBlock,
};
use std::marker::PhantomData;

const SIZE: usize = 64;

shared::depth_classifier!(Avx512VectorIterator64, DelimiterClassifierImpl512, DepthVector64, 64, u64);

#[inline(always)]
fn new_vector<'a, B: InputBlock<'a, SIZE>>(
    bytes: QuoteClassifiedBlock<B, u64, SIZE>,
    classifier: &DelimiterClassifierImpl512,
) -> DepthVector64<'a, B> {
    new_vector_from(bytes, classifier, 0)
}

#[inline(always)]
fn new_vector_from<'a, B: InputBlock<'a, SIZE>>(
    bytes: QuoteClassifiedBlock<B, u64, SIZE>,
    classifier: &DelimiterClassifierImpl512,
    idx: usize,
) -> DepthVector64<'a, B> {
    // SAFETY: target_feature invariant
    unsafe { new_avx512(bytes, classifier, idx) }
}

#[inline(always)]
unsafe fn new_avx512<'a, B: InputBlock<'a, SIZE>>(
    bytes: QuoteClassifiedBlock<B, u64, SIZE>,
    classifier: &DelimiterClassifierImpl512,
    start_idx: usize,
) -> DepthVector64<'a, B> {
    let idx_mask = 0xFFFF_FFFF_FFFF_FFFF_u64 << start_idx;
    let block = &bytes.block;
    let (opening_mask, closing_mask) = classifier.get_opening_and_closing_masks(block);

    let opening_mask = opening_mask & (!bytes.within_quotes_mask) & idx_mask;
    let closing_mask = closing_mask & (!bytes.within_quotes_mask) & idx_mask;

    DepthVector64 {
        quote_classified: bytes,
        opening_mask,
        closing_mask,
        opening_count: opening_mask.count_ones(),
        depth: 0,
        idx: 0,
        phantom: PhantomData,
    }
}
