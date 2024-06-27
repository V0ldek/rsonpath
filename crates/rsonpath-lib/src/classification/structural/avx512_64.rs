use super::{
    shared::{mask_128, vector_512},
    *,
};
use crate::{
    bin_u128,
    classification::mask::m128,
    classification::{QuoteClassifiedBlock, ResumeClassifierBlockState},
    debug,
    input::InputBlock,
};

super::shared::structural_classifier!(Avx512Classifier128, BlockAvx512Classifier128, mask_128, 128, u128);

struct BlockAvx512Classifier128 {
    internal_classifier: vector_512::BlockClassifier512,
}

impl BlockAvx512Classifier128 {
    fn new() -> Self {
        Self {
            // SAFETY: target feature invariant
            internal_classifier: unsafe { vector_512::BlockClassifier512::new() },
        }
    }

    #[inline(always)]
    unsafe fn classify<'i, B: InputBlock<'i, 128>>(
        &mut self,
        quote_classified_block: QuoteClassifiedBlock<B, u128, 128>,
    ) -> mask_128::StructuralsBlock<B> {
        let (block1, block2) = quote_classified_block.block.halves();
        let classification1 = self.internal_classifier.classify_block(block1);
        let classification2 = self.internal_classifier.classify_block(block2);

        let structural = m128::combine_64(classification1.structural, classification2.structural);
        let nonquoted_structural = structural & !quote_classified_block.within_quotes_mask;

        bin_u128!("structural", structural);
        bin_u128!("nonquoted_structural", nonquoted_structural);

        mask_128::StructuralsBlock::new(quote_classified_block, nonquoted_structural)
    }
}
