use super::{
    shared::{mask_64, vector_512},
    *,
};
use crate::{
    bin_u64,
    classification::{QuoteClassifiedBlock, ResumeClassifierBlockState},
    debug,
    input::InputBlock,
};

super::shared::structural_classifier!(Avx512Classifier64, BlockAvx512Classifier64, mask_64, 64, u64);

struct BlockAvx512Classifier64 {
    internal_classifier: vector_512::BlockClassifier512,
}

impl BlockAvx512Classifier64 {
    fn new() -> Self {
        Self {
            // SAFETY: target feature invariant
            internal_classifier: unsafe { vector_512::BlockClassifier512::new() },
        }
    }

    #[inline(always)]
    unsafe fn classify<'i, B: InputBlock<'i, 64>>(
        &mut self,
        quote_classified_block: QuoteClassifiedBlock<B, u64, 64>,
    ) -> mask_64::StructuralsBlock<B> {
        let block = &quote_classified_block.block;
        let classification = self.internal_classifier.classify_block(block);

        let structural = classification.structural;
        let nonquoted_structural = structural & !quote_classified_block.within_quotes_mask;

        bin_u64!("structural", structural);
        bin_u64!("nonquoted_structural", nonquoted_structural);

        mask_64::StructuralsBlock::new(quote_classified_block, nonquoted_structural)
    }
}
