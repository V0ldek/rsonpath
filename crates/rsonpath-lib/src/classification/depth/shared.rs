#[cfg(target_arch = "x86")]
pub(super) mod mask_32;
#[cfg(target_arch = "x86_64")]
pub(super) mod mask_64;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(super) mod vector_128;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(super) mod vector_256;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(super) mod vector_512;

#[allow(unused_macros)]
macro_rules! depth_classifier {
    ($name:ident, $core:ident, $vector:ident, $size:literal, $mask_ty:ty) => {
        pub(crate) struct Constructor;

        impl DepthImpl for Constructor {
            type Classifier<'i, I, Q> = $name<'i, I, Q>
            where I: InputBlockIterator<'i, BLOCK_SIZE>, Q: QuoteClassifiedIterator<'i, I, MaskType, BLOCK_SIZE>;
        }

        pub(crate) struct $name<'i, I, Q> {
            iter: Q,
            classifier: $core,
            were_commas_on: bool,
            were_colons_on: bool,
            phantom: PhantomData<(&'i (), I)>,
        }

        impl<'a, I, Q> FallibleIterator for $name<'a, I, Q>
        where
            I: InputBlockIterator<'a, $size>,
            Q: QuoteClassifiedIterator<'a, I, $mask_ty, $size>,
        {
            type Item = $vector<'a, I::Block>;
            type Error = InputError;

            #[inline(always)]
            fn next(&mut self) -> Result<Option<Self::Item>, Self::Error> {
                let quote_classified = self.iter.next()?;
                Ok(quote_classified.map(|q| new_vector(q, &self.classifier)))
            }
        }

        impl<'a, I, Q> DepthIterator<'a, I, Q, $mask_ty, $size> for $name<'a, I, Q>
        where
            I: InputBlockIterator<'a, $size>,
            Q: QuoteClassifiedIterator<'a, I, $mask_ty, $size>,
        {
            type Block = $vector<'a, I::Block>;

            #[inline(always)]
            fn stop(self, block: Option<Self::Block>) -> ResumeClassifierState<'a, I, Q, $mask_ty, $size> {
                let block_state = block.and_then(|b| {
                    let idx = b.idx;
                    debug!("Depth iterator stopping at index {idx}");
                    if idx >= b.quote_classified.len() {
                        None
                    } else {
                        Some(ResumeClassifierBlockState {
                            block: b.quote_classified,
                            idx,
                        })
                    }
                });

                ResumeClassifierState {
                    iter: self.iter,
                    block: block_state,
                    are_commas_on: self.were_commas_on,
                    are_colons_on: self.were_colons_on,
                }
            }

            #[inline(always)]
            fn resume(
                state: ResumeClassifierState<'a, I, Q, $mask_ty, $size>,
                opening: BracketType,
            ) -> (Option<Self::Block>, Self) {
                let classifier = $core::new(opening);
                let first_block = state.block.and_then(|b| {
                    if b.idx == $size {
                        None
                    } else {
                        Some(new_vector_from(b.block, &classifier, b.idx))
                    }
                });

                (
                    first_block,
                    $name {
                        iter: state.iter,
                        classifier,
                        phantom: PhantomData,
                        were_commas_on: state.are_commas_on,
                        were_colons_on: state.are_colons_on,
                    },
                )
            }
        }
    };
}

#[allow(unused_imports)]
pub(crate) use depth_classifier;
