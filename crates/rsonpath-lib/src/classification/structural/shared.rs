#[cfg(target_arch = "x86")]
pub(super) mod mask_32;
#[cfg(target_arch = "x86_64")]
pub(super) mod mask_64;
#[cfg(target_arch = "x86_64")]
pub(super) mod mask_128;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(super) mod vector_128;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(super) mod vector_256;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(super) mod vector_512;


#[allow(unused_macros)]
macro_rules! structural_classifier {
    ($name:ident, $core:ident, $mask_mod:ident, $size:literal, $mask_ty:ty) => {
        pub(crate) struct Constructor;

        impl StructuralImpl for Constructor {
            type Classifier<'i, I, Q> = $name<'i, I, Q>
            where I: InputBlockIterator<'i, BLOCK_SIZE>, Q: QuoteClassifiedIterator<'i, I, MaskType, BLOCK_SIZE>;

            #[inline]
            #[allow(dead_code)]
            fn new<'i, I, Q>(iter: Q) -> Self::Classifier<'i, I, Q>
            where
                I: InputBlockIterator<'i, BLOCK_SIZE>,
                Q: QuoteClassifiedIterator<'i, I, MaskType, BLOCK_SIZE>,
            {
                Self::Classifier {
                    iter,
                    classifier: $core::new(),
                    block: None,
                    are_commas_on: false,
                    are_colons_on: false,
                }
            }
        }

        pub(crate) struct $name<'a, I, Q>
        where
            I: InputBlockIterator<'a, $size>,
        {
            iter: Q,
            classifier: $core,
            block: Option<$mask_mod::StructuralsBlock<I::Block>>,
            are_commas_on: bool,
            are_colons_on: bool,
        }

        impl<'a, I: InputBlockIterator<'a, $size>, Q: QuoteClassifiedIterator<'a, I, $mask_ty, $size>> $name<'a, I, Q> {
            #[inline(always)]
            fn current_block_is_spent(&self) -> bool {
                self.block
                    .as_ref()
                    .map_or(true, $mask_mod::StructuralsBlock::is_empty)
            }

            #[inline]
            fn reclassify(&mut self, idx: usize) {
                if let Some(block) = self.block.take() {
                    let quote_classified_block = block.quote_classified;
                    let relevant_idx = idx + 1;
                    let block_idx = (idx + 1) % $size;
                    debug!("relevant_idx is {relevant_idx}.");

                    if block_idx != 0 || relevant_idx == self.iter.get_offset() {
                        debug!("need to reclassify.");

                        let mask = <$mask_ty>::MAX << block_idx;
                        // SAFETY: target_feature invariant
                        let mut new_block = unsafe { self.classifier.classify(quote_classified_block) };
                        new_block.structural_mask &= mask;
                        self.block = Some(new_block);
                    }
                }
            }
        }

        impl<'a, I, Q> FallibleIterator for $name<'a, I, Q>
        where
            I: InputBlockIterator<'a, $size>,
            Q: QuoteClassifiedIterator<'a, I, $mask_ty, $size>,
        {
            type Item = Structural;
            type Error = InputError;

            #[inline(always)]
            fn next(&mut self) -> Result<Option<Structural>, Self::Error> {
                while self.current_block_is_spent() {
                    match self.iter.next() {
                        Ok(Some(block)) => {
                            // SAFETY: target_feature invariant
                            self.block = unsafe { Some(self.classifier.classify(block)) };
                        }
                        Ok(None) => {
                            self.block = None;
                            break;
                        }
                        Err(err) => return Err(err),
                    }
                }

                Ok(self
                    .block
                    .as_mut()
                    .and_then(|b| b.next().map(|x| x.offset(self.iter.get_offset()))))
            }
        }

        impl<'a, I, Q> StructuralIterator<'a, I, Q, $mask_ty, $size> for $name<'a, I, Q>
        where
            I: InputBlockIterator<'a, $size>,
            Q: QuoteClassifiedIterator<'a, I, $mask_ty, $size>,
        {
            #[inline(always)]
            fn turn_colons_and_commas_on(&mut self, idx: usize) {
                if !self.are_commas_on && !self.are_colons_on {
                    self.are_commas_on = true;
                    self.are_colons_on = true;
                    debug!("Turning both commas and colons on at {idx}.");
                    // SAFETY: target_feature invariant
                    unsafe { self.classifier.internal_classifier.toggle_colons_and_commas() }

                    self.reclassify(idx);
                } else if !self.are_commas_on {
                    self.turn_commas_on(idx);
                } else if !self.are_colons_on {
                    self.turn_colons_on(idx);
                }
            }

            #[inline(always)]
            fn turn_colons_and_commas_off(&mut self) {
                if self.are_commas_on && self.are_colons_on {
                    self.are_commas_on = false;
                    self.are_colons_on = false;
                    debug!("Turning both commas and colons off.");
                    // SAFETY: target_feature invariant
                    unsafe { self.classifier.internal_classifier.toggle_colons_and_commas() }
                } else if self.are_commas_on {
                    self.turn_commas_off();
                } else if self.are_colons_on {
                    self.turn_colons_off();
                }
            }

            #[inline(always)]
            fn turn_commas_on(&mut self, idx: usize) {
                if !self.are_commas_on {
                    self.are_commas_on = true;
                    debug!("Turning commas on at {idx}.");
                    // SAFETY: target_feature invariant
                    unsafe { self.classifier.internal_classifier.toggle_commas() }

                    self.reclassify(idx);
                }
            }

            #[inline(always)]
            fn turn_commas_off(&mut self) {
                if self.are_commas_on {
                    self.are_commas_on = false;
                    debug!("Turning commas off.");
                    // SAFETY: target_feature invariant
                    unsafe { self.classifier.internal_classifier.toggle_commas() }
                }
            }

            #[inline(always)]
            fn turn_colons_on(&mut self, idx: usize) {
                if !self.are_colons_on {
                    self.are_colons_on = true;
                    debug!("Turning colons on at {idx}.");
                    // SAFETY: target_feature invariant
                    unsafe { self.classifier.internal_classifier.toggle_colons() }

                    self.reclassify(idx);
                }
            }

            #[inline(always)]
            fn turn_colons_off(&mut self) {
                if self.are_colons_on {
                    self.are_colons_on = false;
                    debug!("Turning colons off.");
                    // SAFETY: target_feature invariant
                    unsafe { self.classifier.internal_classifier.toggle_colons() }
                }
            }

            #[inline(always)]
            fn stop(self) -> ResumeClassifierState<'a, I, Q, $mask_ty, $size> {
                let block = self.block.map(|b| ResumeClassifierBlockState {
                    idx: b.get_idx() as usize,
                    block: b.quote_classified,
                });

                ResumeClassifierState {
                    iter: self.iter,
                    block,
                    are_commas_on: self.are_commas_on,
                    are_colons_on: self.are_colons_on,
                }
            }

            #[inline(always)]
            fn resume(state: ResumeClassifierState<'a, I, Q, $mask_ty, $size>) -> Self {
                let mut classifier = $core::new();

                // SAFETY: target_feature invariant
                unsafe {
                    if state.are_commas_on && state.are_colons_on {
                        classifier.internal_classifier.toggle_colons_and_commas();
                    } else {
                        if state.are_commas_on {
                            classifier.internal_classifier.toggle_commas();
                        }
                        if state.are_colons_on {
                            classifier.internal_classifier.toggle_colons();
                        }
                    }
                }

                let block = state.block.map(|b| {
                    // SAFETY: target_feature invariant
                    let mut block = unsafe { classifier.classify(b.block) };
                    let idx_mask = <$mask_ty>::MAX.checked_shl(b.idx as u32).unwrap_or(0);
                    block.structural_mask &= idx_mask;

                    block
                });

                Self {
                    iter: state.iter,
                    block,
                    classifier,
                    are_commas_on: state.are_commas_on,
                    are_colons_on: state.are_colons_on,
                }
            }
        }
    };
}

#[allow(unused_imports)]
pub(crate) use structural_classifier;
