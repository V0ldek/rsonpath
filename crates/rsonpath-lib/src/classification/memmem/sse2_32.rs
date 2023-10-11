use super::{shared::mask_32, shared::vector_128, *};
use crate::{
    classification::mask::m32,
    input::{error::InputError, Input, InputBlock, InputBlockIterator},
    query::JsonString,
    result::InputRecorder,
    FallibleIterator,
};

const SIZE: usize = 32;

pub(crate) struct Constructor;

impl MemmemImpl for Constructor {
    type Classifier<'i, 'b, 'r, I, R> = Sse2MemmemClassifier32<'i, 'b, 'r, I, R>
    where
        I: Input + 'i,
        <I as Input>::BlockIterator<'i, 'r, BLOCK_SIZE, R>: 'b,
        R: InputRecorder<<I as Input>::Block<'i, BLOCK_SIZE>> + 'r,
        'i: 'r;

    fn memmem<'i, 'b, 'r, I, R>(
        input: &'i I,
        iter: &'b mut <I as Input>::BlockIterator<'i, 'r, BLOCK_SIZE, R>,
    ) -> Self::Classifier<'i, 'b, 'r, I, R>
    where
        I: Input,
        R: InputRecorder<<I as Input>::Block<'i, BLOCK_SIZE>>,
        'i: 'r,
    {
        Self::Classifier { input, iter }
    }
}

pub(crate) struct Sse2MemmemClassifier32<'i, 'b, 'r, I, R>
where
    I: Input,
    R: InputRecorder<I::Block<'i, SIZE>> + 'r,
{
    input: &'i I,
    iter: &'b mut I::BlockIterator<'i, 'r, SIZE, R>,
}

impl<'i, 'b, 'r, I, R> Sse2MemmemClassifier32<'i, 'b, 'r, I, R>
where
    I: Input,
    R: InputRecorder<I::Block<'i, SIZE>>,
    'i: 'r,
{
    #[inline]
    #[allow(dead_code)]
    pub(crate) fn new(input: &'i I, iter: &'b mut I::BlockIterator<'i, 'r, SIZE, R>) -> Self {
        Self { input, iter }
    }

    #[target_feature(enable = "sse2")]
    unsafe fn find_empty(
        &mut self,
        label: &JsonString,
        mut offset: usize,
    ) -> Result<Option<(usize, I::Block<'i, SIZE>)>, InputError> {
        let classifier = vector_128::BlockClassifier128::new(b'"', b'"');
        let mut previous_block: u32 = 0;

        while let Some(block) = self.iter.next()? {
            let (block1, block2) = block.halves();
            let classified1 = classifier.classify_block(block1);
            let classified2 = classifier.classify_block(block2);

            let first_bitmask = m32::combine_16(classified1.first, classified2.first);
            let second_bitmask = m32::combine_16(classified1.second, classified2.second);

            let mut result = (previous_block | (first_bitmask << 1)) & second_bitmask;
            while result != 0 {
                let idx = result.trailing_zeros() as usize;
                if self.input.is_member_match(offset + idx - 1, offset + idx, label) {
                    return Ok(Some((offset + idx - 1, block)));
                }
                result &= !(1 << idx);
            }

            offset += SIZE;
            previous_block = first_bitmask >> (SIZE - 1);
        }

        Ok(None)
    }

    // Here we want to detect the pattern `"c"`
    // For interblock communication we need to bit of information that requires extra work to get obtained.
    // one for the block cut being `"` and `c"` and one for `"c` and `"`. We only deal with one of them.
    #[target_feature(enable = "sse2")]
    unsafe fn find_letter(
        &mut self,
        label: &JsonString,
        mut offset: usize,
    ) -> Result<Option<(usize, I::Block<'i, SIZE>)>, InputError> {
        let classifier = vector_128::BlockClassifier128::new(label.bytes()[0], b'"');
        let mut previous_block: u32 = 0;

        while let Some(block) = self.iter.next()? {
            let (block1, block2) = block.halves();
            let classified1 = classifier.classify_block(block1);
            let classified2 = classifier.classify_block(block2);

            let first_bitmask = m32::combine_16(classified1.first, classified2.first);
            let second_bitmask = m32::combine_16(classified1.second, classified2.second);

            if let Some(res) =
                mask_32::find_in_mask(self.input, label, previous_block, first_bitmask, second_bitmask, offset)
            {
                return Ok(Some((res, block)));
            }

            offset += SIZE;
            previous_block = first_bitmask >> (SIZE - 1);
        }

        Ok(None)
    }

    #[target_feature(enable = "sse2")]
    #[inline]
    unsafe fn find_label_sse2(
        &mut self,
        label: &JsonString,
        mut offset: usize,
    ) -> Result<Option<(usize, I::Block<'i, SIZE>)>, InputError> {
        if label.bytes().is_empty() {
            return self.find_empty(label, offset);
        } else if label.bytes().len() == 1 {
            return self.find_letter(label, offset);
        }

        let classifier = vector_128::BlockClassifier128::new(label.bytes()[0], label.bytes()[1]);
        let mut previous_block: u32 = 0;

        while let Some(block) = self.iter.next()? {
            let (block1, block2) = block.halves();
            let classified1 = classifier.classify_block(block1);
            let classified2 = classifier.classify_block(block2);

            let first_bitmask = m32::combine_16(classified1.first, classified2.first);
            let second_bitmask = m32::combine_16(classified1.second, classified2.second);

            if let Some(res) =
                mask_32::find_in_mask(self.input, label, previous_block, first_bitmask, second_bitmask, offset)
            {
                return Ok(Some((res, block)));
            }

            offset += SIZE;
            previous_block = first_bitmask >> (SIZE - 1);
        }

        Ok(None)
    }
}

impl<'i, 'b, 'r, I, R> Memmem<'i, 'b, 'r, I, SIZE> for Sse2MemmemClassifier32<'i, 'b, 'r, I, R>
where
    I: Input,
    R: InputRecorder<I::Block<'i, SIZE>>,
    'i: 'r,
{
    // Output the relative offsets
    fn find_label(
        &mut self,
        first_block: Option<I::Block<'i, SIZE>>,
        start_idx: usize,
        label: &JsonString,
    ) -> Result<Option<(usize, I::Block<'i, SIZE>)>, InputError> {
        if let Some(b) = first_block {
            if let Some(res) = shared::find_label_in_first_block(self.input, b, start_idx, label)? {
                return Ok(Some(res));
            }
        }
        let next_block_offset = self.iter.get_offset();
        // SAFETY: target feature invariant
        unsafe { self.find_label_sse2(label, next_block_offset) }
    }
}
