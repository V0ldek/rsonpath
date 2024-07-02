use super::{shared::mask_64, shared::vector_512, *};
use crate::{
    classification::mask::m64,
    input::{error::InputErrorConvertible, InputBlock, InputBlockIterator},
};

const SIZE: usize = 64;

pub(crate) struct Constructor;

impl MemmemImpl for Constructor {
    type Classifier<'i, 'b, 'r, I, R> = Avx512MemmemClassifier64<'i, 'b, 'r, I, R>
    where
        I: Input + 'i,
        <I as Input>::BlockIterator<'i, 'r, R, BLOCK_SIZE>: 'b,
        R: InputRecorder<<I as Input>::Block<'i, BLOCK_SIZE>> + 'r,
        'i: 'r;

    fn memmem<'i, 'b, 'r, I, R>(
        input: &'i I,
        iter: &'b mut <I as Input>::BlockIterator<'i, 'r, R, BLOCK_SIZE>,
    ) -> Self::Classifier<'i, 'b, 'r, I, R>
    where
        I: Input,
        R: InputRecorder<<I as Input>::Block<'i, BLOCK_SIZE>>,
        'i: 'r,
    {
        Self::Classifier { input, iter }
    }
}

pub(crate) struct Avx512MemmemClassifier64<'i, 'b, 'r, I, R>
where
    I: Input,
    R: InputRecorder<I::Block<'i, SIZE>> + 'r,
{
    input: &'i I,
    iter: &'b mut I::BlockIterator<'i, 'r, R, SIZE>,
}

impl<'i, 'b, 'r, I, R> Avx512MemmemClassifier64<'i, 'b, 'r, I, R>
where
    I: Input,
    R: InputRecorder<I::Block<'i, SIZE>>,
    'i: 'r,
{
    #[inline]
    #[allow(dead_code)]
    pub(crate) fn new(input: &'i I, iter: &'b mut I::BlockIterator<'i, 'r, R, SIZE>) -> Self {
        Self { input, iter }
    }

    #[inline(always)]
    unsafe fn find_empty(
        &mut self,
        label: &JsonString,
        mut offset: usize,
    ) -> Result<Option<(usize, I::Block<'i, SIZE>)>, InputError> {
        let classifier = vector_512::BlockClassifier512::new(b'"', b'"');
        let mut previous_block: u64 = 0;

        // copied from avx2_32.
        while let Some(block) = self.iter.next().e()? {
            let classified = classifier.classify_block(&block);

            let mut result = (previous_block | (classified.first << 1)) & classified.second;
            while result != 0 {
                let idx = result.trailing_zeros() as usize;
                if self
                    .input
                    .is_member_match(offset + idx - 1, offset + idx + 1, label)
                    .e()?
                {
                    return Ok(Some((offset + idx - 1, block)));
                }
                result &= !(1 << idx);
            }

            offset += SIZE;
            previous_block = classified.first >> (SIZE - 1);
        }

        Ok(None)
    }

    // Here we want to detect the pattern `"c"`
    // For interblock communication we need to bit of information that requires extra work to get obtained.
    // one for the block cut being `"` and `c"` and one for `"c` and `"`. We only deal with one of them.
    #[inline(always)]
    unsafe fn find_letter(
        &mut self,
        label: &JsonString,
        mut offset: usize,
    ) -> Result<Option<(usize, I::Block<'i, SIZE>)>, InputError> {
        let classifier = vector_512::BlockClassifier512::new(label.unquoted().as_bytes()[0], b'"');
        let mut previous_block: u64 = 0;

        while let Some(block) = self.iter.next().e()? {
            let classified = classifier.classify_block(&block);

            if let Some(res) = mask_64::find_in_mask(
                self.input,
                label,
                previous_block,
                classified.first,
                classified.second,
                offset,
            )? {
                return Ok(Some((res, block)));
            }

            offset += SIZE;
            previous_block = classified.first >> (SIZE - 1);
        }

        Ok(None)
    }

    #[inline(always)]
    unsafe fn find_label_avx512(
        &mut self,
        label: &JsonString,
        mut offset: usize,
    ) -> Result<Option<(usize, I::Block<'i, SIZE>)>, InputError> {
        if label.unquoted().is_empty() {
            return self.find_empty(label, offset);
        } else if label.unquoted().len() == 1 {
            return self.find_letter(label, offset);
        }

        let classifier =
            vector_512::BlockClassifier512::new(label.unquoted().as_bytes()[0], label.unquoted().as_bytes()[1]);
        let mut previous_block: u64 = 0;

        while let Some(block) = self.iter.next().e()? {
            let classified = classifier.classify_block(&block);

            if let Some(res) = mask_64::find_in_mask(
                self.input,
                label,
                previous_block,
                classified.first,
                classified.second,
                offset,
            )? {
                return Ok(Some((res, block)));
            }

            offset += SIZE;
            previous_block = classified.first >> (SIZE - 1);
        }

        Ok(None)
    }
}

impl<'i, 'b, 'r, I, R> Memmem<'i, 'b, 'r, I, SIZE> for Avx512MemmemClassifier64<'i, 'b, 'r, I, R>
where
    I: Input,
    R: InputRecorder<I::Block<'i, SIZE>>,
    'i: 'r,
{
    #[inline(always)]
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
        unsafe { self.find_label_avx512(label, next_block_offset) }
    }
}
