#[cfg(target_arch = "x86")]
use ::core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use ::core::arch::x86_64::*;

pub(crate) struct BlockClassifier512 {
    first: __m512i,
    second: __m512i,
}

impl BlockClassifier512 {
    #[target_feature(enable = "avx512f")]
    pub(crate) unsafe fn new(first: u8, second: u8) -> Self {
        Self {
            first: _mm512_set1_epi8(first as i8),
            second: _mm512_set1_epi8(second as i8),
        }
    }

    #[target_feature(enable = "avx512f")]
    pub(crate) unsafe fn classify_block(&self, block: &[u8]) -> BlockClassification512 {
        let byte_vector = _mm512_loadu_si512(block.as_ptr().cast::<i32>());

        let first = _mm512_cmpeq_epi8_mask(byte_vector, self.first);
        let second = _mm512_cmpeq_epi8_mask(byte_vector, self.second);

        BlockClassification512 { first, second }
    }
}

pub(crate) struct BlockClassification512 {
    pub(crate) first: u64,
    pub(crate) second: u64,
}
