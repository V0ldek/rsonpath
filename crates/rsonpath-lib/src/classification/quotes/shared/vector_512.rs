#[cfg(target_arch = "x86")]
use ::core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use ::core::arch::x86_64::*;

#[inline(always)]
pub(crate) unsafe fn quote_mask() -> __m512i {
    _mm512_set1_epi8(b'"' as i8)
}

#[inline(always)]
pub(crate) unsafe fn slash_mask() -> __m512i {
    _mm512_set1_epi8(b'\\' as i8)
}

#[target_feature(enable = "avx512f")]
#[target_feature(enable = "avx512bw")]
pub(crate) unsafe fn classify_block(block: &[u8]) -> BlockClassification512 {
    let byte_vector = _mm512_loadu_si512(block.as_ptr().cast::<i32>());

    let slashes = _mm512_cmpeq_epi8_mask(byte_vector, slash_mask());

    let quotes = _mm512_cmpeq_epi8_mask(byte_vector, quote_mask());

    BlockClassification512 { slashes, quotes }
}

pub(crate) struct BlockClassification512 {
    pub(crate) slashes: u64,
    pub(crate) quotes: u64,
}
