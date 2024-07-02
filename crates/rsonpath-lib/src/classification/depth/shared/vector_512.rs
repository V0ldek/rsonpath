use crate::classification::structural::BracketType;

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

pub(crate) struct DelimiterClassifierImpl512 {
    opening: i8,
}

impl DelimiterClassifierImpl512 {
    pub(crate) fn new(opening: BracketType) -> Self {
        let opening = match opening {
            BracketType::Square => b'[',
            BracketType::Curly => b'{',
        };

        Self { opening: opening as i8 }
    }

    #[inline(always)]
    unsafe fn opening_mask(&self) -> __m512i {
        _mm512_set1_epi8(self.opening)
    }

    #[inline(always)]
    unsafe fn closing_mask(&self) -> __m512i {
        _mm512_set1_epi8(self.opening + 2)
    }

    #[target_feature(enable = "avx512f")]
    #[target_feature(enable = "avx512bw")]
    #[inline]
    pub(crate) unsafe fn get_opening_and_closing_masks(&self, bytes: &[u8]) -> (u64, u64) {
        assert_eq!(64, bytes.len());
        // SAFETY: target_feature invariant
        unsafe {
            let byte_vector = _mm512_loadu_si512(bytes.as_ptr().cast::<i32>());
            let opening_mask = _mm512_cmpeq_epi8_mask(byte_vector, self.opening_mask());
            let closing_mask = _mm512_cmpeq_epi8_mask(byte_vector, self.closing_mask());

            (opening_mask, closing_mask)
        }
    }
}
