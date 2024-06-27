#[cfg(target_arch = "x86_64")]
use ::core::arch::x86_64::*;
use crate::classification::mask::m64;


const LOWER_NIBBLE_MASK_ARRAY: [u8; 32] = [
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x03, 0x01, 0x02, 0x01, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x03, 0x01, 0x02, 0x01, 0xff, 0xff,
];
const UPPER_NIBBLE_MASK_ARRAY: [u8; 32] = [
    0xfe, 0xfe, 0x10, 0x10, 0xfe, 0x01, 0xfe, 0x01, 0xfe, 0xfe, 0xfe, 0xfe, 0xfe, 0xfe, 0xfe, 0xfe, 0xfe, 0xfe, 0x10,
    0x10, 0xfe, 0x01, 0xfe, 0x01, 0xfe, 0xfe, 0xfe, 0xfe, 0xfe, 0xfe, 0xfe, 0xfe,
];
const COMMAS_TOGGLE_MASK_ARRAY: [u8; 32] = [
    0x00, 0x00, 0x12, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x12,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
];
const COLON_TOGGLE_MASK_ARRAY: [u8; 32] = [
    0x00, 0x00, 0x00, 0x13, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x13, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
];

#[target_feature(enable = "avx512f")]
#[inline]
pub(crate) unsafe fn upper_nibble_zeroing_mask() -> __m512i {
    _mm512_set1_epi8(0x0F)
}

#[target_feature(enable = "avx512f")]
#[inline]
pub(crate) unsafe fn lower_nibble_mask() -> __m512i {
    _mm512_loadu_si512(LOWER_NIBBLE_MASK_ARRAY.as_ptr().cast::<i32>())
}

#[target_feature(enable = "avx512f")]
#[inline]
pub(crate) unsafe fn upper_nibble_mask() -> __m512i {
    _mm512_loadu_si512(UPPER_NIBBLE_MASK_ARRAY.as_ptr().cast::<i32>())
}

#[target_feature(enable = "avx512f")]
#[inline]
pub(crate) unsafe fn commas_toggle_mask() -> __m512i {
    _mm512_loadu_si512(COMMAS_TOGGLE_MASK_ARRAY.as_ptr().cast::<i32>())
}

#[target_feature(enable = "avx512f")]
#[inline]
pub(crate) unsafe fn colons_toggle_mask() -> __m512i {
    _mm512_loadu_si512(COLON_TOGGLE_MASK_ARRAY.as_ptr().cast::<i32>())
}

#[target_feature(enable = "avx512f")]
#[inline]
pub(crate) unsafe fn colons_and_commas_toggle_mask() -> __m512i {
    _mm512_or_si512(colons_toggle_mask(), commas_toggle_mask())
}

pub(crate) struct BlockClassifier512 {
    upper_nibble_mask: __m512i,
}

impl BlockClassifier512 {
    #[target_feature(enable = "avx512f")]
    #[inline]
    pub(crate) unsafe fn new() -> Self {
        Self {
            upper_nibble_mask: upper_nibble_mask(),
        }
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    pub(crate) unsafe fn toggle_commas(&mut self) {
        self.upper_nibble_mask = _mm512_xor_si512(self.upper_nibble_mask, commas_toggle_mask());
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    pub(crate) unsafe fn toggle_colons(&mut self) {
        self.upper_nibble_mask = _mm512_xor_si512(self.upper_nibble_mask, colons_toggle_mask());
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    pub(crate) unsafe fn toggle_colons_and_commas(&mut self) {
        self.upper_nibble_mask = _mm512_xor_si512(self.upper_nibble_mask, colons_and_commas_toggle_mask());
    }

    #[target_feature(enable = "avx512f")]
    #[target_feature(enable = "avx512bw")]
    #[inline]
    pub(crate) unsafe fn classify_block(&self, block: &[u8]) -> BlockClassification512 {
        let byte_vector = _mm512_loadu_si512(block.as_ptr().cast::<i32>());
        let shifted_byte_vector = _mm512_srli_epi16::<4>(byte_vector);
        let upper_nibble_byte_vector = _mm512_and_si512(shifted_byte_vector, upper_nibble_zeroing_mask());
        let lower_nibble_lookup = _mm512_shuffle_epi8(lower_nibble_mask(), byte_vector);
        let upper_nibble_lookup = _mm512_shuffle_epi8(self.upper_nibble_mask, upper_nibble_byte_vector);
        let structural= _mm512_cmpeq_epi8_mask(lower_nibble_lookup, upper_nibble_lookup);

        BlockClassification512 { structural }
    }
}

pub(crate) struct BlockClassification512 {
    pub(crate) structural: u64,
}
