use crate::bin_u32;
#[cfg(target_arch = "x86")]
use ::core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use ::core::arch::x86_64::*;

/// Bitmask selecting bits on even positions when indexing from zero.
pub(crate) const ODD: u32 = 0b0101_0101_0101_0101_0101_0101_0101_0101_u32;
/// Bitmask selecting bits on odd positions when indexing from zero.
pub(crate) const EVEN: u32 = 0b1010_1010_1010_1010_1010_1010_1010_1010_u32;

#[target_feature(enable = "sse2")]
unsafe fn all_ones128() -> __m128i {
    _mm_set1_epi8(0xFF_u8 as i8)
}

pub(crate) struct BlockClassifier32Bit {
    /// Compressed information about the state from the previous block.
    /// The first bit is lit iff the previous block ended with an unescaped escape character.
    /// The second bit is lit iff the previous block ended with a starting quote,
    /// meaning that it was not escaped, nor was it the closing quote of a quoted sequence.
    prev_block_mask: u8,
}

impl BlockClassifier32Bit {
    pub(crate) fn new() -> Self {
        Self { prev_block_mask: 0 }
    }

    /// Set the inter-block state based on slash overflow and the quotes mask.
    fn update_prev_block_mask(&mut self, set_slash_mask: bool, quotes: u32) {
        let slash_mask = u8::from(set_slash_mask);
        let quote_mask = (((quotes & (1 << 31)) >> 30) as u8) & 0x02;
        self.prev_block_mask = slash_mask | quote_mask;
    }

    /// Flip the inter-block state bit representing the quote state.
    pub(crate) fn flip_prev_quote_mask(&mut self) {
        self.prev_block_mask ^= 0x02;
    }

    /// Returns 0x01 if the last character of the previous block was an unescaped escape character,
    /// zero otherwise.
    fn get_prev_slash_mask(&self) -> u32 {
        u32::from(self.prev_block_mask & 0x01)
    }

    /// Returns 0x01 if the last character of the previous block was an unescaped quote, zero otherwise.
    fn get_prev_quote_mask(&self) -> u32 {
        u32::from((self.prev_block_mask & 0x02) >> 1)
    }

    #[target_feature(enable = "sse2")]
    #[target_feature(enable = "pclmulqdq")]
    pub(crate) unsafe fn classify(&mut self, slashes: u32, quotes: u32) -> u32 {
        let (escaped, set_prev_slash_mask) = if slashes == 0 {
            (self.get_prev_slash_mask(), false)
        } else {
            let slashes_excluding_escaped_first = slashes & !self.get_prev_slash_mask();
            let starts = slashes_excluding_escaped_first & !(slashes_excluding_escaped_first << 1);
            let odd_starts = ODD & starts;
            let even_starts = EVEN & starts;

            let odd_starts_carry = odd_starts.wrapping_add(slashes);
            let (even_starts_carry, set_prev_slash_mask) = even_starts.overflowing_add(slashes);

            let ends_of_odd_starts = odd_starts_carry & !slashes;
            let ends_of_even_starts = even_starts_carry & !slashes;

            let escaped = (ends_of_odd_starts & EVEN) | (ends_of_even_starts & ODD) | self.get_prev_slash_mask();

            (escaped, set_prev_slash_mask)
        };

        let nonescaped_quotes = (quotes & !escaped) ^ self.get_prev_quote_mask();

        let nonescaped_quotes_vector = _mm_set_epi64x(0, i64::from(nonescaped_quotes));
        let cumulative_xor = _mm_clmulepi64_si128::<0>(nonescaped_quotes_vector, all_ones128());

        let within_quotes = _mm_cvtsi128_si32(cumulative_xor) as u32;
        self.update_prev_block_mask(set_prev_slash_mask, within_quotes);

        bin_u32!("slashes", slashes);
        bin_u32!("quotes", quotes);
        bin_u32!("prev_slash_bit", self.get_prev_slash_mask());
        bin_u32!("prev_quote_bit", self.get_prev_quote_mask());
        bin_u32!("escaped", escaped);
        bin_u32!("quotes & !escaped", quotes & !escaped);
        bin_u32!("nonescaped_quotes", nonescaped_quotes);
        bin_u32!("within_quotes", within_quotes);

        within_quotes
    }
}
