use std::ops::Shl;

pub(crate) trait Mask {
    fn is_lit<N>(&self, bit: N) -> bool
    where
        Self: Shl<N, Output = Self>;
}

impl Mask for u128 {
    #[inline(always)]
    fn is_lit<N>(&self, bit: N) -> bool
    where
        Self: Shl<N, Output = Self>,
    {
        (*self & (1 << bit)) != 0
    }
}

impl Mask for u64 {
    #[inline(always)]
    fn is_lit<N>(&self, bit: N) -> bool
    where
        Self: Shl<N, Output = Self>,
    {
        (*self & (1 << bit)) != 0
    }
}

impl Mask for u32 {
    #[inline(always)]
    fn is_lit<N>(&self, bit: N) -> bool
    where
        Self: Shl<N, Output = Self>,
    {
        (*self & (1 << bit)) != 0
    }
}

impl Mask for usize {
    #[inline(always)]
    fn is_lit<N>(&self, bit: N) -> bool
    where
        Self: Shl<N, Output = Self>,
    {
        (*self & (1 << bit)) != 0
    }
}

#[allow(dead_code)]
pub(crate) mod m32 {
    pub(crate) fn combine_16(m1: u16, m2: u16) -> u32 {
        u32::from(m1) | (u32::from(m2) << 16)
    }
}

#[allow(dead_code)]
pub(crate) mod m64 {
    pub(crate) fn combine_16(m1: u16, m2: u16, m3: u16, m4: u16) -> u64 {
        u64::from(m1) | (u64::from(m2) << 16) | (u64::from(m3) << 32) | (u64::from(m4) << 48)
    }

    pub(crate) fn combine_32(m1: u32, m2: u32) -> u64 {
        u64::from(m1) | (u64::from(m2) << 32)
    }
}

#[allow(dead_code)]
pub(crate) mod m128 {
    pub(crate) fn combine_16(m1: u16, m2: u16, m3: u16, m4: u16,
                             m5: u16, m6: u16, m7: u16, m8: u16) -> u128 {
        u128::from(m1) | (u128::from(m2) << 16) | (u128::from(m3) << 32) | (u128::from(m4) << 48) |
        (u128::from(m5) << 64) | (u128::from(m6) << 80) | (u128::from(m7) << 96) | (u128::from(m8) << 112)
    }

    pub(crate) fn combine_32(m1: u32, m2: u32, m3: u32, m4: u32) -> u128 {
        u128::from(m1) | (u128::from(m2) << 32) | (u128::from(m3) << 64) | (u128::from(m4) << 96)
    }

    pub(crate) fn combine_64(m1: u64, m2: u64) -> u128 {
        u128::from(m1) | (u128::from(m2) << 64)
    }
}
