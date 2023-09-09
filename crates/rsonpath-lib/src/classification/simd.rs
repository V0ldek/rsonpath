use super::{
    depth::{DepthImpl, DepthIterator, DepthIteratorResumeOutcome},
    memmem::{Memmem, MemmemImpl},
    quotes::{InnerIter, QuoteClassifiedIterator, QuotesImpl, ResumedQuoteClassifier},
    structural::{BracketType, StructuralImpl, StructuralIterator},
    ResumeClassifierState,
};
use crate::{
    input::{Input, InputBlockIterator},
    result::InputRecorder,
    MaskType, BLOCK_SIZE,
};
use cfg_if::cfg_if;
use log::warn;
use std::{fmt::Display, marker::PhantomData};

pub trait Simd: Copy {
    type QuotesClassifier<'i, I>: QuoteClassifiedIterator<'i, I, MaskType, BLOCK_SIZE> + InnerIter<I>
    where
        I: InputBlockIterator<'i, BLOCK_SIZE>;

    type StructuralClassifier<'i, I>: StructuralIterator<'i, I, Self::QuotesClassifier<'i, I>, MaskType, BLOCK_SIZE>
    where
        I: InputBlockIterator<'i, BLOCK_SIZE>;

    type DepthClassifier<'i, I>: DepthIterator<'i, I, Self::QuotesClassifier<'i, I>, MaskType, BLOCK_SIZE>
    where
        I: InputBlockIterator<'i, BLOCK_SIZE>;

    type MemmemClassifier<'i, 'b, 'r, I, R>: Memmem<'i, 'b, 'r, I, BLOCK_SIZE>
    where
        I: Input + 'i,
        I::BlockIterator<'i, 'r, BLOCK_SIZE, R>: 'b,
        R: InputRecorder<I::Block<'i, BLOCK_SIZE>> + 'r,
        'i: 'r;

    #[must_use]
    fn classify_quoted_sequences<'i, I>(self, iter: I) -> Self::QuotesClassifier<'i, I>
    where
        I: InputBlockIterator<'i, BLOCK_SIZE>;

    #[must_use]
    fn resume_quote_classification<'i, I>(
        self,
        iter: I,
        first_block: Option<I::Block>,
    ) -> ResumedQuoteClassifier<Self::QuotesClassifier<'i, I>, I::Block, MaskType, BLOCK_SIZE>
    where
        I: InputBlockIterator<'i, BLOCK_SIZE>;

    fn classify_structural_characters<'i, I>(
        self,
        iter: Self::QuotesClassifier<'i, I>,
    ) -> Self::StructuralClassifier<'i, I>
    where
        I: InputBlockIterator<'i, BLOCK_SIZE>;

    fn resume_structural_classification<'i, I>(
        self,
        state: ResumeClassifierState<'i, I, Self::QuotesClassifier<'i, I>, MaskType, BLOCK_SIZE>,
    ) -> Self::StructuralClassifier<'i, I>
    where
        I: InputBlockIterator<'i, BLOCK_SIZE>;

    fn classify_depth<'i, I>(
        self,
        iter: Self::QuotesClassifier<'i, I>,
        opening: BracketType,
    ) -> Self::DepthClassifier<'i, I>
    where
        I: InputBlockIterator<'i, BLOCK_SIZE>;

    fn resume_depth_classification<'i, I>(
        self,
        state: ResumeClassifierState<'i, I, Self::QuotesClassifier<'i, I>, MaskType, BLOCK_SIZE>,
        opening: BracketType,
    ) -> DepthIteratorResumeOutcome<
        'i,
        I,
        Self::QuotesClassifier<'i, I>,
        Self::DepthClassifier<'i, I>,
        MaskType,
        BLOCK_SIZE,
    >
    where
        I: InputBlockIterator<'i, BLOCK_SIZE>;

    fn memmem<'i, 'b, 'r, I, R>(
        self,
        input: &'i I,
        iter: &'b mut I::BlockIterator<'i, 'r, BLOCK_SIZE, R>,
    ) -> Self::MemmemClassifier<'i, 'b, 'r, I, R>
    where
        I: Input,
        R: InputRecorder<I::Block<'i, BLOCK_SIZE>>,
        'i: 'r;
}

pub(crate) struct ResolvedSimd<Q, S, D, M> {
    phantom: PhantomData<(Q, S, D, M)>,
}

impl<Q, S, D, M> Clone for ResolvedSimd<Q, S, D, M> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<Q, S, D, M> Copy for ResolvedSimd<Q, S, D, M> {}

impl<Q, S, D, M> ResolvedSimd<Q, S, D, M> {
    pub(crate) fn new() -> Self {
        Self { phantom: PhantomData }
    }
}

impl<Q, S, D, M> Simd for ResolvedSimd<Q, S, D, M>
where
    Q: QuotesImpl,
    S: StructuralImpl,
    D: DepthImpl,
    M: MemmemImpl,
{
    type QuotesClassifier<'i, I> = Q::Classifier<'i, I>
    where
        I: InputBlockIterator<'i, BLOCK_SIZE>;

    type StructuralClassifier<'i, I> = S::Classifier<'i, I, Self::QuotesClassifier<'i, I>>
    where
        I: InputBlockIterator<'i, BLOCK_SIZE>;

    type DepthClassifier<'i, I> = D::Classifier<'i, I, Self::QuotesClassifier<'i, I>>
    where
        I: InputBlockIterator<'i, BLOCK_SIZE>;

    type MemmemClassifier<'i, 'b, 'r, I, R> = M::Classifier<'i, 'b, 'r, I, R>
    where
        I: Input + 'i,
        I::BlockIterator<'i, 'r, BLOCK_SIZE, R>: 'b,
        R: InputRecorder<I::Block<'i, BLOCK_SIZE>> + 'r,
        'i: 'r;

    fn classify_quoted_sequences<'i, I>(self, iter: I) -> Self::QuotesClassifier<'i, I>
    where
        I: InputBlockIterator<'i, BLOCK_SIZE>,
    {
        Q::new(iter)
    }

    fn resume_quote_classification<'i, I>(
        self,
        iter: I,
        first_block: Option<I::Block>,
    ) -> ResumedQuoteClassifier<Self::QuotesClassifier<'i, I>, I::Block, MaskType, BLOCK_SIZE>
    where
        I: InputBlockIterator<'i, BLOCK_SIZE>,
    {
        Q::resume(iter, first_block)
    }

    fn classify_structural_characters<'i, I>(
        self,
        iter: Self::QuotesClassifier<'i, I>,
    ) -> Self::StructuralClassifier<'i, I>
    where
        I: InputBlockIterator<'i, BLOCK_SIZE>,
    {
        S::new(iter)
    }

    fn resume_structural_classification<'i, I>(
        self,
        state: ResumeClassifierState<'i, I, Self::QuotesClassifier<'i, I>, MaskType, BLOCK_SIZE>,
    ) -> Self::StructuralClassifier<'i, I>
    where
        I: InputBlockIterator<'i, BLOCK_SIZE>,
    {
        S::resume(state)
    }

    fn classify_depth<'i, I>(
        self,
        iter: Self::QuotesClassifier<'i, I>,
        opening: BracketType,
    ) -> Self::DepthClassifier<'i, I>
    where
        I: InputBlockIterator<'i, BLOCK_SIZE>,
    {
        D::new(iter, opening)
    }

    fn resume_depth_classification<'i, I>(
        self,
        state: ResumeClassifierState<'i, I, Self::QuotesClassifier<'i, I>, MaskType, BLOCK_SIZE>,
        opening: BracketType,
    ) -> DepthIteratorResumeOutcome<
        'i,
        I,
        Self::QuotesClassifier<'i, I>,
        Self::DepthClassifier<'i, I>,
        MaskType,
        BLOCK_SIZE,
    >
    where
        I: InputBlockIterator<'i, BLOCK_SIZE>,
    {
        D::resume(state, opening)
    }

    fn memmem<'i, 'b, 'r, I, R>(
        self,
        input: &'i I,
        iter: &'b mut I::BlockIterator<'i, 'r, BLOCK_SIZE, R>,
    ) -> Self::MemmemClassifier<'i, 'b, 'r, I, R>
    where
        I: Input,
        R: InputRecorder<I::Block<'i, BLOCK_SIZE>>,
        'i: 'r,
    {
        M::memmem(input, iter)
    }
}

#[derive(Clone, Copy)]
pub enum SimdTag {
    Nosimd,
    Sse2,
    Ssse3,
    Avx2,
}

#[derive(Clone, Copy)]
pub struct SimdConfiguration {
    highest_simd: SimdTag,
    fast_quotes: bool,
    fast_popcnt: bool,
}

impl SimdConfiguration {
    pub(crate) fn highest_simd(&self) -> SimdTag {
        self.highest_simd
    }

    pub(crate) fn fast_quotes(&self) -> bool {
        self.fast_quotes
    }

    pub(crate) fn fast_popcnt(&self) -> bool {
        self.fast_popcnt
    }
}

pub const SIMD_OVERRIDE_ENV_VARIABLE: &str = "RSONPATH_UNSAFE_FORCE_SIMD";

impl SimdConfiguration {
    fn try_parse(str: &str) -> Option<Self> {
        let parts = str.split(';').collect::<Vec<_>>();

        if parts.len() != 3 {
            return None;
        }

        let simd_slug = parts[0];
        let quotes_str = parts[1];
        let popcnt_str = parts[2];

        let simd = match simd_slug.to_ascii_lowercase().as_ref() {
            "nosimd" => Some(SimdTag::Nosimd),
            "sse2+" => Some(SimdTag::Sse2),
            "ssse3+" => Some(SimdTag::Ssse3),
            "avx2+" => Some(SimdTag::Avx2),
            _ => None,
        };
        let quotes = match quotes_str.to_ascii_lowercase().as_ref() {
            "fast_quotes" => Some(true),
            "slow_quotes" => Some(false),
            _ => None,
        };
        let popcnt = match popcnt_str.to_ascii_lowercase().as_ref() {
            "fast_popcnt" => Some(true),
            "slow_popcnt" => Some(false),
            _ => None,
        };

        Some(Self {
            highest_simd: simd?,
            fast_quotes: quotes?,
            fast_popcnt: popcnt?,
        })
    }
}

#[inline]
#[must_use]
pub fn configure() -> SimdConfiguration {
    if let Ok(simd) = std::env::var(SIMD_OVERRIDE_ENV_VARIABLE) {
        warn!(
            r#"cargo:warning=OVERRIDING SIMD SUPPORT TO "{}". THIS IS UNSAFE."#,
            simd
        );
        #[allow(clippy::expect_used)] // This is already an unsafe override, not expected to be used by users.
        return SimdConfiguration::try_parse(&simd).expect("invalid simd configuration override");
    }

    let highest_simd = if is_x86_feature_detected!("avx2") {
        SimdTag::Avx2
    } else if is_x86_feature_detected!("ssse3") {
        SimdTag::Ssse3
    } else if is_x86_feature_detected!("sse2") {
        SimdTag::Sse2
    } else {
        SimdTag::Nosimd
    };

    let fast_quotes = is_x86_feature_detected!("pclmulqdq");
    let fast_popcnt = is_x86_feature_detected!("popcnt");

    SimdConfiguration {
        highest_simd,
        fast_quotes,
        fast_popcnt,
    }
}

impl Display for SimdConfiguration {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let simd_slug = match self.highest_simd {
            SimdTag::Nosimd => "nosimd",
            SimdTag::Sse2 => "sse2+",
            SimdTag::Ssse3 => "ssse3+",
            SimdTag::Avx2 => "avx2+",
        };
        let quote_desc = if self.fast_quotes { "fast_quotes" } else { "slow_quotes" };
        let popcnt_desc = if self.fast_popcnt { "fast_popcnt" } else { "slow_popcnt" };

        write!(f, "{simd_slug};{quote_desc};{popcnt_desc}")
    }
}

cfg_if! {
    if #[cfg(target_arch = "x86_64")] {
        macro_rules! simd_dispatch {
            ($conf:expr => |$simd:ident| $b:block) => {
                {
                    let conf = $conf;

                    match conf.highest_simd() {
                        // AVX2 implies all other optimizations.
                        $crate::classification::simd::SimdTag::Avx2 => {
                            assert!(conf.fast_quotes());
                            assert!(conf.fast_popcnt());
                            let $simd = $crate::classification::simd::ResolvedSimd::<
                                $crate::classification::quotes::avx2_64::Constructor,
                                $crate::classification::structural::avx2_64::Constructor,
                                $crate::classification::depth::avx2_64::Constructor,
                                $crate::classification::memmem::avx2_64::Constructor,
                            >::new();
                            $b
                        }
                        $crate::classification::simd::SimdTag::Ssse3 => {
                            // In SSSE3 we need to check both advanced optimizations.
                            match (conf.fast_quotes(), conf.fast_popcnt()) {
                                (true, true) => {
                                    let $simd = $crate::classification::simd::ResolvedSimd::<
                                        $crate::classification::quotes::sse2_64::Constructor,
                                        $crate::classification::structural::ssse3_64::Constructor,
                                        $crate::classification::depth::sse2_64::Constructor,
                                        $crate::classification::memmem::sse2_64::Constructor,
                                    >::new();
                                    $b
                                }
                                (true, false) => {
                                    let $simd = $crate::classification::simd::ResolvedSimd::<
                                        $crate::classification::quotes::sse2_64::Constructor,
                                        $crate::classification::structural::ssse3_64::Constructor,
                                        $crate::classification::depth::nosimd::Constructor,
                                        $crate::classification::memmem::sse2_64::Constructor,
                                    >::new();
                                    $b
                                }
                                (false, true) => {
                                    let $simd = $crate::classification::simd::ResolvedSimd::<
                                        $crate::classification::quotes::nosimd::Constructor,
                                        $crate::classification::structural::ssse3_64::Constructor,
                                        $crate::classification::depth::sse2_64::Constructor,
                                        $crate::classification::memmem::sse2_64::Constructor,
                                    >::new();
                                    $b
                                }
                                (false, false) => {
                                    let $simd = $crate::classification::simd::ResolvedSimd::<
                                        $crate::classification::quotes::nosimd::Constructor,
                                        $crate::classification::structural::ssse3_64::Constructor,
                                        $crate::classification::depth::nosimd::Constructor,
                                        $crate::classification::memmem::sse2_64::Constructor,
                                    >::new();
                                    $b
                                }
                            }
                        }
                        $crate::classification::simd::SimdTag::Sse2 => {
                            // In SSE2 we need to check both advanced optimizations,
                            // and structural classifier is denied.
                            match (conf.fast_quotes(), conf.fast_popcnt()) {
                                (true, true) => {
                                    let $simd = $crate::classification::simd::ResolvedSimd::<
                                        $crate::classification::quotes::sse2_64::Constructor,
                                        $crate::classification::structural::nosimd::Constructor,
                                        $crate::classification::depth::sse2_64::Constructor,
                                        $crate::classification::memmem::sse2_64::Constructor,
                                    >::new();
                                    $b
                                }
                                (true, false) => {
                                    let $simd = $crate::classification::simd::ResolvedSimd::<
                                        $crate::classification::quotes::sse2_64::Constructor,
                                        $crate::classification::structural::nosimd::Constructor,
                                        $crate::classification::depth::nosimd::Constructor,
                                        $crate::classification::memmem::sse2_64::Constructor,
                                    >::new();
                                    $b
                                }
                                (false, true) => {
                                    let $simd = $crate::classification::simd::ResolvedSimd::<
                                        $crate::classification::quotes::nosimd::Constructor,
                                        $crate::classification::structural::nosimd::Constructor,
                                        $crate::classification::depth::sse2_64::Constructor,
                                        $crate::classification::memmem::sse2_64::Constructor,
                                    >::new();
                                    $b
                                }
                                (false, false) => {
                                    let $simd = $crate::classification::simd::ResolvedSimd::<
                                        $crate::classification::quotes::nosimd::Constructor,
                                        $crate::classification::structural::nosimd::Constructor,
                                        $crate::classification::depth::nosimd::Constructor,
                                        $crate::classification::memmem::sse2_64::Constructor,
                                    >::new();
                                    $b
                                }
                            }
                        }
                        // nosimd denies all optimizations.
                        $crate::classification::simd::SimdTag::Nosimd => {
                            let $simd = $crate::classification::simd::ResolvedSimd::<
                                $crate::classification::quotes::nosimd::Constructor,
                                $crate::classification::structural::nosimd::Constructor,
                                $crate::classification::depth::nosimd::Constructor,
                                $crate::classification::memmem::nosimd::Constructor,
                            >::new();
                            $b
                        }
                    }
                }
            };
        }
    }
    else if #[cfg(arch = "x86")] {
        macro_rules! simd_dispatch {
            ($conf:expr => |$simd:ident| $b:block) => {
                {
                    let conf = $conf;

                    match conf.highest_simd() {
                        // AVX2 implies all other optimizations.
                        $crate::classification::simd::SimdTag::Avx2 => {
                            assert!(conf.fast_quotes());
                            assert!(conf.fast_popcnt());
                            let $simd = $crate::classification::simd::ResolvedSimd::<
                                $crate::classification::quotes::avx2_32::Constructor,
                                $crate::classification::structural::avx2_32::Constructor,
                                $crate::classification::depth::avx2_32::Constructor,
                                $crate::classification::memmem::avx2_32::Constructor,
                            >::new();
                            $b
                        }
                        $crate::classification::simd::SimdTag::Ssse3 => {
                            // In SSSE3 we need to check both advanced optimizations.
                            match (conf.fast_quotes(), conf.fast_popcnt()) {
                                (true, true) => {
                                    let $simd = $crate::classification::simd::ResolvedSimd::<
                                        $crate::classification::quotes::sse2_32::Constructor,
                                        $crate::classification::structural::ssse3_32::Constructor,
                                        $crate::classification::depth::sse2_32::Constructor,
                                        $crate::classification::memmem::sse2_32::Constructor,
                                    >::new();
                                    $b
                                }
                                (true, false) => {
                                    let $simd = $crate::classification::simd::ResolvedSimd::<
                                        $crate::classification::quotes::sse2_32::Constructor,
                                        $crate::classification::structural::ssse3_32::Constructor,
                                        $crate::classification::depth::nosimd::Constructor,
                                        $crate::classification::memmem::sse2_32::Constructor,
                                    >::new();
                                    $b
                                }
                                (false, true) => {
                                    let $simd = $crate::classification::simd::ResolvedSimd::<
                                        $crate::classification::quotes::nosimd::Constructor,
                                        $crate::classification::structural::ssse3_32::Constructor,
                                        $crate::classification::depth::sse2_32::Constructor,
                                        $crate::classification::memmem::sse2_32::Constructor,
                                    >::new();
                                    $b
                                }
                                (false, false) => {
                                    let $simd = $crate::classification::simd::ResolvedSimd::<
                                        $crate::classification::quotes::nosimd::Constructor,
                                        $crate::classification::structural::ssse3_32::Constructor,
                                        $crate::classification::depth::nosimd::Constructor,
                                        $crate::classification::memmem::sse2_32::Constructor,
                                    >::new();
                                    $b
                                }
                            }
                        }
                        $crate::classification::simd::SimdTag::Sse2 => {
                            // In SSE2 we need to check both advanced optimizations,
                            // and structural classifier is denied.
                            match (conf.fast_quotes(), conf.fast_popcnt()) {
                                (true, true) => {
                                    let $simd = $crate::classification::simd::ResolvedSimd::<
                                        $crate::classification::quotes::sse2_32::Constructor,
                                        $crate::classification::structural::nosimd::Constructor,
                                        $crate::classification::depth::sse2_32::Constructor,
                                        $crate::classification::memmem::sse2_32::Constructor,
                                    >::new();
                                    $b
                                }
                                (true, false) => {
                                    let $simd = $crate::classification::simd::ResolvedSimd::<
                                        $crate::classification::quotes::sse2_32::Constructor,
                                        $crate::classification::structural::nosimd::Constructor,
                                        $crate::classification::depth::nosimd::Constructor,
                                        $crate::classification::memmem::sse2_32::Constructor,
                                    >::new();
                                    $b
                                }
                                (false, true) => {
                                    let $simd = $crate::classification::simd::ResolvedSimd::<
                                        $crate::classification::quotes::nosimd::Constructor,
                                        $crate::classification::structural::nosimd::Constructor,
                                        $crate::classification::depth::sse2_32::Constructor,
                                        $crate::classification::memmem::sse2_32::Constructor,
                                    >::new();
                                    $b
                                }
                                (false, false) => {
                                    let $simd = $crate::classification::simd::ResolvedSimd::<
                                        $crate::classification::quotes::nosimd::Constructor,
                                        $crate::classification::structural::nosimd::Constructor,
                                        $crate::classification::depth::nosimd::Constructor,
                                        $crate::classification::memmem::sse2_32::Constructor,
                                    >::new();
                                    $b
                                }
                            }
                        }
                        // nosimd denies all optimizations.
                        $crate::classification::simd::SimdTag::Nosimd => {
                            let $simd = $crate::classification::simd::ResolvedSimd::<
                                $crate::classification::quotes::nosimd::Constructor,
                                $crate::classification::structural::nosimd::Constructor,
                                $crate::classification::depth::nosimd::Constructor,
                                $crate::classification::memmem::nosimd::Constructor,
                            >::new();
                            $b
                        }
                    }
                }
            };
        }
    }
    else {
        macro_rules! simd_dispatch {
            ($conf:expr => |$simd:ident| $b:block) => {
                let $simd = $crate::classification::simd::ResolvedSimd::<
                    $crate::classification::quotes::nosimd::Constructor,
                    $crate::classification::structural::nosimd::Constructor,
                    $crate::classification::depth::nosimd::Constructor,
                    $crate::classification::memmem::nosimd::Constructor,
                >::new();
                $b
            };
        }
    }
}

pub(crate) use simd_dispatch;
