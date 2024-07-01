//! SIMD configuration and runtime dispatch.
//!
//! The core of our approach is the two macros: [`config_simd`] and [`dispatch_simd`].
//!
//! ## What?
//!
//! We need to strike a delicate balance between portable code and compiler optimizations.
//! All SIMD code should be maximally inlined. To that end we need `target_feature` annotated
//! functions, so that SIMD intrinsics are actually emitted, but these are hard barriers for
//! inlining if called from non-`target_feature` functions.
//!
//! The ideal would be to have a single `target_feature`-annotated entry point and then force
//! the compiler to inline everything there. This isn't that easy, because you cannot *really*
//! force the compiler to inline everything, and even if you could that can lead to inefficient compilation
//! (large functions are harder to optimize, large code size negatively impacts caching, etc.).
//!
//! On the other end of portability there's runtime checking of CPU capabilities.
//! That introduces an overhead and cannot be used everywhere, so in any case it'd have to be added
//! in specific places that then call `target_feature`-annotated functions.
//!
//! The [`multiversion`](https://calebzulawski.github.io/rust-simd-book/3.3-multiversion.html) crate
//! provides a near-ideal tradeoff, where one can annotate a function such that multiple definitions of it
//! are expanded with different `target_feature` sets, and an efficient, cached runtime check is performed
//! at entry to that function.
//!
//! For our crate we can do slightly better. The idea is to do the entire configuration of SIMD once and
//! upfront ([`configure`] producing [`SimdConfiguration`]) and save it. Then we can use [`config_simd`]
//! to create a [`Simd`] implementation that encapsulates all the relevant information in its type arguments.
//! An entry point function can take a generic `V: Simd` parameter so that the compiler specializes it
//! for all supported CPU capability configurations. Finally, to get the correct `target_feature` annotation
//! we do a similar thing to `multiversion`, but using a constant value from the `V` type, which allows
//! the compiler to optimize the check away when monomorphizing the function.
//!
//! An example idiomatic usage would be:
//!
//! ```rust,ignore
//! fn entry() -> Result<(), EngineError> {
//!   let configuration = simd::configure();
//!   config_simd!(configuration => |simd| {
//!       run(simd)
//!   })
//! }
//!
//! fn run<V: Simd>(simd: V) -> Result<(), EngineError> {
//!   dispatch_simd!(simd; simd => {
//!     fn<V: Simd>(simd: V) -> Result<(), EngineError>
//!     {
//!       // Actual implementation using SIMD capabilities from `simd`.
//!     }
//!   });
//! }
//! ```
//!
//! Assume for a second we only have 3 SIMD combinations:
//! - `+avx2,+pclmulqdq,+popcnt`
//! - `+sse2,+popcnt`
//! - `nosimd`
//!
//! The above code gets expanded to (approximately):
//!
//! ```rust,ignore
//! fn entry() -> Result<(), EngineError> {
//!   let configuration = simd::configure();
//!   {
//!     match configuration.highest_simd() {
//!       SimdTag::Avx2 => {
//!         let simd = ResolvedSimd::<
//!           quotes::avx2_64::Constructor,
//!           structural::avx2_64::Constructor,
//!           depth::avx2_64::Constructor,
//!           memmem::avx2_64::Constructor,
//!           simd::AVX2_PCLMULQDQ_POPCNT,
//!         >::new();
//!         run(simd)
//!       },
//!       SimdTag::Sse2 if conf.fast_popcnt() => {
//!         let simd = ResolvedSimd::<
//!           quotes::nosimd::Constructor,
//!           structural::nosimd::Constructor,
//!           depth::sse2_64::Constructor,
//!           memmem::sse2_64::Constructor,
//!           simd::SSE2_POPCNT,
//!         >::new();
//!         run(simd)
//!       },
//!       _ => {
//!         let simd = ResolvedSimd::<
//!           quotes::nosimd::Constructor,
//!           structural::nosimd::Constructor,
//!           depth::nosimd::Constructor,
//!           memmem::nosimd::Constructor,
//!           simd::NOSIMD,
//!         >::new();
//!         run(simd)
//!       },
//!     }
//!   }
//! }
//!
//! fn run<V: Simd>(simd: V) -> Result<(), EngineError> {
//!   #[target_feature(enable = "avx2")]
//!   #[target_feature(enable = "pclmulqdq")]
//!   #[target_feature(enable = "popcnt")]
//!   unsafe fn avx2_pclmulqdq_popcnt<V: Simd>(simd: V) -> Result<(), EngineError> {
//!     // Actual implementation using SIMD capabilities from `simd`.
//!   }
//!   #[target_feature(enable = "sse2")]
//!   #[target_feature(enable = "popcnt")]
//!   unsafe fn sse2_popcnt<V: Simd>(simd: V) -> Result<(), EngineError> {
//!     // Actual implementation using SIMD capabilities from `simd`.
//!   }
//!   unsafe fn nosimd<V: Simd>(simd: V) -> Result<(), EngineError> {
//!     // Actual implementation using SIMD capabilities from `simd`.
//!   }
//!   
//!   // SAFETY: depends on the provided SimdConfig, which cannot be incorrectly constructed.
//!   unsafe {
//!       match simd.dispatch_tag() {
//!           simd::AVX2_PCLMULQDQ_POPCNT => avx2_pclmulqdq_popcnt(simd),
//!           simd::SSE2_POPCNT => sse2_popcnt(simd),
//!           _ => nosimd(simd),
//!       }
//!   }
//! }
//! ```
//!
//! Now because all of the logic in the `dispatch_simd` is done over the `V` type constants,
//! the compiler will produce a `run` function for the three possible `ResolvedSimd` concrete
//! types used and then constant-fold the body to produce code equivalent to this (not valid Rust code):
//!
//! ```rust,ignore
//! fn run(simd: ResolvedSimd::<
//!           quotes::avx2_64::Constructor,
//!           structural::avx2_64::Constructor,
//!           depth::avx2_64::Constructor,
//!           memmem::avx2_64::Constructor,
//!           simd::AVX2_PCLMULQDQ_POPCNT,
//!         >) -> Result<(), EngineError> {
//!   #[target_feature(enable = "avx2")]
//!   #[target_feature(enable = "pclmulqdq")]
//!   #[target_feature(enable = "popcnt")]
//!   unsafe fn avx2_pclmulqdq_popcnt(simd: Avx2Simd = ResolvedSimd::<
//!           quotes::avx2_64::Constructor,
//!           structural::avx2_64::Constructor,
//!           depth::avx2_64::Constructor,
//!           memmem::avx2_64::Constructor,
//!           simd::AVX2_PCLMULQDQ_POPCNT,
//!         >) -> Result<(), EngineError> {
//!     // Actual implementation using SIMD capabilities from `simd`.
//!   }
//!
//!   unsafe { avx2_pclmulqdq_popcnt(simd) }
//! }
//!
//! fn run(simd: ResolvedSimd::<
//!           quotes::nosimd::Constructor,
//!           structural::nosimd::Constructor,
//!           depth::sse2_64::Constructor,
//!           memmem::sse2_64::Constructor,
//!           simd::SSE2_POPCNT,
//!         >) -> Result<(), EngineError> {
//!   #[target_feature(enable = "sse2")]
//!   #[target_feature(enable = "popcnt")]
//!   unsafe fn sse2_popcnt(simd: ResolvedSimd::<
//!           quotes::nosimd::Constructor,
//!           structural::nosimd::Constructor,
//!           depth::sse2_64::Constructor,
//!           memmem::sse2_64::Constructor,
//!           simd::SSE2_POPCNT,
//!         >) -> Result<(), EngineError> {
//!     // Actual implementation using SIMD capabilities from `simd`.
//!   }
//!   
//!   unsafe { sse2_popcnt(simd) }
//! }
//!
//! fn run(simd: ResolvedSimd::<
//!           quotes::nosimd::Constructor,
//!           structural::nosimd::Constructor,
//!           depth::nosimd::Constructor,
//!           memmem::nosimd::Constructor,
//!           simd::NOSIMD,
//!         >) -> Result<(), EngineError> {
//!   unsafe fn nosimd(simd: ResolvedSimd::<
//!           quotes::nosimd::Constructor,
//!           structural::nosimd::Constructor,
//!           depth::nosimd::Constructor,
//!           memmem::nosimd::Constructor,
//!           simd::NOSIMD,
//!         >) -> Result<(), EngineError> {
//!     // Actual implementation using SIMD capabilities from `simd`.
//!   }
//!   
//!   unsafe { nosimd(simd) }
//! }
//! ```
//!
//! The compiler is then free to optimize the inner functions fully, and the entire dispatch
//! happens once when `entry` is called.
//!
//! The config dispatch is done at start of the engine in one of the functions that run the executor.
//! The simd dispatch is put into the big entry points of the executor logic - `run_on_subtree`,
//! `run_head_skipping`, and `run_tail_skipping`. These are generally big enough to not be inlined by the compiler,
//! and long-running enough for that to not be an issue.
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
use std::{fmt::Display, marker::PhantomData};

/// All SIMD capabilities of the engine and classifier types.
pub(crate) trait Simd: Copy {
    /// The implementation of [`QuoteClassifiedIterator`] of this SIMD configuration.
    type QuotesClassifier<'i, I>: QuoteClassifiedIterator<'i, I, MaskType, BLOCK_SIZE> + InnerIter<I>
    where
        I: InputBlockIterator<'i, BLOCK_SIZE>;

    /// The implementation of [`StructuralIterator`] of this SIMD configuration.
    type StructuralClassifier<'i, I>: StructuralIterator<'i, I, Self::QuotesClassifier<'i, I>, MaskType, BLOCK_SIZE>
    where
        I: InputBlockIterator<'i, BLOCK_SIZE>;

    /// The implementation of [`DepthIterator`] of this SIMD configuration.
    type DepthClassifier<'i, I>: DepthIterator<'i, I, Self::QuotesClassifier<'i, I>, MaskType, BLOCK_SIZE>
    where
        I: InputBlockIterator<'i, BLOCK_SIZE>;

    /// The implementation of [`Memmem`] of this SIMD configuration.
    type MemmemClassifier<'i, 'b, 'r, I, R>: Memmem<'i, 'b, 'r, I, BLOCK_SIZE>
    where
        I: Input + 'i,
        <I as Input>::BlockIterator<'i, 'r, R, BLOCK_SIZE>: 'b,
        R: InputRecorder<<I as Input>::Block<'i, BLOCK_SIZE>> + 'r,
        'i: 'r;

    /// Get a unique descriptor of the enabled SIMD capabilities.
    ///
    /// The value should correspond to the `const`s defined in [`simd`](`self`),
    /// like [`AVX2_PCLMULQDQ_POPCNT`] or [`NOSIMD`].
    #[must_use]
    #[allow(dead_code)] // Not used in targets that have only one possible tag (NOSIMD for non-x86 for example).
    fn dispatch_tag(self) -> usize;

    /// Walk through the JSON document given by the `iter` and classify quoted sequences.
    #[must_use]
    fn classify_quoted_sequences<'i, I>(self, iter: I) -> Self::QuotesClassifier<'i, I>
    where
        I: InputBlockIterator<'i, BLOCK_SIZE>;

    /// Resume quote classification from an `iter` and, optionally, an already read
    /// block that will be used as the first block to classify.
    #[must_use]
    fn resume_quote_classification<'i, I>(
        self,
        iter: I,
        first_block: Option<I::Block>,
    ) -> ResumedQuoteClassifier<Self::QuotesClassifier<'i, I>, I::Block, MaskType, BLOCK_SIZE>
    where
        I: InputBlockIterator<'i, BLOCK_SIZE>;

    /// Walk through the JSON document quote-classified by `iter` and iterate over all
    /// occurrences of structural characters in it.
    #[must_use]
    fn classify_structural_characters<'i, I>(
        self,
        iter: Self::QuotesClassifier<'i, I>,
    ) -> Self::StructuralClassifier<'i, I>
    where
        I: InputBlockIterator<'i, BLOCK_SIZE>;

    /// Resume classification using a state retrieved from a previously
    /// used classifier via the [`stop`](StructuralIterator::stop) function.
    #[must_use]
    fn resume_structural_classification<'i, I>(
        self,
        state: ResumeClassifierState<'i, I, Self::QuotesClassifier<'i, I>, MaskType, BLOCK_SIZE>,
    ) -> Self::StructuralClassifier<'i, I>
    where
        I: InputBlockIterator<'i, BLOCK_SIZE>;

    /// Resume classification using a state retrieved from a previously
    /// used classifier via the [`stop`](DepthIterator::stop) function.
    #[must_use]
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

    /// Create a classifier that can look for occurrences of a key in the `iter`.
    #[must_use]
    fn memmem<'i, 'b, 'r, I, R>(
        self,
        input: &'i I,
        iter: &'b mut <I as Input>::BlockIterator<'i, 'r, R, BLOCK_SIZE>,
    ) -> Self::MemmemClassifier<'i, 'b, 'r, I, R>
    where
        I: Input,
        R: InputRecorder<<I as Input>::Block<'i, BLOCK_SIZE>>,
        'i: 'r;
}

pub(crate) struct ResolvedSimd<Q, S, D, M, const TARGET: usize> {
    phantom: PhantomData<(Q, S, D, M)>,
}

impl<Q, S, D, M, const TARGET: usize> Clone for ResolvedSimd<Q, S, D, M, TARGET> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<Q, S, D, M, const TARGET: usize> Copy for ResolvedSimd<Q, S, D, M, TARGET> {}

impl<Q, S, D, M, const TARGET: usize> ResolvedSimd<Q, S, D, M, TARGET> {
    pub(crate) fn new() -> Self {
        Self { phantom: PhantomData }
    }
}

impl<Q, S, D, M, const TARGET: usize> Simd for ResolvedSimd<Q, S, D, M, TARGET>
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
        <I as Input>::BlockIterator<'i, 'r, R, BLOCK_SIZE>: 'b,
        R: InputRecorder<<I as Input>::Block<'i, BLOCK_SIZE>> + 'r,
        'i: 'r;

    #[inline(always)]
    fn dispatch_tag(self) -> usize {
        TARGET
    }

    #[inline(always)]
    fn classify_quoted_sequences<'i, I>(self, iter: I) -> Self::QuotesClassifier<'i, I>
    where
        I: InputBlockIterator<'i, BLOCK_SIZE>,
    {
        Q::new(iter)
    }

    #[inline(always)]
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

    #[inline(always)]
    fn classify_structural_characters<'i, I>(
        self,
        iter: Self::QuotesClassifier<'i, I>,
    ) -> Self::StructuralClassifier<'i, I>
    where
        I: InputBlockIterator<'i, BLOCK_SIZE>,
    {
        S::new(iter)
    }

    #[inline(always)]
    fn resume_structural_classification<'i, I>(
        self,
        state: ResumeClassifierState<'i, I, Self::QuotesClassifier<'i, I>, MaskType, BLOCK_SIZE>,
    ) -> Self::StructuralClassifier<'i, I>
    where
        I: InputBlockIterator<'i, BLOCK_SIZE>,
    {
        S::resume(state)
    }

    #[inline(always)]
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

    #[inline(always)]
    fn memmem<'i, 'b, 'r, I, R>(
        self,
        input: &'i I,
        iter: &'b mut <I as Input>::BlockIterator<'i, 'r, R, BLOCK_SIZE>,
    ) -> Self::MemmemClassifier<'i, 'b, 'r, I, R>
    where
        I: Input,
        R: InputRecorder<<I as Input>::Block<'i, BLOCK_SIZE>>,
        'i: 'r,
    {
        M::memmem(input, iter)
    }
}

/// SIMD extension recognized by rsonpath.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum SimdTag {
    /// No SIMD capabilities detected.
    Nosimd,
    /// SSE2 detected.
    Sse2,
    /// SSSE3 detected.
    Ssse3,
    /// AVX2 detected.
    Avx2,
    /// AVX512 detected.
    Avx512
}

/// Runtime-detected SIMD configuration guiding how to construct a [`Simd`] implementation for the engine.
#[derive(Clone, Copy)]
pub(crate) struct SimdConfiguration {
    highest_simd: SimdTag,
    fast_quotes: bool,
    fast_popcnt: bool,
}

/// Name of the env variable that can be used to force a given [`SimdConfiguration`] to be used.
pub(crate) const SIMD_OVERRIDE_ENV_VARIABLE: &str = "RSONPATH_UNSAFE_FORCE_SIMD";

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
            "sse2" => Some(SimdTag::Sse2),
            "ssse3" => Some(SimdTag::Ssse3),
            "avx2" => Some(SimdTag::Avx2),
            "avx512" => Some(SimdTag::Avx512),
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

/// Detect available SIMD features and return the best possible [`SimdConfiguration`]
/// for the current system.
///
/// # Safety
/// If the [`SIMD_OVERRIDE_ENV_VARIABLE`] env variable is defined, it MUST be a valid SIMD
/// configuration for the current system. Otherwise, undefined behavior will follow.
/// For example, setting the value to enable AVX2 on a platform without AVX2 is unsound.
///
/// # Panics
/// If the [`SIMD_OVERRIDE_ENV_VARIABLE`] env variable is defined and does not contain a valid
/// SIMD configuration, an immediate panic is raised.
#[inline]
#[must_use]
pub(crate) fn configure() -> SimdConfiguration {
    if let Ok(simd) = std::env::var(SIMD_OVERRIDE_ENV_VARIABLE) {
        #[allow(clippy::expect_used)] // This is already an unsafe override, not expected to be used by users.
        return SimdConfiguration::try_parse(&simd).expect("invalid simd configuration override");
    }

    cfg_if! {
        if #[cfg(not(feature = "simd"))]
        {
            let highest_simd = SimdTag::Nosimd;
            let fast_quotes = false;
            let fast_popcnt = false;
        }
        else if #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            let highest_simd = if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
                SimdTag::Avx512
            } else if is_x86_feature_detected!("avx2") {
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
        }
        else
        {
            let highest_simd = SimdTag::Nosimd;
            let fast_quotes = false;
            let fast_popcnt = false;
        }
    }

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
            SimdTag::Sse2 => "sse2",
            SimdTag::Ssse3 => "ssse3",
            SimdTag::Avx2 => "avx2",
            SimdTag::Avx512 => "avx512",
        };
        let quote_desc = if self.fast_quotes { "fast_quotes" } else { "slow_quotes" };
        let popcnt_desc = if self.fast_popcnt { "fast_popcnt" } else { "slow_popcnt" };

        write!(f, "{simd_slug};{quote_desc};{popcnt_desc}")
    }
}

pub(crate) const NOSIMD: usize = 0;

cfg_if! { // FIXME co to
    if #[cfg(any(target_arch = "x86_64", target_arch = "x86"))] {
        pub(crate) const AVX512_PCLMULQDQ_POPCNT: usize = 1;
        pub(crate) const AVX2_PCLMULQDQ_POPCNT: usize = 2;
        pub(crate) const SSSE3_PCLMULQDQ_POPCNT: usize = 3;
        pub(crate) const SSSE3_PCLMULQDQ: usize = 4;
        pub(crate) const SSSE3_POPCNT: usize = 5;
        pub(crate) const SSSE3: usize = 6;
        pub(crate) const SSE2_PCLMULQDQ_POPCNT: usize = 7;
        pub(crate) const SSE2_PCLMULQDQ: usize = 8;
        pub(crate) const SSE2_POPCNT: usize = 9;
        pub(crate) const SSE2: usize = 10;

        macro_rules! dispatch_simd {
            ($simd:expr; $( $arg:expr ),* => fn $( $fn:tt )*) => {{
                #[target_feature(enable = "avx512f")]
                #[target_feature(enable = "avx512bw")]
                #[target_feature(enable = "pclmulqdq")]
                #[target_feature(enable = "popcnt")]
                unsafe fn avx512_pclmulqdq_popcnt $($fn)*
                #[target_feature(enable = "avx2")]
                #[target_feature(enable = "pclmulqdq")]
                #[target_feature(enable = "popcnt")]
                unsafe fn avx2_pclmulqdq_popcnt $($fn)*
                #[target_feature(enable = "ssse3")]
                #[target_feature(enable = "pclmulqdq")]
                #[target_feature(enable = "popcnt")]
                unsafe fn ssse3_pclmulqdq_popcnt $($fn)*
                #[target_feature(enable = "ssse3")]
                #[target_feature(enable = "pclmulqdq")]
                unsafe fn ssse3_pclmulqdq $($fn)*
                #[target_feature(enable = "ssse3")]
                #[target_feature(enable = "popcnt")]
                unsafe fn ssse3_popcnt $($fn)*
                #[target_feature(enable = "ssse3")]
                unsafe fn ssse3 $($fn)*
                #[target_feature(enable = "sse2")]
                #[target_feature(enable = "pclmulqdq")]
                #[target_feature(enable = "popcnt")]
                unsafe fn sse2_pclmulqdq_popcnt $($fn)*
                #[target_feature(enable = "sse2")]
                #[target_feature(enable = "pclmulqdq")]
                unsafe fn sse2_pclmulqdq $($fn)*
                #[target_feature(enable = "sse2")]
                #[target_feature(enable = "popcnt")]
                unsafe fn sse2_popcnt $($fn)*
                #[target_feature(enable = "sse2")]
                unsafe fn sse2 $($fn)*
                fn nosimd $($fn)*

                let simd = $simd;

                // SAFETY: depends on the provided SimdConfig, which cannot be incorrectly constructed.
                unsafe {
                    match simd.dispatch_tag() {
                        $crate::classification::simd::AVX512_PCLMULQDQ_POPCNT => avx512_pclmulqdq_popcnt($($arg),*),
                        $crate::classification::simd::AVX2_PCLMULQDQ_POPCNT => avx2_pclmulqdq_popcnt($($arg),*),
                        $crate::classification::simd::SSSE3_PCLMULQDQ_POPCNT => ssse3_pclmulqdq_popcnt($($arg),*),
                        $crate::classification::simd::SSSE3_PCLMULQDQ => ssse3_pclmulqdq($($arg),*),
                        $crate::classification::simd::SSSE3_POPCNT => ssse3_popcnt($($arg),*),
                        $crate::classification::simd::SSSE3 => ssse3($($arg),*),
                        $crate::classification::simd::SSE2_PCLMULQDQ_POPCNT => sse2_pclmulqdq_popcnt($($arg),*),
                        $crate::classification::simd::SSE2_PCLMULQDQ => sse2_pclmulqdq($($arg),*),
                        $crate::classification::simd::SSE2_POPCNT => sse2_popcnt($($arg),*),
                        $crate::classification::simd::SSE2 => sse2($($arg),*),
                        _ => nosimd($($arg),*),
                    }
                }
            }};
        }
    }
    else {
        macro_rules! dispatch_simd {
            ($simd:expr; $( $arg:expr ),* => fn $( $fn:tt )*) => {{
                fn nosimd $($fn)*
                nosimd($($arg),*)
            }};
        }
    }
}

cfg_if! {
    if #[cfg(target_arch = "x86_64")] {
        macro_rules! config_simd {
            ($conf:expr => |$simd:ident| $b:block) => {
                {
                    let conf = $conf;

                    match conf.highest_simd() {
                        // AVX512 implies all other optimizations.
                        $crate::classification::simd::SimdTag::Avx512 => {
                            assert!(conf.fast_quotes());
                            assert!(conf.fast_popcnt());
                            let $simd = $crate::classification::simd::ResolvedSimd::<
                                $crate::classification::quotes::avx512_64::Constructor,
                                $crate::classification::structural::avx512_64::Constructor,
                                $crate::classification::depth::avx512_64::Constructor,
                                $crate::classification::memmem::avx512_64::Constructor,
                                {$crate::classification::simd::AVX512_PCLMULQDQ_POPCNT},
                            >::new();
                            $b
                        }
                        // AVX2 implies all optimizations other than 512.
                        $crate::classification::simd::SimdTag::Avx2 => {
                            assert!(conf.fast_quotes());
                            assert!(conf.fast_popcnt());
                            let $simd = $crate::classification::simd::ResolvedSimd::<
                                $crate::classification::quotes::avx2_64::Constructor,
                                $crate::classification::structural::avx2_64::Constructor,
                                $crate::classification::depth::avx2_64::Constructor,
                                $crate::classification::memmem::avx2_64::Constructor,
                                {$crate::classification::simd::AVX2_PCLMULQDQ_POPCNT},
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
                                        {$crate::classification::simd::SSSE3_PCLMULQDQ_POPCNT},
                                    >::new();
                                    $b
                                }
                                (true, false) => {
                                    let $simd = $crate::classification::simd::ResolvedSimd::<
                                        $crate::classification::quotes::sse2_64::Constructor,
                                        $crate::classification::structural::ssse3_64::Constructor,
                                        $crate::classification::depth::nosimd::Constructor,
                                        $crate::classification::memmem::sse2_64::Constructor,
                                        {$crate::classification::simd::SSSE3_PCLMULQDQ},
                                    >::new();
                                    $b
                                }
                                (false, true) => {
                                    let $simd = $crate::classification::simd::ResolvedSimd::<
                                        $crate::classification::quotes::nosimd::Constructor,
                                        $crate::classification::structural::ssse3_64::Constructor,
                                        $crate::classification::depth::sse2_64::Constructor,
                                        $crate::classification::memmem::sse2_64::Constructor,
                                        {$crate::classification::simd::SSSE3_POPCNT},
                                    >::new();
                                    $b
                                }
                                (false, false) => {
                                    let $simd = $crate::classification::simd::ResolvedSimd::<
                                        $crate::classification::quotes::nosimd::Constructor,
                                        $crate::classification::structural::ssse3_64::Constructor,
                                        $crate::classification::depth::nosimd::Constructor,
                                        $crate::classification::memmem::sse2_64::Constructor,
                                        {$crate::classification::simd::SSSE3},
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
                                        {$crate::classification::simd::SSE2_PCLMULQDQ_POPCNT},
                                    >::new();
                                    $b
                                }
                                (true, false) => {
                                    let $simd = $crate::classification::simd::ResolvedSimd::<
                                        $crate::classification::quotes::sse2_64::Constructor,
                                        $crate::classification::structural::nosimd::Constructor,
                                        $crate::classification::depth::nosimd::Constructor,
                                        $crate::classification::memmem::sse2_64::Constructor,
                                        {$crate::classification::simd::SSE2_PCLMULQDQ},
                                    >::new();
                                    $b
                                }
                                (false, true) => {
                                    let $simd = $crate::classification::simd::ResolvedSimd::<
                                        $crate::classification::quotes::nosimd::Constructor,
                                        $crate::classification::structural::nosimd::Constructor,
                                        $crate::classification::depth::sse2_64::Constructor,
                                        $crate::classification::memmem::sse2_64::Constructor,
                                        {$crate::classification::simd::SSE2_POPCNT},
                                    >::new();
                                    $b
                                }
                                (false, false) => {
                                    let $simd = $crate::classification::simd::ResolvedSimd::<
                                        $crate::classification::quotes::nosimd::Constructor,
                                        $crate::classification::structural::nosimd::Constructor,
                                        $crate::classification::depth::nosimd::Constructor,
                                        $crate::classification::memmem::sse2_64::Constructor,
                                        {$crate::classification::simd::SSE2},
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
                                {$crate::classification::simd::NOSIMD}
                            >::new();
                            $b
                        }
                    }
                }
            };
        }
    }
    else if #[cfg(target_arch = "x86")] {
        macro_rules! config_simd {
            ($conf:expr => |$simd:ident| $b:block) => {
                {
                    let conf = $conf;

                    match conf.highest_simd() {
                        // AVX2 implies all other optimizations.
                        // AVX512 on x86 is yet to be implemented.
                        $crate::classification::simd::SimdTag::Avx2 | 
                        $crate::classification::simd::SimdTag::Avx512 => {
                            assert!(conf.fast_quotes());
                            assert!(conf.fast_popcnt());
                            let $simd = $crate::classification::simd::ResolvedSimd::<
                                $crate::classification::quotes::avx2_32::Constructor,
                                $crate::classification::structural::avx2_32::Constructor,
                                $crate::classification::depth::avx2_32::Constructor,
                                $crate::classification::memmem::avx2_32::Constructor,
                                {$crate::classification::simd::AVX2_PCLMULQDQ_POPCNT},
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
                                        {$crate::classification::simd::SSSE3_PCLMULQDQ_POPCNT}
                                    >::new();
                                    $b
                                }
                                (true, false) => {
                                    let $simd = $crate::classification::simd::ResolvedSimd::<
                                        $crate::classification::quotes::sse2_32::Constructor,
                                        $crate::classification::structural::ssse3_32::Constructor,
                                        $crate::classification::depth::nosimd::Constructor,
                                        $crate::classification::memmem::sse2_32::Constructor,
                                        {$crate::classification::simd::SSSE3_PCLMULQDQ}
                                    >::new();
                                    $b
                                }
                                (false, true) => {
                                    let $simd = $crate::classification::simd::ResolvedSimd::<
                                        $crate::classification::quotes::nosimd::Constructor,
                                        $crate::classification::structural::ssse3_32::Constructor,
                                        $crate::classification::depth::sse2_32::Constructor,
                                        $crate::classification::memmem::sse2_32::Constructor,
                                        {$crate::classification::simd::SSSE3_POPCNT}
                                    >::new();
                                    $b
                                }
                                (false, false) => {
                                    let $simd = $crate::classification::simd::ResolvedSimd::<
                                        $crate::classification::quotes::nosimd::Constructor,
                                        $crate::classification::structural::ssse3_32::Constructor,
                                        $crate::classification::depth::nosimd::Constructor,
                                        $crate::classification::memmem::sse2_32::Constructor,
                                        {$crate::classification::simd::SSSE3}
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
                                        {$crate::classification::simd::SSE2_PCLMULQDQ_POPCNT}
                                    >::new();
                                    $b
                                }
                                (true, false) => {
                                    let $simd = $crate::classification::simd::ResolvedSimd::<
                                        $crate::classification::quotes::sse2_32::Constructor,
                                        $crate::classification::structural::nosimd::Constructor,
                                        $crate::classification::depth::nosimd::Constructor,
                                        $crate::classification::memmem::sse2_32::Constructor,
                                        {$crate::classification::simd::SSE2_PCLMULQDQ}
                                    >::new();
                                    $b
                                }
                                (false, true) => {
                                    let $simd = $crate::classification::simd::ResolvedSimd::<
                                        $crate::classification::quotes::nosimd::Constructor,
                                        $crate::classification::structural::nosimd::Constructor,
                                        $crate::classification::depth::sse2_32::Constructor,
                                        $crate::classification::memmem::sse2_32::Constructor,
                                        {$crate::classification::simd::SSE2_POPCNT}
                                    >::new();
                                    $b
                                }
                                (false, false) => {
                                    let $simd = $crate::classification::simd::ResolvedSimd::<
                                        $crate::classification::quotes::nosimd::Constructor,
                                        $crate::classification::structural::nosimd::Constructor,
                                        $crate::classification::depth::nosimd::Constructor,
                                        $crate::classification::memmem::sse2_32::Constructor,
                                        {$crate::classification::simd::SSE2}
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
                                {$crate::classification::simd::NOSIMD}
                            >::new();
                            $b
                        }
                    }
                }
            };
        }
    }
    else {
        macro_rules! config_simd {
            ($conf:expr => |$simd:ident| $b:block) => {
                {
                    let conf = $conf;
                    assert_eq!(conf.highest_simd(), $crate::classification::simd::SimdTag::Nosimd);
                    assert!(!conf.fast_quotes());
                    assert!(!conf.fast_popcnt());
                    let $simd = $crate::classification::simd::ResolvedSimd::<
                        $crate::classification::quotes::nosimd::Constructor,
                        $crate::classification::structural::nosimd::Constructor,
                        $crate::classification::depth::nosimd::Constructor,
                        $crate::classification::memmem::nosimd::Constructor,
                        {$crate::classification::simd::NOSIMD},
                    >::new();
                    $b
                }
            };
        }
    }
}

pub(crate) use config_simd;
pub(crate) use dispatch_simd;
