use super::error::ArrayIndexError;
use std::fmt::{self, Display, Formatter};

/// Array index to search for in a JSON document.
///
/// Represents a specific location from the front of the list in a json array.
/// Provides the [IETF-conforming index value](https://www.rfc-editor.org/rfc/rfc7493.html#section-2).  Values are \[0, (2^53)-1].
/// # Examples
///
/// ```
/// # use rsonpath_lib::query::NonNegativeArrayIndex;
///
/// let idx = NonNegativeArrayIndex::new(2);
///
/// assert_eq!(idx.get_index(), 2);
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Debug, PartialOrd, Ord)]
pub struct NonNegativeArrayIndex(u64);

/// The upper inclusive bound on index values.
pub(crate) const ARRAY_INDEX_ULIMIT: u64 = (1 << 53) - 1;

impl TryFrom<u64> for NonNegativeArrayIndex {
    type Error = ArrayIndexError;

    #[inline]
    fn try_from(value: u64) -> Result<Self, ArrayIndexError> {
        if value > ARRAY_INDEX_ULIMIT {
            Err(ArrayIndexError::ExceedsUpperLimitError(value.to_string()))
        } else {
            Ok(Self(value))
        }
    }
}

impl NonNegativeArrayIndex {
    /// Create a new search index from a u64.
    #[must_use]
    #[inline]
    pub const fn new(index: u64) -> Self {
        Self(index)
    }

    /// Create a new search index from a u64.
    #[must_use]
    #[inline]
    pub fn increment(&self) -> Self {
        NonNegativeArrayIndex::new(&self.0 + 1)
    }

    /// Return the index stored.
    #[must_use]
    #[inline]
    pub fn get_index(&self) -> u64 {
        self.0
    }

    /// Return a display object with a UTF8 representation of this index.
    #[must_use]
    #[inline(always)]
    pub fn display(&self) -> impl Display + '_ {
        self
    }
}

impl From<NonNegativeArrayIndex> for u64 {
    #[inline(always)]
    fn from(val: NonNegativeArrayIndex) -> Self {
        val.0
    }
}

impl Display for NonNegativeArrayIndex {
    #[inline]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::{NonNegativeArrayIndex, ARRAY_INDEX_ULIMIT};

    #[test]
    fn index_ulimit_sanity_check() {
        assert_eq!(9007199254740991, ARRAY_INDEX_ULIMIT);
    }

    #[test]
    fn index_ulimit_parse_check() {
        NonNegativeArrayIndex::try_from(ARRAY_INDEX_ULIMIT)
            .expect("Array index ulimit should be convertible.");

        NonNegativeArrayIndex::try_from(ARRAY_INDEX_ULIMIT + 1)
            .expect_err("Values in excess of array index ulimit should not be convertible.");
    }
}