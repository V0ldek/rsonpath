//! Result types that can be returned by a JSONPath query engine.
use crate::debug;
use crate::lib::fmt::{self, Display};
use crate::lib::Vec;

/// Result that can be reported during query execution.
pub trait QueryResult: Default + Display + PartialEq {
    /// Report a match of the query.
    fn report(&mut self, index: usize);
}

/// Result informing on the number of values matching the executed query.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CountResult {
    count: usize,
}

impl CountResult {
    /// Number of values matched by the executed query.
    #[must_use]
    #[inline(always)]
    pub fn get(&self) -> usize {
        self.count
    }
}

impl Display for CountResult {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.count)
    }
}

impl QueryResult for CountResult {
    #[inline(always)]
    fn report(&mut self, _item: usize) {
        debug!("Reporting result: {_item}");
        self.count += 1;
    }
}

/// Query result containing all indices of colons that constitute a
/// match.
#[derive(Debug, Default, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg(feature = "alloc")]
pub struct IndexResult {
    indices: Vec<usize>,
}

#[cfg(feature = "alloc")]
impl IndexResult {
    /// Get indices of colons constituting matches of the query.
    #[must_use]
    #[inline(always)]
    pub fn get(&self) -> &[usize] {
        &self.indices
    }
}

#[cfg(feature = "alloc")]
impl From<IndexResult> for Vec<usize> {
    #[inline(always)]
    fn from(result: IndexResult) -> Self {
        result.indices
    }
}

#[cfg(feature = "alloc")]
impl Display for IndexResult {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.indices)
    }
}

#[cfg(feature = "alloc")]
impl QueryResult for IndexResult {
    #[inline(always)]
    fn report(&mut self, item: usize) {
        debug!("Reporting result: {item}");
        self.indices.push(item);
    }
}
