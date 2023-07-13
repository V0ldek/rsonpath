//! [`QueryResult`] and [`Recorder`] implementation collecting the bytes of all matches.
//!
//! This is the heaviest recorder. It will copy all bytes of all matches into [`Vecs`](`Vec`).
//!
// There is number of invariants that are hard to enforce on the type level,
// and handling of Depth that should be properly error-handled by the engine, not here.
// Using `expect` here is idiomatic.
#![allow(clippy::expect_used)]
use super::*;
use crate::{debug, depth::Depth};
use std::{
    cell::RefCell,
    fmt::{self, Debug, Display},
    str::{self, Utf8Error},
};

/// [`QueryResult`] that collects all byte spans of matched values.
#[derive(Debug, Default, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodesResult {
    spans: Vec<Vec<u8>>,
}

impl NodesResult {
    /// Get slices of raw bytes of all matched nodes.
    #[must_use]
    #[inline(always)]
    pub fn get(&self) -> &[impl AsRef<[u8]>] {
        &self.spans
    }

    /// Iterate over all slices interpreted as valid UTF8.
    #[must_use]
    #[inline(always)]
    pub fn iter_as_utf8(&self) -> impl IntoIterator<Item = Result<&str, Utf8Error>> {
        self.spans.iter().map(|x| str::from_utf8(x))
    }

    /// Return the inner buffers consuming the result.
    #[must_use]
    #[inline(always)]
    pub fn into_inner(self) -> Vec<Vec<u8>> {
        self.spans
    }
}

impl From<NodesResult> for Vec<Vec<u8>> {
    #[inline(always)]
    fn from(result: NodesResult) -> Self {
        result.spans
    }
}

impl Display for NodesResult {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for span in &self.spans {
            let display = String::from_utf8_lossy(span);
            writeln!(f, "{display}")?;
        }

        Ok(())
    }
}

impl QueryResult for NodesResult {}

/// Recorder for [`NodesResult`].
pub struct NodesRecorder {
    internal: RefCell<InternalRecorder>,
}

impl InputRecorder for NodesRecorder {
    #[inline(always)]
    fn record_block_end(&self, new_block: &[u8]) {
        self.internal.borrow_mut().record_block(new_block)
    }
}

impl Recorder for NodesRecorder {
    type Result = NodesResult;

    #[inline]
    fn new() -> Self {
        Self {
            internal: RefCell::new(InternalRecorder::new()),
        }
    }

    #[inline]
    fn record_match(&self, idx: usize, depth: Depth, ty: MatchedNodeType) {
        debug!("Recording match at {idx}");
        self.internal.borrow_mut().record_match(idx, depth, ty)
    }

    #[inline]
    fn record_value_terminator(&self, idx: usize, depth: Depth) {
        self.internal.borrow_mut().record_value_terminator(idx, depth)
    }

    #[inline]
    fn finish(self) -> Self::Result {
        debug!("Finish recording.");
        self.internal.into_inner().finish()
    }
}

struct InternalRecorder {
    idx: usize,
    stack: Vec<PartialNode>,
    ready: Vec<PreparedNode>,
    finished: Vec<Vec<u8>>,
}

struct PartialNode {
    start_idx: usize,
    start_depth: Depth,
    buf: Vec<u8>,
    ty: MatchedNodeType,
}

struct PreparedNode {
    start_idx: usize,
    buf: Vec<u8>,
    end_idx: usize,
    ty: MatchedNodeType,
}

impl PartialNode {
    fn prepare(self, end_idx: usize) -> PreparedNode {
        PreparedNode {
            start_idx: self.start_idx,
            buf: self.buf,
            end_idx,
            ty: self.ty,
        }
    }
}

impl InternalRecorder {
    fn new() -> Self {
        Self {
            idx: 0,
            stack: vec![],
            ready: vec![],
            finished: vec![],
        }
    }

    fn record_block(&mut self, block: &[u8]) {
        mov(self.idx, &mut self.ready, &mut self.finished, block);

        for node in &mut self.stack {
            debug!("Continuing node: {node:?}, idx is {}", self.idx);
            Self::append_block(&mut node.buf, block, self.idx, node.start_idx)
        }

        self.idx += block.len();

        fn mov(idx: usize, ready: &mut Vec<PreparedNode>, bufs: &mut Vec<Vec<u8>>, block: &[u8]) {
            for mut top in ready.drain(..) {
                debug!("Final block for {top:?} starting at {idx}");
                InternalRecorder::append_final_block(&mut top.buf, block, idx, top.start_idx, top.end_idx);
                finalize_node(bufs, top);
            }
        }

        fn finalize_node(finished: &mut Vec<Vec<u8>>, mut node: PreparedNode) {
            debug!("Finalizing node: {node:?}");

            if node.ty == MatchedNodeType::Atomic {
                // Atomic nodes are finished when the next structural character is matched.
                // The buffer includes that character and all preceding whitespace.
                // We need to remove it before saving the result.
                let mut i = node.buf.len() - 2;
                while node.buf[i] == b' ' || node.buf[i] == b'\t' || node.buf[i] == b'\n' || node.buf[i] == b'\r' {
                    i -= 1;
                }

                node.buf.truncate(i + 1);
            }

            debug!("Committing node: {node:?}");
            finished.push(node.buf);
        }
    }

    fn append_final_block(dest: &mut Vec<u8>, src: &[u8], src_start: usize, read_start: usize, read_end: usize) {
        debug_assert!(read_end >= src_start);
        let in_block_start = if read_start > src_start {
            read_start - src_start
        } else {
            0
        };
        let in_block_end = read_end - src_start;

        dest.extend(&src[in_block_start..in_block_end]);
    }

    fn append_block(dest: &mut Vec<u8>, src: &[u8], src_start: usize, read_start: usize) {
        if read_start >= src_start + src.len() {
            return;
        }

        let to_extend = if read_start > src_start {
            let in_block_start = read_start - src_start;
            &src[in_block_start..]
        } else {
            src
        };

        dest.extend(to_extend);
    }

    fn record_match(&mut self, idx: usize, depth: Depth, ty: MatchedNodeType) {
        // In case of atomic types, any structural event that happens
        // at or above current depth marks the end. For complex types,
        // we first get the opening structural event, so the end is marked
        // by a depth decrease of 1.
        let start_depth = match ty {
            MatchedNodeType::Atomic => (depth + 1).expect("depth not above limit"),
            MatchedNodeType::Complex => depth,
        };

        let node = PartialNode {
            start_idx: idx,
            start_depth: depth,
            buf: vec![],
            ty,
        };

        debug!("New node {node:?}");
        self.stack.push(node);
    }

    #[inline]
    fn record_value_terminator(&mut self, idx: usize, depth: Depth) {
        debug!("Value terminator at {idx}, depth {depth}");
        while let Some(node) = self.stack.last() {
            if node.start_depth >= depth {
                debug!("Mark node {node:?} as ended at {}", idx + 1);
                let node = self.stack.pop().expect("last was Some, pop must succeed");
                let prepared_node = node.prepare(idx + 1);
                self.ready.push(prepared_node);
            } else {
                break;
            }
        }
    }

    fn finish(self) -> NodesResult {
        debug_assert!(self.stack.is_empty());

        NodesResult { spans: self.finished }
    }
}

impl Debug for PartialNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PartialNode")
            .field("start_idx", &self.start_idx)
            .field("start_depth", &self.start_depth)
            .field("ty", &self.ty)
            .field("buf", &str::from_utf8(&self.buf).unwrap_or("[invalid utf8]"))
            .finish()
    }
}

impl Debug for PreparedNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PartialNode")
            .field("start_idx", &self.start_idx)
            .field("end_idx", &self.end_idx)
            .field("ty", &self.ty)
            .field("buf", &str::from_utf8(&self.buf).unwrap_or("[invalid utf8]"))
            .finish()
    }
}
