use std::{fmt::Display, ops::BitOr};

/// Attributes that may be associated with a DFA's [`State`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u8)]
pub(crate) enum StateAttribute {
    /// Marks that the [`State`] is accepting.
    Accepting = 0x01,
    /// Marks that the [`State`] is rejecting,
    /// i.e. there is no possible path to any accepting state from it.
    Rejecting = 0x02,
    /// Marks that the [`State`] is _unitary_.
    /// A state is _unitary_ if it contains exactly one labelled transition
    /// and its fallback transition is [`Rejecting`](`StateAttribute::Rejecting`).
    Unitary = 0x04,
    /// Marks that the [`State`] contains some transition
    /// (labelled or fallback) to an [`Accepting`](`StateAttribute::Accepting`) state.
    TransitionsToAccepting = 0x08,
}

pub(crate) struct StateAttributesBuilder {
    attrs: StateAttributes,
}

/// A set of attributes that can be associated with a [`State`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct StateAttributes(u8);

impl StateAttributesBuilder {
    pub(crate) fn new() -> Self {
        Self {
            attrs: StateAttributes(0),
        }
    }

    pub(crate) fn accepting(self) -> Self {
        self.set(StateAttribute::Accepting)
    }

    pub(crate) fn rejecting(self) -> Self {
        self.set(StateAttribute::Rejecting)
    }

    pub(crate) fn unitary(self) -> Self {
        self.set(StateAttribute::Unitary)
    }

    pub(crate) fn transitions_to_accepting(self) -> Self {
        self.set(StateAttribute::TransitionsToAccepting)
    }

    pub(crate) fn build(self) -> StateAttributes {
        self.attrs
    }

    fn set(self, attr: StateAttribute) -> Self {
        Self {
            attrs: StateAttributes(self.attrs.0 | attr as u8),
        }
    }
}

impl From<StateAttributesBuilder> for StateAttributes {
    #[inline(always)]
    fn from(value: StateAttributesBuilder) -> Self {
        value.build()
    }
}

impl BitOr for StateAttributes {
    type Output = Self;

    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl StateAttributes {
    /// Marks that the [`State`] is accepting.
    pub const ACCEPTING: Self = Self(StateAttribute::Accepting as u8);
    /// Set with no attributes.
    pub const EMPTY: Self = Self(0);
    /// Marks that the [`State`] is rejecting,
    /// i.e. there is no possible path to any accepting state from it.
    pub const REJECTING: Self = Self(StateAttribute::Rejecting as u8);
    /// Marks that the [`State`] contains some transition
    /// (labelled or fallback) to an [`Accepting`](`StateAttributes::is_accepting`) state.
    pub const TRANSITIONS_TO_ACCEPTING: Self = Self(StateAttribute::TransitionsToAccepting as u8);
    /// Marks that the [`State`] is _unitary_.
    /// A state is _unitary_ if it contains exactly one labelled transition
    /// and its fallback transition is [`Rejecting`](`StateAttributes::is_rejecting`).
    pub const UNITARY: Self = Self(StateAttribute::Unitary as u8);

    /// Check if the the state is accepting.
    #[inline(always)]
    #[must_use]
    pub fn is_accepting(&self) -> bool {
        self.is_set(StateAttribute::Accepting)
    }

    /// Check if the state is rejecting,
    /// i.e. there is no possible path to any accepting state from it.
    #[inline(always)]
    #[must_use]
    pub fn is_rejecting(&self) -> bool {
        self.is_set(StateAttribute::Rejecting)
    }

    /// Marks that the [`State`] contains some transition
    /// (labelled or fallback) to an [`Accepting`](`StateAttributes::is_accepting`) state.
    #[inline(always)]
    #[must_use]
    pub fn has_transition_to_accepting(&self) -> bool {
        self.is_set(StateAttribute::TransitionsToAccepting)
    }

    /// Marks that the [`State`] is _unitary_.
    /// A state is _unitary_ if it contains exactly one labelled transition
    /// and its fallback transition is [`Rejecting`](`StateAttributes::is_rejecting`).
    #[inline(always)]
    #[must_use]
    pub fn is_unitary(&self) -> bool {
        self.is_set(StateAttribute::Unitary)
    }

    #[inline(always)]
    #[must_use]
    fn is_set(&self, attr: StateAttribute) -> bool {
        (self.0 & attr as u8) != 0
    }
}

/// State of an [`Automaton`](`super::Automaton`). Thin wrapper over a state's identifier.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct State(pub(crate) u8);

impl Display for State {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DFA({})", self.0)
    }
}
