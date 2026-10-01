//! Directed-capacity maximum-flow service.

pub mod api;
pub mod core;
pub mod service;

pub use core::graph::{Capacity, EdgeSpec, Network, NetworkSpec, NodeId};

pub const MAX_NODES: usize = 10_000;
pub const MAX_EDGES: usize = 100_000;
