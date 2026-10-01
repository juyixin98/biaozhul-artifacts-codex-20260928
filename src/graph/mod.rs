mod input;
mod model;
mod normalize;

pub use input::*;
pub use model::*;
pub use normalize::*;

pub type Vertex = usize;
pub type EdgeId = usize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Endpoints {
    pub u: Vertex,
    pub v: Vertex,
}

impl Endpoints {
    /// Canonical order for an undirected simple graph edge.
    #[must_use]
    pub fn new(u: Vertex, v: Vertex) -> Self {
        if u <= v {
            Self { u, v }
        } else {
            Self { u: v, v: u }
        }
    }

    #[must_use]
    pub fn other(&self, vertex: Vertex) -> Option<Vertex> {
        if vertex == self.u {
            Some(self.v)
        } else if vertex == self.v {
            Some(self.u)
        } else {
            None
        }
    }
}

