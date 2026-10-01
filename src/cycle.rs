//! Iterative directed-cycle detection with concrete evidence.

use serde::Serialize;

use crate::model::{Graph, Vertex};

const UNVISITED: u8 = 0;
const ACTIVE: u8 = 1;
const DONE: u8 = 2;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CycleEvidence {
    /// Closed vertex sequence. It starts and ends at the same vertex.
    pub vertices: Vec<usize>,
    /// Directed edges belonging to `vertices` in order.
    pub edges: Vec<usize>,
}

#[derive(Debug, Clone)]
struct Frame {
    vertex: Vertex,
    next_neighbor: usize,
}

/// Returns evidence for the first deterministic cycle found, or `None` for a DAG.
pub fn find_directed_cycle(graph: &Graph) -> Option<CycleEvidence> {
    let n = graph.vertex_count();
    let mut state = vec![UNVISITED; n];

    for root in 0..n {
        if state[root] != UNVISITED {
            continue;
        }

        state[root] = ACTIVE;
        let mut stack = vec![Frame {
            vertex: root,
            next_neighbor: 0,
        }];

        while let Some(frame) = stack.last_mut() {
            let u = frame.vertex;

            if frame.next_neighbor >= graph.neighbors(u).len() {
                state[u] = DONE;
                stack.pop();
                continue;
            }

            let v = graph.neighbors(u)[frame.next_neighbor];
            frame.next_neighbor += 1;

            if state[v] == ACTIVE {
                let start = stack
                    .iter()
                    .position(|item| item.vertex == v)
                    .expect("active vertex must be represented on the DFS stack");
                let vertices: Vec<usize> = stack[start..]
                    .iter()
                    .map(|item| item.vertex)
                    .chain(std::iter::once(v))
                    .collect();
                let edges = (0..vertices.len() - 1).collect();
                return Some(CycleEvidence { vertices, edges });
            }

            if state[v] == UNVISITED {
                state[v] = ACTIVE;
                stack.push(Frame {
                    vertex: v,
                    next_neighbor: 0,
                });
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Edge, GraphInput};

    fn graph(n: usize, edges: &[(usize, usize)]) -> Graph {
        Graph::new(
            &GraphInput {
                vertices: n,
                edges: edges.iter().map(|&(from, to)| Edge { from, to }).collect(),
            },
            100,
        )
        .expect("test graph")
    }

    #[test]
    fn accepts_diamond_and_disconnected_dag() {
        let graph = graph(6, &[(0, 1), (0, 2), (1, 3), (2, 3), (4, 5)]);

        assert!(find_directed_cycle(&graph).is_none());
    }

    #[test]
    fn reports_self_loop_evidence() {
        let graph = graph(1, &[(0, 0)]);

        assert_eq!(
            find_directed_cycle(&graph),
            Some(CycleEvidence {
                vertices: vec![0, 0],
                edges: vec![0]
            })
        );
    }

    #[test]
    fn reports_closed_directed_cycle() {
        let graph = graph(4, &[(0, 1), (1, 2), (2, 3), (3, 1)]);

        let evidence = find_directed_cycle(&graph).expect("cycle");
        assert_eq!(
            *evidence.vertices.first().unwrap(),
            *evidence.vertices.last().unwrap()
        );
        assert_eq!(evidence.edges.len(), evidence.vertices.len() - 1);
        for window in evidence.vertices.windows(2) {
            assert!(graph.has_edge(window[0], window[1]));
        }
        assert_eq!(evidence.vertices, vec![1, 2, 3, 1]);
    }
}
