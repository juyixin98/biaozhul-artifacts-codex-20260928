use std::collections::VecDeque;

use super::{
    initial_support, DecompositionEvent, DecompositionReport, EdgeTrussResult, WitnessAssignment,
};
use crate::graph::{QueueOrder, SimpleGraph};
use crate::witness::{assign_stage_witnesses, Witness};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DecompositionOrder {
    #[default]
    Initial,
    ReverseInitial,
    Lifo,
}

impl From<QueueOrder> for DecompositionOrder {
    fn from(order: QueueOrder) -> Self {
        match order {
            QueueOrder::Initial => Self::Initial,
            QueueOrder::ReverseInitial => Self::ReverseInitial,
            QueueOrder::Lifo => Self::Lifo,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TrussKernel {
    graph: SimpleGraph,
    support: Vec<usize>,
    in_queue: Vec<bool>,
}

impl TrussKernel {
    #[must_use]
    pub fn new(graph: SimpleGraph) -> Self {
        let support = initial_support(&graph);
        let edge_count = graph.edges().len();
        Self {
            graph,
            support,
            in_queue: vec![false; edge_count],
        }
    }

    #[must_use]
    pub fn graph(&self) -> &SimpleGraph {
        &self.graph
    }

    #[must_use]
    pub fn support(&self) -> &[usize] {
        &self.support
    }

    #[must_use]
    pub fn triangle_count(&self) -> usize {
        self.graph.triangles().len()
    }

    pub fn decompose(
        &mut self,
        input_edge_count: usize,
        duplicate_edge_count: usize,
        include_events: bool,
    ) -> DecompositionReport {
        self.decompose_with_order(
            DecompositionOrder::Initial,
            input_edge_count,
            duplicate_edge_count,
            include_events,
        )
    }

    pub fn decompose_with_order(
        &mut self,
        order: DecompositionOrder,
        input_edge_count: usize,
        duplicate_edge_count: usize,
        include_events: bool,
    ) -> DecompositionReport {
        let initial_total_support: usize = self.support.iter().sum();
        let initial_support_values = self.support.clone();
        let mut levels = vec![0; self.graph.edges().len()];
        let mut witness_id_by_edge = vec![0; self.graph.edges().len()];
        let mut witnesses: Vec<Witness> = Vec::new();
        let mut events = Vec::new();
        let mut next_witness_id = 0;

        let mut stage_level: usize = 2;
        loop {
            let threshold = stage_level.saturating_sub(2);
            let mut low_edges: Vec<usize> = self
                .graph
                .active_edges()
                .into_iter()
                .filter(|edge| !self.in_queue[*edge] && self.support[*edge] <= threshold)
                .collect();

            if matches!(order, DecompositionOrder::ReverseInitial) {
                low_edges.reverse();
            }

            let mut queue = VecDeque::new();
            let mut stack = Vec::new();
            for edge in low_edges {
                self.in_queue[edge] = true;
                if matches!(order, DecompositionOrder::Lifo) {
                    stack.push(edge);
                } else {
                    queue.push_back(edge);
                }
            }

            if queue.is_empty() && stack.is_empty() {
                if self.graph.active_edge_count() == 0 {
                    break;
                }
                stage_level += 1;
                continue;
            }

            let mut stage_assignments: Vec<WitnessAssignment> = Vec::new();

            loop {
                let edge = if matches!(order, DecompositionOrder::Lifo) {
                    stack.pop()
                } else {
                    queue.pop_front()
                };
                let Some(edge) = edge else { break };

                if !self.graph.is_active(edge) {
                    self.in_queue[edge] = false;
                    continue;
                }

                if self.support[edge] > threshold {
                    self.in_queue[edge] = false;
                    continue;
                }

                if stage_assignments.is_empty() {
                    stage_assignments =
                        assign_stage_witnesses(&self.graph, stage_level, next_witness_id);
                    next_witness_id += stage_assignments.len();
                }

                let witness_id = stage_assignments
                    .iter()
                    .find_map(|assignment| {
                        assignment
                            .subgraph
                            .edges
                            .iter()
                            .any(|snapshot| snapshot.edge_id == edge)
                            .then_some(assignment.witness_id)
                    })
                    .unwrap_or(0);

                let support_before_removal = self.support[edge];
                let endpoints = self.graph.edge(edge).expect("active edge exists");
                let mut affected_triangles = self
                    .graph
                    .triangles()
                    .into_iter()
                    .filter(|triangle| {
                        triangle.vertices().contains(&endpoints.u)
                            && triangle.vertices().contains(&endpoints.v)
                    })
                    .collect::<Vec<_>>();
                affected_triangles.sort_unstable();
                affected_triangles.dedup();

                self.graph.remove_edge(edge);
                self.in_queue[edge] = false;

                for triangle in &affected_triangles {
                    for (u, v) in triangle.edges() {
                        if (u, v) == (endpoints.u, endpoints.v)
                            || (u, v) == (endpoints.v, endpoints.u)
                        {
                            continue;
                        }
                        if let Some(other_edge) = self.graph.edge_id(u, v) {
                            if self.graph.is_active(other_edge) {
                                self.support[other_edge] =
                                    self.support[other_edge].saturating_sub(1);
                                if self.support[other_edge] <= threshold
                                    && !self.in_queue[other_edge]
                                {
                                    self.in_queue[other_edge] = true;
                                    if matches!(order, DecompositionOrder::Lifo) {
                                        stack.push(other_edge);
                                    } else {
                                        queue.push_back(other_edge);
                                    }
                                }
                            }
                        }
                    }
                }

                levels[edge] = stage_level;
                witness_id_by_edge[edge] = witness_id;
                if include_events {
                    events.push(DecompositionEvent {
                        stage_level,
                        edge_id: edge,
                        u: endpoints.u,
                        v: endpoints.v,
                        support_before_removal,
                        affected_triangle_count: affected_triangles.len(),
                        witness_id,
                    });
                }
            }

            for assignment in stage_assignments {
                witnesses.push(Witness {
                    witness_id: assignment.witness_id,
                    truss_level: stage_level,
                    support_threshold: threshold,
                    subgraph: assignment.subgraph,
                });
            }
        }

        let edge_results = self
            .graph
            .edges()
            .iter()
            .enumerate()
            .map(|(edge_id, endpoints)| EdgeTrussResult {
                edge_id,
                u: endpoints.u,
                v: endpoints.v,
                initial_support: initial_support_values[edge_id],
                truss_level: levels[edge_id],
                witness_id: witness_id_by_edge[edge_id],
            })
            .collect();

        DecompositionReport {
            vertex_count: self.graph.vertex_count(),
            input_edge_count,
            normalized_edge_count: self.graph.edges().len(),
            duplicate_edge_count,
            triangle_count: initial_total_support / 3,
            initial_total_support,
            levels: edge_results,
            witnesses,
            events,
        }
    }
}
