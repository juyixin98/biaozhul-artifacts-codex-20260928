use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use super::errors::{KernelError, ResourceLimit};
use super::graph::Network;
use super::progress::{PathEdge, ProgressEvent, SolveLimits};

#[derive(Debug, Clone, Default)]
pub struct CancellationToken {
    canceled: Arc<AtomicBool>,
}

impl CancellationToken {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cancel(&self) {
        self.canceled.store(true, Ordering::SeqCst);
    }

    #[must_use]
    pub fn is_canceled(&self) -> bool {
        self.canceled.load(Ordering::SeqCst)
    }
}

#[derive(Debug, Clone, Copy)]
struct ResidualEdge {
    head: usize,
    capacity: u64,
    opposite: usize,
    original_index: usize,
    is_reverse: bool,
}

#[derive(Debug, Clone)]
struct DfsStep {
    from: usize,
    edge_id: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SolveStatistics {
    pub phases: usize,
    pub augmentations: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SolveResult {
    pub run_id: String,
    pub flow_value: u64,
    pub edge_flows: Vec<u64>,
    pub source_side: Option<Vec<usize>>,
    pub sink_side: Option<Vec<usize>>,
    pub statistics: SolveStatistics,
    #[serde(skip)]
    pub events: Vec<ProgressEvent>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SolveOutcome {
    Complete(SolveResult),
    Canceled(SolveResult),
}

impl SolveOutcome {
    #[must_use]
    pub fn result(&self) -> &SolveResult {
        match self {
            SolveOutcome::Complete(result) | SolveOutcome::Canceled(result) => result,
        }
    }

    #[must_use]
    pub fn is_optimal(&self) -> bool {
        matches!(self, SolveOutcome::Complete(_))
    }
}

pub struct Dinic {
    node_count: usize,
    source: usize,
    sink: usize,
    adjacency: Vec<Vec<usize>>,
    edges: Vec<ResidualEdge>,
    original_capacities: Vec<u64>,
    flows: Vec<u64>,
}

struct ProgressLog {
    events: Vec<ProgressEvent>,
}

impl ProgressLog {
    fn new() -> Self {
        Self { events: Vec::new() }
    }

    fn record(&mut self, event: ProgressEvent, observer: &dyn Fn(&ProgressEvent)) {
        observer(&event);
        self.events.push(event);
    }
}

impl Dinic {
    #[must_use]
    pub fn new(network: &Network) -> Self {
        let node_count = network.node_count();
        let mut solver = Self {
            node_count,
            source: network.source(),
            sink: network.sink(),
            adjacency: vec![Vec::new(); node_count],
            edges: Vec::with_capacity(network.edges().len() * 2),
            original_capacities: Vec::with_capacity(network.edges().len()),
            flows: vec![0; network.edges().len()],
        };

        for (index, edge) in network.edges().iter().enumerate() {
            solver.original_capacities.push(edge.capacity);
            let forward_id = solver.edges.len();
            let reverse_id = forward_id + 1;
            solver.adjacency[edge.from].push(forward_id);
            solver.adjacency[edge.to].push(reverse_id);
            solver.edges.push(ResidualEdge {
                head: edge.to,
                capacity: edge.capacity,
                opposite: reverse_id,
                original_index: index,
                is_reverse: false,
            });
            solver.edges.push(ResidualEdge {
                head: edge.from,
                capacity: 0,
                opposite: forward_id,
                original_index: index,
                is_reverse: true,
            });
        }

        solver
    }

    pub fn solve(
        &mut self,
        run_id: &str,
        limits: SolveLimits,
        cancellation: &CancellationToken,
        observer: &dyn Fn(&ProgressEvent),
    ) -> Result<SolveOutcome, KernelError> {
        let mut events = ProgressLog::new();
        let mut flow_value: u64 = 0;
        let mut phases = 0;
        let mut augmentations = 0;

        events.record(
            ProgressEvent::Started {
                run_id: run_id.to_string(),
                node_count: self.node_count,
                edge_count: self.flows.len(),
                source: self.source,
                sink: self.sink,
            },
            observer,
        );

        let mut final_levels = vec![None; self.node_count];

        loop {
            if cancellation.is_canceled() {
                return Ok(SolveOutcome::Canceled(self.make_result(
                    run_id,
                    flow_value,
                    phases,
                    augmentations,
                    None,
                    events,
                    "cancellation observed before next Dinic phase",
                    observer,
                )?));
            }

            phases += 1;
            if let Some(max_phases) = limits.max_phases {
                if phases > max_phases {
                    return Err(KernelError::ResourceExhausted(ResourceLimit {
                        name: "dinic_phases".to_string(),
                        limit: max_phases as u64,
                        actual: phases as u64,
                    }));
                }
            }

            events.record(
                ProgressEvent::PhaseBegin {
                    run_id: run_id.to_string(),
                    phase: phases,
                    flow_value,
                    reason: "building BFS level graph".to_string(),
                },
                observer,
            );

            let (levels, reached_sink) = self.bfs_levels();
            final_levels = levels.clone();
            events.record(
                ProgressEvent::LevelGraph {
                    run_id: run_id.to_string(),
                    phase: phases,
                    levels: levels.clone(),
                    reached_sink,
                    reason: if reached_sink {
                        "sink is reachable in the residual graph"
                    } else {
                        "sink is unreachable; current cut is a minimum cut"
                    }
                    .to_string(),
                },
                observer,
            );

            if !reached_sink {
                break;
            }

            let mut current_arc = vec![0usize; self.node_count];
            let mut phase_augmentations = 0;

            loop {
                if cancellation.is_canceled() {
                    return Ok(SolveOutcome::Canceled(self.make_result(
                        run_id,
                        flow_value,
                        phases,
                        augmentations,
                        None,
                        events,
                        "cancellation observed while searching the level graph",
                        observer,
                    )?));
                }

                let Some(steps) = self.find_path(&levels, &mut current_arc) else {
                    break;
                };
                let bottleneck = self.path_bottleneck(&steps)?;
                let residual_path = self.describe_path(&steps);

                for step in &steps {
                    self.push_flow(step.edge_id, bottleneck)?;
                }
                flow_value = flow_value
                    .checked_add(bottleneck)
                    .ok_or(KernelError::ArithmeticOverflow)?;
                augmentations += 1;
                phase_augmentations += 1;

                if let Some(max_augmentations) = limits.max_augmentations {
                    if augmentations > max_augmentations {
                        return Err(KernelError::ResourceExhausted(ResourceLimit {
                            name: "dinic_augmentations".to_string(),
                            limit: max_augmentations as u64,
                            actual: augmentations as u64,
                        }));
                    }
                }

                events.record(
                    ProgressEvent::PathFound {
                        run_id: run_id.to_string(),
                        phase: phases,
                        augmentation: augmentations,
                        residual_path,
                        bottleneck,
                        flow_value,
                        reason: "augmented along a shortest residual path".to_string(),
                    },
                    observer,
                );
            }

            events.record(
                ProgressEvent::PhaseEnd {
                    run_id: run_id.to_string(),
                    phase: phases,
                    augmentations: phase_augmentations,
                    flow_value,
                    reason: "current-arc DFS exhausted the blocking-flow search".to_string(),
                },
                observer,
            );
        }

        let (source_side, sink_side) = self.partition_from_levels(&final_levels);
        Ok(SolveOutcome::Complete(self.make_result(
            run_id,
            flow_value,
            phases,
            augmentations,
            Some((source_side, sink_side)),
            events,
            "final residual BFS certified flow value and cut capacity",
            observer,
        )?))
    }

    fn bfs_levels(&self) -> (Vec<Option<usize>>, bool) {
        let mut levels = vec![None; self.node_count];
        levels[self.source] = Some(0);
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(self.source);

        while let Some(node) = queue.pop_front() {
            let next_level = levels[node].expect("queued node has a level") + 1;
            for &edge_id in &self.adjacency[node] {
                let edge = &self.edges[edge_id];
                if edge.capacity > 0 && levels[edge.head].is_none() {
                    levels[edge.head] = Some(next_level);
                    if edge.head == self.sink {
                        return (levels, true);
                    }
                    queue.push_back(edge.head);
                }
            }
        }

        (levels, false)
    }

    fn find_path(
        &self,
        levels: &[Option<usize>],
        current_arc: &mut [usize],
    ) -> Option<Vec<DfsStep>> {
        let mut steps: Vec<DfsStep> = Vec::new();

        loop {
            let node = steps
                .last()
                .map_or(self.source, |step| self.edges[step.edge_id].head);

            if node == self.sink {
                return Some(steps);
            }
            if current_arc[node] >= self.adjacency[node].len() {
                if node == self.source {
                    return None;
                }
                let dead_step = steps.pop()?;
                current_arc[dead_step.from] += 1;
                continue;
            }

            let edge_index = current_arc[node];
            let edge_id = self.adjacency[node][edge_index];
            let edge = &self.edges[edge_id];
            let expected_level = levels[node].map(|level| level + 1);

            if edge.capacity == 0 || levels[edge.head] != expected_level {
                current_arc[node] += 1;
                continue;
            }

            steps.push(DfsStep {
                from: node,
                edge_id,
            });
        }
    }

    fn path_bottleneck(&self, steps: &[DfsStep]) -> Result<u64, KernelError> {
        steps
            .iter()
            .map(|step| self.edges[step.edge_id].capacity)
            .min()
            .ok_or(KernelError::InternalError {
                message: "Dinic produced an empty augmenting path".to_string(),
            })
    }

    fn describe_path(&self, steps: &[DfsStep]) -> Vec<PathEdge> {
        steps
            .iter()
            .map(|step| {
                let edge = &self.edges[step.edge_id];
                PathEdge {
                    from: step.from,
                    to: edge.head,
                    is_reverse: edge.is_reverse,
                    original_edge: edge.original_index,
                    residual_capacity: edge.capacity,
                }
            })
            .collect()
    }

    fn push_flow(&mut self, edge_id: usize, amount: u64) -> Result<(), KernelError> {
        let edge = &mut self.edges[edge_id];
        let original_index = edge.original_index;
        let is_reverse = edge.is_reverse;
        let opposite_id = edge.opposite;

        edge.capacity = edge
            .capacity
            .checked_sub(amount)
            .ok_or(KernelError::ArithmeticOverflow)?;
        self.edges[opposite_id].capacity = self.edges[opposite_id]
            .capacity
            .checked_add(amount)
            .ok_or(KernelError::ArithmeticOverflow)?;

        if is_reverse {
            self.flows[original_index] = self.flows[original_index]
                .checked_sub(amount)
                .ok_or(KernelError::ArithmeticOverflow)?;
        } else {
            self.flows[original_index] = self.flows[original_index]
                .checked_add(amount)
                .ok_or(KernelError::ArithmeticOverflow)?;
        }

        Ok(())
    }

    fn partition_from_levels(&self, levels: &[Option<usize>]) -> (Vec<usize>, Vec<usize>) {
        let mut source_side = Vec::new();
        let mut sink_side = Vec::new();
        for (node, level) in levels.iter().enumerate() {
            if level.is_some() {
                source_side.push(node);
            } else {
                sink_side.push(node);
            }
        }
        (source_side, sink_side)
    }

    fn make_result(
        &self,
        run_id: &str,
        flow_value: u64,
        phases: usize,
        augmentations: usize,
        cut: Option<(Vec<usize>, Vec<usize>)>,
        mut events: ProgressLog,
        final_reason: &str,
        observer: &dyn Fn(&ProgressEvent),
    ) -> Result<SolveResult, KernelError> {
        let (source_side, sink_side, final_event) = match cut {
            Some((source_side, sink_side)) => {
                let event = ProgressEvent::Completed {
                    run_id: run_id.to_string(),
                    flow_value,
                    phases,
                    augmentations,
                    reason: final_reason.to_string(),
                };
                (Some(source_side), Some(sink_side), event)
            }
            None => {
                let event = ProgressEvent::Canceled {
                    run_id: run_id.to_string(),
                    flow_value,
                    phases,
                    augmentations,
                    reason: final_reason.to_string(),
                };
                (None, None, event)
            }
        };
        events.record(final_event, observer);

        Ok(SolveResult {
            run_id: run_id.to_string(),
            flow_value,
            edge_flows: self.flows.clone(),
            source_side,
            sink_side,
            statistics: SolveStatistics {
                phases,
                augmentations,
            },
            events: events.events,
        })
    }
}
