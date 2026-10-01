use crate::error::AppError;
use crate::graph::{Capacity, GraphSpec, Network, Node};
use crate::logging::{EventLog, ProgressEvent, RunId};
use crate::witness::verify_flow_certificate;
pub use crate::witness::WitnessFailure;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SolveOptions {
    pub max_phases: Option<usize>,
}

impl Default for SolveOptions {
    fn default() -> Self {
        Self { max_phases: None }
    }
}

#[derive(Debug, Clone, Default)]
pub struct CancelToken {
    canceled: Arc<AtomicBool>,
}

impl CancelToken {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cancel(&self) {
        self.canceled.store(true, Ordering::SeqCst);
    }

    pub fn is_canceled(&self) -> bool {
        self.canceled.load(Ordering::SeqCst)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SolveStatus {
    Completed,
    Canceled,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EdgeFlow {
    pub edge_id: String,
    pub flow: Capacity,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Cut {
    pub source_side: Vec<Node>,
    pub sink_side: Vec<Node>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FlowCertificate {
    pub value: Capacity,
    pub edge_flows: Vec<EdgeFlow>,
    pub cut: Option<Cut>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SolveOutcome {
    pub run_id: String,
    pub status: SolveStatus,
    pub optimal: bool,
    pub certificate: FlowCertificate,
    pub feasibility_valid: bool,
    pub feasibility_reasons: Vec<String>,
    pub events: Vec<crate::logging::LogEntry>,
}

#[derive(Debug, Clone)]
pub enum SolveError {
    InvalidGraph(AppError),
    CapacityOverflow,
    WitnessCheckFailed(WitnessFailure),
}

impl SolveError {
    pub fn message(&self) -> String {
        match self {
            Self::InvalidGraph(error) => error.to_string(),
            Self::CapacityOverflow => "safe integer capacity arithmetic overflowed".to_string(),
            Self::WitnessCheckFailed(reason) => {
                format!("solver-produced certificate failed witness check: {reason:?}")
            }
        }
    }
}

struct Dinic<'a> {
    network: &'a mut Network,
    level: Vec<Option<usize>>,
    next_edge: Vec<usize>,
}

impl<'a> Dinic<'a> {
    fn new(network: &'a mut Network) -> Self {
        let node_count = network.node_count();
        Self {
            network,
            level: vec![None; node_count],
            next_edge: vec![0; node_count],
        }
    }

    fn bfs(&mut self) -> bool {
        for level in &mut self.level {
            *level = None;
        }
        let source = self.network.spec.source;
        let sink = self.network.spec.sink;
        self.level[source] = Some(0);
        let mut queue = VecDeque::from([source]);

        while let Some(node) = queue.pop_front() {
            let next_level = self.level[node].expect("queued node has a level") + 1;
            for &edge_index in &self.network.adjacency[node] {
                let edge = self.network.residual[edge_index];
                if edge.remaining > 0 && self.level[edge.to].is_none() {
                    self.level[edge.to] = Some(next_level);
                    queue.push_back(edge.to);
                }
            }
        }

        self.level[sink].is_some()
    }

    fn dfs(&mut self, node: Node, sink: Node, pushed: Capacity) -> Result<Capacity, SolveError> {
        if node == sink || pushed == 0 {
            return Ok(pushed);
        }

        while self.next_edge[node] < self.network.adjacency[node].len() {
            let edge_index = self.network.adjacency[node][self.next_edge[node]];
            let edge = self.network.residual[edge_index];
            let can_advance = self.level[edge.to] == self.level[node].map(|level| level + 1)
                && edge.remaining > 0;

            if !can_advance {
                self.next_edge[node] += 1;
                continue;
            }

            let available = pushed.min(edge.remaining);
            let sent = self.dfs(edge.to, sink, available)?;
            if sent == 0 {
                self.next_edge[node] += 1;
                continue;
            }

            let reverse_index = edge.reverse;
            self.network.residual[edge_index].remaining = self.network.residual[edge_index]
                .remaining
                .checked_sub(sent)
                .ok_or(SolveError::CapacityOverflow)?;
            self.network.residual[reverse_index].remaining = self.network.residual[reverse_index]
                .remaining
                .checked_add(sent)
                .ok_or(SolveError::CapacityOverflow)?;
            return Ok(sent);
        }

        Ok(0)
    }

    fn reachable_from_source(&self) -> Vec<bool> {
        let mut reachable = vec![false; self.network.node_count()];
        let mut queue = VecDeque::from([self.network.spec.source]);
        reachable[self.network.spec.source] = true;

        while let Some(node) = queue.pop_front() {
            for &edge_index in &self.network.adjacency[node] {
                let edge = self.network.residual[edge_index];
                if edge.remaining > 0 && !reachable[edge.to] {
                    reachable[edge.to] = true;
                    queue.push_back(edge.to);
                }
            }
        }

        reachable
    }
}

pub fn max_flow(
    spec: GraphSpec,
    options: SolveOptions,
    cancel: CancelToken,
    run_id: RunId,
) -> Result<SolveOutcome, SolveError> {
    spec.validate().map_err(SolveError::InvalidGraph)?;
    let mut network = Network::new(spec.clone()).map_err(SolveError::InvalidGraph)?;
    let mut log = EventLog::new(run_id.clone());
    log.record(ProgressEvent::SolveStarted {
        node_count: spec.node_count,
        edge_count: spec.edges.len(),
        source: spec.source,
        sink: spec.sink,
        max_phases: options.max_phases,
    });

    let mut total_flow: Capacity = 0;
    let mut phases = 0;
    let mut status = SolveStatus::Completed;

    loop {
        if options.max_phases == Some(phases) {
            status = SolveStatus::Canceled;
            log.record(ProgressEvent::CancellationRequested {
                phase: Some(phases),
                reason: "phase budget reached".to_string(),
            });
            break;
        }

        if cancel.is_canceled() {
            status = SolveStatus::Canceled;
            log.record(ProgressEvent::CancellationRequested {
                phase: Some(phases),
                reason: "client requested cancellation".to_string(),
            });
            break;
        }

        log.record(ProgressEvent::PhaseStarted {
            phase: phases + 1,
            total_flow,
        });
        let mut dinic = Dinic::new(&mut network);
        let sink_reachable = dinic.bfs();
        let reachable_count = dinic.level.iter().filter(|level| level.is_some()).count();
        log.record(ProgressEvent::BfsResult {
            phase: phases + 1,
            reachable_from_source: reachable_count,
            sink_reachable,
            level: dinic.level.clone(),
        });

        if !sink_reachable {
            break;
        }

        let mut pushed_in_phase: Capacity = 0;
        loop {
            for cursor in &mut dinic.next_edge {
                *cursor = 0;
            }
            let pushed = dinic.dfs(spec.source, spec.sink, Capacity::MAX)?;
            if pushed == 0 {
                break;
            }
            pushed_in_phase = pushed_in_phase
                .checked_add(pushed)
                .ok_or(SolveError::CapacityOverflow)?;
        }

        total_flow = total_flow
            .checked_add(pushed_in_phase)
            .ok_or(SolveError::CapacityOverflow)?;
        phases += 1;

        let residual_caps = network
            .residual
            .iter()
            .enumerate()
            .filter(|(index, _)| index % 2 == 0)
            .map(|(_, edge)| edge.remaining)
            .collect();
        log.record(ProgressEvent::BlockingFlow {
            phase: phases,
            pushed_in_phase,
            total_flow,
            residual_caps,
        });

        if cancel.is_canceled() {
            status = SolveStatus::Canceled;
            log.record(ProgressEvent::CanceledAfterPhase {
                completed_phases: phases,
                total_flow,
                reason: "client requested cancellation".to_string(),
            });
            break;
        }
    }

    let edge_flows = (0..spec.edges.len())
        .map(|index| EdgeFlow {
            edge_id: spec.edge_ids[index].clone(),
            flow: network.edge_flow(index),
        })
        .collect();

    let cut = if status == SolveStatus::Completed {
        let final_dinic = Dinic::new(&mut network);
        let reachable = final_dinic.reachable_from_source();
        let source_side: Vec<Node> = reachable
            .iter()
            .enumerate()
            .filter_map(|(node, is_reachable)| is_reachable.then_some(node))
            .collect();
        let sink_side: Vec<Node> = reachable
            .iter()
            .enumerate()
            .filter_map(|(node, is_reachable)| (!is_reachable).then_some(node))
            .collect();
        Some(Cut {
            source_side,
            sink_side,
        })
    } else {
        None
    };

    let certificate = FlowCertificate {
        value: total_flow,
        edge_flows,
        cut,
    };
    let witness = verify_flow_certificate(&spec, &certificate, status == SolveStatus::Completed);
    log.record(ProgressEvent::WitnessChecked {
        valid: witness.valid,
        reasons: witness.failure_strings(),
    });

    if !witness.valid {
        return Err(SolveError::WitnessCheckFailed(
            witness
                .failures
                .into_iter()
                .next()
                .expect("invalid witness has at least one failure"),
        ));
    }

    Ok(SolveOutcome {
        run_id: run_id.to_string(),
        status,
        optimal: status == SolveStatus::Completed,
        certificate,
        feasibility_valid: true,
        feasibility_reasons: Vec::new(),
        events: log.entries().to_vec(),
    })
}
