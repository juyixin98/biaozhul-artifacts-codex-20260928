//! Transactional service boundary around the dynamic MSF core.

use std::sync::{Arc, Mutex};

use serde::Serialize;

use crate::core::{CoreError, DynamicMsf, StepEvent};
use crate::model::{Edge, EdgeId};
use crate::operations::Operation;
use crate::witness::{self, WitnessFailure, WitnessReport};

/// Shared graph service state.
#[derive(Clone)]
pub struct UpdateService {
    state: Arc<Mutex<DynamicMsf>>,
}

impl Default for UpdateService {
    fn default() -> Self {
        Self::new()
    }
}

impl UpdateService {
    /// Create a service around an empty graph.
    pub fn new() -> Self {
        Self {
            state: Arc::new(Mutex::new(DynamicMsf::new())),
        }
    }

    /// Read a complete published state snapshot.
    pub fn snapshot(&self) -> StateSnapshot {
        let state = self.state.lock().expect("MSF state mutex is not poisoned");
        StateSnapshot::from_core(&state)
    }

    /// Apply a batch to a private draft and publish it only if witness accepts.
    pub fn apply_checked(&self, batch: &[Operation]) -> Result<UpdateOutcome, ServiceError> {
        let mut draft = {
            let published = self.state.lock().expect("MSF state mutex is not poisoned");
            published.clone()
        };

        let version_from = draft.version();
        let steps = draft.apply_batch(batch).map_err(ServiceError::Core)?;
        let witness_report =
            witness::verify(draft.graph(), draft.forest_ids(), draft.total_weight());

        if !witness_report.accepted {
            return Err(ServiceError::WitnessRejected(WitnessRejection {
                rejected_version: draft.version(),
                report: witness_report,
                steps,
            }));
        }

        {
            let mut published = self.state.lock().expect("MSF state mutex is not poisoned");
            *published = draft;
        }

        let state = self.state.lock().expect("MSF state mutex is not poisoned");
        Ok(UpdateOutcome {
            version_from,
            version_to: state.version(),
            state: StateSnapshot::from_core(&state),
            steps,
            witness: WitnessSummary::from_report(&witness_report),
        })
    }
}

/// Complete published graph and forest state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct StateSnapshot {
    /// Complete-graph version.
    pub version: u64,
    /// Exact forest total.
    pub total_weight: i128,
    /// All current edges, including non-tree edges and self-loops.
    pub edges: Vec<Edge>,
    /// Selected forest edge IDs in stable order.
    pub forest_edge_ids: Vec<EdgeId>,
    /// Number of full-graph connected components, including isolated vertices.
    pub graph_component_count: usize,
}

impl StateSnapshot {
    fn from_core(core: &DynamicMsf) -> Self {
        let probe_report = witness::verify(core.graph(), core.forest_ids(), core.total_weight());
        Self {
            version: core.version(),
            total_weight: core.total_weight(),
            edges: core.graph().edges().cloned().collect(),
            forest_edge_ids: core.forest_ids().iter().cloned().collect(),
            graph_component_count: probe_report.graph_component_count,
        }
    }
}

/// Successful batch update result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct UpdateOutcome {
    /// Published version used as the starting point.
    pub version_from: u64,
    /// Newly published complete-graph version.
    pub version_to: u64,
    /// Complete state after the publish.
    pub state: StateSnapshot,
    /// Explainable core steps.
    pub steps: Vec<StepEvent>,
    /// Independent witness summary.
    pub witness: WitnessSummary,
}

/// Short witness summary included with successful responses.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct WitnessSummary {
    /// Always true in a successful outcome.
    pub accepted: bool,
    /// Independently recomputed forest total.
    pub expected_total_weight: i128,
    /// Independent full-graph component count.
    pub graph_component_count: usize,
}

impl WitnessSummary {
    fn from_report(report: &WitnessReport) -> Self {
        Self {
            accepted: report.accepted,
            expected_total_weight: report.expected_total_weight,
            graph_component_count: report.graph_component_count,
        }
    }
}

/// Witness rejection details. The version is not published.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct WitnessRejection {
    /// Draft version that was withheld.
    pub rejected_version: u64,
    /// Complete independent report.
    pub report: WitnessReport,
    /// Core steps that led to the rejected draft, useful for diagnosis.
    pub steps: Vec<StepEvent>,
}

/// Service-level error category.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ServiceError {
    /// Batch validation or core operation error.
    Core(CoreError),
    /// Core produced a draft that independent witness checks rejected.
    WitnessRejected(WitnessRejection),
}

impl ServiceError {
    /// Machine-readable error category.
    pub fn category(&self) -> &'static str {
        match self {
            ServiceError::Core(error) => match error {
                CoreError::EmptyBatch => "empty_batch",
                CoreError::UnknownEdge { .. } => "unknown_edge",
                CoreError::InvalidEdge { .. } => "invalid_edge",
            },
            ServiceError::WitnessRejected(_) => "witness_rejected",
        }
    }

    /// Concrete witness failures, when this is a witness rejection.
    pub fn witness_failures(&self) -> Vec<WitnessFailure> {
        match self {
            ServiceError::WitnessRejected(rejection) => rejection.report.failures.clone(),
            ServiceError::Core(_) => Vec::new(),
        }
    }
}

impl std::fmt::Display for ServiceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ServiceError::Core(error) => write!(f, "{error}"),
            ServiceError::WitnessRejected(rejection) => write!(
                f,
                "witness rejected draft version {} with {} failure(s)",
                rejection.rejected_version,
                rejection.report.failures.len()
            ),
        }
    }
}

impl std::error::Error for ServiceError {}
