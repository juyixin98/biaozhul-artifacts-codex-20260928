use serde::{Deserialize, Serialize};

use crate::core::dinic::SolveStatistics;
use crate::core::graph::NetworkSpec;
use crate::core::progress::ProgressEvent;
use crate::core::witness::OptimalityCertificate;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ComputeRequest {
    #[serde(flatten)]
    pub network: NetworkSpec,
    #[serde(default)]
    pub limits: SolveLimitsRequest,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SolveLimitsRequest {
    #[serde(default)]
    pub max_phases: Option<usize>,
    #[serde(default)]
    pub max_augmentations: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CreateJobRequest {
    #[serde(flatten)]
    pub compute: ComputeRequest,
    #[serde(default = "default_autostart")]
    pub autostart: bool,
}

fn default_autostart() -> bool {
    true
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FlowResultResponse {
    pub run_id: String,
    pub network: NetworkSpec,
    pub edges: Vec<EdgeFlowResponse>,
    pub flow_value: u64,
    pub certificate: OptimalityCertificate,
    pub statistics: SolveStatistics,
    pub progress: Vec<ProgressEvent>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EdgeFlowResponse {
    pub index: usize,
    pub from: usize,
    pub to: usize,
    pub capacity: u64,
    pub flow: u64,
}

impl EdgeFlowResponse {
    #[must_use]
    pub fn from_network(network: &NetworkSpec, flows: &[u64]) -> Vec<Self> {
        network
            .edges
            .iter()
            .zip(flows.iter())
            .enumerate()
            .map(|(index, (edge, flow))| Self {
                index,
                from: edge.from,
                to: edge.to,
                capacity: edge.capacity,
                flow: *flow,
            })
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum JobResponse {
    Queued {
        run_id: String,
        network: NetworkSpec,
    },
    Running {
        run_id: String,
        network: NetworkSpec,
        progress: Vec<ProgressEvent>,
    },
    CancellationRequested {
        run_id: String,
        network: NetworkSpec,
        progress: Vec<ProgressEvent>,
    },
    Completed(FlowResultResponse),
    Canceled(FlowResultResponse),
    Failed {
        run_id: String,
        network: NetworkSpec,
        category: String,
        error: serde_json::Value,
    },
}
