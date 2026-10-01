use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum ProgressEvent {
    Started {
        run_id: String,
        node_count: usize,
        edge_count: usize,
        source: usize,
        sink: usize,
    },
    PhaseBegin {
        run_id: String,
        phase: usize,
        flow_value: u64,
        reason: String,
    },
    LevelGraph {
        run_id: String,
        phase: usize,
        levels: Vec<Option<usize>>,
        reached_sink: bool,
        reason: String,
    },
    PathFound {
        run_id: String,
        phase: usize,
        augmentation: usize,
        residual_path: Vec<PathEdge>,
        bottleneck: u64,
        flow_value: u64,
        reason: String,
    },
    PhaseEnd {
        run_id: String,
        phase: usize,
        augmentations: usize,
        flow_value: u64,
        reason: String,
    },
    Completed {
        run_id: String,
        flow_value: u64,
        phases: usize,
        augmentations: usize,
        reason: String,
    },
    Canceled {
        run_id: String,
        flow_value: u64,
        phases: usize,
        augmentations: usize,
        reason: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PathEdge {
    pub from: usize,
    pub to: usize,
    /// `true` means this residual step traverses a reverse edge and removes flow.
    pub is_reverse: bool,
    /// Index of the original edge whose forward/reverse residual edge was used.
    pub original_edge: usize,
    pub residual_capacity: u64,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct SolveLimits {
    pub max_phases: Option<usize>,
    pub max_augmentations: Option<usize>,
}

impl SolveLimits {
    #[must_use]
    pub fn unbounded() -> Self {
        Self::default()
    }
}
