use serde::Serialize;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static RUN_SEQUENCE: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct RunId(pub String);

impl RunId {
    pub fn new() -> Self {
        let millis = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock is after Unix epoch")
            .as_millis();
        let sequence = RUN_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        Self(format!("run-{millis}-{sequence:04}"))
    }
}

impl Default for RunId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for RunId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum ProgressEvent {
    SolveStarted {
        node_count: usize,
        edge_count: usize,
        source: usize,
        sink: usize,
        max_phases: Option<usize>,
    },
    PhaseStarted {
        phase: usize,
        total_flow: u64,
    },
    BfsResult {
        phase: usize,
        reachable_from_source: usize,
        sink_reachable: bool,
        level: Vec<Option<usize>>,
    },
    BlockingFlow {
        phase: usize,
        pushed_in_phase: u64,
        total_flow: u64,
        residual_caps: Vec<u64>,
    },
    CancellationRequested {
        phase: Option<usize>,
        reason: String,
    },
    CanceledAfterPhase {
        completed_phases: usize,
        total_flow: u64,
        reason: String,
    },
    Completed {
        phases: usize,
        total_flow: u64,
        source_side: Vec<usize>,
        sink_side: Vec<usize>,
    },
    WitnessChecked {
        valid: bool,
        reasons: Vec<String>,
    },
    Message {
        message: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct LogEntry {
    pub run_id: String,
    pub sequence: usize,
    pub event: ProgressEvent,
}

#[derive(Debug, Clone, Default)]
pub struct EventLog {
    run_id: RunId,
    entries: Vec<LogEntry>,
}

impl EventLog {
    pub fn new(run_id: RunId) -> Self {
        Self {
            run_id,
            entries: Vec::new(),
        }
    }

    pub fn run_id(&self) -> &RunId {
        &self.run_id
    }

    pub fn record(&mut self, event: ProgressEvent) {
        self.entries.push(LogEntry {
            run_id: self.run_id.0.clone(),
            sequence: self.entries.len() + 1,
            event,
        });
    }

    pub fn entries(&self) -> &[LogEntry] {
        &self.entries
    }
}
