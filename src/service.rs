use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;
use serde_json::json;

use crate::algorithm::{minimum_path_cover, PathCover};
use crate::model::{LabeledGraph, LabeledGraphError};
use crate::topology::CycleEvidence;
use crate::witness::{verify_path_cover, WitnessCheck, WitnessFailure};
use crate::{model::Graph, APP_VERSION};

static RUN_COUNTER: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone)]
pub struct RunContext {
    pub run_id: String,
    pub input_id: String,
    pub case_id: Option<String>,
    pub version: String,
    pub steps: Vec<RunStep>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct RunStep {
    pub order: usize,
    pub name: String,
    pub status: String,
    pub detail: String,
}

impl RunContext {
    fn new(case_id: Option<String>, input_id: String) -> Self {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_nanos())
            .unwrap_or(0);
        let counter = RUN_COUNTER.fetch_add(1, Ordering::Relaxed);
        let run_number = (nanos as u64).wrapping_add(counter.wrapping_mul(0x9e37_79b9_7f4a_7c15));
        Self {
            run_id: format!("run-{run_number:016x}"),
            input_id,
            case_id,
            version: APP_VERSION.to_string(),
            steps: Vec::new(),
        }
    }

    pub fn record(&mut self, name: &str, status: &str, detail: String) {
        self.steps.push(RunStep {
            order: self.steps.len() + 1,
            name: name.to_string(),
            status: status.to_string(),
            detail,
        });
    }
}

#[derive(Debug, Clone)]
pub struct AnalysisSuccess {
    pub context: RunContext,
    pub graph: Graph,
    pub labels: Vec<String>,
    pub cover: PathCover,
    pub witness: WitnessCheck,
}

#[derive(Debug, Clone)]
pub struct AnalysisFailure {
    pub status: u16,
    pub category: String,
    pub message: String,
    pub context: RunContext,
    pub details: serde_json::Value,
}

pub fn analyze_named_graph(
    case_id: Option<String>,
    graph: LabeledGraph,
    max_vertices: usize,
    max_edges: usize,
) -> Result<AnalysisSuccess, AnalysisFailure> {
    let input_id = format!("in-{:016x}", stable_input_fingerprint(&graph));
    let mut context = RunContext::new(case_id.clone(), input_id);
    context.record(
        "receive",
        "ok",
        format!(
            "received {} candidate vertices and {} edges",
            graph.vertices.len(),
            graph.edges.len()
        ),
    );

    if graph.vertices.len() > max_vertices {
        return Err(fail(
            context,
            400,
            "INPUT_TOO_LARGE",
            "vertex count exceeds configured limit",
            json!({
                "limit": max_vertices,
                "actual": graph.vertices.len(),
                "field": "graph.vertices"
            }),
            "validate_limits",
        ));
    }
    if graph.edges.len() > max_edges {
        return Err(fail(
            context,
            400,
            "INPUT_TOO_LARGE",
            "edge count exceeds configured limit",
            json!({
                "limit": max_edges,
                "actual": graph.edges.len(),
                "field": "graph.edges"
            }),
            "validate_limits",
        ));
    }

    let (core_graph, labels) = graph
        .into_core_graph(max_vertices, max_edges)
        .map_err(|error| labeled_graph_error(error, &mut context))?;
    context.record(
        "validate_graph",
        "ok",
        "all vertex IDs are unique and all edge endpoints are known".to_string(),
    );

    let cover = match minimum_path_cover(&core_graph) {
        Ok(cover) => cover,
        Err(crate::algorithm::PathCoverError::Cycle(cycle)) => {
            return Err(cycle_failure(cycle, &labels, &mut context));
        }
    };
    context.record(
        "cycle_check",
        "ok",
        "DFS found no active-back edge, so the graph is acyclic".to_string(),
    );
    context.record(
        "bipartite_matching",
        "ok",
        format!(
            "Kuhn matching found {} matched edge(s); independent relation gives |V|-|M|={}",
            cover.matching.size, cover.minimum_path_count
        ),
    );
    context.record(
        "restore_paths",
        "ok",
        format!(
            "matching successors reconstructed {} path(s)",
            cover.paths.len()
        ),
    );

    let witness = verify_path_cover(&core_graph, &cover);
    if !witness.valid {
        let failures: Vec<String> = witness
            .failures
            .iter()
            .map(|failure| failure.category().to_string())
            .collect();
        return Err(fail(
            context,
            500,
            "WITNESS_INVALID",
            "the algorithm produced a path cover that failed independent witness checking",
            json!({ "failures": failures }),
            "witness_check",
        ));
    }
    context.record(
        "witness_check",
        "ok",
        "every vertex appears exactly once, all adjacencies and the cardinality relation hold"
            .to_string(),
    );

    Ok(AnalysisSuccess {
        context,
        graph: core_graph,
        labels,
        cover,
        witness,
    })
}

fn fail(
    mut context: RunContext,
    status: u16,
    category: &str,
    message: &str,
    details: serde_json::Value,
    step: &str,
) -> AnalysisFailure {
    context.record(step, "failed", message.to_string());
    AnalysisFailure {
        status,
        category: category.to_string(),
        message: message.to_string(),
        context,
        details,
    }
}

fn labeled_graph_error(error: LabeledGraphError, context: &mut RunContext) -> AnalysisFailure {
    let (category, message, details) = match error {
        LabeledGraphError::EmptyVertexId { index } => (
            "EMPTY_VERTEX_ID",
            "vertex ID must not be empty",
            json!({ "index": index }),
        ),
        LabeledGraphError::EmptyEdgeEndpoint { edge_index, side } => (
            "EMPTY_EDGE_ENDPOINT",
            "edge endpoint must not be empty",
            json!({ "edge_index": edge_index, "side": side.as_str() }),
        ),
        LabeledGraphError::DuplicateVertex {
            vertex,
            first_index,
            duplicate_index,
        } => (
            "DUPLICATE_VERTEX",
            "vertex IDs must be unique",
            json!({
                "vertex": vertex,
                "first_index": first_index,
                "duplicate_index": duplicate_index
            }),
        ),
        LabeledGraphError::UnknownEdgeEndpoint {
            edge_index,
            vertex,
            side,
        } => (
            "UNKNOWN_EDGE_ENDPOINT",
            "edge endpoint must be declared as a vertex",
            json!({
                "edge_index": edge_index,
                "vertex": vertex,
                "side": side.as_str()
            }),
        ),
        LabeledGraphError::SelfLoop { edge_index, vertex } => (
            "SELF_LOOP",
            "self loops are cycles and cannot belong to a DAG",
            json!({ "edge_index": edge_index, "vertex": vertex }),
        ),
        LabeledGraphError::DuplicateEdge {
            edge_index,
            from,
            to,
        } => (
            "DUPLICATE_EDGE",
            "duplicate directed edges are not accepted",
            json!({ "edge_index": edge_index, "from": from, "to": to }),
        ),
        LabeledGraphError::VertexLimitExceeded { limit, actual } => (
            "INPUT_TOO_LARGE",
            "vertex count exceeds configured limit",
            json!({ "limit": limit, "actual": actual }),
        ),
        LabeledGraphError::EdgeLimitExceeded { limit, actual } => (
            "INPUT_TOO_LARGE",
            "edge count exceeds configured limit",
            json!({ "limit": limit, "actual": actual }),
        ),
    };

    context.record("validate_graph", "failed", message.to_string());
    AnalysisFailure {
        status: 400,
        category: category.to_string(),
        message: message.to_string(),
        context: std::mem::replace(context, RunContext::new(None, String::new())),
        details,
    }
}

fn cycle_failure(
    cycle: CycleEvidence,
    labels: &[String],
    context: &mut RunContext,
) -> AnalysisFailure {
    let label = |id: usize| labels.get(id).cloned().unwrap_or_else(|| format!("v{id}"));
    let vertices: Vec<String> = cycle.vertices.iter().map(|id| label(*id)).collect();
    let edges: Vec<serde_json::Value> = cycle
        .edges
        .iter()
        .map(|(from, to)| json!({ "from": label(*from), "to": label(*to) }))
        .collect();
    context.record(
        "cycle_check",
        "failed",
        format!(
            "DFS found a directed cycle through {} vertex(vertices)",
            vertices.len()
        ),
    );
    AnalysisFailure {
        status: 422,
        category: "CYCLE_DETECTED".to_string(),
        message: "graph must be acyclic; the reported directed edges form cycle evidence"
            .to_string(),
        context: clone_context_without_replacement(context),
        details: json!({ "cycle_vertices": vertices, "cycle_edges": edges }),
    }
}

fn clone_context_without_replacement(context: &RunContext) -> RunContext {
    context.clone()
}

fn stable_input_fingerprint(graph: &LabeledGraph) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    hash = mix_u64(hash, graph.vertices.len() as u64);
    for vertex in &graph.vertices {
        hash = mix_bytes(hash, vertex.as_bytes());
    }
    hash = mix_u64(hash, graph.edges.len() as u64);
    for (from, to) in &graph.edges {
        hash = mix_bytes(hash, from.as_bytes());
        hash = mix_bytes(hash, to.as_bytes());
    }
    hash
}

fn mix_bytes(mut hash: u64, bytes: &[u8]) -> u64 {
    hash = mix_u64(hash, bytes.len() as u64);
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn mix_u64(mut hash: u64, value: u64) -> u64 {
    for shift in [0, 8, 16, 24, 32, 40, 48, 56] {
        hash ^= (value >> shift) & 0xff;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

#[allow(dead_code)]
fn assert_witness_failure_naming(failure: &WitnessFailure) -> &'static str {
    failure.category()
}
