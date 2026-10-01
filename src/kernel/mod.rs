use serde_json::json;

use crate::graph::Graph;
use crate::witness::{check_induced_cycle, CycleWitness, WitnessCheck};

mod clique;
mod cycle;
mod peo;

pub use clique::{build_clique_tree, check_clique_tree, Clique, CliqueTree, CliqueTreeCheck};
pub use cycle::{find_cycle_from_peo_failure, DEFAULT_CYCLE_SEARCH_BUDGET};
pub use peo::{
    check_perfect_elimination_order, maximum_cardinality_order, McsStep, PeoFailure, PeoReport,
};

#[derive(Debug)]
pub enum AnalysisError {
    ResourceExhausted(String),
    ComputationFailure(String),
}

#[derive(Debug, Clone)]
pub struct TraceEvent {
    pub stage: String,
    pub status: String,
    pub message: String,
    pub details: serde_json::Value,
}

impl TraceEvent {
    fn new(
        stage: impl Into<String>,
        status: impl Into<String>,
        message: impl Into<String>,
        details: serde_json::Value,
    ) -> Self {
        Self {
            stage: stage.into(),
            status: status.into(),
            message: message.into(),
            details,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ChordalityRun {
    pub chordal: bool,
    pub mcs_selection_order: Vec<usize>,
    pub peo_order: Vec<usize>,
    pub mcs_steps: Vec<McsStep>,
    pub peo_report: PeoReport,
    pub witness: Option<CycleWitness>,
    pub witness_check: Option<WitnessCheck>,
    pub maximal_cliques: Vec<Clique>,
    pub maximum_clique_ids: Vec<usize>,
    pub clique_tree: Option<CliqueTree>,
    pub clique_tree_check: Option<CliqueTreeCheck>,
    pub trace: Vec<TraceEvent>,
}

#[derive(Debug, Clone)]
pub struct OrderVerification {
    pub peo_report: PeoReport,
    pub witness: Option<CycleWitness>,
    pub witness_check: Option<WitnessCheck>,
    pub trace: Vec<TraceEvent>,
}

pub fn analyze_chordality(
    graph: &Graph,
    cycle_search_budget: u64,
) -> Result<ChordalityRun, AnalysisError> {
    let mut trace = vec![TraceEvent::new(
        "input",
        "normalized",
        "normalized simple undirected graph is ready",
        json!({
            "vertex_count": graph.vertex_count(),
            "edge_count": graph.edge_count(),
            "duplicate_edge_count": graph.duplicate_edge_count(),
        }),
    )];

    let (mcs_selection_order, mcs_steps) = maximum_cardinality_order(graph);
    let peo_order: Vec<usize> = mcs_selection_order.iter().rev().copied().collect();
    trace.push(TraceEvent::new(
        "mcs",
        "complete",
        "maximum cardinality search selected a numbering order; its reverse is the PEO candidate",
        json!({
            "mcs_selection_order": mcs_selection_order,
            "perfect_elimination_candidate": peo_order,
            "step_count": mcs_steps.len(),
            "tie_breaker": "lowest normalized vertex index",
        }),
    ));

    let peo_report = check_perfect_elimination_order(graph, &peo_order);
    trace.push(TraceEvent::new(
        "peo_check",
        if peo_report.valid { "valid" } else { "invalid" },
        peo_report.reason.clone(),
        json!({
            "checked_order": peo_order,
            "first_failure_position": peo_report.first_failure.as_ref().map(|failure| failure.position),
        }),
    ));

    if !peo_report.valid {
        let failure = peo_report
            .first_failure
            .as_ref()
            .expect("invalid PEO report must carry a failure");
        let witness_option = find_cycle_from_peo_failure(graph, failure, cycle_search_budget)?;
        let Some(witness) = witness_option else {
            return Err(AnalysisError::ComputationFailure(
                "MCS PEO failure did not yield a cycle witness; this violates the search invariant"
                    .to_owned(),
            ));
        };
        let witness_check = check_induced_cycle(graph, &witness);
        {
            let cycle = &witness;
            let check = &witness_check;
            trace.push(TraceEvent::new(
                "witness",
                if check.valid { "verified" } else { "invalid" },
                check.reason.clone(),
                json!({
                    "cycle": cycle.cycle,
                    "cycle_length": cycle.cycle.len(),
                    "construction": "shortest path between a nonadjacent successor pair after blocking the failed vertex and all other later neighbors",
                }),
            ));
            if !check.valid {
                return Err(AnalysisError::ComputationFailure(
                    "constructed cycle failed independent induced-cycle verification".to_owned(),
                ));
            }
        }

        return Ok(ChordalityRun {
            chordal: false,
            mcs_selection_order,
            peo_order,
            mcs_steps,
            peo_report,
            witness: Some(witness),
            witness_check: Some(witness_check),
            maximal_cliques: Vec::new(),
            maximum_clique_ids: Vec::new(),
            clique_tree: None,
            clique_tree_check: None,
            trace,
        });
    }

    let maximal_cliques = extract_maximal_cliques(graph, &peo_order);
    let maximum_size = maximal_cliques
        .iter()
        .map(|clique| clique.vertices.len())
        .max()
        .unwrap_or(0);
    let maximum_clique_ids: Vec<usize> = maximal_cliques
        .iter()
        .filter(|clique| clique.vertices.len() == maximum_size)
        .map(|clique| clique.id)
        .collect();
    trace.push(TraceEvent::new(
        "maximal_cliques",
        "complete",
        "suffix-neighbor cliques were retained only when not contained in an earlier clique",
        json!({
            "count": maximal_cliques.len(),
            "maximum_size": maximum_size,
            "maximum_clique_ids": maximum_clique_ids,
            "cliques": maximal_cliques.iter().map(|clique| json!({
                "id": clique.id,
                "source_position": clique.source_position,
                "source_vertex": clique.source_vertex,
                "vertices": clique.vertices,
            })).collect::<Vec<_>>(),
        }),
    ));

    let clique_tree = build_clique_tree(graph, &maximal_cliques, &peo_order);
    let clique_tree_check = check_clique_tree(graph, &clique_tree);
    trace.push(TraceEvent::new(
        "clique_tree",
        if clique_tree_check.valid {
            "valid"
        } else {
            "invalid"
        },
        clique_tree_check.reason.clone(),
        json!({
            "root": clique_tree.root,
            "edges": clique_tree.edges.iter().map(|edge| json!({
                "parent": edge.parent,
                "child": edge.child,
                "separator": edge.separator,
            })).collect::<Vec<_>>(),
        }),
    ));

    if !clique_tree_check.valid {
        return Err(AnalysisError::ComputationFailure(
            "generated clique tree failed independent running-intersection validation".to_owned(),
        ));
    }

    Ok(ChordalityRun {
        chordal: true,
        mcs_selection_order,
        peo_order,
        mcs_steps,
        peo_report,
        witness: None,
        witness_check: None,
        maximal_cliques,
        maximum_clique_ids,
        clique_tree: Some(clique_tree),
        clique_tree_check: Some(clique_tree_check),
        trace,
    })
}

pub fn verify_custom_order(
    graph: &Graph,
    order: &[usize],
    cycle_search_budget: u64,
) -> Result<OrderVerification, AnalysisError> {
    let mut trace = vec![TraceEvent::new(
        "input",
        "normalized",
        "verifying a caller-supplied elimination order",
        json!({
            "vertex_count": graph.vertex_count(),
            "order_length": order.len(),
        }),
    )];

    let peo_report = check_perfect_elimination_order(graph, order);
    trace.push(TraceEvent::new(
        "peo_check",
        if peo_report.valid { "valid" } else { "invalid" },
        peo_report.reason.clone(),
        json!({
            "checked_order": order,
            "first_failure_position": peo_report.first_failure.as_ref().map(|failure| failure.position),
        }),
    ));

    if peo_report.valid {
        return Ok(OrderVerification {
            peo_report,
            witness: None,
            witness_check: None,
            trace,
        });
    }

    let Some(failure) = peo_report.first_failure.as_ref() else {
        return Ok(OrderVerification {
            peo_report,
            witness: None,
            witness_check: None,
            trace,
        });
    };

    let witness_option = find_cycle_from_peo_failure(graph, failure, cycle_search_budget)?;
    let (witness, witness_check) = if let Some(witness) = witness_option {
        let check = check_induced_cycle(graph, &witness);
        trace.push(TraceEvent::new(
            "witness",
            if check.valid { "verified" } else { "invalid" },
            check.reason.clone(),
            json!({
                "cycle": witness.cycle,
                "cycle_length": witness.cycle.len(),
            }),
        ));
        (Some(witness), Some(check))
    } else {
        trace.push(TraceEvent::new(
            "witness",
            "not_required",
            "a chordal graph can reject a tampered order without containing a chordless cycle; the missing-edge failure is independently checkable",
            json!({}),
        ));
        (None, None)
    };

    Ok(OrderVerification {
        peo_report,
        witness,
        witness_check,
        trace,
    })
}

pub use clique::extract_maximal_cliques;
