use serde_json::{json, Value};

use crate::dto::{CheckReport, GraphSummary, OrderFailureDto, WitnessDto};
use crate::graph::Graph;
use crate::kernel::{ChordalityRun, OrderVerification, PeoFailure};
use crate::witness::WitnessCheck;

pub fn analysis_response(graph: &Graph, run: &ChordalityRun) -> Value {
    let witness = run.witness.as_ref().map(|cycle| WitnessDto {
        kind: "induced_cycle",
        cycle_length: cycle.cycle.len(),
        cycle: map_labels(graph, &cycle.cycle),
        cycle_edges: map_edge_pairs(graph, &cycle.edges()),
        induced: run.witness_check.as_ref().is_some_and(|check| check.valid),
        independent_check: map_check(run.witness_check.as_ref()),
    });

    let clique_tree = run.clique_tree.as_ref().map(|tree| {
        json!({
            "root": tree.root,
            "edges": tree.edges.iter().map(|edge| {
                json!({
                    "parent": edge.parent,
                    "child": edge.child,
                    "separator": map_labels(graph, &edge.separator),
                })
            }).collect::<Vec<_>>(),
        })
    });

    let clique_tree_check = run.clique_tree_check.as_ref().map(|check| {
        json!({
            "valid": check.valid,
            "reason": check.reason,
            "clique_cover_check": check_report(&check.clique_cover_check),
            "edge_cover_check": check_report(&check.edge_cover_check),
            "separator_check": check_report(&check.separator_check),
            "running_intersection_check": check_report(&check.running_intersection_check),
        })
    });

    json!({
        "chordal": run.chordal,
        "normalized_graph": graph_summary(graph),
        "maximum_cardinality_search": {
            "selection_order": map_labels(graph, &run.mcs_selection_order),
            "peo_candidate": map_labels(graph, &run.peo_order),
            "steps": run.mcs_steps.iter().map(|step| {
                json!({
                    "step": step.step,
                    "selected_vertex": graph.label(step.selected_vertex),
                    "weight_before": step.selected_weight,
                    "weights_before": step.weights_before,
                    "earlier_neighbors": map_labels(graph, &step.earlier_neighbors),
                })
            }).collect::<Vec<_>>(),
        },
        "perfect_elimination_order": {
            "order": map_labels(graph, &run.peo_order),
            "valid": run.peo_report.valid,
            "reason": run.peo_report.reason,
            "first_failure": run.peo_report.first_failure.as_ref().map(|failure| failure_dto(graph, failure)),
        },
        "chordless_cycle_witness": witness,
        "maximal_cliques": run.maximal_cliques.iter().map(|clique| {
            json!({
                "id": clique.id,
                "source_position": clique.source_position,
                "source_vertex": clique.source_vertex.map(|vertex| graph.label(vertex)),
                "vertices": map_labels(graph, &clique.vertices),
                "size": clique.vertices.len(),
            })
        }).collect::<Vec<_>>(),
        "maximum_clique": {
            "size": run
                .maximum_clique_ids
                .iter()
                .filter_map(|id| run.maximal_cliques.get(*id))
                .map(|clique| clique.vertices.len())
                .max()
                .unwrap_or(0),
            "clique_ids": run.maximum_clique_ids,
        },
        "clique_tree": clique_tree,
        "clique_tree_check": clique_tree_check,
        "algorithm": {
            "chordality_test": "maximum cardinality search plus successor-neighbor clique check",
            "witness_check": "independent induced-cycle boundary/chord verification",
            "clique_tree_check": "coverage, maximality, tree connectivity and running intersection verification",
        },
        "trace": trace_events(run),
    })
}

pub fn order_verification_response(
    graph: &Graph,
    order: &[usize],
    verification: &OrderVerification,
) -> Value {
    let witness = verification.witness.as_ref().map(|cycle| WitnessDto {
        kind: "induced_cycle",
        cycle_length: cycle.cycle.len(),
        cycle: map_labels(graph, &cycle.cycle),
        cycle_edges: map_edge_pairs(graph, &cycle.edges()),
        induced: verification
            .witness_check
            .as_ref()
            .is_some_and(|check| check.valid),
        independent_check: map_check(verification.witness_check.as_ref()),
    });

    json!({
        "normalized_graph": graph_summary(graph),
        "order_check": {
            "valid": verification.peo_report.valid,
            "checked_order": map_labels(graph, order),
            "first_failure": verification.peo_report.first_failure.as_ref().map(|failure| failure_dto(graph, failure)),
            "reason": verification.peo_report.reason,
        },
        "chordless_cycle_witness": witness,
        "trace": verification.trace.iter().map(|event| {
            json!({
                "stage": event.stage,
                "status": event.status,
                "message": event.message,
                "details": event.details,
            })
        }).collect::<Vec<_>>(),
    })
}

fn trace_events(run: &ChordalityRun) -> Vec<Value> {
    run.trace
        .iter()
        .map(|event| {
            json!({
                "stage": event.stage,
                "status": event.status,
                "message": event.message,
                "details": event.details,
            })
        })
        .collect()
}

fn graph_summary(graph: &Graph) -> GraphSummary {
    GraphSummary {
        vertex_count: graph.vertex_count(),
        edge_count: graph.edge_count(),
        duplicate_edge_count: graph.duplicate_edge_count(),
        vertices: graph.labels().to_vec(),
        normalized_edges: map_edge_pairs(graph, &graph.normalized_edges_indexed()),
    }
}

fn map_labels(graph: &Graph, ids: &[usize]) -> Vec<String> {
    ids.iter().map(|id| graph.label(*id).to_owned()).collect()
}

fn map_edge_pairs(graph: &Graph, edges: &[(usize, usize)]) -> Vec<[String; 2]> {
    edges
        .iter()
        .map(|(u, v)| [graph.label(*u).to_owned(), graph.label(*v).to_owned()])
        .collect()
}

fn map_check(check: Option<&WitnessCheck>) -> CheckReport {
    check.map(check_report).unwrap_or(CheckReport {
        valid: false,
        reason: "check was not produced".to_owned(),
    })
}

fn check_report(check: &WitnessCheck) -> CheckReport {
    CheckReport {
        valid: check.valid,
        reason: check.reason.clone(),
    }
}

fn failure_dto(graph: &Graph, failure: &PeoFailure) -> OrderFailureDto {
    OrderFailureDto {
        position: failure.position,
        vertex: graph.label(failure.vertex).to_owned(),
        later_neighbors: map_labels(graph, &failure.later_neighbors),
        missing_edges: map_edge_pairs(graph, &failure.missing_edges),
        failure_category: "successor_neighbors_not_clique",
    }
}
