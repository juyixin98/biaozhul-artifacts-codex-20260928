#[path = "reference/mod.rs"]
mod reference;

use ktruss_backend::algorithm::{
    delete_edge_once_per_triangle, DecompositionOrder, TrussKernel,
};
use ktruss_backend::graph::{
    normalize_input, GraphInput, InputEdge, MultiEdgePolicy, SimpleGraph,
};
use ktruss_backend::witness::check_report_witnesses;

fn edges(values: &[(usize, usize)]) -> Vec<InputEdge> {
    values
        .iter()
        .map(|(u, v)| InputEdge { u: *u, v: *v })
        .collect()
}

fn graph(vertex_count: usize, values: &[(usize, usize)]) -> SimpleGraph {
    let endpoints = values
        .iter()
        .map(|(u, v)| ktruss_backend::graph::Endpoints::new(*u, *v));
    SimpleGraph::from_endpoints(vertex_count, endpoints)
}

fn level_map(
    graph: &SimpleGraph,
    report: &ktruss_backend::algorithm::DecompositionReport,
) -> std::collections::BTreeMap<(usize, usize), usize> {
    report
        .levels
        .iter()
        .map(|edge| {
            let endpoints = graph.edge(edge.edge_id).unwrap();
            ((endpoints.u, endpoints.v), edge.truss_level)
        })
        .collect()
}

#[test]
fn exhaustive_small_graphs_match_triple_enumeration_and_full_recompute() {
    for vertex_count in 0..=4usize {
        for reference_graph in reference::all_graphs(vertex_count) {
            let reference_levels = reference_graph.clone().truss_levels();
            let reference_support = reference_graph.support();
            let reference_edges = reference_graph.edges();
            let mut production = graph(vertex_count, &reference_edges);
            let initial_support = ktruss_backend::algorithm::initial_support(&production);

            assert_eq!(production.triangles().len(), reference_graph.triangles().len());
            for (edge_id, endpoints) in production.edges().iter().enumerate() {
                let key = (endpoints.u, endpoints.v);
                assert_eq!(initial_support[edge_id], reference_support[&key]);
            }

            let baseline = {
                let mut kernel = TrussKernel::new(production.clone());
                kernel.decompose_with_order(
                    DecompositionOrder::Initial,
                    reference_edges.len(),
                    0,
                    false,
                )
            };
            assert_eq!(level_map(&production, &baseline), reference_levels);
            check_report_witnesses(&baseline).unwrap();

            for order in [
                DecompositionOrder::ReverseInitial,
                DecompositionOrder::Lifo,
            ] {
                let mut kernel = TrussKernel::new(production.clone());
                let report =
                    kernel.decompose_with_order(order, reference_edges.len(), 0, true);
                assert_eq!(level_map(&production, &report), reference_levels);
                check_report_witnesses(&report).unwrap();
            }

        }
    }
}

#[test]
fn zero_triangle_graph_assigns_level_two_and_threshold_zero_witnesses() {
    let values = vec![(0, 1), (1, 2), (2, 3)];
    let production = graph(5, &values);
    let mut kernel = TrussKernel::new(production.clone());
    let report = kernel.decompose(values.len(), 0, true);

    assert_eq!(report.triangle_count, 0);
    assert_eq!(report.initial_total_support, 0);
    for edge in &report.levels {
        assert_eq!(edge.initial_support, 0);
        assert_eq!(edge.truss_level, 2);
    }
    assert_eq!(report.witnesses.len(), 1);
    assert_eq!(report.witnesses[0].truss_level, 2);
    assert_eq!(report.witnesses[0].support_threshold, 0);
    assert_eq!(report.witnesses[0].subgraph.edges.len(), 3);
    check_report_witnesses(&report).unwrap();
}

#[test]
fn shared_edge_many_triangles_has_exact_supports_and_level_three() {
    let values = vec![(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (0, 4), (1, 4)];
    let production = graph(5, &values);
    let mut kernel = TrussKernel::new(production.clone());
    let report = kernel.decompose(values.len(), 0, true);

    assert_eq!(report.triangle_count, 3);
    let by_edge = level_map(&production, &report);
    assert!(by_edge.values().all(|level| *level == 3));
    assert_eq!(
        report
            .levels
            .iter()
            .find(|edge| edge.u == 0 && edge.v == 1)
            .unwrap()
            .initial_support,
        3
    );
    for edge in report.levels.iter().filter(|edge| !(edge.u == 0 && edge.v == 1)) {
        assert_eq!(edge.initial_support, 1);
    }
    assert_eq!(report.witnesses[0].truss_level, 3);
    assert_eq!(report.witnesses[0].support_threshold, 1);
    check_report_witnesses(&report).unwrap();
}

#[test]
fn clique_connected_by_thin_bridge_separates_levels_four_three_and_two() {
    let mut values = vec![
        (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3),
        (4, 5), (5, 6), (4, 6),
        (3, 4),
    ];
    values.sort_unstable();
    let production = graph(7, &values);
    let mut kernel = TrussKernel::new(production.clone());
    let report = kernel.decompose(values.len(), 0, true);
    let by_edge = level_map(&production, &report);

    assert_eq!(by_edge[&(3, 4)], 2);
    for edge in [(4, 5), (5, 6), (4, 6)] {
        assert_eq!(by_edge[&edge], 3);
    }
    for a in 0..4usize {
        for b in (a + 1)..4 {
            assert_eq!(by_edge[&(a, b)], 4);
        }
    }
    assert_eq!(
        report.witnesses.iter().filter(|w| w.truss_level == 4).count(),
        1
    );
    assert_eq!(
        report.witnesses.iter().filter(|w| w.truss_level == 3).count(),
        1
    );
    assert_eq!(
        report.witnesses.iter().filter(|w| w.truss_level == 2).count(),
        2
    );
    check_report_witnesses(&report).unwrap();
}

#[test]
fn k4_satisfies_level_four_support_threshold_two() {
    let values = vec![(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
    let production = graph(4, &values);
    let report = TrussKernel::new(production).decompose(6, 0, true);

    assert_eq!(report.triangle_count, 4);
    assert!(report.levels.iter().all(|edge| {
        edge.initial_support == 2 && edge.truss_level == 4
    }));
    assert_eq!(report.witnesses.len(), 1);
    assert_eq!(report.witnesses[0].truss_level, 4);
    assert_eq!(report.witnesses[0].support_threshold, 2);
    check_report_witnesses(&report).unwrap();
}

#[test]
fn multiedges_are_deduplicated_by_declared_policy() {
    let input = GraphInput {
        vertices: Some(vec![0, 1, 2]),
        edges: edges(&[(0, 1), (1, 0), (1, 2), (2, 0)]),
        multi_edge_policy: MultiEdgePolicy::Deduplicate,
    };
    let normalized = normalize_input(&input).unwrap();
    assert_eq!(normalized.duplicate_edge_count, 1);
    assert_eq!(normalized.graph.edges().len(), 3);

    let mut kernel = TrussKernel::new(normalized.graph);
    let report = kernel.decompose(normalized.input_edge_count, normalized.duplicate_edge_count, true);
    assert_eq!(report.input_edge_count, 4);
    assert_eq!(report.normalized_edge_count, 3);
    assert_eq!(report.triangle_count, 1);
    assert!(report.levels.iter().all(|edge| edge.truss_level == 3));
    check_report_witnesses(&report).unwrap();
}

#[test]
fn reject_policy_reports_duplicate_edge_failure_category() {
    let input = GraphInput {
        vertices: Some(vec![0, 1, 2]),
        edges: edges(&[(0, 1), (1, 0)]),
        multi_edge_policy: MultiEdgePolicy::Reject,
    };
    let error = normalize_input(&input).unwrap_err();
    assert_eq!(
        error.category,
        ktruss_backend::diagnostics::ErrorCategory::DuplicateEdge
    );
    assert!(error.message.contains("parallel/multi-edges"));
}

#[test]
fn self_loops_are_rejected_before_simple_graph_construction() {
    let input = GraphInput {
        vertices: Some(vec![0, 1]),
        edges: edges(&[(0, 0), (0, 1)]),
        multi_edge_policy: MultiEdgePolicy::Deduplicate,
    };
    let error = normalize_input(&input).unwrap_err();
    assert_eq!(
        error.category,
        ktruss_backend::diagnostics::ErrorCategory::SelfLoop
    );
    assert_eq!(error.diagnostics.state["self_loop_count"], 1);
}

#[test]
fn deleting_k4_edge_decrements_each_affected_triangle_support_once() {
    let values = vec![(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
    let mut production = graph(4, &values);
    let removed = production.edge_id(0, 1).unwrap();
    let report = delete_edge_once_per_triangle(&mut production, removed).unwrap();

    assert_eq!(report.affected_triangles.len(), 2);
    assert_eq!(report.triangle_count_before, 4);
    assert_eq!(report.triangle_count_after, 2);

    let support = |u: usize, v: usize| {
        let edge = report
            .support_after
            .iter()
            .enumerate()
            .map(|(id, _)| id)
            .find(|id| {
                let endpoints = production.edge(*id).unwrap();
                endpoints.u == u && endpoints.v == v
            })
            .unwrap();
        report.support_after[edge]
    };
    assert_eq!(support(0, 2), 1);
    assert_eq!(support(1, 2), 1);
    assert_eq!(support(0, 3), 1);
    assert_eq!(support(1, 3), 1);
    assert_eq!(support(2, 3), 2);
}
