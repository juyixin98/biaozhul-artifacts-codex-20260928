mod common;

use common::ReferenceGraph;
use truss_backend::graph::{MultipleEdgePolicy, SimpleGraph};
use truss_backend::kernel::{PeelingOrder, decompose};
use truss_backend::witness::verify_decomposition;

fn analyze(fixture: &str, order: PeelingOrder) -> truss_backend::kernel::Decomposition {
    let input = common::load_fixture(fixture);
    let (graph, _) = SimpleGraph::from_input(input).expect("fixture must normalize");
    let decomposition = decompose(&graph, order).expect("kernel must finish");
    let report = verify_decomposition(&graph, &decomposition);
    assert!(
        report.accepted,
        "witness check failed: {:?}",
        report.diagnostics
    );
    decomposition
}

fn production_levels(fixture: &str) -> std::collections::BTreeMap<String, usize> {
    let input = common::load_fixture(fixture);
    let reference = ReferenceGraph::from_input(&input);
    let expected = reference.truss_levels();

    let orders = [
        PeelingOrder::Declaration,
        PeelingOrder::ReverseDeclaration,
        PeelingOrder::EndpointLex,
    ];
    let mut result = None;
    for order in orders {
        let (graph, _) = SimpleGraph::from_input(input.clone()).unwrap();
        let decomposition = decompose(&graph, order).unwrap();
        let report = verify_decomposition(&graph, &decomposition);
        assert!(report.accepted, "{fixture}: {:?}", report.diagnostics);

        let actual: std::collections::BTreeMap<String, usize> = decomposition
            .edges
            .iter()
            .map(|edge| (edge.edge_id.clone(), edge.level))
            .collect();
        assert_eq!(
            actual,
            expected.iter().map(|(k, v)| (k.clone(), *v)).collect(),
            "production levels differ from triple-enumerating oracle for {fixture:?}"
        );

        if result.is_none() {
            result = Some(actual);
        } else {
            assert_eq!(&result, &Some(actual), "peeling order changed levels");
        }
    }
    result.unwrap()
}

#[test]
fn shared_edge_counts_each_triangle_once_during_first_support_removal() {
    let decomposition = analyze("shared_edge_multi_triangle.json", PeelingOrder::Declaration);

    assert_eq!(decomposition.counters.triangle_count, 3);
    let shared = decomposition
        .edges
        .iter()
        .find(|edge| edge.edge_id == "ab")
        .unwrap();
    assert_eq!(shared.initial_support, 3);
    assert_eq!(shared.level, 3);

    // During the k=4 round the three triangles disappear one at a time. Each triangle
    // is discovered exactly once, with two support decrements per triangle.
    assert_eq!(decomposition.counters.affected_triangle_updates, 3);
    assert_eq!(decomposition.counters.support_decrements, 6);
}

#[test]
fn shared_edge_levels_match_brute_force_oracle() {
    let levels = production_levels("shared_edge_multi_triangle.json");
    assert_eq!(levels["ab"], 3);
    assert_eq!(levels["a1"], 3);
    assert_eq!(levels["a2"], 3);
    assert_eq!(levels["a3"], 3);
    assert_eq!(levels["b1"], 3);
    assert_eq!(levels["b2"], 3);
    assert_eq!(levels["b3"], 3);
}

#[test]
fn clique_bridge_keeps_dense_component_levels_and_bridge_at_base() {
    let levels = production_levels("clique_bridge.json");
    assert_eq!(levels["k4_ab"], 4);
    assert_eq!(levels["k4_ac"], 4);
    assert_eq!(levels["k4_ad"], 4);
    assert_eq!(levels["k4_bc"], 4);
    assert_eq!(levels["k4_bd"], 4);
    assert_eq!(levels["k4_cd"], 4);
    assert_eq!(levels["k3_xy"], 3);
    assert_eq!(levels["k3_xz"], 3);
    assert_eq!(levels["k3_yz"], 3);
    assert_eq!(levels["bridge_dx"], 2);
}

#[test]
fn zero_triangle_graph_has_all_edges_at_base_level() {
    let input = common::load_fixture("zero_triangle.json");
    let reference = ReferenceGraph::from_input(&input);
    assert_eq!(reference.triangle_count(), 0);

    let (graph, _) = SimpleGraph::from_input(input).unwrap();
    assert_eq!(graph.triangle_count(), 0);
    let decomposition = decompose(&graph, PeelingOrder::Declaration).unwrap();
    let levels = production_levels("zero_triangle.json");

    assert!(levels.values().all(|level| *level == 2));
    assert_eq!(decomposition.counters.affected_triangle_updates, 0);
    assert_eq!(decomposition.counters.support_decrements, 0);
    assert_eq!(decomposition.witnesses.len(), 1);
    assert_eq!(decomposition.witnesses[0].minimum_support, 0);
}

#[test]
fn complete_graphs_verify_every_support_threshold() {
    for complete_size in 3..=6 {
        let mut nodes = Vec::new();
        let mut edges = Vec::new();
        for index in 0..complete_size {
            nodes.push(serde_json::json!({ "id": format!("n{index}") }));
        }
        for u in 0..complete_size {
            for v in (u + 1)..complete_size {
                edges.push(serde_json::json!({
                    "id": format!("n{u}_n{v}"),
                    "u": format!("n{u}"),
                    "v": format!("n{v}")
                }));
            }
        }
        let input: truss_backend::graph::GraphInput = serde_json::from_value(serde_json::json!({
            "name": format!("K{complete_size}"),
            "nodes": nodes,
            "edges": edges
        }))
        .unwrap();

        let reference = ReferenceGraph::from_input(&input);
        let expected_levels = reference.truss_levels();
        let (graph, _) = SimpleGraph::from_input(input).unwrap();
        let decomposition = decompose(&graph, PeelingOrder::Declaration).unwrap();
        let report = verify_decomposition(&graph, &decomposition);
        assert!(report.accepted, "K{complete_size}: {report:?}");

        for edge in &decomposition.edges {
            assert_eq!(edge.level, complete_size);
            assert_eq!(edge.initial_support, complete_size - 2);
            assert_eq!(expected_levels[&edge.edge_id], complete_size);
        }
        assert_eq!(
            decomposition.counters.triangle_count,
            complete_size * (complete_size - 1) * (complete_size - 2) / 6
        );
    }
}

#[test]
fn duplicate_edges_are_rejected_only_when_declared_policy_says_so() {
    let deduplicated = common::load_fixture("duplicate_edges.json");
    assert!(SimpleGraph::from_input(deduplicated).is_ok());

    let mut rejected = common::load_fixture("duplicate_edges.json");
    rejected.multiple_edge_policy = MultipleEdgePolicy::Reject;
    let error = SimpleGraph::from_input(rejected).unwrap_err();
    assert_eq!(error.status, 422);
    assert_eq!(error.diagnostics[0].code, "MULTIPLE_EDGE");
}
