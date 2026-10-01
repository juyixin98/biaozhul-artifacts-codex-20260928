use msf_service::witness::{verify_claim, WitnessClaimInput, WitnessEdge, WitnessFailureCategory};
use std::collections::{BTreeMap, BTreeSet};

fn edges(pairs: [(&str, i64, i64, i64); 3]) -> BTreeMap<String, WitnessEdge> {
    pairs
        .into_iter()
        .map(|(id, u, v, weight)| (id.to_string(), WitnessEdge { u, v, weight }))
        .collect()
}

fn triangle_edges() -> BTreeMap<String, WitnessEdge> {
    edges([("a", 1, 2, 1), ("b", 2, 3, 1), ("c", 1, 3, 1)])
}

#[test]
fn accepts_independently_computed_stable_forest_and_total() {
    let claim = WitnessClaimInput {
        vertices: BTreeSet::from([1, 2, 3]),
        edges: triangle_edges(),
        forest_edge_ids: vec!["a".into(), "b".into()],
    };
    let report = verify_claim(&claim, Some(2));
    assert_eq!(report.status, "accepted");
    assert!(report.failures.is_empty());
    assert_eq!(report.expected_total_weight, "2");
}

#[test]
fn rejects_unknown_forest_edge() {
    let claim = WitnessClaimInput {
        vertices: BTreeSet::from([1, 2, 3]),
        edges: triangle_edges(),
        forest_edge_ids: vec!["a".into(), "missing".into()],
    };
    let report = verify_claim(&claim, Some(2));
    assert_eq!(report.status, "rejected");
    assert_eq!(
        report.failures[0].category,
        WitnessFailureCategory::UnknownEdge
    );
    assert_eq!(
        report.failures[0].message,
        "forest edge id `missing` is not present in the full edge set"
    );
}

#[test]
fn rejects_self_loop_forest_edge_with_specific_category() {
    let mut all_edges = triangle_edges();
    all_edges.insert(
        "loop".into(),
        WitnessEdge {
            u: 4,
            v: 4,
            weight: -9,
        },
    );
    let claim = WitnessClaimInput {
        vertices: BTreeSet::from([1, 2, 3, 4]),
        edges: all_edges,
        forest_edge_ids: vec!["a".into(), "b".into(), "loop".into()],
    };
    let report = verify_claim(&claim, Some(-7));
    assert!(report
        .failures
        .iter()
        .any(|failure| failure.category == WitnessFailureCategory::ForestEdgeIsSelfLoop));
}

#[test]
fn rejects_duplicate_forest_edge_even_though_json_is_an_array() {
    let claim = WitnessClaimInput {
        vertices: BTreeSet::from([1, 2, 3]),
        edges: triangle_edges(),
        forest_edge_ids: vec!["a".into(), "a".into(), "b".into()],
    };
    let report = verify_claim(&claim, Some(2));
    assert!(report
        .failures
        .iter()
        .any(|failure| failure.category == WitnessFailureCategory::DuplicateForestEdge));
}

#[test]
fn rejects_forest_cycle_and_reports_edge_id() {
    let claim = WitnessClaimInput {
        vertices: BTreeSet::from([1, 2, 3]),
        edges: triangle_edges(),
        forest_edge_ids: vec!["a".into(), "b".into(), "c".into()],
    };
    let report = verify_claim(&claim, Some(3));
    assert!(report.failures.iter().any(|failure| failure.category
        == WitnessFailureCategory::ForestCycle
        && failure.message.contains("`c`")));
}

#[test]
fn rejects_non_spanning_forest_with_component_count() {
    let claim = WitnessClaimInput {
        vertices: BTreeSet::from([1, 2, 3]),
        edges: triangle_edges(),
        forest_edge_ids: vec!["a".into()],
    };
    let report = verify_claim(&claim, Some(1));
    assert!(report.failures.iter().any(|failure| failure.category
        == WitnessFailureCategory::NotSpanning
        && failure.message.contains("2 components")
        && failure.message.contains("has 1 components")));
}

#[test]
fn rejects_wrong_weight_with_actual_and_expected_values() {
    let claim = WitnessClaimInput {
        vertices: BTreeSet::from([1, 2, 3]),
        edges: triangle_edges(),
        forest_edge_ids: vec!["a".into(), "b".into()],
    };
    let report = verify_claim(&claim, Some(99));
    assert!(report.failures.iter().any(|failure| failure.category
        == WitnessFailureCategory::WrongWeight
        && failure.message.contains("claimed total weight 99")
        && failure.message.contains("actual forest sum 2")
        && failure.message.contains("Kruskal total 2")));
}

#[test]
fn rejects_non_minimum_or_unstable_tie_choice() {
    let claim = WitnessClaimInput {
        vertices: BTreeSet::from([1, 2, 3]),
        edges: triangle_edges(),
        forest_edge_ids: vec!["a".into(), "c".into()],
    };
    let report = verify_claim(&claim, Some(2));
    assert!(report
        .failures
        .iter()
        .any(|failure| failure.category == WitnessFailureCategory::UnstableOrNonMinimum));
}
