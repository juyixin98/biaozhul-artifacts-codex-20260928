mod common;

use std::collections::BTreeSet;

use chordal_api::kernel::{
    analyze_chordality, check_perfect_elimination_order, maximum_cardinality_order,
    DEFAULT_CYCLE_SEARCH_BUDGET,
};
use chordal_api::witness::check_induced_cycle;

use common::{
    graph, implicit_graph, labels, reference_check_running_intersection,
    reference_has_induced_cycle, reference_maximal_cliques,
};

#[test]
fn empty_graph_has_empty_maximal_clique_and_singleton_clique_tree() {
    let graph = graph(&[], &[]);
    let run = analyze_chordality(&graph, DEFAULT_CYCLE_SEARCH_BUDGET).unwrap();

    assert!(run.chordal);
    assert_eq!(run.peo_order, Vec::<usize>::new());
    assert_eq!(run.maximal_cliques.len(), 1);
    assert_eq!(run.maximal_cliques[0].vertices, Vec::<usize>::new());
    assert_eq!(run.maximum_clique_ids, vec![0]);
    let tree = run.clique_tree.as_ref().unwrap();
    assert_eq!(tree.edges.len(), 0);
    assert!(run.clique_tree_check.as_ref().unwrap().valid);
    assert!(!reference_has_induced_cycle(&graph));
}

#[test]
fn tree_star_is_chordal_and_tree_cliques_have_rip() {
    let graph = graph(&["c", "a", "b", "d"], &[("a", "c"), ("b", "c"), ("c", "d")]);
    let run = analyze_chordality(&graph, DEFAULT_CYCLE_SEARCH_BUDGET).unwrap();

    assert!(run.chordal);
    assert_eq!(labels(&graph, &run.peo_order), vec!["d", "b", "a", "c"]);
    assert!(run.peo_report.valid);
    assert_eq!(run.maximal_cliques.len(), 3);
    assert_eq!(run.maximum_clique_ids, vec![0, 1, 2]);

    let mut actual: Vec<BTreeSet<usize>> = run
        .maximal_cliques
        .iter()
        .map(|clique| clique.vertices.iter().copied().collect())
        .collect();
    actual.sort();
    let expected = reference_maximal_cliques(&graph);
    assert_eq!(actual, expected);

    let tree = run.clique_tree.as_ref().unwrap();
    let edges: Vec<(usize, usize)> = tree
        .edges
        .iter()
        .map(|edge| (edge.parent, edge.child))
        .collect();
    assert!(reference_check_running_intersection(
        &graph, &actual, &edges
    ));
    assert!(run.clique_tree_check.as_ref().unwrap().valid);
    assert!(!reference_has_induced_cycle(&graph));
}

#[test]
fn complete_k4_returns_one_maximum_clique() {
    let vertices = ["a", "b", "c", "d"];
    let edges = [
        ("a", "b"),
        ("a", "c"),
        ("a", "d"),
        ("b", "c"),
        ("b", "d"),
        ("c", "d"),
    ];
    let graph = graph(&vertices, &edges);
    let run = analyze_chordality(&graph, DEFAULT_CYCLE_SEARCH_BUDGET).unwrap();

    assert!(run.chordal);
    assert_eq!(run.maximal_cliques.len(), 1);
    let k4: BTreeSet<usize> = run.maximal_cliques[0].vertices.iter().copied().collect();
    assert_eq!(k4, [0, 1, 2, 3].into_iter().collect());
    assert_eq!(run.maximum_clique_ids, vec![0]);
    assert_eq!(run.clique_tree.as_ref().unwrap().edges.len(), 0);
    assert_eq!(reference_maximal_cliques(&graph).len(), 1);
}

#[test]
fn four_cycle_is_non_chordal_with_independently_checked_witness() {
    let graph = graph(
        &["a", "b", "c", "d"],
        &[("a", "b"), ("b", "c"), ("c", "d"), ("d", "a")],
    );
    let run = analyze_chordality(&graph, DEFAULT_CYCLE_SEARCH_BUDGET).unwrap();

    assert!(!run.chordal);
    assert!(reference_has_induced_cycle(&graph));
    let witness = run.witness.as_ref().unwrap();
    let check = check_induced_cycle(&graph, witness);
    assert!(check.valid, "{}", check.reason);
    assert_eq!(witness.cycle.len(), 4);
    let mut cycle = labels(&graph, &witness.cycle);
    cycle.sort();
    assert_eq!(cycle, vec!["a", "b", "c", "d"]);
    assert!(run.maximal_cliques.is_empty());
    assert!(run.clique_tree.is_none());
}

#[test]
fn chordal_diamond_and_cycle_reference_agree() {
    let graph = graph(
        &["a", "b", "c", "d"],
        &[("a", "b"), ("a", "c"), ("b", "c"), ("b", "d"), ("c", "d")],
    );
    let run = analyze_chordality(&graph, DEFAULT_CYCLE_SEARCH_BUDGET).unwrap();

    assert!(run.chordal);
    assert!(!reference_has_induced_cycle(&graph));
    let mut actual: Vec<BTreeSet<usize>> = run
        .maximal_cliques
        .iter()
        .map(|clique| clique.vertices.iter().copied().collect())
        .collect();
    actual.sort();
    assert_eq!(actual, reference_maximal_cliques(&graph));
    assert_eq!(run.maximum_clique_ids, vec![0, 1]);

    let tree = run.clique_tree.as_ref().unwrap();
    assert_eq!(tree.edges.len(), 1);
    assert_eq!(tree.edges[0].separator, vec![1, 2]);
    assert!(run.clique_tree_check.as_ref().unwrap().valid);
}

#[test]
fn disconnected_chordal_graph_uses_empty_separators_to_connect_components() {
    let graph = graph(
        &["a", "b", "c", "x", "y"],
        &[("a", "b"), ("b", "c"), ("a", "c"), ("x", "y")],
    );
    let run = analyze_chordality(&graph, DEFAULT_CYCLE_SEARCH_BUDGET).unwrap();

    assert!(run.chordal);
    let mut actual: Vec<BTreeSet<usize>> = run
        .maximal_cliques
        .iter()
        .map(|clique| clique.vertices.iter().copied().collect())
        .collect();
    actual.sort();
    assert_eq!(actual, reference_maximal_cliques(&graph));
    let tree = run.clique_tree.as_ref().unwrap();
    assert_eq!(tree.edges.len(), run.maximal_cliques.len() - 1);
    assert!(tree.edges.iter().any(|edge| edge.separator.is_empty()));
    let edges: Vec<(usize, usize)> = tree
        .edges
        .iter()
        .map(|edge| (edge.parent, edge.child))
        .collect();
    assert!(reference_check_running_intersection(
        &graph, &actual, &edges
    ));
}

#[test]
fn duplicate_and_reversed_edges_are_normalized() {
    let graph = implicit_graph(&[("a", "b"), ("b", "a"), ("a", "b"), ("b", "c")]);

    assert_eq!(graph.vertex_count(), 3);
    assert_eq!(graph.edge_count(), 2);
    assert_eq!(graph.duplicate_edge_count(), 2);
    assert_eq!(graph.normalized_edges_indexed(), vec![(0, 1), (1, 2)]);
}

#[test]
fn tampered_peo_fails_with_specific_missing_edge_category() {
    let graph = graph(
        &["a", "b", "c", "d"],
        &[("a", "b"), ("a", "c"), ("b", "c"), ("b", "d"), ("c", "d")],
    );
    let order = vec![
        graph.vertex_id("b").unwrap(),
        graph.vertex_id("a").unwrap(),
        graph.vertex_id("d").unwrap(),
        graph.vertex_id("c").unwrap(),
    ];
    let report = check_perfect_elimination_order(&graph, &order);

    assert!(!report.valid);
    let failure = report.first_failure.unwrap();
    assert_eq!(failure.vertex, graph.vertex_id("b").unwrap());
    assert_eq!(failure.position, 0);
    assert_eq!(
        failure.later_neighbors,
        vec![
            graph.vertex_id("a").unwrap(),
            graph.vertex_id("c").unwrap(),
            graph.vertex_id("d").unwrap()
        ]
    );
    assert!(failure
        .missing_edges
        .contains(&(graph.vertex_id("a").unwrap(), graph.vertex_id("d").unwrap())));
    assert!(!reference_has_induced_cycle(&graph));
}

#[test]
fn mcs_reverse_order_is_peo_for_known_cases() {
    let cases = [
        (vec!["a", "b"], vec![("a", "b")]),
        (
            vec!["a", "b", "c", "d"],
            vec![("a", "b"), ("b", "c"), ("c", "d"), ("a", "d")],
        ),
    ];

    for (vertices, edges) in cases {
        let edge_refs: Vec<(&str, &str)> = edges;
        let graph = graph(&vertices, &edge_refs);
        let (selection, _) = maximum_cardinality_order(&graph);
        let candidate: Vec<usize> = selection.into_iter().rev().collect();
        let report = check_perfect_elimination_order(&graph, &candidate);
        assert_eq!(
            reference_has_induced_cycle(&graph),
            !report.valid,
            "MCS candidate disagreed with cycle enumeration"
        );
    }
}
