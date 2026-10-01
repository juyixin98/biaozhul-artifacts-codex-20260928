mod common;

use common::{assert_acyclic, kruskal, RefGraph, Rng};
use msf_service::kernel::{DynamicMsf, MaintenanceOp};
use msf_service::model::Edge;

fn edge(id: &str, u: i64, v: i64, weight: i64) -> Edge {
    Edge::new(id, u, v, weight)
}

fn assert_matches_reference(kernel: &DynamicMsf, graph_id: &str, reference: &RefGraph) {
    let snapshot = kernel.snapshot(graph_id, 0);
    let expected = kruskal(reference);
    assert_acyclic(reference, &expected.forest);
    assert_eq!(snapshot.forest_edge_ids, expected.forest);
    assert_eq!(snapshot.total_weight, expected.total_weight);
}

#[test]
fn disconnected_parallel_edges_and_self_loops_have_specific_forest() {
    let mut kernel = DynamicMsf::new(100, 128);
    let mut reference = RefGraph::default();

    for (index, values) in [
        ("loop", 1, 1, -5),
        ("a", 1, 2, 3),
        ("a-parallel", 1, 2, 2),
        ("b", 2, 3, 4),
        ("other", 8, 9, 7),
    ]
    .into_iter()
    .enumerate()
    {
        let (id, u, v, weight) = values;
        reference.add_edge(id, u, v, weight);
        kernel
            .apply(index, MaintenanceOp::AddEdge(edge(id, u, v, weight)))
            .unwrap();
    }

    let snapshot = kernel.snapshot("g", 1);
    assert_eq!(
        snapshot.forest_edge_ids.into_iter().collect::<Vec<_>>(),
        vec![
            "a-parallel".to_string(),
            "b".to_string(),
            "other".to_string()
        ]
    );
    assert_eq!(snapshot.total_weight, 13);
    assert_matches_reference(&kernel, "g", &reference);
}

#[test]
fn equal_weight_cycle_uses_stable_edge_id_at_every_step() {
    let mut kernel = DynamicMsf::new(100, 128);
    let mut reference = RefGraph::default();
    let additions = [("c", 2, 3), ("b", 1, 3), ("a", 1, 2)];

    for (index, (id, u, v)) in additions.into_iter().enumerate() {
        reference.add_edge(id, u, v, 5);
        let step = kernel
            .apply(index, MaintenanceOp::AddEdge(edge(id, u, v, 5)))
            .unwrap();
        assert_matches_reference(&kernel, "equal", &reference);
        if id == "a" {
            assert_eq!(step.action, "replaced_heavier_cycle_edge");
            assert_eq!(step.forest_edge_removed.as_deref(), Some("c"));
            assert_eq!(step.forest_edge_added.as_deref(), Some("a"));
        }
    }

    let snapshot = kernel.snapshot("equal", 3);
    assert_eq!(
        snapshot.forest_edge_ids.into_iter().collect::<Vec<_>>(),
        vec!["a".to_string(), "b".to_string()]
    );
    assert_eq!(snapshot.total_weight, 10);
}

#[test]
fn deleting_tree_edges_distinguishes_bridge_from_minimum_reconnect() {
    let mut kernel = DynamicMsf::new(100, 128);
    let additions = [
        ("tree", 1, 2, 1),
        ("cheap", 2, 3, 2),
        ("bridge", 3, 4, 3),
        ("parallel-1", 2, 3, 5),
        ("parallel-0", 2, 3, 4),
    ];

    for (index, (id, u, v, weight)) in additions.into_iter().enumerate() {
        kernel
            .apply(index, MaintenanceOp::AddEdge(edge(id, u, v, weight)))
            .unwrap();
    }

    let reconnect = kernel
        .apply(5, MaintenanceOp::DeleteEdge { id: "cheap".into() })
        .unwrap();
    assert_eq!(reconnect.action, "deleted_tree_edge_and_reconnected");
    assert_eq!(reconnect.forest_edge_added.as_deref(), Some("parallel-0"));

    let bridge = kernel
        .apply(
            6,
            MaintenanceOp::DeleteEdge {
                id: "bridge".into(),
            },
        )
        .unwrap();
    assert_eq!(bridge.action, "deleted_tree_edge_bridge");
    assert!(bridge.forest_edge_added.is_none());
    assert_eq!(bridge.total_weight_delta, "-3");
}

#[test]
fn tree_weight_increase_prefers_lowest_weight_then_stable_id() {
    let mut kernel = DynamicMsf::new(100, 128);
    for (index, values) in [("t", 1, 2, 1), ("alt-z", 1, 2, 5), ("alt-a", 1, 2, 5)]
        .into_iter()
        .enumerate()
    {
        let (id, u, v, weight) = values;
        kernel
            .apply(index, MaintenanceOp::AddEdge(edge(id, u, v, weight)))
            .unwrap();
    }

    let step = kernel
        .apply(
            3,
            MaintenanceOp::SetWeight {
                id: "t".into(),
                weight: 6,
            },
        )
        .unwrap();
    assert_eq!(step.action, "tree_weight_increased_and_reconnected");
    assert_eq!(step.forest_edge_removed.as_deref(), Some("t"));
    assert_eq!(step.forest_edge_added.as_deref(), Some("alt-a"));
    assert_eq!(step.total_weight_delta, "4");
}

#[test]
fn non_tree_decrease_replaces_max_path_edge_with_stable_tie_break() {
    let mut kernel = DynamicMsf::new(100, 128);
    for (index, values) in [
        ("low", 1, 2, 1),
        ("m-z", 2, 3, 5),
        ("m-a", 1, 3, 5),
        ("extra", 1, 3, 9),
    ]
    .into_iter()
    .enumerate()
    {
        let (id, u, v, weight) = values;
        kernel
            .apply(index, MaintenanceOp::AddEdge(edge(id, u, v, weight)))
            .unwrap();
    }

    let step = kernel
        .apply(
            4,
            MaintenanceOp::SetWeight {
                id: "extra".into(),
                weight: 4,
            },
        )
        .unwrap();
    assert_eq!(step.action, "non_tree_weight_decreased_entered_forest");
    assert_eq!(step.forest_edge_removed.as_deref(), Some("m-a"));
    assert_eq!(step.forest_edge_added.as_deref(), Some("extra"));
    assert_eq!(step.total_weight_delta, "-1");
}

#[test]
fn random_small_graphs_match_full_kruskal_after_every_step() {
    for seed in 0..300u64 {
        let mut rng = Rng::new(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1));
        let mut kernel = DynamicMsf::new(128, 128);
        let mut reference = RefGraph::default();
        let vertex_count = 2 + rng.below(5) as i64;

        for step in 0..80usize {
            let existing_ids: Vec<String> = reference.edges.keys().cloned().collect();
            let choice = rng.below(10);
            let operation = if choice < 6 || existing_ids.is_empty() {
                let id = format!("e{step:02}");
                let u = 1 + rng.below(vertex_count as usize) as i64;
                let v = if rng.below(10) == 0 {
                    u
                } else {
                    1 + rng.below(vertex_count as usize) as i64
                };
                let weight = (rng.below(11) as i64) - 3;
                reference.add_edge(&id, u, v, weight);
                MaintenanceOp::AddEdge(edge(&id, u, v, weight))
            } else if choice < 8 {
                let id = existing_ids[rng.below(existing_ids.len())].clone();
                reference.delete_edge(&id);
                MaintenanceOp::DeleteEdge { id }
            } else {
                let id = existing_ids[rng.below(existing_ids.len())].clone();
                let weight = (rng.below(11) as i64) - 3;
                reference.set_weight(&id, weight);
                MaintenanceOp::SetWeight { id, weight }
            };

            kernel.apply(step, operation).unwrap_or_else(|error| {
                panic!("seed {seed}, step {step}: dynamic operation failed: {error}")
            });
            assert_matches_reference(&kernel, "random", &reference);
        }
    }
}
