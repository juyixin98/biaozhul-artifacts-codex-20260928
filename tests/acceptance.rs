//! Independent acceptance tests with a separate full-Kruskal reference.

use std::collections::{BTreeMap, BTreeSet};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use msf_service::api;
use msf_service::model::{Edge, Weight};
use msf_service::operations::Operation;
use msf_service::service::UpdateService;
use tower::ServiceExt;

#[derive(Clone, Debug, PartialEq, Eq)]
struct RefEdge {
    id: String,
    u: String,
    v: String,
    weight: Weight,
}

impl RefEdge {
    fn new(id: &str, u: &str, v: &str, weight: Weight) -> Self {
        Self {
            id: id.to_string(),
            u: u.to_string(),
            v: v.to_string(),
            weight,
        }
    }

    fn upsert(id: &str, u: &str, v: &str, weight: Weight) -> Operation {
        Operation::UpsertEdge(Edge {
            id: id.to_string(),
            u: u.to_string(),
            v: v.to_string(),
            weight,
        })
    }

    fn update_weight(id: &str, weight: Weight) -> Operation {
        Operation::UpdateWeight {
            id: id.to_string(),
            weight,
        }
    }

    fn delete(id: &str) -> Operation {
        Operation::DeleteEdge { id: id.to_string() }
    }
}

#[derive(Default)]
struct ReferenceGraph {
    edges: BTreeMap<String, RefEdge>,
}

impl ReferenceGraph {
    fn apply(&mut self, operation: &Operation) {
        match operation {
            Operation::UpsertEdge(edge) => {
                self.edges.insert(
                    edge.id.clone(),
                    RefEdge::new(&edge.id, &edge.u, &edge.v, edge.weight),
                );
            }
            Operation::UpdateWeight { id, weight } => {
                if let Some(edge) = self.edges.get_mut(id) {
                    edge.weight = *weight;
                }
            }
            Operation::DeleteEdge { id } => {
                self.edges.remove(id);
            }
        }
    }

    fn canonical_msf(&self) -> (BTreeSet<String>, i128) {
        let mut parent: BTreeMap<String, String> = BTreeMap::new();
        for edge in self.edges.values() {
            parent
                .entry(edge.u.clone())
                .or_insert_with(|| edge.u.clone());
            parent
                .entry(edge.v.clone())
                .or_insert_with(|| edge.v.clone());
        }

        let mut edges: Vec<&RefEdge> = self
            .edges
            .values()
            .filter(|edge| edge.u != edge.v)
            .collect();
        edges.sort_by(|a, b| (a.weight, a.id.as_str()).cmp(&(b.weight, b.id.as_str())));

        let mut forest = BTreeSet::new();
        let mut total: i128 = 0;
        for edge in edges {
            if reference_union(&mut parent, &edge.u, &edge.v) {
                forest.insert(edge.id.clone());
                total += i128::from(edge.weight);
            }
        }
        (forest, total)
    }
}

fn reference_find(parent: &mut BTreeMap<String, String>, vertex: &str) -> String {
    let mut current = vertex.to_string();
    while let Some(next) = parent.get(&current) {
        if *next == current {
            return current;
        }
        current = next.clone();
    }
    current
}

fn reference_union(parent: &mut BTreeMap<String, String>, left: &str, right: &str) -> bool {
    let left_root = reference_find(parent, left);
    let right_root = reference_find(parent, right);
    if left_root == right_root {
        return false;
    }
    parent.insert(right_root.clone(), left_root.clone());
    true
}

fn operation_id(operation: &Operation) -> &str {
    match operation {
        Operation::UpsertEdge(edge) => &edge.id,
        Operation::UpdateWeight { id, .. } | Operation::DeleteEdge { id } => id,
    }
}

fn assert_matches_reference(service: &UpdateService, reference: &ReferenceGraph) {
    let snapshot = service.snapshot();
    let (expected_forest, expected_total) = reference.canonical_msf();
    let actual_forest: BTreeSet<String> = snapshot.forest_edge_ids.iter().cloned().collect();
    assert_eq!(actual_forest, expected_forest);
    assert_eq!(snapshot.total_weight, expected_total);
}

fn assert_forest_is_acyclic(service: &UpdateService) {
    let snapshot = service.snapshot();
    let edges: BTreeMap<String, &Edge> = snapshot
        .forest_edge_ids
        .iter()
        .filter_map(|id| {
            snapshot
                .edges
                .iter()
                .find(|edge| &edge.id == id)
                .map(|edge| (id.clone(), edge))
        })
        .collect();

    for tree_id in &snapshot.forest_edge_ids {
        let edge = edges[tree_id];
        assert!(
            !connects_without(&edges, tree_id, &edge.u, &edge.v),
            "cycle found around {tree_id}"
        );
    }
}

fn connects_without(
    edges: &BTreeMap<String, &Edge>,
    excluded: &str,
    source: &str,
    target: &str,
) -> bool {
    let mut adjacency: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (id, edge) in edges {
        if id == excluded {
            continue;
        }
        adjacency
            .entry(edge.u.clone())
            .or_default()
            .push(edge.v.clone());
        adjacency
            .entry(edge.v.clone())
            .or_default()
            .push(edge.u.clone());
    }

    let mut seen = BTreeSet::from([source.to_string()]);
    let mut stack = vec![source.to_string()];
    while let Some(vertex) = stack.pop() {
        if vertex == target {
            return true;
        }
        if let Some(neighbors) = adjacency.get(&vertex) {
            for neighbor in neighbors {
                if seen.insert(neighbor.clone()) {
                    stack.push(neighbor.clone());
                }
            }
        }
    }
    false
}

#[test]
fn increasing_tree_edge_weight_selects_minimum_replacement() {
    let service = UpdateService::new();
    service
        .apply_checked(&[
            RefEdge::upsert("tree", "a", "b", 1),
            RefEdge::upsert("heavy", "b", "c", 5),
            RefEdge::upsert("choice2", "a", "c", 4),
            RefEdge::upsert("choice1", "a", "c", 3),
        ])
        .unwrap();

    let outcome = service
        .apply_checked(&[RefEdge::update_weight("tree", 6)])
        .unwrap();

    assert_eq!(
        outcome.state.forest_edge_ids,
        vec!["choice1".to_string(), "heavy".to_string()]
    );
    assert_eq!(outcome.state.total_weight, 8);
    assert!(outcome.steps.iter().any(|step| {
        step.step == "tree_edge_replaced_after_increase"
            && step.candidate_edge_id.as_deref() == Some("choice1")
    }));
}

#[test]
fn bridge_delete_splits_and_reconnect_uses_minimum_cut_edge() {
    let disconnected = UpdateService::new();
    disconnected
        .apply_checked(&[RefEdge::upsert("only", "a", "b", 2)])
        .unwrap();
    let split = disconnected
        .apply_checked(&[RefEdge::delete("only")])
        .unwrap();
    assert!(split.state.forest_edge_ids.is_empty());
    assert_eq!(split.state.total_weight, 0);
    assert_eq!(split.state.graph_component_count, 2);
    assert!(split
        .steps
        .iter()
        .any(|step| step.step == "bridge_removal_split_component"));

    let service = UpdateService::new();
    service
        .apply_checked(&[
            RefEdge::upsert("bridge2", "a", "b", 1),
            RefEdge::upsert("bridge1", "a", "b", 2),
            RefEdge::upsert("left", "x", "a", 3),
            RefEdge::upsert("right", "b", "y", 4),
        ])
        .unwrap();

    let outcome = service
        .apply_checked(&[RefEdge::delete("bridge2")])
        .unwrap();
    assert_eq!(
        outcome.state.forest_edge_ids,
        vec![
            "bridge1".to_string(),
            "left".to_string(),
            "right".to_string()
        ]
    );
    assert_eq!(outcome.state.total_weight, 9);
    assert!(outcome.steps.iter().any(|step| {
        step.step == "minimum_cut_edge_selected"
            && step.candidate_edge_id.as_deref() == Some("bridge1")
    }));
}

#[test]
fn equal_weight_cycle_uses_stable_edge_id() {
    let service = UpdateService::new();
    service
        .apply_checked(&[
            RefEdge::upsert("later", "a", "b", 5),
            RefEdge::upsert("middle", "a", "b", 5),
            RefEdge::upsert("earlier", "a", "b", 5),
        ])
        .unwrap();

    assert_eq!(
        service.snapshot().forest_edge_ids,
        vec!["earlier".to_string()]
    );

    let outcome = service
        .apply_checked(&[RefEdge::upsert("earlier", "a", "b", 6)])
        .unwrap();
    assert_eq!(outcome.state.forest_edge_ids, vec!["middle".to_string()]);
    assert_eq!(outcome.state.total_weight, 5);
}

#[test]
fn self_loops_parallel_edges_and_disconnected_vertices_are_supported() {
    let service = UpdateService::new();
    let outcome = service
        .apply_checked(&[
            RefEdge::upsert("loop", "a", "a", -100),
            RefEdge::upsert("p2", "a", "b", 8),
            RefEdge::upsert("p1", "a", "b", 3),
        ])
        .unwrap();

    assert_eq!(outcome.state.forest_edge_ids, vec!["p1".to_string()]);
    assert_eq!(outcome.state.total_weight, 3);
    assert_eq!(outcome.state.graph_component_count, 1);
    assert_eq!(
        outcome
            .state
            .edges
            .iter()
            .find(|edge| edge.id == "loop")
            .unwrap()
            .weight,
        -100
    );
}

#[test]
fn invalid_batch_is_atomic_and_reports_specific_failure_category() {
    let service = UpdateService::new();
    service
        .apply_checked(&[RefEdge::upsert("existing", "a", "b", 1)])
        .unwrap();

    let error = service
        .apply_checked(&[
            RefEdge::delete("existing"),
            RefEdge::update_weight("existing", 2),
        ])
        .unwrap_err();

    assert_eq!(error.category(), "unknown_edge");
    assert_eq!(service.snapshot().version, 1);
    assert_eq!(
        service.snapshot().forest_edge_ids,
        vec!["existing".to_string()]
    );
    assert_eq!(service.snapshot().total_weight, 1);
}

#[tokio::test]
async fn http_endpoints_return_request_id_and_specific_errors() {
    let app = api::app(UpdateService::new());

    let update = serde_json::json!({
        "operations": [
            {"type": "upsert_edge", "id": "e1", "u": "a", "v": "b", "weight": 1},
            {"type": "upsert_edge", "id": "e2", "u": "b", "v": "c", "weight": 2}
        ]
    });
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/updates/batch")
                .header("content-type", "application/json")
                .header("x-request-id", "req-fixture-001")
                .body(Body::from(update.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = hyper_body(response.into_body()).await;
    assert!(body.contains("req-fixture-001"));
    assert!(body.contains("batch_committed"));

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/updates/batch")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::json!({
                        "operations": [
                            {"type": "delete_edge", "id": "missing"}
                        ]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    let body = hyper_body(response.into_body()).await;
    let value: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(value["error"]["category"], "unknown_edge");
    assert_eq!(value["error"]["details"]["operation_index"], 0);
}

async fn hyper_body(body: axum::body::Body) -> String {
    let bytes = axum::body::to_bytes(body, usize::MAX)
        .await
        .expect("response body is readable");
    String::from_utf8(bytes.to_vec()).expect("response is UTF-8")
}

#[test]
fn randomized_small_graphs_match_full_kruskal_after_every_step() {
    let mut state = 0x1234_5678_9abc_def0_u64;
    let mut next_u32 = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state &= u64::MAX;
        (state & u32::MAX as u64) as u32
    };

    let service = UpdateService::new();
    let mut reference = ReferenceGraph::default();
    let mut committed_steps = 0;

    for step in 0..2500 {
        let id_number = next_u32() % 28;
        let id = format!("e{id_number:03}");
        let operation = if step < 12 || next_u32() % 10 < 7 {
            let u = next_u32() % 6;
            let v = next_u32() % 6;
            let weight = (next_u32() % 9) as Weight - 2;
            RefEdge::upsert(&id, &format!("v{u}"), &format!("v{v}"), weight)
        } else if next_u32() % 2 == 0 {
            RefEdge::update_weight(&id, (next_u32() % 9) as Weight - 2)
        } else {
            RefEdge::delete(&id)
        };

        let needs_existing = matches!(
            operation,
            Operation::UpdateWeight { .. } | Operation::DeleteEdge { .. }
        );
        if needs_existing && !reference.edges.contains_key(operation_id(&operation)) {
            assert!(service
                .apply_checked(std::slice::from_ref(&operation))
                .is_err());
            continue;
        }

        let outcome = service
            .apply_checked(std::slice::from_ref(&operation))
            .expect("valid randomized step is accepted");
        reference.apply(&operation);
        committed_steps += 1;
        assert_eq!(outcome.version_to, committed_steps);
        assert_matches_reference(&service, &reference);
        assert_forest_is_acyclic(&service);
    }
}
