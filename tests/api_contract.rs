mod common;

use common::{post_json, RefGraph};
use serde_json::{json, Value};
use std::net::SocketAddr;
use tokio::net::TcpListener;

async fn spawn_server() -> u16 {
    let listener = TcpListener::bind(SocketAddr::from(([127, 0, 0, 1], 0)))
        .await
        .unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
        axum::serve(listener, chordal_graph_backend::api::router())
            .await
            .unwrap();
    });
    std::thread::sleep(std::time::Duration::from_millis(50));
    port
}

fn graph_body(vertices: &[&str], edges: &[(&str, &str)]) -> Value {
    json!({
        "vertices": vertices,
        "edges": edges
            .iter()
            .map(|(u, v)| json!({"u": u, "v": v}))
            .collect::<Vec<_>>()
    })
}

fn ids(names: &[&str], vertices: &[&str]) -> Vec<String> {
    names
        .iter()
        .map(|name| vertices[vertices.iter().position(|item| item == name).unwrap()].to_string())
        .collect()
}

#[tokio::test]
async fn analyzes_empty_and_disconnected_graphs_with_duplicate_edge_normalization() {
    let port = spawn_server().await;

    let empty = graph_body(&[], &[]);
    let (status, body) = post_json(port, "/api/graphs/analyze", empty);
    assert_eq!(status, 200);
    assert_eq!(body["chordal"], true);
    assert_eq!(body["graph"]["vertex_count"], 0);
    assert_eq!(body["graph"]["edge_count"], 0);
    assert_eq!(body["graph"]["components"], json!([]));
    assert_eq!(body["maximal_cliques"], json!([]));
    assert_eq!(body["clique_forest"]["edges"], json!([]));
    assert!(body["run_id"].as_str().unwrap().starts_with("run-"));

    let vertices = ["a", "b", "c", "d"];
    let disconnected = graph_body(
        &vertices,
        &[
            ("a", "b"),
            ("b", "a"),
            ("a", "b"),
            ("c", "d"),
            ("d", "c"),
        ],
    );
    let (status, body) = post_json(port, "/api/graphs/analyze", disconnected);
    assert_eq!(status, 200);
    assert_eq!(body["chordal"], true);
    assert_eq!(body["graph"]["edge_count"], 2);
    assert_eq!(body["graph"]["duplicate_edge_count"], 3);
    assert_eq!(body["graph"]["components"], json!([["a", "b"], ["c", "d"]]));
    assert_eq!(body["max_clique_size"], 2);
    assert_eq!(body["maximal_cliques"].as_array().unwrap().len(), 2);
    assert_eq!(body["clique_forest"]["component_count"], 2);
    assert_eq!(body["clique_forest"]["edges"].as_array().unwrap().len(), 0);
    common::assert_forest_rip(&body, &vertices, &[("a", "b"), ("c", "d")]);
}

#[tokio::test]
async fn returns_concrete_forest_for_tree_and_single_clique_for_complete_graph() {
    let port = spawn_server().await;

    let tree_vertices = ["a", "b", "c", "d"];
    let tree_edges = [("a", "b"), ("a", "c"), ("c", "d")];
    let tree = graph_body(&tree_vertices, &tree_edges);
    let (status, body) = post_json(port, "/api/graphs/analyze", tree);
    assert_eq!(status, 200);
    assert_eq!(body["chordal"], true);
    assert_eq!(body["max_clique_size"], 2);
    assert_eq!(body["maximal_cliques"].as_array().unwrap().len(), 3);
    assert_eq!(body["clique_forest"]["edges"].as_array().unwrap().len(), 2);
    for edge in body["clique_forest"]["edges"].as_array().unwrap() {
        assert_eq!(edge["weight"], 1);
    }
    common::assert_forest_rip(&body, &tree_vertices, &tree_edges);

    let complete_vertices = ["k0", "k1", "k2", "k3", "k4"];
    let complete_edges: Vec<(&str, &str)> = (0..5)
        .flat_map(|i| {
            ((i + 1)..5)
                .map(move |j| (complete_vertices[i], complete_vertices[j]))
        })
        .collect();
    let complete = graph_body(&complete_vertices, &complete_edges);
    let (status, body) = post_json(port, "/api/graphs/analyze", complete);
    assert_eq!(status, 200);
    assert_eq!(body["chordal"], true);
    assert_eq!(body["max_clique_size"], 5);
    assert_eq!(
        body["maximal_cliques"],
        json!([["k0", "k1", "k2", "k3", "k4"]])
    );
    assert_eq!(body["clique_forest"]["edges"], json!([]));
    assert_eq!(body["checks"]["running_intersection"]["passed"], true);
    common::assert_forest_rip(&body, &complete_vertices, &complete_edges);
}

#[tokio::test]
async fn returns_independently_checkable_witness_for_four_cycle() {
    let port = spawn_server().await;
    let vertices = ["a", "b", "c", "d"];
    let edges = [("a", "b"), ("b", "c"), ("c", "d"), ("d", "a")];
    let (status, body) = post_json(port, "/api/graphs/analyze", graph_body(&vertices, &edges));

    assert_eq!(status, 200);
    assert_eq!(body["chordal"], false);
    assert_eq!(body["chordless_cycle"]["length"], 4);

    let cycle: Vec<usize> = body["chordless_cycle"]["vertex_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_u64().unwrap() as usize)
        .collect();
    let reference = RefGraph::new(
        4,
        &[(0, 1), (1, 2), (2, 3), (3, 0)],
    );
    assert!(common::is_induced_cycle(&reference, &cycle));
    assert_eq!(body["peo_violation"]["missing_edge"].as_array().unwrap().len(), 2);
    assert_eq!(body["maximal_cliques"], json!([]));
    assert_eq!(body["clique_forest"]["nodes"], json!([]));
}

#[tokio::test]
async fn diamond_is_chordal_but_tampered_order_fails_as_state_conflict() {
    let port = spawn_server().await;
    let vertices = ["a", "b", "c", "d"];
    let edges = [
        ("a", "b"),
        ("a", "c"),
        ("b", "c"),
        ("c", "d"),
        ("b", "d"),
    ];

    let (status, body) = post_json(port, "/api/graphs/analyze", graph_body(&vertices, &edges));
    assert_eq!(status, 200);
    assert_eq!(body["chordal"], true);
    let canonical: Vec<String> = body["elimination_order"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_str().unwrap().to_string())
        .collect();

    let mut request = graph_body(&vertices, &edges);
    request["order"] = json!(canonical);
    let (status, body) = post_json(port, "/api/elimination-orders/verify", request);
    assert_eq!(status, 200);
    assert_eq!(body["valid"], true);

    let tampered = ids(&["c", "a", "b", "d"], &vertices);
    let mut request = graph_body(&vertices, &edges);
    request["order"] = json!(tampered);
    let (status, body) = post_json(port, "/api/elimination-orders/verify", request);
    assert_eq!(status, 409);
    assert_eq!(body["category"], "state_conflict");
    assert_eq!(body["code"], "supplied_order_not_peo");
    assert_eq!(body["details"]["supplied_peo_violation"]["vertex"], "c");
    assert_eq!(
        body["details"]["supplied_peo_violation"]["missing_edge"],
        json!(["a", "d"])
    );
    assert!(body["run_id"].as_str().unwrap().starts_with("run-"));
}

#[tokio::test]
async fn verifying_order_against_cycle_returns_non_chordal_state_conflict() {
    let port = spawn_server().await;
    let vertices = ["a", "b", "c", "d"];
    let edges = [("a", "b"), ("b", "c"), ("c", "d"), ("d", "a")];
    let mut request = graph_body(&vertices, &edges);
    request["order"] = json!(ids(&["a", "b", "c", "d"], &vertices));

    let (status, body) = post_json(port, "/api/elimination-orders/verify", request);
    assert_eq!(status, 409);
    assert_eq!(body["category"], "state_conflict");
    assert_eq!(body["code"], "non_chordal_graph");
    assert_eq!(body["details"]["chordless_cycle"]["length"], 4);
}

#[tokio::test]
async fn malformed_order_and_resource_limits_use_distinct_categories() {
    let port = spawn_server().await;
    let vertices = ["a", "b", "c"];
    let edges = [("a", "b"), ("b", "c")];

    let mut wrong_length = graph_body(&vertices, &edges);
    wrong_length["order"] = json!(["a", "b"]);
    let (status, body) = post_json(port, "/api/elimination-orders/verify", wrong_length);
    assert_eq!(status, 400);
    assert_eq!(body["category"], "input_malformed");
    assert_eq!(body["code"], "invalid_order_length");

    let mut unknown = graph_body(&vertices, &edges);
    unknown["order"] = json!(["a", "b", "x"]);
    let (status, body) = post_json(port, "/api/elimination-orders/verify", unknown);
    assert_eq!(status, 400);
    assert_eq!(body["code"], "unknown_order_vertex");

    let too_many: Vec<String> = (0..129).map(|index| format!("v{index}")).collect();
    let (status, body) = post_json(
        port,
        "/api/graphs/analyze",
        json!({"vertices": too_many, "edges": []}),
    );
    assert_eq!(status, 413);
    assert_eq!(body["category"], "resource_exhausted");
    assert_eq!(body["code"], "too_many_vertices");

    let many_vertices: Vec<String> = (0..128).map(|index| format!("v{index}")).collect();
    let many_edges: Vec<Value> = (0..8193)
        .map(|index| {
            json!({
                "u": many_vertices[index % 128],
                "v": many_vertices[(index + 1) % 128]
            })
        })
        .collect();
    let (status, body) = post_json(
        port,
        "/api/graphs/analyze",
        json!({"vertices": many_vertices, "edges": many_edges}),
    );
    assert_eq!(status, 413);
    assert_eq!(body["category"], "resource_exhausted");
    assert_eq!(body["code"], "too_many_edges");
}
