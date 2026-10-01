use dag_path_cover::{
    analyze_named_graph,
    config::Config,
    model::LabeledGraph,
    witness::{verify_path_cover, WitnessFailure},
    AnalysisFailure, AnalysisSuccess,
};
use serde_json::Value;
use std::{
    collections::BTreeSet,
    path::{Path, PathBuf},
    process::Command,
};

#[derive(Clone)]
struct NamedFixture {
    name: String,
    document: Value,
    path: PathBuf,
}

fn fixture(name: &str) -> NamedFixture {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join(format!("{name}.json"));
    let text = std::fs::read_to_string(&path).unwrap();
    let document: Value = serde_json::from_str(&text).unwrap();
    NamedFixture {
        name: name.to_string(),
        document,
        path,
    }
}

fn labeled_graph(document: &Value) -> LabeledGraph {
    LabeledGraph {
        vertices: document["graph"]["vertices"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_str().unwrap().to_string())
            .collect(),
        edges: document["graph"]["edges"]
            .as_array()
            .unwrap()
            .iter()
            .map(|edge| {
                (
                    edge["from"].as_str().unwrap().to_string(),
                    edge["to"].as_str().unwrap().to_string(),
                )
            })
            .collect(),
    }
}

fn analyze(document: &Value) -> AnalysisSuccess {
    let config = Config::default();
    analyze_named_graph(
        Some("unit-case".to_string()),
        labeled_graph(document),
        config.max_vertices,
        config.max_edges,
    )
    .expect("DAG fixture should analyze successfully")
}

fn paths_of(success: &AnalysisSuccess) -> Vec<Vec<String>> {
    success
        .cover
        .paths
        .iter()
        .map(|path| path.iter().map(|id| success.labels[*id].clone()).collect())
        .collect()
}

fn matching_of(success: &AnalysisSuccess) -> Vec<(String, String)> {
    success
        .cover
        .matching
        .pairs
        .iter()
        .map(|(from, to)| (success.labels[*from].clone(), success.labels[*to].clone()))
        .collect()
}

#[test]
fn zero_edge_graph_returns_all_singleton_paths() {
    let fixture = fixture("zero_edges");
    let success = analyze(&fixture.document);
    assert_eq!(success.cover.minimum_path_count, 3);
    assert_eq!(success.cover.matching.size, 0);
    assert_eq!(paths_of(&success), vec![vec!["a"], vec!["b"], vec!["c"]]);
    assert!(success.witness.valid);
}

#[test]
fn diamond_has_concrete_two_path_cover() {
    let fixture = fixture("diamond");
    let success = analyze(&fixture.document);
    assert_eq!(success.cover.minimum_path_count, 2);
    assert_eq!(success.cover.matching.size, 2);
    assert_eq!(paths_of(&success), vec![vec!["s", "a", "t"], vec!["b"]]);
    assert_eq!(
        matching_of(&success),
        vec![
            ("a".to_string(), "t".to_string()),
            ("s".to_string(), "a".to_string())
        ]
    );
    assert!(success.witness.valid);
}

#[test]
fn multi_source_multi_sink_cover_has_no_repeats_or_gaps() {
    let fixture = fixture("multi_source_sink");
    let success = analyze(&fixture.document);
    let paths = paths_of(&success);

    assert_eq!(success.cover.minimum_path_count, 3);
    assert_eq!(success.cover.matching.size, 2);
    assert_eq!(paths, vec![vec!["a", "c", "d"], vec!["b"], vec!["e"]]);

    let flattened: BTreeSet<String> = paths.iter().flatten().cloned().collect();
    let expected: BTreeSet<String> = ["a", "b", "c", "d", "e"]
        .into_iter()
        .map(str::to_string)
        .collect();
    assert_eq!(flattened, expected);
    assert_eq!(flattened.len(), paths.iter().flatten().count());
    assert_eq!(success.graph.sources(), vec![0, 1]);
    assert_eq!(success.graph.sinks(), vec![3, 4]);
}

#[test]
fn disconnected_components_are_covered_independently() {
    let fixture = fixture("disconnected_components");
    let success = analyze(&fixture.document);
    assert_eq!(success.cover.minimum_path_count, 4);
    assert_eq!(success.cover.matching.size, 3);
    assert_eq!(
        paths_of(&success),
        vec![vec!["s", "p", "t"], vec!["x", "y"], vec!["q"], vec!["z"]]
    );
    assert!(success.witness.valid);
}

#[test]
fn cycle_returns_exact_evidence_and_failure_category() {
    let fixture = fixture("cycle");
    let config = Config::default();
    let error: AnalysisFailure = analyze_named_graph(
        Some("cycle-case".to_string()),
        labeled_graph(&fixture.document),
        config.max_vertices,
        config.max_edges,
    )
    .expect_err("a cyclic graph must not produce a path cover");

    assert_eq!(error.status, 422);
    assert_eq!(error.category, "CYCLE_DETECTED");
    assert_eq!(
        error.details["cycle_vertices"],
        serde_json::json!(["a", "b", "c"])
    );
    assert_eq!(
        error.details["cycle_edges"],
        serde_json::json!([
            {"from": "a", "to": "b"},
            {"from": "b", "to": "c"},
            {"from": "c", "to": "a"}
        ])
    );
    assert!(error
        .context
        .steps
        .iter()
        .any(|step| step.name == "cycle_check" && step.status == "failed"));
}

#[test]
fn malformed_inputs_have_specific_failure_categories() {
    let config = Config::default();
    let unknown = LabeledGraph {
        vertices: vec!["a".to_string(), "b".to_string()],
        edges: vec![("a".to_string(), "ghost".to_string())],
    };
    let error = analyze_named_graph(None, unknown, config.max_vertices, config.max_edges)
        .expect_err("unknown endpoint must fail");
    assert_eq!(error.status, 400);
    assert_eq!(error.category, "UNKNOWN_EDGE_ENDPOINT");
    assert_eq!(error.details["vertex"], "ghost");
    assert_eq!(error.details["side"], "to");

    let duplicate_vertex = LabeledGraph {
        vertices: vec!["a".to_string(), "a".to_string()],
        edges: vec![],
    };
    let error = analyze_named_graph(
        None,
        duplicate_vertex,
        config.max_vertices,
        config.max_edges,
    )
    .expect_err("duplicate vertex must fail");
    assert_eq!(error.category, "DUPLICATE_VERTEX");
    assert_eq!(error.details["first_index"], 0);
    assert_eq!(error.details["duplicate_index"], 1);
}

#[test]
fn witness_reports_concrete_failure_categories() {
    let fixture = fixture("diamond");
    let success = analyze(&fixture.document);

    let mut repeated = success.cover.clone();
    repeated.paths = vec![vec![0, 1, 3], vec![0]];
    let check = verify_path_cover(&success.graph, &repeated);
    assert!(!check.valid);
    assert!(check
        .failures
        .contains(&WitnessFailure::RepeatedVertex { vertex: 0 }));
    assert!(check
        .failures
        .contains(&WitnessFailure::MissingVertex { vertex: 2 }));

    let mut bad_edge = success.cover.clone();
    bad_edge.paths = vec![vec![0, 2, 1, 3]];
    let check = verify_path_cover(&success.graph, &bad_edge);
    assert!(check
        .failures
        .iter()
        .any(|failure| failure.category() == "UNKNOWN_PATH_EDGE"));

    let mut wrong_count = success.cover.clone();
    wrong_count.minimum_path_count = 99;
    let check = verify_path_cover(&success.graph, &wrong_count);
    assert!(check
        .failures
        .iter()
        .any(|failure| failure.category() == "PATH_COUNT_MISMATCH"));

    let mut wrong_relation = success.cover.clone();
    wrong_relation.matching.size = 1;
    let check = verify_path_cover(&success.graph, &wrong_relation);
    assert!(check
        .failures
        .iter()
        .any(|failure| failure.category() == "CARDINALITY_RELATION_FAILED"));
}

#[test]
fn independent_python_oracle_agrees_with_all_dag_fixtures() {
    let fixtures = [
        fixture("zero_edges"),
        fixture("diamond"),
        fixture("multi_source_sink"),
        fixture("disconnected_components"),
    ];
    let output = Command::new("python3")
        .arg("tests/python/reference_oracle.py")
        .args(fixtures.iter().map(|fixture| fixture.path.as_os_str()))
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("python3 reference oracle must run");

    eprintln!(
        "oracle run input={:?}\n{}",
        fixtures
            .iter()
            .map(|fixture| fixture.name.clone())
            .collect::<Vec<_>>(),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        output.status.success(),
        "oracle failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let references: Value = serde_json::from_slice(&output.stdout).unwrap();
    for fixture in fixtures.clone() {
        let success = analyze(&fixture.document);
        let actual = paths_of(&success);
        let name = fixture.name.clone();
        let reference = &references[name.as_str()];
        assert_eq!(
            success.cover.minimum_path_count,
            reference["minimum_path_count"].as_u64().unwrap() as usize,
            "fixture {name}"
        );
        let accepted: Vec<Vec<Vec<String>>> =
            serde_json::from_value(reference["minimum_covers"].clone()).unwrap();
        assert!(
            accepted.contains(&actual),
            "fixture {name} produced {actual:?}, oracle accepted {accepted:?}"
        );
        assert_eq!(
            success.cover.minimum_path_count,
            success.graph.vertex_count - success.cover.matching.size
        );
        assert_eq!(
            accepted
                .iter()
                .flatten()
                .flatten()
                .collect::<BTreeSet<_>>()
                .len(),
            success.graph.vertex_count
        );
    }
}
