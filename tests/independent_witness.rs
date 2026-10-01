use maxflow_server::flow::{max_flow, CancelToken, FlowCertificate, SolveOptions, SolveStatus};
use maxflow_server::graph::{EdgeSpec, GraphSpec};
use maxflow_server::witness::{verify_flow_certificate, WitnessFailure};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

#[derive(Debug, Clone)]
struct FixtureEdge {
    id: String,
    from: usize,
    to: usize,
    capacity: u64,
}

#[derive(Debug, Clone)]
struct Fixture {
    raw: Value,
    node_count: usize,
    source: usize,
    sink: usize,
    edges: Vec<FixtureEdge>,
}

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join(format!("{name}.json"))
}

fn load_fixture(name: &str) -> Fixture {
    let raw: Value = serde_json::from_slice(
        &std::fs::read(fixture_path(name)).expect("fixture file should be readable"),
    )
    .expect("fixture should be JSON");
    let node_count = raw["node_count"].as_u64().unwrap() as usize;
    let source = raw["source"].as_u64().unwrap() as usize;
    let sink = raw["sink"].as_u64().unwrap() as usize;
    let edges = raw["edges"]
        .as_array()
        .unwrap()
        .iter()
        .map(|edge| FixtureEdge {
            id: edge["id"].as_str().unwrap().to_string(),
            from: edge["from"].as_u64().unwrap() as usize,
            to: edge["to"].as_u64().unwrap() as usize,
            capacity: edge["capacity"].as_u64().unwrap(),
        })
        .collect();

    Fixture {
        raw,
        node_count,
        source,
        sink,
        edges,
    }
}

impl Fixture {
    fn spec(&self) -> GraphSpec {
        GraphSpec {
            node_count: self.node_count,
            source: self.source,
            sink: self.sink,
            edge_ids: self.edges.iter().map(|edge| edge.id.clone()).collect(),
            edges: self
                .edges
                .iter()
                .map(|edge| EdgeSpec {
                    from: edge.from,
                    to: edge.to,
                    capacity: edge.capacity,
                })
                .collect(),
        }
    }

    fn expected_u64(&self, key: &str) -> u64 {
        self.raw[key].as_u64().expect("expected u64 field")
    }

    fn expected_nodes(&self, key: &str) -> Vec<usize> {
        self.raw[key]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_u64().unwrap() as usize)
            .collect()
    }

    fn expected_strings(&self, key: &str) -> Vec<String> {
        self.raw[key]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_str().unwrap().to_string())
            .collect()
    }
}

/// Enumerates every source-side subset with source in and sink out.
/// This is deliberately written independently from Dinic and witness.rs.
fn enumerate_min_cut(fixture: &Fixture) -> (u128, Vec<usize>, Vec<String>) {
    let optional_nodes: Vec<usize> = (0..fixture.node_count)
        .filter(|node| *node != fixture.source && *node != fixture.sink)
        .collect();
    let mut best: Option<(u128, Vec<usize>, Vec<String>)> = None;

    for mask in 0u64..(1u64 << optional_nodes.len()) {
        let mut source_side = HashSet::from([fixture.source]);
        for (bit, node) in optional_nodes.iter().enumerate() {
            if mask & (1u64 << bit) != 0 {
                source_side.insert(*node);
            }
        }

        let mut capacity: u128 = 0;
        let mut cut_edges = Vec::new();
        for edge in &fixture.edges {
            if source_side.contains(&edge.from) && !source_side.contains(&edge.to) {
                capacity += u128::from(edge.capacity);
                cut_edges.push(edge.id.clone());
            }
        }
        cut_edges.sort();
        let mut source_side = Vec::from_iter(source_side);
        source_side.sort();

        if best
            .as_ref()
            .is_none_or(|candidate: &(u128, Vec<usize>, Vec<String>)| capacity < candidate.0)
        {
            best = Some((capacity, source_side, cut_edges));
        }
    }

    best.expect("there is at least one s-t cut")
}

fn independently_check_flow(
    fixture: &Fixture,
    certificate: &FlowCertificate,
) -> Result<u64, String> {
    if certificate.edge_flows.len() != fixture.edges.len() {
        return Err("edge flow count mismatch".to_string());
    }

    let capacity_by_id: HashMap<_, _> = fixture
        .edges
        .iter()
        .map(|edge| (edge.id.as_str(), edge.capacity))
        .collect();
    let mut flow_by_id = HashMap::new();
    for edge_flow in &certificate.edge_flows {
        if flow_by_id
            .insert(edge_flow.edge_id.as_str(), edge_flow.flow)
            .is_some()
        {
            return Err(format!("duplicate flow entry {}", edge_flow.edge_id));
        }
        if edge_flow.flow
            > *capacity_by_id
                .get(edge_flow.edge_id.as_str())
                .ok_or_else(|| format!("unknown edge id in flow {}", edge_flow.edge_id))?
        {
            return Err(format!("flow exceeds capacity on {}", edge_flow.edge_id));
        }
    }

    let mut balance = vec![0i128; fixture.node_count];
    for edge in &fixture.edges {
        let flow = i128::from(flow_by_id[edge.id.as_str()]);
        balance[edge.from] += flow;
        balance[edge.to] -= flow;
    }

    for node in 0..fixture.node_count {
        if node != fixture.source && node != fixture.sink && balance[node] != 0 {
            return Err(format!(
                "node {node} violates conservation: {}",
                balance[node]
            ));
        }
    }

    let value = u64::try_from(balance[fixture.source])
        .map_err(|_| "source net outflow is not a non-negative u64".to_string())?;
    if balance[fixture.sink] != -i128::from(value) {
        return Err("sink inflow does not match source outflow".to_string());
    }
    if value != certificate.value {
        return Err("declared value differs from independently computed value".to_string());
    }
    Ok(value)
}

fn assert_fixture_completed(name: &str) {
    let fixture = load_fixture(name);
    let outcome = max_flow(
        fixture.spec(),
        SolveOptions { max_phases: None },
        CancelToken::new(),
        maxflow_server::logging::RunId::new(),
    )
    .expect("complete solve should succeed");

    assert_eq!(outcome.status, SolveStatus::Completed);
    assert!(outcome.optimal);
    let expected_value = fixture.expected_u64("expected_value");
    assert_eq!(outcome.certificate.value, expected_value);
    assert_eq!(
        independently_check_flow(&fixture, &outcome.certificate).unwrap(),
        expected_value
    );

    let cut = outcome
        .certificate
        .cut
        .as_ref()
        .expect("completed result has cut");
    assert_eq!(
        cut.source_side,
        fixture.expected_nodes("expected_source_side")
    );
    assert_eq!(cut.sink_side, fixture.expected_nodes("expected_sink_side"));

    let (min_cut_value, independent_side, independent_edges) = enumerate_min_cut(&fixture);
    assert_eq!(min_cut_value, u128::from(expected_value));
    assert_eq!(cut.source_side, independent_side);
    let mut actual_cut_edges: Vec<String> = Vec::new();
    let source_side: HashSet<usize> = cut.source_side.iter().copied().collect();
    for edge in &fixture.edges {
        if source_side.contains(&edge.from) && !source_side.contains(&edge.to) {
            actual_cut_edges.push(edge.id.clone());
        }
    }
    actual_cut_edges.sort();
    assert_eq!(actual_cut_edges, independent_edges);
    let mut expected_cut_edges = fixture.expected_strings("expected_cut_edges");
    expected_cut_edges.sort();
    assert_eq!(actual_cut_edges, expected_cut_edges);

    let report = verify_flow_certificate(&fixture.spec(), &outcome.certificate, true);
    assert!(report.valid, "certificate should pass witness: {report:?}");
    assert_eq!(report.cut.as_ref().unwrap().capacity, expected_value);

    let phase_logs: Vec<_> = outcome
        .events
        .iter()
        .filter(|entry| {
            serde_json::to_value(&entry.event)
                .unwrap()
                .get("event")
                .is_some_and(|event| event == "blocking_flow" || event == "bfs_result")
        })
        .collect();
    assert!(!phase_logs.is_empty(), "run log retains phase states");
    assert!(
        outcome
            .events
            .iter()
            .any(|entry| serde_json::to_value(&entry.event)
                .unwrap()
                .get("event")
                .is_some_and(|event| event == "witness_checked")),
        "run log retains witness judgment"
    );
}

#[test]
fn enumerates_cuts_for_all_required_small_fixtures() {
    for name in [
        "bottleneck",
        "reverse_reroute",
        "disconnected",
        "large_capacity",
        "parallel_zero",
    ] {
        assert_fixture_completed(name);
    }
}

#[test]
fn phase_budget_returns_a_feasible_but_unproven_flow() {
    let fixture = load_fixture("reverse_reroute");
    let outcome = max_flow(
        fixture.spec(),
        SolveOptions {
            max_phases: Some(1),
        },
        CancelToken::new(),
        maxflow_server::logging::RunId::new(),
    )
    .unwrap();

    assert_eq!(outcome.status, SolveStatus::Canceled);
    assert!(!outcome.optimal);
    assert!(outcome.certificate.cut.is_none());
    let expected = fixture.expected_u64("cancel_after_phase_1_value");
    assert_eq!(outcome.certificate.value, expected);
    independently_check_flow(&fixture, &outcome.certificate).unwrap();
    assert!(
        !outcome.optimal,
        "a phase-bounded run must not claim optimality"
    );
}

#[test]
fn zero_phase_budget_returns_trivial_feasible_flow() {
    let fixture = load_fixture("bottleneck");
    let outcome = max_flow(
        fixture.spec(),
        SolveOptions {
            max_phases: Some(0),
        },
        CancelToken::new(),
        maxflow_server::logging::RunId::new(),
    )
    .unwrap();

    assert_eq!(outcome.status, SolveStatus::Canceled);
    assert_eq!(outcome.certificate.value, 0);
    assert!(outcome
        .certificate
        .edge_flows
        .iter()
        .all(|edge| edge.flow == 0));
}

fn certificate_with_changed_flow(fixture: &Fixture, edge_id: &str, flow: u64) -> FlowCertificate {
    let outcome = max_flow(
        fixture.spec(),
        SolveOptions::default(),
        CancelToken::new(),
        maxflow_server::logging::RunId::new(),
    )
    .unwrap();
    let mut certificate = outcome.certificate;
    certificate
        .edge_flows
        .iter_mut()
        .find(|edge| edge.edge_id == edge_id)
        .unwrap()
        .flow = flow;
    certificate
}

#[test]
fn witness_names_specific_failure_categories() {
    let bottleneck = load_fixture("bottleneck");
    let over_capacity = certificate_with_changed_flow(&bottleneck, "at", 11);
    let report = verify_flow_certificate(&bottleneck.spec(), &over_capacity, true);
    assert!(!report.valid);
    assert!(report
        .failures
        .contains(&WitnessFailure::FlowExceedsCapacity));

    let disconnected = load_fixture("disconnected");
    let mut zero_flow = max_flow(
        disconnected.spec(),
        SolveOptions::default(),
        CancelToken::new(),
        maxflow_server::logging::RunId::new(),
    )
    .unwrap()
    .certificate;
    zero_flow.value = 1;
    let report = verify_flow_certificate(&disconnected.spec(), &zero_flow, true);
    assert!(report.failures.contains(&WitnessFailure::FlowValueMismatch));

    let mut missing_cut = max_flow(
        bottleneck.spec(),
        SolveOptions::default(),
        CancelToken::new(),
        maxflow_server::logging::RunId::new(),
    )
    .unwrap()
    .certificate;
    missing_cut.cut = None;
    let report = verify_flow_certificate(&bottleneck.spec(), &missing_cut, true);
    assert!(report.failures.contains(&WitnessFailure::CutMissing));
}
