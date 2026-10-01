mod support;

use directed_maxflow_backend::core::dinic::{CancellationToken, Dinic, SolveLimits, SolveOutcome};
use directed_maxflow_backend::core::graph::{EdgeSpec, Network, NetworkSpec};
use directed_maxflow_backend::core::progress::ProgressEvent;
use directed_maxflow_backend::core::witness;
use serde_json::Value;
use support::{append_log, run_id};

#[derive(Debug)]
struct Case {
    id: String,
    network: NetworkSpec,
    expected_flow: u64,
    expected_source_side: Vec<usize>,
    expected_sink_side: Vec<usize>,
    expected_reverse_event: Option<Value>,
}

fn load_cases() -> Vec<Case> {
    let raw = std::fs::read_to_string("fixtures/cases.json").expect("fixture cases");
    let values: Vec<Value> = serde_json::from_str(&raw).expect("parse fixture cases");
    values
        .into_iter()
        .map(|value| Case {
            id: value["id"].as_str().expect("case id").to_string(),
            network: serde_json::from_value(value["network"].clone()).expect("network fixture"),
            expected_flow: value["expected"]["flow_value"]
                .as_u64()
                .expect("expected flow"),
            expected_source_side: serde_json::from_value(value["expected"]["source_side"].clone())
                .expect("source side"),
            expected_sink_side: serde_json::from_value(value["expected"]["sink_side"].clone())
                .expect("sink side"),
            expected_reverse_event: value.get("expected_reverse_event").cloned(),
        })
        .collect()
}

#[test]
fn fixtures_match_independent_exhaustive_cut_oracle() {
    for case in load_cases() {
        let id = run_id("core_exhaustive", &case.id);
        let network = Network::build(case.network.clone()).expect("valid fixture network");
        let mut solver = Dinic::new(&network);
        let outcome = solver
            .solve(
                &id,
                SolveLimits::unbounded(),
                &CancellationToken::new(),
                &|_| {},
            )
            .expect("solve fixture");
        let result = outcome.result();

        let independent_cut = exhaustive_min_cut(&case.network);
        assert_eq!(result.flow_value, independent_cut.capacity, "{}", case.id);
        assert_eq!(result.flow_value, case.expected_flow, "{}", case.id);
        assert!(
            independent_cut
                .source_sides
                .contains(result.source_side.as_ref().expect("complete cut")),
            "{} cut was not one of the independently enumerated minimum cuts",
            case.id
        );
        assert_eq!(
            result.source_side.as_deref(),
            Some(case.expected_source_side.as_slice()),
            "{}",
            case.id
        );
        assert_eq!(
            result.sink_side.as_deref(),
            Some(case.expected_sink_side.as_slice()),
            "{}",
            case.id
        );

        witness::verify_optimal(
            &network,
            result.flow_value,
            &result.edge_flows,
            result.source_side.as_ref().expect("cut"),
        )
        .expect("independent flow and cut certificate");

        append_log(
            &id,
            "core_case",
            "flow equals the minimum independently enumerated cut capacity and the witness checks capacity, conservation, partition, and equality",
            &serde_json::json!({
                "case_id": case.id,
                "network": case.network,
                "flow_value": result.flow_value,
                "source_side": result.source_side,
                "sink_side": result.sink_side,
                "edge_flows": result.edge_flows,
                "all_min_cut_source_sides": independent_cut.source_sides,
                "events": result.events,
            }),
        );
    }
}

#[test]
fn reverse_reroute_logs_a_reverse_residual_step() {
    let case = load_cases()
        .into_iter()
        .find(|case| case.id == "reverse_reroute")
        .expect("reverse case");
    let id = run_id("core_reverse", &case.id);
    let network = Network::build(case.network.clone()).unwrap();
    let mut solver = Dinic::new(&network);
    let outcome = solver
        .solve(
            &id,
            SolveLimits::unbounded(),
            &CancellationToken::new(),
            &|_| {},
        )
        .unwrap();
    let result = outcome.result();

    let reverse_step = result
        .events
        .iter()
        .flat_map(|event| match event {
            ProgressEvent::PathFound { residual_path, .. } => residual_path.clone(),
            _ => Vec::new(),
        })
        .find(|step| step.is_reverse && step.original_edge == 2 && step.from == 1 && step.to == 2)
        .expect("reverse residual withdrawal is explicitly logged");

    append_log(
        &id,
        "reverse_withdrawal",
        "a shortest residual path traverses residual edge 2 from 1 to 2, proving previous flow was undone rather than only sending additional forward flow",
        &serde_json::json!({
            "case_id": case.id,
            "reverse_step": reverse_step,
            "final_flows": result.edge_flows,
            "events": result.events,
        }),
    );
}

#[test]
fn cancellation_returns_feasible_flow_without_optimality_claim() {
    let raw = std::fs::read_to_string("fixtures/cancel_network.json").expect("cancel fixture");
    let fixture: Value = serde_json::from_str(&raw).unwrap();
    let spec: NetworkSpec = serde_json::from_value(fixture["network"].clone()).unwrap();
    let expected_flows: Vec<u64> =
        serde_json::from_value(fixture["expected_partial"]["edge_flows"].clone()).unwrap();
    let id = run_id("cancel", fixture["id"].as_str().unwrap());
    let network = Network::build(spec.clone()).unwrap();
    let token = CancellationToken::new();
    let cancel_token = token.clone();
    let mut augmentation_count = 0;

    let outcome = Dinic::new(&network)
        .solve(&id, SolveLimits::unbounded(), &token, &move |event| {
            if let ProgressEvent::PathFound {
                augmentation,
                flow_value,
                ..
            } = event
            {
                augmentation_count = *augmentation;
                if *flow_value == 4 && *augmentation == 2 {
                    cancel_token.cancel();
                }
            }
        })
        .expect("cancellation is a controlled outcome");

    assert!(matches!(outcome, SolveOutcome::Canceled(_)));
    let result = outcome.result();
    assert_eq!(result.flow_value, 4);
    assert_eq!(result.edge_flows, expected_flows);
    assert_eq!(result.source_side, None);
    assert_eq!(result.sink_side, None);
    assert_eq!(augmentation_count, 2);
    let certificate = witness::verify_feasible(
        &network,
        result.flow_value,
        &result.edge_flows,
        "test cancellation",
    )
    .unwrap();
    assert!(matches!(
        certificate,
        witness::OptimalityCertificate::NotProved { .. }
    ));

    append_log(
        &id,
        "cancellation",
        "cancel was observed after two augmentations; returned flow is feasible but no source/sink cut is claimed",
        &serde_json::json!({
            "network": spec,
            "flow_value": result.flow_value,
            "edge_flows": result.edge_flows,
            "events": result.events,
        }),
    );
}

#[test]
fn arithmetic_overflow_is_computation_failure_not_input_failure() {
    let spec = NetworkSpec {
        node_count: 4,
        source: 0,
        sink: 3,
        edges: vec![
            EdgeSpec::new(0, 1, u64::MAX),
            EdgeSpec::new(0, 2, u64::MAX),
            EdgeSpec::new(1, 3, u64::MAX),
            EdgeSpec::new(2, 3, u64::MAX),
        ],
    };
    let id = run_id("overflow", "two_u64_max_paths");
    let network = Network::build(spec.clone()).unwrap();
    let error = Dinic::new(&network)
        .solve(
            &id,
            SolveLimits::unbounded(),
            &CancellationToken::new(),
            &|_| {},
        )
        .expect_err("two u64::MAX paths overflow a u64 total");

    assert!(matches!(
        error,
        directed_maxflow_backend::core::errors::KernelError::ArithmeticOverflow
    ));
    append_log(
        &id,
        "computation_failure",
        "each input capacity is legal, but summing two maxima exceeds the safe u64 result domain",
        &serde_json::json!({ "network": spec, "error": error }),
    );
}

#[derive(Debug)]
struct CutOracle {
    capacity: u64,
    source_sides: Vec<Vec<usize>>,
}

/// Independent brute-force oracle: enumerate every binary node partition with
/// source on the left and sink on the right and sum original forward edges
/// crossing from left to right.
fn exhaustive_min_cut(spec: &NetworkSpec) -> CutOracle {
    let mut minimum = u64::MAX;
    let mut source_sides = Vec::new();
    let masks = 1u64 << spec.node_count;

    for mask in 0..masks {
        let source_side = (mask >> spec.source) & 1 == 1;
        let sink_side = (mask >> spec.sink) & 1 == 1;
        if !source_side || sink_side {
            continue;
        }

        let capacity = spec
            .edges
            .iter()
            .filter(|edge| ((mask >> edge.from) & 1 == 1) && ((mask >> edge.to) & 1 == 0))
            .try_fold(0u64, |sum, edge: &EdgeSpec| sum.checked_add(edge.capacity))
            .expect("small fixture cut capacity fits u64");
        let nodes: Vec<usize> = (0..spec.node_count)
            .filter(|node| (mask >> node) & 1 == 1)
            .collect();

        match capacity.cmp(&minimum) {
            std::cmp::Ordering::Less => {
                minimum = capacity;
                source_sides = vec![nodes];
            }
            std::cmp::Ordering::Equal => source_sides.push(nodes),
            std::cmp::Ordering::Greater => {}
        }
    }

    CutOracle {
        capacity: minimum,
        source_sides,
    }
}
