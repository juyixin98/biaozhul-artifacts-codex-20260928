mod support;

use directed_maxflow_backend::core::errors::{KernelError, WitnessFailure};
use directed_maxflow_backend::core::graph::{EdgeSpec, Network, NetworkSpec};
use directed_maxflow_backend::core::witness;
use support::{append_log, run_id};

fn tiny_network() -> Network {
    Network::build(NetworkSpec {
        node_count: 4,
        source: 0,
        sink: 3,
        edges: vec![
            EdgeSpec::new(0, 1, 5),
            EdgeSpec::new(0, 2, 5),
            EdgeSpec::new(1, 3, 5),
            EdgeSpec::new(2, 3, 5),
        ],
    })
    .unwrap()
}

#[test]
fn witness_rejects_capacity_violation() {
    let network = tiny_network();
    let run = run_id("witness", "capacity");
    let error = witness::verify_flow(&network, 10, &[6, 4, 5, 5]).unwrap_err();
    match error {
        KernelError::WitnessInvalid {
            reason:
                WitnessFailure::FlowExceedsCapacity {
                    edge_index,
                    capacity,
                    flow,
                },
        } => {
            assert_eq!((edge_index, capacity, flow), (0, 5, 6));
        }
        other => panic!("wrong failure category: {other:?}"),
    }
    append_log(
        &run,
        "witness_failure",
        "flow 6 exceeds the u64 capacity 5 on original edge index 0",
        &serde_json::json!({"flows": [6, 4, 5, 5], "failure": error}),
    );
}

#[test]
fn witness_rejects_conservation_violation() {
    let network = tiny_network();
    let run = run_id("witness", "balance");
    let error = witness::verify_flow(&network, 10, &[5, 5, 4, 5]).unwrap_err();
    match error {
        KernelError::WitnessInvalid {
            reason: WitnessFailure::BalanceViolation { node, balance },
        } => {
            assert_eq!((node, balance), (1, 1));
        }
        other => panic!("wrong failure category: {other:?}"),
    }
    append_log(
        &run,
        "witness_failure",
        "node 1 sends one more unit than it receives, so its net outflow is 1 instead of 0",
        &serde_json::json!({"flows": [5, 5, 4, 5], "failure": error}),
    );
}

#[test]
fn witness_rejects_invalid_cut_side_and_unequal_capacity() {
    let network = tiny_network();
    let run = run_id("witness", "cut_membership");
    let error = witness::verify_cut(&network, 10, &[0, 3]).unwrap_err();
    assert!(matches!(
        error,
        KernelError::WitnessInvalid {
            reason: WitnessFailure::CutContainsSink { sink: 3 }
        }
    ));
    append_log(
        &run,
        "witness_failure",
        "source-side partition contains sink 3, so it cannot define an s-t cut",
        &serde_json::json!({"source_side": [0, 3], "failure": error}),
    );

    let run = run_id("witness", "cut_value");
    let error = witness::verify_cut(&network, 10, &[0, 1]).unwrap_err();
    assert!(matches!(
        error,
        KernelError::WitnessInvalid {
            reason: WitnessFailure::CutCapacityMismatch {
                computed: 5,
                claimed: 10,
            }
        }
    ));
    append_log(
        &run,
        "witness_failure",
        "partition {0,1} has crossing capacity 5 (edge 1->3), which cannot certify flow 10",
        &serde_json::json!({"source_side": [0, 1], "failure": error}),
    );
}

#[test]
fn graph_validation_reports_exact_input_categories() {
    let run = run_id("input", "negative_is_u64_rejected_by_schema");
    let raw = serde_json::json!({
        "node_count": 2,
        "source": 0,
        "sink": 1,
        "edges": [{"from": 0, "to": 1, "capacity": -1}]
    });
    let parse_error = serde_json::from_value::<NetworkSpec>(raw).unwrap_err();
    append_log(
        &run,
        "input_failure",
        "negative signed integer cannot deserialize into the safe u64 capacity type",
        &serde_json::json!({"failure": parse_error.to_string()}),
    );

    assert_eq!(
        Network::build(NetworkSpec {
            node_count: 2,
            source: 0,
            sink: 0,
            edges: Vec::new()
        })
        .unwrap_err(),
        directed_maxflow_backend::core::errors::InputError::SourceEqualsSink { node: 0 }
    );
    assert_eq!(
        Network::build(NetworkSpec {
            node_count: 2,
            source: 3,
            sink: 1,
            edges: Vec::new()
        })
        .unwrap_err(),
        directed_maxflow_backend::core::errors::InputError::SourceOutOfRange {
            source: 3,
            node_count: 2
        }
    );
}
