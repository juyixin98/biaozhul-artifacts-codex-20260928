use serde::{Deserialize, Serialize};

use super::errors::{KernelError, WitnessFailure};
use super::graph::Network;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FlowCheck {
    pub flow_value: u64,
    /// Positive balance means net outflow. It is flow_value at source and zero
    /// at internal nodes. Sink has net inflow, represented as `-flow_value`.
    pub balances: Vec<i128>,
    pub source_net_flow: i128,
    pub sink_net_flow: i128,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CutCheck {
    pub source_side: Vec<usize>,
    pub sink_side: Vec<usize>,
    pub cut_capacity: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum OptimalityCertificate {
    Proved {
        flow_check: FlowCheck,
        cut_check: CutCheck,
    },
    NotProved {
        flow_check: FlowCheck,
        reason: String,
    },
}

pub fn verify_flow(
    network: &Network,
    flow_value: u64,
    edge_flows: &[u64],
) -> Result<FlowCheck, KernelError> {
    if edge_flows.len() != network.edges().len() {
        return Err(KernelError::WitnessInvalid {
            reason: WitnessFailure::EdgeFlowCountMismatch {
                expected: network.edges().len(),
                actual: edge_flows.len(),
            },
        });
    }

    for (edge_index, (edge, flow)) in network.edges().iter().zip(edge_flows.iter()).enumerate() {
        if *flow > edge.capacity {
            return Err(KernelError::WitnessInvalid {
                reason: WitnessFailure::FlowExceedsCapacity {
                    edge_index,
                    capacity: edge.capacity,
                    flow: *flow,
                },
            });
        }
    }

    let mut balances = vec![0i128; network.node_count()];
    for (edge, flow) in network.edges().iter().zip(edge_flows.iter()) {
        let signed_flow = i128::try_from(*flow).map_err(|_| KernelError::ArithmeticOverflow)?;
        balances[edge.from] = balances[edge.from]
            .checked_add(signed_flow)
            .ok_or(KernelError::ArithmeticOverflow)?;
        balances[edge.to] = balances[edge.to]
            .checked_sub(signed_flow)
            .ok_or(KernelError::ArithmeticOverflow)?;
    }

    let source_net_flow = balances[network.source()];
    let sink_net_flow = balances[network.sink()];
    let signed_value = i128::try_from(flow_value).map_err(|_| KernelError::ArithmeticOverflow)?;

    if source_net_flow != signed_value {
        return Err(KernelError::WitnessInvalid {
            reason: WitnessFailure::SourceNetFlowMismatch {
                expected: flow_value,
                actual: source_net_flow,
            },
        });
    }
    if sink_net_flow != -signed_value {
        return Err(KernelError::WitnessInvalid {
            reason: WitnessFailure::SinkNetFlowMismatch {
                expected: flow_value,
                actual: sink_net_flow,
            },
        });
    }

    for (node, balance) in balances.iter().copied().enumerate() {
        if node == network.source() || node == network.sink() {
            continue;
        }
        if balance != 0 {
            return Err(KernelError::WitnessInvalid {
                reason: WitnessFailure::BalanceViolation { node, balance },
            });
        }
    }

    Ok(FlowCheck {
        flow_value,
        balances,
        source_net_flow,
        sink_net_flow,
    })
}

pub fn verify_cut(
    network: &Network,
    claimed_flow_value: u64,
    source_side: &[usize],
) -> Result<CutCheck, KernelError> {
    let mut membership = vec![None; network.node_count()];

    for &node in source_side {
        if node >= network.node_count() {
            return Err(KernelError::WitnessInvalid {
                reason: WitnessFailure::CutNodeOutOfRange {
                    node,
                    node_count: network.node_count(),
                },
            });
        }
        if membership[node].is_some() {
            return Err(KernelError::WitnessInvalid {
                reason: WitnessFailure::CutDuplicateNode { node },
            });
        }
        membership[node] = Some(true);
    }

    if membership[network.source()] != Some(true) {
        return Err(KernelError::WitnessInvalid {
            reason: WitnessFailure::CutMissingSource {
                source: network.source(),
            },
        });
    }
    if membership[network.sink()] == Some(true) {
        return Err(KernelError::WitnessInvalid {
            reason: WitnessFailure::CutContainsSink {
                sink: network.sink(),
            },
        });
    }

    let mut cut_capacity: u64 = 0;
    for edge in network.edges() {
        if membership[edge.from] == Some(true) && membership[edge.to].is_none() {
            cut_capacity = cut_capacity
                .checked_add(edge.capacity)
                .ok_or(KernelError::ArithmeticOverflow)?;
        }
    }

    if cut_capacity != claimed_flow_value {
        return Err(KernelError::WitnessInvalid {
            reason: WitnessFailure::CutCapacityMismatch {
                computed: cut_capacity,
                claimed: claimed_flow_value,
            },
        });
    }

    let mut sink_side = Vec::new();
    for (node, on_source_side) in membership.iter().enumerate() {
        match on_source_side {
            Some(true) => {}
            None => sink_side.push(node),
            Some(false) => unreachable!("only source-side nodes are written"),
        }
    }

    if source_side.len() + sink_side.len() != network.node_count() {
        return Err(KernelError::WitnessInvalid {
            reason: WitnessFailure::CutDoesNotCoverAllNodes,
        });
    }

    let mut normalized_source = source_side.to_vec();
    normalized_source.sort_unstable();
    sink_side.sort_unstable();

    Ok(CutCheck {
        source_side: normalized_source,
        sink_side,
        cut_capacity,
    })
}

pub fn verify_optimal(
    network: &Network,
    flow_value: u64,
    edge_flows: &[u64],
    source_side: &[usize],
) -> Result<OptimalityCertificate, KernelError> {
    let flow_check = verify_flow(network, flow_value, edge_flows)?;
    let cut_check = verify_cut(network, flow_value, source_side)?;
    if cut_check.cut_capacity != flow_check.flow_value {
        return Err(KernelError::WitnessInvalid {
            reason: WitnessFailure::FlowCutMismatch {
                flow_value,
                cut_capacity: cut_check.cut_capacity,
            },
        });
    }
    Ok(OptimalityCertificate::Proved {
        flow_check,
        cut_check,
    })
}

pub fn verify_feasible(
    network: &Network,
    flow_value: u64,
    edge_flows: &[u64],
    reason: &str,
) -> Result<OptimalityCertificate, KernelError> {
    let flow_check = verify_flow(network, flow_value, edge_flows)?;
    Ok(OptimalityCertificate::NotProved {
        flow_check,
        reason: reason.to_string(),
    })
}
