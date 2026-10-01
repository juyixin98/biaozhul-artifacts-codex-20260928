use crate::flow::{Cut, FlowCertificate};
use crate::graph::{Capacity, GraphSpec, Node};
use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WitnessFailure {
    EdgeCountMismatch,
    EdgeIdMismatch,
    FlowExceedsCapacity,
    FlowConservationFailed,
    FlowValueMismatch,
    CutMissing,
    CutNotPartition,
    CutNodeOutOfRange,
    CutSidesOverlap,
    CutEdgeFromSinkToSource,
    CutCapacityOverflow,
    CutCapacityMismatch,
}

impl WitnessFailure {
    pub fn explanation(self) -> &'static str {
        match self {
            Self::EdgeCountMismatch => {
                "the certificate must contain one flow for each original edge"
            }
            Self::EdgeIdMismatch => "certificate edge ids must match graph edge ids positionally",
            Self::FlowExceedsCapacity => "an edge carries more flow than its non-negative capacity",
            Self::FlowConservationFailed => "an internal node violates flow conservation",
            Self::FlowValueMismatch => "the declared value does not equal net source outflow",
            Self::CutMissing => "an optimal certificate must include both cut sides",
            Self::CutNotPartition => "cut sides must partition every node exactly once",
            Self::CutNodeOutOfRange => "a cut node id is outside the graph",
            Self::CutSidesOverlap => "a node appears on both cut sides",
            Self::CutEdgeFromSinkToSource => "a T-to-S edge carries residual reverse flow",
            Self::CutCapacityOverflow => "summing cut capacity overflowed the safe integer type",
            Self::CutCapacityMismatch => "max-flow value and min-cut capacity differ",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct NodeBalance {
    pub node: Node,
    pub inflow: u128,
    pub outflow: u128,
    pub net: i128,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FlowReport {
    pub value: Capacity,
    pub balances: Vec<NodeBalance>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CutReport {
    pub forward_edges: Vec<String>,
    pub reverse_edges: Vec<String>,
    pub capacity: Capacity,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct WitnessReport {
    pub valid: bool,
    pub failures: Vec<WitnessFailure>,
    pub flow: Option<FlowReport>,
    pub cut: Option<CutReport>,
}

impl WitnessReport {
    pub fn failure_strings(&self) -> Vec<String> {
        self.failures
            .iter()
            .map(|failure| failure.explanation().to_string())
            .collect()
    }
}

fn signed_add(target: &mut i128, value: u64, sign: i128) -> Option<()> {
    *target = target.checked_add(sign * i128::from(value))?;
    Some(())
}

pub fn verify_flow_certificate(
    spec: &GraphSpec,
    certificate: &FlowCertificate,
    require_cut: bool,
) -> WitnessReport {
    let mut failures = Vec::new();

    if certificate.edge_flows.len() != spec.edges.len() {
        failures.push(WitnessFailure::EdgeCountMismatch);
    }

    for ((certificate_edge, edge_id), edge) in certificate
        .edge_flows
        .iter()
        .zip(spec.edge_ids.iter())
        .zip(spec.edges.iter())
    {
        if &certificate_edge.edge_id != edge_id {
            failures.push(WitnessFailure::EdgeIdMismatch);
        }
        if certificate_edge.flow > edge.capacity {
            failures.push(WitnessFailure::FlowExceedsCapacity);
        }
    }

    let mut inflow = vec![0u128; spec.node_count];
    let mut outflow = vec![0u128; spec.node_count];
    for (certificate_edge, edge) in certificate.edge_flows.iter().zip(spec.edges.iter()) {
        outflow[edge.from] = outflow[edge.from].saturating_add(u128::from(certificate_edge.flow));
        inflow[edge.to] = inflow[edge.to].saturating_add(u128::from(certificate_edge.flow));
    }

    let mut balances = Vec::with_capacity(spec.node_count);
    let mut conservation_ok = true;
    let mut source_net: i128 = 0;
    let mut sink_net: i128 = 0;
    for node in 0..spec.node_count {
        let net = i128::try_from(inflow[node]).ok().and_then(|inflow| {
            i128::try_from(outflow[node])
                .ok()
                .and_then(|outflow| outflow.checked_sub(inflow))
        });
        let net = match net {
            Some(net) => net,
            None => {
                conservation_ok = false;
                i128::MAX
            }
        };

        if node == spec.source {
            source_net = net;
        } else if node == spec.sink {
            sink_net = net;
        } else if net != 0 {
            conservation_ok = false;
        }
        balances.push(NodeBalance {
            node,
            inflow: inflow[node],
            outflow: outflow[node],
            net,
        });
    }

    if !conservation_ok {
        failures.push(WitnessFailure::FlowConservationFailed);
    }

    let flow_value = if source_net >= 0 {
        u64::try_from(source_net).ok()
    } else {
        None
    };
    match flow_value {
        Some(value) if value == certificate.value && source_net + sink_net == 0 => {}
        _ => failures.push(WitnessFailure::FlowValueMismatch),
    }

    let mut cut_report = None;
    if require_cut {
        match certificate.cut.as_ref() {
            None => failures.push(WitnessFailure::CutMissing),
            Some(cut) => {
                verify_cut(spec, certificate, cut, &mut failures, &mut cut_report);
            }
        }
    }

    WitnessReport {
        valid: failures.is_empty(),
        failures,
        flow: Some(FlowReport {
            value: certificate.value,
            balances,
        }),
        cut: cut_report,
    }
}

fn verify_cut(
    spec: &GraphSpec,
    certificate: &FlowCertificate,
    cut: &Cut,
    failures: &mut Vec<WitnessFailure>,
    cut_report: &mut Option<CutReport>,
) {
    let mut side = vec![None; spec.node_count];
    let mut partition_valid = true;

    for node in &cut.source_side {
        if *node >= spec.node_count {
            failures.push(WitnessFailure::CutNodeOutOfRange);
            partition_valid = false;
        } else if side[*node].is_some() {
            failures.push(WitnessFailure::CutNotPartition);
            partition_valid = false;
        } else {
            side[*node] = Some(true);
        }
    }
    for node in &cut.sink_side {
        if *node >= spec.node_count {
            failures.push(WitnessFailure::CutNodeOutOfRange);
            partition_valid = false;
        } else if side[*node].is_some() {
            failures.push(WitnessFailure::CutSidesOverlap);
            partition_valid = false;
        } else {
            side[*node] = Some(false);
        }
    }

    if side.iter().any(Option::is_none) || !partition_valid {
        failures.push(WitnessFailure::CutNotPartition);
    }
    if side[spec.source] != Some(true) {
        failures.push(WitnessFailure::CutNotPartition);
    }
    if side[spec.sink] != Some(false) {
        failures.push(WitnessFailure::CutNotPartition);
    }

    let mut capacity: u128 = 0;
    let mut forward_edges = Vec::new();
    let mut reverse_edges = Vec::new();
    for (edge_id, edge) in spec.edge_ids.iter().zip(spec.edges.iter()) {
        let from_source = side[edge.from] == Some(true);
        let to_source = side[edge.to] == Some(true);
        if from_source && !to_source {
            capacity = capacity.saturating_add(u128::from(edge.capacity));
            forward_edges.push(edge_id.clone());
        }
        if !from_source && to_source {
            let Some(flow_entry) = certificate
                .edge_flows
                .iter()
                .find(|entry| &entry.edge_id == edge_id)
            else {
                failures.push(WitnessFailure::EdgeIdMismatch);
                continue;
            };
            if flow_entry.flow != 0 {
                failures.push(WitnessFailure::CutEdgeFromSinkToSource);
            } else {
                reverse_edges.push(edge_id.clone());
            }
        }
    }

    match u64::try_from(capacity) {
        Ok(capacity_u64) => {
            if capacity_u64 != certificate.value {
                failures.push(WitnessFailure::CutCapacityMismatch);
            }
            *cut_report = Some(CutReport {
                forward_edges,
                reverse_edges,
                capacity: capacity_u64,
            });
        }
        Err(_) => {
            failures.push(WitnessFailure::CutCapacityOverflow);
        }
    }
}

#[allow(dead_code)]
fn type_check_for_signed_add() {
    let mut value = 0i128;
    let _ = signed_add(&mut value, 1, 1);
}
