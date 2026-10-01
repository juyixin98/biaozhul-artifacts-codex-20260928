use crate::error::{AppError, InputErrorCode};
use crate::flow::{Cut, EdgeFlow, FlowCertificate, SolveOptions, SolveOutcome, SolveStatus};
use crate::graph::{EdgeSpec, GraphSpec};
use crate::witness::WitnessReport;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EdgeInput {
    pub id: String,
    pub from: i128,
    pub to: i128,
    pub capacity: i128,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SolveJobRequest {
    pub node_count: i128,
    pub source: i128,
    pub sink: i128,
    pub edges: Vec<EdgeInput>,
    pub max_phases: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct EdgeOutput {
    pub id: String,
    pub flow: u64,
}

#[derive(Debug, Serialize)]
pub struct CutOutput {
    pub source_side: Vec<usize>,
    pub sink_side: Vec<usize>,
}

#[derive(Debug, Serialize)]
pub struct CertificateOutput {
    pub value: u64,
    pub edge_flows: Vec<EdgeOutput>,
    pub cut: Option<CutOutput>,
}

#[derive(Debug, Serialize)]
pub struct JobResponse {
    pub id: String,
    pub run_id: String,
    pub state: &'static str,
    pub optimal: bool,
    pub status: Option<&'static str>,
    pub certificate: Option<CertificateOutput>,
    pub feasibility_valid: Option<bool>,
    pub feasibility_reasons: Option<Vec<String>>,
    pub error: Option<AppError>,
}

#[derive(Debug, Serialize)]
pub struct MessageResponse {
    pub message: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WitnessEdgeInput {
    pub edge_id: String,
    pub flow: i128,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WitnessCutInput {
    pub source_side: Vec<i128>,
    pub sink_side: Vec<i128>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WitnessRequest {
    pub node_count: i128,
    pub source: i128,
    pub sink: i128,
    pub edges: Vec<EdgeInput>,
    pub value: i128,
    pub edge_flows: Vec<WitnessEdgeInput>,
    pub cut: Option<WitnessCutInput>,
}

#[derive(Debug, Serialize)]
pub struct WitnessResponse {
    pub valid: bool,
    pub failures: Vec<String>,
    pub report: WitnessReport,
}

fn non_negative_usize(value: i128, field: &str) -> Result<usize, AppError> {
    if value < 0 {
        return Err(AppError::input(
            InputErrorCode::NodeCountOutOfRange,
            format!("{field} must be non-negative; got {value}"),
        ));
    }
    usize::try_from(value).map_err(|_| {
        AppError::input(
            InputErrorCode::NodeCountOutOfRange,
            format!("{field} is too large for this platform: {value}"),
        )
    })
}

fn capacity(capacity: i128, edge_id: &str) -> Result<u64, AppError> {
    if capacity < 0 {
        return Err(AppError::input(
            InputErrorCode::CapacityOutOfRange,
            format!("edge {edge_id} has negative capacity {capacity}"),
        ));
    }
    u64::try_from(capacity).map_err(|_| {
        AppError::input(
            InputErrorCode::CapacityOutOfRange,
            format!("edge {edge_id} capacity {capacity} exceeds u64::MAX"),
        )
    })
}

impl SolveJobRequest {
    pub fn into_spec(self) -> Result<(GraphSpec, SolveOptions), AppError> {
        let node_count = non_negative_usize(self.node_count, "node_count")?;
        let source = non_negative_usize(self.source, "source")?;
        let sink = non_negative_usize(self.sink, "sink")?;

        let mut edge_ids = Vec::with_capacity(self.edges.len());
        let mut edges = Vec::with_capacity(self.edges.len());
        for edge in self.edges {
            let capacity = capacity(edge.capacity, &edge.id)?;
            let from = non_negative_usize(edge.from, "edge.from")?;
            let to = non_negative_usize(edge.to, "edge.to")?;
            edge_ids.push(edge.id);
            edges.push(EdgeSpec { from, to, capacity });
        }

        let spec = GraphSpec {
            node_count,
            source,
            sink,
            edge_ids,
            edges,
        };
        spec.validate()?;

        Ok((
            spec,
            SolveOptions {
                max_phases: self.max_phases,
            },
        ))
    }
}

impl WitnessRequest {
    pub fn into_parts(self) -> Result<(GraphSpec, FlowCertificate), AppError> {
        let node_count = non_negative_usize(self.node_count, "node_count")?;
        let source = non_negative_usize(self.source, "source")?;
        let sink = non_negative_usize(self.sink, "sink")?;
        let value = if self.value < 0 {
            return Err(AppError::input(
                InputErrorCode::CapacityOutOfRange,
                format!("flow value must be non-negative; got {}", self.value),
            ));
        } else {
            u64::try_from(self.value).map_err(|_| {
                AppError::input(
                    InputErrorCode::CapacityOutOfRange,
                    format!("flow value {} exceeds u64::MAX", self.value),
                )
            })?
        };

        let mut edge_ids = Vec::new();
        let mut edges = Vec::new();
        for edge in self.edges {
            edge_ids.push(edge.id.clone());
            edges.push(EdgeSpec {
                from: non_negative_usize(edge.from, "edge.from")?,
                to: non_negative_usize(edge.to, "edge.to")?,
                capacity: capacity(edge.capacity, &edge.id)?,
            });
        }

        let mut edge_flows = Vec::new();
        for edge in self.edge_flows {
            if edge.flow < 0 {
                return Err(AppError::input(
                    InputErrorCode::CapacityOutOfRange,
                    format!("flow on edge {} is negative", edge.edge_id),
                ));
            }
            let flow = u64::try_from(edge.flow).map_err(|_| {
                AppError::input(
                    InputErrorCode::CapacityOutOfRange,
                    format!("flow on edge {} exceeds u64::MAX", edge.edge_id),
                )
            })?;
            edge_flows.push(EdgeFlow {
                edge_id: edge.edge_id,
                flow,
            });
        }

        let cut = match self.cut {
            None => None,
            Some(cut) => Some(Cut {
                source_side: cut
                    .source_side
                    .into_iter()
                    .map(|node| non_negative_usize(node, "cut node"))
                    .collect::<Result<Vec<_>, _>>()?,
                sink_side: cut
                    .sink_side
                    .into_iter()
                    .map(|node| non_negative_usize(node, "cut node"))
                    .collect::<Result<Vec<_>, _>>()?,
            }),
        };

        let spec = GraphSpec {
            node_count,
            source,
            sink,
            edge_ids,
            edges,
        };
        spec.validate()?;

        Ok((
            spec,
            FlowCertificate {
                value,
                edge_flows,
                cut,
            },
        ))
    }
}

pub fn outcome_to_certificate(outcome: &SolveOutcome) -> CertificateOutput {
    CertificateOutput {
        value: outcome.certificate.value,
        edge_flows: outcome
            .certificate
            .edge_flows
            .iter()
            .map(|edge| EdgeOutput {
                id: edge.edge_id.clone(),
                flow: edge.flow,
            })
            .collect(),
        cut: outcome.certificate.cut.as_ref().map(|cut| CutOutput {
            source_side: cut.source_side.clone(),
            sink_side: cut.sink_side.clone(),
        }),
    }
}

pub fn status_name(status: SolveStatus) -> &'static str {
    match status {
        SolveStatus::Completed => "completed",
        SolveStatus::Canceled => "canceled",
    }
}
