use uuid::Uuid;

use crate::core::dinic::{CancellationToken, Dinic, SolveOutcome};
use crate::core::errors::{KernelError, ServiceError};
use crate::core::graph::Network;
use crate::core::progress::SolveLimits;
use crate::core::witness;

use super::contracts::{ComputeRequest, EdgeFlowResponse, FlowResultResponse};

pub fn new_run_id() -> String {
    Uuid::new_v4().to_string()
}

pub fn solve_request(
    request: ComputeRequest,
    cancellation: &CancellationToken,
    run_id: &str,
    observer: &dyn Fn(&crate::core::progress::ProgressEvent),
) -> Result<FlowResultResponse, ServiceError> {
    let limits = SolveLimits {
        max_phases: request.limits.max_phases,
        max_augmentations: request.limits.max_augmentations,
    };
    let network = Network::build(request.network.clone())?;
    let mut solver = Dinic::new(&network);
    let outcome = solver.solve(run_id, limits, cancellation, observer)?;
    let result = outcome.result();

    let certificate = match outcome {
        SolveOutcome::Complete(_) => {
            let source_side = result.source_side.clone().ok_or_else(|| {
                ServiceError::ComputationFailed(KernelError::InternalError {
                    message: "completed run did not produce a source-side cut".to_string(),
                })
            })?;
            witness::verify_optimal(
                &network,
                result.flow_value,
                &result.edge_flows,
                &source_side,
            )?
        }
        SolveOutcome::Canceled(_) => witness::verify_feasible(
            &network,
            result.flow_value,
            &result.edge_flows,
            "canceled before optimality was proved; no cut certificate is returned",
        )?,
    };

    Ok(FlowResultResponse {
        run_id: run_id.to_string(),
        network: network.spec(),
        edges: EdgeFlowResponse::from_network(&network.spec(), &result.edge_flows),
        flow_value: result.flow_value,
        certificate,
        statistics: result.statistics.clone(),
        progress: result.events.clone(),
    })
}
