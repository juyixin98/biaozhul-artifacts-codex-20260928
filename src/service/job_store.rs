use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::core::dinic::CancellationToken;
use crate::core::errors::{ErrorCategory, ResourceLimit, ServiceError};
use crate::core::graph::Network;
use crate::core::witness;

use super::contracts::{CreateJobRequest, FlowResultResponse, JobResponse};
use super::flow_service::{new_run_id, solve_request};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JobStatus {
    Queued,
    Running,
    CancellationRequested,
    Completed,
    Canceled,
    Failed,
}

#[derive(Clone)]
pub struct JobStore {
    inner: Arc<Mutex<HashMap<String, Job>>>,
    max_jobs: usize,
}

struct Job {
    request: CreateJobRequest,
    status: JobStatus,
    cancellation: CancellationToken,
    result: Option<FlowResultResponse>,
    failure: Option<FailureRecord>,
    progress: Arc<Mutex<Vec<crate::core::progress::ProgressEvent>>>,
}

#[derive(Debug, Clone)]
struct FailureRecord {
    category: ErrorCategory,
    error: ServiceError,
}

impl JobStore {
    #[must_use]
    pub fn new(max_jobs: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(HashMap::new())),
            max_jobs,
        }
    }

    pub fn create(&self, request: CreateJobRequest) -> Result<(String, JobResponse), ServiceError> {
        // Validate before consuming a job slot.
        let network = Network::build(request.compute.network.clone())?;
        drop(network);

        let run_id = new_run_id();
        let mut jobs = self.inner.lock().expect("job store mutex is not poisoned");
        if jobs.len() >= self.max_jobs {
            return Err(ServiceError::ResourceExhausted(ResourceLimit {
                name: "jobs".to_string(),
                limit: self.max_jobs as u64,
                actual: jobs.len() as u64,
            }));
        }

        let cancellation = CancellationToken::new();
        let job = Job {
            request: request.clone(),
            status: JobStatus::Queued,
            cancellation: cancellation.clone(),
            result: None,
            failure: None,
            progress: Arc::new(Mutex::new(Vec::new())),
        };
        jobs.insert(run_id.clone(), job);
        drop(jobs);

        let response = JobResponse::Queued {
            run_id: run_id.clone(),
            network: request.compute.network.clone(),
        };

        if request.autostart {
            self.start(&run_id)?;
        }

        Ok((run_id, response))
    }

    pub fn start(&self, run_id: &str) -> Result<JobStatus, ServiceError> {
        let (request, cancellation, should_cancel) = {
            let mut jobs = self.inner.lock().expect("job store mutex is not poisoned");
            let Some(job) = jobs.get_mut(run_id) else {
                return Err(ServiceError::JobNotFound {
                    run_id: run_id.to_string(),
                });
            };

            match job.status {
                JobStatus::Queued => {
                    job.status = JobStatus::Running;
                    (job.request.clone(), job.cancellation.clone(), false)
                }
                JobStatus::CancellationRequested => {
                    job.status = JobStatus::Running;
                    (job.request.clone(), job.cancellation.clone(), true)
                }
                other => {
                    return Err(ServiceError::InvalidState {
                        run_id: run_id.to_string(),
                        message: format!("job cannot start from status {other:?}"),
                    })
                }
            }
        };
        if should_cancel {
            cancellation.cancel();
        }

        let progress_handle = {
            let jobs = self.inner.lock().expect("job store mutex is not poisoned");
            jobs.get(run_id)
                .expect("just-started job exists")
                .progress
                .clone()
        };

        let store = self.clone();
        let run_id_owned = run_id.to_string();
        tokio::task::spawn_blocking(move || {
            let solve_result =
                solve_request(request.compute, &cancellation, &run_id_owned, &|event| {
                    if let Ok(mut progress) = progress_handle.lock() {
                        progress.push(event.clone());
                    }
                });
            let mut jobs = store.inner.lock().expect("job store mutex is not poisoned");
            if let Some(job) = jobs.get_mut(&run_id_owned) {
                match solve_result {
                    Ok(result) => {
                        let status = match result.certificate {
                            crate::core::witness::OptimalityCertificate::Proved { .. } => {
                                JobStatus::Completed
                            }
                            crate::core::witness::OptimalityCertificate::NotProved { .. } => {
                                JobStatus::Canceled
                            }
                        };
                        job.status = status;
                        job.result = Some(result);
                    }
                    Err(error) => {
                        let category = error.category();
                        job.status = JobStatus::Failed;
                        job.failure = Some(FailureRecord {
                            category,
                            error: error.clone(),
                        });
                    }
                }
            }
        });

        Ok(JobStatus::Running)
    }

    pub fn request_cancellation(&self, run_id: &str) -> Result<JobResponse, ServiceError> {
        let mut jobs = self.inner.lock().expect("job store mutex is not poisoned");
        let Some(job) = jobs.get_mut(run_id) else {
            return Err(ServiceError::JobNotFound {
                run_id: run_id.to_string(),
            });
        };

        match job.status {
            JobStatus::Queued => {
                job.status = JobStatus::Canceled;
                let response = immediate_canceled_response(
                    run_id,
                    &job.request.compute,
                    "canceled while queued; no computation was started",
                );
                job.result = Some(response.clone());
                Ok(JobResponse::Canceled(response))
            }
            JobStatus::Running => {
                job.status = JobStatus::CancellationRequested;
                job.cancellation.cancel();
                let progress = job.progress.lock().expect("progress mutex valid").clone();
                Ok(JobResponse::CancellationRequested {
                    run_id: run_id.to_string(),
                    network: job.request.compute.network.clone(),
                    progress,
                })
            }
            JobStatus::CancellationRequested => Ok(JobResponse::CancellationRequested {
                run_id: run_id.to_string(),
                network: job.request.compute.network.clone(),
                progress: job.progress.lock().expect("progress mutex valid").clone(),
            }),
            JobStatus::Completed | JobStatus::Canceled | JobStatus::Failed => {
                Err(ServiceError::InvalidState {
                    run_id: run_id.to_string(),
                    message: "job is already terminal".to_string(),
                })
            }
        }
    }

    pub fn get(&self, run_id: &str) -> Result<(JobStatus, JobResponse), ServiceError> {
        let jobs = self.inner.lock().expect("job store mutex is not poisoned");
        let Some(job) = jobs.get(run_id) else {
            return Err(ServiceError::JobNotFound {
                run_id: run_id.to_string(),
            });
        };

        let response = match job.status {
            JobStatus::Queued => JobResponse::Queued {
                run_id: run_id.to_string(),
                network: job.request.compute.network.clone(),
            },
            JobStatus::Running => JobResponse::Running {
                run_id: run_id.to_string(),
                network: job.request.compute.network.clone(),
                progress: job.progress.lock().expect("progress mutex valid").clone(),
            },
            JobStatus::CancellationRequested => JobResponse::CancellationRequested {
                run_id: run_id.to_string(),
                network: job.request.compute.network.clone(),
                progress: job.progress.lock().expect("progress mutex valid").clone(),
            },
            JobStatus::Completed | JobStatus::Canceled => {
                let result = job
                    .result
                    .clone()
                    .expect("terminal successful job has a result");
                if job.status == JobStatus::Completed {
                    JobResponse::Completed(result)
                } else {
                    JobResponse::Canceled(result)
                }
            }
            JobStatus::Failed => {
                let failure = job
                    .failure
                    .clone()
                    .expect("failed job has a failure record");
                JobResponse::Failed {
                    run_id: run_id.to_string(),
                    network: job.request.compute.network.clone(),
                    category: failure.category.as_str().to_string(),
                    error: serde_json::to_value(&failure.error).expect("service error serializes"),
                }
            }
        };

        Ok((job.status, response))
    }
}

fn immediate_canceled_response(
    run_id: &str,
    request: &super::contracts::ComputeRequest,
    reason: &str,
) -> FlowResultResponse {
    let network = Network::build(request.network.clone()).expect("request was validated");
    let edge_flows = vec![0u64; network.edges().len()];
    let certificate =
        witness::verify_feasible(&network, 0, &edge_flows, reason).expect("zero flow is feasible");

    FlowResultResponse {
        run_id: run_id.to_string(),
        network: network.spec(),
        edges: super::contracts::EdgeFlowResponse::from_network(&network.spec(), &edge_flows),
        flow_value: 0,
        certificate,
        statistics: crate::core::dinic::SolveStatistics {
            phases: 0,
            augmentations: 0,
        },
        progress: vec![crate::core::progress::ProgressEvent::Canceled {
            run_id: run_id.to_string(),
            flow_value: 0,
            phases: 0,
            augmentations: 0,
            reason: reason.to_string(),
        }],
    }
}
