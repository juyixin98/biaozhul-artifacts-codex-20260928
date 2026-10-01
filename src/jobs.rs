use crate::dto::{outcome_to_certificate, status_name, JobResponse, SolveJobRequest};
use crate::error::{AppError, ComputationErrorCode, ResourceErrorCode, MAX_ACTIVE_JOBS};
use crate::flow::{max_flow, CancelToken, SolveError, SolveOutcome};
use crate::logging::RunId;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

#[derive(Clone)]
pub struct SharedRegistry {
    pub inner: Arc<Mutex<HashMap<String, JobEntry>>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JobState {
    Running,
    Completed,
    Canceled,
    Failed,
}

pub struct JobEntry {
    pub state: JobState,
    pub cancel: CancelToken,
    pub outcome: Option<SolveOutcome>,
    pub error: Option<AppError>,
    pub created_at: Instant,
    pub run_id: String,
}

impl SharedRegistry {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn submit(&self, request: SolveJobRequest) -> Result<String, AppError> {
        let (spec, options) = request.into_spec()?;
        let run_id = RunId::new();
        let job_id = format!("job-{}", run_id);
        let cancel = CancelToken::new();

        let mut jobs = self.lock()?;
        let active_count = jobs
            .values()
            .filter(|entry| entry.state == JobState::Running)
            .count();
        if active_count >= MAX_ACTIVE_JOBS {
            return Err(AppError::resource(
                ResourceErrorCode::JobRegistryFull,
                format!("more than {MAX_ACTIVE_JOBS} jobs are already running"),
            ));
        }

        jobs.insert(
            job_id.clone(),
            JobEntry {
                state: JobState::Running,
                cancel: cancel.clone(),
                outcome: None,
                error: None,
                created_at: Instant::now(),
                run_id: run_id.to_string(),
            },
        );
        drop(jobs);

        let registry = self.clone();
        let submitted_job_id = job_id.clone();
        tokio::task::spawn_blocking(move || {
            let result = max_flow(spec, options, cancel, run_id);
            let mut jobs = match registry.lock() {
                Ok(jobs) => jobs,
                Err(_) => return,
            };
            let Some(entry) = jobs.get_mut(&submitted_job_id) else {
                return;
            };
            match result {
                Ok(outcome) => {
                    entry.state = match outcome.status {
                        crate::flow::SolveStatus::Completed => JobState::Completed,
                        crate::flow::SolveStatus::Canceled => JobState::Canceled,
                    };
                    entry.outcome = Some(outcome);
                }
                Err(solve_error) => {
                    entry.state = JobState::Failed;
                    entry.error = Some(match solve_error {
                        SolveError::InvalidGraph(app_error) => app_error,
                        SolveError::CapacityOverflow => AppError::computation(
                            ComputationErrorCode::CapacityOverflow,
                            solve_error.message(),
                        ),
                        SolveError::WitnessCheckFailed(_) => AppError::computation(
                            ComputationErrorCode::WitnessCheckFailed,
                            solve_error.message(),
                        ),
                    });
                }
            }
        });

        Ok(job_id)
    }

    pub fn cancel(&self, job_id: &str) -> Result<(), AppError> {
        let mut jobs = self.lock()?;
        let entry = jobs
            .get_mut(job_id)
            .ok_or_else(|| AppError::not_found(format!("job {job_id} does not exist")))?;
        if entry.state != JobState::Running {
            return Err(AppError::conflict(format!(
                "job {job_id} is no longer running"
            )));
        }
        entry.cancel.cancel();
        Ok(())
    }

    pub fn response(&self, job_id: &str) -> Result<JobResponse, AppError> {
        let jobs = self.lock()?;
        let entry = jobs
            .get(job_id)
            .ok_or_else(|| AppError::not_found(format!("job {job_id} does not exist")))?;
        Ok(entry.response(job_id))
    }

    fn lock(&self) -> Result<std::sync::MutexGuard<'_, HashMap<String, JobEntry>>, AppError> {
        self.inner.lock().map_err(|_| {
            AppError::computation(
                ComputationErrorCode::InternalStatePoisoned,
                "job registry mutex was poisoned",
            )
        })
    }
}

impl JobEntry {
    pub fn response(&self, job_id: &str) -> JobResponse {
        let (state, optimal, status) = match self.state {
            JobState::Running => ("running", false, None),
            JobState::Completed => (
                "completed",
                true,
                Some(status_name(crate::flow::SolveStatus::Completed)),
            ),
            JobState::Canceled => (
                "canceled",
                false,
                Some(status_name(crate::flow::SolveStatus::Canceled)),
            ),
            JobState::Failed => ("failed", false, None),
        };

        JobResponse {
            id: job_id.to_string(),
            run_id: self.run_id.clone(),
            state,
            optimal,
            status,
            certificate: self.outcome.as_ref().map(outcome_to_certificate),
            feasibility_valid: self.outcome.as_ref().map(|_| true),
            feasibility_reasons: self.outcome.as_ref().map(|_| Vec::new()),
            error: self.error.clone(),
        }
    }
}
