use std::sync::Arc;

use crate::service::job_store::JobStore;

#[derive(Clone)]
pub struct AppState {
    pub jobs: Arc<JobStore>,
}

impl AppState {
    #[must_use]
    pub fn new(max_jobs: usize) -> Self {
        Self {
            jobs: Arc::new(JobStore::new(max_jobs)),
        }
    }
}
