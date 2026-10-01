mod dtos;
mod handlers;
mod state;

pub use state::AppState;

use axum::{routing::{get, post}, Router};

#[must_use]
pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(handlers::health))
        .route("/api/v1/graph/decompose", post(handlers::decompose))
        .route("/api/v1/graph/delete-edge", post(handlers::delete_edge))
        .route("/api/v1/witness/check", post(handlers::check_witness))
        .with_state(state)
}

