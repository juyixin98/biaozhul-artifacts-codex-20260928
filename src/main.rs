use std::env;

use directed_maxflow_backend::api;
use directed_maxflow_backend::api::state::AppState;
use tokio::net::TcpListener;

#[tokio::main]
async fn main() {
    let host = env::var("MAXFLOW_HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
    let port = env::var("MAXFLOW_PORT")
        .ok()
        .and_then(|value| value.parse::<u16>().ok())
        .unwrap_or(8080);
    let max_jobs = env::var("MAXFLOW_MAX_JOBS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(64);

    let state = AppState::new(max_jobs);
    let app = api::router(state);
    let listener = TcpListener::bind((host.as_str(), port))
        .await
        .expect("bind HTTP listener");

    println!("maximum-flow backend listening on http://{host}:{port}");
    axum::serve(listener, app).await.expect("serve HTTP API");
}
