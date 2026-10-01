use axum::extract::DefaultBodyLimit;
use std::net::SocketAddr;

use maxflow_server::api;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let host = std::env::var("MAXFLOW_HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
    let port = std::env::var("MAXFLOW_PORT")
        .ok()
        .map(|value| value.parse::<u16>())
        .transpose()?
        .unwrap_or(8080);
    let address: SocketAddr = format!("{host}:{port}").parse()?;
    let app = api::router()
        .fallback(api::not_found)
        .layer(DefaultBodyLimit::max(1024 * 1024));

    let listener = tokio::net::TcpListener::bind(address).await?;
    println!("maxflow-server listening on http://{address}");
    axum::serve(listener, app).await?;
    Ok(())
}
