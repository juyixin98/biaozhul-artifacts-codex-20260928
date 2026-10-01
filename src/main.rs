use std::net::SocketAddr;

use chordal_graph_backend::api;
use tokio::net::TcpListener;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(8080);
    let address = SocketAddr::from(([127, 0, 0, 1], port));
    let listener = TcpListener::bind(address).await?;
    println!("chordal graph backend listening on http://{address}");
    axum::serve(listener, api::router()).await?;
    Ok(())
}
