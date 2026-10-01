use std::net::SocketAddr;

use msf_service::{api, config::Config, service::UpdateService};
use tracing_subscriber::{fmt, EnvFilter};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = Config::from_env();
    init_tracing(&config.log_level);

    let address: SocketAddr = config.bind_address.parse()?;
    let service = UpdateService::new();
    let listener = tokio::net::TcpListener::bind(address).await?;

    tracing::info!(
        bind_address = %address,
        log_level = %config.log_level,
        "dynamic MSF service listening"
    );

    axum::serve(listener, api::app(service)).await?;
    Ok(())
}

fn init_tracing(level: &str) {
    let filter = EnvFilter::try_new(level).unwrap_or_else(|_| EnvFilter::new("info"));
    fmt()
        .with_env_filter(filter)
        .with_target(true)
        .json()
        .init();
}
