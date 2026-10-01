use truss_backend::{api, config};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "truss_backend=info,tower_http=warn".into()),
        )
        .json()
        .init();

    let settings = config::Settings::from_env()?;
    let address = settings.bind_address();
    let listener = tokio::net::TcpListener::bind(&address).await?;
    tracing::info!(address = %address, "starting truss backend");
    axum::serve(listener, api::router(settings)).await?;
    Ok(())
}
