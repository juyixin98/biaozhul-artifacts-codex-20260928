use dag_path_cover::{api::router, config::Config, APP_NAME, APP_VERSION};
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = Config::from_env()?;

    let env_filter = EnvFilter::try_new(&config.log_level)
        .or_else(|_| EnvFilter::try_new("info"))
        .expect("a valid fallback EnvFilter");
    tracing_subscriber::fmt()
        .with_env_filter(env_filter)
        .json()
        .init();

    let listener = tokio::net::TcpListener::bind(config.bind_address).await?;
    tracing::info!(
        service = APP_NAME,
        version = APP_VERSION,
        bind = %config.bind_address,
        "starting HTTP API"
    );
    axum::serve(listener, router(config.clone()))
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("Ctrl-C signal handler can be installed");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("SIGTERM signal handler can be installed")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}
