//! Process-wide logging initialization.

use std::sync::Once;

static INIT: Once = Once::new();

pub fn init(level: &str) {
    INIT.call_once(|| {
        let filter = tracing_subscriber::EnvFilter::try_new(format!(
            "dag_path_cover={level},tower_http=off"
        ))
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));

        tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_target(true)
            .with_level(true)
            .init();
    });
}
