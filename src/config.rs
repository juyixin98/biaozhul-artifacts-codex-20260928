//! Process configuration loaded from environment variables.

use std::env;

use serde::Deserialize;

/// Runtime service configuration.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct Config {
    /// Socket address used by the HTTP server.
    pub bind_address: String,
    /// Minimum severity for tracing output, for example `info`.
    pub log_level: String,
}

impl Config {
    /// Load configuration from `MSF_BIND_ADDRESS` and `MSF_LOG_LEVEL`.
    pub fn from_env() -> Self {
        Self {
            bind_address: env::var("MSF_BIND_ADDRESS")
                .unwrap_or_else(|_| "127.0.0.1:8080".to_string()),
            log_level: env::var("MSF_LOG_LEVEL").unwrap_or_else(|_| "info".to_string()),
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            bind_address: "127.0.0.1:8080".to_string(),
            log_level: "info".to_string(),
        }
    }
}
