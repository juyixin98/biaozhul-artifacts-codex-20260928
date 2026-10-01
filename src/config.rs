use std::env;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AppConfig {
    pub bind_address: String,
    pub include_events_by_default: bool,
    pub redact_labels: bool,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            bind_address: "127.0.0.1:8080".to_string(),
            include_events_by_default: false,
            redact_labels: true,
        }
    }
}

impl AppConfig {
    #[must_use]
    pub fn from_env() -> Self {
        let mut config = Self::default();
        if let Ok(address) = env::var("KTRUSS_BIND_ADDRESS") {
            if !address.trim().is_empty() {
                config.bind_address = address;
            }
        }
        if let Ok(value) = env::var("KTRUSS_INCLUDE_EVENTS") {
            config.include_events_by_default = matches!(value.as_str(), "1" | "true" | "yes");
        }
        if let Ok(value) = env::var("KTRUSS_REDACT_LABELS") {
            config.redact_labels = !matches!(value.as_str(), "0" | "false" | "no");
        }
        config
    }
}

