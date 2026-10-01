//! Process configuration loaded from explicit local environment variables.

use serde::Serialize;
use std::env;

#[derive(Debug, Clone, Serialize)]
pub struct Settings {
    pub host: String,
    pub port: u16,
    pub max_nodes: usize,
    pub max_edges: usize,
    pub redact_diagnostics: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 8080,
            max_nodes: 5_000,
            max_edges: 20_000,
            redact_diagnostics: false,
        }
    }
}

impl Settings {
    pub fn from_env() -> Result<Self, String> {
        let defaults = Self::default();
        Ok(Self {
            host: env::var("TRUSS_HOST").unwrap_or(defaults.host),
            port: parse_env("TRUSS_PORT", defaults.port)?,
            max_nodes: parse_env("TRUSS_MAX_NODES", defaults.max_nodes)?,
            max_edges: parse_env("TRUSS_MAX_EDGES", defaults.max_edges)?,
            redact_diagnostics: parse_env("TRUSS_REDACT_DIAGNOSTICS", defaults.redact_diagnostics)?,
        })
    }

    pub fn bind_address(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }
}

fn parse_env<T>(name: &str, default_value: T) -> Result<T, String>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    match env::var(name) {
        Ok(raw) => raw
            .parse::<T>()
            .map_err(|error| format!("invalid {name}: {error}")),
        Err(_) => Ok(default_value),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reports_invalid_numeric_configuration() {
        unsafe { env::set_var("TRUSS_PORT", "not-a-port") };
        let result = Settings::from_env();
        unsafe { env::remove_var("TRUSS_PORT") };
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("TRUSS_PORT"));
    }
}
