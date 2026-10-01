//! File and environment based application configuration.

use serde::Deserialize;
use std::net::{AddrParseError, SocketAddr};

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub host: String,
    pub port: u16,
    pub max_vertices: usize,
    pub log_level: String,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 8080,
            max_vertices: 2000,
            log_level: "info".to_string(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigError {
    ReadToml(String),
    InvalidHost(String),
    InvalidPort(u16),
    InvalidMaxVertices(usize),
    InvalidLogLevel(String),
}

impl ConfigError {
    pub fn code(&self) -> &'static str {
        match self {
            Self::ReadToml(_) => "config_read_toml",
            Self::InvalidHost(_) => "config_invalid_host",
            Self::InvalidPort(_) => "config_invalid_port",
            Self::InvalidMaxVertices(_) => "config_invalid_max_vertices",
            Self::InvalidLogLevel(_) => "config_invalid_log_level",
        }
    }

    pub fn message(&self) -> String {
        match self {
            Self::ReadToml(error) => format!("failed to parse TOML configuration: {error}"),
            Self::InvalidHost(host) => format!("invalid host `{host}`"),
            Self::InvalidPort(port) => format!("invalid port `{port}`"),
            Self::InvalidMaxVertices(value) => {
                format!("max_vertices must be between 1 and 2000, got {value}")
            }
            Self::InvalidLogLevel(level) => {
                format!("log_level must be error|warn|info|debug|trace, got `{level}`")
            }
        }
    }
}

impl Config {
    pub fn load(path: Option<&str>) -> Result<Self, ConfigError> {
        let mut config = match path {
            Some(path) => {
                let contents = std::fs::read_to_string(path)
                    .map_err(|error| ConfigError::ReadToml(error.to_string()))?;
                toml::from_str(&contents)
                    .map_err(|error| ConfigError::ReadToml(error.to_string()))?
            }
            None => Config::default(),
        };

        if let Ok(host) = std::env::var("DAG_PATH_COVER_HOST") {
            config.host = host;
        }
        if let Ok(port) = std::env::var("DAG_PATH_COVER_PORT") {
            config.port = port
                .parse()
                .map_err(|_: std::num::ParseIntError| ConfigError::InvalidPort(0))?;
        }
        if let Ok(max_vertices) = std::env::var("DAG_PATH_COVER_MAX_VERTICES") {
            config.max_vertices = max_vertices
                .parse()
                .map_err(|_: std::num::ParseIntError| ConfigError::InvalidMaxVertices(0))?;
        }
        if let Ok(log_level) = std::env::var("DAG_PATH_COVER_LOG_LEVEL") {
            config.log_level = log_level;
        }

        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.host.parse::<SocketAddr>().is_ok() {
            // A full socket address is allowed only when the port is the one embedded in it.
            return Err(ConfigError::InvalidHost(self.host.clone()));
        }
        format!("{}:{}", self.host, self.port)
            .parse::<SocketAddr>()
            .map_err(|_: AddrParseError| ConfigError::InvalidHost(self.host.clone()))?;
        if self.port == 0 {
            return Err(ConfigError::InvalidPort(self.port));
        }
        if self.max_vertices == 0 || self.max_vertices > 2000 {
            return Err(ConfigError::InvalidMaxVertices(self.max_vertices));
        }
        if !matches!(
            self.log_level.as_str(),
            "error" | "warn" | "info" | "debug" | "trace"
        ) {
            return Err(ConfigError::InvalidLogLevel(self.log_level.clone()));
        }
        Ok(())
    }

    pub fn socket_addr(&self) -> SocketAddr {
        format!("{}:{}", self.host, self.port)
            .parse()
            .expect("validated configuration")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_valid() {
        let config = Config::default();
        assert_eq!(config.socket_addr().to_string(), "127.0.0.1:8080");
        assert_eq!(config.max_vertices, 2000);
    }

    #[test]
    fn rejects_zero_port() {
        let config = Config {
            port: 0,
            ..Config::default()
        };

        assert_eq!(
            config.validate().expect_err("port must be nonzero"),
            ConfigError::InvalidPort(0)
        );
    }

    #[test]
    fn rejects_oversized_algorithm_limit() {
        let config = Config {
            max_vertices: 2001,
            ..Config::default()
        };

        assert_eq!(
            config.validate().expect_err("limit"),
            ConfigError::InvalidMaxVertices(2001)
        );
    }
}
