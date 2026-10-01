use std::env;
use std::error::Error;
use std::fmt;
use std::net::{IpAddr, SocketAddr};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Config {
    pub bind_address: SocketAddr,
    pub max_vertices: usize,
    pub max_edges: usize,
    pub log_level: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigError {
    pub variable: String,
    pub value: String,
    pub reason: String,
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "invalid {}={}: {}",
            self.variable, self.value, self.reason
        )
    }
}

impl Error for ConfigError {}

impl Default for Config {
    fn default() -> Self {
        Self {
            bind_address: SocketAddr::new(IpAddr::from([127, 0, 0, 1]), 8080),
            max_vertices: 1000,
            max_edges: 5000,
            log_level: "info".to_string(),
        }
    }
}

impl Config {
    pub fn from_env() -> Result<Self, ConfigError> {
        Self::from_iter(env::vars())
    }

    pub fn from_iter(
        iter: impl IntoIterator<Item = (String, String)>,
    ) -> Result<Self, ConfigError> {
        let mut config = Self::default();
        for (key, value) in iter {
            match key.as_str() {
                "DAG_BIND_ADDRESS" => {
                    config.bind_address = value.parse().map_err(|error| ConfigError {
                        variable: key,
                        value,
                        reason: format!("invalid socket address: {error}"),
                    })?;
                }
                "DAG_HOST" => {
                    let host: IpAddr = value.parse().map_err(|error| ConfigError {
                        variable: key,
                        value: value.clone(),
                        reason: format!("invalid IP address: {error}"),
                    })?;
                    config.bind_address.set_ip(host);
                }
                "DAG_PORT" => {
                    let port: u16 = value.parse().map_err(|error| ConfigError {
                        variable: key,
                        value: value.clone(),
                        reason: format!("invalid TCP port: {error}"),
                    })?;
                    config.bind_address.set_port(port);
                }
                "DAG_MAX_VERTICES" => {
                    config.max_vertices = parse_usize(&key, &value)?;
                }
                "DAG_MAX_EDGES" => {
                    config.max_edges = parse_usize(&key, &value)?;
                }
                "DAG_LOG_LEVEL" => config.log_level = value,
                _ => {}
            }
        }
        Ok(config)
    }
}

fn parse_usize(variable: &str, value: &str) -> Result<usize, ConfigError> {
    value.parse::<usize>().map_err(|error| ConfigError {
        variable: variable.to_string(),
        value: value.to_string(),
        reason: format!("expected a non-negative integer: {error}"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_local_and_limited() {
        let config = Config::from_iter([("UNUSED".to_string(), "x".to_string())]).unwrap();
        assert_eq!(config.bind_address.to_string(), "127.0.0.1:8080");
        assert_eq!(config.max_vertices, 1000);
        assert_eq!(config.max_edges, 5000);
    }

    #[test]
    fn overrides_host_port_and_limits() {
        let config = Config::from_iter([
            ("DAG_HOST".to_string(), "0.0.0.0".to_string()),
            ("DAG_PORT".to_string(), "9099".to_string()),
            ("DAG_MAX_VERTICES".to_string(), "7".to_string()),
            ("DAG_MAX_EDGES".to_string(), "11".to_string()),
        ])
        .unwrap();
        assert_eq!(config.bind_address.to_string(), "0.0.0.0:9099");
        assert_eq!(config.max_vertices, 7);
        assert_eq!(config.max_edges, 11);
    }

    #[test]
    fn invalid_port_is_named() {
        let error = Config::from_iter([("DAG_PORT".to_string(), "nope".to_string())]).unwrap_err();
        assert_eq!(error.variable, "DAG_PORT");
        assert_eq!(error.value, "nope");
        assert!(error.reason.contains("TCP port"));
    }
}
