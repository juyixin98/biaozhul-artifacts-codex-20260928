use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AppConfig {
    pub host: String,
    pub port: u16,
    pub max_edges: usize,
    pub max_operations_per_batch: usize,
    pub max_edge_id_length: usize,
    pub max_graph_id_length: usize,
    pub max_history_versions: u64,
    pub verify_after_batch: bool,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 8080,
            max_edges: 10_000,
            max_operations_per_batch: 500,
            max_edge_id_length: 128,
            max_graph_id_length: 128,
            max_history_versions: 100,
            verify_after_batch: true,
        }
    }
}

impl AppConfig {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|source| ConfigError::Read {
            path: path.to_path_buf(),
            source: source.to_string(),
        })?;
        let config: Self =
            toml::from_str(&text).map_err(|source| ConfigError::Parse(source.to_string()))?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.host.is_empty() {
            return Err(ConfigError::Invalid("host must not be empty"));
        }
        if self.max_edges == 0
            || self.max_operations_per_batch == 0
            || self.max_edge_id_length == 0
            || self.max_graph_id_length == 0
            || self.max_history_versions == 0
        {
            return Err(ConfigError::Invalid("all limits must be greater than zero"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigError {
    Read {
        path: std::path::PathBuf,
        source: String,
    },
    Parse(String),
    Invalid(&'static str),
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Read { path, source } => {
                write!(f, "failed to read config {}: {source}", path.display())
            }
            Self::Parse(source) => write!(f, "invalid config TOML: {source}"),
            Self::Invalid(message) => write!(f, "invalid config: {message}"),
        }
    }
}

impl std::error::Error for ConfigError {}
