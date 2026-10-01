//! Minimum vertex-disjoint path cover for a validated DAG.

pub mod algorithm;
pub mod config;
pub mod cycle;
pub mod logging;
pub mod matching;
pub mod model;
pub mod path_cover;
pub mod query;
pub mod witness;

pub use algorithm::{solve_dag_path_cover, SolveError, SolveResponse};
pub use config::{Config, ConfigError};
pub use model::{Edge, Graph, GraphInput};
pub use query::router;
