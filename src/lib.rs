pub mod algorithm;
pub mod api;
pub mod config;
pub mod dto;
pub mod fixtures;
pub mod model;
pub mod service;
pub mod topology;
pub mod witness;

pub use service::{analyze_named_graph, AnalysisFailure, AnalysisSuccess, RunContext};

pub const APP_NAME: &str = env!("CARGO_PKG_NAME");
pub const APP_VERSION: &str = env!("CARGO_PKG_VERSION");
