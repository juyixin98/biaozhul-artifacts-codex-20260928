use serde::{Deserialize, Serialize};

use crate::dto::{FixtureSummary, GraphDto};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FixtureDocument {
    pub name: String,
    pub description: String,
    pub expected_category: String,
    pub graph: GraphDto,
}

const FIXTURE_NAMES: [&str; 5] = [
    "zero_edges",
    "diamond",
    "multi_source_sink",
    "disconnected_components",
    "cycle",
];

pub fn all_documents() -> Vec<FixtureDocument> {
    FIXTURE_NAMES
        .iter()
        .map(|name| document(name).expect("embedded fixture is valid JSON"))
        .collect()
}

pub fn document(name: &str) -> Option<FixtureDocument> {
    let path = match name {
        "zero_edges" => include_str!("../fixtures/zero_edges.json"),
        "diamond" => include_str!("../fixtures/diamond.json"),
        "multi_source_sink" => include_str!("../fixtures/multi_source_sink.json"),
        "disconnected_components" => include_str!("../fixtures/disconnected_components.json"),
        "cycle" => include_str!("../fixtures/cycle.json"),
        _ => return None,
    };
    serde_json::from_str(path).ok()
}

pub fn summaries() -> Vec<FixtureSummary> {
    all_documents()
        .into_iter()
        .map(|document| FixtureSummary {
            name: document.name,
            vertex_count: document.graph.vertices.len(),
            edge_count: document.graph.edges.len(),
            expected_category: document.expected_category,
            description: document.description,
        })
        .collect()
}
