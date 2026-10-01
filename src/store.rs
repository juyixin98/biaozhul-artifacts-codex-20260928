use crate::config::AppConfig;
use crate::kernel::DynamicMsf;
use crate::model::{Edge, ModelError};
use crate::witness::{self, WitnessReport};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Mutex, MutexGuard};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Snapshot {
    pub graph_id: String,
    pub version: u64,
    pub vertices: BTreeSet<i64>,
    pub edges: BTreeMap<String, Edge>,
    pub forest_edge_ids: BTreeSet<String>,
    #[serde(with = "i128_string")]
    pub total_weight: i128,
}

pub mod i128_string {
    use serde::{Deserialize, Deserializer, Serializer};
    use std::str::FromStr;

    pub fn serialize<S: Serializer>(value: &i128, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&value.to_string())
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<i128, D::Error> {
        let text = String::deserialize(deserializer)?;
        i128::from_str(&text).map_err(serde::de::Error::custom)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StoreError {
    GraphAlreadyExists(String),
    GraphNotFound(String),
    VersionNotFound {
        graph_id: String,
        version: u64,
    },
    ExpectedVersionMismatch {
        graph_id: String,
        actual: u64,
        expected: u64,
    },
    WitnessRejected(WitnessReport),
    Model(ModelError),
}

impl From<ModelError> for StoreError {
    fn from(value: ModelError) -> Self {
        Self::Model(value)
    }
}

#[derive(Default)]
struct GraphRecord {
    versions: BTreeMap<u64, Snapshot>,
}

#[derive(Default)]
pub struct GraphStore {
    records: Mutex<BTreeMap<String, GraphRecord>>,
}

impl GraphStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn create_graph(&self, graph_id: &str, config: &AppConfig) -> Result<Snapshot, StoreError> {
        let mut records = self.lock();
        if records.contains_key(graph_id) {
            return Err(StoreError::GraphAlreadyExists(graph_id.to_string()));
        }
        let snapshot =
            DynamicMsf::new(config.max_edges, config.max_edge_id_length).snapshot(graph_id, 0);
        records
            .entry(graph_id.to_string())
            .or_default()
            .versions
            .insert(0, snapshot.clone());
        Ok(snapshot)
    }

    pub fn get_version(&self, graph_id: &str, version: u64) -> Result<Snapshot, StoreError> {
        let records = self.lock();
        let record = records
            .get(graph_id)
            .ok_or_else(|| StoreError::GraphNotFound(graph_id.to_string()))?;
        record
            .versions
            .get(&version)
            .cloned()
            .ok_or_else(|| StoreError::VersionNotFound {
                graph_id: graph_id.to_string(),
                version,
            })
    }

    pub fn latest_version(&self, graph_id: &str) -> Result<Snapshot, StoreError> {
        let records = self.lock();
        let record = records
            .get(graph_id)
            .ok_or_else(|| StoreError::GraphNotFound(graph_id.to_string()))?;
        record
            .versions
            .values()
            .next_back()
            .cloned()
            .ok_or_else(|| StoreError::GraphNotFound(graph_id.to_string()))
    }

    pub fn list_versions(&self, graph_id: &str) -> Result<Vec<u64>, StoreError> {
        let records = self.lock();
        let record = records
            .get(graph_id)
            .ok_or_else(|| StoreError::GraphNotFound(graph_id.to_string()))?;
        Ok(record.versions.keys().copied().collect())
    }

    pub fn mutate(
        &self,
        graph_id: &str,
        expected_version: Option<u64>,
        config: &AppConfig,
        f: impl FnOnce(&mut DynamicMsf) -> Result<(), ModelError>,
    ) -> Result<Snapshot, StoreError> {
        let mut records = self.lock();
        let record = records
            .get_mut(graph_id)
            .ok_or_else(|| StoreError::GraphNotFound(graph_id.to_string()))?;
        let current = record
            .versions
            .values()
            .next_back()
            .expect("created graph has version zero")
            .clone();

        if let Some(expected) = expected_version {
            if current.version != expected {
                return Err(StoreError::ExpectedVersionMismatch {
                    graph_id: graph_id.to_string(),
                    actual: current.version,
                    expected,
                });
            }
        }

        let mut kernel =
            DynamicMsf::from_snapshot(&current, config.max_edges, config.max_edge_id_length);
        f(&mut kernel)?;
        let next_version = current.version + 1;
        let snapshot = kernel.snapshot(graph_id, next_version);
        if config.verify_after_batch {
            let report = witness::verify_snapshot(&snapshot);
            if report.status != "accepted" {
                return Err(StoreError::WitnessRejected(report));
            }
        }
        record.versions.insert(next_version, snapshot.clone());
        self.trim(record, config);
        Ok(snapshot)
    }

    fn trim(&self, record: &mut GraphRecord, config: &AppConfig) {
        while record.versions.len() > config.max_history_versions as usize {
            if let Some(oldest) = record.versions.keys().next().copied() {
                record.versions.remove(&oldest);
            } else {
                break;
            }
        }
    }

    fn lock(&self) -> MutexGuard<'_, BTreeMap<String, GraphRecord>> {
        self.records.lock().expect("graph store mutex poisoned")
    }
}
