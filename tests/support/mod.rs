#![allow(dead_code)]

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;

static CASE_SEQUENCE: AtomicU64 = AtomicU64::new(1);

#[must_use]
pub fn log_dir() -> PathBuf {
    let path = std::env::var("MAXFLOW_TEST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("test-logs/rust"));
    fs::create_dir_all(&path).expect("create test log directory");
    path
}

#[must_use]
pub fn run_id(test_name: &str, case_id: &str) -> String {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time")
        .as_millis();
    let sequence = CASE_SEQUENCE.fetch_add(1, Ordering::SeqCst);
    format!("rust-{timestamp:016}-{sequence:04}-{test_name}-{case_id}")
}

pub fn append_log<T>(run_id: &str, kind: &str, judgment: &str, payload: &T)
where
    T: Serialize,
{
    let file = log_dir().join("cases.jsonl");
    let mut handle = OpenOptions::new()
        .create(true)
        .append(true)
        .open(file)
        .expect("open test log");
    let line = serde_json::json!({
        "run_id": run_id,
        "kind": kind,
        "judgment": judgment,
        "payload": payload,
    });
    writeln!(handle, "{line}").expect("write test log");
}
