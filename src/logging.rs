use std::fs::{create_dir_all, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::sync::Mutex;

use serde_json::json;

#[derive(Debug)]
pub struct RunLogger {
    path: std::path::PathBuf,
    writer: Mutex<()>,
}

impl Clone for RunLogger {
    fn clone(&self) -> Self {
        Self {
            path: self.path.clone(),
            writer: Mutex::new(()),
        }
    }
}

impl RunLogger {
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            writer: Mutex::new(()),
        }
    }

    pub fn append(&self, event: serde_json::Value) -> Result<(), String> {
        if let Some(parent) = self.path.parent() {
            if !parent.as_os_str().is_empty() {
                create_dir_all(parent).map_err(|error| error.to_string())?;
            }
        }
        let _guard = self.writer.lock().map_err(|error| error.to_string())?;
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .map_err(|error| error.to_string())?;
        let line = serde_json::to_string(&event).map_err(|error| error.to_string())?;
        writeln!(file, "{line}").map_err(|error| error.to_string())
    }
}

pub fn new_run_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    let thread = format!("{:?}", std::thread::current().id());
    let thread_suffix: String = thread
        .chars()
        .filter(|character| character.is_ascii_digit())
        .collect();
    format!("run-{nanos}-{thread_suffix}")
}

pub fn error_event(
    run_id: &str,
    endpoint: &str,
    kind: &str,
    code: &str,
    message: &str,
) -> serde_json::Value {
    json!({
        "timestamp": current_timestamp(),
        "run_id": run_id,
        "endpoint": endpoint,
        "status": "error",
        "error_kind": kind,
        "error_code": code,
        "message": message,
    })
}

pub fn current_timestamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0);
    format!("unix:{seconds}")
}
