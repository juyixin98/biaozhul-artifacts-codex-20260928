use serde_json::Value;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static SEQUENCE: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone)]
pub struct RunContext {
    run_id: String,
    log_path: Option<PathBuf>,
}

impl RunContext {
    pub fn new() -> Self {
        let millis = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);
        let seq = SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let pid = std::process::id();
        Self {
            run_id: format!("run-{millis}-pid{pid}-{seq:04}"),
            log_path: default_log_path(),
        }
    }

    pub fn for_tests() -> Self {
        let mut ctx = Self::new();
        ctx.log_path = Some(PathBuf::from("logs/test-runs.jsonl"));
        ctx
    }

    pub fn id(&self) -> &str {
        &self.run_id
    }

    pub fn event(&self, stage: &str, status: &str, message: &str, data: Value) {
        let Some(path) = &self.log_path else {
            return;
        };
        let line = serde_json::json!({
            "run_id": self.run_id,
            "timestamp_ms": SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_millis())
                .unwrap_or(0),
            "stage": stage,
            "status": status,
            "message": message,
            "data": data,
        });
        if let Err(error) = append_json_line(path, &line) {
            eprintln!(
                "warning: unable to append run log for {} to {}: {error}",
                self.run_id,
                path.display()
            );
        }
    }
}

impl Default for RunContext {
    fn default() -> Self {
        Self::new()
    }
}

fn default_log_path() -> Option<PathBuf> {
    if std::env::var("CHORDAL_DISABLE_LOG").as_deref() == Ok("1") {
        return None;
    }
    if let Ok(path) = std::env::var("CHORDAL_LOG_FILE") {
        return Some(PathBuf::from(path));
    }
    Some(PathBuf::from("logs/run-log.jsonl"))
}

fn append_json_line(path: &Path, value: &Value) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    let mut bytes = serde_json::to_vec(value).map_err(std::io::Error::other)?;
    bytes.push(b'\n');
    file.write_all(&bytes)?;
    Ok(())
}
