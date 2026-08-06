//! Reader for `logs/status.json` — the bot's "right now" snapshot.
//!
//! The file is replaced atomically (temp + rename) by the Python side, so a
//! read either sees the previous snapshot or the next one, never a partial
//! object. We still treat a parse failure as "stale" rather than an error:
//! the dashboard's job during a wobble is to say *I don't know*, not to 500.
//!
//! Freshness is the load-bearing part. A snapshot that stopped updating an
//! hour ago looks identical to a healthy one if you only read its fields —
//! which is exactly how a dead bot gets reported as fine. Every response
//! carries `age_seconds` and a `fresh` verdict derived from the bar interval.

use std::path::Path;
use std::time::SystemTime;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Snapshot {
    /// Parsed contents of status.json, verbatim.
    pub data: serde_json::Value,
    /// Seconds since the file was last modified.
    pub age_seconds: f64,
    /// False once the snapshot is older than roughly two bar periods.
    pub fresh: bool,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum StatusReport {
    Ok(Snapshot),
    /// The bot has never written a snapshot, or the file was removed.
    Missing { path: String },
    /// Present but unreadable — reported honestly rather than as healthy.
    Unreadable { path: String, error: String },
}

/// Read and classify the snapshot. `bar_seconds` sets the freshness bar.
pub fn read(path: &Path, bar_seconds: f64) -> StatusReport {
    let meta = match std::fs::metadata(path) {
        Ok(m) => m,
        Err(_) => return StatusReport::Missing { path: path.display().to_string() },
    };
    let age_seconds = meta
        .modified()
        .ok()
        .and_then(|m| SystemTime::now().duration_since(m).ok())
        .map(|d| d.as_secs_f64())
        .unwrap_or(f64::INFINITY);

    match std::fs::read_to_string(path).map_err(|e| e.to_string()).and_then(|s| {
        serde_json::from_str::<serde_json::Value>(&s).map_err(|e| e.to_string())
    }) {
        Ok(data) => StatusReport::Ok(Snapshot {
            data,
            age_seconds,
            // Two bar periods of slack: one missed write is a hiccup, two is
            // a symptom.
            fresh: age_seconds <= bar_seconds * 2.0,
        }),
        Err(error) => StatusReport::Unreadable { path: path.display().to_string(), error },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn reads_a_good_snapshot() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("status.json");
        let mut f = std::fs::File::create(&p).unwrap();
        f.write_all(br#"{"pid":42,"positions":{}}"#).unwrap();

        match read(&p, 900.0) {
            StatusReport::Ok(s) => {
                assert_eq!(s.data["pid"], 42);
                assert!(s.fresh);
            }
            other => panic!("expected Ok, got {other:?}"),
        }
    }

    #[test]
    fn missing_file_is_reported_not_faked() {
        assert!(matches!(
            read(Path::new("/nope/status.json"), 900.0),
            StatusReport::Missing { .. }
        ));
    }

    #[test]
    fn garbage_is_unreadable_not_healthy() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("status.json");
        std::fs::write(&p, b"{ half written").unwrap();
        assert!(matches!(read(&p, 900.0), StatusReport::Unreadable { .. }));
    }

    #[test]
    fn stale_snapshot_is_not_fresh() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("status.json");
        std::fs::write(&p, b"{}").unwrap();
        // Zero-length freshness window: anything already written is stale.
        match read(&p, 0.0) {
            StatusReport::Ok(s) => assert!(!s.fresh),
            other => panic!("expected Ok, got {other:?}"),
        }
    }
}
