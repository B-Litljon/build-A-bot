//! Runtime configuration, all from the environment.
//!
//! Deliberately has no `oanda_token` field yet — the token arrives with
//! `oanda.rs`. Everything here is a path or a bind address, so a
//! misconfiguration is a 404, never a credential leak.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct Config {
    /// Repo root; every other path is resolved against it.
    pub repo_dir: PathBuf,
    /// Where the bot writes events-*.jsonl and status.json.
    pub logs_dir: PathBuf,
    /// Built front-end assets, served at `/` when the directory exists.
    pub web_dir: PathBuf,
    /// Bind address. Defaults to loopback ON PURPOSE: remote access is meant
    /// to go through `tailscale serve`, which gives tailnet TLS without this
    /// process ever listening on a public interface.
    pub bind: SocketAddr,
    /// How often the JSONL tailer looks for new lines.
    pub tail_interval: Duration,
    /// Cap on events held in memory, oldest evicted first.
    pub ring_capacity: usize,
}

impl Config {
    pub fn from_env() -> anyhow::Result<Self> {
        let repo_dir = PathBuf::from(
            std::env::var("SOAK_REPO_DIR")
                .unwrap_or_else(|_| "/mnt/storage/mystuf/development/build-A-bot".into()),
        );
        let logs_dir = std::env::var("EVENTS_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| repo_dir.join("logs"));
        let web_dir = std::env::var("DASHBOARD_WEB_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| repo_dir.join("dashboard/web/dist"));
        let bind: SocketAddr = std::env::var("DASHBOARD_BIND")
            .unwrap_or_else(|_| "127.0.0.1:8787".into())
            .parse()?;
        let tail_interval = Duration::from_millis(
            std::env::var("DASHBOARD_TAIL_MS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1000),
        );
        let ring_capacity = std::env::var("DASHBOARD_RING")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(20_000);

        Ok(Self { repo_dir, logs_dir, web_dir, bind, tail_interval, ring_capacity })
    }

    pub fn status_path(&self) -> PathBuf {
        self.logs_dir.join("status.json")
    }
}
