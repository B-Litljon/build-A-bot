//! Read-only observability API for the V5 OANDA soak.
//!
//! Reads what the bot writes (`logs/events-*.jsonl`, `logs/status.json`) and
//! serves it over HTTP. It never writes to the repo, never talks to a broker
//! on a mutating endpoint, and registers no route that can change the bot's
//! behaviour — the strongest safety property available here is that the
//! capability is simply absent.
//!
//! Scaffold state: `/api/health`, `/api/status` and `/api/events` are live.
//! `oanda.rs` (broker truth: balance, fills, realized P&L), `derive.rs`
//! (computed views) and the remaining routes are Brandon's to write; see
//! `dashboard/README.md` for the contract each one owes the front-end.

mod config;
mod events;
mod status;

use std::sync::Arc;

use axum::{
    extract::{Query, State},
    http::StatusCode,
    routing::get,
    Json, Router,
};
use serde::Deserialize;
use tokio::sync::RwLock;
use tower_http::services::ServeDir;

use config::Config;
use events::EventRing;

#[derive(Clone)]
struct AppState {
    cfg: Arc<Config>,
    ring: Arc<RwLock<EventRing>>,
    started: std::time::Instant,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "soak_dashboard=info,tower_http=warn".into()),
        )
        .init();

    let cfg = Arc::new(Config::from_env()?);
    tracing::info!(?cfg, "starting soak dashboard");

    let ring = Arc::new(RwLock::new(EventRing::new(cfg.ring_capacity)));
    {
        // Backfill before serving, so the first request is not empty.
        let mut guard = ring.write().await;
        let n = guard.ingest_dir(&cfg.logs_dir, 7).unwrap_or(0);
        tracing::info!(events = n, "backfilled event ring");
    }

    // Tailer: the only background task in the scaffold.
    {
        let (cfg, ring) = (cfg.clone(), ring.clone());
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(cfg.tail_interval);
            loop {
                ticker.tick().await;
                let mut guard = ring.write().await;
                if let Err(e) = guard.ingest_dir(&cfg.logs_dir, 2) {
                    tracing::warn!(error = %e, "tail failed");
                }
            }
        });
    }

    let state = AppState { cfg: cfg.clone(), ring, started: std::time::Instant::now() };

    let api = Router::new()
        .route("/health", get(health))
        .route("/status", get(get_status))
        .route("/events", get(get_events));

    let mut app = Router::new().nest("/api", api).with_state(state);
    if cfg.web_dir.is_dir() {
        app = app.fallback_service(ServeDir::new(&cfg.web_dir));
    }
    let app = app.layer(tower_http::compression::CompressionLayer::new());

    let listener = tokio::net::TcpListener::bind(cfg.bind).await?;
    tracing::info!("listening on http://{}", cfg.bind);
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}

/// Is the dashboard up, and does it believe the BOT is up?
///
/// The second half is the point: an API that answers 200 while the bot is dead
/// is worse than no dashboard, because it reads as reassurance.
async fn health(State(st): State<AppState>) -> Json<serde_json::Value> {
    let bar_seconds = 15.0 * 60.0;
    let report = status::read(&st.cfg.status_path(), bar_seconds);
    let ring = st.ring.read().await;
    let last_event = ring.recent(1, None).into_iter().next();

    Json(serde_json::json!({
        "api": {
            "ok": true,
            "uptime_seconds": st.started.elapsed().as_secs(),
            "events_buffered": ring.len(),
            "lines_skipped": ring.skipped_lines,
        },
        "bot": report,
        "last_event": last_event,
    }))
}

async fn get_status(State(st): State<AppState>) -> Json<status::StatusReport> {
    Json(status::read(&st.cfg.status_path(), 15.0 * 60.0))
}

#[derive(Debug, Deserialize)]
struct EventQuery {
    /// Filter by event kind (`bar`, `entry`, `exit`, `gate_veto`, ...).
    ev: Option<String>,
    limit: Option<usize>,
}

async fn get_events(
    State(st): State<AppState>,
    Query(q): Query<EventQuery>,
) -> Result<Json<Vec<events::Event>>, StatusCode> {
    let limit = q.limit.unwrap_or(200).min(5_000);
    let ring = st.ring.read().await;
    Ok(Json(ring.recent(limit, q.ev.as_deref())))
}
