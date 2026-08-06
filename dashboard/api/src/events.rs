//! JSONL tailer: `logs/events-YYYY-MM-DD.jsonl` -> an in-memory ring.
//!
//! Two properties matter more than anything else here:
//!
//! 1. **A malformed line is skipped, never fatal.** The writer appends and
//!    flushes per line, so a read can catch a half-written tail line. That is
//!    normal operation, not corruption — the next poll sees it complete.
//! 2. **Reads are incremental.** We remember the byte offset per file and read
//!    only what is new, so tailing a growing file stays O(new bytes).
//!
//! Events are kept as a typed envelope plus the raw JSON object. The envelope
//! covers what every consumer needs (`ts`, `ev`, `sym`); the rest stays
//! untyped so a new field on the Python side does not require a Rust change
//! before it can reach the browser.

use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    pub ts: String,
    pub ev: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sym: Option<String>,
    /// Everything else, verbatim.
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// Bounded, newest-last store of parsed events.
#[derive(Debug)]
pub struct EventRing {
    events: VecDeque<Event>,
    capacity: usize,
    /// Per-file read offset, so each poll reads only the new bytes.
    offsets: std::collections::HashMap<PathBuf, u64>,
    pub skipped_lines: u64,
}

impl EventRing {
    pub fn new(capacity: usize) -> Self {
        Self {
            events: VecDeque::with_capacity(capacity.min(4096)),
            capacity,
            offsets: Default::default(),
            skipped_lines: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.events.len()
    }

    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    pub fn push(&mut self, ev: Event) {
        if self.events.len() == self.capacity {
            self.events.pop_front();
        }
        self.events.push_back(ev);
    }

    /// Newest events first, at most `limit`, optionally filtered by kind.
    pub fn recent(&self, limit: usize, kind: Option<&str>) -> Vec<Event> {
        self.events
            .iter()
            .rev()
            .filter(|e| kind.is_none_or(|k| e.ev == k))
            .take(limit)
            .cloned()
            .collect()
    }

    /// Read new bytes from one file into the ring. Returns events added.
    pub fn ingest_file(&mut self, path: &Path) -> anyhow::Result<usize> {
        let file = match std::fs::File::open(path) {
            Ok(f) => f,
            // A day with no events yet is not an error.
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(0),
            Err(e) => return Err(e.into()),
        };
        let size = file.metadata()?.len();
        let offset = *self.offsets.get(path).unwrap_or(&0);
        // Truncated or rotated underneath us: start over rather than seek past
        // the end and read garbage.
        let start = if size < offset { 0 } else { offset };

        let mut reader = BufReader::new(file);
        reader.seek(SeekFrom::Start(start))?;

        let mut consumed = start;
        let mut added = 0usize;
        let mut line = String::new();
        loop {
            line.clear();
            let n = reader.read_line(&mut line)?;
            if n == 0 {
                break;
            }
            // A line without a trailing newline is still being written; leave
            // the offset before it so the next poll re-reads it whole.
            if !line.ends_with('\n') {
                break;
            }
            consumed += n as u64;
            match serde_json::from_str::<Event>(line.trim_end()) {
                Ok(ev) => {
                    self.push(ev);
                    added += 1;
                }
                Err(_) => self.skipped_lines += 1,
            }
        }
        self.offsets.insert(path.to_path_buf(), consumed);
        Ok(added)
    }

    /// Ingest the newest `days` event files (today first is not required —
    /// files are read oldest to newest so ring order stays chronological).
    pub fn ingest_dir(&mut self, dir: &Path, days: usize) -> anyhow::Result<usize> {
        let mut files: Vec<PathBuf> = match std::fs::read_dir(dir) {
            Ok(rd) => rd
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| {
                    p.file_name()
                        .and_then(|n| n.to_str())
                        .is_some_and(|n| n.starts_with("events-") && n.ends_with(".jsonl"))
                })
                .collect(),
            Err(_) => return Ok(0),
        };
        // Filenames are ISO dates, so lexical order is chronological order.
        files.sort();
        if files.len() > days {
            files.drain(..files.len() - days);
        }
        let mut added = 0;
        for f in files {
            added += self.ingest_file(&f)?;
        }
        Ok(added)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write(dir: &Path, name: &str, body: &str) -> PathBuf {
        let p = dir.join(name);
        let mut f = std::fs::File::create(&p).unwrap();
        f.write_all(body.as_bytes()).unwrap();
        p
    }

    #[test]
    fn parses_events_and_keeps_extra_fields() {
        let dir = tempfile::tempdir().unwrap();
        let p = write(
            dir.path(),
            "events-2026-07-30.jsonl",
            r#"{"ts":"2026-07-30T13:45:01+00:00","ev":"entry","sym":"NZD_JPY","units":1000,"entry":94.182}
"#,
        );
        let mut ring = EventRing::new(100);
        assert_eq!(ring.ingest_file(&p).unwrap(), 1);
        let ev = &ring.recent(1, None)[0];
        assert_eq!(ev.ev, "entry");
        assert_eq!(ev.sym.as_deref(), Some("NZD_JPY"));
        assert_eq!(ev.rest["units"], 1000);
    }

    #[test]
    fn malformed_line_is_skipped_not_fatal() {
        let dir = tempfile::tempdir().unwrap();
        let p = write(
            dir.path(),
            "events-2026-07-30.jsonl",
            "{not json at all}\n{\"ts\":\"t\",\"ev\":\"bar\"}\n",
        );
        let mut ring = EventRing::new(100);
        assert_eq!(ring.ingest_file(&p).unwrap(), 1);
        assert_eq!(ring.skipped_lines, 1);
    }

    #[test]
    fn partial_last_line_is_reread_when_complete() {
        // The writer flushes per line; a reader can still land mid-append.
        let dir = tempfile::tempdir().unwrap();
        let p = write(dir.path(), "events-2026-07-30.jsonl", "{\"ts\":\"t\",\"ev\":\"ba");
        let mut ring = EventRing::new(100);
        assert_eq!(ring.ingest_file(&p).unwrap(), 0);

        write(dir.path(), "events-2026-07-30.jsonl", "{\"ts\":\"t\",\"ev\":\"bar\"}\n");
        assert_eq!(ring.ingest_file(&p).unwrap(), 1);
        assert_eq!(ring.skipped_lines, 0);
    }

    #[test]
    fn incremental_reads_do_not_duplicate() {
        let dir = tempfile::tempdir().unwrap();
        let p = write(dir.path(), "events-2026-07-30.jsonl", "{\"ts\":\"t\",\"ev\":\"a\"}\n");
        let mut ring = EventRing::new(100);
        ring.ingest_file(&p).unwrap();

        let mut f = std::fs::OpenOptions::new().append(true).open(&p).unwrap();
        f.write_all(b"{\"ts\":\"t\",\"ev\":\"b\"}\n").unwrap();

        assert_eq!(ring.ingest_file(&p).unwrap(), 1);
        assert_eq!(ring.len(), 2);
    }

    #[test]
    fn ring_evicts_oldest_at_capacity() {
        let mut ring = EventRing::new(2);
        for ev in ["a", "b", "c"] {
            ring.push(Event {
                ts: "t".into(),
                ev: ev.into(),
                sym: None,
                rest: Default::default(),
            });
        }
        assert_eq!(ring.len(), 2);
        assert_eq!(ring.recent(9, None)[0].ev, "c");
    }

    #[test]
    fn filters_by_kind() {
        let mut ring = EventRing::new(10);
        for ev in ["bar", "entry", "bar"] {
            ring.push(Event {
                ts: "t".into(),
                ev: ev.into(),
                sym: None,
                rest: Default::default(),
            });
        }
        assert_eq!(ring.recent(10, Some("bar")).len(), 2);
    }

    #[test]
    fn missing_file_is_not_an_error() {
        let mut ring = EventRing::new(10);
        assert_eq!(ring.ingest_file(Path::new("/nope/events-2026-01-01.jsonl")).unwrap(), 0);
    }
}
