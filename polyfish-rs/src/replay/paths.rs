//! The one authority for canonical replay file names.
//!
//! The server's save endpoints, the startup loader, `import_replays` and
//! `upload_replays` all derive names here, so a producer can never write a name
//! its consumers filter out.

use std::path::{Path, PathBuf};

pub const REPLAY_DIR: &str = "replays";
pub const CANONICAL_REPLAY_SUFFIX: &str = ".replay.json";
/// Where a payload the save endpoints refused is parked, so a rejection costs
/// a capture session nothing.
pub const REJECTED_REPLAY_DIR: &str = "replays/rejected";
pub const REJECTED_PAYLOAD_SUFFIX: &str = ".rejected.json";

/// Lowercased `[a-z0-9_]`; every other run of characters, `-` included,
/// collapses to a single `-`. Never empty, so the stem always names something.
pub fn sanitize_storage_key(name: &str) -> String {
    let mut result = String::new();
    let mut last_was_dash = false;
    for c in name.chars() {
        if c.is_ascii_alphanumeric() || c == '_' {
            result.push(c.to_ascii_lowercase());
            last_was_dash = false;
        } else if !last_was_dash && !result.is_empty() {
            result.push('-');
            last_was_dash = true;
        }
    }
    let trimmed = result.trim_matches('-');
    if trimmed.is_empty() {
        "replay".to_string()
    } else {
        trimmed.to_string()
    }
}

/// The canonical base name, used verbatim as the Supabase storage key.
pub fn canonical_replay_file_name(game_name: &str, timestamp: u64) -> String {
    format!(
        "{}_{}{}",
        sanitize_storage_key(game_name),
        timestamp,
        CANONICAL_REPLAY_SUFFIX
    )
}

pub fn local_replay_path(game_name: &str, timestamp: u64) -> PathBuf {
    Path::new(REPLAY_DIR).join(canonical_replay_file_name(game_name, timestamp))
}

pub fn is_canonical_replay_file(path: &Path) -> bool {
    path.to_string_lossy().ends_with(CANONICAL_REPLAY_SUFFIX)
}

/// The quarantine file for a payload that could not be accepted. Its sibling
/// `<stem>.error.txt` holds the reason.
pub fn rejected_payload_path(game_name: &str, timestamp: u64) -> PathBuf {
    Path::new(REJECTED_REPLAY_DIR).join(format!(
        "{}_{}{}",
        sanitize_storage_key(game_name),
        timestamp,
        REJECTED_PAYLOAD_SUFFIX
    ))
}
