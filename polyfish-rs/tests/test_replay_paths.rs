//! Producer and consumer must agree on the canonical replay name.
//!
//! The no-Supabase fallback used to write `replays/{name}_{ts}.json` while the
//! startup loader, `import_replays` and `upload_replays` all filtered for
//! `.replay.json`, so every accepted replay was saved under a name no consumer
//! could see.

use polyfish::replay::{
    CANONICAL_REPLAY_SUFFIX, REPLAY_DIR, canonical_replay_file_name, converted_payload_stem,
    is_canonical_replay_file, local_replay_path, sanitize_storage_key,
};
use std::path::Path;

#[test]
fn every_produced_name_is_one_a_consumer_accepts() {
    for name in ["Some Game 1", "", "  ", "a-b_c", "Zoë's Game!!"] {
        let path = local_replay_path(name, 1_756_000_000);
        assert!(
            is_canonical_replay_file(&path),
            "{path:?} is not a canonical replay name"
        );
        assert_eq!(path.parent(), Some(Path::new(REPLAY_DIR)));
    }
}

#[test]
fn file_name_is_the_storage_key() {
    let path = local_replay_path("Some Game 1", 42);
    assert_eq!(
        path.file_name().and_then(|n| n.to_str()),
        Some(canonical_replay_file_name("Some Game 1", 42).as_str())
    );
    assert_eq!(
        canonical_replay_file_name("Some Game 1", 42),
        format!("some-game-1_42{CANONICAL_REPLAY_SUFFIX}")
    );
}

#[test]
fn sanitizer_collapses_runs_and_never_returns_an_empty_stem() {
    assert_eq!(
        sanitize_storage_key("The Winter of Love"),
        "the-winter-of-love"
    );
    assert_eq!(
        sanitize_storage_key("game-4-(yădakk-qualifiers-"),
        "game-4-y-dakk-qualifiers"
    );
    assert_eq!(sanitize_storage_key("Hello World!!!"), "hello-world");
    assert_eq!(
        sanitize_storage_key("---Multiple---Dashes---"),
        "multiple-dashes"
    );
    assert_eq!(sanitize_storage_key("UPPER_case_123"), "upper_case_123");
    assert_eq!(sanitize_storage_key("a-b_c"), "a-b_c");
    assert_eq!(sanitize_storage_key(""), "replay");
    assert_eq!(sanitize_storage_key("!@#$%^&*()"), "replay");
}

#[test]
fn legacy_fallback_name_is_rejected() {
    assert!(!is_canonical_replay_file(Path::new("replays/game_1.json")));
    assert!(is_canonical_replay_file(Path::new(
        "replays/game_1.replay.json"
    )));
}

/// `convert-legacy --recursive` flattens a tree into one output directory, so two
/// captures sharing a basename used to write the same file: the second truncated
/// the first and both were still reported CONVERTED.
#[test]
fn converted_stems_survive_a_flattened_tree() {
    let input = Path::new("/captures");
    let first = converted_payload_stem(input, Path::new("/captures/day1/capture.json"));
    let second = converted_payload_stem(input, Path::new("/captures/day2/capture.json"));
    assert_ne!(
        first, second,
        "same basename in different subdirectories must not share an output name"
    );

    // A file given directly, and a payload directly under the input dir, both keep
    // the plain stem a caller expects.
    let direct = Path::new("/captures/day1/capture.json");
    assert_eq!(converted_payload_stem(direct, direct), "capture");
    assert_eq!(
        converted_payload_stem(input, Path::new("/captures/capture.json")),
        "capture"
    );
}
