//! Pins the default ruleset so `GameSettings` cannot silently drift away from
//! the version every generated map and every training game plays (#51).

use polyfish::mapgen::MapGenSettings;
use polyfish::states::{GameSettings, GameState};
use polyfish::version_sync::{CURRENT_VERSION, GameVersion};

#[test]
fn default_state_runs_the_current_ruleset() {
    assert_eq!(GameState::default().settings.version, CURRENT_VERSION);
}

#[test]
fn mapgen_default_matches_state_default() {
    assert_eq!(
        MapGenSettings::default().version,
        GameState::default().settings.version
    );
}

#[test]
fn serde_default_matches_struct_default() {
    let settings: GameSettings = serde_json::from_str(r#"{"mode":0,"size":11}"#).unwrap();
    assert_eq!(settings.version, CURRENT_VERSION);
}

/// Scraper-era data pins itself to 0 explicitly; that must keep meaning legacy
/// rules and legacy feature scaling, not the current ruleset.
#[test]
fn explicit_zero_still_means_legacy() {
    let settings: GameSettings =
        serde_json::from_str(r#"{"mode":0,"size":11,"version":0}"#).unwrap();
    assert_eq!(settings.version, GameVersion::Legacy as i32);
}
