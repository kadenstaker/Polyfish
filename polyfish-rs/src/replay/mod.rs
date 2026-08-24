//! Versioned, engine-authoritative replay support.
//!
//! JSON parsing, structural validation, legal-move resolution, playback and
//! training extraction deliberately share this one domain model.

pub mod command;
pub mod errors;
pub mod executor;
pub mod legacy;
pub mod loader;
pub mod outcome;
pub mod paths;
pub mod playback;
pub mod recorder;
pub mod schema;
pub mod training;
pub mod validator;
pub mod verify;

pub use command::ReplayCommand;
pub use errors::{ReplayError, ReplayMoveContext};
pub use executor::{NoopReplayObserver, ReplayExecutor, ReplayObserver};
pub use legacy::{ConvertedCommand, convert_mod_payload, is_legacy_mod_payload};
pub use loader::{load_replay, load_replay_reader, save_replay};
pub use outcome::derive_result;
pub use paths::{
    CANONICAL_REPLAY_SUFFIX, REJECTED_PAYLOAD_SUFFIX, REJECTED_REPLAY_DIR, REPLAY_DIR,
    canonical_replay_file_name, is_canonical_replay_file, local_replay_path, rejected_payload_path,
    sanitize_storage_key,
};
pub use playback::ReplayPlayback;
pub use recorder::ReplayRecorder;
pub use schema::*;
pub use validator::{
    MAX_SUPPORTED_GAME_VERSION, MIN_SUPPORTED_GAME_VERSION, TrainingEligibility, VersionSupport,
    classify_game_version, validate_replay, validate_training_eligibility,
    validate_training_eligibility_with,
};
pub use verify::{DivergenceVerifier, PairObserver};

pub const REPLAY_SCHEMA_VERSION: u32 = 1;
pub const DATASET_SCHEMA_VERSION: u32 = 1;
pub const FEATURE_SCHEMA_VERSION: u32 = 1;
pub const ACTION_SCHEMA_VERSION: u32 = 1;

#[cfg(test)]
mod tests;
