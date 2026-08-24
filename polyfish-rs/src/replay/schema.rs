use crate::states::{GameState, PlayerId};
use crate::types::{ModeType, TribeType};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use super::ReplayCommand;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Replay {
    pub schema_version: u32,
    pub metadata: ReplayMetadata,
    pub initial_state: GameState,
    pub turns: Vec<ReplayTurn>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<ReplayResult>,
}

impl Replay {
    pub fn command_count(&self) -> usize {
        self.turns.iter().map(|turn| turn.commands.len()).sum()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ReplayMetadata {
    pub source: ReplaySource,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub game_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created_at: Option<String>,
    pub map_width: usize,
    pub map_height: usize,
    pub max_turns: i32,
    pub game_mode: ModeType,
    pub players: Vec<ReplayPlayerMetadata>,
    /// Source-only diagnostics. Execution never reads this field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_diagnostics: Option<serde_json::Value>,
}

impl ReplayMetadata {
    /// The source game's per-EndTurn checkpoints, empty when it wrote none.
    /// Unknown diagnostic keys stay tolerated: only `endTurnCheckpoints` is read.
    pub fn end_turn_checkpoints(&self) -> Result<Vec<SourceCheckpoint>, serde_json::Error> {
        let Some(value) = self
            .source_diagnostics
            .as_ref()
            .and_then(|diagnostics| diagnostics.get("endTurnCheckpoints"))
        else {
            return Ok(Vec::new());
        };
        serde_json::from_value(value.clone())
    }
}

/// What the capturing source observed for one player immediately before it
/// issued that player's EndTurn. Compared by `replay::verify`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct SourceCheckpoint {
    pub turn_number: i32,
    pub player_id: PlayerId,
    pub score: i32,
    pub stars: i32,
    pub unit_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum ReplaySource {
    PolyfishSelfPlay,
    PolyfishUi,
    PolytopiaProfessional,
    Other,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ReplayPlayerMetadata {
    pub player_id: PlayerId,
    pub tribe: TribeType,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// One contiguous action segment for one player. `turn_number` is the exact
/// Polyfish engine turn counter before the first command in this segment.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ReplayTurn {
    pub turn_number: i32,
    pub player_id: PlayerId,
    pub commands: Vec<ReplayCommand>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ReplayResult {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub winner_player_id: Option<PlayerId>,
    #[serde(default)]
    pub draw: bool,
    #[serde(default)]
    pub scores: BTreeMap<PlayerId, i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}
