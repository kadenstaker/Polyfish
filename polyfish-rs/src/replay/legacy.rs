//! Converts the capture payload the Polytopia mod produces into a canonical
//! `Replay`.
//!
//! The payload predates the canonical schema: it carries Polytopia's own
//! command ids, a Nature tribe, a 0-based seat index where a player id belongs,
//! and commands the engine has no move for. Conversion walks the engine rather
//! than reshaping the JSON, so every segment's turn number and player id come
//! from the engine itself and a converted replay always re-executes.

use serde_json::{Map, Value};

use crate::game::Game;
use crate::states::GameState;
use crate::types::{
    AbilityType, CityRewardType, RuinsRewardType, StructureType, TechnologyType, UnitType,
};

use super::{
    NoopReplayObserver, Replay, ReplayCommand, ReplayError, ReplayExecutor, ReplayMetadata,
    ReplayMoveContext, ReplayPlayerMetadata, ReplayRecorder, ReplaySource, validate_replay,
};

/// Polytopia's animal player. Never a replay participant.
const NATURE_PLAYER_ID: i32 = 255;
/// `ImprovementData.Type.StarFishing`, which the source reports as a build but
/// the engine models as a capture.
const STAR_FISHING_IMPROVEMENT: i64 = 46;
/// The mod cannot snapshot the game before the game plays its own forced
/// opening, so the captured state can already reflect the first commands it
/// then reports. Those are skipped and counted. The allowance is deliberately
/// tight: past it, an illegal opening is a conversion defect, not a snapshot
/// that ran late, and must fail loudly.
const MAX_PRE_APPLIED_COMMANDS: usize = 2;

/// One source command, or the reason it carries nothing canonical.
#[derive(Debug, Clone, PartialEq)]
pub enum ConvertedCommand {
    Command(ReplayCommand),
    /// `startmatch`/`endmatch`: match bookkeeping, not a move.
    NonMove,
    /// `generate_legal_moves` never emits `Resign`, so a canonical resign
    /// command could not be replayed.
    Resign,
}

pub fn is_legacy_mod_payload(payload: &Value) -> bool {
    payload.get("schemaVersion").is_none()
        && payload.get("gameState").is_some()
        && payload.get("turns").is_some_and(Value::is_array)
}

/// Maps one source command onto its canonical counterpart. Enum ids go through
/// `From<i32>`, never `Deserialize`, so an id this engine build does not know
/// degrades to `None`/`Basic` instead of failing the whole capture.
pub fn convert_command(raw: &Value) -> Result<ConvertedCommand, ReplayError> {
    if let Some(error) = raw.get("error").and_then(Value::as_str) {
        let tid = raw.get("tid").and_then(Value::as_str).unwrap_or("unknown");
        return Err(ReplayError::validation(format!(
            "source could not serialize command `{tid}`: {error}"
        )));
    }
    let command = match int_field(raw, "moveType")? {
        -1 => return Ok(ConvertedCommand::NonMove),
        11 => return Ok(ConvertedCommand::Resign),
        1 => ReplayCommand::Step {
            source: idx(raw, "src")?,
            target: idx(raw, "target")?,
        },
        2 => ReplayCommand::Attack {
            source: idx(raw, "src")?,
            target: idx(raw, "target")?,
        },
        // The source writes whichever slot the Polytopia command carries, and
        // the engine's ability moves expose the matching one, `breakice`
        // included (`BreakIceMove::source_idx` is the ice tile). Passing both
        // through unchanged is what makes `matches_move`'s exact slot
        // comparison agree.
        3 => ReplayCommand::Ability {
            source: opt_idx(raw, "src")?,
            target: opt_idx(raw, "target")?,
            ability: AbilityType::from(int_field(raw, "type")? as i32),
        },
        // Polytopia `train` and `upgrade` both arrive as 4 with no
        // distinguishing field; `Summon` matches SummonMove and UpgradeMove
        // alike, while the canonical `Upgrade` variant matches neither.
        4 => ReplayCommand::Summon {
            target: idx(raw, "src")?,
            unit: UnitType::from(int_field(raw, "type")? as i32),
        },
        5 => ReplayCommand::Harvest {
            target: idx(raw, "target")?,
        },
        6 => {
            let improvement = int_field(raw, "type")?;
            if improvement == STAR_FISHING_IMPROVEMENT {
                ReplayCommand::Capture {
                    source: idx(raw, "target")?,
                    reward: None,
                    revealed_tiles: None,
                    technology: None,
                }
            } else {
                ReplayCommand::Build {
                    target: idx(raw, "target")?,
                    structure: StructureType::from(improvement as i32),
                }
            }
        }
        7 => ReplayCommand::Research {
            technology: TechnologyType::from(int_field(raw, "type")? as i32),
        },
        8 => ReplayCommand::Capture {
            source: idx(raw, "src")?,
            reward: opt_int(raw, "_reward").map(|value| RuinsRewardType::from(value as i32)),
            revealed_tiles: revealed_tiles(raw)?,
            technology: tech_hint(raw),
        },
        9 => ReplayCommand::Reward {
            target: idx(raw, "target")?,
            reward: CityRewardType::from(int_field(raw, "type")? as i32),
            revealed_tiles: revealed_tiles(raw)?,
        },
        10 => ReplayCommand::EndTurn,
        other => {
            return Err(ReplayError::validation(format!(
                "unknown source moveType {other} in {raw}"
            )));
        }
    };
    Ok(ConvertedCommand::Command(command))
}

pub fn convert_mod_payload(payload: &Value) -> Result<Replay, ReplayError> {
    let state_value = payload
        .get("gameState")
        .ok_or_else(|| ReplayError::validation("source payload has no `gameState`"))?;
    let mut state: GameState = serde_json::from_value(state_value.clone()).map_err(|source| {
        ReplayError::validation(format!(
            "source `gameState` is not an engine GameState: {source}"
        ))
    })?;

    let dropped_players: Vec<i32> = state
        .tribes
        .keys()
        .copied()
        .filter(|&id| !is_replay_player(id))
        .collect();
    state.tribes.retain(|&id, _| is_replay_player(id));

    let segments = segments(payload)?;

    // The source writes its 0-based seat index here, not a player id.
    if !state
        .tribes
        .contains_key(&state.settings.current_player_turn_id)
    {
        state.settings.current_player_turn_id = segments
            .first()
            .map(|segment| segment.player_id)
            .filter(|id| state.tribes.contains_key(id))
            .or_else(|| state.tribes.keys().copied().next())
            .ok_or_else(|| ReplayError::validation("source payload has no players"))?;
    }

    let size = state.settings.size.max(0) as usize;
    let metadata = ReplayMetadata {
        source: ReplaySource::PolytopiaProfessional,
        game_id: payload
            .get("uuid")
            .and_then(Value::as_str)
            .filter(|uuid| !uuid.is_empty())
            .map(str::to_string),
        created_at: None,
        map_width: size,
        map_height: size,
        max_turns: state.settings.max_turns,
        game_mode: state.settings.mode,
        players: state
            .tribes
            .iter()
            .map(|(&player_id, tribe)| ReplayPlayerMetadata {
                player_id,
                tribe: tribe.tribe_type,
                name: Some(tribe.username.clone()).filter(|name| !name.is_empty()),
            })
            .collect(),
        source_diagnostics: None,
    };

    let mut game = Game::new();
    game.state = state.clone();
    game.post_load();
    let mut recorder = ReplayRecorder::new(state, metadata);

    let mut counts = DropCounts::default();
    let mut source_commands = 0usize;
    let mut converted_commands = 0usize;
    let mut pre_applied: Vec<String> = Vec::new();

    for (turn_index, segment) in segments.iter().enumerate() {
        let mut seat_checked = false;
        for (command_index, raw) in segment.commands.iter().enumerate() {
            source_commands += 1;
            let at = |error: ReplayError| {
                ReplayError::validation(format!(
                    "source turn {} player {} command {command_index}: {error}",
                    segment.turn, segment.player_id
                ))
            };
            let command = match convert_command(raw).map_err(at)? {
                ConvertedCommand::Command(command) => command,
                ConvertedCommand::NonMove => {
                    counts.non_move += 1;
                    continue;
                }
                ConvertedCommand::Resign => {
                    counts.resign += 1;
                    continue;
                }
            };
            if raw.get("_revealedTiles").is_some()
                && !matches!(
                    command,
                    ReplayCommand::Capture { .. } | ReplayCommand::Reward { .. }
                )
            {
                counts.revealed_tile_hints += 1;
            }
            let active = game.state.settings.current_player_turn_id;
            if !seat_checked {
                if segment.player_id != active {
                    return Err(ReplayError::validation(format!(
                        "source turn {} declares player {}, but the engine is on player {active} after {converted_commands} commands",
                        segment.turn, segment.player_id
                    )));
                }
                seat_checked = true;
            }
            let context = ReplayMoveContext {
                turn_index,
                turn_number: game.state.settings.turn,
                player_id: active,
                command_index,
                global_command_index: converted_commands,
            };
            match ReplayExecutor::execute_command(
                &mut game,
                &command,
                &context,
                &mut NoopReplayObserver,
            ) {
                Ok(()) => {}
                Err(error) => {
                    let snapshot_ran_late = converted_commands == 0
                        && pre_applied.len() < MAX_PRE_APPLIED_COMMANDS
                        && matches!(error, ReplayError::IllegalCommand { .. });
                    if !snapshot_ran_late {
                        return Err(at(error));
                    }
                    pre_applied.push(format!("{command:?}"));
                    continue;
                }
            }
            recorder.record_command(context.turn_number, active, command)?;
            converted_commands += 1;
        }
    }

    let mut diagnostics = Map::new();
    diagnostics.insert("converter".into(), Value::from("legacyModPayload"));
    diagnostics.insert("sourceCommands".into(), Value::from(source_commands));
    diagnostics.insert("convertedCommands".into(), Value::from(converted_commands));
    diagnostics.insert(
        "droppedNonMoveCommands".into(),
        Value::from(counts.non_move),
    );
    diagnostics.insert("droppedResignCommands".into(), Value::from(counts.resign));
    diagnostics.insert(
        "droppedRevealedTileHints".into(),
        Value::from(counts.revealed_tile_hints),
    );
    diagnostics.insert("droppedPlayers".into(), Value::from(dropped_players));
    diagnostics.insert("preAppliedCommands".into(), Value::from(pre_applied));

    let mut replay = recorder.finish(None);
    replay.metadata.source_diagnostics = Some(Value::Object(diagnostics));
    validate_replay(&replay, None)?;
    Ok(replay)
}

#[derive(Default)]
struct DropCounts {
    non_move: usize,
    resign: usize,
    revealed_tile_hints: usize,
}

struct LegacySegment<'a> {
    turn: i32,
    player_id: i32,
    commands: &'a [Value],
}

fn is_replay_player(id: i32) -> bool {
    id > 0 && id != NATURE_PLAYER_ID
}

/// Flattens `turns[].players[]` into play order. The source's `turn` key is
/// kept for error messages only; the engine owns the real turn counter.
fn segments(payload: &Value) -> Result<Vec<LegacySegment<'_>>, ReplayError> {
    let turns = payload
        .get("turns")
        .and_then(Value::as_array)
        .ok_or_else(|| ReplayError::validation("source payload has no `turns` array"))?;
    let mut segments = Vec::new();
    for (turn_index, turn) in turns.iter().enumerate() {
        let key = turn
            .get("turn")
            .and_then(Value::as_i64)
            .unwrap_or(turn_index as i64) as i32;
        let players = turn
            .get("players")
            .and_then(Value::as_array)
            .ok_or_else(|| {
                ReplayError::validation(format!("source turn {key} has no `players` array"))
            })?;
        for player in players {
            let player_id = player
                .get("playerId")
                .and_then(Value::as_i64)
                .ok_or_else(|| {
                    ReplayError::validation(format!("source turn {key} has a player with no id"))
                })? as i32;
            if !is_replay_player(player_id) {
                continue;
            }
            segments.push(LegacySegment {
                turn: key,
                player_id,
                commands: player
                    .get("commands")
                    .and_then(Value::as_array)
                    .map(Vec::as_slice)
                    .unwrap_or(&[]),
            });
        }
    }
    Ok(segments)
}

fn int_field(raw: &Value, key: &str) -> Result<i64, ReplayError> {
    raw.get(key).and_then(Value::as_i64).ok_or_else(|| {
        ReplayError::validation(format!(
            "source command is missing integer field `{key}`: {raw}"
        ))
    })
}

fn idx(raw: &Value, key: &str) -> Result<i32, ReplayError> {
    Ok(int_field(raw, key)? as i32)
}

fn opt_idx(raw: &Value, key: &str) -> Result<Option<i32>, ReplayError> {
    match raw.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(_) => idx(raw, key).map(Some),
    }
}

fn opt_int(raw: &Value, key: &str) -> Option<i64> {
    raw.get(key).and_then(Value::as_i64)
}

/// The mod writes the free-tech id under `_type`; older captures used
/// `_techHint` or `tech_hint`.
fn tech_hint(raw: &Value) -> Option<TechnologyType> {
    ["_type", "_techHint", "tech_hint"]
        .iter()
        .find_map(|key| opt_int(raw, key))
        .map(|value| TechnologyType::from(value as i32))
}

fn revealed_tiles(raw: &Value) -> Result<Option<Vec<i32>>, ReplayError> {
    match raw.get("_revealedTiles") {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Array(items)) => items
            .iter()
            .map(|item| {
                item.as_i64().map(|value| value as i32).ok_or_else(|| {
                    ReplayError::validation(format!("`_revealedTiles` holds a non-integer: {item}"))
                })
            })
            .collect::<Result<Vec<_>, _>>()
            .map(Some),
        Some(other) => Err(ReplayError::validation(format!(
            "`_revealedTiles` must be an array, got {other}"
        ))),
    }
}
