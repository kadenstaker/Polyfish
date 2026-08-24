//! The mod POSTs a pre-canonical capture payload; these pin the conversion that
//! makes `/replay/save` accept it again.

use polyfish::game::Game;
use polyfish::mapgen::{MapGenSettings, generate};
use polyfish::replay::legacy::convert_command;
use polyfish::replay::{
    ConvertedCommand, ReplayCommand, ReplayExecutor, ReplaySource, convert_mod_payload,
    is_legacy_mod_payload, validate_training_eligibility,
};
use polyfish::types::{
    AbilityType, CityRewardType, MapSize, MapType, MoveType, RuinsRewardType, StructureType,
    TechnologyType, TribeType, UnitType,
};
use serde_json::{Value, json};

const REAL_CAPTURE: &str = include_str!("fixtures/mod_replay_legacy_v114.json");

/// A capture payload shaped exactly like the mod's: a 0-based seat index where a
/// player id belongs, a Nature tribe, and match bookkeeping among the commands.
fn mod_payload() -> Value {
    let state = generate(MapGenSettings {
        size: MapSize::Tiny,
        map_type: MapType::Drylands,
        tribes: vec![TribeType::Imperius, TribeType::Bardur],
        seed: 731,
        version: 115,
        symmetric: false,
    });
    let mut game = Game::new();
    game.state = state.clone();
    game.post_load();
    let step = game
        .legal_moves()
        .into_iter()
        .find(|m| m.move_type() == MoveType::Step)
        .expect("the opening position offers a step");

    let mut game_state = serde_json::to_value(&state).unwrap();
    game_state["settings"]["currentPlayerTurnId"] = json!(0);
    let mut nature = game_state["tribes"]["1"].clone();
    nature["id"] = json!(255);
    nature["username"] = json!("Nature");
    nature["units"] = json!([]);
    nature["cities"] = json!([]);
    game_state["tribes"]["255"] = nature;

    json!({
        "uuid": "capture-under-test",
        "turns": [{
            "turn": 0,
            "players": [
                {
                    "playerId": 1,
                    "commands": [
                        {"moveType": -1},
                        {"moveType": 1, "src": step.source_idx().unwrap(), "target": step.target_idx().unwrap()},
                        {"moveType": 10}
                    ]
                },
                {
                    "playerId": 2,
                    "commands": [{"moveType": 10}, {"moveType": 11}]
                },
                {
                    "playerId": 255,
                    "commands": [{"moveType": 10}]
                }
            ]
        }],
        "gameState": game_state
    })
}

fn command_of(raw: Value) -> ReplayCommand {
    match convert_command(&raw).unwrap() {
        ConvertedCommand::Command(command) => command,
        other => panic!("expected a canonical command, got {other:?}"),
    }
}

#[test]
fn recognises_the_mod_payload_shape() {
    assert!(is_legacy_mod_payload(&mod_payload()));
    assert!(is_legacy_mod_payload(
        &serde_json::from_str::<Value>(REAL_CAPTURE).unwrap()
    ));

    let canonical = serde_json::to_value(convert_mod_payload(&mod_payload()).unwrap()).unwrap();
    assert!(!is_legacy_mod_payload(&canonical));
}

#[test]
fn converts_a_capture_the_engine_can_replay() {
    let replay = convert_mod_payload(&mod_payload()).unwrap();

    assert_eq!(replay.schema_version, 1);
    assert_eq!(replay.metadata.source, ReplaySource::PolytopiaProfessional);
    assert_eq!(
        replay.metadata.game_id.as_deref(),
        Some("capture-under-test")
    );
    assert_eq!(replay.metadata.map_width, 11);
    assert_eq!(replay.metadata.map_height, 11);
    assert_eq!(
        replay.metadata.max_turns,
        replay.initial_state.settings.max_turns
    );
    assert_eq!(
        replay.metadata.game_mode,
        replay.initial_state.settings.mode
    );

    // Nature is dropped, and the source's seat index becomes a real player id.
    let mut ids: Vec<_> = replay
        .metadata
        .players
        .iter()
        .map(|player| player.player_id)
        .collect();
    ids.sort();
    assert_eq!(ids, vec![1, 2]);
    assert!(!replay.initial_state.tribes.contains_key(&255));
    assert_eq!(replay.initial_state.settings.current_player_turn_id, 1);

    // startmatch and resign carry nothing canonical and are counted, not translated.
    assert_eq!(replay.command_count(), 3);
    assert!(
        !replay
            .turns
            .iter()
            .any(|turn| turn.commands.contains(&ReplayCommand::Resign))
    );
    let diagnostics = replay.metadata.source_diagnostics.as_ref().unwrap();
    assert_eq!(diagnostics["droppedNonMoveCommands"], json!(1));
    assert_eq!(diagnostics["droppedResignCommands"], json!(1));
    assert_eq!(diagnostics["droppedPlayers"], json!([255]));
    assert_eq!(diagnostics["sourceCommands"], json!(5));
    assert_eq!(diagnostics["convertedCommands"], json!(3));

    // The segments carry engine counters, not the source's timeline keys.
    assert_eq!(
        replay
            .turns
            .iter()
            .map(|turn| (turn.turn_number, turn.player_id))
            .collect::<Vec<_>>(),
        vec![(0, 1), (0, 2)]
    );
    ReplayExecutor::execute(&replay).unwrap();
}

#[test]
fn maps_every_source_command_slot() {
    assert_eq!(
        command_of(json!({"moveType": 1, "src": 4, "target": 5})),
        ReplayCommand::Step {
            source: 4,
            target: 5
        }
    );
    assert_eq!(
        command_of(json!({"moveType": 2, "src": 4, "target": 5})),
        ReplayCommand::Attack {
            source: 4,
            target: 5
        }
    );
    // `train` and `upgrade` both arrive as 4 keyed on `src`.
    assert_eq!(
        command_of(json!({"moveType": 4, "src": 7, "type": 2})),
        ReplayCommand::Summon {
            target: 7,
            unit: UnitType::Warrior
        }
    );
    assert_eq!(
        command_of(json!({"moveType": 5, "target": 9})),
        ReplayCommand::Harvest { target: 9 }
    );
    assert_eq!(
        command_of(json!({"moveType": 6, "target": 9, "type": 13})),
        ReplayCommand::Build {
            target: 9,
            structure: StructureType::Sawmill
        }
    );
    assert_eq!(
        command_of(json!({"moveType": 7, "type": 6})),
        ReplayCommand::Research {
            technology: TechnologyType::from(6)
        }
    );
    assert_eq!(command_of(json!({"moveType": 10})), ReplayCommand::EndTurn);

    // StarFishing is a build in the source and a capture in the engine.
    assert_eq!(
        command_of(json!({"moveType": 6, "target": 12, "type": 46})),
        ReplayCommand::Capture {
            source: 12,
            reward: None,
            revealed_tiles: None,
            technology: None
        }
    );
}

#[test]
fn ability_slots_survive_exactly_as_the_source_wrote_them() {
    // Forest actions report only a target; `matches_move` compares both slots.
    assert_eq!(
        command_of(json!({"moveType": 3, "target": 30, "type": 2})),
        ReplayCommand::Ability {
            source: None,
            target: Some(30),
            ability: AbilityType::ClearForest
        }
    );
    // Recover reports only a source.
    assert_eq!(
        command_of(json!({"moveType": 3, "src": 30, "type": 7})),
        ReplayCommand::Ability {
            source: Some(30),
            target: None,
            ability: AbilityType::Recover
        }
    );
    // The ice tile stays in `source`: `BreakIceMove::source_idx` is that tile,
    // and it exposes no target at all.
    assert_eq!(
        command_of(json!({"moveType": 3, "src": 44, "type": 17})),
        ReplayCommand::Ability {
            source: Some(44),
            target: None,
            ability: AbilityType::BreakIce
        }
    );
    // Diplomacy keys `src` on the opponent id, which is what the move reports.
    assert_eq!(
        command_of(json!({"moveType": 3, "src": 2, "target": 1, "type": 19})),
        ReplayCommand::Ability {
            source: Some(2),
            target: Some(1),
            ability: AbilityType::PeaceRequestResponse
        }
    );
}

#[test]
fn ruins_and_city_reward_hints_reach_the_canonical_command() {
    assert_eq!(
        command_of(json!({"moveType": 8, "src": 21, "_reward": 4, "_type": 6})),
        ReplayCommand::Capture {
            source: 21,
            reward: Some(RuinsRewardType::from(4)),
            revealed_tiles: None,
            technology: Some(TechnologyType::from(6))
        }
    );
    // Older captures spelled the free-tech hint differently.
    assert_eq!(
        command_of(json!({"moveType": 8, "src": 21, "_reward": 4, "tech_hint": 6})),
        ReplayCommand::Capture {
            source: 21,
            reward: Some(RuinsRewardType::from(4)),
            revealed_tiles: None,
            technology: Some(TechnologyType::from(6))
        }
    );
    assert_eq!(
        command_of(json!({"moveType": 9, "target": 33, "type": 3, "_revealedTiles": [1, 2, 3]})),
        ReplayCommand::Reward {
            target: 33,
            reward: CityRewardType::from(3),
            revealed_tiles: Some(vec![1, 2, 3])
        }
    );
}

#[test]
fn drops_what_the_engine_has_no_move_for() {
    assert_eq!(
        convert_command(&json!({"moveType": -1})).unwrap(),
        ConvertedCommand::NonMove
    );
    assert_eq!(
        convert_command(&json!({"moveType": 11})).unwrap(),
        ConvertedCommand::Resign
    );
}

#[test]
fn refuses_commands_it_cannot_resolve() {
    let error = convert_command(&json!({"error": "Unknown command", "tid": "swarm"}))
        .unwrap_err()
        .to_string();
    assert!(error.contains("swarm"), "{error}");

    let error = convert_command(&json!({"moveType": 99}))
        .unwrap_err()
        .to_string();
    assert!(error.contains("unknown source moveType 99"), "{error}");

    let error = convert_command(&json!({"moveType": 1, "src": 4}))
        .unwrap_err()
        .to_string();
    assert!(error.contains("`target`"), "{error}");
}

/// The one real capture kept in-tree. It is a 14x14 game, so it converts and
/// replays but can never become training data: `features::MAP_SIZE` is 11.
#[test]
fn real_mod_capture_converts_but_is_not_training_eligible() {
    let payload: Value = serde_json::from_str(REAL_CAPTURE).unwrap();
    let replay = convert_mod_payload(&payload).unwrap();

    assert_eq!(replay.initial_state.settings.version, 114);
    assert_eq!(replay.metadata.map_width, 14);
    assert!(!replay.initial_state.tribes.contains_key(&255));
    assert_eq!(replay.command_count(), 247);
    ReplayExecutor::execute(&replay).unwrap();

    // The mod snapshots after the game plays its own forced opening, so exactly
    // one reported command is already in the captured state.
    let diagnostics = replay.metadata.source_diagnostics.as_ref().unwrap();
    assert_eq!(diagnostics["sourceCommands"], json!(251));
    assert_eq!(diagnostics["droppedNonMoveCommands"], json!(2));
    assert_eq!(diagnostics["droppedResignCommands"], json!(1));
    assert_eq!(diagnostics["droppedPlayers"], json!([255]));
    assert_eq!(
        diagnostics["preAppliedCommands"],
        json!(["Step { source: 163, target: 148 }"])
    );

    let error = validate_training_eligibility(&replay)
        .unwrap_err()
        .to_string();
    assert!(error.contains("14x14"), "{error}");
}
