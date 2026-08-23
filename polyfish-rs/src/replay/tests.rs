use super::*;
use crate::ai::features;
use crate::ai::mapper::NUM_MOVE_OPTIONS;
use crate::ai::network::NUM_ACTION_TYPES;
use crate::mapgen::{MapGenSettings, generate};
use crate::moves::{EndTurnMove, Move};
use crate::types::{MapSize, MapType, TribeType};
use std::collections::BTreeMap;

fn replay_with(commands: Vec<ReplayCommand>) -> Replay {
    let initial_state = generate(MapGenSettings {
        size: MapSize::Tiny,
        map_type: MapType::Drylands,
        tribes: vec![TribeType::Imperius, TribeType::Bardur],
        seed: 731,
        version: 115,
        symmetric: false,
    });
    Replay {
        schema_version: REPLAY_SCHEMA_VERSION,
        metadata: ReplayMetadata {
            source: ReplaySource::Other,
            game_id: Some("replay-test".into()),
            created_at: None,
            map_width: 11,
            map_height: 11,
            max_turns: initial_state.settings.max_turns,
            game_mode: initial_state.settings.mode,
            players: initial_state
                .tribes
                .iter()
                .map(|(&player_id, tribe)| ReplayPlayerMetadata {
                    player_id,
                    tribe: tribe.tribe_type,
                    name: None,
                })
                .collect(),
            source_diagnostics: None,
        },
        turns: vec![ReplayTurn {
            turn_number: initial_state.settings.turn,
            player_id: initial_state.settings.current_player_turn_id,
            commands,
        }],
        initial_state,
        result: None,
    }
}

#[test]
fn valid_replay_plays_every_command() {
    let replay = replay_with(vec![ReplayCommand::EndTurn]);
    let game = ReplayExecutor::execute(&replay).unwrap();
    assert_eq!(game.state.settings.current_player_turn_id, 2);
}

#[test]
fn rejects_invalid_schema_version() {
    let mut replay = replay_with(vec![ReplayCommand::EndTurn]);
    replay.schema_version = 99;
    assert!(
        validate_replay(&replay, None)
            .unwrap_err()
            .to_string()
            .contains("unsupported schemaVersion 99")
    );
}

#[test]
fn rejects_incorrect_dimensions_and_training_size() {
    let mut replay = replay_with(vec![ReplayCommand::EndTurn]);
    replay.metadata.map_width = 14;
    assert!(
        validate_replay(&replay, None)
            .unwrap_err()
            .to_string()
            .contains("square engine maps")
    );

    let mut replay = replay_with(vec![ReplayCommand::EndTurn]);
    replay.metadata.map_width = 14;
    replay.metadata.map_height = 14;
    assert!(
        validate_training_eligibility(&replay)
            .unwrap_err()
            .to_string()
            .contains("supports only 11x11")
    );
}

#[test]
fn rejects_out_of_range_tile_unknown_player_and_wrong_active_player() {
    let replay = replay_with(vec![ReplayCommand::Step {
        source: 0,
        target: 121,
    }]);
    assert!(
        validate_replay(&replay, None)
            .unwrap_err()
            .to_string()
            .contains("outside 0..121")
    );

    let mut replay = replay_with(vec![ReplayCommand::EndTurn]);
    replay.turns[0].player_id = 77;
    assert!(
        validate_replay(&replay, None)
            .unwrap_err()
            .to_string()
            .contains("unknown player 77")
    );

    let mut replay = replay_with(vec![ReplayCommand::EndTurn]);
    replay.turns[0].player_id = 2;
    let error = ReplayExecutor::execute(&replay).unwrap_err();
    assert!(matches!(
        error,
        ReplayError::ActivePlayer {
            declared: 2,
            actual: 1,
            ..
        }
    ));
}

#[test]
fn rejects_illegal_and_detects_ambiguous_commands() {
    let replay = replay_with(vec![ReplayCommand::Step {
        source: 0,
        target: 0,
    }]);
    assert!(matches!(
        ReplayExecutor::execute(&replay).unwrap_err(),
        ReplayError::IllegalCommand { .. }
    ));

    let duplicate: Vec<Box<dyn Move>> = vec![Box::new(EndTurnMove), Box::new(EndTurnMove)];
    assert_eq!(
        super::executor::matching_move_indices(&ReplayCommand::EndTurn, &duplicate),
        vec![0, 1]
    );
}

#[test]
fn recorder_keeps_end_turn_and_playback_reconstructs_backwards() {
    let replay = replay_with(Vec::new());
    let mut recorder = ReplayRecorder::new(replay.initial_state.clone(), replay.metadata.clone());
    recorder.record_move(0, 1, &EndTurnMove).unwrap();
    let replay = recorder.finish(None);
    assert_eq!(replay.command_count(), 1);
    assert_eq!(replay.turns[0].commands, vec![ReplayCommand::EndTurn]);

    let mut playback = ReplayPlayback::new(replay).unwrap();
    playback.seek(1).unwrap();
    assert_eq!(playback.game().state.settings.current_player_turn_id, 2);
    playback.seek(0).unwrap();
    assert_eq!(playback.cursor(), 0);
    assert_eq!(playback.game().state.settings.current_player_turn_id, 1);
}

#[test]
fn training_records_atomic_end_turn_with_pov_value_and_zero_optional_heads() {
    let mut replay = replay_with(vec![ReplayCommand::EndTurn]);
    replay.turns.push(ReplayTurn {
        turn_number: replay.initial_state.settings.turn,
        player_id: 2,
        commands: vec![ReplayCommand::EndTurn],
    });
    replay.result = Some(ReplayResult {
        winner_player_id: Some(1),
        draw: false,
        scores: BTreeMap::from([(1, 100), (2, 50)]),
        reason: Some("test".into()),
    });
    let mut collector = crate::replay::training::TrainingCollector::new(&replay).unwrap();
    let game = ReplayExecutor::execute_with_observer(&replay, &mut collector).unwrap();
    assert_eq!(collector.len(), 2);
    let samples = collector
        .finish(
            &game,
            replay.result.as_ref(),
            std::path::Path::new("test.replay.json"),
        )
        .unwrap();
    assert_eq!(samples.len(), 2);
    assert_eq!(samples[0].value(), 1.0);
    assert_eq!(samples[1].value(), -1.0);
    assert_eq!(samples[0].targets().action_type, 10);
    assert_eq!(samples[0].targets().source_spatial, None);
    assert_eq!(samples[0].targets().target_spatial, None);
    assert_eq!(samples[0].targets().target_type, None);
}

#[test]
fn safetensors_writer_emits_expected_shapes() {
    let mut replay = replay_with(vec![ReplayCommand::EndTurn]);
    replay.result = Some(ReplayResult {
        winner_player_id: Some(2),
        draw: false,
        scores: BTreeMap::from([(1, 10), (2, 20)]),
        reason: None,
    });
    let mut collector = crate::replay::training::TrainingCollector::new(&replay).unwrap();
    let game = ReplayExecutor::execute_with_observer(&replay, &mut collector).unwrap();
    let samples = collector
        .finish(
            &game,
            replay.result.as_ref(),
            std::path::Path::new("shape.replay.json"),
        )
        .unwrap();
    let dir = std::env::temp_dir().join(format!("polyfish-replay-test-{}", std::process::id()));
    let paths = crate::replay::training::write_training_files(&samples, &dir, 10).unwrap();
    let tensors = candle_core::safetensors::load(&paths[0], &candle_core::Device::Cpu).unwrap();
    assert_eq!(
        tensors["spatial_maps"].dims(),
        &[1, features::NUM_CHANNELS * 121]
    );
    assert_eq!(
        tensors["player_states"].dims(),
        &[1, features::RawFeatures::PLAYER_STATE_DIM]
    );
    assert_eq!(tensors["action_type"].dims(), &[1, NUM_ACTION_TYPES]);
    assert_eq!(tensors["source_spatial"].dims(), &[1, 121]);
    assert_eq!(tensors["target_spatial"].dims(), &[1, 121]);
    assert_eq!(tensors["move_option"].dims(), &[1, NUM_MOVE_OPTIONS]);
    let manifest: serde_json::Value = serde_json::from_slice(
        &std::fs::read(format!("{}.manifest.json", paths[0].display())).unwrap(),
    )
    .unwrap();
    assert_eq!(
        manifest["moveOptionDim"].as_u64(),
        Some(NUM_MOVE_OPTIONS as u64)
    );
    assert_eq!(
        manifest["numActionTypes"].as_u64(),
        Some(NUM_ACTION_TYPES as u64)
    );
    assert_eq!(
        manifest["derivedResultSourceFiles"],
        serde_json::json!([]),
        "a captured result must not be marked as derived"
    );
    assert_eq!(tensors["progress_mask"].dims(), &[1]);
    assert_eq!(tensors["aux_mask"].dims(), &[1]);
    let action_type = tensors["action_type"].to_vec2::<f32>().unwrap();
    assert_eq!(action_type[0].iter().sum::<f32>(), 1.0);
    assert_eq!(action_type[0][10], 1.0);
    assert_eq!(
        tensors["source_spatial"].to_vec2::<f32>().unwrap()[0]
            .iter()
            .sum::<f32>(),
        0.0
    );
    assert_eq!(
        tensors["target_spatial"].to_vec2::<f32>().unwrap()[0]
            .iter()
            .sum::<f32>(),
        0.0
    );
    assert_eq!(
        tensors["move_option"].to_vec2::<f32>().unwrap()[0]
            .iter()
            .sum::<f32>(),
        0.0
    );
    std::fs::remove_file(format!("{}.manifest.json", paths[0].display())).unwrap();
    std::fs::remove_file(&paths[0]).unwrap();
    std::fs::remove_dir(&dir).unwrap();
}

/// Terminal-ish state built by hand: `(id, score, killed)` per tribe.
/// Mirrors `ai::mcts_common::tests::terminal_game`.
fn game_with_tribes(tribes: &[(i32, i32, bool)], game_over: bool) -> crate::game::Game {
    let mut game = crate::game::Game::new();
    game.state.tribes.clear();
    for &(id, score, killed) in tribes {
        game.state.tribes.insert(
            id,
            crate::states::TribeState {
                id,
                score,
                killed_turn: if killed { 3 } else { 0 },
                ..Default::default()
            },
        );
    }
    game.state.settings._game_over = game_over;
    game
}

#[test]
fn derives_elimination_winner_over_a_dead_score_leader() {
    let game = game_with_tribes(&[(1, 40, false), (2, 900, true)], true);
    let result = derive_result(&game).unwrap();
    assert_eq!(result.winner_player_id, Some(1));
    assert!(!result.draw);
    assert_eq!(result.reason.as_deref(), Some("derived:elimination"));
    assert_eq!(result.scores, BTreeMap::from([(1, 40), (2, 900)]));
}

#[test]
fn derives_turn_limit_winner_from_score_among_the_living() {
    let game = game_with_tribes(&[(1, 100, false), (2, 40, false), (3, 900, true)], true);
    let result = derive_result(&game).unwrap();
    assert_eq!(result.winner_player_id, Some(1));
    assert_eq!(result.reason.as_deref(), Some("derived:scoreAtLimit"));
}

#[test]
fn derives_draw_on_mutual_elimination() {
    let game = game_with_tribes(&[(1, 100, true), (2, 40, true)], true);
    let result = derive_result(&game).unwrap();
    assert_eq!(result.winner_player_id, None);
    assert!(result.draw);
    assert_eq!(result.reason.as_deref(), Some("derived:mutualElimination"));
}

#[test]
fn derives_draw_on_a_score_tie_at_the_turn_limit() {
    let game = game_with_tribes(&[(1, 40, false), (2, 40, false)], true);
    let result = derive_result(&game).unwrap();
    assert_eq!(result.winner_player_id, None);
    assert!(result.draw);
    assert_eq!(result.reason.as_deref(), Some("derived:scoreTieAtLimit"));
}

#[test]
fn refuses_to_derive_from_a_non_terminal_state() {
    let game = game_with_tribes(&[(1, 100, false), (2, 40, false)], false);
    let error = derive_result(&game).unwrap_err().to_string();
    assert!(
        error.contains("cannot derive a result"),
        "unexpected error: {error}"
    );
}

#[test]
fn finish_derives_a_result_when_the_replay_has_none() {
    let mut replay = replay_with(vec![ReplayCommand::EndTurn]);
    replay.turns.push(ReplayTurn {
        turn_number: replay.initial_state.settings.turn,
        player_id: 2,
        commands: vec![ReplayCommand::EndTurn],
    });
    assert!(replay.result.is_none());
    let mut collector = crate::replay::training::TrainingCollector::new(&replay).unwrap();
    let mut game = ReplayExecutor::execute_with_observer(&replay, &mut collector).unwrap();
    game.state.tribes.get_mut(&2).unwrap().killed_turn = 3;

    assert_eq!(
        derive_result(&game).unwrap().reason.as_deref(),
        Some("derived:elimination")
    );
    let samples = collector
        .finish(&game, None, std::path::Path::new("derived.replay.json"))
        .unwrap();
    assert_eq!(samples.len(), 2);
    assert_eq!(samples[0].value(), 1.0);
    assert_eq!(samples[1].value(), -1.0);

    let dir = std::env::temp_dir().join(format!("polyfish-derived-test-{}", std::process::id()));
    let paths = crate::replay::training::write_training_files(&samples, &dir, 10).unwrap();
    let manifest_path = format!("{}.manifest.json", paths[0].display());
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
    assert_eq!(
        manifest["derivedResultSourceFiles"],
        serde_json::json!(["derived.replay.json"])
    );
    std::fs::remove_file(&manifest_path).unwrap();
    std::fs::remove_file(&paths[0]).unwrap();
    std::fs::remove_dir(&dir).unwrap();
}

#[test]
fn finish_still_refuses_a_result_less_unfinished_replay() {
    let replay = replay_with(vec![ReplayCommand::EndTurn]);
    let mut collector = crate::replay::training::TrainingCollector::new(&replay).unwrap();
    let game = ReplayExecutor::execute_with_observer(&replay, &mut collector).unwrap();
    let error = match collector.finish(&game, None, std::path::Path::new("truncated.replay.json")) {
        Ok(samples) => panic!("expected a refusal, got {} samples", samples.len()),
        Err(error) => error.to_string(),
    };
    assert!(
        error.contains("cannot derive a result"),
        "unexpected error: {error}"
    );
}

#[test]
fn reports_the_game_version_and_gates_one_outside_the_supported_range() {
    let replay = replay_with(vec![ReplayCommand::EndTurn]);
    let eligibility = validate_training_eligibility(&replay).unwrap();
    assert_eq!(eligibility.game_version, 115);
    assert_eq!(eligibility.version_support, VersionSupport::Supported);

    let mut ancient = replay_with(vec![ReplayCommand::EndTurn]);
    ancient.initial_state.settings.version = MIN_SUPPORTED_GAME_VERSION - 1;
    let error = validate_training_eligibility(&ancient)
        .unwrap_err()
        .to_string();
    assert!(
        error.contains(&format!("game version {}", MIN_SUPPORTED_GAME_VERSION - 1))
            && error.contains(&format!(
                "{MIN_SUPPORTED_GAME_VERSION}..={MAX_SUPPORTED_GAME_VERSION}"
            )),
        "unexpected error: {error}"
    );

    let mut future = replay_with(vec![ReplayCommand::EndTurn]);
    future.initial_state.settings.version = MAX_SUPPORTED_GAME_VERSION + 4;
    assert!(validate_training_eligibility(&future).is_err());
    assert!(crate::replay::training::TrainingCollector::new(&future).is_err());

    let eligibility = validate_training_eligibility_with(&future, true).unwrap();
    assert_eq!(eligibility.version_support, VersionSupport::TooNew);
    assert!(crate::replay::training::TrainingCollector::new_with(&future, true).is_ok());
}

#[test]
fn illegal_command_error_carries_the_game_version() {
    let replay = replay_with(vec![ReplayCommand::Step {
        source: 0,
        target: 0,
    }]);
    assert!(matches!(
        ReplayExecutor::execute(&replay).unwrap_err(),
        ReplayError::IllegalCommand {
            game_version: 115,
            ..
        }
    ));
}

/// Reads the checkpoint off the engine's own post-load state, which is what
/// `before_move` compares against.
fn checkpoint_for(replay: &Replay, player_id: crate::states::PlayerId) -> SourceCheckpoint {
    let game = ReplayExecutor::initialize(replay).unwrap();
    let tribe = &game.state.tribes[&player_id];
    SourceCheckpoint {
        turn_number: replay.turns[0].turn_number,
        player_id,
        score: tribe.score,
        stars: tribe.stars,
        unit_count: tribe.units.len(),
    }
}

#[test]
fn verifier_is_inert_without_diagnostics_and_catches_a_diverged_star_count() {
    let replay = replay_with(vec![ReplayCommand::EndTurn]);
    let mut inert = DivergenceVerifier::new(replay.metadata.end_turn_checkpoints().unwrap());
    assert!(inert.is_inert());
    ReplayExecutor::execute_with_observer(&replay, &mut inert).unwrap();

    let mut replay = replay_with(vec![ReplayCommand::EndTurn]);
    let checkpoint = checkpoint_for(&replay, 1);
    replay.metadata.source_diagnostics = Some(serde_json::json!({
        "endTurnCheckpoints": [checkpoint],
        "someUnknownSourceKey": 7,
    }));
    let mut verifier = DivergenceVerifier::new(replay.metadata.end_turn_checkpoints().unwrap());
    assert!(!verifier.is_inert());
    ReplayExecutor::execute_with_observer(&replay, &mut verifier).unwrap();
    assert!(verifier.score_notes().is_empty());

    let mut drifted = checkpoint;
    drifted.score += 5;
    replay.metadata.source_diagnostics =
        Some(serde_json::json!({ "endTurnCheckpoints": [drifted] }));
    let mut verifier = DivergenceVerifier::new(replay.metadata.end_turn_checkpoints().unwrap());
    ReplayExecutor::execute_with_observer(&replay, &mut verifier).unwrap();
    assert_eq!(verifier.score_notes().len(), 1);

    let mut drifted = checkpoint;
    drifted.stars += 1;
    replay.metadata.source_diagnostics =
        Some(serde_json::json!({ "endTurnCheckpoints": [drifted] }));
    let mut verifier = DivergenceVerifier::new(replay.metadata.end_turn_checkpoints().unwrap());
    let error = ReplayExecutor::execute_with_observer(&replay, &mut verifier).unwrap_err();
    assert!(
        matches!(
            error,
            ReplayError::SourceDivergence {
                field: "stars",
                player_id: 1,
                ..
            }
        ),
        "unexpected error: {error}"
    );
    assert!(error.to_string().contains("source recorded"));
}
