use std::collections::HashMap;

use crate::functions::calculate_detailed_tribe_score;
use crate::game::Game;
use crate::moves::Move;
use crate::states::PlayerId;

use super::{ReplayCommand, ReplayError, ReplayMoveContext, ReplayObserver, SourceCheckpoint};

/// Cross-checks engine state against the source game's EndTurn checkpoints, so
/// a rules divergence that keeps every command legal still fails the import.
/// Inert on a replay that carries no checkpoints.
pub struct DivergenceVerifier {
    checkpoints: HashMap<(i32, PlayerId), SourceCheckpoint>,
    score_notes: Vec<String>,
}

impl DivergenceVerifier {
    pub fn new(checkpoints: Vec<SourceCheckpoint>) -> Self {
        Self {
            checkpoints: checkpoints
                .into_iter()
                .map(|checkpoint| ((checkpoint.turn_number, checkpoint.player_id), checkpoint))
                .collect(),
            score_notes: Vec::new(),
        }
    }

    pub fn is_inert(&self) -> bool {
        self.checkpoints.is_empty()
    }

    /// Score mismatches, reported rather than failed: `tribe.score` is an
    /// incremental counter that provably drifts from a recomputed score.
    pub fn score_notes(&self) -> &[String] {
        &self.score_notes
    }
}

impl ReplayObserver for DivergenceVerifier {
    /// Compares before the command, not after: `Game::end_turn` advances the
    /// active player and pays income in the same call.
    fn before_move(
        &mut self,
        game: &Game,
        context: &ReplayMoveContext,
        _legal_moves: &[Box<dyn Move>],
        _selected_move: &dyn Move,
        command: &ReplayCommand,
    ) -> Result<(), ReplayError> {
        if !matches!(command, ReplayCommand::EndTurn) {
            return Ok(());
        }
        let Some(checkpoint) = self
            .checkpoints
            .get(&(context.turn_number, context.player_id))
        else {
            return Ok(());
        };
        let Some(tribe) = game.state.tribes.get(&context.player_id) else {
            return Ok(());
        };
        for (field, expected, actual) in [
            ("stars", checkpoint.stars as i64, tribe.stars as i64),
            (
                "unitCount",
                checkpoint.unit_count as i64,
                tribe.units.len() as i64,
            ),
        ] {
            if expected != actual {
                return Err(ReplayError::SourceDivergence {
                    context: context.clone(),
                    player_id: context.player_id,
                    field,
                    expected,
                    actual,
                });
            }
        }
        if checkpoint.score != tribe.score {
            let recomputed = calculate_detailed_tribe_score(&game.state, context.player_id);
            self.score_notes.push(format!(
                "{context}: player {} score is {} (recomputed {recomputed}), source recorded {}",
                context.player_id, tribe.score, checkpoint.score
            ));
        }
        Ok(())
    }
}

/// Runs two observers over one execution.
pub struct PairObserver<A, B>(pub A, pub B);

impl<A: ReplayObserver, B: ReplayObserver> ReplayObserver for PairObserver<A, B> {
    fn before_move(
        &mut self,
        game: &Game,
        context: &ReplayMoveContext,
        legal_moves: &[Box<dyn Move>],
        selected_move: &dyn Move,
        command: &ReplayCommand,
    ) -> Result<(), ReplayError> {
        self.0
            .before_move(game, context, legal_moves, selected_move, command)?;
        self.1
            .before_move(game, context, legal_moves, selected_move, command)
    }

    fn after_move(
        &mut self,
        game: &Game,
        context: &ReplayMoveContext,
        command: &ReplayCommand,
    ) -> Result<(), ReplayError> {
        self.0.after_move(game, context, command)?;
        self.1.after_move(game, context, command)
    }
}
