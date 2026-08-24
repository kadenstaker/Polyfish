use crate::functions::is_game_over;
use crate::game::Game;
use crate::states::PlayerId;
use std::collections::BTreeMap;

use super::{ReplayError, ReplayResult};

/// Synthesize the result of a capture that carries none, from its executed
/// final state. Survival decides first (training plays Domination, and most
/// score outlives its owner); score breaks a genuine turn-limit terminal, and
/// only among the living. Mirrors `ai::mcts_common::compute_terminal_outcome`.
///
/// Scores are the incremental `tribe.score` counter, which is known to drift
/// from the canonical recompute (#40) - the same currency the in-tree rule
/// uses. A score tie among the living collapses to a draw, which hands an
/// already-dead third tribe 0.0 instead of -1.0; the `reason` string names
/// that case so a consumer can tell it apart.
pub fn derive_result(game: &Game) -> Result<ReplayResult, ReplayError> {
    let mut alive: Vec<PlayerId> = game
        .state
        .tribes
        .iter()
        .filter(|(_, tribe)| tribe.killed_turn <= 0 && tribe.resigned_turn <= 0)
        .map(|(&id, _)| id)
        .collect();
    alive.sort_unstable();

    if !is_game_over(&game.state) {
        return Err(ReplayError::Training(format!(
            "cannot derive a result: the replay ends at turn {} of {} with {} tribes still alive; \
             supply replay.result or capture the finished game",
            game.state.settings.turn,
            game.state.settings.max_turns,
            alive.len(),
        )));
    }

    let scores: BTreeMap<PlayerId, i32> = game
        .state
        .tribes
        .iter()
        .map(|(&id, tribe)| (id, tribe.score))
        .collect();
    let score_of = |id: &PlayerId| scores.get(id).copied().unwrap_or(0);

    let (winner_player_id, reason) = match alive.as_slice() {
        [] => (None, "derived:mutualElimination"),
        [sole] => (Some(*sole), "derived:elimination"),
        _ => {
            let best = alive.iter().map(score_of).max().unwrap_or(0);
            let leaders: Vec<PlayerId> = alive
                .iter()
                .copied()
                .filter(|id| score_of(id) == best)
                .collect();
            match leaders.as_slice() {
                [sole] => (Some(*sole), "derived:scoreAtLimit"),
                _ => (None, "derived:scoreTieAtLimit"),
            }
        }
    };

    Ok(ReplayResult {
        winner_player_id,
        draw: winner_player_id.is_none(),
        scores,
        reason: Some(reason.into()),
    })
}
