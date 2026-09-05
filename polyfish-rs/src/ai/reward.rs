//! Shared per-move reward definition for TD value labels (self_play) and
//! reward-aware MCTS backup (gumbel_mcts). One source of truth so a move's
//! score gain is normalized identically whether it's being summed into a
//! training label or backed up through the search tree.

use crate::states::GameState;
use crate::types::ModeType;

/// Turn-boundary discount for TD backup/labels: `γ^Δturn` applied only when
/// an edge crosses into a new game turn (within-turn moves are undiscounted).
/// ~10-turn effective horizon; gives a strict banked-now > pending-later
/// ordering independent of noise, unlike the old fixed forward-window MC
/// label it replaces. See notes.md, decision-trace section.
pub const GAMMA_TURN: f32 = 0.9;

/// Weight of the relative (vs opponent) component in every reward and value
/// label — the single zero-sum convention for both the TD body here and the
/// final-outcome tail in self_play. 1.0 = pure relative. The backup negates
/// across every player-turn boundary (mcts_common.rs), which is only valid
/// when v(mine) = -v(theirs); absolute own-progress is NOT antisymmetric (the
/// opponent's progress isn't my loss), so any abs share is corrupted through
/// EndTurn-crossing lines, worse as search deepens. The measured mirror-play
/// failure that once motivated an abs share (decision traces, Jul 7-8 2026:
/// relative swings net to ~0 in mirror play, so the label was empty) is
/// attacked in the DATA instead — greedy-anchor games (--anchor-frac) make
/// passivity actually lose. See notes.md, "Phase-1 training-signal fixes".
pub const REL_W: f32 = 1.0;

/// Reward normalization scales with the game's economy: a saturating swing
/// is ~15% of combined score, floored for the small opening turns.
pub const NORM_FRAC: f32 = 0.15;
pub const NORM_FLOOR: f32 = 600.0;

/// Normalization denominator for a reward measured from a state where `my`/
/// `opp` are the pre-transition scores.
pub fn score_norm(my: i32, opp: i32) -> f32 {
    (NORM_FRAC * (my + opp) as f32).max(NORM_FLOOR)
}

/// Normalized reward for a transition `(my_pre, opp_pre) -> (my_post,
/// opp_post)`, blending absolute (my own score gain) and relative (my gain
/// vs the opponent's) progress. Not clamped — callers accumulate/discount
/// multiple rewards before clamping the final label.
pub fn normalized_reward(my_pre: i32, opp_pre: i32, my_post: i32, opp_post: i32) -> f32 {
    let norm = score_norm(my_pre, opp_pre);
    let delta_abs = (my_post - my_pre) as f32 / norm;
    let delta_rel = ((my_post - opp_post) - (my_pre - opp_pre)) as f32 / norm;
    REL_W * delta_rel + (1.0 - REL_W) * delta_abs
}

/// Extra weight on army value in the Domination label (EXP_LABEL_004). The
/// scoreboard already prices a unit at cost x `UNIT_COST_SCORE` and never
/// discounts damage; audit A2b measured unit count ~8pp better than score at
/// predicting a Domination winner at every turn, and score *falling* late.
/// 8.0 puts a mid-game army at rough parity with the rest of the scoreboard.
/// 0.0 restores the raw-score label of the Sep 4 2026 baseline.
pub const ARMY_W: f32 = 8.0;

/// HP-weighted scoreboard value of `player`'s live army. A unit counts for
/// its `score::unit_score` scaled by remaining health, so taking damage
/// registers as a loss and healing as a gain.
pub fn army_value(state: &GameState, player: i32) -> i32 {
    let Some(tribe) = state.tribes.get(&player) else {
        return 0;
    };
    tribe
        .units
        .iter()
        .map(|u| {
            let max_hp = crate::functions::get_unit_max_health(u).max(1.0);
            let frac = (u.health / max_hp).clamp(0.0, 1.0);
            (crate::score::unit_score(u) as f32 * frac).round() as i32
        })
        .sum()
}

/// The quantity every reward and value label is built from: raw `score`,
/// plus `ARMY_W` x army value when the game is Domination. Other modes keep
/// the scoreboard, which is their win condition.
pub fn progress(state: &GameState, player: i32) -> i32 {
    let score = state.tribes.get(&player).map(|t| t.score).unwrap_or(0);
    if state.settings.mode == ModeType::Domination && ARMY_W > 0.0 {
        score + (ARMY_W * army_value(state, player) as f32).round() as i32
    } else {
        score
    }
}

/// `(my_progress, best_opponent_progress)` for `player` in `state`. Shared
/// snapshot helper for reward computation at both a tree edge (gumbel_mcts)
/// and a self-play history step.
pub fn progress_snapshot(state: &GameState, player: i32) -> (i32, i32) {
    let my = progress(state, player);
    let opp = state
        .tribes
        .keys()
        .filter(|id| **id != player)
        .map(|id| progress(state, *id))
        .max()
        .unwrap_or(0);
    (my, opp)
}

/// `(my_score, best_opponent_score)` for `player` in `state` — the raw
/// scoreboard, untouched by `ARMY_W`. Kept for callers that report score.
pub fn score_snapshot(state: &GameState, player: i32) -> (i32, i32) {
    let my = state.tribes.get(&player).map(|t| t.score).unwrap_or(0);
    let opp = state
        .tribes
        .iter()
        .filter(|(id, _)| **id != player)
        .map(|(_, t)| t.score)
        .max()
        .unwrap_or(0);
    (my, opp)
}

/// `(my_spt, best_opponent_spt)` for `player` in `state`. Feeds the
/// potential-based SPT shaping term in self_play value labels.
pub fn spt_snapshot(state: &GameState, player: i32) -> (i32, i32) {
    let my = state
        .tribes
        .get(&player)
        .map(|t| crate::functions::get_tribe_spt(state, t))
        .unwrap_or(0);
    let opp = state
        .tribes
        .iter()
        .filter(|(id, _)| **id != player)
        .map(|(_, t)| crate::functions::get_tribe_spt(state, t))
        .max()
        .unwrap_or(0);
    (my, opp)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::states::{TribeState, UnitState};
    use crate::types::UnitType;

    fn state_with(mode: ModeType) -> GameState {
        let mut st = GameState::default();
        st.settings.mode = mode;
        for id in [1, 2] {
            let mut t = TribeState::default();
            t.score = 1000;
            st.tribes.insert(id, t);
        }
        st
    }

    fn unit(owner: i32, health: f32) -> UnitState {
        let mut u = UnitState::default();
        u.owner = owner;
        u.unit_type = UnitType::Warrior;
        u.health = health;
        u
    }

    #[test]
    fn army_value_scales_with_health_and_ignores_others() {
        let mut st = state_with(ModeType::Domination);
        st.tribes[&1].units.push(unit(1, 10.0));
        st.tribes[&1].units.push(unit(1, 5.0));
        st.tribes[&2].units.push(unit(2, 10.0));
        let full = crate::score::unit_score(&unit(1, 10.0));
        assert_eq!(army_value(&st, 1), full + full / 2);
        assert_eq!(army_value(&st, 2), full);
    }

    #[test]
    fn progress_adds_army_only_in_domination() {
        let mut st = state_with(ModeType::Domination);
        st.tribes[&1].units.push(unit(1, 10.0));
        let full = crate::score::unit_score(&unit(1, 10.0));
        assert_eq!(
            progress(&st, 1),
            1000 + (ARMY_W * full as f32).round() as i32
        );
        assert_eq!(progress(&st, 2), 1000);
        st.settings.mode = ModeType::Perfection;
        assert_eq!(progress(&st, 1), 1000);
        assert_eq!(progress_snapshot(&st, 1), score_snapshot(&st, 1));
    }

    #[test]
    fn damage_is_a_negative_reward() {
        let mut st = state_with(ModeType::Domination);
        st.tribes[&1].units.push(unit(1, 10.0));
        let (my_pre, opp_pre) = progress_snapshot(&st, 1);
        st.tribes[&1].units[0].health = 4.0;
        let (my_post, opp_post) = progress_snapshot(&st, 1);
        assert!(normalized_reward(my_pre, opp_pre, my_post, opp_post) < 0.0);
        assert_eq!(score_snapshot(&st, 1), (1000, 1000));
    }
}
