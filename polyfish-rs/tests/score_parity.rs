//! `tribe.score` is the reward and value-label currency, and it is maintained
//! incrementally while `score::breakdown` recomputes it from the state. When
//! the two disagree, every TD label built from the incremental one is wrong in
//! a way no training metric shows (#40).
//!
//! This is the cheap gate arm: a handful of random games, parity asserted after
//! every move so the failing move names itself. `examples/score_parity_fuzz`
//! is the same check at fuzz depth, with a per-move-kind ranking.

use polyfish::game::Game;
use polyfish::score::{ScoreBreakdown, breakdown};
use polyfish::states::{GameState, PlayerId};
use polyfish::types::{MapSize, MapType, ModeType, TribeType};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;

const TRIBES: [TribeType; 6] = [
    TribeType::Imperius,
    TribeType::Bardur,
    TribeType::Aquarion,
    TribeType::Elyrion,
    TribeType::Cymanti,
    TribeType::Luxidoor,
];

fn snapshot(state: &GameState) -> HashMap<PlayerId, (i32, ScoreBreakdown)> {
    state
        .tribes
        .iter()
        .filter(|(_, t)| t.killed_turn <= 0 && t.resigned_turn <= 0)
        .map(|(id, t)| (*id, (t.score, breakdown(state, *id))))
        .collect()
}

fn play_and_check(seed: u64, t1: TribeType, t2: TribeType, mode: ModeType, max_turns: i32) {
    let gen_settings = polyfish::mapgen::MapGenSettings {
        size: MapSize::Tiny,
        map_type: MapType::Drylands,
        tribes: vec![t1, t2],
        seed: seed as i64,
        symmetric: true,
        ..Default::default()
    };

    let mut game = Game::new();
    game.state = polyfish::mapgen::generate(gen_settings);
    game.state.settings.mode = mode;
    game.state.settings.max_turns = max_turns;
    game.post_load();

    let mut rng = StdRng::seed_from_u64(seed);
    let mut before = snapshot(&game.state);
    let mut moves = 0;

    while !game.state.settings._game_over && moves < 1500 {
        let legal = game.legal_moves();
        if legal.is_empty() {
            break;
        }
        let m = &legal[rng.random_range(0..legal.len())];
        let desc = m.describe(&game.state);
        assert!(
            game.play_move(m.as_ref()).is_some(),
            "seed {seed}: legal move rejected: {desc}"
        );
        moves += 1;

        let after = snapshot(&game.state);
        for (id, (incr_after, canon_after)) in &after {
            let (incr_before, canon_before) = before
                .get(id)
                .copied()
                .unwrap_or((*incr_after, *canon_after));
            let delta = (incr_after - canon_after.total()) - (incr_before - canon_before.total());
            assert_eq!(
                delta,
                0,
                "seed {seed} turn {} player {id}: `{desc}` moved incremental score {:+} \
                 while the recompute moved {:+} ({:?})",
                game.state.settings.turn,
                incr_after - incr_before,
                canon_after.total() - canon_before.total(),
                canon_after.diff(&canon_before),
            );
        }
        before = after;
    }
}

/// The cheap arm, so a regression reds the normal gate rather than waiting for
/// someone to run the fuzz. The seeds are picked to fail before the fix: 20 and
/// 44 build and destroy a temple, 32 embarks a unit, 50 captures a village and
/// a parked city; 1 and 2 are quiet games, for breadth.
#[test]
fn incremental_score_tracks_the_recompute() {
    for seed in [1u64, 2, 20, 32, 44, 50] {
        let t1 = TRIBES[(seed as usize) % TRIBES.len()];
        let t2 = TRIBES[(seed as usize + 3) % TRIBES.len()];
        let mode = if seed % 2 == 0 {
            ModeType::Domination
        } else {
            ModeType::Perfection
        };
        play_and_check(seed, t1, t2, mode, 30);
    }
}

/// Capture moves a city, its territory and everything standing on it between
/// two tribes; both sides of that transfer have to price it identically. Long
/// Domination games, so `#[ignore]`d - `examples/score_parity_fuzz` is the
/// wider version of the same check.
#[test]
#[ignore]
fn capture_transfers_the_whole_city_contribution() {
    for seed in 20..=25u64 {
        play_and_check(
            seed,
            TribeType::Imperius,
            TribeType::Bardur,
            ModeType::Domination,
            45,
        );
    }
}
