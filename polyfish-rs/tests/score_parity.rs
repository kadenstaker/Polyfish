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
use polyfish::types::{MapSize, MapType, ModeType, MoveType, TribeType};
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

/// Every tribe, dead ones included: an elimination is the one move that
/// re-prices a whole tribe at once, and its score still feeds the others'
/// relative reward.
fn snapshot(state: &GameState) -> HashMap<PlayerId, (i32, ScoreBreakdown)> {
    state
        .tribes
        .iter()
        .map(|(id, t)| (*id, (t.score, breakdown(state, *id))))
        .collect()
}

struct Playout {
    seed: u64,
    tribes: [TribeType; 2],
    mode: ModeType,
    map_type: MapType,
    max_turns: i32,
    /// Take a capture or attack whenever one is legal; random play rarely conquers.
    conquest: bool,
}

fn play_and_check(p: Playout) {
    let Playout {
        seed,
        tribes: [t1, t2],
        mode,
        map_type,
        max_turns,
        conquest,
    } = p;
    let gen_settings = polyfish::mapgen::MapGenSettings {
        size: MapSize::Tiny,
        map_type,
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
        let pick = conquest
            .then(|| {
                legal
                    .iter()
                    .position(|m| matches!(m.move_type(), MoveType::Capture | MoveType::Attack))
            })
            .flatten()
            .unwrap_or_else(|| rng.random_range(0..legal.len()));
        let m = &legal[pick];
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
        play_and_check(Playout {
            seed,
            tribes: [t1, t2],
            mode,
            map_type: MapType::Drylands,
            max_turns: 30,
            conquest: false,
        });
    }
}

/// Capturing a tribe's last city re-prices everything it owned in one move.
/// The recompute must price the dead tribe's leftover territory the same way
/// the claim that took it did, so this walks eliminations on every map type
/// (the Archipelago case drifted by one tile before the fix).
#[test]
fn elimination_settles_every_owned_tile() {
    for (seed, map_type) in [
        (2u64, MapType::Archipelago),
        (5, MapType::Archipelago),
        (3, MapType::Continents),
        (4, MapType::Drylands),
    ] {
        play_and_check(Playout {
            seed,
            tribes: [TribeType::Luxidoor, TribeType::Elyrion],
            mode: ModeType::Domination,
            map_type,
            max_turns: 30,
            conquest: true,
        });
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
        play_and_check(Playout {
            seed,
            tribes: [TribeType::Imperius, TribeType::Bardur],
            mode: ModeType::Domination,
            map_type: MapType::Drylands,
            max_turns: 45,
            conquest: false,
        });
    }
}
