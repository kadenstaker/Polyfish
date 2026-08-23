//! Fuzz probe for MCTS state-restoration bugs: walks random game paths and
//! verifies that (a) undoing a simulated move restores the exact prior state,
//! and (b) re-simulating the same move reproduces the exact same result.
//! Either failing explains "Insufficient stars" errors during tree descent.
//!
//! Four arms, because the failure modes differ: legacy vs adversarial in-tree
//! EndTurn, and descending on the real state vs on a fogged `clone_for_mcts`
//! view (where the in-tree opponent plays against cleared explorers).

use polyfish::game::Game;
use polyfish::state_fingerprint::undo_fingerprint;
use polyfish::types::{MapSize, MapType, TribeType};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

mod common;
use common::AdversarialModeGuard;

const TRIBES: [TribeType; 12] = [
    TribeType::Imperius,
    TribeType::Bardur,
    TribeType::Oumaji,
    TribeType::XinXi,
    TribeType::Vengir,
    TribeType::Hoodrick,
    TribeType::Luxidoor,
    TribeType::Zebasi,
    TribeType::AiMo,
    TribeType::Quetzali,
    TribeType::Yadakk,
    TribeType::Kickoo,
];

/// JSON plus the order/shadow-state fingerprint the JSON cannot see.
type Snapshot = (serde_json::Value, Vec<String>);

fn snap(game: &Game) -> Snapshot {
    (
        serde_json::to_value(&game.state).unwrap(),
        undo_fingerprint(&game.state),
    )
}

/// Return the paths of the first few differing fields between two JSON values.
fn diff_paths(a: &serde_json::Value, b: &serde_json::Value, path: String, out: &mut Vec<String>) {
    if out.len() >= 5 {
        return;
    }
    match (a, b) {
        (serde_json::Value::Object(ma), serde_json::Value::Object(mb)) => {
            for k in ma.keys().chain(mb.keys()) {
                let pa = ma.get(k).unwrap_or(&serde_json::Value::Null);
                let pb = mb.get(k).unwrap_or(&serde_json::Value::Null);
                if pa != pb {
                    diff_paths(pa, pb, format!("{}.{}", path, k), out);
                }
            }
        }
        (serde_json::Value::Array(va), serde_json::Value::Array(vb)) => {
            if va.len() != vb.len() {
                out.push(format!("{}: array len {} vs {}", path, va.len(), vb.len()));
                return;
            }
            for (i, (pa, pb)) in va.iter().zip(vb.iter()).enumerate() {
                if pa != pb {
                    diff_paths(pa, pb, format!("{}[{}]", path, i), out);
                }
            }
        }
        _ => out.push(format!("{}: {} vs {}", path, a, b)),
    }
}

fn assert_same(before: &Snapshot, after: &Snapshot, ctx: &str) {
    if before == after {
        return;
    }
    let mut diffs = Vec::new();
    diff_paths(&before.0, &after.0, String::new(), &mut diffs);
    // Positional, so a fingerprint line that appears or vanishes (a
    // `_sim_explored` entry whose undo was dropped) is reported, not zipped away.
    let missing = "<missing>";
    for i in 0..before.1.len().max(after.1.len()) {
        let want = before.1.get(i).map(String::as_str).unwrap_or(missing);
        let got = after.1.get(i).map(String::as_str).unwrap_or(missing);
        if want != got {
            diffs.push(format!("fingerprint: {} vs {}", want, got));
        }
    }
    panic!("STATE MISMATCH {}\nDiffs:\n  {}", ctx, diffs.join("\n  "));
}

/// Recursively simulate a random move, descend deeper, then unwind and verify
/// this level's undo restores the exact prior state, and that replaying the
/// move is deterministic. Inner levels verify themselves first, so a mismatch
/// always implicates this level's move.
fn descent_check(game: &mut Game, rng: &mut StdRng, depth: u32) {
    if depth == 0 || game.state.settings._game_over {
        return;
    }
    let moves = game.legal_moves();
    if moves.is_empty() {
        return;
    }
    let m = &moves[rng.random_range(0..moves.len())];
    let desc = m.describe(&game.state);
    let turn = game.state.settings.turn;

    let before = snap(game);
    let Some(undo) = game.simulate_move(m.as_ref()) else {
        panic!("LEGAL MOVE REFUSED in simulation: {} (turn {})", desc, turn);
    };
    let after_first = snap(game);

    descent_check(game, rng, depth - 1);

    undo(&mut game.state);
    assert_same(
        &before,
        &snap(game),
        &format!("after undoing [{}] (turn {})", desc, turn),
    );

    // Replay determinism: same state + same move must give the same result.
    let Some(undo2) = game.simulate_move(m.as_ref()) else {
        panic!(
            "REPLAY REFUSED: {} executed once but not twice (turn {})",
            desc, turn
        );
    };
    assert_same(
        &after_first,
        &snap(game),
        &format!("replaying [{}] (turn {})", desc, turn),
    );
    undo2(&mut game.state);
    assert_same(
        &before,
        &snap(game),
        &format!("after second undo of [{}] (turn {})", desc, turn),
    );
}

/// One arm of the probe. Holds the adversarial guard for its whole body: the
/// switch is process-wide and `simulate_move` reads it on every in-tree
/// EndTurn, so a concurrent arm would flip the mode mid-descent.
fn run_arm(seeds: std::ops::Range<i64>, steps: usize, depth: u32, adversarial: bool, fogged: bool) {
    let _guard = AdversarialModeGuard::set(adversarial);

    for game_seed in seeds {
        let mut rng = StdRng::seed_from_u64(game_seed as u64);
        let t1 = TRIBES[rng.random_range(0..TRIBES.len())];
        let mut t2 = TRIBES[rng.random_range(0..TRIBES.len())];
        while t2 == t1 {
            t2 = TRIBES[rng.random_range(0..TRIBES.len())];
        }

        let gen_settings = polyfish::mapgen::MapGenSettings {
            size: MapSize::Tiny,
            map_type: MapType::Drylands,
            tribes: vec![t1, t2],
            seed: game_seed,
            symmetric: true,
            ..Default::default()
        };
        let mut game = Game::new();
        game.state = polyfish::mapgen::generate(gen_settings);
        game.state.settings.max_turns = 12;
        game.post_load();

        // Random real playout; at every step fuzz simulated descents first.
        for _step in 0..steps {
            if game.state.settings._game_over {
                break;
            }
            if fogged {
                let mut view = game.clone_for_mcts(game.current_player_id());
                descent_check(&mut view, &mut rng, depth);
            } else {
                descent_check(&mut game, &mut rng, depth);
            }

            let moves = game.legal_moves();
            if moves.is_empty() {
                break;
            }
            let m = &moves[rng.random_range(0..moves.len())];
            if game.play_move(m.as_ref()).is_none() {
                break;
            }
        }
        eprintln!(
            "seed {} ok ({:?} vs {:?}, adversarial={} fogged={}, turn {})",
            game_seed, t1, t2, adversarial, fogged, game.state.settings.turn
        );
    }
}

const ARMS: [(bool, bool); 4] = [(false, false), (true, false), (false, true), (true, true)];

/// Cheap pass over all four arms so a broken arm reds the PR gate instead of
/// waiting for the nightly.
#[test]
fn undo_arms_smoke() {
    use std::sync::atomic::Ordering;
    let before = polyfish::game::SIM_END_TURN_EDGES.load(Ordering::Relaxed);
    for (adversarial, fogged) in ARMS {
        run_arm(0..2, 6, 3, adversarial, fogged);
    }
    // Seeds are pinned, so this is deterministic. If it trips, the smoke stopped
    // reaching the in-tree turn boundary the adversarial arm exists to cover.
    assert!(
        polyfish::game::SIM_END_TURN_EDGES.load(Ordering::Relaxed) > before,
        "no in-tree EndTurn simulated: the arms are not exercising the handover"
    );
}

// Heavy fuzz probe (~6s release per arm, minutes in debug) — the whole file is
// run nightly by .github/workflows/undo_fuzz.yml, or on demand:
//   cargo test --release --test undo_integrity -- --ignored
#[test]
#[ignore]
fn undo_integrity_legacy() {
    run_arm(0..30, 60, 5, false, false);
}

#[test]
#[ignore]
fn undo_integrity_adversarial() {
    run_arm(0..20, 60, 5, true, false);
}

#[test]
#[ignore]
fn undo_integrity_fogged_clone() {
    run_arm(0..20, 60, 5, false, true);
}

#[test]
#[ignore]
fn undo_integrity_adversarial_fogged_clone() {
    run_arm(0..20, 60, 5, true, true);
}
