//! Record-then-replay round trip over the whole command matrix.
//!
//! Every executed command in the replay unit suite was `EndTurn` or a
//! deliberately illegal `Step`, so ten of `matches_move`'s eleven arms had
//! never run against a real position. Here a network-free driver plays a real
//! engine game, `ReplayRecorder` records it, the replay goes through
//! `save_replay`/`load_replay`, and `ReplayExecutor` re-executes it. The
//! assertion is per command, not just on the final state: the executor must
//! resolve the *same* move the driver played, which catches `matches_move`
//! selecting a different legal move that merely satisfies its predicate.

use polyfish::game::Game;
use polyfish::mapgen::{MapGenSettings, generate};
use polyfish::moves::Move;
use polyfish::replay::{
    Replay, ReplayCommand, ReplayError, ReplayExecutor, ReplayMetadata, ReplayMoveContext,
    ReplayObserver, ReplayPlayerMetadata, ReplayRecorder, ReplaySource, load_replay, save_replay,
};
use polyfish::states::GameState;
use polyfish::types::{MapSize, MapType, MoveType, TribeType};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::BTreeMap;

const MAX_TURNS: i32 = 20;
const MOVE_CAP: usize = 300;

/// Command variants the default-gate run must reach. `Upgrade` needs a Raft
/// plus a navy tech and `Resign` is never emitted by `generate_legal_moves`,
/// so neither can be required of a bounded game.
const REQUIRED_COMMANDS: [&str; 10] = [
    "Step", "Attack", "Capture", "Build", "Research", "Summon", "Ability", "Reward", "Harvest",
    "EndTurn",
];

fn command_name(command: &ReplayCommand) -> &'static str {
    match command {
        ReplayCommand::Step { .. } => "Step",
        ReplayCommand::Attack { .. } => "Attack",
        ReplayCommand::Capture { .. } => "Capture",
        ReplayCommand::Build { .. } => "Build",
        ReplayCommand::Research { .. } => "Research",
        ReplayCommand::Summon { .. } => "Summon",
        ReplayCommand::Upgrade { .. } => "Upgrade",
        ReplayCommand::Ability { .. } => "Ability",
        ReplayCommand::Reward { .. } => "Reward",
        ReplayCommand::Harvest { .. } => "Harvest",
        ReplayCommand::EndTurn => "EndTurn",
        ReplayCommand::Resign => "Resign",
    }
}

/// Cheap O(1) coverage key. Abilities and structures are split by sub-type so
/// the driver spreads across them instead of hammering the first one it finds.
fn bucket_key(m: &dyn Move) -> (i32, i32) {
    let move_type = m.move_type();
    let sub = match move_type {
        MoveType::Ability => m.ability_type().map(|a| a as i32).unwrap_or(-1),
        MoveType::Build => m.structure_type().map(|s| s as i32).unwrap_or(-1),
        MoveType::Reward => m.reward_type().map(|r| r as i32).unwrap_or(-1),
        _ => 0,
    };
    (move_type as i32, sub)
}

/// Least-visited bucket wins, ties broken by the seeded RNG. `EndTurn` is only
/// taken when nothing else is legal, which is what drives turns to completion.
fn choose_move(
    moves: &[Box<dyn Move>],
    seen: &BTreeMap<(i32, i32), usize>,
    rng: &mut StdRng,
) -> usize {
    let keys: Vec<(i32, i32)> = moves.iter().map(|m| bucket_key(m.as_ref())).collect();
    let end_turn = MoveType::EndTurn as i32;
    let has_other = keys.iter().any(|k| k.0 != end_turn);
    let mut best = 0;
    let mut best_count = usize::MAX;
    let mut ties = 0;
    for (index, key) in keys.iter().enumerate() {
        if has_other && key.0 == end_turn {
            continue;
        }
        let count = *seen.get(key).unwrap_or(&0);
        if count < best_count {
            best_count = count;
            best = index;
            ties = 1;
        } else if count == best_count {
            ties += 1;
            if rng.random_range(0..ties) == 0 {
                best = index;
            }
        }
    }
    best
}

fn metadata_for(state: &GameState, game_id: &str) -> ReplayMetadata {
    ReplayMetadata {
        source: ReplaySource::Other,
        game_id: Some(game_id.to_string()),
        created_at: None,
        map_width: state.settings.size as usize,
        map_height: state.settings.size as usize,
        max_turns: state.settings.max_turns,
        game_mode: state.settings.mode,
        players: state
            .tribes
            .iter()
            .map(|(&player_id, tribe)| ReplayPlayerMetadata {
                player_id,
                tribe: tribe.tribe_type,
                name: None,
            })
            .collect(),
        source_diagnostics: None,
    }
}

struct Recorded {
    replay: Replay,
    final_state: GameState,
    /// `format!("{:?}", played_move)` per recorded command, in global order.
    expected_moves: Vec<String>,
    coverage: BTreeMap<&'static str, usize>,
    /// Commands whose move was generated more than once (see `Stats`).
    duplicate_matches: usize,
}

fn record_game(seed: i64, map_type: MapType, tribes: Vec<TribeType>, move_cap: usize) -> Recorded {
    let mut game = Game::new();
    game.state = generate(MapGenSettings {
        size: MapSize::Tiny,
        map_type,
        tribes,
        seed,
        ..Default::default()
    });
    game.post_load();
    game.state.settings.max_turns = MAX_TURNS;

    let initial_state = game.state.clone();
    let metadata = metadata_for(&initial_state, &format!("round-trip-{seed}"));
    let mut recorder = ReplayRecorder::new(initial_state, metadata);

    let mut rng = StdRng::seed_from_u64(seed as u64);
    let mut seen: BTreeMap<(i32, i32), usize> = BTreeMap::new();
    let mut expected_moves = Vec::new();
    let mut coverage: BTreeMap<&'static str, usize> = BTreeMap::new();
    let mut duplicate_matches = 0;

    for _ in 0..move_cap {
        if game.state.settings._game_over {
            break;
        }
        let moves = game.legal_moves();
        if moves.is_empty() {
            break;
        }
        let index = choose_move(&moves, &seen, &mut rng);
        *seen.entry(bucket_key(moves[index].as_ref())).or_insert(0) += 1;

        let played = moves[index].as_ref();
        let turn = game.state.settings.turn;
        let player_id = game.state.settings.current_player_turn_id;
        let command = ReplayCommand::from_move(played)
            .unwrap_or_else(|e| panic!("seed {seed}: {played:?} is not recordable: {e}"));
        *coverage.entry(command_name(&command)).or_insert(0) += 1;
        let played_debug = format!("{played:?}");
        if moves
            .iter()
            .filter(|other| format!("{:?}", other.as_ref()) == played_debug)
            .count()
            > 1
        {
            duplicate_matches += 1;
        }
        expected_moves.push(played_debug);
        recorder
            .record_command(turn, player_id, command)
            .unwrap_or_else(|e| panic!("seed {seed}: recorder rejected turn {turn}: {e}"));

        assert!(
            game.play_move(played).is_some(),
            "seed {seed}: engine refused its own legal move {played:?} on turn {turn}"
        );
    }

    let final_state = game.state.clone();
    Recorded {
        replay: recorder.finish(None),
        final_state,
        expected_moves,
        coverage,
        duplicate_matches,
    }
}

/// Fails the replay at the first command whose resolved move differs from the
/// one the driver actually played.
struct IdentityObserver {
    expected: Vec<String>,
    checked: usize,
}

impl ReplayObserver for IdentityObserver {
    fn before_move(
        &mut self,
        _game: &Game,
        context: &ReplayMoveContext,
        _legal_moves: &[Box<dyn Move>],
        selected_move: &dyn Move,
        command: &ReplayCommand,
    ) -> Result<(), ReplayError> {
        let expected = &self.expected[context.global_command_index];
        let actual = format!("{selected_move:?}");
        assert_eq!(
            *expected, actual,
            "{context}: command {command:?} resolved to a different move than was recorded"
        );
        self.checked += 1;
        Ok(())
    }
}

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
                    diff_paths(pa, pb, format!("{path}.{k}"), out);
                }
            }
        }
        (serde_json::Value::Array(va), serde_json::Value::Array(vb)) => {
            if va.len() != vb.len() {
                out.push(format!("{path}: array len {} vs {}", va.len(), vb.len()));
                return;
            }
            for (i, (pa, pb)) in va.iter().zip(vb.iter()).enumerate() {
                if pa != pb {
                    diff_paths(pa, pb, format!("{path}[{i}]"), out);
                }
            }
        }
        _ => out.push(format!("{path}: {a} vs {b}")),
    }
}

/// Fields backed by a `HashSet`, so their serialized array order carries no
/// information. Every other array stays ordered, since unit/city/history order
/// is real state a replay must reproduce.
const UNORDERED_ARRAY_FIELDS: [&str; 4] = [
    "explorers",
    "effects",
    "builtUniqueImprovements",
    "knownPlayers",
];

fn canonicalize(value: &mut serde_json::Value, key: &str) {
    match value {
        serde_json::Value::Object(map) => {
            for (child_key, child) in map.iter_mut() {
                canonicalize(child, child_key);
            }
        }
        serde_json::Value::Array(items) => {
            for item in items.iter_mut() {
                canonicalize(item, "");
            }
            if UNORDERED_ARRAY_FIELDS.contains(&key) {
                items.sort_by_key(|item| item.to_string());
            }
        }
        _ => {}
    }
}

/// `execute_command` clears `_messages` after every command and `play_move`
/// does not, so the field is normalized away before comparing.
fn comparable(state: &GameState) -> serde_json::Value {
    let mut state = state.clone();
    state._messages.clear();
    let mut value = serde_json::to_value(&state).unwrap();
    canonicalize(&mut value, "");
    value
}

/// Aggregated evidence one round trip produced.
#[derive(Default)]
struct Stats {
    coverage: BTreeMap<&'static str, usize>,
    /// Commands whose move `generate_legal_moves` emitted more than once, so
    /// the executor had to collapse indistinguishable matches instead of
    /// refusing as ambiguous. Non-zero here is the engine defect, not a
    /// replay-layer one: a tile inside two of a player's city territories is
    /// walked once per city.
    duplicate_matches: usize,
}

fn round_trip(seed: i64, map_type: MapType, tribes: Vec<TribeType>) -> Stats {
    let recorded = record_game(seed, map_type, tribes, MOVE_CAP);
    let commands = recorded.replay.command_count();
    assert_eq!(commands, recorded.expected_moves.len());
    assert!(
        commands > 50,
        "seed {seed}: only {commands} commands recorded, too few to cover the matrix"
    );

    let path = std::env::temp_dir().join(format!(
        "polyfish-round-trip-{}-{seed}.replay.json",
        std::process::id()
    ));
    save_replay(&recorded.replay, &path)
        .unwrap_or_else(|e| panic!("seed {seed}: recorded replay failed validation: {e}"));
    let loaded = load_replay(&path)
        .unwrap_or_else(|e| panic!("seed {seed}: saved replay failed to load back: {e}"));
    let _ = std::fs::remove_file(&path);

    let mut observer = IdentityObserver {
        expected: recorded.expected_moves,
        checked: 0,
    };
    let replayed = ReplayExecutor::execute_with_observer(&loaded, &mut observer)
        .unwrap_or_else(|e| panic!("seed {seed}: re-execution failed: {e}"));
    assert_eq!(observer.checked, commands);

    let live = comparable(&recorded.final_state);
    let again = comparable(&replayed.state);
    if live != again {
        let mut diffs = Vec::new();
        diff_paths(&live, &again, String::new(), &mut diffs);
        panic!(
            "seed {seed}: replayed state diverged after {commands} commands\nDiffs:\n  {}",
            diffs.join("\n  ")
        );
    }
    Stats {
        coverage: recorded.coverage,
        duplicate_matches: recorded.duplicate_matches,
    }
}

fn merge(into: &mut Stats, from: Stats) {
    for (name, count) in from.coverage {
        *into.coverage.entry(name).or_insert(0) += count;
    }
    into.duplicate_matches += from.duplicate_matches;
}

fn assert_matrix_covered(stats: &Stats) {
    let coverage = &stats.coverage;
    eprintln!(
        "replay command coverage: {coverage:?} (duplicate-move commands: {})",
        stats.duplicate_matches
    );
    let missing: Vec<&str> = REQUIRED_COMMANDS
        .iter()
        .copied()
        .filter(|name| !coverage.contains_key(name))
        .collect();
    assert!(
        missing.is_empty(),
        "round trip never exercised {missing:?}; it has degraded into a narrower test. Covered: {coverage:?}"
    );
}

#[test]
fn record_then_replay_round_trips() {
    let mut stats = Stats::default();
    for (seed, map_type, tribes) in [
        (
            20260823i64,
            MapType::Drylands,
            [TribeType::Imperius, TribeType::Bardur],
        ),
        (7, MapType::Lakes, [TribeType::Kickoo, TribeType::Vengir]),
        (5, MapType::Lakes, [TribeType::Imperius, TribeType::Bardur]),
    ] {
        merge(&mut stats, round_trip(seed, map_type, tribes.to_vec()));
    }
    assert_matrix_covered(&stats);
    assert!(
        stats.duplicate_matches > 0,
        "no command hit a duplicated legal move, so the executor's collapse of \
         indistinguishable matches went untested. If movegen no longer emits \
         duplicates, drop this assertion; otherwise pick a seed reaching a tile \
         shared by two city territories"
    );
}

#[test]
#[ignore = "sweep: 20 seeds across four map types, ~10s in debug"]
fn record_then_replay_round_trips_many_seeds() {
    let mut stats = Stats::default();
    for seed in 0..20i64 {
        let map_type = match seed % 4 {
            0 => MapType::Drylands,
            1 => MapType::Lakes,
            2 => MapType::Continents,
            _ => MapType::Archipelago,
        };
        merge(
            &mut stats,
            round_trip(seed, map_type, vec![TribeType::Imperius, TribeType::Bardur]),
        );
    }
    assert_matrix_covered(&stats);
}
