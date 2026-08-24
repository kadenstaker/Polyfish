//! Attributes score drift to the move that caused it (#40).
//!
//! The self-play probe reports one aggregate number at game end, which cannot
//! say which mutation is wrong. This plays random games and, after every real
//! move, compares each living tribe's incremental `score` against
//! `calculate_detailed_tribe_score`. A *change* in that difference is charged to
//! the move that just ran, so the report is a ranking of offending move types.
//!
//! Usage: cargo run --release --example score_parity_fuzz -- \
//!          [num_seeds] [start_seed] [--tribes=core|all] \
//!          [--mode=alternate|perfection|domination] [--max-turns=N] [--first]

use polyfish::game::Game;
use polyfish::score::{ScoreBreakdown, breakdown};
use polyfish::states::{GameState, PlayerId};
use polyfish::types::{ModeType, TribeType};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;

const CORE_TRIBES: [TribeType; 12] = [
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

const SPECIAL_TRIBES: [TribeType; 3] =
    [TribeType::Aquarion, TribeType::Elyrion, TribeType::Cymanti];

const USAGE: &str = "usage: score_parity_fuzz [num_seeds] [start_seed] \
[--tribes=core|all] [--mode=alternate|perfection|domination] [--max-turns=N] [--first]";

struct Args {
    num_seeds: u64,
    start_seed: u64,
    tribes_all: bool,
    mode: Option<ModeType>,
    max_turns: i32,
    stop_on_first: bool,
}

fn parse_args() -> Args {
    let mut positionals: Vec<String> = Vec::new();
    let mut tribes_all = false;
    let mut mode = None;
    let mut max_turns = 30;
    let mut stop_on_first = false;

    for arg in std::env::args().skip(1) {
        let Some(body) = arg.strip_prefix("--") else {
            positionals.push(arg);
            continue;
        };
        let (key, value) = body.split_once('=').unwrap_or((body, ""));
        match (key, value) {
            ("tribes", "core") => tribes_all = false,
            ("tribes", "all") => tribes_all = true,
            ("mode", "perfection") => mode = Some(ModeType::Perfection),
            ("mode", "domination") => mode = Some(ModeType::Domination),
            ("mode", "alternate") => mode = None,
            ("max-turns", v) => max_turns = v.parse().unwrap_or(30),
            ("first", "") => stop_on_first = true,
            _ => {
                eprintln!("unrecognised flag: {}\n{}", arg, USAGE);
                std::process::exit(2);
            }
        }
    }

    let positional = |i: usize, default: u64| -> u64 {
        match positionals.get(i) {
            Some(s) => s.parse().unwrap_or_else(|_| {
                eprintln!("not a number: {}\n{}", s, USAGE);
                std::process::exit(2);
            }),
            None => default,
        }
    };

    Args {
        num_seeds: positional(0, 20),
        start_seed: positional(1, 1),
        tribes_all,
        mode,
        max_turns,
        stop_on_first,
    }
}

/// Per-tribe incremental score and canonical breakdown, for tribes still alive.
fn snapshot(state: &GameState) -> HashMap<PlayerId, (i32, ScoreBreakdown)> {
    state
        .tribes
        .iter()
        .filter(|(_, t)| t.killed_turn <= 0 && t.resigned_turn <= 0)
        .map(|(id, t)| (*id, (t.score, breakdown(state, *id))))
        .collect()
}

/// `describe` carries indices; the ranking wants the move kind only.
fn move_kind(desc: &str) -> String {
    desc.split([' ', '('])
        .take(2)
        .collect::<Vec<_>>()
        .join(" ")
        .trim()
        .to_string()
}

fn main() {
    let args = parse_args();

    let mut pool = CORE_TRIBES.to_vec();
    if args.tribes_all {
        pool.extend_from_slice(&SPECIAL_TRIBES);
    }

    // kind -> (occurrences, summed |delta|, worst single delta, first example)
    let mut charges: HashMap<String, (usize, i64, i32, String)> = HashMap::new();
    let mut total_moves = 0u64;
    let mut drifting_games = 0usize;

    for seed in args.start_seed..args.start_seed + args.num_seeds {
        let mut rng = StdRng::seed_from_u64(seed);
        let t1 = pool[rng.random_range(0..pool.len())];
        let mut t2 = pool[rng.random_range(0..pool.len())];
        while t2 == t1 {
            t2 = pool[rng.random_range(0..pool.len())];
        }

        let mode = args.mode.unwrap_or(if seed % 2 == 0 {
            ModeType::Domination
        } else {
            ModeType::Perfection
        });

        let gen_settings = polyfish::mapgen::MapGenSettings {
            size: polyfish::types::MapSize::Tiny,
            map_type: polyfish::types::MapType::Drylands,
            tribes: vec![t1, t2],
            seed: seed as i64,
            symmetric: true,
            ..Default::default()
        };

        let mut game = Game::new();
        game.state = polyfish::mapgen::generate(gen_settings);
        game.state.settings.mode = mode;
        game.state.settings.max_turns = args.max_turns;
        game.post_load();

        let mut before = snapshot(&game.state);
        let mut moves = 0usize;
        let mut game_drifted = false;

        while !game.state.settings._game_over && moves < 2000 {
            let legal = game.legal_moves();
            if legal.is_empty() {
                break;
            }
            let m = &legal[rng.random_range(0..legal.len())];
            let desc = m.describe(&game.state);
            if game.play_move(m.as_ref()).is_none() {
                println!("REAL MOVE FAILED: seed {} move {}", seed, desc);
                break;
            }
            moves += 1;
            total_moves += 1;

            let after = snapshot(&game.state);
            for (id, (incr_after, canon_after)) in &after {
                let (incr_before, canon_before) = before
                    .get(id)
                    .copied()
                    .unwrap_or((*incr_after, *canon_after));
                let d_before = incr_before - canon_before.total();
                let d_after = incr_after - canon_after.total();
                let delta = d_after - d_before;
                if delta == 0 {
                    continue;
                }
                game_drifted = true;
                let components: Vec<String> = canon_after
                    .diff(&canon_before)
                    .into_iter()
                    .map(|(name, d)| format!("{} {:+}", name, d))
                    .collect();
                let example = format!(
                    "seed {} turn {} player {} move `{}` charged {:+} (drift {} -> {}); incremental {:+}, canonical [{}]",
                    seed,
                    game.state.settings.turn,
                    id,
                    desc,
                    delta,
                    d_before,
                    d_after,
                    incr_after - incr_before,
                    components.join(", ")
                );
                let entry = charges
                    .entry(move_kind(&desc))
                    .or_insert((0, 0, 0, example.clone()));
                entry.0 += 1;
                entry.1 += delta.unsigned_abs() as i64;
                entry.2 = entry.2.max(delta.abs());
                if args.stop_on_first {
                    println!("FIRST DRIFT: {}", example);
                    std::process::exit(1);
                }
            }
            before = after;
        }

        if game_drifted {
            drifting_games += 1;
        }
        println!(
            "seed {} done ({:?} vs {:?}, mode {:?}, {} moves, drifted: {})",
            seed, t1, t2, mode, moves, game_drifted
        );
    }

    println!(
        "\n{} of {} games drifted over {} moves",
        drifting_games, args.num_seeds, total_moves
    );
    if charges.is_empty() {
        println!("SCORE PARITY CLEAN");
        return;
    }

    let mut ranked: Vec<_> = charges.into_iter().collect();
    ranked.sort_by_key(|(_, (_, sum, _, _))| -*sum);
    println!("\nmove kind                  charges   |sum|   worst");
    for (kind, (n, sum, worst, example)) in &ranked {
        println!("{:26} {:7} {:7} {:7}", kind, n, sum, worst);
        println!("    e.g. {}", example);
    }
    std::process::exit(1);
}
