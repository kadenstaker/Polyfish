//! Perft-style make/unmake verifier for the MCTS simulate/undo path.
//!
//! For each seed: build a game exactly like self_play, then interleave
//! random real moves with random MCTS-like descents (simulate_move × depth,
//! undo all). After every descent the full state is deep-compared against
//! the pre-descent snapshot; the first mismatch prints the offending move
//! sequence, the arm it was on, and a single-seed repro command, then exits
//! non-zero.
//!
//! The JSON comparison is order-blind (`serde_json` sorts object keys) and
//! cannot see `#[serde(skip)]` shadow state, so every snapshot also carries
//! `state_fingerprint::undo_fingerprint` (#50, #47).
//!
//! Four descent batches run between real moves, and by default they rotate the
//! arms: legacy/real, adversarial/fogged, legacy/fogged, adversarial/real. The
//! fogged arms descend inside a `clone_for_mcts` view, which is where the
//! in-tree opponent plays a belief-state army against cleared explorers.
//!
//! Usage: cargo run --release --example undo_fuzz -- \
//!          [num_seeds] [start_seed] [max_depth] \
//!          [--adversarial=auto|on|off] [--view=auto|real|fogged] \
//!          [--tribes=core|all] [--mode=alternate|perfection|domination] \
//!          [--symmetric=true|false]

use polyfish::game::Game;
use polyfish::state_fingerprint::undo_fingerprint;
use polyfish::states::PlayerId;
use polyfish::types::{ModeType, TribeType};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde_json::Value;

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

/// Special mechanics (fungi, mycelium, sanctuary) on thinner engine coverage.
/// Polaris is deliberately absent: out of scope project-wide.
const SPECIAL_TRIBES: [TribeType; 3] =
    [TribeType::Aquarion, TribeType::Elyrion, TribeType::Cymanti];

#[derive(Clone, Copy, PartialEq)]
enum ModeSel {
    Perfection,
    Domination,
    Alternate,
}

struct Args {
    num_seeds: u64,
    start_seed: u64,
    max_depth: usize,
    /// `None` rotates the arm per descent batch.
    adversarial: Option<bool>,
    fogged: Option<bool>,
    tribes_all: bool,
    mode: ModeSel,
    symmetric: bool,
}

/// Which arm a descent batch is running, for the failure report.
struct ArmCtx {
    seed: u64,
    tribes: (TribeType, TribeType),
    mode: ModeType,
    adversarial: bool,
    fogged: bool,
    pov: PlayerId,
    real_moves: usize,
    repro: String,
}

const USAGE: &str = "usage: undo_fuzz [num_seeds] [start_seed] [max_depth] \
[--adversarial=auto|on|off] [--view=auto|real|fogged] [--tribes=core|all] \
[--mode=alternate|perfection|domination] [--symmetric=true|false]";

fn parse_args() -> Args {
    let mut positionals: Vec<String> = Vec::new();
    let mut adversarial = None;
    let mut fogged = None;
    let mut tribes_all = false;
    let mut mode = ModeSel::Alternate;
    let mut symmetric = true;

    for arg in std::env::args().skip(1) {
        let Some(body) = arg.strip_prefix("--") else {
            positionals.push(arg);
            continue;
        };
        let (key, value) = body.split_once('=').unwrap_or((body, ""));
        match (key, value) {
            ("adversarial", "auto") => adversarial = None,
            ("adversarial", "on") => adversarial = Some(true),
            ("adversarial", "off") => adversarial = Some(false),
            ("view", "auto") => fogged = None,
            ("view", "real") => fogged = Some(false),
            ("view", "fogged") => fogged = Some(true),
            ("tribes", "core") => tribes_all = false,
            ("tribes", "all") => tribes_all = true,
            ("mode", "perfection") => mode = ModeSel::Perfection,
            ("mode", "domination") => mode = ModeSel::Domination,
            ("mode", "alternate") => mode = ModeSel::Alternate,
            ("symmetric", "true") => symmetric = true,
            ("symmetric", "false") => symmetric = false,
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
    if positionals.len() > 3 {
        eprintln!("too many positional arguments\n{}", USAGE);
        std::process::exit(2);
    }

    Args {
        num_seeds: positional(0, 50),
        start_seed: positional(1, 1),
        max_depth: positional(2, 16) as usize,
        adversarial,
        fogged,
        tribes_all,
        mode,
        symmetric,
    }
}

fn json_diff(prefix: String, a: &Value, b: &Value, out: &mut Vec<String>) {
    if out.len() >= 20 {
        return;
    }
    match (a, b) {
        (Value::Object(ma), Value::Object(mb)) => {
            for (k, va) in ma {
                match mb.get(k) {
                    Some(vb) => json_diff(format!("{}/{}", prefix, k), va, vb, out),
                    None => out.push(format!("{}/{}: {} -> <missing>", prefix, k, va)),
                }
            }
            for (k, vb) in mb {
                if !ma.contains_key(k) {
                    out.push(format!("{}/{}: <missing> -> {}", prefix, k, vb));
                }
            }
        }
        (Value::Array(aa), Value::Array(ab)) => {
            if aa.len() != ab.len() {
                out.push(format!(
                    "{}: array len {} -> {}",
                    prefix,
                    aa.len(),
                    ab.len()
                ));
            }
            for (i, (va, vb)) in aa.iter().zip(ab.iter()).enumerate() {
                json_diff(format!("{}[{}]", prefix, i), va, vb, out);
            }
        }
        _ => {
            if a != b {
                out.push(format!("{}: {} -> {}", prefix, a, b));
            }
        }
    }
}

/// Positional, so a fingerprint line that appears or vanishes (a `_sim_explored`
/// entry whose undo was dropped) is reported rather than skipped by a zip.
fn fingerprint_diff(want: &[String], got: &[String], out: &mut Vec<String>) {
    let missing = "<missing>";
    for i in 0..want.len().max(got.len()) {
        let a = want.get(i).map(String::as_str).unwrap_or(missing);
        let b = got.get(i).map(String::as_str).unwrap_or(missing);
        if a != b && out.len() < 20 {
            out.push(format!("fingerprint: {} -> {}", a, b));
        }
    }
}

fn snapshot(game: &Game) -> (Value, Vec<String>) {
    (
        serde_json::to_value(&game.state).expect("serialize state"),
        undo_fingerprint(&game.state),
    )
}

/// One MCTS-like descent: simulate a random line, then unwind it verifying that
/// each undo restores the snapshot taken before its move. Returns the number of
/// simulated moves; exits the process on the first mismatch.
fn descent_batch(game: &mut Game, rng: &mut StdRng, max_depth: usize, ctx: &ArmCtx) -> usize {
    let mut snapshots: Vec<(Value, Vec<String>)> = vec![snapshot(game)];
    let mut undos = Vec::new();
    let mut seq: Vec<String> = Vec::new();

    let depth = rng.random_range(1..=max_depth);
    for _ in 0..depth {
        if game.state.settings._game_over {
            break;
        }
        let legal = game.legal_moves();
        if legal.is_empty() {
            break;
        }
        let m = &legal[rng.random_range(0..legal.len())];
        let desc = m.describe(&game.state);
        match game.simulate_move(m.as_ref()) {
            Some(u) => {
                seq.push(desc);
                undos.push(u);
                snapshots.push(snapshot(game));
            }
            None => {
                println!("SIMULATE FAILED (legal move rejected)");
                println!("  seed: {}  {}", ctx.seed, arm_line(ctx));
                println!("  path: {:?}", seq);
                println!("  move: {}", desc);
                println!("  repro: {}", ctx.repro);
                std::process::exit(2);
            }
        }
    }

    let sim_moves = seq.len();
    while let Some(u) = undos.pop() {
        u(&mut game.state);
        snapshots.pop();
        let (expected, expected_order) = snapshots.last().expect("snapshot underflow");
        let (restored, restored_order) = snapshot(game);
        if expected != &restored || expected_order != &restored_order {
            let bad_idx = undos.len();
            let mut diffs = Vec::new();
            json_diff(String::new(), expected, &restored, &mut diffs);
            fingerprint_diff(expected_order, &restored_order, &mut diffs);
            println!("UNDO MISMATCH");
            println!(
                "  seed: {}  (tribes {:?} vs {:?})",
                ctx.seed, ctx.tribes.0, ctx.tribes.1
            );
            println!("  {}", arm_line(ctx));
            println!("  after real move #{}", ctx.real_moves);
            println!("  descent path ({} moves):", seq.len());
            for (i, d) in seq.iter().enumerate() {
                let marker = if i == bad_idx { "  <-- BAD UNDO" } else { "" };
                println!("    {:2}. {}{}", i, d, marker);
            }
            println!("  differing fields (max 20):");
            for d in &diffs {
                println!("    {}", d);
            }
            println!("  repro: {}", ctx.repro);
            std::process::exit(1);
        }
    }
    sim_moves
}

fn arm_line(ctx: &ArmCtx) -> String {
    format!(
        "arm: adversarial={} view={} mode={:?} pov={}",
        if ctx.adversarial { "on" } else { "off" },
        if ctx.fogged { "fogged" } else { "real" },
        ctx.mode,
        ctx.pov
    )
}

fn main() {
    let args = parse_args();

    let mut pool = CORE_TRIBES.to_vec();
    if args.tribes_all {
        pool.extend_from_slice(&SPECIAL_TRIBES);
    }

    let mut total_descents = 0u64;
    let mut total_sim_moves = 0u64;

    for seed in args.start_seed..args.start_seed + args.num_seeds {
        let mut rng = StdRng::seed_from_u64(seed);
        let t1 = pool[rng.random_range(0..pool.len())];
        let mut t2 = pool[rng.random_range(0..pool.len())];
        while t2 == t1 {
            t2 = pool[rng.random_range(0..pool.len())];
        }

        let mode = match args.mode {
            ModeSel::Perfection => ModeType::Perfection,
            ModeSel::Domination => ModeType::Domination,
            ModeSel::Alternate if seed % 2 == 0 => ModeType::Domination,
            ModeSel::Alternate => ModeType::Perfection,
        };

        let gen_settings = polyfish::mapgen::MapGenSettings {
            size: polyfish::types::MapSize::Tiny,
            map_type: polyfish::types::MapType::Drylands,
            tribes: vec![t1, t2],
            seed: seed as i64,
            symmetric: args.symmetric,
            ..Default::default()
        };

        let mut game = Game::new();
        game.state = polyfish::mapgen::generate(gen_settings);
        game.state.settings.mode = mode;
        game.state.settings.max_turns = 30;
        game.post_load();

        let mut real_moves = 0usize;
        'game: while !game.state.settings._game_over && real_moves < 400 {
            // A few MCTS-like descents from the current position, rotating the
            // arms unless the caller pinned one.
            for batch in 0..4 {
                let adversarial = args.adversarial.unwrap_or(batch % 2 == 1);
                let fogged = args.fogged.unwrap_or(batch == 1 || batch == 2);
                // `clone_for_mcts` reads the switch to decide whether to confine
                // the in-tree opponent's vision, so set it before cloning.
                polyfish::game::set_adversarial_search(adversarial);

                let pov = game.current_player_id();
                let ctx = ArmCtx {
                    seed,
                    tribes: (t1, t2),
                    mode,
                    adversarial,
                    fogged,
                    pov,
                    real_moves,
                    repro: repro_command(&args, seed),
                };

                let mut view;
                let target = if fogged {
                    view = game.clone_for_mcts(pov);
                    &mut view
                } else {
                    &mut game
                };
                total_sim_moves += descent_batch(target, &mut rng, args.max_depth, &ctx) as u64;
                total_descents += 1;
            }
            polyfish::game::set_adversarial_search(false);

            // Advance the real game with one random legal move.
            let legal = game.legal_moves();
            if legal.is_empty() {
                break 'game;
            }
            let m = &legal[rng.random_range(0..legal.len())];
            if game.play_move(m.as_ref()).is_none() {
                println!(
                    "REAL MOVE FAILED: seed {} move {}",
                    seed,
                    m.describe(&game.state)
                );
                break 'game;
            }
            real_moves += 1;
        }
        println!(
            "seed {} ok ({:?} vs {:?}, mode {:?}, {} real moves, {} descents so far, {} sim moves)",
            seed, t1, t2, mode, real_moves, total_descents, total_sim_moves
        );
    }
    println!(
        "ALL CLEAN: {} descents, {} simulated moves verified",
        total_descents, total_sim_moves
    );
}

/// The exact command that replays a failure. Each seed owns its RNG, so the
/// run's own flags must be echoed verbatim: pinning the resolved arm instead
/// would change which state each batch descends on and diverge the trajectory.
fn repro_command(args: &Args, seed: u64) -> String {
    let tri = |v: Option<bool>, on: &str, off: &str| match v {
        Some(true) => on.to_string(),
        Some(false) => off.to_string(),
        None => "auto".to_string(),
    };
    format!(
        "cargo run --release --example undo_fuzz -- 1 {} {} --adversarial={} --view={} --tribes={} --mode={} --symmetric={}",
        seed,
        args.max_depth,
        tri(args.adversarial, "on", "off"),
        tri(args.fogged, "fogged", "real"),
        if args.tribes_all { "all" } else { "core" },
        match args.mode {
            ModeSel::Perfection => "perfection",
            ModeSel::Domination => "domination",
            ModeSel::Alternate => "alternate",
        },
        args.symmetric
    )
}
