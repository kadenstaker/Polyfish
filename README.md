# Polyfish

An `AI NN` + MCTS `Rust` capable of playing the award winning `Polytopia` strategy game. 

## Features
- **Replica Simulation**: I painstakingly rebuilt the entire Polytopia game logic in Typescript, then translated it to Rust for performance and AI training.
- **Web UI**: Mimicry of the original game's UI, fully interactive and served by the Rust backend.
- **AI Engine**: A hybrid MCTS (Monte Carlo Tree Search) + Neural Network (Alpha-Zero style) approach, trained by self-play.
- **Strength Ladder**: Frozen-anchor Elo measurement (`arena` + `ladder.py` + `elo.py`) wired into the training loop, with plateau early-stop.
- **Training Dashboard**: Live charts of loss, move mix, and value distribution, served by the same backend.
- **Game RIPPER**: C++ injection script that extracts live game states from the Steam version of Polytopia.

## Quick Start

1. Install **Rust**, **Node**, and **Python 3**.
2. *(optional)* Build the React SPA: `cd polyfish-ui && npm install && npm run build` — needs Node ≥20.19 (vite 8). The simulator and the training dashboard are plain static files under `src/public` and need no build.
3. Run `./run-server.sh` (frees port 3000, then runs the `polyfish` binary from `polyfish-rs/`).
4. Open `http://localhost:3000/simulator/index.html` for the simulator and `http://localhost:3000/training.html` for the training dashboard. `http://localhost:3000` is the React SPA once built, and falls back to the static simulator otherwise.

On startup the server loads `polyfish-rs/live_game.json`, then `saved_state.json`, then the newest canonical `replays/*.replay.json`, before falling back to a generated map. Pre-canonical mod captures are converted on the way in by `/replay/save`, or offline with `import_replays convert-legacy`.

## Open Work

There is no TODO list here — it drifts. The live lists are:

- **`expert_pipeline_audit.md`** — the current open-work list for the training pipeline. Every item has a status and a command that re-checks it. Start here.
- **`hypothesis_driven_improvements.md`** — pre-registered experiments with COMMITTED/REJECTED verdicts. Read before proposing a change; several obvious ideas have already been measured.
- **`expert_review.md`** / **`expert_boost_throughput.md`** — prior search/learning-signal review and a measured throughput investigation.
- [GitHub issues](https://github.com/kadenstaker/Polyfish/issues) for everything else.

The one perennial item: training the network well still needs monster compute.

## Training

- **`polyfish-rs/local_setup.sh`**: Creates `polyfish-rs/.venv` from `polyfish-rs/requirements.txt`. `remote_setup.sh` / `vast_setup.sh` do the same on a GPU box and also install PyTorch.
- **`polyfish-rs/run_training_loop.sh`**: The driver — `init_model.py` → `self_play` (Rust, writes `games_*.safetensors`) → `train.py` (PyTorch, updates `model.safetensors`) → log a row and checkpoint, plus a strength-gauge arena match every few iterations.
- **`polyfish-rs/train.py`**: The PyTorch trainer. Its network definition must stay byte-compatible with `polyfish-rs/src/ai/network.rs` — both read and write the same `model.safetensors`.
- Head-to-head evaluation: `cargo run --release --bin arena -- --model1 a.safetensors --model2 b.safetensors --games 32 --mcts 64` (each seed is played twice with sides swapped).
- **Windows**: train under WSL2 with `polyfish-rs/run_training_loop.sh`. The PowerShell chain (`run_training_loop.ps1`, `run-training.bat`, `auto_train.ps1`, `start-hidden.vbs`) was deleted because it trained with no strength gauge, no anchor freeze and no plateau stop; git history has it. `run-server.bat` is unaffected and still launches the server.
- **`auto_train.sh`** (repo root, Linux/GNOME only): idle babysitter that runs the loop with `--resume` while the desktop is idle and halts the whole process group when it is not.

## Core Modules

- **`polyfish-rs/`**: The Rust game engine, AI, web backend, and every training binary. Almost all work happens here.
- **`polyfish-ui/`**: Vite/React frontend. The Rust server serves its `dist/` build at `/`, `/assets` and `/static`; `dist/` is gitignored, so build it before you expect the SPA.
- **`src/public/`**: The static Web UI (JS/HTML/CSS) — the simulator and the training dashboard, and the only copy of either. The server mounts it at `/simulator/*` and as the fallback for `/`; `polyfish-ui` iframes `/simulator/index.html` and `/simulator/training.html` rather than keeping a fork.
- **`polyfish-mod/`**: C# BepInEx/PolyMod mod that auto-plays replays inside the real Steam game and POSTs captured states to the local server.
- **`polyfish-scraper/`**: Utilities for gathering game data and assets.
- **`CLAUDE.md`**: The per-file map of the tree, including the traps. The deep reference this README summarises.
- **`notes.md` / `notes-heuristics.md` / `notes-memory.md`**: Architectural research, branching-factor analysis, and the observation-memory channels.

## AI Architecture

- **`polyfish-rs/src/ai/gumbel_mcts.rs`**: Gumbel Alpha-Zero search — the one self-play training actually runs.
- **`polyfish-rs/src/ai/mcts_zero.rs`**: The PUCT Alpha-Zero MCTS (`original_mcts_zero.rs` is an older implementation kept for comparison; the unused `mcts.rs` has been deleted).
- **`polyfish-rs/src/ai/heuristic_mcts.rs`**: Network-free MCTS for UI analysis and the interactive `trainer` binary.
- **`polyfish-rs/src/ai/network.rs`**: `PolyZeroNet` — candle ResNet trunk plus cross-attention, a decomposed policy, and a value head.
- **`polyfish-rs/src/ai/mapper.rs`**: Maps moves onto the four policy heads (action type, source, target, option) so the policy is independent of legal-move ordering.
- **`polyfish-rs/src/ai/features.rs`**: Logic for encoding GameState into NN tensors (11x11 maps).
- **`polyfish-rs/src/ai/evaluator/`**: Modular logic for Economy, Military, Research, Exploration, and Expansion evaluation.
- **`polyfish-rs/src/ai/book.rs`**: Opening move library for standardized tribe starts. Data only — no search agent consults it.
- **`polyfish-rs/src/bin/`**: The CLI tools — `self_play`, `arena`, `train`, `trainer`, plus benchmarks, replay management, and debug probes.

## Inference Backends

Three implementations read the same `model.safetensors`, picked by Cargo feature:

- **`network.rs`** (candle) — the default, and the only backend on non-Apple hardware. `cuda`/`cudnn` opt into the GPU.
- **`tch_network.rs`** (`tch-eval`) — libtorch/MPS on macOS; needs the env vars documented in `Cargo.toml`.
- **`metal_network.rs`** (`metal-eval`) — hand-composed MPSGraph, fastest on Apple silicon.

`eval_backend.rs` / `eval_server.rs` are the batching layer that fans leaf evaluations from many actors onto whichever backend is selected.

## GameState Ripper (Steam)

`polyfish-reader/` is a separate C++ tree and is **not checked into this repo**. `scan.sh` finds the running `Polytopia.exe` and runs the compiled reader against it, dumping live game state as JSON the simulator can load; it needs `g++` and `sudo`.

- **`polyfish-reader.cpp`**: Memory reader that extracts live game state for the simulator.
- **`inputerv2`**: Interactive tool for manipulating live game memory.
- **`polyfish-scanner.cpp`**: Utility for finding memory offsets in new game versions.
