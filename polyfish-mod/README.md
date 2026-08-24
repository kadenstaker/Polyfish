# PolyfishAI Mod

This is a mod for **The Battle of Polytopia** that helps capture replay data for the PolyAI project. 

Currently it basically sits inside the game and "scrapes" what happens in replays (like what an explorer finds or what rewards come from ruins) so we can use that data to train the AI.

## 🛠 What it does
- **Auto-Replay**: Opens and runs through replays on its own.
- **Fast Forward**: Speeds up the game logic by 20x to get through games quickly.
- **Data Capture**: Sends game states and moves to a local server.

## 📥 How to use it
1. Make sure you have **PolyMod** or **BepInEx** installed.
2. Put the `.dll` (compiled code) in your game's `plugins` folder.
3. Keep the **PolyAI/polyfish-rs** server running in the background while the game is open.

## ⚙️ Environment overrides

The capture rig's paths were hardcoded to one machine. They now read env vars, falling back to the original literals so an existing rig keeps working:

- `POLYFISH_SCRAPER_DATA` — directory holding `replays_*.txt` and `replays_all.txt` (default `/home/henry/Desktop/Coding/PolyAI/polyfish-scraper/data/`).
- `POLYFISH_REPLAY_QUEUE` — the replay-uuid queue file (default `/tmp/polyfish_replays.txt`).

## 📦 Capture payload

The mod still sends the pre-canonical payload shape. The server converts it on the way in, so **no mod rebuild is needed for a capture to be accepted**. A payload the server refuses is written to `polyfish-rs/replays/rejected/` with the reason beside it, and can be re-imported later with `import_replays convert-legacy`.

Note `PolyfishAPI.SaveReplaySync` is fire-and-forget and never inspects the response — the server answers 200 even on refusal — so the quarantine directory, not the mod's log, is where a failed capture shows up. Nothing in this repo compiles or lints C#, so mod-side edits are unverified here.
