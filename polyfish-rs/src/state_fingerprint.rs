//! The half of an undo comparison that a JSON snapshot cannot see.
//!
//! `serde_json` sorts object keys, so an `IndexMap` entry that undo re-appended
//! instead of restoring to its slot compares equal, and `#[serde(skip)]` shadow
//! state is invisible entirely. Undo probes compare this alongside the JSON.

use crate::states::GameState;

/// Iteration order of every `IndexMap` in the state plus the simulation-only
/// shadow sets, one line each. Iteration order is load-bearing — temple growth,
/// fungi spread, sanctuary spawns and mycelium heal caps all walk these in
/// order, and unit spawn order feeds movegen order.
pub fn undo_fingerprint(state: &GameState) -> Vec<String> {
    let join = |keys: Vec<String>| keys.join(",");
    let ints = |keys: Vec<i32>| join(keys.into_iter().map(|k| k.to_string()).collect());

    let mut lines = vec![
        format!("tiles: {}", ints(state.tiles.keys().copied().collect())),
        format!(
            "structures: {}",
            ints(state.structures.keys().copied().collect())
        ),
        format!(
            "resources: {}",
            ints(state.resources.keys().copied().collect())
        ),
        format!("tribes: {}", ints(state.tribes.keys().copied().collect())),
    ];
    for (id, tribe) in &state.tribes {
        lines.push(format!(
            "tribe {} relations: {}",
            id,
            ints(tribe.relations.keys().copied().collect())
        ));
        lines.push(format!(
            "tribe {} memory_units: {}",
            id,
            ints(tribe.memory_units.keys().copied().collect())
        ));
        lines.push(format!(
            "tribe {} memory_attacks: {}",
            id,
            ints(tribe.memory_attacks.keys().copied().collect())
        ));
    }

    if let Some(pred) = &state._prediction {
        lines.push(format!(
            "prediction villages: {}",
            ints(pred._villages.keys().copied().collect())
        ));
        lines.push(format!(
            "prediction terrain: {}",
            ints(pred._terrain.keys().copied().collect())
        ));
    }

    // `_sim_explored` is `#[serde(skip)]`, so a dropped undo of the exploration
    // credit is invisible to JSON. Empty sets are skipped because `discover_tiles`
    // inserts via `entry().or_default()` but its undo removes only the indices,
    // leaving a permanent empty entry that is not drift.
    let mut sim_explored: Vec<(i32, Vec<i32>)> = state
        .settings
        ._sim_explored
        .iter()
        .filter(|(_, idxs)| !idxs.is_empty())
        .map(|(pov, idxs)| {
            let mut idxs: Vec<i32> = idxs.iter().copied().collect();
            idxs.sort_unstable();
            (*pov, idxs)
        })
        .collect();
    sim_explored.sort_unstable_by_key(|(pov, _)| *pov);
    for (pov, idxs) in sim_explored {
        lines.push(format!("sim_explored {}: {}", pov, ints(idxs)));
    }

    lines
}
