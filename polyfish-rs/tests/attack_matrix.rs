//! Direct coverage for `actions::units::attack_unit`, which resolves every fight the
//! agent plays and every reward the search backs up. It had none: the index re-find
//! after a splash kill, the retaliation gates, the Persist/DoubleAttack/Escape flag
//! block and the melee move-in were all reachable only through the random-move fuzz.
//!
//! States are hand-built on flat Field terrain so `get_defense_bonus` is exactly 1.0
//! and every expected number below is the bare `calculate_combat` formula.

use polyfish::actions::units::{attack_unit, calculate_combat};
use polyfish::coords::Coords;
use polyfish::states::{GameState, PlayerId, TileState, TribeState, UnitState};
use polyfish::types::{TerrainType, TribeType, UnitEffect, UnitType};

const SIZE: i32 = 11;

/// Two mutually hostile tribes on an all-Field map. `get_defense_bonus` unwraps on
/// both the owning tribe and the unit's tile, so every tile must exist up front.
fn flat_state() -> GameState {
    let mut state = GameState::default();
    for i in 0..(SIZE * SIZE) {
        state.tiles.insert(
            i,
            TileState {
                coords: Coords::from_index(i, SIZE),
                terrain_type: TerrainType::Field,
                ..Default::default()
            },
        );
    }
    for id in [1, 2] {
        state.tribes.insert(
            id,
            TribeState {
                id,
                tribe_type: TribeType::Imperius,
                ..Default::default()
            },
        );
    }
    state
}

fn place(
    state: &mut GameState,
    owner: PlayerId,
    unit_type: UnitType,
    idx: i32,
    health: f32,
) -> usize {
    let unit = UnitState {
        owner,
        unit_type,
        health,
        coords: Coords::from_index(idx, SIZE),
        prev_coords: Coords::from_index(idx, SIZE),
        ..Default::default()
    };
    let tribe = state.tribes.get_mut(&owner).unwrap();
    tribe.units.push(unit);
    let unit_idx = tribe.units.len() - 1;
    state.tiles.get_mut(&idx).unwrap()._unit_owner_id = Some(owner);
    unit_idx
}

fn unit_at(state: &GameState, owner: PlayerId, idx: i32) -> Option<&UnitState> {
    state.tribes[&owner]
        .units
        .iter()
        .find(|u| u.coords.idx == idx)
}

fn hp(state: &GameState, owner: PlayerId, idx: i32) -> Option<f32> {
    unit_at(state, owner, idx).map(|u| u.health)
}

fn snapshot(state: &GameState) -> serde_json::Value {
    serde_json::to_value(state).unwrap()
}

#[test]
fn melee_trade_applies_both_damages() {
    let mut state = flat_state();
    let atk = place(&mut state, 1, UnitType::Warrior, 59, 10.0);
    place(&mut state, 2, UnitType::Warrior, 60, 10.0);
    let before = snapshot(&state);

    let undo = attack_unit(&mut state, 1, atk, 2, Some(0));

    assert_eq!(hp(&state, 2, 60), Some(5.0), "defender damage");
    assert_eq!(hp(&state, 1, 59), Some(5.0), "retaliation damage");
    let attacker = unit_at(&state, 1, 59).unwrap();
    assert!(attacker.attacked);
    assert!(attacker.moved);
    assert_eq!(
        attacker.last_attack_coords.as_ref().map(|c| c.idx),
        Some(60)
    );
    // prev_coords is mirrored across the attacker so push logic reads the attack as a step in.
    assert_eq!(attacker.prev_coords.idx, 58);

    undo(&mut state);
    assert_eq!(snapshot(&state), before, "melee trade undo");
}

#[test]
fn melee_kill_moves_attacker_in() {
    let mut state = flat_state();
    let atk = place(&mut state, 1, UnitType::Warrior, 59, 10.0);
    place(&mut state, 2, UnitType::Warrior, 60, 4.0);
    let before = snapshot(&state);

    let undo = attack_unit(&mut state, 1, atk, 2, Some(0));

    assert!(
        state.tribes[&2].units.is_empty(),
        "defender was not removed"
    );
    let attacker = &state.tribes[&1].units[0];
    assert_eq!(attacker.coords.idx, 60, "melee attacker did not move in");
    assert_eq!(attacker.health, 10.0, "a dead defender must not retaliate");
    assert_eq!(attacker.kills, 1);
    assert_eq!(state.tiles[&60]._unit_owner_id, Some(1));
    assert_eq!(state.tiles[&59]._unit_owner_id, None);
    assert_eq!(state.tribes[&1].kills, 1);
    assert_eq!(state.tribes[&2].casualties, 1);

    undo(&mut state);
    assert_eq!(snapshot(&state), before, "melee kill undo");
}

#[test]
fn ranged_kill_does_not_move_attacker() {
    let mut state = flat_state();
    let atk = place(&mut state, 1, UnitType::Archer, 58, 10.0);
    place(&mut state, 2, UnitType::Warrior, 60, 4.0);
    let before = snapshot(&state);

    let undo = attack_unit(&mut state, 1, atk, 2, Some(0));

    assert!(
        state.tribes[&2].units.is_empty(),
        "defender was not removed"
    );
    assert_eq!(state.tribes[&1].units[0].coords.idx, 58);
    assert_eq!(state.tiles[&58]._unit_owner_id, Some(1));
    assert_eq!(state.tiles[&60]._unit_owner_id, None);

    undo(&mut state);
    assert_eq!(snapshot(&state), before, "ranged kill undo");
}

#[test]
fn retaliation_requires_reach_and_no_surprise() {
    // Identical damage arithmetic in all three legs; only the retaliation gate differs.
    for (name, unit_type, from, expect_attacker_hp) in [
        ("out of the defender's range", UnitType::Archer, 58, 10.0),
        ("in the defender's range", UnitType::Archer, 59, 5.0),
        ("Surprise", UnitType::Dagger, 59, 10.0),
    ] {
        let mut state = flat_state();
        let atk = place(&mut state, 1, unit_type, from, 10.0);
        place(&mut state, 2, UnitType::Warrior, 60, 10.0);
        let before = snapshot(&state);

        let undo = attack_unit(&mut state, 1, atk, 2, Some(0));

        assert_eq!(hp(&state, 2, 60), Some(5.0), "{name}: defender damage");
        assert_eq!(
            hp(&state, 1, from),
            Some(expect_attacker_hp),
            "{name}: attacker health after retaliation"
        );

        undo(&mut state);
        assert_eq!(snapshot(&state), before, "{name}: undo");
    }
}

#[test]
fn stiff_defender_does_not_retaliate() {
    let mut state = flat_state();
    let atk = place(&mut state, 1, UnitType::Warrior, 59, 10.0);
    place(&mut state, 2, UnitType::MindBender, 60, 10.0);

    // The retaliation damage exists; Stiff is what suppresses it.
    assert_eq!(
        calculate_combat(2.0, 10.0, 10.0, 1.0, 10.0, 10.0, 1.0).defense_damage,
        2.0
    );

    let before = snapshot(&state);
    let undo = attack_unit(&mut state, 1, atk, 2, Some(0));

    assert_eq!(hp(&state, 2, 60), Some(4.0));
    assert_eq!(hp(&state, 1, 59), Some(10.0), "Stiff defender retaliated");

    undo(&mut state);
    assert_eq!(snapshot(&state), before, "stiff defender undo");
}

#[test]
fn retaliation_can_kill_the_attacker() {
    let mut state = flat_state();
    let atk = place(&mut state, 1, UnitType::Warrior, 59, 1.0);
    place(&mut state, 2, UnitType::Defender, 60, 15.0);
    let before = snapshot(&state);

    let undo = attack_unit(&mut state, 1, atk, 2, Some(0));

    assert!(
        state.tribes[&1].units.is_empty(),
        "attacker survived 13 retaliation damage"
    );
    assert_eq!(hp(&state, 2, 60), Some(14.0));
    assert_eq!(state.tribes[&2].kills, 1);
    assert_eq!(state.tribes[&2].units[0].kills, 1);
    assert_eq!(state.tribes[&1].casualties, 1);
    assert_eq!(state.tiles[&59]._unit_owner_id, None);

    undo(&mut state);
    assert_eq!(snapshot(&state), before, "retaliation kill undo");
}

/// The headline case: splash kills a unit of the defender's own tribe, shifting the
/// defender's index in `tribe.units`. Without the re-find before removal the engine
/// would delete the wrong unit.
#[test]
fn splash_kill_shifts_the_defender_index() {
    let mut state = flat_state();
    let dragon = place(&mut state, 1, UnitType::FireDragon, 58, 20.0);
    place(&mut state, 2, UnitType::Warrior, 61, 1.0); // splashed, dies, index 0
    let defender = place(&mut state, 2, UnitType::Warrior, 60, 10.0);
    place(&mut state, 2, UnitType::Warrior, 5, 10.0); // far bystander, must survive
    let before = snapshot(&state);

    let undo = attack_unit(&mut state, 1, dragon, 2, Some(defender));

    assert_eq!(
        state.tribes[&2].units.len(),
        1,
        "expected exactly the bystander to survive"
    );
    let survivor = &state.tribes[&2].units[0];
    assert_eq!(survivor.coords.idx, 5, "the wrong unit was removed");
    assert_eq!(survivor.health, 10.0);
    assert_eq!(state.tiles[&60]._unit_owner_id, None);
    assert_eq!(state.tiles[&61]._unit_owner_id, None);
    assert_eq!(state.tribes[&1].kills, 2);
    assert_eq!(state.tribes[&1].units[0].kills, 2);
    assert_eq!(state.tribes[&2].casualties, 2);
    assert_eq!(
        state.tribes[&1].units[0].coords.idx, 58,
        "a range-2 attacker must not move in"
    );

    undo(&mut state);
    assert_eq!(snapshot(&state), before, "splash kill undo");
}

#[test]
fn persist_leaves_a_ranged_attacker_unspent_on_a_kill() {
    let mut state = flat_state();
    let atk = place(&mut state, 1, UnitType::Tridention, 58, 10.0);
    place(&mut state, 2, UnitType::Warrior, 60, 4.0);
    let before = snapshot(&state);

    let undo = attack_unit(&mut state, 1, atk, 2, Some(0));

    assert!(state.tribes[&2].units.is_empty());
    let attacker = &state.tribes[&1].units[0];
    assert!(
        !attacker.attacked,
        "Persist must not spend the attacker on a kill"
    );
    assert_eq!(attacker.attacks_performed, 1);

    undo(&mut state);
    assert_eq!(snapshot(&state), before, "persist kill undo");
}

#[test]
fn persist_does_not_apply_when_the_defender_survives() {
    let mut state = flat_state();
    let atk = place(&mut state, 1, UnitType::Knight, 59, 10.0);
    place(&mut state, 2, UnitType::Defender, 60, 15.0);
    let before = snapshot(&state);

    let undo = attack_unit(&mut state, 1, atk, 2, Some(0));

    assert_eq!(hp(&state, 2, 60), Some(7.0));
    assert_eq!(hp(&state, 1, 59), Some(4.0), "Knight took retaliation");
    assert!(unit_at(&state, 1, 59).unwrap().attacked);

    undo(&mut state);
    assert_eq!(snapshot(&state), before, "persist non-kill undo");
}

#[test]
fn double_attack_spends_the_unit_on_the_second_attack() {
    let mut state = flat_state();
    let atk = place(&mut state, 1, UnitType::Phychi, 58, 5.0);
    place(&mut state, 2, UnitType::Warrior, 60, 10.0);
    let before = snapshot(&state);

    let undo_first = attack_unit(&mut state, 1, atk, 2, Some(0));
    assert_eq!(hp(&state, 2, 60), Some(9.0), "first attack damage");
    assert!(
        state.tribes[&2].units[0]
            .effects
            .contains(&UnitEffect::Poison),
        "Poison was not applied"
    );
    let attacker = &state.tribes[&1].units[0];
    assert_eq!(attacker.attacks_performed, 1);
    assert!(
        !attacker.attacked,
        "DoubleAttack must allow a second attack"
    );

    let undo_second = attack_unit(&mut state, 1, atk, 2, Some(0));
    assert_eq!(hp(&state, 2, 60), Some(7.0), "second attack damage");
    let attacker = &state.tribes[&1].units[0];
    assert_eq!(attacker.attacks_performed, 2);
    assert!(attacker.attacked, "the second attack must spend the unit");

    undo_second(&mut state);
    undo_first(&mut state);
    assert_eq!(snapshot(&state), before, "double attack undo");
}

#[test]
fn escape_clears_moved_even_after_the_move_in() {
    let mut state = flat_state();
    let atk = place(&mut state, 1, UnitType::Rider, 59, 10.0);
    place(&mut state, 2, UnitType::Warrior, 60, 3.0);
    let before = snapshot(&state);

    let undo = attack_unit(&mut state, 1, atk, 2, Some(0));

    assert!(state.tribes[&2].units.is_empty());
    let attacker = &state.tribes[&1].units[0];
    assert_eq!(attacker.coords.idx, 60);
    assert!(
        !attacker.moved,
        "Escape must clear the move-in's moved flag"
    );
    assert!(attacker.attacked);

    undo(&mut state);
    assert_eq!(snapshot(&state), before, "escape undo");
}

/// Regression: the Frozen undo used to remove the effect unconditionally, so attacking
/// an already-Frozen defender and undoing stripped an effect the attack never added.
#[test]
fn freeze_undo_round_trips_on_an_already_frozen_defender() {
    for already_frozen in [false, true] {
        let mut state = flat_state();
        let atk = place(&mut state, 1, UnitType::IceArcher, 58, 10.0);
        place(&mut state, 2, UnitType::Warrior, 60, 10.0);
        if already_frozen {
            state.tribes.get_mut(&2).unwrap().units[0]
                .effects
                .insert(UnitEffect::Frozen);
        }
        let before = snapshot(&state);

        let undo = attack_unit(&mut state, 1, atk, 2, Some(0));

        assert_eq!(hp(&state, 2, 60), Some(8.0));
        assert!(
            state.tribes[&2].units[0]
                .effects
                .contains(&UnitEffect::Frozen),
            "already_frozen={already_frozen}: Freeze was not applied"
        );

        undo(&mut state);
        assert_eq!(
            state.tribes[&2].units[0]
                .effects
                .contains(&UnitEffect::Frozen),
            already_frozen,
            "already_frozen={already_frozen}: undo did not restore the Frozen effect"
        );
        assert_eq!(snapshot(&state), before, "freeze undo");
    }
}

/// Current behaviour, not asserted intent: on a MELEE Persist kill the move-in
/// `step_unit` sets `attacked` before the Persist branch runs, and that branch only
/// declines to set the flag rather than clearing it, so a Knight cannot chain-kill.
/// Pinned so a fix is a deliberate, measured change rather than a silent one.
#[test]
fn melee_persist_kill_is_still_spent_by_the_move_in() {
    let mut state = flat_state();
    let atk = place(&mut state, 1, UnitType::Knight, 59, 10.0);
    place(&mut state, 2, UnitType::Warrior, 60, 9.0);
    let before = snapshot(&state);

    let undo = attack_unit(&mut state, 1, atk, 2, Some(0));

    assert!(state.tribes[&2].units.is_empty());
    let attacker = &state.tribes[&1].units[0];
    assert_eq!(attacker.coords.idx, 60);
    assert!(attacker.attacked);
    assert!(attacker.moved);
    assert_eq!(attacker.attacks_performed, 1);

    undo(&mut state);
    assert_eq!(snapshot(&state), before, "melee persist kill undo");
}
