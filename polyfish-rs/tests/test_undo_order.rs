//! Regression for #50: removals that `shift_remove` an `IndexMap` entry used to
//! undo with a plain `insert`, which re-appends. The state compared equal but
//! iterated differently, and temple growth, fungi, sanctuary spawns and mycelium
//! healing all walk these maps in order — so an undone in-tree state could play
//! out differently than the original. The undo fuzz could not see it: its
//! `serde_json` comparison sorts object keys.

use polyfish::actions::resource::consume_resource;
use polyfish::actions::structure::destroy_structure;
use polyfish::game::Game;
use polyfish::mapgen::{MapGenSettings, generate};
use polyfish::memory::note_unit_removed;
use polyfish::states::{MemUnit, ResourceState, StructureState};
use polyfish::types::{MapSize, MapType, ResourceType, StructureType, TribeType, UnitType};

fn game(seed: i64) -> Game {
    let mut game = Game::new();
    game.state = generate(MapGenSettings {
        size: MapSize::Small,
        map_type: MapType::Continents,
        tribes: vec![TribeType::Imperius, TribeType::Bardur],
        seed,
        ..Default::default()
    });
    game.post_load();
    game
}

#[test]
fn destroy_structure_undo_restores_slot_order() {
    let mut game = game(7);
    let turn = game.state.settings.turn;

    game.state.structures.clear();
    for idx in [12, 34, 56, 78] {
        game.state.structures.insert(
            idx,
            Some(StructureState {
                structure_type: StructureType::Ruin,
                level: 1,
                founded: turn,
            }),
        );
    }
    let before: Vec<i32> = game.state.structures.keys().copied().collect();

    let undo = destroy_structure(&mut game.state, 34);
    assert_eq!(
        game.state.structures.keys().copied().collect::<Vec<_>>(),
        vec![12, 56, 78]
    );
    undo(&mut game.state);

    assert_eq!(
        game.state.structures.keys().copied().collect::<Vec<_>>(),
        before,
        "undo re-appended the structure instead of restoring its slot"
    );
}

#[test]
fn consume_resource_undo_restores_slot_order() {
    let mut game = game(7);

    game.state.resources.clear();
    for idx in [12, 34, 56, 78] {
        game.state.resources.insert(
            idx,
            Some(ResourceState {
                resource_type: ResourceType::Fruit,
            }),
        );
    }
    let before: Vec<i32> = game.state.resources.keys().copied().collect();

    let undo = consume_resource(&mut game.state, 34, None);
    assert_eq!(
        game.state.resources.keys().copied().collect::<Vec<_>>(),
        vec![12, 56, 78]
    );
    undo(&mut game.state);

    assert_eq!(
        game.state.resources.keys().copied().collect::<Vec<_>>(),
        before,
        "undo re-appended the resource instead of restoring its slot"
    );
}

#[test]
fn note_unit_removed_undo_restores_slot_order() {
    let mut game = game(7);
    let observer = *game.state.tribes.keys().next().unwrap();
    let turn = game.state.settings.turn;

    for idx in [12, 34, 56] {
        game.state
            .tribes
            .get_mut(&observer)
            .unwrap()
            .memory_units
            .insert(
                idx,
                MemUnit {
                    unit_type: UnitType::Warrior,
                    hp_norm: 1.0,
                    last_seen_turn: turn,
                },
            );
        game.state
            .tiles
            .get_mut(&idx)
            .unwrap()
            .explorers
            .insert(observer);
    }
    let before: Vec<i32> = game.state.tribes[&observer]
        .memory_units
        .keys()
        .copied()
        .collect();

    let undo = note_unit_removed(&mut game.state, 34);
    assert_eq!(
        game.state.tribes[&observer]
            .memory_units
            .keys()
            .copied()
            .collect::<Vec<_>>(),
        vec![12, 56]
    );
    undo(&mut game.state);

    assert_eq!(
        game.state.tribes[&observer]
            .memory_units
            .keys()
            .copied()
            .collect::<Vec<_>>(),
        before,
        "undo re-appended the ghost instead of restoring its slot"
    );
}
