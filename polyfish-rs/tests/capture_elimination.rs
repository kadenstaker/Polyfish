//! Capturing a tribe's last city eliminates it: every one of its units is swept off the
//! board in reverse index order and the city changes hands. Nothing covered that path,
//! and it is the only exercise of `capture_city`'s `old_city_idx` re-insert on undo.

use polyfish::game::Game;
use polyfish::mapgen::{MapGenSettings, generate};
use polyfish::moves::capture::CaptureMove;
use polyfish::states::UnitState;
use polyfish::types::{MapSize, MapType, MoveType, StructureType, TribeType, UnitType};

fn two_tribe_game(seed: i64) -> Game {
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
fn capturing_the_last_city_eliminates_the_tribe_and_undo_restores_it() {
    let mut game = two_tribe_game(20260823);
    let size = game.state.settings.size;

    assert_eq!(
        game.state.tribes[&2].cities.len(),
        1,
        "tribe 2 should start with one city"
    );
    let cap = game.state.tribes[&2].cities[0].idx;
    assert_eq!(
        game.state
            .structures
            .get(&cap)
            .and_then(|s| s.as_ref())
            .map(|s| s.structure_type),
        Some(StructureType::Village),
        "capture requires a Village structure on the tile"
    );

    // Clear tribe 2's starting units, then give it one unit somewhere else so the
    // elimination sweep has something to remove.
    for unit in std::mem::take(&mut game.state.tribes.get_mut(&2).unwrap().units) {
        game.state
            .tiles
            .get_mut(&unit.coords.idx)
            .unwrap()
            ._unit_owner_id = None;
    }
    let stranded = game
        .state
        .tiles
        .iter()
        .map(|(&idx, _)| idx)
        .find(|&idx| {
            idx != cap
                && polyfish::functions::get_unit_at(&game.state, idx).is_none()
                && !game.state.tiles[&idx].is_water_terrain()
        })
        .expect("no free land tile");
    let mut victim = UnitState {
        owner: 2,
        unit_type: UnitType::Warrior,
        health: 10.0,
        ..Default::default()
    };
    victim.coords.set_at(stranded, size);
    game.state.tribes.get_mut(&2).unwrap().units.push(victim);
    game.state.tiles.get_mut(&stranded).unwrap()._unit_owner_id = Some(2);

    let mut capturer = UnitState {
        owner: 1,
        unit_type: UnitType::Warrior,
        health: 10.0,
        ..Default::default()
    };
    capturer.coords.set_at(cap, size);
    game.state.tribes.get_mut(&1).unwrap().units.push(capturer);
    game.state.tiles.get_mut(&cap).unwrap()._unit_owner_id = Some(1);
    game.state.settings.current_player_turn_id = 1;

    let cities_before = game.state.tribes[&1].cities.len();
    let kills_before = game.state.tribes[&1].kills;
    let turn = game.state.settings.turn;
    assert_eq!(
        game.state.tribes[&2].units.len(),
        1,
        "the sweep needs a unit to remove"
    );
    assert!(
        game.legal_moves().iter().any(
            |m| m.move_type() == MoveType::Capture && m.source_idx().ok() == Some(cap as usize)
        ),
        "no CaptureMove generated on the enemy capital"
    );

    let before = serde_json::to_value(&game.state).unwrap();
    let undo = game
        .simulate_move(&CaptureMove::new(cap))
        .expect("capture was refused");

    assert!(
        game.state.tribes[&2].cities.is_empty(),
        "the city did not change hands"
    );
    assert!(
        game.state.tribes[&2].units.is_empty(),
        "elimination left units behind"
    );
    assert_eq!(game.state.tribes[&2].killer_id, 1);
    assert_eq!(game.state.tribes[&2].killed_turn, turn);
    assert_eq!(game.state.tribes[&1].cities.len(), cities_before + 1);
    assert_eq!(game.state.tiles[&cap].owner, 1);
    assert_eq!(game.state.tiles[&stranded]._unit_owner_id, None);
    assert_eq!(game.state.tribes[&2].casualties, 1);
    // No killer unit is passed for the sweep, so the capturing tribe books no kills.
    assert_eq!(game.state.tribes[&1].kills, kills_before);

    undo(&mut game.state);
    assert_eq!(
        serde_json::to_value(&game.state).unwrap(),
        before,
        "capture-with-elimination undo did not round-trip"
    );
}
