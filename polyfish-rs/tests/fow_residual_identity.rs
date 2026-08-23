//! `clone_for_mcts` is the single choke point that keeps hidden information out of
//! the search. These pin the tile-level invariant: a tile the POV has never explored
//! carries no field that still names its owner, its capital, or its road history.

use polyfish::coords::Coords;
use polyfish::game::Game;
use polyfish::states::{GameState, TileState, TribeState};
use polyfish::types::{TerrainType, TribeType};

const SIZE: i32 = 11;

fn two_tribe_state() -> GameState {
    let mut state = GameState::default();
    state.settings._fow = true;
    state.settings.size = SIZE;

    state.tribes.insert(
        1,
        TribeState {
            id: 1,
            tribe_type: TribeType::Imperius,
            ..Default::default()
        },
    );
    state.tribes.insert(
        2,
        TribeState {
            id: 2,
            tribe_type: TribeType::Bardur,
            ..Default::default()
        },
    );

    // Tile 0: player 2's capital centre, never explored by player 1.
    state.tiles.insert(
        0,
        TileState {
            coords: Coords::from_index(0, SIZE),
            terrain_type: TerrainType::Mountain,
            owner: 2,
            capital_of: 2,
            climate: TribeType::Bardur,
            ruling_city_coords: Some(Coords::from_index(0, SIZE)),
            had_route: true,
            skin_type: 3,
            explorers: [2].into_iter().collect(),
            ..Default::default()
        },
    );

    // Tile 1: player 1's own capital centre, explored.
    state.tiles.insert(
        1,
        TileState {
            coords: Coords::from_index(1, SIZE),
            terrain_type: TerrainType::Field,
            owner: 1,
            capital_of: 1,
            climate: TribeType::Imperius,
            ruling_city_coords: Some(Coords::from_index(1, SIZE)),
            had_route: true,
            skin_type: 2,
            explorers: [1].into_iter().collect(),
            ..Default::default()
        },
    );

    state
}

fn mcts_view() -> Game {
    Game {
        state: two_tribe_state(),
    }
    .clone_for_mcts(1)
}

#[test]
fn hidden_tiles_carry_no_residual_identity() {
    let view = mcts_view();
    let h = view.state.tiles.get(&0).unwrap();

    assert_eq!(h.capital_of, 0, "hidden tile leaks capital_of");
    assert_eq!(h.climate, TribeType::Nature, "hidden tile leaks climate");
    assert!(
        h.ruling_city_coords.is_none(),
        "hidden tile leaks ruling_city_coords"
    );
    assert!(!h.had_route, "hidden tile leaks had_route");
    assert_eq!(h.skin_type, 0, "hidden tile leaks skin_type");
}

#[test]
fn hidden_enemy_capital_no_longer_reads_as_a_city() {
    let view = mcts_view();

    // `is_city` reads only `ruling_city_coords`, so before the clear a hidden enemy
    // centre was saved by its callers' owner gate rather than by the state itself.
    assert!(
        !polyfish::functions::is_city(&view.state, 0),
        "hidden enemy city centre still reads as a city"
    );
    assert!(polyfish::functions::is_city(&view.state, 1));
}

#[test]
fn explored_tiles_keep_their_identity() {
    let view = mcts_view();
    let s = view.state.tiles.get(&1).unwrap();

    assert_eq!(s.capital_of, 1);
    assert_eq!(s.climate, TribeType::Imperius);
    assert_eq!(s.ruling_city_coords.as_ref().map(|c| c.idx), Some(1));
    assert!(s.had_route);
    assert_eq!(s.skin_type, 2);
    assert_eq!(s.owner, 1);
    assert_eq!(s.terrain_type, TerrainType::Field);
}

#[test]
fn fow_disabled_leaves_every_tile_untouched() {
    let mut state = two_tribe_state();
    state.settings._fow = false;
    let view = Game { state }.clone_for_mcts(1);
    let h = view.state.tiles.get(&0).unwrap();

    assert_eq!(h.capital_of, 2);
    assert_eq!(h.climate, TribeType::Bardur);
    assert!(h.ruling_city_coords.is_some());
    assert!(h.had_route);
    assert_eq!(h.skin_type, 3);
}
