//! The opening book supplies data, not a veto. It used to hard-force a
//! uniform-random pick with a fabricated one-hot policy target (`mcts_zero`)
//! and to replace every node's untried set (`heuristic_mcts`). These pin the
//! forcing out: at a position where the book IS non-empty, both agents must
//! still search the whole legal set.

use candle_core::Device;
use candle_nn::VarMap;
use polyfish::ai::book::Book;
use polyfish::ai::eval_server::{Evaluator, InlineEvalHandle};
use polyfish::ai::heuristic_mcts::HeuristicMctsAgent;
use polyfish::ai::mcts_zero::ZeroMctsAgent;
use polyfish::ai::network::PolyZeroNet;
use polyfish::game::Game;
use polyfish::mapgen::{MapGenSettings, generate};
use polyfish::states::ResourceState;
use polyfish::types::{MapSize, ResourceType, TerrainType, TribeType};
use std::sync::Arc;

/// Imperius on turn 0 with a Fruit under its starting unit, so the book's
/// Harvest/Step/Reward line matches at least one legal move. Mirrors the
/// setup `tests/verify_opening_book.rs` uses.
fn book_position() -> Game {
    let settings = MapGenSettings {
        size: MapSize::Tiny,
        tribes: vec![TribeType::Imperius, TribeType::Oumaji],
        seed: 123,
        ..Default::default()
    };

    let mut game = Game::new();
    game.state = generate(settings);
    game.post_load();

    let imp_id = game
        .state
        .tribes
        .iter()
        .find(|(_, t)| t.tribe_type == TribeType::Imperius)
        .map(|(&id, _)| id)
        .unwrap();

    game.state.settings.turn = 0;
    game.state.settings.current_player_turn_id = imp_id;
    game.state.tribes.get_mut(&imp_id).unwrap().stars = 5;

    let unit_pos = game.state.tribes.get(&imp_id).unwrap().units[0].coords;
    let unit_tile_idx = game
        .state
        .tiles
        .iter()
        .find(|(_, t)| t.coords == unit_pos)
        .map(|(&i, _)| i)
        .unwrap();
    game.state.resources.insert(
        unit_tile_idx,
        Some(ResourceState {
            resource_type: ResourceType::Fruit,
        }),
    );
    game.state
        .tiles
        .get_mut(&unit_tile_idx)
        .unwrap()
        .terrain_type = TerrainType::Field;

    game
}

#[test]
fn heuristic_root_is_not_restricted_to_the_book() {
    let mut game = book_position();

    let book_len = Book::recommend(&game).len();
    assert!(
        book_len > 0,
        "setup is only meaningful while the book fires here"
    );
    let legal_len = game.legal_moves().len();
    assert!(
        legal_len > book_len,
        "position must offer moves the book does not, got {legal_len} legal vs {book_len} book"
    );

    let agent = HeuristicMctsAgent::with_exploration(legal_len + 4, 0.4).with_search_seed(20260823);
    let (_best, analysis) = agent.select_move_with_analysis(&mut game);

    assert!(
        analysis.evaluations.len() > book_len,
        "root searched {} moves; the book recommends {book_len}, and forcing it back \
         would cap the root at that",
        analysis.evaluations.len()
    );
}

#[test]
fn zero_policy_target_is_search_visits_not_a_book_one_hot() {
    let mut game = book_position();
    assert!(
        !Book::recommend(&game).is_empty(),
        "setup is only meaningful while the book fires here"
    );

    let varmap = VarMap::new();
    let vs = candle_nn::VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &Device::Cpu);
    let network = Arc::new(PolyZeroNet::new(vs).unwrap());
    let evaluator = Evaluator::Inline(InlineEvalHandle::new(network));

    let agent = ZeroMctsAgent::new(&evaluator, 8).with_search_seed(20260823);
    let (best, visits) = agent.select_move_with_decomposed_visits(&mut game, 0);

    assert!(best.is_some(), "search must return a move");
    assert!(
        visits.len() > 1,
        "the book path emitted exactly one MoveVisit carrying the full iteration \
         count as a fabricated target; got {} entries",
        visits.len()
    );
}
