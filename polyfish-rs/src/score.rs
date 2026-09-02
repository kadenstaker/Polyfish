//! The canonical score definition, and the per-component breakdown the
//! incremental mutations in `actions/` have to agree with (#40).

use crate::settings::structures::get_structure_setting;
use crate::states::{CityState, GameState, PlayerId, StructureState};
use crate::types::{StructureType, TribeType};

pub const CITY_BASE_SCORE: i32 = 100;
pub const CITY_LEVEL_UP_SCORE: i32 = 50;
pub const CITY_POPULATION_SCORE: i32 = 5;
pub const CITY_TERRITORY_SCORE: i32 = 20;
pub const PARK_SCORE: i32 = 250;
pub const TEMPLE_LEVEL_SCORE: i32 = 100;
pub const EXPLORED_TILE_SCORE: i32 = 5;
pub const UNIT_COST_SCORE: i32 = 5;
pub const TECH_TIER_SCORE: i32 = 100;

pub fn is_temple(structure_type: StructureType) -> bool {
    matches!(
        structure_type,
        StructureType::Temple
            | StructureType::WaterTemple
            | StructureType::ForestTemple
            | StructureType::MountainTemple
            | StructureType::IceTemple
    )
}

/// What a structure is worth to whoever owns the tile it stands on. The one
/// definition both the canonical recompute and every incremental mutation
/// read, so a build/destroy/capture cannot price it differently.
pub fn structure_score(structure: &StructureState) -> i32 {
    if is_temple(structure.structure_type) {
        structure.level * TEMPLE_LEVEL_SCORE
    } else {
        get_structure_setting(structure.structure_type).reward_score
    }
}

/// Score a unit contributes: its own cost plus its passenger's, at 5/star.
/// Converted units are worth nothing to their new owner.
pub fn unit_score(unit: &crate::states::UnitState) -> i32 {
    if unit.converted {
        return 0;
    }
    let cost = crate::settings::units::get_unit_setting(unit.unit_type).cost
        + unit
            .passenger_type
            .map(|p| crate::settings::units::get_unit_setting(p).cost)
            .unwrap_or(0);
    cost * UNIT_COST_SCORE
}

/// What a city is worth to its owner on its own: the city, its population and
/// its parks. Territory and the structures on it are priced per *tile*, by
/// ownership (see `breakdown`), so they are not part of this - a capture moves
/// them through the territory claim that follows it.
pub fn city_core_score(city: &CityState) -> i32 {
    let mut b = ScoreBreakdown::default();
    add_city_core(city, &mut b);
    b.total()
}

/// What one tile is worth to whoever owns it: the territory itself plus
/// anything standing on it.
pub fn tile_score(state: &GameState, idx: i32) -> i32 {
    CITY_TERRITORY_SCORE
        + crate::functions::get_structure_at(state, idx)
            .map(structure_score)
            .unwrap_or(0)
}

/// Canonical score split by source. `total()` is `calculate_detailed_tribe_score`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ScoreBreakdown {
    pub cities: i32,
    pub territory: i32,
    pub structures: i32,
    pub parks: i32,
    pub population: i32,
    pub explored: i32,
    pub units: i32,
    pub tech: i32,
}

impl ScoreBreakdown {
    pub fn total(&self) -> i32 {
        self.cities
            + self.territory
            + self.structures
            + self.parks
            + self.population
            + self.explored
            + self.units
            + self.tech
    }

    /// Component-wise difference, for attributing a drift to its source.
    pub fn diff(&self, other: &ScoreBreakdown) -> Vec<(&'static str, i32)> {
        [
            ("cities", self.cities - other.cities),
            ("territory", self.territory - other.territory),
            ("structures", self.structures - other.structures),
            ("parks", self.parks - other.parks),
            ("population", self.population - other.population),
            ("explored", self.explored - other.explored),
            ("units", self.units - other.units),
            ("tech", self.tech - other.tech),
        ]
        .into_iter()
        .filter(|(_, d)| *d != 0)
        .collect()
    }
}

/// The canonical score, by component, as per Polytopia rules.
pub fn breakdown(state: &GameState, player_id: PlayerId) -> ScoreBreakdown {
    let tribe = match state.tribes.get(&player_id) {
        Some(t) => t,
        None => return ScoreBreakdown::default(),
    };

    let mut b = ScoreBreakdown::default();

    for city in &tribe.cities {
        add_city_core(city, &mut b);
    }

    // Territory is priced off `tile.owner`, the one bit `claim_territory`
    // scores transitions of. City `_territory` lists are not consulted: cities
    // two tiles apart share entries, a stolen tile stays listed in the city
    // that lost it, and an eliminated tribe keeps tiles no city lists (#40).
    for (&idx, tile) in &state.tiles {
        if tile.owner != player_id {
            continue;
        }
        b.territory += CITY_TERRITORY_SCORE;
        if let Some(structure) = crate::functions::get_structure_at(state, idx) {
            b.structures += structure_score(structure);
        }
    }

    // Luxidoor starts with a level 3 capital, whose extra population was never earned.
    if tribe.tribe_type == TribeType::Luxidoor {
        b.population -= 5 * CITY_POPULATION_SCORE;
    }

    b.explored = state
        .tiles
        .values()
        .filter(|t| t.explorers.contains(&player_id))
        .count() as i32
        * EXPLORED_TILE_SCORE;

    for unit in &tribe.units {
        b.units += unit_score(unit);
    }

    for tech in &tribe.tech_vanilla {
        let tier = crate::settings::technology::get_technology_setting(tech.tech_type)
            .tier
            .unwrap_or(1);
        b.tech += TECH_TIER_SCORE * tier;
    }

    b
}

fn add_city_core(city: &CityState, b: &mut ScoreBreakdown) {
    if city.level >= 1 {
        b.cities += CITY_BASE_SCORE + (city.level - 1) * CITY_LEVEL_UP_SCORE;
    }

    // Each Park reward is worth 250 - level 5+ can be spent on Park again.
    b.parks += city
        .rewards
        .iter()
        .filter(|r| **r == crate::types::CityRewardType::Park)
        .count() as i32
        * PARK_SCORE;

    b.population += city.population * CITY_POPULATION_SCORE;
}
