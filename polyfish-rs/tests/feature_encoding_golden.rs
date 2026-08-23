//! Golden encoding checksum for `state_to_cpu_features`.
//!
//! `MAP_FINGERPRINT` pins the fixture map (mapgen / DotNetRandom). `GOLDEN_*`
//! pin what the encoder makes of it. Both moving means mapgen changed - re-bless
//! both. Only the goldens moving means the channel layout or an encoding rule
//! changed, which silently invalidates every checkpoint and every archived
//! `games_*.safetensors`.

use polyfish::ai::features::{
    CH_CITY_PRESENT, CH_MEM_ATTACKED_HERE, CH_MEM_ENEMY_ATTACK, CH_MEM_ENEMY_HP,
    CH_MEM_ENEMY_NAVAL, CH_MEM_ENEMY_RANGED, CH_MEM_ENEMY_SEEN, CH_RESOURCE_END, CH_RESOURCE_START,
    CH_STRUCTURE_END, CH_STRUCTURE_START, CH_TERRAIN_END, CH_TERRAIN_START, CH_TILE_IS_EXPLORED,
    CH_UNIT_END, CH_UNIT_OWNER, CH_UNIT_START, MAP_SIZE, NUM_CHANNELS, RawFeatures,
    state_to_cpu_features,
};
use polyfish::game::Game;
use polyfish::mapgen::{MapGenSettings, generate};
use polyfish::states::{GameState, MemUnit};
use polyfish::types::{MapSize, MapType, TribeType, UnitType};

const SEED: i64 = 20260823;

const MAP_FINGERPRINT: u64 = 0xb153_ca4d_a00f_4483;
const GOLDEN_CHANNELS: &[(usize, u64)] = &[
    (3, 0x8be89c74942feff8),
    (4, 0x3213f64aaf877a55),
    (5, 0x9925af7778d6f525),
    (12, 0x3439bb2183191208),
    (13, 0xddafc8628edb6918),
    (14, 0x8bc3d4a9369acf08),
    (15, 0x8bc3d4a9369acf08),
    (23, 0x0b16622657532a05),
    (28, 0xd95f795e3c970608),
    (29, 0x3d730400bcc6f605),
    (63, 0xd95f795e3c970608),
    (108, 0xd95f795e3c970608),
    (109, 0xd95f795e3c970608),
    (124, 0xd95f795e3c970608),
    (125, 0xd95f795e3c970608),
    (126, 0x8d4f37cff92b0417),
    (127, 0x0eb3fca6fed69cda),
    (128, 0xd95f795e3c970608),
    (133, 0x49e150ecd70f7288),
    (136, 0xc08594a4b1c6240c),
    (137, 0x5c8e1ecccf10bc48),
    (138, 0x773d8ddbb9bff314),
    (139, 0xb9bc9a3b1b76ccc8),
    (140, 0xf60451bfe10e9648),
    (141, 0x1fec64fa6dffd1a0),
];
const GOLDEN_PLAYER: &[f32] = &[
    0.06666667,
    1.0,
    0.16666667,
    0.05,
    0.02475,
    0.08,
    0.4117647,
    1.0,
    0.0,
    0.2,
    0.033333335,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
];

/// Self-contained FNV-1a: no dependency bump may move these digests.
struct Fnv(u64);

impl Fnv {
    fn new() -> Self {
        Fnv(0xcbf2_9ce4_8422_2325)
    }

    fn push(&mut self, bytes: &[u8]) {
        for b in bytes {
            self.0 ^= *b as u64;
            self.0 = self.0.wrapping_mul(0x100_0000_01b3);
        }
    }

    fn push_i64(&mut self, v: i64) {
        self.push(&v.to_le_bytes());
    }

    fn push_f32(&mut self, v: f32) {
        self.push(&v.to_bits().to_le_bytes());
    }
}

fn hash_f32s(vals: &[f32]) -> u64 {
    let mut h = Fnv::new();
    for v in vals {
        h.push_f32(*v);
    }
    h.0
}

/// Everything the encoder reads off the map, in `IndexMap` order.
fn map_fingerprint(state: &GameState) -> u64 {
    let mut h = Fnv::new();
    for (&idx, tile) in &state.tiles {
        h.push_i64(idx as i64);
        h.push_i64(tile.terrain_type as i64);
        h.push_i64(tile.owner as i64);
        h.push_i64(tile.capital_of as i64);
        h.push_i64(tile.has_road as i64);
        h.push_i64(tile.climate as i8 as i64);
        h.push_i64(
            state
                .structures
                .get(&idx)
                .and_then(|s| s.as_ref())
                .map(|s| s.structure_type as i64)
                .unwrap_or(-1),
        );
        h.push_i64(
            state
                .resources
                .get(&idx)
                .and_then(|r| r.as_ref())
                .map(|r| r.resource_type as i64)
                .unwrap_or(-1),
        );
    }
    for (&player_id, tribe) in &state.tribes {
        h.push_i64(player_id as i64);
        h.push_i64(tribe.tribe_type as i8 as i64);
        for unit in &tribe.units {
            h.push_i64(unit.unit_type as i64);
            h.push_i64(unit.coords.idx as i64);
            h.push_f32(unit.health);
        }
        for city in &tribe.cities {
            h.push_i64(city.idx as i64);
            h.push_i64(city.level as i64);
        }
    }
    h.0
}

/// Tiny is 11x11, i.e. the full `MAP_SIZE` grid, so no channel is padded away.
fn fixture() -> Game {
    let mut game = Game::new();
    game.state = generate(MapGenSettings {
        size: MapSize::Tiny,
        map_type: MapType::Drylands,
        tribes: vec![TribeType::Imperius, TribeType::Imperius],
        seed: SEED,
        symmetric: true,
        ..Default::default()
    });
    game.post_load();

    // Turn 0 leaves all six fog-memory channels empty; plant remembered contacts
    // so 136..142 are covered. Ages stay in {0,1,2}: `MEM_DECAY.powi(age)` is
    // then 1.0, one IEEE multiply, or two, and cannot drift across targets.
    game.state.settings.turn = 3;
    let tribe = game.state.tribes.get_mut(&1).unwrap();
    tribe.memory_units.insert(
        13,
        MemUnit {
            unit_type: UnitType::Rider,
            hp_norm: 0.5,
            last_seen_turn: 2,
        },
    );
    tribe.memory_units.insert(
        57,
        MemUnit {
            unit_type: UnitType::Archer,
            hp_norm: 0.75,
            last_seen_turn: 1,
        },
    );
    tribe.memory_units.insert(
        97,
        MemUnit {
            unit_type: UnitType::Rammership,
            hp_norm: 1.0,
            last_seen_turn: 3,
        },
    );
    tribe.memory_attacks.insert(40, 2);
    game
}

#[test]
fn feature_encoding_is_golden() {
    let game = fixture();
    assert_eq!(
        map_fingerprint(&game.state),
        MAP_FINGERPRINT,
        "the fixture map moved (mapgen/rng), not the encoder - re-bless both goldens"
    );

    let raw = state_to_cpu_features(&game.state, 1).unwrap();
    assert_eq!(raw.spatial.len(), NUM_CHANNELS * MAP_SIZE * MAP_SIZE);
    assert_eq!(raw.player.len(), RawFeatures::PLAYER_STATE_DIM);
    assert!(raw.spatial.iter().chain(&raw.player).all(|v| v.is_finite()));

    let hw = MAP_SIZE * MAP_SIZE;
    let actual: Vec<(usize, u64)> = (0..NUM_CHANNELS)
        .filter(|&c| raw.spatial[c * hw..(c + 1) * hw].iter().any(|v| *v != 0.0))
        .map(|c| (c, hash_f32s(&raw.spatial[c * hw..(c + 1) * hw])))
        .collect();
    if actual != GOLDEN_CHANNELS {
        // A shifted occupied-channel LIST is the signature of a mid-enum
        // insertion; the same list with different digests is an encoding rule.
        let first = actual
            .iter()
            .zip(GOLDEN_CHANNELS)
            .position(|(a, g)| a != g)
            .map(|i| {
                format!(
                    "entry {i}: golden {:?}, actual {:?}",
                    GOLDEN_CHANNELS[i], actual[i]
                )
            })
            .unwrap_or_else(|| format!("length {} != {}", actual.len(), GOLDEN_CHANNELS.len()));
        let lines: Vec<String> = actual
            .iter()
            .map(|(c, h)| format!("    ({c}, 0x{h:016x}),"))
            .collect();
        panic!(
            "channel encoding drifted ({first}).\nre-bless GOLDEN_CHANNELS with:\n{}",
            lines.join("\n")
        );
    }
    assert_eq!(
        raw.player.as_slice(),
        GOLDEN_PLAYER,
        "player-state vector drifted; actual {:?}",
        raw.player
    );

    // The goldens prove nothing about a block the fixture never populates.
    for ch in [
        CH_MEM_ENEMY_SEEN,
        CH_MEM_ENEMY_HP,
        CH_MEM_ENEMY_ATTACK,
        CH_MEM_ENEMY_RANGED,
        CH_MEM_ENEMY_NAVAL,
        CH_MEM_ATTACKED_HERE,
        CH_CITY_PRESENT,
        CH_UNIT_OWNER,
        CH_TILE_IS_EXPLORED,
    ] {
        assert!(
            actual.iter().any(|(c, _)| *c == ch),
            "fixture never sets channel {ch}"
        );
    }
    for block in [
        (CH_TERRAIN_START, CH_TERRAIN_END),
        (CH_RESOURCE_START, CH_RESOURCE_END),
        (CH_STRUCTURE_START, CH_STRUCTURE_END),
        (CH_UNIT_START, CH_UNIT_END),
    ] {
        assert!(
            actual.iter().any(|(c, _)| (block.0..block.1).contains(c)),
            "fixture never sets a channel in block {block:?}"
        );
    }
}
