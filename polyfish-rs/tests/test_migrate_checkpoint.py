#!/usr/bin/env python3
"""Tests for train.py's _migrate_checkpoint, and for the down-conversion that
scripts/make_parity_fixtures.py builds the legacy forward-parity fixture from.

Every checkpoint older than the current architecture takes this path on load,
and no test reached any of its branches: a fresh init_model.py checkpoint fires
none of them, so the migration only ever ran on real training data where a wrong
result looks like a bad run rather than a bug. The last case is the glue - it
fails if the fixture generator stops reaching a branch, which would otherwise
turn the CI forward-parity pass into a second copy of the base pass.

    python3 -m unittest discover -s tests -p 'test_*.py'
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

try:
    import torch

    import train

    import make_parity_fixtures

    HAVE_TORCH = True
except ImportError:  # pragma: no cover - exercised only on a torch-less runner
    HAVE_TORCH = False
    torch = train = make_parity_fixtures = None

PLAYER_STATE_DIM = 16
SPATIAL_CHANNELS = 142


@unittest.skipUnless(HAVE_TORCH, "train.py requires torch")
class MigrateCheckpointTest(unittest.TestCase):
    def setUp(self):
        self.model = train.PolyZeroNet(SPATIAL_CHANNELS, PLAYER_STATE_DIM, 11, 11)
        self.current = {k: v.clone() for k, v in self.model.state_dict().items()}

    def _migrate(self, state):
        return train._migrate_checkpoint(state, self.model, PLAYER_STATE_DIM)

    @staticmethod
    def _fired(migrations, marker):
        return [m for m in migrations if marker in m]

    def test_a_current_checkpoint_is_left_alone(self):
        out, migrations = self._migrate(dict(self.current))
        self.assertEqual(migrations, [])
        for key, value in self.current.items():
            self.assertTrue(torch.equal(out[key], value), key)

    def test_the_old_value_head_is_dropped_and_the_new_one_seeded(self):
        legacy = {k: v for k, v in self.current.items() if not k.startswith(("v_fc1.", "v_fc2."))}
        legacy["v_pool_conv.weight"] = torch.zeros(1, 64, 1, 1)
        legacy["v_fc_shared.weight"] = torch.zeros(64, 121)

        out, migrations = self._migrate(legacy)
        self.assertTrue(self._fired(migrations, "value-head"), migrations)
        self.assertNotIn("v_pool_conv.weight", out)
        self.assertNotIn("v_fc_shared.weight", out)
        self.assertEqual(out["v_fc1.weight"].shape, self.current["v_fc1.weight"].shape)
        self.assertEqual(out["v_fc2.weight"].shape, self.current["v_fc2.weight"].shape)

    def test_narrow_player_embeddings_are_widened_keeping_their_rows(self):
        legacy = dict(self.current)
        for name in ("player_pos_embeddings", "player_feature_embeddings"):
            legacy[name] = self.current[name][:12].clone()

        out, migrations = self._migrate(legacy)
        for name in ("player_pos_embeddings", "player_feature_embeddings"):
            self.assertTrue(self._fired(migrations, name), migrations)
            self.assertEqual(out[name].shape[0], PLAYER_STATE_DIM)
            self.assertTrue(torch.equal(out[name][:12], self.current[name][:12]), name)

    def test_a_missing_progress_head_is_created(self):
        legacy = {k: v for k, v in self.current.items() if not k.startswith("v_progress.")}

        out, migrations = self._migrate(legacy)
        self.assertTrue(self._fired(migrations, "v_progress"), migrations)
        self.assertEqual(out["v_progress.weight"].shape, self.current["v_progress.weight"].shape)
        self.assertEqual(out["v_progress.bias"].shape, self.current["v_progress.bias"].shape)

    def test_conv1_is_zero_padded_on_the_fog_memory_channels(self):
        legacy = dict(self.current)
        legacy["conv1.weight"] = self.current["conv1.weight"][:, :136].clone()

        out, migrations = self._migrate(legacy)
        self.assertTrue(self._fired(migrations, "conv1.weight"), migrations)
        self.assertEqual(out["conv1.weight"].shape[1], SPATIAL_CHANNELS)
        # Padding at the END is what makes the old channels still mean what they
        # meant; padding at the front would silently relabel all 136.
        self.assertTrue(
            torch.equal(out["conv1.weight"][:, :136], self.current["conv1.weight"][:, :136])
        )
        self.assertEqual(float(out["conv1.weight"][:, 136:].abs().sum()), 0.0)

    def test_the_action_head_is_padded_to_the_current_width(self):
        legacy = dict(self.current)
        legacy["pi_action.weight"] = self.current["pi_action.weight"][:11].clone()
        legacy["pi_action.bias"] = self.current["pi_action.bias"][:11].clone()

        out, migrations = self._migrate(legacy)
        self.assertTrue(self._fired(migrations, "pi_action"), migrations)
        self.assertEqual(out["pi_action.weight"].shape[0], train.NUM_ACTION_TYPES)
        self.assertEqual(out["pi_action.bias"].shape[0], train.NUM_ACTION_TYPES)
        self.assertTrue(
            torch.equal(out["pi_action.weight"][:11], self.current["pi_action.weight"][:11])
        )
        self.assertEqual(float(out["pi_action.weight"][11:].abs().sum()), 0.0)

    def test_the_parity_fixture_down_conversion_reaches_every_branch(self):
        legacy = make_parity_fixtures.downconvert_to_legacy(self.current)
        _, migrations = self._migrate(legacy)
        unfired = [
            b
            for b in make_parity_fixtures.MIGRATION_BRANCHES
            if not self._fired(migrations, b)
        ]
        self.assertEqual(
            unfired, [],
            "scripts/make_parity_fixtures.py no longer reaches these _migrate_checkpoint "
            f"branches, so the CI legacy parity pass is testing less than it claims: {migrations}",
        )


if __name__ == "__main__":
    unittest.main()
