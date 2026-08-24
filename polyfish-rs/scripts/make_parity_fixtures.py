#!/usr/bin/env python3
"""Build the extra checkpoints scripts/run_forward_parity.sh compares on.

In CI the forward-parity check only ever sees a fresh init_model.py checkpoint,
and that checkpoint hides two whole classes of drift between network.rs and
train.py: init_model.py sets every GroupNorm weight to 1 and every GroupNorm,
Conv2d and Linear bias to 0, so a norm-affine or bias that one side applies in
the wrong place — or not at all — is numerically invisible. None of
_migrate_checkpoint's five branches executes on it either, so the migration path
that every legacy checkpoint takes is never exercised.

Two fixtures close that:
  perturbed.safetensors        every norm weight and every bias moved off its
                               identity value by a seeded randn
  legacy_migrated.safetensors  BASE_MODEL down-converted to the pre-migration
                               format, then run through train._migrate_checkpoint

The migration is applied and saved HERE, not left to py_parity.py: the Rust side
loads the file straight through candle's VarBuilder with no migration path, so a
raw legacy file would fail the load rather than test anything. Migrating once and
comparing both implementations against the same migrated file is what makes the
branches observable and asserts the migrated result is Rust-loadable.

    make_parity_fixtures.py OUT_DIR BASE_MODEL

Prints the fixtures it wrote, one path per line.
"""
import contextlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

# Fixed so a parity failure reproduces. Small on purpose: py_parity.py compares
# raw logits at an absolute 1e-3, so a large perturbation would turn a port
# check into a float-noise check.
SEED = 20260823
SCALE = 0.05

# One marker per message _migrate_checkpoint emits (the embedding branch emits
# one per tensor). An unfired marker means the down-conversion missed a branch.
MIGRATION_BRANCHES = (
    "value-head",
    "player_pos_embeddings",
    "player_feature_embeddings",
    "v_progress",
    "conv1.weight",
    "pi_action",
)

LEGACY_PLAYER_DIM = 12
LEGACY_SPATIAL_CHANNELS = 136
LEGACY_ACTION_TYPES = 11


def is_norm_weight(name, tensor, keys):
    """Norm affines are the only 1-D `.weight` tensors; conv/linear are 2-D+."""
    if not name.endswith(".weight") or tensor.dim() != 1:
        return False
    return name[: -len("weight")] + "bias" in keys


def is_bias(name):
    # Not `.bias`: nn.MultiheadAttention names its packed bias `in_proj_bias`,
    # and that one is exactly the kind of mapping that can drift silently.
    return name.endswith("bias")


def perturb(state):
    """Move every norm weight off 1 and every bias off 0."""
    gen = torch.Generator().manual_seed(SEED)
    keys = set(state)
    out = {}
    touched = 0
    for name, tensor in state.items():
        if not tensor.is_floating_point():
            out[name] = tensor
            continue
        noise = torch.randn(tensor.shape, generator=gen, dtype=torch.float32) * SCALE
        if is_bias(name):
            out[name] = noise.to(tensor.dtype)
            touched += 1
        elif is_norm_weight(name, tensor, keys):
            out[name] = (1.0 + noise).to(tensor.dtype)
            touched += 1
        else:
            out[name] = tensor
    if touched == 0:
        sys.exit("FAIL: perturbed nothing — the norm/bias naming convention moved")
    return out, touched


def downconvert_to_legacy(state):
    """The pre-migration shape of a checkpoint: one edit per _migrate_checkpoint branch."""
    legacy = {k: v.clone() for k, v in state.items()}

    # 1. old value head: v_pool_conv / v_fc_shared instead of v_fc1 / v_fc2
    for key in [k for k in legacy if k.startswith(("v_fc1.", "v_fc2."))]:
        del legacy[key]
    legacy["v_pool_conv.weight"] = torch.zeros(1, 64, 1, 1)
    legacy["v_fc_shared.weight"] = torch.zeros(64, 121)

    # 2. narrower player-state embeddings
    for name in ("player_pos_embeddings", "player_feature_embeddings"):
        legacy[name] = legacy[name][:LEGACY_PLAYER_DIM].clone()

    # 3. no v_progress head
    legacy.pop("v_progress.weight", None)
    legacy.pop("v_progress.bias", None)

    # 4. pre-fog-memory conv1 input width
    legacy["conv1.weight"] = legacy["conv1.weight"][:, :LEGACY_SPATIAL_CHANNELS].clone()

    # 5. action_type head before Resign widened it
    legacy["pi_action.weight"] = legacy["pi_action.weight"][:LEGACY_ACTION_TYPES].clone()
    legacy["pi_action.bias"] = legacy["pi_action.bias"][:LEGACY_ACTION_TYPES].clone()
    return legacy


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: make_parity_fixtures.py OUT_DIR BASE_MODEL")
    out_dir, base_path = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)

    # train.py prints a device banner on import; stdout here is the path list.
    with contextlib.redirect_stdout(sys.stderr):
        import train

    base = {k: v.float() for k, v in load_file(base_path).items()}

    perturbed_path = os.path.join(out_dir, "perturbed.safetensors")
    perturbed, touched = perturb(base)
    save_file(perturbed, perturbed_path)

    torch.manual_seed(SEED)  # the migration seeds fresh heads from a fresh model
    model = train.PolyZeroNet(142, 16, 11, 11)
    migrated, migrations = train._migrate_checkpoint(downconvert_to_legacy(base), model, 16)
    unfired = [b for b in MIGRATION_BRANCHES if not any(b in m for m in migrations)]
    if unfired:
        sys.exit(f"FAIL: _migrate_checkpoint branches {unfired} did not fire: {migrations}")
    legacy_path = os.path.join(out_dir, "legacy_migrated.safetensors")
    save_file({k: v.contiguous() for k, v in migrated.items()}, legacy_path)

    print(f"perturbed {touched} norm weights/biases", file=sys.stderr)
    print(perturbed_path)
    print(legacy_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
