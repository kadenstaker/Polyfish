#!/bin/bash
# Rust<->Python forward parity (audit T1): load the same model.safetensors into
# network.rs (candle, CPU) and train.py (PyTorch, CPU) and compare raw outputs.
#
# The two are separate implementations of one architecture that read and write
# the same file, and nothing enforced that they agree. This is the check that
# found the strided cross-attention bug — see the note in
# hypothesis_driven_improvements.md.
#
#   scripts/run_forward_parity.sh [model.safetensors]
#
# With no argument it uses model.safetensors if present, otherwise it builds a
# fresh one with init_model.py in a scratch dir (so a clean checkout can run it),
# and then compares two further fixtures derived from it. That matters because
# in CI the base is always a fresh init: every GroupNorm affine sits at identity
# and every bias at zero (init_model.py), so affine/bias mapping drift is
# numerically invisible there, and none of _migrate_checkpoint's branches runs.
# The `perturbed` fixture moves those values off their identity, the
# `legacy_migrated` one has taken the migration path. See make_parity_fixtures.py.
#
# An explicit model argument skips both: a hand-run against a real trained
# checkpoint stays a single fast comparison.
set -euo pipefail

cd "$(dirname "$0")/.."
REPO="$PWD"

if [ -x .venv/bin/python ]; then
    PY="$PWD/.venv/bin/python"
else
    PY=$(command -v python3)
fi

if ! "$PY" -c 'import torch' 2>/dev/null; then
    echo "forward parity needs torch (polyfish-rs/.venv, or see local_setup.sh)" >&2
    exit 1
fi

MODEL="${1:-}"
EXPLICIT=1
SCRATCH=""
if [ -z "$MODEL" ]; then
    EXPLICIT=0
    SCRATCH=$(mktemp -d)
    if [ -f model.safetensors ]; then
        MODEL=model.safetensors
    else
        echo "no model.safetensors — initialising one in $SCRATCH"
        # init_model.py imports PolyZeroNet from train.py and writes into cwd,
        # so run it from the scratch dir with this tree on PYTHONPATH.
        ( cd "$SCRATCH" && PYTHONPATH="$REPO" "$PY" "$REPO/init_model.py" >/dev/null )
        MODEL="$SCRATCH/model.safetensors"
    fi
fi

RUST_JSON=$(mktemp)
trap 'rm -f "$RUST_JSON"; [ -n "$SCRATCH" ] && rm -rf "$SCRATCH"' EXIT

FIXTURES=("$MODEL")
if [ "$EXPLICIT" -eq 0 ]; then
    # Command substitution, not a process substitution: `while read < <(cmd)`
    # would swallow a failure and quietly leave the run comparing the base only.
    EXTRA=$("$PY" scripts/make_parity_fixtures.py "$SCRATCH" "$MODEL")
    while IFS= read -r f; do
        if [ -n "$f" ]; then FIXTURES+=("$f"); fi
    done <<< "$EXTRA"
    if [ "${#FIXTURES[@]}" -ne 3 ]; then
        echo "expected 2 generated fixtures, got ${#FIXTURES[@]} entries" >&2
        exit 1
    fi
fi

for f in "${FIXTURES[@]}"; do
    echo
    echo "=== $(basename "$f")"
    cargo run --quiet --no-default-features --example py_parity -- "$f" > "$RUST_JSON"
    "$PY" scripts/py_parity.py "$RUST_JSON" "$f"
done
