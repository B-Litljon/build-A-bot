"""Throwaway M2M driver (2026-08-29): run the FULL production gate with
capacity params patched from outside, leaving src/ untouched.

Reads GATE_N_ESTIMATORS / GATE_NUM_LEAVES / GATE_MIN_CHILD; everything else
uses the standard retrain env vars. Patches BOTH Angel and Devil, matching
scripts/capacity_sweep.py's override semantics.
"""

import logging
import os
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "src")
logging.basicConfig(level=logging.INFO)

from core import retrainer as R  # noqa: E402

_N = int(os.environ["GATE_N_ESTIMATORS"])
_L = int(os.environ["GATE_NUM_LEAVES"])
_M = int(os.environ["GATE_MIN_CHILD"])

_orig = R.get_hyperparameters


def _patched(asset_class):
    angel, devil = _orig(asset_class)
    for params in (angel, devil):
        params.update(
            n_estimators=_N, num_leaves=_L, min_child_samples=_M
        )
    return angel, devil


R.get_hyperparameters = _patched

if __name__ == "__main__":
    raise SystemExit(R.main())
