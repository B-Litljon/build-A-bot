"""Throwaway M2M driver (2026-08-29): run the FULL production gate with
capacity params patched from outside, leaving src/ untouched.

Shared overrides: GATE_N_ESTIMATORS / GATE_NUM_LEAVES / GATE_MIN_CHILD /
GATE_MAX_DEPTH apply to BOTH models, matching scripts/capacity_sweep.py's
override semantics. Per-stage overrides GATE_ANGEL_* / GATE_DEVIL_* (same
suffixes) take precedence over the shared value for that model, so
asymmetric historical configs (e.g. the shipped Angel-63-leaves/depth-10 vs
Devil-31/8) can be reproduced. Everything else uses the standard retrain
env vars. Note the Devil's min_child_samples is further auto-scaled to its
approved population inside refit_models (capped at whatever is set here).
"""

import logging
import os
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "src")
logging.basicConfig(level=logging.INFO)

from core import retrainer as R  # noqa: E402

_ENV_KEYS = {
    "GATE_N_ESTIMATORS": "n_estimators",
    "GATE_NUM_LEAVES": "num_leaves",
    "GATE_MIN_CHILD": "min_child_samples",
    "GATE_MAX_DEPTH": "max_depth",
}


def _overrides(stage: str) -> dict:
    """Shared GATE_* values, overridden by GATE_{stage}_* when present."""
    out = {}
    for env_name, param in _ENV_KEYS.items():
        shared = os.environ.get(env_name, "").strip()
        specific = os.environ.get(f"GATE_{stage}_{env_name[5:]}", "").strip()
        value = specific or shared
        if value:
            out[param] = int(value)
    return out


_orig = R.get_hyperparameters


def _patched(asset_class):
    angel, devil = _orig(asset_class)
    angel.update(_overrides("ANGEL"))
    devil.update(_overrides("DEVIL"))
    return angel, devil


R.get_hyperparameters = _patched

if __name__ == "__main__":
    raise SystemExit(R.main())
