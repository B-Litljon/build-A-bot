"""FeatureSpec -- the frozen definition of one feature-lab experiment.

An experiment is "what data, which features, which labels, which bracket".
One immutable, hashable object carries all of it, and its content hash is the
frame-cache key, so changing a single lookback is automatically a fresh frame
and never a stale hit. See src/lab/README.md for the loop.

Glossary:
    DEFAULT_TRADEABLE_6 -- the six OANDA-tradeable fiat crosses (metals are
        deliberately excluded: this practice account rejects them, which is why
        UNTRADEABLE_SYMBOLS excludes XAU/XAG from gate scoring).
    GeometrySpec -- the ATR bracket (stop multiple, target multiple, max hold
        bars) used for BOTH label construction and gate/backtest evaluation.
        One object on purpose: the plan originally split label and evaluation
        geometry, but a silent disagreement between them would make the gate
        score a bracket the labels were never built for.
    LabelSpec -- the label vocabulary knobs the retrainer exposes without an
        environment change: survival_bars, the Angel's independent momentum
        multiple (None = the asset class's production default), and kind
        ("survival" | "macro") selecting RETRAIN_DEVIL_LABEL's value for the
        gate run. Targets are otherwise deliberately fixed in v1.
    GateConfig -- model_family (the retrainer's MODEL_FAMILY seam; it is read
        at retrainer import time, so the lab validates the spec against the
        loaded value rather than patching it) and n_folds.
    FeatureSpec -- the whole experiment: data window, feature families,
        geometry, labels, cost-table switch, estimator family.
    content_hash -- sha256 of the resolved spec plus the environment that
        affects frame contents: spread-table bytes, risk-profile gate values,
        behavior-veto env, every registered family's (name, version), and the
        resolved STATE of any extra generator (constructor args included —
        two instances of one class with different args hash differently, and
        a generator whose state is not JSON-serializable raises rather than
        degrading to a class-name-only hash). Sixteen hex chars; the frame
        cache key. model_family is deliberately EXCLUDED — it changes the
        estimator, not the frame (it belongs to run provenance).
    family_versions -- the (name, version) pairs hashed into content_hash,
        resolved through the registry; an unregistered family raises.
    generator_state -- an extra generator's resolved __dict__, JSON-serialized
        deterministically for hashing; raises on a non-serializable value.
    asset_class -- "forex" selects the 2.0x/4.0x profile and the class's Angel
        multiple; only forex is exercised by the seeds today.
    htf_timeframe -- the higher-timeframe context bar size, which MUST match
        run_oanda.py's _GRANULARITY_PROFILES pairing for the spec's bar size.
        The M15 pairing is "1h"; a mismatch is train/serve skew in the htf_*
        columns.
    use_spread_table / spread_table_path -- whether the per-instrument cost
        table is active (adds cost_ratio, applies measured alphas to the veto).
    extra_generators -- escape hatch for one-off generators that are not
        registered; each must expose a ``feature_cols`` attribute naming the
        columns it adds.
    kind -- see LabelSpec.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ml.core.interfaces import BaseFeatureGenerator

# The six fiat crosses this account can trade — mirrors the gate's tradeable
# universe (config/spread_alphas_m15.json lists exactly these). Metals stay out:
# the practice account rejects XAU/XAG orders (INSTRUMENT_NOT_TRADEABLE).
DEFAULT_TRADEABLE_6: Tuple[str, ...] = (
    "AUD_JPY",
    "EUR_JPY",
    "GBP_JPY",
    "NZD_JPY",
    "GBP_AUD",
    "GBP_NZD",
)

_SPEC_SCHEMA_VERSION = 2

# Spec fields that are RUN PROVENANCE, not frame content: model_family
# selects the estimator and n_folds shapes the walk-forward, but neither
# changes a single row of the built frame. Excluded so W4's estimator A/B
# reuses the same cached frame by construction.
_HASH_EXCLUDED_FIELDS = frozenset({"gate"})

# Frame contents that come from the environment rather than the spec. Hashed
# explicitly so a veto toggle invalidates cached frames.
_FRAME_ENV_KEYS = (
    "RETRAIN_BEHAVIOR_VETO",
    "RISK_CHOP_FILTER_ENABLED",
    "RISK_SPREAD_GATE_ENABLED",
    "RISK_REGIME_GATE_ENABLED",
    "RISK_TIME_GATE_ENABLED",
    "RISK_BLACKOUT_ET",
    "RISK_SPREAD_K",
    "RISK_SPREAD_K_COUPLING",
    "RISK_COUPLING_MODE",
    "RISK_REGIME_PCTILE",
    "RISK_REGIME_WINDOW",
    "RISK_REGIME_MIN_SAMPLES",
    "RISK_SPREAD_ATR_ALPHA",
)


@dataclass(frozen=True)
class GeometrySpec:
    """ATR bracket used for labels and evaluation (2.0x/4.0x/45 = the served one)."""

    sl_mult: float = 2.0
    tp_mult: float = 4.0
    max_hold: int = 45


@dataclass(frozen=True)
class LabelSpec:
    """The retrainer label knobs the lab may vary without touching code."""

    survival_bars: int = 5
    # None -> the asset class's production Angel multiple (forex: 1.0), NOT
    # sl_mult; a float pins it explicitly. See _ANGEL_ATR_MULT_BY_CLASS.
    angel_mult: Optional[float] = None
    # Which column the Devil trains/scores on: "survival" (devil_target) or
    # "macro" (devil_target_macro). Applied around the gate run via
    # RETRAIN_DEVIL_LABEL, which the retrainer reads per call.
    kind: str = "survival"


@dataclass(frozen=True)
class GateConfig:
    """Estimator family and fold count for the validation gate."""

    model_family: str = "lightgbm"
    n_folds: int = 3


@dataclass(frozen=True)
class FeatureSpec:
    """One feature-lab experiment. Frozen: build a new one to change anything.

    Frame-cache contract (W1, 2026-09-22): ``content_hash`` covers the
    registered families' (name, version) pairs and every extra generator's
    RESOLVED state — editing a lookback constant inside a registered generator
    requires bumping that family's registration version (registration without
    a version raises), and a configured extra generator hashes its constructor
    args. ``gate`` is excluded: it is run provenance, not frame content.
    """

    name: str = "v3_base_control"
    symbols: Tuple[str, ...] = DEFAULT_TRADEABLE_6
    granularity: int = 15
    days_back: int = 730
    feature_sets: Tuple[str, ...] = ("v3_base",)
    geometry: GeometrySpec = field(default_factory=GeometrySpec)
    label: LabelSpec = field(default_factory=LabelSpec)
    gate: GateConfig = field(default_factory=GateConfig)
    use_spread_table: bool = False
    spread_table_path: str = "config/spread_alphas_m15.json"
    asset_class: str = "forex"
    htf_timeframe: str = "1h"
    extra_generators: Tuple[BaseFeatureGenerator, ...] = ()

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("FeatureSpec.name must be a non-empty slug")
        if not self.symbols:
            raise ValueError("FeatureSpec.symbols must not be empty")
        if not self.feature_sets and not self.extra_generators:
            raise ValueError(
                "FeatureSpec needs at least one feature family or extra generator"
            )
        if self.granularity <= 0 or self.days_back <= 0:
            raise ValueError("granularity and days_back must be positive")
        if self.label.kind not in ("survival", "macro"):
            raise ValueError(
                f"label.kind must be 'survival' or 'macro', got {self.label.kind!r}"
            )
        for gen in self.extra_generators:
            if not getattr(gen, "feature_cols", None):
                raise ValueError(
                    f"extra generator {type(gen).__name__} must expose a "
                    "'feature_cols' attribute naming the columns it adds"
                )

    def alpha_table(self) -> Optional[Dict[str, float]]:
        """Resolve the spread-cost table (None when the switch is off)."""
        if not self.use_spread_table:
            return None
        from core.retrainer._common import _load_spread_table

        return _load_spread_table(self.spread_table_path)["alphas"]

    def content_hash(self) -> str:
        """Cache key for a built frame: the resolved spec + frame-affecting env.

        Covers generator state explicitly: every registered family hashes as
        (name, version), and every extra generator hashes its resolved
        __dict__ alongside its class id. A generator whose state cannot be
        JSON-serialized RAISES here — a class-name-only fallback would silently
        reuse a frame built by a different configuration.
        """
        table_sha = None
        if self.use_spread_table:
            path = Path(self.spread_table_path)
            table_sha = (
                hashlib.sha256(path.read_bytes()).hexdigest()[:16]
                if path.is_file()
                else "missing"
            )
        payload = {
            "schema": _SPEC_SCHEMA_VERSION,
            "spec": {
                f.name: (
                    _family_version_entry(getattr(self, f.name))
                    if f.name == "feature_sets"
                    else _jsonable(getattr(self, f.name))
                )
                for f in fields(self)
                if f.name not in _HASH_EXCLUDED_FIELDS
            },
            "spread_table_sha256": table_sha,
            "risk_profile": _risk_profile_fingerprint(self.asset_class),
            "frame_env": {k: os.getenv(k, "") for k in _FRAME_ENV_KEYS},
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def family_versions(self) -> Tuple[Tuple[str, int], ...]:
        """(name, version) for every family in feature_sets, registry-resolved."""
        from lab.registry import family_version

        return tuple(
            (name, family_version(name)) for name in self.feature_sets
        )


def _generator_id(gen: BaseFeatureGenerator) -> str:
    cls = type(gen)
    return f"{cls.__module__}.{cls.__qualname__}"


def _generator_state(gen: BaseFeatureGenerator) -> dict:
    """An extra generator's resolved state, JSON-ready and deterministically
    ordered. Raises on anything not JSON-serializable — the frame cache key
    must never degrade to a class-name-only hash for a configured generator."""
    if is_dataclass(gen) and not isinstance(gen, type):
        raw: Dict[str, Any] = {f.name: getattr(gen, f.name) for f in fields(gen)}
    else:
        raw = dict(vars(gen))
    try:
        return json.loads(
            json.dumps({"class": _generator_id(gen), "state": raw}, sort_keys=True)
        )
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"extra generator {_generator_id(gen)} holds state that cannot be "
            f"JSON-serialized ({exc}); FeatureSpec.content_hash refuses to hash "
            "it by class name alone — make the state serializable or register "
            "the family with a bumped version instead."
        ) from exc


def _jsonable(obj: Any) -> Any:
    """Dataclasses/tuples/generators -> plain JSON-serializable structures.

    Registered families resolve to (name, version) pairs through the registry;
    extra generators resolve to their full resolved state (class id + __dict__).
    """
    if isinstance(obj, BaseFeatureGenerator):
        return _generator_state(obj)
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _jsonable(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, (tuple, list)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in sorted(obj.items())}
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return obj


def _family_version_entry(feature_sets: Tuple[str, ...]) -> List[Any]:
    """Hash form of feature_sets: plain names, plus each registered family's
    version folded alongside so a bumped family version invalidates frames."""
    from lab.registry import family_version

    return [
        {"family": name, "version": family_version(name)}
        for name in feature_sets
    ]


def _risk_profile_fingerprint(asset_class: str) -> dict:
    """The gate values that decide which rows survive the chop veto."""
    from execution.risk_manager import RiskProfile, _chop_filter_enabled

    profile = RiskProfile.for_asset_class(asset_class)
    return {
        "profile": _jsonable(profile),
        "chop_filter_enabled": _chop_filter_enabled(),
    }
