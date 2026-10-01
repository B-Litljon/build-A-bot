"""Spread-adjusted Devil label brackets (2026-09-27 fix).

Pins the contract that the two Devil target generators simulate the
round-trip spread a live trade pays (entry at Ask, exit at Bid, historicals
at Mid) when an alpha_table is provided, and that they stay BYTE-IDENTICAL
to the frictionless Mid labels when it is not:

  1. alpha_table=None (and {}) == a verbatim frictionless reference walk.
  2. A Mid move of exactly tp_mult*ATR that won at zero toll now misses TP.
  3. A Mid move of tp_mult*ATR + spread still wins.
  4. The SL edge moves up too (a surviving-at-Mid path can stop out).
  5. Symbols missing from the table fall back to DEFAULT_SPREAD_ALPHA
     (including the placeholder empty symbol of frames without a symbol
     column).
  6. Wiring: engineer_features_and_labels / apply_labels_and_veto actually
     thread alpha_table into both generators, and a zero-alpha table is
     indistinguishable from no table while a huge-alpha table prices every
     entry out of its bracket.

Spread formula unit convention (matched, not invented): the live Gate A
proxy prices ``spread_proxy = alpha * baseline_atr_abs``
(src/execution/risk_manager.py:696-698) and V3CostFeatures prices
``cost_ratio = alpha * baseline_natr / natr_14``
(src/ml/features/v3_features.py:267) — alpha is a dimensionless fraction of
ATR, so here ``spread_price = alpha * atr_abs``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from core.retrainer import (
    DEFAULT_SPREAD_ALPHA,
    _compute_devil_survival_target,
    _compute_devil_targets_atr,
    apply_labels_and_veto,
    engineer_features_and_labels,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Frame builders + the frictionless reference walk (pre-spread semantics)
# ═══════════════════════════════════════════════════════════════════════════════


def _frame(symbols, n, seed):
    """Multi-symbol random-walk OHLCV frame with a deterministic natr_14."""
    parts = []
    for k, sym in enumerate(symbols):
        rng = np.random.RandomState(seed + k)
        closes = 100.0 + np.cumsum(rng.randn(n) * 0.1)
        highs = closes + rng.uniform(0.05, 0.30, size=n)
        lows = closes - rng.uniform(0.05, 0.30, size=n)
        parts.append(
            pl.DataFrame(
                {
                    "timestamp": pl.datetime_range(
                        pl.datetime(2026, 1, 1, 0, 0),
                        pl.datetime(2026, 1, 1, 0, 0) + pl.duration(minutes=n - 1),
                        interval="1m",
                        eager=True,
                    ),
                    "symbol": [sym] * n,
                    "open": (highs + lows) / 2.0,
                    "high": highs,
                    "low": lows,
                    "close": closes,
                    "volume": rng.uniform(100.0, 1000.0, size=n),
                    "natr_14": rng.uniform(0.08, 0.35, size=n),
                }
            )
        )
    return pl.concat(parts).sort(["symbol", "timestamp"])


def _ref_targets_atr(df, sl_mult, tp_mult, max_hold):
    """Verbatim pre-2026-09-27 _compute_devil_targets_atr (frictionless Mid)."""
    close = df["close"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    natr = df["natr_14"].to_numpy()
    symbol = (
        df["symbol"].to_numpy()
        if "symbol" in df.columns
        else np.array([""] * len(close))
    )
    n = len(close)
    out = np.zeros(n, dtype=np.int8)
    for i in range(n - 1):
        atr_abs = close[i] * natr[i] / 100.0
        if np.isnan(atr_abs) or atr_abs <= 0:
            continue
        sl_price = close[i] - sl_mult * atr_abs
        tp_price = close[i] + tp_mult * atr_abs
        for j in range(i + 1, min(i + max_hold + 1, n)):
            if symbol[j] != symbol[i]:
                break
            if low[j] <= sl_price:
                out[i] = 0
                break
            if high[j] >= tp_price:
                out[i] = 1
                break
    return out


def _ref_survival(df, sl_mult, survival_bars):
    """Verbatim pre-2026-09-27 _compute_devil_survival_target (Mid only)."""
    close = df["close"].to_numpy()
    low = df["low"].to_numpy()
    natr = df["natr_14"].to_numpy()
    symbol = (
        df["symbol"].to_numpy()
        if "symbol" in df.columns
        else np.array([""] * len(close))
    )
    n = len(close)
    out = np.zeros(n, dtype=np.int8)
    for i in range(n - 1):
        if i + survival_bars >= n or symbol[i + survival_bars] != symbol[i]:
            continue
        atr_abs = close[i] * natr[i] / 100.0
        if np.isnan(atr_abs) or atr_abs <= 0:
            continue
        sl_price = close[i] - sl_mult * atr_abs
        survived = True
        for j in range(i + 1, min(i + survival_bars + 1, n)):
            if symbol[j] != symbol[i]:
                survived = False
                break
            if low[j] <= sl_price:
                survived = False
                break
        out[i] = np.int8(1) if survived else np.int8(0)
    return out


def _entry_frame(after_entry, *, natr=1.0, close0=10.0, symbol="EUR_USD"):
    """
    Two-bar frame: bar 0 is the entry (close=close0), bar 1 carries whatever
    high/low the test sets. natr is a percentage (1.0 → ATR_abs = 0.1 at
    close0=10: frictionless SL=9.8, TP=10.4 under 2.0x/4.0x).
    """
    return pl.DataFrame(
        {
            "timestamp": pl.datetime_range(
                pl.datetime(2026, 1, 1, 0, 0),
                pl.datetime(2026, 1, 1, 0, 15),
                interval="15m",
                eager=True,
            ),
            "symbol": [symbol] * 2,
            "open": [close0, close0],
            "high": [close0, after_entry["high"]],
            "low": [close0, after_entry["low"]],
            "close": [close0, after_entry["close"]],
            "volume": [100.0, 100.0],
            "natr_14": [natr, natr],
        }
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 1. alpha_table=None / {} is byte-identical to the frictionless reference
# ═══════════════════════════════════════════════════════════════════════════════


class TestDefaultIsFrictionless:
    def test_macro_none_and_empty_match_reference(self):
        df = _frame(["EUR_USD", "GBP_JPY", "GBP_NZD"], 220, seed=19)
        for sl, tp, hold in ((2.0, 4.0, 10), (0.5, 3.0, 45), (1.0, 1.0, 3)):
            ref = _ref_targets_atr(df, sl, tp, hold)
            np.testing.assert_array_equal(
                _compute_devil_targets_atr(df, sl_mult=sl, tp_mult=tp, max_hold=hold),
                ref,
                err_msg=f"None-table drift at sl={sl} tp={tp} hold={hold}",
            )
            np.testing.assert_array_equal(
                _compute_devil_targets_atr(
                    df, sl_mult=sl, tp_mult=tp, max_hold=hold, alpha_table=None
                ),
                ref,
            )
            np.testing.assert_array_equal(
                _compute_devil_targets_atr(
                    df, sl_mult=sl, tp_mult=tp, max_hold=hold, alpha_table={}
                ),
                ref,
                err_msg=f"empty-table drift at sl={sl} tp={tp} hold={hold}",
            )

    def test_survival_none_and_empty_match_reference(self):
        df = _frame(["EUR_USD", "NZD_JPY"], 160, seed=7)
        for sl, bars in ((0.5, 5), (2.0, 3), (1.0, 8)):
            ref = _ref_survival(df, sl, bars)
            np.testing.assert_array_equal(
                _compute_devil_survival_target(df, sl_mult=sl, survival_bars=bars),
                ref,
                err_msg=f"None-table drift at sl={sl} bars={bars}",
            )
            np.testing.assert_array_equal(
                _compute_devil_survival_target(
                    df, sl_mult=sl, survival_bars=bars, alpha_table=None
                ),
                ref,
            )
            np.testing.assert_array_equal(
                _compute_devil_survival_target(
                    df, sl_mult=sl, survival_bars=bars, alpha_table={}
                ),
                ref,
            )

    def test_zero_alpha_table_equals_frictionless(self):
        """A listed alpha of 0.0 prices no toll — same as no table at all."""
        df = _frame(["GBP_JPY"], 200, seed=5)
        zip_alpha = {s: 0.0 for s in ["GBP_JPY"]}
        np.testing.assert_array_equal(
            _compute_devil_targets_atr(df, alpha_table=zip_alpha),
            _compute_devil_targets_atr(df),
        )
        np.testing.assert_array_equal(
            _compute_devil_survival_target(df, alpha_table=zip_alpha),
            _compute_devil_survival_target(df),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 2/3. The bracket edges move up through Mid by alpha*ATR
# ═══════════════════════════════════════════════════════════════════════════════


class TestBracketEdgesShiftUp:
    # ATR_abs = 0.1 at close0=10, natr=1.0. Thresholds below are built with
    # the same floating-point expression chain the generators use (close*natr/
    # 100, then +spread), so equality at the boundary is bit-exact, not
    # decimal-lucky.
    SL_MULT, TP_MULT = 2.0, 4.0
    ATR = 10.0 * 1.0 / 100.0
    SL0 = 10.0 - 2.0 * ATR
    TP0 = 10.0 + 4.0 * ATR

    def _macro0(self, df, **kw):
        return _compute_devil_targets_atr(
            df, sl_mult=self.SL_MULT, tp_mult=self.TP_MULT, max_hold=5, **kw
        )[0]

    def test_exact_tp_move_no_longer_wins(self):
        """Mid high = tp_mult*ATR exactly: wins at zero toll, loses once the
        spread is paid (TP needs tp_mult*ATR + spread through Mid); low stays
        clear of both stops so the flip is purely the TP edge."""
        spread = 0.5 * self.ATR
        df = _entry_frame({"high": self.TP0, "low": self.SL0 + 3 * self.ATR,
                           "close": 10.1})
        assert self._macro0(df) == 1
        assert self._macro0(df, alpha_table={"EUR_USD": 0.5}) == 0

    def test_tp_plus_spread_move_still_wins(self):
        """Mid high = tp_mult*ATR + spread clears the spread-adjusted TP."""
        alpha = 0.5
        tp_t = 10.0 + self.TP_MULT * self.ATR + alpha * self.ATR
        df = _entry_frame({"high": tp_t, "low": self.SL0 + 3 * self.ATR,
                           "close": 10.1})
        assert self._macro0(df) == 1
        assert self._macro0(df, alpha_table={"EUR_USD": alpha}) == 1

    def test_sl_edge_moves_up_through_mid(self):
        """The stop rises too: a low between the frictionless and effective
        stops now stops out (SL checked first, even though TP would clear)."""
        alpha = 0.5
        sl_t = 10.0 - self.SL_MULT * self.ATR + alpha * self.ATR
        tp_t = 10.0 + self.TP_MULT * self.ATR + alpha * self.ATR
        low_mid = (self.SL0 + sl_t) / 2.0  # strictly above SL0, strictly below sl_t
        df = _entry_frame({"high": tp_t, "low": low_mid, "close": 10.1})
        assert self._macro0(df) == 1
        assert self._macro0(df, alpha_table={"EUR_USD": alpha}) == 0

    def test_survival_boundary_moves_by_spread(self):
        """Survival stops out at SL + spread: a Mid-path below the effective
        stop no longer survives; the inclusive-breach semantics are unchanged
        at the new edge; survival can only flip 1→0, never 0→1."""
        alpha = 0.5
        sl_t = 10.0 - self.SL_MULT * self.ATR + alpha * self.ATR
        mid = (self.SL0 + sl_t) / 2.0
        surv = lambda low, **kw: _compute_devil_survival_target(
            _entry_frame({"high": 9.9, "low": low, "close": 10.0}),
            sl_mult=self.SL_MULT, survival_bars=1, **kw,
        )[0]
        assert surv(mid) == 1  # above the frictionless SL, survives at Mid
        assert surv(mid, alpha_table={"EUR_USD": alpha}) == 0  # pays the toll
        assert surv(sl_t, alpha_table={"EUR_USD": alpha}) == 0  # inclusive
        assert surv(np.nextafter(sl_t, np.float64(np.inf)),
                    alpha_table={"EUR_USD": alpha}) == 1  # just clear survives
        assert surv(self.SL0, alpha_table={"EUR_USD": alpha}) == 0  # also breached

    def test_survival_wins_subset(self):
        """Tabled survival approvals can never exceed frictionless ones on
        identical bars — the shifted stop only adds breaches."""
        df = _frame(["EUR_USD"], 200, seed=13)
        np.testing.assert_array_less(
            _compute_devil_survival_target(df, sl_mult=2.0, survival_bars=5,
                                           alpha_table={"EUR_USD": 0.5}),
            _compute_devil_survival_target(df, sl_mult=2.0, survival_bars=5) + 1,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Missing-symbol fallback to DEFAULT_SPREAD_ALPHA
# ═══════════════════════════════════════════════════════════════════════════════


class TestMissingSymbolFallback:
    def test_unlisted_symbol_charged_default_alpha(self):
        """Only GBP_JPY is listed (alpha 0.01); EUR_USD must be charged the
        0.15 default: a move that clears the 0.01 toll but not the 0.15 one.
        ATR_abs = 0.1 at close0=10 → spread(0.01)=0.001 → TP at 10.401;
        spread(0.15)=0.015 → TP at 10.415. high 10.41 wins at the listed
        alpha, loses at the default (thresholds at bit-exact edges below)."""
        close0, natr = 10.0, 1.0
        atr = close0 * natr / 100.0
        tp_at_alpha01 = close0 + 4.0 * atr + 0.01 * atr
        tp_at_default = close0 + 4.0 * atr + 0.15 * atr
        high = (tp_at_alpha01 + tp_at_default) / 2.0  # strictly between the two TPs
        assert tp_at_alpha01 < high < tp_at_default
        df = _entry_frame(
            {"high": high, "low": 10.0, "close": 10.4}, symbol="EUR_USD"
        )
        macro = lambda **kw: _compute_devil_targets_atr(
            df, sl_mult=2.0, tp_mult=4.0, max_hold=5, **kw
        )[0]
        assert macro(alpha_table={"EUR_USD": 0.15}) == 0
        assert macro(alpha_table={"EUR_USD": 0.01}) == 1  # listed → cheap toll
        assert macro(alpha_table={"GBP_JPY": 0.01}) == 0  # unlisted → default
        assert macro(alpha_table={"GBP_JPY": 0.01}) == macro(
            alpha_table={"EUR_USD": 0.15}
        )

        # Survival mirror: a low between the two effective stops (alpha 0.01
        # → 9.801: survives; default 0.15 → 9.815: breaches).
        sl_at_alpha01 = close0 - 2.0 * atr + 0.01 * atr
        sl_at_default = close0 - 2.0 * atr + 0.15 * atr
        low_mid = (sl_at_alpha01 + sl_at_default) / 2.0
        assert sl_at_alpha01 < low_mid < sl_at_default
        df_s = _entry_frame(
            {"high": 9.9, "low": low_mid, "close": 10.0}, symbol="EUR_USD"
        )
        surv = lambda **kw: _compute_devil_survival_target(
            df_s, sl_mult=2.0, survival_bars=1, **kw
        )[0]
        assert surv(alpha_table={"EUR_USD": 0.15}) == 0
        assert surv(alpha_table={"EUR_USD": 0.01}) == 1  # would survive, if listed
        assert surv(alpha_table={"GBP_JPY": 0.01}) == 0  # unlisted → default charged
        assert surv(alpha_table={"GBP_JPY": 0.01}) == surv(
            alpha_table={"EUR_USD": 0.15}
        )

    def test_placeholder_symbol_without_symbol_column_uses_default(self):
        """Frames without a symbol column walk under the '' placeholder; the
        table can never match it, so the default alpha must apply."""
        close0, natr = 10.0, 1.0
        atr = close0 * natr / 100.0
        high = (close0 + 4.0 * atr + 0.01 * atr + close0 + 4.0 * atr
                + 0.15 * atr) / 2.0  # strictly between listed-0.01 and default TPs
        df = _entry_frame({"high": high, "low": 10.0, "close": 10.4}).drop("symbol")
        macro = lambda **kw: _compute_devil_targets_atr(
            df, sl_mult=2.0, tp_mult=4.0, max_hold=5, **kw
        )[0]
        assert macro(alpha_table={"EUR_USD": 0.01}) == 0  # 0.15 default charged
        assert macro(alpha_table={"EUR_USD": 0.15}) == 0
        assert macro(alpha_table={}) == 1  # empty table → frictionless

    def test_default_alpha_matches_risk_profile_placeholder(self):
        """The fallback is the same 0.15 placeholder Gate A uses — one toll
        convention across training and execution."""
        from src.execution.risk_manager import RiskProfile

        assert DEFAULT_SPREAD_ALPHA == RiskProfile.for_asset_class(
            "forex"
        ).spread_atr_alpha == 0.15


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Dominance theorem: spread-adjusted wins ⊆ frictionless wins
# ═══════════════════════════════════════════════════════════════════════════════


class TestSpreadOnlyMakesTradesHarder:
    def test_tabled_wins_are_subset_of_frictionless_wins(self):
        """If the spread-adjusted walk wins, the frictionless walk must win
        too: TP_T > TP and SL_T > SL imply the clearing bar clears both."""
        df = _frame(["EUR_USD", "GBP_JPY", "GBP_AUD", "NZD_JPY"], 240, seed=23)
        table = {s: a for s, a in zip(
            ["EUR_USD", "GBP_JPY", "GBP_AUD", "NZD_JPY"], (0.3, 0.9, 0.15, 0.55)
        )}
        np.testing.assert_array_less(
            _compute_devil_targets_atr(df, sl_mult=2.0, tp_mult=4.0,
                                       max_hold=20, alpha_table=table),
            _compute_devil_targets_atr(df, sl_mult=2.0, tp_mult=4.0, max_hold=20) + 1,
        )
        np.testing.assert_array_less(
            _compute_devil_survival_target(df, sl_mult=2.0, survival_bars=5,
                                           alpha_table=table),
            _compute_devil_survival_target(df, sl_mult=2.0, survival_bars=5) + 1,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Wiring — alpha_table reaches both generators
# ═══════════════════════════════════════════════════════════════════════════════


def _minimal_label_frame(n=30, symbol="GBP_JPY"):
    close0 = 150.0
    rng = np.random.RandomState(3)
    closes = close0 + np.cumsum(rng.randn(n) * 0.2)
    return pl.DataFrame(
        {
            "timestamp": pl.datetime_range(
                pl.datetime(2026, 1, 1, 0, 0),
                pl.datetime(2026, 1, 1, 0, 0) + pl.duration(minutes=n - 1),
                interval="1m",
                eager=True,
            ),
            "symbol": [symbol] * n,
            "open": closes,
            "high": closes + 0.1,
            "low": closes - 0.1,
            "close": closes,
            "volume": np.arange(n, dtype=float),
            "natr_14": [0.2] * n,
        }
    )


def _trend_frame():
    """30 flat warmup bars then 60 bars climbing +0.15/bar (highs close+0.05,
    lows close-0.05 — pullbacks far smaller than the 2×ATR stop). Frictionless
    brackets win on many entries (TP = 4×ATR ≈ 0.8 is reached in ~6 bars, the
    stop is never touched); a 10-alpha toll lifts the TP by ~2.0 and the
    effective stop ABOVE the entry, so every tabled entry stops out."""
    closes, k = [], 0
    for bar in range(90):
        closes.append(100.0 + (0.15 * bar if bar >= 30 else 0.0))
    highs = [c + 0.05 for c in closes]
    lows = [c - 0.05 for c in closes]
    n = len(closes)
    return pl.DataFrame(
        {
            "timestamp": pl.datetime_range(
                pl.datetime(2026, 3, 2, 0, 0),
                pl.datetime(2026, 3, 2, 0, 0) + pl.duration(minutes=n - 1),
                interval="1m",
                eager=True,
            ),
            "symbol": ["EUR_USD"] * n,
            "open": closes,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [100.0] * n,
        }
    )


class TestWiringThreadsAlphaTable:
    def test_apply_labels_and_veto_passes_alpha_table_to_both_generators(
        self, monkeypatch
    ):
        import core.retrainer._features as features_mod

        captured: dict = {}
        real_atr = features_mod._compute_devil_targets_atr
        real_surv = features_mod._compute_devil_survival_target

        def spy_atr(df, **kw):
            captured["macro"] = kw.get("alpha_table", "MISSING")
            return real_atr(df, **kw)

        def spy_surv(df, **kw):
            captured["survival"] = kw.get("alpha_table", "MISSING")
            return real_surv(df, **kw)

        monkeypatch.setattr(features_mod, "_compute_devil_targets_atr", spy_atr)
        monkeypatch.setattr(features_mod, "_compute_devil_survival_target", spy_surv)

        table = {"GBP_JPY": 0.55}
        apply_labels_and_veto(
            _minimal_label_frame(), ["natr_14"], alpha_table=table
        )
        assert captured["macro"] == table
        assert captured["survival"] == table

        captured.clear()
        apply_labels_and_veto(_minimal_label_frame(), ["natr_14"])
        assert captured["macro"] is None
        assert captured["survival"] is None

    def _engineer(self, raw, **kw):
        feats, _, _ = engineer_features_and_labels(
            raw, sl_mult=2.0, tp_mult=4.0, max_hold=20, survival_bars=5,
            htf_timeframe="5m", **kw,
        )
        return feats.select(
            "timestamp", "devil_target_macro", "devil_target"
        )

    def test_engineer_features_applies_the_toll(self):
        """End-to-end on engineer_features_and_labels: zero-alpha table is
        byte-identical to no table; a huge-alpha table prices entries out of
        their brackets — every tabled label is <= the frictionless one on the
        matched timestamps, and at least one macro win is eaten by the toll."""
        raw = _trend_frame()
        plain = self._engineer(raw)
        zero = self._engineer(raw, alpha_table={"EUR_USD": 0.0})
        assert zero.height == plain.height
        # Row sets are identical (alpha affects labels, not the cleaned
        # feature columns), so a positional compare is row-aligned.
        np.testing.assert_array_equal(
            zero["devil_target_macro"].to_numpy(),
            plain["devil_target_macro"].to_numpy(),
        )
        np.testing.assert_array_equal(
            zero["devil_target"].to_numpy(),
            plain["devil_target"].to_numpy(),
        )

        huge = self._engineer(raw, alpha_table={"EUR_USD": 10.0})
        assert huge.height == plain.height
        joined = plain.join(huge, on="timestamp", suffix="_huge")
        plain_macro = joined["devil_target_macro"].to_numpy()
        huge_macro = joined["devil_target_macro_huge"].to_numpy()
        plain_surv = joined["devil_target"].to_numpy()
        huge_surv = joined["devil_target_huge"].to_numpy()
        np.testing.assert_array_less(huge_macro, plain_macro + 1)
        np.testing.assert_array_less(huge_surv, plain_surv + 1)
        assert (plain_macro == 1).sum() >= 1, "fixture must contain a win"
        assert (huge_macro == 1).sum() < (plain_macro == 1).sum() or (
            (plain_macro == 1) & (huge_macro == 0)
        ).any(), "the toll must eat at least one frictionless win"