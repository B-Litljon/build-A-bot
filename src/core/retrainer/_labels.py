"""Devil target construction (bar-by-bar bracket simulation + survival variant) and
_compute_chop_veto_mask, the training-side mirror of the live RiskManager gates.
Split out of core/retrainer.py on 2026-09-16.
"""
from __future__ import annotations

from ._common import (
    MAX_HOLD_BARS,
    Optional,
    RiskProfile,
    SL_ATR_MULTIPLIER,
    SURVIVAL_BARS,
    TP_ATR_MULTIPLIER,
    _chop_filter_enabled,
    coupled_keff,
    logger,
    np,
    pl,
)


# ═══════════════════════════════════════════════════════════════════════════════
# ATR-DYNAMIC DEVIL TARGET (Bar-by-Bar Bracket Simulation)
# ═══════════════════════════════════════════════════════════════════════════════


def _compute_devil_targets_atr(
    df: pl.DataFrame,
    sl_mult: float = SL_ATR_MULTIPLIER,
    tp_mult: float = TP_ATR_MULTIPLIER,
    max_hold: int = MAX_HOLD_BARS,
) -> np.ndarray:
    """
    Compute Devil targets using dynamic ATR brackets with bar-by-bar resolution.

    For each bar i, simulates a bracket order:
        SL = close[i] - sl_mult * ATR_abs[i]
        TP = close[i] + tp_mult * ATR_abs[i]

    Then walks forward up to max_hold bars checking:
        - If low[j] <= SL → loss (0)
        - If high[j] >= TP → win (1)
        - SL is checked FIRST (conservative, matches evaluate_performance.py)
        - If neither hit in max_hold bars → loss (0, timeout)

    This avoids rolling max/min which does not respect the temporal ordering
    of SL vs TP hits.  Complexity: O(n × max_hold).  At 60 days × 5 tickers
    × ~390 bars/day ≈ 117k rows × 15 bars = ~1.75M iterations — runs in
    under 2 seconds on modern hardware.

    Args:
        df: DataFrame containing 'close', 'high', 'low', 'natr_14' columns.
        sl_mult: ATR multiplier for stop-loss (default: SL_ATR_MULTIPLIER).
        tp_mult: ATR multiplier for take-profit (default: TP_ATR_MULTIPLIER).
        max_hold: Maximum bars to hold before timeout (default: MAX_HOLD_BARS).

    Returns:
        NumPy array of int8 (0 = loss/timeout, 1 = win), same length as df.
        NaN/invalid entries at the tail are set to 0.
    """
    close = df["close"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    natr = df["natr_14"].to_numpy()
    symbol = df["symbol"].to_numpy() if "symbol" in df.columns else np.array([""] * len(close))
    n = len(close)
    targets = np.zeros(n, dtype=np.int8)

    for i in range(n - 1):
        atr_abs = close[i] * natr[i] / 100.0
        if np.isnan(atr_abs) or atr_abs <= 0:
            continue

        sl_price = close[i] - sl_mult * atr_abs
        tp_price = close[i] + tp_mult * atr_abs

        for j in range(i + 1, min(i + max_hold + 1, n)):
            if symbol[j] != symbol[i]:
                break
            # SL checked first (conservative — matches evaluate_performance.py)
            if low[j] <= sl_price:
                targets[i] = 0
                break
            if high[j] >= tp_price:
                targets[i] = 1
                break
        # If loop completes without break → timeout → 0 (already default)

    return targets


def _compute_devil_survival_target(
    df: pl.DataFrame,
    sl_mult: float = SL_ATR_MULTIPLIER,
    survival_bars: int = SURVIVAL_BARS,
) -> np.ndarray:
    """
    Compute Devil survival targets: whether price survives the SL for the
    next `survival_bars` bars after each row.

    Phase 5.5 — Temporal Realignment:
        The Devil's 1m microstructure features (wick toxicity, range
        compression) operate at a 1–5 minute horizon.  Asking the Devil
        to predict 45-bar macro outcomes (the old devil_target) creates
        an unlearnable temporal gap.  Asking it to predict 5-bar SL
        survival aligns the learning objective with the feature horizon.

    Survival definition:
        target[i] = 1  if  low[j] > SL_price  for ALL j in [i+1, i+SURVIVAL_BARS]
        target[i] = 0  if  low[j] <= SL_price  for ANY j in that window

    SL price is computed identically to the live bracket:
        SL = close[i] - sl_mult * ATR_abs[i]
        ATR_abs = close[i] * natr_14[i] / 100.0

    Args:
        df:             DataFrame with 'close', 'low', 'natr_14' columns.
        sl_mult:        ATR multiplier for stop-loss (default: SL_ATR_MULTIPLIER).
        survival_bars:  Number of bars to check for SL breach (default: SURVIVAL_BARS).

    Returns:
        NumPy int8 array of length len(df).
        1 = survived (no SL breach in window), 0 = stopped out.
        Last `survival_bars` rows are always 0 (insufficient lookahead).
    """
    close = df["close"].to_numpy()
    low = df["low"].to_numpy()
    natr = df["natr_14"].to_numpy()
    symbol = df["symbol"].to_numpy() if "symbol" in df.columns else np.array([""] * len(close))
    n = len(close)
    targets = np.zeros(n, dtype=np.int8)

    for i in range(n - 1):
        # Insufficient lookahead safety: check if symbol changes before survival window completes
        if i + survival_bars >= n or symbol[i + survival_bars] != symbol[i]:
            continue  # leaves targets[i] = 0 (default)

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

        targets[i] = np.int8(1) if survived else np.int8(0)

    return targets


# ═══════════════════════════════════════════════════════════════════════════════
# HYBRID CHOP VETO (symmetric with live RiskManager.calculate_bracket)
# ═══════════════════════════════════════════════════════════════════════════════


def _compute_chop_veto_mask(
    df: pl.DataFrame,
    profile: RiskProfile,
    sl_mult: float,
    alpha_table: Optional[dict] = None,
) -> np.ndarray:
    """
    Vectorized hybrid chop veto, mirroring ``RiskManager._evaluate_dynamic_gates``
    so the model trains only on the live-tradeable population.

    For each row, using the trailing ``regime_window`` of ``natr_14`` per symbol:
      * pctile_rank = fraction of the window <= the current bar's NATR
      * Gate B (regime): veto if ``pctile_rank < regime_pctile/100``
      * Gate A (cost): veto if ``sl_mult·natr < k_eff · alpha · baseline_natr``
        (the live inequality ``sl_dist < k_eff·spread_proxy`` with the
        volatility-scaled proxy; ``close`` cancels on both sides). The spread
        proxy scales with each era's *baseline* (median-window) volatility —
        not a static historical constant — so it is era-robust.

    ``alpha_table`` (2026-07-07): optional per-instrument spread alphas
    ({symbol: alpha_emp} from a bake_spread_alphas.py table). When provided,
    each symbol's measured alpha replaces the flat ``profile.spread_atr_alpha``
    in Gate A — instruments the table doesn't list fall back to the profile
    value. This fixes the asymmetry where training priced GBP_NZD (measured
    ~0.90) at the flat 0.15 and kept setups live always vetoes.

    Returns a boolean array (True = veto/drop) aligned to ``df`` rows. Rows are
    dropped only as trade *entries*; the bracket walk in the target functions
    still sees the full contiguous price path (so this must run AFTER target
    generation, not before).

    Gate C (time-of-day blackout) is mirrored here since 2026-09-09: the same
    America/New_York window as ``RiskManager._in_blackout`` (DST-correct,
    start-inclusive / end-exclusive, midnight wrap), applied to the rows' UTC
    timestamps (naive timestamps are assumed UTC, matching the live gate). The
    live bot drops NY-rollover entries (≈16:55–17:30 ET); training now vetoes
    the same bars instead of labelling un-executable rollover entries.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    n_total = df.height
    veto = np.zeros(n_total, dtype=bool)
    if not _chop_filter_enabled() or n_total == 0:
        return veto

    # Gate C (time-of-day blackout) — vectorized mirror of
    # RiskManager._in_blackout, DST-correct via America/New_York conversion.
    # Naive timestamps are assumed UTC (the live gate does the same).
    if (
        profile.time_gate_enabled
        and profile.blackout_start is not None
        and profile.blackout_end is not None
        and "timestamp" in df.columns
    ):
        ts = df["timestamp"]
        if getattr(ts.dtype, "time_zone", None) is None:
            ts = ts.dt.replace_time_zone("UTC")
        ny = ts.dt.convert_time_zone("America/New_York")
        # .dt.hour() is Int8 — cast before multiplying or 22*3600 overflows.
        secs = (
            ny.dt.hour().cast(pl.Int64) * 3600
            + ny.dt.minute().cast(pl.Int64) * 60
            + ny.dt.second().cast(pl.Int64)
        )
        start_s = profile.blackout_start.hour * 3600 + profile.blackout_start.minute * 60
        end_s = profile.blackout_end.hour * 3600 + profile.blackout_end.minute * 60
        if start_s <= end_s:
            gate_c = (secs >= start_s) & (secs < end_s)
        else:
            gate_c = (secs >= start_s) | (secs < end_s)  # window wraps midnight
        veto |= gate_c.to_numpy()

    w = int(profile.regime_window)
    mins = int(profile.regime_min_samples)
    p_thresh = profile.regime_pctile / 100.0

    symbols = df["symbol"].to_numpy() if "symbol" in df.columns else np.zeros(n_total)
    natr_all = df["natr_14"].to_numpy().astype(float)

    for sym in np.unique(symbols):
        # Per-instrument measured alpha when a table is provided; flat profile
        # value otherwise (and for symbols the table doesn't list).
        alpha = (alpha_table or {}).get(str(sym), profile.spread_atr_alpha)
        idx = np.where(symbols == sym)[0]  # contiguous, time-ordered per symbol
        natr = natr_all[idx]
        m = len(natr)
        # rank_actual feeds Gate B (regime); rank_eff feeds the coupling and is
        # held neutral (0.5) until the window is warm — exactly as the live gate
        # holds pctile_rank=0.5 below regime_min_samples.
        rank_actual = np.full(m, 0.5)
        rank_eff = np.full(m, 0.5)
        baseline = np.full(m, np.nan)  # expanding/rolling median of the window

        # Full-window region (vectorized): rows i >= w-1 (always warm: w >= mins).
        if m >= w:
            sw = sliding_window_view(natr, w)  # (m-w+1, w) → rows w-1 .. m-1
            last = sw[:, -1]
            fr = (sw <= last[:, None]).mean(axis=1)
            rank_actual[w - 1:] = fr
            rank_eff[w - 1:] = fr
            baseline[w - 1:] = np.median(sw, axis=1)

        # Expanding region (all earlier rows): baseline is always computable, so
        # Gate A (cost) runs from the first bar; Gate B only once warm.
        for i in range(0, min(w - 1, m)):
            win = natr[: i + 1]
            rank_actual[i] = float(np.mean(win <= natr[i]))
            baseline[i] = float(np.median(win))
            if (i + 1) >= mins:  # warm → real rank couples; else stay neutral 0.5
                rank_eff[i] = rank_actual[i]

        warm = (np.arange(m) + 1) >= mins

        gate_b = np.zeros(m, dtype=bool)
        if profile.regime_gate_enabled:
            gate_b = warm & (rank_actual < p_thresh)

        gate_a = np.zeros(m, dtype=bool)
        if profile.spread_gate_enabled:
            k_eff = coupled_keff(
                profile.spread_k_base, profile.spread_k_coupling,
                profile.spread_k_coupling_mode, rank_eff,
            )
            with np.errstate(invalid="ignore"):
                gate_a = (sl_mult * natr) < (k_eff * alpha * baseline)
            gate_a &= np.isfinite(baseline)

        # OR into the existing veto (Gate C may already have marked rows):
        # assigning here would silently clobber the blackout mask.
        veto[idx] |= gate_a | gate_b
        # Per-symbol diagnostics — critical when alpha_table is active: an
        # expensive instrument (GBP_NZD ~0.90) should thin dramatically.
        logger.info(
            "  Chop veto [%s]: alpha=%.4f | gate_a=%d gate_b=%d gate_c=%d | vetoed %d/%d (%.1f%%)",
            sym, alpha, int(gate_a.sum()), int(gate_b.sum()),
            int(veto[idx].sum() - (gate_a | gate_b).sum()),
            int(veto[idx].sum()), m,
            100.0 * veto[idx].mean() if m else 0.0,
        )

    return veto
