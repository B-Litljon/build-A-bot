"""
MAE/MFE excursion labels — the "right answer" columns for quantile barrier
models.

For a long entry at price ``e`` (the sealed bar's close) over the next
``horizon`` bars:

    MAE (maximum adverse excursion)  = e - min(low[t+1 .. t+horizon])
    MFE (maximum favorable excursion) = max(high[t+1 .. t+horizon]) - e

both clipped at 0 (a walk that never crosses the entry has no excursion that
side) and scaled by the bar's own absolute ATR (``close * natr_14 / 100``), so
one model spans JPY crosses and metals — same convention as every other
volatility-scaled quantity in this repo.

The SHORT direction needs no separate labels: a short's adverse excursion is
the long's favorable excursion and vice versa (short MAE ≡ long MFE, short MFE
≡ long MAE), so the two long-side label columns serve both directions. This
identity is load-bearing for the estimator design — see ml/barriers/README.md.

Labels for the last ``horizon`` bars are null: their walk runs off the end of
the frame. So are labels whose forward window contains even one null high or
low: ``nanmin``/``nanmax`` would silently skip the gap bar and return a finite
minimum from fewer than ``horizon`` observations, marking the row resolvable
with an understated excursion. A partial window is unresolvable, never
best-effort. Those rows must be dropped before fitting (the same unresolvable-
tail principle as the retrainer's boundary purge), never zero-filled — a
timeout is not the same as "no excursion", and inventing 0s would teach the
quantile model that the tail is thinner than it is.

Glossary:
    MAE / mae_natr -- maximum adverse excursion over the forward window, as a
        multiple of the entry bar's absolute ATR. The stop-side label.
    MFE / mfe_natr -- maximum favorable excursion over the same window, same
        units. The target-side label.
    horizon -- how many sealed bars forward the walk looks (default 45,
        matching the retrainer's max_hold; the label horizon and the live
        hold limit must agree or the geometry the model learns is not the
        geometry the trade experiences).
    excursions -- the output frame: input columns plus mae_natr / mfe_natr /
        resolvable, null where the forward window is incomplete.
"""

from typing import Optional

import numpy as np
import polars as pl

DEFAULT_HORIZON = 45


def compute_excursions(df: pl.DataFrame, horizon: int = DEFAULT_HORIZON) -> pl.DataFrame:
    """
    Append mae_natr / mfe_natr / resolvable columns to a single-symbol OHLC
    frame.

    Requires columns: open, high, low, close, natr_14. Rows whose forward
    window is incomplete — including windows that contain a single null high
    or low — get null labels (resolvable=False), as do rows with a null or
    non-positive natr_14 — a bracket cannot be sized off a missing volatility
    estimate.
    """
    for col in ("high", "low", "close", "natr_14"):
        if col not in df.columns:
            raise ValueError(f"compute_excursions requires column '{col}'")

    low = df["low"].to_numpy().astype(float)
    high = df["high"].to_numpy().astype(float)
    close = df["close"].to_numpy().astype(float)
    natr = df["natr_14"].to_numpy().astype(float)
    n = len(close)

    fwd_low = np.concatenate([low[1:], np.full(horizon, np.nan)])
    fwd_high = np.concatenate([high[1:], np.full(horizon, np.nan)])
    if n - 1 >= horizon:
        win_low = np.lib.stride_tricks.sliding_window_view(fwd_low, horizon)
        win_high = np.lib.stride_tricks.sliding_window_view(fwd_high, horizon)
        fmin = np.full(n, np.nan)
        fmax = np.full(n, np.nan)
        # A window may extend past the tail padding only via the
        # index-bounded slice below, but a NaN *inside* low/high (a gap bar)
        # must poison every window containing it — otherwise nanmin/nanmax
        # quietly compute the excursion from a shortened window and the row
        # looks resolvable.
        win_ok = np.full(n, False)
        m = n - horizon  # slice [:m] covers exactly the all-resolvable rows
        win_ok[:m] = (
            np.isfinite(win_low[:m]).all(axis=1)
            & np.isfinite(win_high[:m]).all(axis=1)
        )
        fmin[:m] = np.where(win_ok[:m], np.min(win_low[:m], axis=1), np.nan)
        fmax[:m] = np.where(win_ok[:m], np.max(win_high[:m], axis=1), np.nan)
    else:
        fmin = np.full(n, np.nan)
        fmax = np.full(n, np.nan)
        win_ok = np.zeros(n, dtype=bool)

    atr_abs = close * natr / 100.0
    with np.errstate(invalid="ignore"):
        mae = np.clip(close - fmin, 0.0, None) / atr_abs
        mfe = np.clip(fmax - close, 0.0, None) / atr_abs

    resolvable = (
        (np.arange(n) + horizon <= n - 1)
        & np.isfinite(atr_abs)
        & (atr_abs > 0)
        & win_ok
    )
    mae = np.where(resolvable, mae, np.nan)
    mfe = np.where(resolvable, mfe, np.nan)

    return df.with_columns(
        pl.Series("mae_natr", mae).fill_nan(None),
        pl.Series("mfe_natr", mfe).fill_nan(None),
        pl.Series("resolvable", resolvable),
    )


def pinball_loss(y: np.ndarray, q: np.ndarray, tau: float) -> float:
    """
    Mean pinball (quantile) loss, the objective the barrier estimators
    minimise. Lower is better; it is the metric the learned barriers must
    beat the static-multiple baseline on, per fold and on the holdout.
    """
    y = np.asarray(y, dtype=float)
    q = np.asarray(q, dtype=float)
    ok = np.isfinite(y) & np.isfinite(q)
    if not ok.any():
        return float("nan")
    u = y[ok] - q[ok]
    return float(np.mean(np.where(u >= 0, tau * u, (tau - 1.0) * u)))