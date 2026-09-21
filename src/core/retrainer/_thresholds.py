"""EV-optimal Devil threshold search (calibration-set sweep) and
_find_optimal_angel_threshold (OOF-calibrated Angel proposal bar).
Split out of core/retrainer.py on 2026-09-16.
"""
from __future__ import annotations

from ._common import (
    MIN_ANGEL_PROPOSALS,
    SL_ATR_MULTIPLIER,
    TP_ATR_MULTIPLIER,
    Tuple,
    logger,
    np,
)


# ═══════════════════════════════════════════════════════════════════════════════
# DYNAMIC THRESHOLD SELECTION
# ═══════════════════════════════════════════════════════════════════════════════


def _find_optimal_threshold(
    devil_probs: np.ndarray,
    survival_targets: np.ndarray,
    macro_targets: np.ndarray,
    sl_mult: float = SL_ATR_MULTIPLIER,
    tp_mult: float = TP_ATR_MULTIPLIER,
    min_trades: int = 5,
) -> Tuple[float, float]:
    """
    Sweep thresholds to find the one that maximizes Expected Value.

    Phase 5.5 — Two-Target EV Calibration:
        The Devil is trained on `survival_targets` (5-bar SL survival).
        EV calibration must use `macro_targets` (45-bar bracket outcome) to
        reflect the actual asymmetric R:R payload delivered by the live system.

        Separating these two concerns is critical:
        - Using survival_targets for EV would compute "expected value of not
          getting stopped in 5 bars" — meaningless for bracket sizing.
        - Using macro_targets for training would reintroduce the temporal
          mismatch that caused the Devil to flatline.

    For each candidate threshold:
        1. Filter to approved trades: devil_prob >= threshold
        2. Compute realized win rate from MACRO outcomes on approved trades
        3. Compute EV = win_rate * (tp_mult / sl_mult) - (1 - win_rate)

    Args:
        devil_probs:      Array of Devil's predicted probabilities (survival).
        survival_targets: 5-bar SL survival ground truth (Devil's training target).
                          Passed for signature consistency; not used in EV sweep.
        macro_targets:    45-bar bracket outcome ground truth (0/1).
                          Used to compute realized win rate and EV.
        sl_mult:          Stop-loss ATR multiplier.
        tp_mult:          Take-profit ATR multiplier.
        min_trades:       Minimum approved trades for a threshold to be valid.

    Returns:
        Tuple of (optimal_threshold, best_ev)
    """
    thresholds = np.arange(0.10, 0.66, 0.02)  # 0.10, 0.12, ..., 0.64
    best_threshold = 0.20  # fallback default
    best_ev = -float("inf")

    for t in thresholds:
        mask = devil_probs >= t
        n_approved = int(mask.sum())

        if n_approved < min_trades:
            continue

        # EV is computed from MACRO outcomes (45-bar bracket), not survival.
        # This correctly prices the asymmetric R:R of the live bracket system.
        approved_macro = macro_targets[mask]
        win_rate = float(approved_macro.mean())

        # EV in R-multiples: wins pay (tp_mult / sl_mult) R, losses pay -1R
        rr_ratio = tp_mult / sl_mult
        ev = win_rate * rr_ratio - (1.0 - win_rate)

        if ev > best_ev:
            best_ev = ev
            best_threshold = float(t)

    return best_threshold, float(best_ev)


def _find_optimal_angel_threshold(
    angel_probs: np.ndarray,
    macro_targets: np.ndarray,
    sl_mult: float = SL_ATR_MULTIPLIER,
    tp_mult: float = TP_ATR_MULTIPLIER,
    min_proposals: int = MIN_ANGEL_PROPOSALS,
) -> Tuple[float, float, int]:
    """
    Calibrate the Angel's proposal bar from out-of-fold probabilities.

    Why this exists: the Angel threshold was a global constant (0.40) while
    model capacity changes the score distribution underneath it. The trim
    100x15/min_child=80 architecture compresses predicted probabilities so
    severely that a fixed 0.40 bar passed 11/3/11 proposals per ~103k-row
    fold (2026-08-29 matrix) — starving both the Devil's training population
    and the fold gate's trade count. Expressing the bar in the model's OWN
    score units (quantiles of its OOF distribution) makes the proposal rate
    a property of the evidence, not of an arbitrary constant.

    Discipline mirrors _find_optimal_threshold (Devil), with one role
    difference: the Angel is the RECALL stage, so the sweep maximizes EV
    subject to a minimum proposal count that keeps the Devil's training
    population learnable — precision is the Devil's job downstream.

    Candidates are quantiles of the OOF scores (median through max), so the
    grid adapts to any distribution shape, including compressed ones. The
    sweep runs on TRAIN-frame OOF probabilities only; the chosen value is
    then applied frozen to validation/holdout/live scoring, so no validation
    information leaks into the parameter (same discipline as the Devil's
    frozen calibration_threshold).

    Args:
        angel_probs:   OOF Angel probabilities on the training frame.
        macro_targets: 45-bar bracket outcome ground truth (0/1), aligned
                       with angel_probs. Used for the EV objective.
        sl_mult:       Stop-loss ATR multiplier.
        tp_mult:       Take-profit ATR multiplier.
        min_proposals: Minimum proposal count for a candidate to be valid
                       (guarantees the Devil a training population).

    Returns:
        Tuple of (threshold, ev_at_threshold, n_proposals). When no candidate
        yields min_proposals (frame smaller than min_proposals), returns the
        observed minimum score — propose everything and let the gate judge.
    """
    n = len(angel_probs)
    if n == 0:
        return 0.5, -float("inf"), 0

    # Quantile grid from the median up to the observed max; dedupe guards a
    # distribution compressed to (near-)constant scores.
    grid_q = np.linspace(0.50, 1.0 - 1.0 / n, 400)
    candidates = np.unique(np.quantile(angel_probs, grid_q))

    rr_ratio = tp_mult / sl_mult
    best_threshold = float(candidates[0])
    best_ev = -float("inf")
    best_n = 0

    for t in candidates:
        mask = angel_probs >= t
        n_approved = int(mask.sum())
        if n_approved < min_proposals:
            continue
        win_rate = float(macro_targets[mask].mean())
        ev = win_rate * rr_ratio - (1.0 - win_rate)
        if ev > best_ev:
            best_ev = ev
            best_threshold = float(t)
            best_n = n_approved

    if best_ev == -float("inf"):
        # No candidate met the proposal floor (frame smaller than
        # min_proposals): propose EVERYTHING — the frame is degenerate
        # anyway, and starving the Devil of the little data that exists
        # only makes it worse. The fold gate judges the result.
        best_threshold = float(angel_probs.min())
        mask = angel_probs >= best_threshold
        best_n = int(mask.sum())
        win_rate = float(macro_targets[mask].mean()) if best_n else 0.0
        best_ev = win_rate * rr_ratio - (1.0 - win_rate)
        logger.warning(
            "Angel threshold sweep: no candidate reached min_proposals=%d "
            "(frame n=%d) — falling back to propose-everything %.4f "
            "(%d proposals)",
            min_proposals, n, best_threshold, best_n,
        )

    return best_threshold, float(best_ev), best_n
