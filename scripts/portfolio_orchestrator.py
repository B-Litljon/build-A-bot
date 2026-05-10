"""
V4 Portfolio Orchestrator — monthly cron-driven rebalance.

End-to-end pipeline executed once per cron invocation:

    1. Refresh raw data       — subprocess: investor_data_miner.py
                                (OHLCV, macro, SimFin fundamentals)
    2. Engineer features      — subprocess: investor_feature_pipeline.py
                                --inference  (embargo rows retained so
                                today's row survives)
    3. Predict & rank         — load models/v4_investor_lgbm.txt,
                                align inference frame to
                                booster.feature_name() (59 features),
                                predict scores, take top-K.
    4. Alpaca rebalance       — liquidate positions outside top-K,
                                rebalance top-K to TARGET_WEIGHT each
                                via fractional market orders.  Sells
                                are sized from a live Alpaca quote
                                fetched at execute time.

Architectural decisions (Gemini 3.1 Pro Rev 1):
    * SDK    : alpaca-py (project standard).  Env vars
               ALPACA_API_KEY / ALPACA_SECRET_KEY.
    * Truth  : booster.feature_name() — not a hard-coded N.
    * Refresh: subprocess to existing scripts (no inline duplication).
    * Sells  : live quote at execute time → exact qty.
    * Timing : MarketOrderRequest + TimeInForce.DAY; out-of-hours
               queueing is Alpaca's responsibility.
    * Safety : --dry-run skips order submission, logs intents.

Invocation:
    pipenv run python scripts/portfolio_orchestrator.py
    pipenv run python scripts/portfolio_orchestrator.py --dry-run

Cron (monthly, first day of month, 16:30 ET — after close):
    30 16 1 * *  cd /path/to/project && pipenv run python scripts/portfolio_orchestrator.py
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from dotenv import load_dotenv

# ── path setup (mirrors investor_data_miner.py) ──────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _PROJECT_ROOT / "src"
sys.path.insert(0, str(_SRC_DIR))

load_dotenv(_PROJECT_ROOT / ".env")

from alpaca.common.exceptions import APIError  # noqa: E402
from alpaca.data.historical.stock import StockHistoricalDataClient  # noqa: E402
from alpaca.data.requests import StockLatestQuoteRequest  # noqa: E402
from alpaca.trading.client import TradingClient  # noqa: E402
from alpaca.trading.enums import OrderSide, TimeInForce  # noqa: E402
from alpaca.trading.requests import MarketOrderRequest  # noqa: E402

# ── logging ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger("portfolio_orchestrator")

# ── strategy configuration ────────────────────────────────────────────
UNIVERSE: list[str] = ["AAPL", "MSFT", "NVDA", "JPM", "XOM", "WMT", "JNJ"]
TOP_K: int = 2
TARGET_WEIGHT: float = 0.50  # per top-K holding, applied to usable_equity

# Reserve 1% of equity as a cash buffer so penny-rounding across multiple
# notional orders cannot exceed buying power. Without this, allocating
# exactly 100% of equity (50% × 2) caused Alpaca to reject the second
# order on a 1¢ overdraft.
EQUITY_BUFFER: float = 0.99

# Skip rebalance trades smaller than this fraction of equity to avoid
# commission/slippage churn from sub-dollar drift orders.
REBALANCE_DEADBAND: float = 0.005  # 0.5% of equity

# ── paths ─────────────────────────────────────────────────────────────
_RAW_DATA_PATH = _PROJECT_ROOT / "data" / "raw" / "v4_investor_data.parquet"
_INFERENCE_PATH = _PROJECT_ROOT / "data" / "processed" / "v4_inference_features.parquet"
_MODEL_PATH = _PROJECT_ROOT / "models" / "v4_investor_lgbm.txt"
_MINER_SCRIPT = _PROJECT_ROOT / "scripts" / "investor_data_miner.py"
_FEATURE_SCRIPT = _PROJECT_ROOT / "scripts" / "investor_feature_pipeline.py"


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _sanitize(name: str) -> str:
    """Mirror the column-name sanitizer used by investor_train_model.py."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", name).strip("_")


@dataclass
class TradeIntent:
    """A single rebalance leg, ready for submission."""
    symbol: str
    side: OrderSide
    notional: float | None = None
    qty: float | None = None
    reason: str = ""
    extras: dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────
# Stage 1 + 2 — Subprocess refresh
# ─────────────────────────────────────────────────────────────────────

def _run_subprocess(label: str, cmd: list[str]) -> None:
    """Spawn a subprocess, stream its output, raise on non-zero exit."""
    logger.info("─" * 70)
    logger.info("[%s] %s", label, " ".join(cmd))
    logger.info("─" * 70)

    try:
        completed = subprocess.run(
            cmd,
            cwd=str(_PROJECT_ROOT),
            check=False,
            capture_output=False,  # let child stream its own logs
        )
    except OSError as exc:
        raise RuntimeError(f"{label} failed to spawn: {exc}") from exc

    if completed.returncode != 0:
        raise RuntimeError(
            f"{label} exited with code {completed.returncode}. "
            "See subprocess output above for details."
        )


def refresh_data() -> None:
    """Re-run the V4 data miner — produces v4_investor_data.parquet."""
    if not _MINER_SCRIPT.exists():
        raise FileNotFoundError(f"Miner script missing: {_MINER_SCRIPT}")
    _run_subprocess("Stage 1/4 — Data miner", [sys.executable, str(_MINER_SCRIPT)])

    if not _RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Miner ran but did not produce {_RAW_DATA_PATH}."
        )
    logger.info("Raw parquet refreshed: %s", _RAW_DATA_PATH)


def build_inference_features() -> None:
    """Run the feature pipeline in --inference mode (embargo retained)."""
    if not _FEATURE_SCRIPT.exists():
        raise FileNotFoundError(f"Feature pipeline missing: {_FEATURE_SCRIPT}")
    _run_subprocess(
        "Stage 2/4 — Feature pipeline (--inference)",
        [sys.executable, str(_FEATURE_SCRIPT), "--inference"],
    )

    if not _INFERENCE_PATH.exists():
        raise FileNotFoundError(
            f"Feature pipeline ran but did not produce {_INFERENCE_PATH}."
        )
    logger.info("Inference parquet refreshed: %s", _INFERENCE_PATH)


# ─────────────────────────────────────────────────────────────────────
# Stage 3 — Ranking inference
# ─────────────────────────────────────────────────────────────────────

def latest_per_symbol(parquet_path: Path, universe: list[str]) -> pd.DataFrame:
    """Load the inference parquet and return one row per symbol — newest date."""
    df = pd.read_parquet(parquet_path)
    if df.index.name == "date":
        df = df.reset_index()

    snapshot = (
        df[df["symbol"].isin(universe)]
        .sort_values("date")
        .groupby("symbol", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )

    missing = sorted(set(universe) - set(snapshot["symbol"].tolist()))
    if missing:
        logger.warning("No inference data for symbol(s): %s", missing)

    if snapshot.empty:
        raise RuntimeError(
            "No symbol data available for inference — universe missing entirely."
        )

    obs_dates = sorted(snapshot["date"].unique())
    logger.info(
        "Inference snapshot: %d symbols | observation date(s): %s",
        len(snapshot),
        [pd.Timestamp(d).date().isoformat() for d in obs_dates],
    )
    return snapshot


def predict_and_rank(
    booster: lgb.Booster,
    snapshot: pd.DataFrame,
    k: int,
) -> tuple[list[str], pd.DataFrame]:
    """Score every snapshot row, return top-K symbols and the ranked frame."""
    logger.info("─" * 70)
    logger.info("[Stage 3/4] Ranking inference (LightGBM LambdaRank)")
    logger.info("─" * 70)

    feature_names = booster.feature_name()
    logger.info("Booster feature count (source of truth): %d", len(feature_names))

    # Sanitize column names exactly as investor_train_model.py did
    rename_map = {
        col: _sanitize(col)
        for col in snapshot.columns
        if col not in ("date", "symbol")
    }
    X_full = snapshot.rename(columns=rename_map).copy()

    missing_in_data = [f for f in feature_names if f not in X_full.columns]
    if missing_in_data:
        logger.warning(
            "%d/%d model features absent from inference frame — filled NaN: %s",
            len(missing_in_data), len(feature_names),
            missing_in_data[:5] + (["..."] if len(missing_in_data) > 5 else []),
        )
        for f in missing_in_data:
            X_full[f] = float("nan")

    # Strict column order to match the booster's training-time layout
    X = X_full[feature_names].apply(pd.to_numeric, errors="coerce")
    scores = booster.predict(X)

    ranked = (
        snapshot.assign(score=scores)
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )

    logger.info("Cross-sectional ranking:")
    for _, row in ranked.iterrows():
        close = row.get("close")
        close_str = f"{float(close):.2f}" if pd.notna(close) else "n/a"
        logger.info(
            "  %-6s  score=%+.4f  close=%s", row["symbol"], row["score"], close_str
        )

    top_k = ranked.head(k)["symbol"].tolist()
    logger.info(
        "Top %d (target weight %.0f%% each): %s", k, TARGET_WEIGHT * 100, top_k
    )
    return top_k, ranked


# ─────────────────────────────────────────────────────────────────────
# Stage 4 — Alpaca portfolio execution
# ─────────────────────────────────────────────────────────────────────

def _build_alpaca_clients() -> tuple[TradingClient, StockHistoricalDataClient]:
    """Construct paper-trading + market-data clients from the environment."""
    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        raise RuntimeError(
            "Missing ALPACA_API_KEY or ALPACA_SECRET_KEY in environment."
        )
    trading = TradingClient(api_key=api_key, secret_key=secret_key, paper=True)
    data = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)
    return trading, data


def _fetch_live_quote(
    data_client: StockHistoricalDataClient,
    symbol: str,
) -> float | None:
    """
    Return a single representative price for *symbol* via Alpaca's latest
    quote endpoint.  Uses the bid for sell-sizing (conservative — bid is
    the price we'd actually receive).  Returns ``None`` on any failure.
    """
    try:
        req = StockLatestQuoteRequest(symbol_or_symbols=[symbol])
        quotes = data_client.get_stock_latest_quote(req)
        quote = quotes.get(symbol)
        if quote is None:
            return None
        bid = float(getattr(quote, "bid_price", 0) or 0)
        ask = float(getattr(quote, "ask_price", 0) or 0)
        if bid > 0:
            return bid
        if ask > 0:
            return ask
        return None
    except (APIError, KeyError, AttributeError, ValueError) as exc:
        logger.warning("  live-quote fetch for %s failed: %s", symbol, exc)
        return None


def execute_rebalance(
    trading: TradingClient,
    data_client: StockHistoricalDataClient,
    top_k: list[str],
    *,
    dry_run: bool,
) -> None:
    """Liquidate non-top-K positions and rebalance top-K to TARGET_WEIGHT each."""
    logger.info("─" * 70)
    logger.info("[Stage 4/4] Alpaca rebalance (dry_run=%s)", dry_run)
    logger.info("─" * 70)

    # ── Account snapshot ────────────────────────────────────────────
    try:
        account = trading.get_account()
    except APIError as exc:
        raise RuntimeError(f"get_account failed: {exc}") from exc

    equity = float(account.equity)
    buying_power = float(account.buying_power)
    cash = float(account.cash)
    usable_equity = equity * EQUITY_BUFFER
    logger.info(
        "Account: equity=$%.2f | cash=$%.2f | buying_power=$%.2f | status=%s",
        equity, cash, buying_power, account.status,
    )
    logger.info(
        "Usable equity (after %.0f%% buffer): $%.2f",
        (1 - EQUITY_BUFFER) * 100, usable_equity,
    )

    # ── Current positions ───────────────────────────────────────────
    try:
        positions = trading.get_all_positions()
    except APIError as exc:
        raise RuntimeError(f"get_all_positions failed: {exc}") from exc

    current_positions: dict[str, tuple[float, float]] = {
        p.symbol: (float(p.qty), float(p.market_value)) for p in positions
    }
    if current_positions:
        logger.info(
            "Open positions: %s",
            {s: f"qty={q:.4f} mv=${mv:.2f}" for s, (q, mv) in current_positions.items()},
        )
    else:
        logger.info("Open positions: (none)")

    # ── Stage 4a — Liquidate positions outside top-K ────────────────
    to_liquidate = [s for s in current_positions if s not in top_k]
    logger.info(
        "Positions to liquidate (not in top-%d): %s",
        TOP_K, to_liquidate or "(none)",
    )

    for symbol in to_liquidate:
        if dry_run:
            logger.info("  [DRY-RUN] would close_position(%s)", symbol)
            continue
        try:
            order = trading.close_position(symbol)
            logger.info(
                "  Liquidated %s — order_id=%s status=%s",
                symbol, order.id, order.status,
            )
        except APIError as exc:
            logger.error("  close_position(%s) failed: %s", symbol, exc)

    # ── Stage 4b — Plan rebalance trades for top-K ──────────────────
    target_value = usable_equity * TARGET_WEIGHT
    intents: list[TradeIntent] = []

    for symbol in top_k:
        current_qty, current_mv = current_positions.get(symbol, (0.0, 0.0))
        delta = target_value - current_mv

        if abs(delta) < REBALANCE_DEADBAND * equity:
            logger.info(
                "  %s within deadband (delta=$%.2f, target=$%.2f) — no trade.",
                symbol, delta, target_value,
            )
            continue

        if delta > 0:
            intents.append(TradeIntent(
                symbol=symbol,
                side=OrderSide.BUY,
                notional=round(delta, 2),
                reason=f"buy ${delta:.2f} → 50% target (current=${current_mv:.2f})",
            ))
            continue

        # delta < 0  →  sell to reduce overweight
        # Fractional sells require qty.  Pull a fresh quote at execute
        # time per architectural decision (live-quote sizing).
        price = _fetch_live_quote(data_client, symbol)
        if price is None or price <= 0:
            logger.warning(
                "  %s overweight by $%.2f but no live quote — "
                "fully closing position then re-buying to target.",
                symbol, -delta,
            )
            intents.append(TradeIntent(
                symbol=symbol,
                side=OrderSide.SELL,
                qty=current_qty,
                reason="overweight, no quote → full close",
            ))
            intents.append(TradeIntent(
                symbol=symbol,
                side=OrderSide.BUY,
                notional=round(target_value, 2),
                reason="re-establish 50% target after full close",
            ))
            continue

        sell_qty = round(min(abs(delta) / price, current_qty), 4)
        if sell_qty <= 0:
            continue
        intents.append(TradeIntent(
            symbol=symbol,
            side=OrderSide.SELL,
            qty=sell_qty,
            reason=(
                f"sell {sell_qty:.4f} @ ~${price:.2f} → 50% target "
                f"(current=${current_mv:.2f}, overweight=${-delta:.2f})"
            ),
        ))

    # ── Stage 4c — Submit sells first, then buys ────────────────────
    # Sells settle into cash before buys consume it (instant on paper).
    intents.sort(key=lambda t: 0 if t.side == OrderSide.SELL else 1)

    if not intents and not to_liquidate:
        logger.info("Portfolio already at target — no trades submitted.")
        return

    for intent in intents:
        logger.info(
            "  → %-4s %-6s | %s",
            intent.side.value.upper(), intent.symbol, intent.reason,
        )
        if dry_run:
            logger.info(
                "    [DRY-RUN] would submit MarketOrder("
                "symbol=%s, side=%s, notional=%s, qty=%s, tif=DAY)",
                intent.symbol, intent.side.value, intent.notional, intent.qty,
            )
            continue

        try:
            kwargs: dict = dict(
                symbol=intent.symbol,
                side=intent.side,
                time_in_force=TimeInForce.DAY,
            )
            if intent.notional is not None:
                kwargs["notional"] = intent.notional
            if intent.qty is not None:
                kwargs["qty"] = intent.qty

            order_request = MarketOrderRequest(**kwargs)
            order = trading.submit_order(order_request)
            logger.info(
                "    Submitted | order_id=%s status=%s",
                order.id, order.status,
            )
        except APIError as exc:
            logger.error("    submit_order failed for %s: %s", intent.symbol, exc)


# ─────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Monthly portfolio rebalance for the V4 investor pipeline."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip Alpaca order submission. Logs intended allocations only.",
    )
    parser.add_argument(
        "--skip-refresh",
        action="store_true",
        help=(
            "Use existing v4_inference_features.parquet without re-running "
            "the miner + feature pipeline subprocesses."
        ),
    )
    args = parser.parse_args(argv)

    logger.info("=" * 70)
    logger.info("V4 Portfolio Orchestrator")
    logger.info("Universe : %s", UNIVERSE)
    logger.info("Top-K    : %d  | Target weight per holding: %.0f%%",
                TOP_K, TARGET_WEIGHT * 100)
    logger.info("Mode     : %s", "DRY-RUN" if args.dry_run else "LIVE (paper)")
    logger.info("=" * 70)

    try:
        # ── Stages 1 & 2 — refresh data + features ───────────────────
        if args.skip_refresh:
            logger.info("Skipping data refresh (--skip-refresh).")
            if not _INFERENCE_PATH.exists():
                raise FileNotFoundError(
                    f"{_INFERENCE_PATH} missing — run without "
                    "--skip-refresh first."
                )
        else:
            refresh_data()
            build_inference_features()

        # ── Stage 3 — load model + predict ───────────────────────────
        if not _MODEL_PATH.exists():
            raise FileNotFoundError(f"Model artifact missing: {_MODEL_PATH}")
        booster = lgb.Booster(model_file=str(_MODEL_PATH))

        snapshot = latest_per_symbol(_INFERENCE_PATH, UNIVERSE)
        top_k, _ranked = predict_and_rank(booster, snapshot, TOP_K)

        # ── Stage 4 — Alpaca rebalance ───────────────────────────────
        trading, data_client = _build_alpaca_clients()
        execute_rebalance(trading, data_client, top_k, dry_run=args.dry_run)

        logger.info("=" * 70)
        logger.info("Portfolio orchestrator complete.")
        logger.info("=" * 70)
        return 0

    except Exception as exc:
        logger.exception("Orchestrator failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
