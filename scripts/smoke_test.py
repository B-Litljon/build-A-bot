"""
Smoke test — Factory path verification (DO NOT COMMIT)

Verifies:
  1. FactoryOrchestrator instantiates cleanly against Alpaca Paper keys.
  2. A3 chop filter: calculate_bracket() returns None when 0.5x ATR < 0.15% floor.
  3. $50 min notional: calculate_quantity() returns 0.0 for zombie trades.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from execution.risk_manager import RiskManager, RiskProfile
from execution.factory_orchestrator import FactoryOrchestrator
from strategies.concrete_strategies.ml_strategy import MLStrategy
from data.feed import AlpacaCryptoFeed

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

def check(label, condition, got):
    status = PASS if condition else FAIL
    print(f"  [{status}] {label}")
    if not condition:
        print(f"         got: {got!r}")
    return condition

all_passed = True

# ── Test 1: FactoryOrchestrator instantiation ─────────────────────────────────
print("\n=== Test 1: FactoryOrchestrator instantiation ===")
try:
    api_key    = os.environ["ALPACA_API_KEY"]
    secret_key = os.environ["ALPACA_SECRET_KEY"]

    rm       = RiskManager()
    strategy = MLStrategy()
    feed     = AlpacaCryptoFeed(api_key=api_key, secret_key=secret_key)

    fo = FactoryOrchestrator(
        symbols=["BTC/USD"],
        api_key=api_key,
        secret_key=secret_key,
        strategy=strategy,
        risk_manager=rm,
        feed=feed,
        paper=True,
    )
    ok = check("FactoryOrchestrator instantiates without exception", True, None)
except Exception as e:
    ok = check("FactoryOrchestrator instantiates without exception", False, e)
all_passed = all_passed and ok

# ── Test 2: A3 chop filter ────────────────────────────────────────────────────
print("\n=== Test 2: A3 chop filter (calculate_bracket returns None) ===")
# entry=100.0, raw_atr=0.2
# sl_dist = 0.5 * 0.2 = 0.10
# floor   = 100.0 * 0.0015 = 0.15
# 0.10 < 0.15 → must return None
rm2    = RiskManager(RiskProfile())
result = rm2.calculate_bracket(entry_price=100.0, raw_atr=0.2)
ok = check("calculate_bracket(entry=100.0, raw_atr=0.2) == None", result is None, result)
all_passed = all_passed and ok

# Bonus: confirm a healthy signal passes through (atr large enough)
# sl_dist = 0.5 * 0.5 = 0.25 > 100.0 * 0.0015 = 0.15 → should return tuple
result_ok = rm2.calculate_bracket(entry_price=100.0, raw_atr=0.5)
ok2 = check("calculate_bracket(entry=100.0, raw_atr=0.5) returns a tuple",
            isinstance(result_ok, tuple) and len(result_ok) == 2, result_ok)
if ok2:
    print(f"         got: sl_dist={result_ok[0]}, tp_dist={result_ok[1]}")
all_passed = all_passed and ok2

# ── Test 3: $50 minimum notional ──────────────────────────────────────────────
print("\n=== Test 3: $50 minimum notional (calculate_quantity returns 0.0) ===")
# cash=1000, is_crypto=True, entry=3000, sl=2900
# risk_dollars = equity * 0.02 — use equity=1000 too
# risk_per_share = 3000 - 2900 = 100
# risk_qty = (1000 * 0.02) / 100 = 0.2
# bp_qty   = (1000 * 0.95) / 3000 ≈ 0.3167
# notional_qty = 100000 / 3000 ≈ 33.3
# final_qty = min(0.2, 33.3, 0.3167) = 0.2
# notional  = 0.2 * 3000 = 600 → this passes the floor
#
# To trigger the $50 floor we need a tiny account.
# cash=50, equity=50: risk_qty=(50*0.02)/100=0.01; notional=0.01*3000=30 < 50 → 0.0
qty = rm2.calculate_quantity(
    equity=50.0,
    buying_power=0.0,
    entry_price=3000.0,
    sl_price=2900.0,
    cash=50.0,
    is_crypto=True,
)
ok = check("calculate_quantity returns 0.0 when notional < $50", qty == 0.0, qty)
all_passed = all_passed and ok

# Bonus: confirm a healthy size passes through
qty_ok = rm2.calculate_quantity(
    equity=10000.0,
    buying_power=0.0,
    entry_price=3000.0,
    sl_price=2900.0,
    cash=10000.0,
    is_crypto=True,
)
ok2 = check("calculate_quantity returns > 0.0 for healthy account", qty_ok > 0.0, qty_ok)
if ok2:
    print(f"         got: qty={qty_ok} (notional=${qty_ok * 3000:.2f})")
all_passed = all_passed and ok2

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
if all_passed:
    print(f"  [{PASS}] All smoke tests passed. Factory path is GO.")
else:
    print(f"  [{FAIL}] One or more tests failed. See above.")
print("=" * 50 + "\n")

sys.exit(0 if all_passed else 1)
