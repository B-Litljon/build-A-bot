#!/usr/bin/env python3
import sys, os, logging
logging.disable(logging.CRITICAL)
for name in logging.Logger.manager.loggerDict:
    logging.getLogger(name).setLevel(logging.CRITICAL)
    logging.getLogger(name).propagate = False
sys.path.insert(0, os.path.abspath('src'))
import polars as pl
from datetime import datetime, timezone
from strategies.concrete_strategies.ml_strategy import MLStrategy
from utils.bar_aggregator import LiveBarAggregator

df = pl.read_parquet('data/raw/SPY_1min.parquet')
df = df.with_columns(pl.col('timestamp').dt.replace_time_zone('UTC'))
start = datetime(2024, 1, 1, tzinfo=timezone.utc)
test_df = df.filter(pl.col('timestamp') >= start)
print(f'Threshold 0.60 Test: {len(test_df)} bars')

strategy = MLStrategy(model_path='src/ml/models/rf_model.joblib', threshold=0.60)
op = strategy.get_order_params()

class BOM:
    def __init__(self, op, cap):
        self.capital = cap
        self.order_params = op
        self.active_orders = {}
        self.trades = []
    def place_order(self, sig, cap):
        risk = cap * self.order_params.risk_percentage
        qty = risk / sig.price
        sl = sig.price * self.order_params.sl_multiplier
        tp = sig.price * self.order_params.tp_multiplier
        oid = f'o{len(self.trades)}'
        self.active_orders[oid] = {'symbol': sig.symbol, 'entry_price': sig.price, 'quantity': qty, 'stop_loss': sl, 'take_profit': tp}
        return oid
    def monitor_orders(self, md):
        for oid, det in list(self.active_orders.items()):
            sym = det['symbol']
            if sym not in md: continue
            bar = md[sym]
            exit_price = None
            if bar['low'] <= det['stop_loss']:
                exit_price = det['stop_loss']
            elif bar['high'] >= det['take_profit']:
                exit_price = det['take_profit']
            if exit_price:
                pnl = (exit_price - det['entry_price']) * det['quantity']
                self.capital += pnl
                self.trades.append({'pnl': pnl})
                del self.active_orders[oid]

bom = BOM(op, 10000.0)
lba = LiveBarAggregator(timeframe=1, history_size=400)
symbol = 'SPY'

print('Running...')
for i, row in enumerate(test_df.iter_rows(named=True)):
    if i % 100000 == 0 and i > 0:
        print(f'  {i}/{len(test_df)} bars, {len(bom.trades)} trades')
    bom.monitor_orders({symbol: row})
    is_new = lba.add_bar({'timestamp': row['timestamp'], 'open': row['open'], 'high': row['high'], 'low': row['low'], 'close': row['close'], 'volume': row['volume']})
    if is_new:
        hist = lba.history_df
        if len(hist) >= strategy.warmup_period:
            sigs = strategy.analyze({symbol: hist})
            for sig in sigs:
                if sig.type == 'BUY' and not any(d['symbol'] == symbol for d in bom.active_orders.values()):
                    bom.place_order(sig, bom.capital)

trades = bom.trades
print(f'\nDone! {len(trades)} trades')
if trades:
    total = sum(t['pnl'] for t in trades)
    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] < 0]
    gp = sum(t['pnl'] for t in wins)
    gl = abs(sum(t['pnl'] for t in losses))
    pf = gp / gl if gl > 0 else float('inf')
    wr = len(wins) / len(trades)
    print(f'  Total Trades: {len(trades)}')
    print(f'  Win Rate: {wr:.1%}')
    print(f'  Profit Factor: {pf:.2f}')
    print(f'  Total PnL: ${total:.2f}')
    if pf > 1.5:
        print('  ✅ PROFIT FACTOR > 1.5 - SNIPER CONFIRMED!')
    else:
        print(f'  ❌ PF = {pf:.2f} (need > 1.5)')
