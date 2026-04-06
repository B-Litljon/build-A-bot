import asyncio
import unittest
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime, timezone
import sys
import os
import polars as pl

# Ensure src is in path
sys.path.insert(0, os.path.abspath("src"))

from execution.factory_orchestrator import FactoryOrchestrator

class TestWarmupSequence(unittest.IsolatedAsyncioTestCase):
    async def test_warmup_injection(self):
        # Mock dependencies
        mock_strategy = MagicMock()
        mock_risk = MagicMock()

        # Setup Orchestrator
        with patch('execution.factory_orchestrator.TradingClient'), \
             patch('execution.factory_orchestrator.AlpacaCryptoFeed') as MockFeed:

            # Setup mock feed behavior
            mock_feed_instance = MockFeed.return_value
            mock_feed_instance.warmup_history = AsyncMock()

            # Create dummy historical data
            now = datetime.now(timezone.utc)
            dummy_df = pl.DataFrame({
                "timestamp": [now],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.5],
                "volume": [10.0]
            })
            mock_feed_instance.warmup_history.return_value = {"BTC/USD": dummy_df}

            orchestrator = FactoryOrchestrator(
                symbols=["BTC/USD"],
                api_key="fake",
                secret_key="fake",
                strategy=mock_strategy,
                risk_manager=mock_risk
            )

            # Trigger run but stop early using the shutdown event
            orchestrator._shutdown_event.set()

            # We wrap the run call to see if it executes the warmup
            # Since we set the shutdown event, it should pull history and then stop
            await orchestrator.run()

            # VERIFY: warmup_history was called
            mock_feed_instance.warmup_history.assert_called_once_with(["BTC/USD"], lookback_minutes=300)

            # VERIFY: Aggregator has the data
            agg = orchestrator.aggregators["BTC/USD"]
            self.assertEqual(len(agg.history_df), 1)
            self.assertEqual(agg.history_df["close"][0], 100.5)
            print("\n✅ Verification Success: Warmup data correctly injected into Aggregator.")

if __name__ == "__main__":
    # Since we can't run this normally due to missing polars in the main env,
    # this script is for documentation of the verification logic.
    pass
