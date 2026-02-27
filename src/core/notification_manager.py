import os
import logging
import requests
from datetime import datetime
from typing import Optional
from core.signal import Signal, SignalType

logger = logging.getLogger(__name__)


class NotificationManager:
    def __init__(self, webhook_url: Optional[str] = None):
        self.webhook_url = webhook_url or os.getenv("DISCORD_WEBHOOK_URL")
        if not self.webhook_url:
            logger.warning("DISCORD_WEBHOOK_URL not set. Notifications disabled.")

    def send_trade_alert(self, signal: Signal, action: str = "ENTRY"):
        """Sends a formatted Meta-Labeling trade alert to Discord."""
        if not self.webhook_url:
            return

        if action == "ENTRY":
            title = f"🎯 UNIVERSAL SCALPER: {signal.type.value} {signal.symbol}"
            color = 0x00FF00 if signal.type == SignalType.BUY else 0xFF0000
        else:
            title = f"🏁 TRADE CLOSED: {signal.symbol}"
            color = 0x00A2FF

        description = f"💵 **Price:** ${signal.price:.2f}\n"

        # Check for Meta-Labeling data
        if "angel_prob" in signal.metadata and "devil_prob" in signal.metadata:
            angel_pct = signal.metadata["angel_prob"] * 100
            devil_pct = signal.metadata["devil_prob"] * 100
            description += f"👼 **Angel (Direction):** {angel_pct:.1f}%\n"
            description += f"😈 **Devil (Conviction):** {devil_pct:.1f}%\n"
        else:
            description += f"📊 **Confidence:** {signal.confidence * 100:.1f}%\n"

        # Append Stop Loss and Take Profit for Entry alerts
        if (
            action == "ENTRY"
            and "sl_price" in signal.metadata
            and "tp_price" in signal.metadata
        ):
            description += f"\n🛑 **Stop Loss:** ${signal.metadata['sl_price']:.4f}\n"
            description += f"🚀 **Take Profit:** ${signal.metadata['tp_price']:.4f}\n"

        payload = {
            "username": "Build-A-Bot Executive",
            "embeds": [
                {
                    "title": title,
                    "description": description,
                    "color": color,
                    "footer": {"text": f"Timestamp: {signal.timestamp}"},
                }
            ],
        }

        try:
            response = requests.post(self.webhook_url, json=payload, timeout=5)
            response.raise_for_status()
        except Exception as e:
            logger.error(f"Failed to send Discord notification: {e}")

    def send_system_message(self, message: str):
        """Sends generic system status updates."""
        if not self.webhook_url:
            return

        payload = {"content": f"🤖 **System Update:** {message}"}
        try:
            requests.post(self.webhook_url, json=payload, timeout=5)
        except Exception as e:
            logger.error(f"Failed to send Discord system message: {e}")

    def send_drift_alert(self, metrics: dict):
        """Sends a critical drift alert to Discord for The Accountant."""
        if not self.webhook_url:
            return

        # Determine alert severity
        brier = metrics.get("brier_score", 0)
        ev = metrics.get("expected_value", 0)

        if brier > 0.30 or ev < -0.001:
            severity = "🔴 CRITICAL"
            color = 0xFF0000
        else:
            severity = "⚠️ WARNING"
            color = 0xFFA500

        # Build description
        description = (
            f"**Model Performance Degradation Detected**\n\n"
            f"📊 **Win Rate:** {metrics.get('win_rate', 0):.2%}\n"
            f"💰 **Expected Value:** {metrics.get('expected_value', 0):.4f} ({metrics.get('expected_value', 0) * 100:.2f}%)\n"
            f"🎯 **Brier Score:** {metrics.get('brier_score', 0):.4f}\n"
            f"📉 **Log Loss:** {metrics.get('log_loss', 0):.4f}\n\n"
            f"⚡ **Recommendation:** Review model retraining pipeline"
        )

        payload = {
            "username": "The Accountant",
            "embeds": [
                {
                    "title": f"{severity}: DEVIL MODEL DRIFT",
                    "description": description,
                    "color": color,
                    "footer": {"text": f"Alert Time: {datetime.now().isoformat()}"},
                }
            ],
        }

        try:
            response = requests.post(self.webhook_url, json=payload, timeout=5)
            response.raise_for_status()
            logger.info(f"Drift alert sent to Discord")
        except Exception as e:
            logger.error(f"Failed to send drift alert: {e}")
