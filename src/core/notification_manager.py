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
