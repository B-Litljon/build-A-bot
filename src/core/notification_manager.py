import logging
import requests
import json
from datetime import datetime
from typing import Optional


class NotificationManager:
    def __init__(self, webhook_url: Optional[str]):
        self.webhook_url = webhook_url
        if not self.webhook_url:
            logging.warning("⚠️ No Discord Webhook URL provided. Notifications disabled.")

    def send_message(self, title: str, description: str, color: str = "blue"):
        """
        Sends a rich embed message to Discord.
        Colors: 'green' (Buy), 'red' (Sell), 'blue' (Info), 'yellow' (Warning), 'grey' (Crash)
        """
        if not self.webhook_url:
            return

        # Map colors to decimal integers for Discord Embeds
        color_map = {
            "green": 5763719,   # Success
            "red": 15548997,    # Error/Sell
            "blue": 3447003,    # Info
            "yellow": 16776960, # Warning
            "grey": 9807270     # Neutral
        }

        decimal_color = color_map.get(color, 3447003)

        payload = {
            "embeds": [
                {
                    "title": title,
                    "description": description,
                    "color": decimal_color,
                    "footer": {
                        "text": f"Build-A-Bot • {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    }
                }
            ]
        }

        try:
            response = requests.post(
                self.webhook_url,
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"}
            )
            if response.status_code != 204:
                logging.error(f"Failed to send Discord notification: {response.status_code}")
        except Exception as e:
            logging.error(f"Notification Error: {e}")

    def notify_startup(self, symbols: list):
        self.send_message(
            "🚀 Bot Started",
            f"Trading Engine is Live.\n**Mode:** Paper Trading\n**Targets:** {', '.join(symbols)}",
            "blue"
        )

    def notify_trade(self, action: str, symbol: str, price: float, quantity: float, reason: str):
        color = "green" if action == "BUY" else "red"
        emoji = "🟢" if action == "BUY" else "🔴"

        desc = (
            f"**Symbol:** {symbol}\n"
            f"**Price:** ${price:.2f}\n"
            f"**Qty:** {quantity:.4f}\n"
            f"**Total:** ${price * quantity:.2f}\n"
            f"**Reason:** {reason}"
        )
        self.send_message(f"{emoji} {action} EXECUTED", desc, color)

    def notify_error(self, context: str, error: str):
        self.send_message(
            "💀 Critical Error",
            f"**Context:** {context}\n**Error:** {str(error)}",
            "grey"
        )
