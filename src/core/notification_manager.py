"""
Discord webhook notifications -- the bot's only outbound human channel.

Every alert the system sends (entries, exits, system status, retraining
verdicts, drift warnings) is posted here as a Discord embed. If the webhook
environment variable is unset the whole class degrades to a silent no-op, so
the trading paths can call it unconditionally. Network failures are logged and
swallowed: a Discord outage must never take down live trading.

Called from the live orchestrators (Alpaca and OANDA), MLStrategy, and the
offline feedback_loop / retrainer pipelines.

Glossary:
    webhook_url -- the Discord incoming-webhook URL, taken from the
        DISCORD_WEBHOOK_URL environment variable unless passed explicitly.
        None/unset disables every send in this class.
    send_trade_alert -- entry/exit embed for the Alpaca path. Takes a
        ``core.signal.Signal`` and reads angel_prob, devil_prob, sl_price,
        tp_price and expected_pct_growth out of its ``metadata`` dict.
        On close actions metadata may also carry ``close_price`` and
        ``hit_level`` ("TP"/"SL"), which replace the price line.
    send_oanda_trade_alert -- the same idea for the forex path, but takes loose
        primitives instead of a Signal, because the OANDA strategy emits the
        other Signal shape (``strategies.base.Signal``). Keyword-only.
        On close actions ``close_price`` and ``hit_level`` provide the
        exit price and which bracket was breached.
    send_system_message -- one-line plain status update, no embed. Used for
        startup, shutdown and connection events.
    send_retraining_report -- posts the full validation-gate verdict after a
        retrain: per-fold Brier/EV/win-rate/trade-count, the aggregate scores,
        and PROMOTED vs REJECTED with the rejection reasons. Takes the
        ``ValidationReport`` from retrainer.validate_candidate().
    send_drift_alert -- posts degradation warnings from feedback_loop.py's
        DriftEvaluator; escalates from WARNING to CRITICAL when Brier > 0.30 or
        expected value < -0.001.
    action -- "ENTRY" vs anything else; ENTRY embeds are green/red by direction
        and include the bracket levels, closes are blue and show close_price
        with the exit reason (hit_level: "TP" or "SL").
    close_price -- the price at which the position was closed (exit-side only).
        Displayed prominently; none produces the legacy single-price line.
    hit_level -- "TP" or "SL" — which bracket was breached, shown with emoji.
    username -- the Discord display name, used as a crude persona tag so the
        source is obvious at a glance: "Build-A-Bot Executive" (Alpaca),
        "Build-A-Bot V5 Forex" (OANDA), "The Accountant" (retraining and
        drift).
    color -- Discord embed sidebar colour as an integer: 0x00FF00 green (long /
        promoted), 0xFF0000 red (short / rejected / critical), 0x00A2FF blue
        (trade closed), 0xFFA500 orange (warning).
    timeout=5 -- every POST is capped at 5 seconds so a hung webhook cannot
        stall the caller's thread.
"""

import os
import logging
import requests
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


class NotificationManager:
    def __init__(self, webhook_url: Optional[str] = None):
        self.webhook_url = webhook_url or os.getenv("DISCORD_WEBHOOK_URL")
        if not self.webhook_url:
            logger.warning("DISCORD_WEBHOOK_URL not set. Notifications disabled.")

    def send_trade_alert(self, signal, action: str = "ENTRY"):
        """Sends a formatted Meta-Labeling trade alert to Discord."""
        from core.signal import (
            SignalType,
        )  # lazy import — only needed in live orchestrator context

        if not self.webhook_url:
            return

        if action == "ENTRY":
            title = f"🎯 UNIVERSAL SCALPER: {signal.type.value} {signal.symbol}"
            color = 0x00FF00 if signal.type == SignalType.BUY else 0xFF0000
        else:
            title = f"🏁 TRADE CLOSED: {signal.symbol}"
            color = 0x00A2FF

        close_price = signal.metadata.get("close_price") if action != "ENTRY" else None
        if close_price is not None:
            description = f"💵 **Close Price:** ${close_price:.2f}\n"
            if signal.price:
                description += f"📥 **Entry Price:** ${signal.price:.2f}\n"
            hit = signal.metadata.get("hit_level")
            if hit:
                emoji = "🚀" if hit == "TP" else "🛑"
                description += f"{emoji} **{hit} HIT**\n"
        else:
            description = f"💵 **Price:** ${signal.price:.2f}\n"

        # Check for Meta-Labeling data
        if "angel_prob" in signal.metadata and "devil_prob" in signal.metadata:
            angel_pct = signal.metadata["angel_prob"] * 100
            devil_pct = signal.metadata["devil_prob"] * 100
            description += f"👼 **Angel (Direction):** {angel_pct:.1f}%\n"
            description += f"😈 **Devil (Conviction):** {devil_pct:.1f}%\n"
        else:
            description += f"📊 **Confidence:** {signal.confidence * 100:.1f}%\n"

        # Append Stop Loss and Take Profit for both entry and close alerts
        if (
            "sl_price" in signal.metadata
            and "tp_price" in signal.metadata
        ):
            description += f"\n🛑 **Stop Loss:** ${signal.metadata['sl_price']:.4f}\n"
            description += f"🚀 **Take Profit:** ${signal.metadata['tp_price']:.4f}\n"
            if action == "ENTRY" and "expected_pct_growth" in signal.metadata:
                description += f"📈 **Projected Growth:** {signal.metadata['expected_pct_growth']:.2f}%\n"

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

    def send_oanda_trade_alert(
        self,
        *,
        symbol: str,
        direction: str,
        action: str,
        price: float,
        units: int,
        sl_price: Optional[float] = None,
        tp_price: Optional[float] = None,
        angel_prob: Optional[float] = None,
        devil_prob: Optional[float] = None,
        reason: Optional[str] = None,
        timestamp: Optional[str] = None,
        close_price: Optional[float] = None,
        hit_level: Optional[str] = None,
    ) -> None:
        """
        Discord alert for the OANDA forex bot path.

        Decoupled from core.signal.Signal because the V5 forex strategy
        emits a different Signal shape (strategies.base.Signal). Accepts
        primitives so each orchestrator passes whatever fields it has.
        """
        if not self.webhook_url:
            return

        is_long = direction.lower() == "long"
        if action == "ENTRY":
            emoji = "📈" if is_long else "📉"
            title = f"{emoji} OANDA FOREX ENTRY: {direction.upper()} {symbol}"
            color = 0x00FF00 if is_long else 0xFF0000
        else:
            title = f"🏁 OANDA FOREX {action}: {symbol}"
            color = 0x00A2FF

        if close_price is not None:
            description = f"💵 **Close Price:** {close_price:.5f}\n"
            if price:
                description += f"📥 **Entry Price:** {price:.5f}\n"
            if hit_level:
                emoji = "🚀" if hit_level == "TP" else "🛑"
                description += f"{emoji} **{hit_level} HIT**\n"
        else:
            description = f"💵 **Price:** {price:.5f}\n"
        description += f"📦 **Units:** {units}\n"

        if angel_prob is not None and devil_prob is not None:
            description += f"👼 **Angel:** {angel_prob * 100:.1f}%\n"
            description += f"😈 **Devil:** {devil_prob * 100:.1f}%\n"

        if sl_price is not None and tp_price is not None:
            description += f"\n🛑 **SL:** {sl_price:.5f}\n"
            description += f"🚀 **TP:** {tp_price:.5f}\n"

        if reason:
            description += f"\n📝 **Reason:** {reason}\n"

        payload = {
            "username": "Build-A-Bot V5 Forex",
            "embeds": [
                {
                    "title": title,
                    "description": description,
                    "color": color,
                    "footer": {
                        "text": f"Timestamp: {timestamp or datetime.now(timezone.utc).isoformat()}"
                    },
                }
            ],
        }

        try:
            response = requests.post(self.webhook_url, json=payload, timeout=5)
            response.raise_for_status()
        except Exception as e:
            logger.error(f"Failed to send Discord OANDA trade alert: {e}")

    def send_retraining_report(self, report, promoted: bool):
        """
        Sends a detailed retraining report to Discord.

        Uses "The Accountant" persona.  Shows per-fold metrics, aggregate
        scores, and the final PROMOTED / REJECTED verdict.

        Args:
            report: ValidationReport dataclass from retrainer.validate_candidate()
            promoted: Whether the model was promoted (True) or rejected (False)
        """
        if not self.webhook_url:
            return

        # Build per-fold summary
        fold_lines = []
        for fm in report.fold_metrics:
            fold_lines.append(
                f"Fold {fm.fold_number}: "
                f"Brier={fm.brier_score:.4f} | "
                f"EV={fm.expected_value:.6f} | "
                f"WR={fm.win_rate:.1%} | "
                f"Trades={fm.devil_approved_trades}"
            )
        fold_text = "\n".join(fold_lines)

        if promoted:
            verdict = "✅ PROMOTED"
            color = 0x00FF00  # Green
            desc_header = "**New models passed all validation gates and are now live.**"
        else:
            verdict = "🚫 REJECTED"
            color = 0xFF0000  # Red
            reasons_text = "\n".join(f"• {r}" for r in report.rejection_reasons)
            desc_header = (
                f"**Models failed validation. Production weights retained.**\n\n"
                f"**Rejection Reasons:**\n{reasons_text}"
            )

        description = (
            f"{desc_header}\n\n"
            f"**Per-Fold Results:**\n```\n{fold_text}\n```\n\n"
            f"📊 **Mean Brier Score:** {report.mean_brier:.4f} (threshold ≤ 0.25)\n"
            f"💰 **Mean EV:** {report.mean_ev:.6f} (threshold ≥ 0.0005)\n"
            f"📈 **Final Profit Factor:** {report.final_profit_factor:.2f} (threshold ≥ 1.2)\n"
            f"🎯 **Final Win Rate:** {report.final_win_rate:.1%}\n"
            f"📋 **Final Trades:** {report.final_total_trades}"
        )

        payload = {
            "username": "The Accountant",
            "embeds": [
                {
                    "title": f"{verdict}: MODEL RETRAINING REPORT",
                    "description": description,
                    "color": color,
                    "footer": {"text": f"Report Time: {datetime.now().isoformat()}"},
                }
            ],
        }

        try:
            response = requests.post(self.webhook_url, json=payload, timeout=5)
            response.raise_for_status()
            logger.info(f"Retraining report sent to Discord (verdict: {verdict})")
        except Exception as e:
            logger.error(f"Failed to send retraining report: {e}")

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
