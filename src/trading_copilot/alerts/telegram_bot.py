"""Telegram alerter — sends trading signals via Telegram Bot API."""

from __future__ import annotations

import time

import requests

from trading_copilot.alerts.formatter import format_backtest, format_signal
from trading_copilot.logging_setup import get_logger

log = get_logger("alerts.telegram")

_API_BASE = "https://api.telegram.org/bot{token}/sendMessage"

# Minimum interval between messages (seconds) to respect Telegram rate limits.
_MIN_INTERVAL = 1.0


class TelegramAlerter:
    """Sends trading alerts to a Telegram chat.

    If *bot_token* or *chat_id* are empty / None the alerter becomes a no-op
    and simply logs a warning on each send attempt.
    """

    def __init__(self, bot_token: str | None = None, chat_id: str | None = None) -> None:
        self._token = bot_token or ""
        self._chat_id = chat_id or ""
        self._last_send: float = 0.0
        self._configured = bool(self._token and self._chat_id)
        if not self._configured:
            log.warning("telegram_alerter_not_configured", hint="Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID")

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def configured(self) -> bool:
        return self._configured

    def send_signal_alert(self, signal: object) -> bool:
        """Format and send a Signal as a Telegram message. Returns True on success."""
        text = format_signal(signal)
        return self._send(text, parse_mode="MarkdownV2")

    def send_backtest_summary(self, result: object) -> bool:
        """Format and send a BacktestResult summary. Returns True on success."""
        text = format_backtest(result)
        return self._send(text, parse_mode="MarkdownV2")

    def send_text(self, message: str) -> bool:
        """Send a plain text message (no markdown). Returns True on success."""
        return self._send(message, parse_mode=None)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _send(self, text: str, *, parse_mode: str | None) -> bool:
        if not self._configured:
            log.warning("telegram_send_skipped", reason="not configured")
            return False

        # Rate-limit: wait if needed.
        elapsed = time.monotonic() - self._last_send
        if elapsed < _MIN_INTERVAL:
            time.sleep(_MIN_INTERVAL - elapsed)

        url = _API_BASE.format(token=self._token)
        payload: dict[str, str] = {"chat_id": self._chat_id, "text": text}
        if parse_mode:
            payload["parse_mode"] = parse_mode

        try:
            resp = requests.post(url, json=payload, timeout=10)
            self._last_send = time.monotonic()
            if resp.status_code == 200:
                log.info("telegram_message_sent", chat_id=self._chat_id)
                return True
            log.error("telegram_api_error", status=resp.status_code, body=resp.text[:300])
            return False
        except (requests.RequestException, OSError) as exc:
            self._last_send = time.monotonic()
            log.error("telegram_send_failed", error=str(exc))
            return False
