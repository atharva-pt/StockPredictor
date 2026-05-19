"""Tests for the Telegram alert system."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from trading_copilot.alerts.formatter import format_backtest, format_signal
from trading_copilot.alerts.telegram_bot import TelegramAlerter
from trading_copilot.backtest.types import BacktestConfig, BacktestResult
from trading_copilot.signals.generator import Signal

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_signal() -> Signal:
    return Signal(
        ticker="RELIANCE",
        timestamp=datetime(2025, 6, 1, 10, 30, tzinfo=UTC),
        action="BUY",
        confidence=0.72,
        risk_level="LOW",
        up_prob=0.68,
        down_prob=0.32,
        reasoning="Strong uptrend with positive momentum divergence.",
        factors=["RSI > 50", "MACD bullish crossover", "Volume surge"],
        model_name="xgb_v1",
    )


@pytest.fixture()
def sample_backtest() -> BacktestResult:
    return BacktestResult(
        config=BacktestConfig(),
        total_return_pct=18.42,
        annualized_return_pct=12.5,
        sharpe_ratio=1.35,
        max_drawdown_pct=-6.2,
        win_rate=0.58,
        profit_factor=1.8,
        total_trades=42,
        avg_hold_days=4.3,
    )


# ---------------------------------------------------------------------------
# Formatter tests (no network)
# ---------------------------------------------------------------------------

class TestFormatter:
    def test_format_signal_contains_action(self, sample_signal: Signal) -> None:
        msg = format_signal(sample_signal)
        assert "BUY" in msg

    def test_format_signal_contains_ticker(self, sample_signal: Signal) -> None:
        msg = format_signal(sample_signal)
        assert "RELIANCE" in msg

    def test_format_signal_contains_confidence(self, sample_signal: Signal) -> None:
        msg = format_signal(sample_signal)
        assert "72" in msg  # 72.0%

    def test_format_signal_contains_factors(self, sample_signal: Signal) -> None:
        msg = format_signal(sample_signal)
        assert "RSI" in msg
        assert "MACD" in msg

    def test_format_signal_truncates_reasoning(self, sample_signal: Signal) -> None:
        sample_signal.reasoning = "x" * 600
        msg = format_signal(sample_signal)
        # Reasoning should be truncated with "..."
        assert "\\.\\.\\." in msg or len(msg) <= 4096

    def test_format_backtest_contains_metrics(self, sample_backtest: BacktestResult) -> None:
        msg = format_backtest(sample_backtest)
        assert "18" in msg  # total return
        assert "1\\.35" in msg or "1.35" in msg  # sharpe
        assert "42" in msg  # total trades

    def test_format_signal_sell(self) -> None:
        sig = Signal(
            ticker="TCS",
            timestamp=datetime(2025, 6, 1, tzinfo=UTC),
            action="SELL",
            confidence=0.61,
            risk_level="HIGH",
            up_prob=0.35,
            down_prob=0.65,
            reasoning="Bearish breakdown below support.",
        )
        msg = format_signal(sig)
        assert "SELL" in msg
        assert "TCS" in msg


# ---------------------------------------------------------------------------
# TelegramAlerter tests (mocked network)
# ---------------------------------------------------------------------------

class TestTelegramAlerter:
    def test_not_configured_logs_warning(self) -> None:
        alerter = TelegramAlerter()
        assert not alerter.configured

    def test_not_configured_send_returns_false(self) -> None:
        alerter = TelegramAlerter()
        assert alerter.send_text("hello") is False

    @patch("trading_copilot.alerts.telegram_bot.requests.post")
    def test_send_text_success(self, mock_post: MagicMock) -> None:
        mock_post.return_value = MagicMock(status_code=200)
        alerter = TelegramAlerter(bot_token="fake:token", chat_id="12345")
        assert alerter.configured
        result = alerter.send_text("test message")
        assert result is True
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        assert call_kwargs[1]["json"]["chat_id"] == "12345"
        assert call_kwargs[1]["json"]["text"] == "test message"

    @patch("trading_copilot.alerts.telegram_bot.requests.post")
    def test_send_signal_alert(self, mock_post: MagicMock, sample_signal: Signal) -> None:
        mock_post.return_value = MagicMock(status_code=200)
        alerter = TelegramAlerter(bot_token="fake:token", chat_id="12345")
        result = alerter.send_signal_alert(sample_signal)
        assert result is True
        payload = mock_post.call_args[1]["json"]
        assert payload["parse_mode"] == "MarkdownV2"
        assert "BUY" in payload["text"]

    @patch("trading_copilot.alerts.telegram_bot.requests.post")
    def test_send_backtest_summary(self, mock_post: MagicMock, sample_backtest: BacktestResult) -> None:
        mock_post.return_value = MagicMock(status_code=200)
        alerter = TelegramAlerter(bot_token="fake:token", chat_id="12345")
        result = alerter.send_backtest_summary(sample_backtest)
        assert result is True

    @patch("trading_copilot.alerts.telegram_bot.requests.post")
    def test_api_error_returns_false(self, mock_post: MagicMock) -> None:
        mock_post.return_value = MagicMock(status_code=400, text="Bad Request")
        alerter = TelegramAlerter(bot_token="fake:token", chat_id="12345")
        result = alerter.send_text("test")
        assert result is False

    @patch("trading_copilot.alerts.telegram_bot.requests.post")
    def test_network_error_returns_false(self, mock_post: MagicMock) -> None:
        mock_post.side_effect = ConnectionError("Network down")
        alerter = TelegramAlerter(bot_token="fake:token", chat_id="12345")
        result = alerter.send_text("test")
        assert result is False


# ---------------------------------------------------------------------------
# Rate limiting test
# ---------------------------------------------------------------------------

class TestRateLimiting:
    @patch("trading_copilot.alerts.telegram_bot.requests.post")
    @patch("trading_copilot.alerts.telegram_bot.time.sleep")
    def test_rate_limit_sleeps_between_sends(self, mock_sleep: MagicMock, mock_post: MagicMock) -> None:
        mock_post.return_value = MagicMock(status_code=200)
        alerter = TelegramAlerter(bot_token="fake:token", chat_id="12345")

        alerter.send_text("first")
        alerter.send_text("second")

        # The second send should trigger a sleep since <1s elapsed.
        assert mock_sleep.called
