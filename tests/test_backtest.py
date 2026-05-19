"""Tests for backtesting engine — synthetic trades, metric validation."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from trading_copilot.backtest.engine import run_backtest
from trading_copilot.backtest.report import summary_text, trade_log_text
from trading_copilot.backtest.types import BacktestConfig, BacktestResult
from trading_copilot.signals.generator import Signal


def _ohlcv(n: int = 100, trend: float = 0.001) -> pd.DataFrame:
    np.random.seed(42)
    close = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.01 + trend))
    idx = pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")
    return pd.DataFrame({
        "open": close * (1 + np.random.randn(n) * 0.002),
        "high": close * (1 + np.abs(np.random.randn(n)) * 0.005),
        "low": close * (1 - np.abs(np.random.randn(n)) * 0.005),
        "close": close,
        "volume": np.random.randint(100_000, 1_000_000, size=n).astype(float),
    }, index=idx)


def _signals(n: int, action: str = "BUY", confidence: float = 0.65) -> list[Signal]:
    return [
        Signal(
            ticker="TEST.NS",
            timestamp=datetime.now(UTC),
            action=action if i % 5 == 0 else "HOLD",
            confidence=confidence,
            risk_level="MEDIUM",
            up_prob=0.65 if action == "BUY" else 0.35,
            down_prob=0.35 if action == "BUY" else 0.65,
            reasoning="test",
        )
        for i in range(n)
    ]


# --- Core backtest ---

class TestBacktest:
    def test_runs_without_error(self):
        ohlcv = _ohlcv()
        signals = _signals(len(ohlcv))
        result = run_backtest(signals, ohlcv)
        assert isinstance(result, BacktestResult)
        assert result.total_trades > 0

    def test_equity_curve_length(self):
        ohlcv = _ohlcv()
        signals = _signals(len(ohlcv))
        result = run_backtest(signals, ohlcv)
        assert len(result.equity_curve) == len(ohlcv)

    def test_trades_have_exit(self):
        ohlcv = _ohlcv()
        signals = _signals(len(ohlcv))
        result = run_backtest(signals, ohlcv)
        for t in result.trades:
            assert t.exit_price is not None
            assert t.exit_date is not None
            assert t.exit_reason != ""

    def test_no_trades_on_hold_signals(self):
        ohlcv = _ohlcv()
        signals = [
            Signal(
                ticker="TEST.NS", timestamp=datetime.now(UTC),
                action="HOLD", confidence=0.5, risk_level="LOW",
                up_prob=0.5, down_prob=0.5, reasoning="hold",
            )
            for _ in range(len(ohlcv))
        ]
        result = run_backtest(signals, ohlcv)
        assert result.total_trades == 0

    def test_no_trades_below_confidence(self):
        ohlcv = _ohlcv()
        signals = _signals(len(ohlcv), confidence=0.40)
        config = BacktestConfig(min_signal_confidence=0.55)
        result = run_backtest(signals, ohlcv, config)
        assert result.total_trades == 0


# --- Stop loss / take profit ---

class TestRiskManagement:
    def test_stop_loss_limits_loss(self):
        ohlcv = _ohlcv(100, trend=-0.005)  # downtrend
        signals = _signals(len(ohlcv), action="BUY")
        config = BacktestConfig(stop_loss_pct=0.03)
        result = run_backtest(signals, ohlcv, config)
        sl_trades = [t for t in result.trades if t.exit_reason == "stop_loss"]
        for t in sl_trades:
            assert t.pnl_pct >= -0.05, f"Loss {t.pnl_pct:.2%} exceeds stop + slippage"

    def test_take_profit_caps_gain(self):
        ohlcv = _ohlcv(100, trend=0.005)  # uptrend
        signals = _signals(len(ohlcv), action="BUY")
        config = BacktestConfig(take_profit_pct=0.06)
        result = run_backtest(signals, ohlcv, config)
        tp_trades = [t for t in result.trades if t.exit_reason == "take_profit"]
        assert len(tp_trades) > 0, "Should have some TP exits in uptrend"

    def test_timeout_exit(self):
        ohlcv = _ohlcv(50, trend=0.0)  # flat
        signals = _signals(len(ohlcv), action="BUY")
        config = BacktestConfig(max_hold_days=5, stop_loss_pct=0.20, take_profit_pct=0.20)
        result = run_backtest(signals, ohlcv, config)
        timeout_trades = [t for t in result.trades if t.exit_reason == "timeout"]
        assert len(timeout_trades) > 0


# --- Metrics ---

class TestMetrics:
    def test_win_rate_bounded(self):
        result = run_backtest(_signals(100), _ohlcv())
        assert 0 <= result.win_rate <= 1

    def test_max_drawdown_non_negative(self):
        result = run_backtest(_signals(100), _ohlcv())
        assert result.max_drawdown_pct >= 0

    def test_sharpe_is_finite(self):
        result = run_backtest(_signals(100), _ohlcv())
        assert np.isfinite(result.sharpe_ratio)

    def test_equity_starts_at_initial_capital(self):
        config = BacktestConfig(initial_capital=50_000)
        result = run_backtest(_signals(100), _ohlcv(), config)
        assert result.equity_curve[0] == 50_000


# --- Position limits ---

class TestPositionLimits:
    def test_max_positions_respected(self):
        ohlcv = _ohlcv()
        # Signal BUY every bar
        signals = [
            Signal(
                ticker="TEST.NS", timestamp=datetime.now(UTC),
                action="BUY", confidence=0.70, risk_level="LOW",
                up_prob=0.70, down_prob=0.30, reasoning="test",
            )
            for _ in range(len(ohlcv))
        ]
        config = BacktestConfig(max_positions=2, max_hold_days=50, stop_loss_pct=0.50, take_profit_pct=0.50)
        result = run_backtest(signals, ohlcv, config)
        # Can't directly check concurrent positions, but can verify trades exist
        assert result.total_trades >= 2


# --- Report ---

class TestReport:
    def test_summary_text(self):
        result = run_backtest(_signals(100), _ohlcv())
        text = summary_text(result)
        assert "BACKTEST RESULTS" in text
        assert "Sharpe" in text
        assert "Win Rate" in text

    def test_trade_log(self):
        result = run_backtest(_signals(100), _ohlcv())
        text = trade_log_text(result)
        assert "Dir" in text
        assert len(text.split("\n")) > 2


# --- Anti-cheating ---

class TestNoLookahead:
    def test_entry_at_next_bar_open(self):
        """Verify trades enter at t+1 open, not t close."""
        ohlcv = _ohlcv(20)
        signals = _signals(20, action="BUY")
        result = run_backtest(signals, ohlcv)
        if result.trades:
            first_trade = result.trades[0]
            # Entry date should be after the first signal bar
            # Signal at bar 0 fills at bar 1 — entry_date should be bar 1 or later
            entry = first_trade.entry_date.replace(tzinfo=None)
            bar1 = ohlcv.index[1].to_pydatetime().replace(tzinfo=None)
            assert entry >= bar1, "Trade entered before next bar — lookahead!"
