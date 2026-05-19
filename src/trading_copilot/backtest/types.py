"""Backtest data types."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Trade:
    ticker: str
    entry_date: datetime
    entry_price: float
    exit_date: datetime | None = None
    exit_price: float | None = None
    direction: str = "LONG"  # LONG | SHORT
    size: float = 1.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    exit_reason: str = ""  # signal | stop_loss | take_profit | timeout


@dataclass
class BacktestConfig:
    initial_capital: float = 100_000.0
    position_size_pct: float = 0.1  # risk 10% of capital per trade
    max_positions: int = 3
    stop_loss_pct: float = 0.03  # 3% stop loss
    take_profit_pct: float = 0.06  # 6% take profit (2:1 R/R)
    max_hold_days: int = 10
    slippage_pct: float = 0.001  # 0.1% slippage per side
    min_signal_confidence: float = 0.55


@dataclass
class BacktestResult:
    config: BacktestConfig
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    equity_dates: list[datetime] = field(default_factory=list)

    # Computed metrics
    total_return_pct: float = 0.0
    annualized_return_pct: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    win_rate: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    avg_hold_days: float = 0.0
