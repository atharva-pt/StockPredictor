"""Backtesting engine — simulates trading signals on historical data.

IMPORTANT RULES (to prevent misleading results):
1. Signals at bar t are executed at bar t+1 OPEN (not close)
2. Slippage is applied on both entry and exit
3. Stop loss / take profit checked against t+1 high/low, not close
4. No lookahead: signal generation uses only data available at time t
5. Position sizing respects capital limits
"""

from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import pandas as pd

from trading_copilot.backtest.types import BacktestConfig, BacktestResult, Trade
from trading_copilot.logging_setup import get_logger
from trading_copilot.signals.generator import Signal

log = get_logger("backtest.engine")


def run_backtest(
    signals: list[Signal],
    ohlcv: pd.DataFrame,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """Run backtest on historical signals.

    signals and ohlcv must be aligned — signal[i] corresponds to ohlcv.iloc[i].
    Signal at index i is FILLED at index i+1 open price.
    """
    if config is None:
        config = BacktestConfig()

    result = BacktestResult(config=config)
    capital = config.initial_capital
    open_trades: list[Trade] = []
    all_trades: list[Trade] = []

    equity = [capital]
    dates = [ohlcv.index[0].to_pydatetime() if hasattr(ohlcv.index[0], "to_pydatetime") else ohlcv.index[0]]

    for i in range(len(ohlcv) - 1):
        ohlcv.index[i]
        next_bar = ohlcv.iloc[i + 1]
        next_date = ohlcv.index[i + 1]
        next_open = next_bar["open"]
        next_high = next_bar["high"]
        next_low = next_bar["low"]

        # --- Check exits on open trades ---
        closed_trades: list[Trade] = []
        for trade in open_trades:
            exit_price, exit_reason = _check_exit(
                trade, next_open, next_high, next_low, next_date, config,
            )
            if exit_price is not None:
                trade.exit_date = _to_datetime(next_date)
                trade.exit_price = exit_price
                trade.exit_reason = exit_reason
                trade.pnl, trade.pnl_pct = _calc_pnl(trade, config.slippage_pct)
                capital += trade.pnl
                all_trades.append(trade)
                closed_trades.append(trade)

        for t in closed_trades:
            open_trades.remove(t)

        # --- Check for new entries ---
        if i < len(signals):
            sig = signals[i]
            if (
                sig.action in ("BUY", "SELL")
                and sig.confidence >= config.min_signal_confidence
                and len(open_trades) < config.max_positions
            ):
                entry_price = next_open * (1 + config.slippage_pct)
                position_capital = capital * config.position_size_pct
                size = position_capital / entry_price if entry_price > 0 else 0

                if size > 0:
                    trade = Trade(
                        ticker=sig.ticker,
                        entry_date=_to_datetime(next_date),
                        entry_price=entry_price,
                        direction="LONG" if sig.action == "BUY" else "SHORT",
                        size=size,
                    )
                    open_trades.append(trade)

        # --- Mark-to-market for equity curve ---
        unrealized = sum(_unrealized_pnl(t, next_bar["close"]) for t in open_trades)
        equity.append(capital + unrealized)
        dates.append(_to_datetime(next_date))

    # Force-close remaining positions at last bar close
    last_close = ohlcv.iloc[-1]["close"]
    last_date = ohlcv.index[-1]
    for trade in open_trades:
        trade.exit_date = _to_datetime(last_date)
        trade.exit_price = last_close * (1 - config.slippage_pct)
        trade.exit_reason = "end_of_data"
        trade.pnl, trade.pnl_pct = _calc_pnl(trade, config.slippage_pct)
        capital += trade.pnl
        all_trades.append(trade)

    result.trades = all_trades
    result.equity_curve = equity
    result.equity_dates = dates
    _compute_metrics(result)

    log.info(
        "backtest_done",
        trades=result.total_trades,
        win_rate=f"{result.win_rate:.1%}",
        total_return=f"{result.total_return_pct:.1%}",
        sharpe=f"{result.sharpe_ratio:.2f}",
        max_dd=f"{result.max_drawdown_pct:.1%}",
    )
    return result


def _check_exit(
    trade: Trade,
    next_open: float,
    next_high: float,
    next_low: float,
    next_date,
    config: BacktestConfig,
) -> tuple[float | None, str]:
    """Check if trade should exit. Returns (exit_price, reason) or (None, "")."""
    entry = trade.entry_price

    if trade.direction == "LONG":
        sl_price = entry * (1 - config.stop_loss_pct)
        tp_price = entry * (1 + config.take_profit_pct)
        if next_low <= sl_price:
            return sl_price, "stop_loss"
        if next_high >= tp_price:
            return tp_price, "take_profit"
    else:
        sl_price = entry * (1 + config.stop_loss_pct)
        tp_price = entry * (1 - config.take_profit_pct)
        if next_high >= sl_price:
            return sl_price, "stop_loss"
        if next_low <= tp_price:
            return tp_price, "take_profit"

    # Timeout
    if trade.entry_date:
        hold_days = (_to_datetime(next_date) - trade.entry_date).days
        if hold_days >= config.max_hold_days:
            return next_open, "timeout"

    return None, ""


def _calc_pnl(trade: Trade, slippage_pct: float) -> tuple[float, float]:
    if trade.exit_price is None:
        return 0.0, 0.0
    exit_adj = trade.exit_price * (1 - slippage_pct)
    if trade.direction == "LONG":
        pnl_per_unit = exit_adj - trade.entry_price
    else:
        pnl_per_unit = trade.entry_price - exit_adj
    pnl = pnl_per_unit * trade.size
    pnl_pct = pnl_per_unit / trade.entry_price if trade.entry_price > 0 else 0
    return pnl, pnl_pct


def _unrealized_pnl(trade: Trade, current_price: float) -> float:
    if trade.direction == "LONG":
        return (current_price - trade.entry_price) * trade.size
    return (trade.entry_price - current_price) * trade.size


def _to_datetime(dt) -> datetime:
    if hasattr(dt, "to_pydatetime"):
        return dt.to_pydatetime()
    return dt


def _compute_metrics(result: BacktestResult) -> None:
    trades = result.trades
    config = result.config
    equity = result.equity_curve

    result.total_trades = len(trades)
    if not trades:
        return

    # Win rate
    winners = [t for t in trades if t.pnl > 0]
    losers = [t for t in trades if t.pnl <= 0]
    result.win_rate = len(winners) / len(trades) if trades else 0

    # Avg win/loss
    result.avg_win_pct = float(np.mean([t.pnl_pct for t in winners])) if winners else 0
    result.avg_loss_pct = float(np.mean([t.pnl_pct for t in losers])) if losers else 0

    # Profit factor
    gross_profit = sum(t.pnl for t in winners)
    gross_loss = abs(sum(t.pnl for t in losers))
    result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Total return
    result.total_return_pct = (equity[-1] - config.initial_capital) / config.initial_capital

    # Annualized return
    if result.equity_dates and len(result.equity_dates) > 1:
        days = (result.equity_dates[-1] - result.equity_dates[0]).days
        if days > 0:
            result.annualized_return_pct = (1 + result.total_return_pct) ** (365 / days) - 1

    # Sharpe ratio (daily returns, annualized)
    if len(equity) > 1:
        daily_returns = np.diff(equity) / equity[:-1]
        if np.std(daily_returns) > 0:
            result.sharpe_ratio = float(np.mean(daily_returns) / np.std(daily_returns) * math.sqrt(252))

    # Max drawdown
    peak = equity[0]
    max_dd = 0.0
    for val in equity:
        peak = max(peak, val)
        dd = (peak - val) / peak
        max_dd = max(max_dd, dd)
    result.max_drawdown_pct = max_dd

    # Avg hold days
    hold_days = []
    for t in trades:
        if t.entry_date and t.exit_date:
            hold_days.append((t.exit_date - t.entry_date).days)
    result.avg_hold_days = float(np.mean(hold_days)) if hold_days else 0
