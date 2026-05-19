"""Backtest reporting — text summary and trade log."""

from __future__ import annotations

from trading_copilot.backtest.types import BacktestResult


def summary_text(result: BacktestResult) -> str:
    lines = [
        "=" * 50,
        "BACKTEST RESULTS",
        "=" * 50,
        f"Initial Capital:      {result.config.initial_capital:>12,.0f}",
        f"Final Equity:         {result.equity_curve[-1]:>12,.0f}" if result.equity_curve else "",
        f"Total Return:         {result.total_return_pct:>11.2%}",
        f"Annualized Return:    {result.annualized_return_pct:>11.2%}",
        f"Sharpe Ratio:         {result.sharpe_ratio:>11.2f}",
        f"Max Drawdown:         {result.max_drawdown_pct:>11.2%}",
        "-" * 50,
        f"Total Trades:         {result.total_trades:>11d}",
        f"Win Rate:             {result.win_rate:>11.1%}",
        f"Avg Win:              {result.avg_win_pct:>11.2%}",
        f"Avg Loss:             {result.avg_loss_pct:>11.2%}",
        f"Profit Factor:        {result.profit_factor:>11.2f}",
        f"Avg Hold (days):      {result.avg_hold_days:>11.1f}",
        "-" * 50,
        f"Position Size:        {result.config.position_size_pct:>11.0%}",
        f"Stop Loss:            {result.config.stop_loss_pct:>11.1%}",
        f"Take Profit:          {result.config.take_profit_pct:>11.1%}",
        f"Max Hold Days:        {result.config.max_hold_days:>11d}",
        f"Slippage:             {result.config.slippage_pct:>11.2%}",
        "=" * 50,
    ]
    return "\n".join(lines)


def trade_log_text(result: BacktestResult, max_trades: int = 20) -> str:
    if not result.trades:
        return "No trades executed."
    lines = [f"{'Date':>12} {'Dir':>5} {'Entry':>10} {'Exit':>10} {'PnL%':>8} {'Reason':>12}"]
    lines.append("-" * 60)
    for t in result.trades[:max_trades]:
        date_str = t.entry_date.strftime("%Y-%m-%d") if t.entry_date else "?"
        lines.append(
            f"{date_str:>12} {t.direction:>5} {t.entry_price:>10.2f} "
            f"{t.exit_price or 0:>10.2f} {t.pnl_pct:>7.2%} {t.exit_reason:>12}"
        )
    if len(result.trades) > max_trades:
        lines.append(f"... and {len(result.trades) - max_trades} more trades")
    return "\n".join(lines)
