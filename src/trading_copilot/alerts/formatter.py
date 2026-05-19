"""Telegram message formatting for trading signals and backtest results."""

from __future__ import annotations

import re
from datetime import UTC, datetime


def _escape_md(text: str) -> str:
    """Escape special characters for Telegram MarkdownV2."""
    special = r"_*[]()~`>#+-=|{}.!"
    return re.sub(f"([{re.escape(special)}])", r"\\\1", str(text))


_ACTION_EMOJI = {"BUY": "\U0001f7e2", "SELL": "\U0001f534", "HOLD": "\U0001f7e1"}
_RISK_EMOJI = {"LOW": "✅", "MEDIUM": "⚠️", "HIGH": "\U0001f6a8"}

# Telegram message limit is 4096 characters.
_MAX_MSG_LEN = 4096
_REASONING_MAX = 500


def format_signal(signal: object) -> str:
    """Format a Signal dataclass into a Telegram MarkdownV2 message."""
    action = getattr(signal, "action", "HOLD")
    ticker = getattr(signal, "ticker", "???")
    confidence = getattr(signal, "confidence", 0.0)
    risk_level = getattr(signal, "risk_level", "MEDIUM")
    up_prob = getattr(signal, "up_prob", 0.0)
    down_prob = getattr(signal, "down_prob", 0.0)
    reasoning = getattr(signal, "reasoning", "")
    factors: list[str] = getattr(signal, "factors", [])
    model_name = getattr(signal, "model_name", "")
    timestamp: datetime = getattr(signal, "timestamp", datetime.now(UTC))

    action_emoji = _ACTION_EMOJI.get(action, "❓")
    risk_emoji = _RISK_EMOJI.get(risk_level, "")

    # Truncate reasoning
    if len(reasoning) > _REASONING_MAX:
        reasoning = reasoning[:_REASONING_MAX] + "..."

    lines = [
        f"{action_emoji} *{_escape_md(action)}* \\| *{_escape_md(ticker)}*",
        "",
        f"\U0001f4ca *Confidence:* {_escape_md(f'{confidence:.1%}')}",
        f"{risk_emoji} *Risk:* {_escape_md(risk_level)}",
        "",
        f"\U0001f4c8 Up: {_escape_md(f'{up_prob:.1%}')}  \\|  \U0001f4c9 Down: {_escape_md(f'{down_prob:.1%}')}",
    ]

    if factors:
        lines.append("")
        lines.append("*Key Factors:*")
        for f in factors[:8]:
            lines.append(f"  • {_escape_md(f)}")

    if reasoning:
        lines.append("")
        lines.append(f"*Reasoning:* {_escape_md(reasoning)}")

    if model_name:
        lines.append("")
        lines.append(f"\U0001f916 {_escape_md(model_name)}")

    lines.append(f"\U0001f552 {_escape_md(timestamp.strftime('%Y-%m-%d %H:%M UTC'))}")

    msg = "\n".join(lines)
    if len(msg) > _MAX_MSG_LEN:
        msg = msg[: _MAX_MSG_LEN - 3] + "\\.\\.\\."
    return msg


def format_backtest(result: object) -> str:
    """Format a BacktestResult dataclass into a Telegram MarkdownV2 message."""
    total_return = getattr(result, "total_return_pct", 0.0)
    annual_return = getattr(result, "annualized_return_pct", 0.0)
    sharpe = getattr(result, "sharpe_ratio", 0.0)
    max_dd = getattr(result, "max_drawdown_pct", 0.0)
    win_rate = getattr(result, "win_rate", 0.0)
    profit_factor = getattr(result, "profit_factor", 0.0)
    total_trades = getattr(result, "total_trades", 0)
    avg_hold = getattr(result, "avg_hold_days", 0.0)

    ret_emoji = "\U0001f4b0" if total_return >= 0 else "\U0001f4b8"

    lines = [
        f"{ret_emoji} *Backtest Summary*",
        "",
        f"*Total Return:* {_escape_md(f'{total_return:.2f}%')}",
        f"*Annualized:* {_escape_md(f'{annual_return:.2f}%')}",
        f"*Sharpe Ratio:* {_escape_md(f'{sharpe:.2f}')}",
        f"*Max Drawdown:* {_escape_md(f'{max_dd:.2f}%')}",
        "",
        f"*Win Rate:* {_escape_md(f'{win_rate:.1%}')}",
        f"*Profit Factor:* {_escape_md(f'{profit_factor:.2f}')}",
        f"*Total Trades:* {_escape_md(str(total_trades))}",
        f"*Avg Hold Days:* {_escape_md(f'{avg_hold:.1f}')}",
    ]

    return "\n".join(lines)
