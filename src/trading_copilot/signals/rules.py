"""Signal confirmation rules — technical, sentiment, volatility filters."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class ConfirmationResult:
    technical_score: float  # -1 to +1
    sentiment_score: float  # -1 to +1
    volatility_ok: bool
    factors: list[str]


def technical_confirmation(row: pd.Series) -> tuple[float, list[str]]:
    """Score technical alignment from indicator columns. Returns (score, factors)."""
    score = 0.0
    factors: list[str] = []

    # RSI
    rsi = row.get("rsi_14", 50)
    if rsi < 30:
        score += 0.3
        factors.append(f"RSI oversold ({rsi:.0f})")
    elif rsi > 70:
        score -= 0.3
        factors.append(f"RSI overbought ({rsi:.0f})")

    # MACD histogram direction
    macd_hist = row.get("macd_hist", 0)
    macd_hist_diff = row.get("macd_hist_diff", 0)
    if macd_hist > 0 and macd_hist_diff > 0:
        score += 0.2
        factors.append("MACD histogram expanding bullish")
    elif macd_hist < 0 and macd_hist_diff < 0:
        score -= 0.2
        factors.append("MACD histogram expanding bearish")

    # EMA alignment
    dist_20 = row.get("dist_ema_20", 0)
    dist_50 = row.get("dist_ema_50", 0)
    if dist_20 > 0 and dist_50 > 0:
        score += 0.15
        factors.append("Price above EMA20 & EMA50")
    elif dist_20 < 0 and dist_50 < 0:
        score -= 0.15
        factors.append("Price below EMA20 & EMA50")

    # Bollinger position
    bb_pos = row.get("bb_position", 0.5)
    if bb_pos < 0.1:
        score += 0.15
        factors.append("Near Bollinger lower band")
    elif bb_pos > 0.9:
        score -= 0.15
        factors.append("Near Bollinger upper band")

    # ADX trend strength
    adx = row.get("adx", 0)
    if adx > 25:
        factors.append(f"Strong trend (ADX={adx:.0f})")

    # Volume spike
    if row.get("volume_spike", False) or row.get("vol_ratio_20", 1) > 1.5:
        factors.append("Elevated volume")

    return max(-1, min(1, score)), factors


def sentiment_confirmation(row: pd.Series) -> tuple[float, list[str]]:
    """Score sentiment alignment. Returns (score, factors)."""
    factors: list[str] = []
    sent_score = row.get("sent_score", 0)
    sent_conf = row.get("sent_confidence", 0)
    article_count = row.get("sent_article_count", 0)
    sent_shift = row.get("sent_shift", 0)

    if article_count == 0:
        return 0.0, ["No recent news"]

    score = sent_score * min(sent_conf, 1.0)

    if sent_score > 0.2:
        factors.append(f"Bullish sentiment ({sent_score:+.2f})")
    elif sent_score < -0.2:
        factors.append(f"Bearish sentiment ({sent_score:+.2f})")
    else:
        factors.append("Neutral sentiment")

    if abs(sent_shift) > 0.1:
        direction = "improving" if sent_shift > 0 else "deteriorating"
        factors.append(f"Sentiment {direction}")

    factors.append(f"{int(article_count)} articles analyzed")
    return max(-1, min(1, score)), factors


def volatility_filter(row: pd.Series, max_atr_pct: float = 0.05) -> tuple[bool, list[str]]:
    """Check if volatility is within acceptable range for signal generation."""
    atr_pct = row.get("atr_pct", 0)
    factors: list[str] = []

    if atr_pct > max_atr_pct:
        factors.append(f"High volatility (ATR={atr_pct:.1%}) — reduced confidence")
        return False, factors

    factors.append(f"Normal volatility (ATR={atr_pct:.1%})")
    return True, factors


def compute_confirmation(row: pd.Series, max_atr_pct: float = 0.05) -> ConfirmationResult:
    tech_score, tech_factors = technical_confirmation(row)
    sent_score, sent_factors = sentiment_confirmation(row)
    vol_ok, vol_factors = volatility_filter(row, max_atr_pct)

    return ConfirmationResult(
        technical_score=tech_score,
        sentiment_score=sent_score,
        volatility_ok=vol_ok,
        factors=tech_factors + sent_factors + vol_factors,
    )
