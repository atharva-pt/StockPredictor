"""Plotly chart builders for the Streamlit dashboard."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def candlestick_chart(
    ohlcv: pd.DataFrame,
    indicators: pd.DataFrame | None = None,
    signals: list | None = None,
    title: str = "Price Chart",
) -> go.Figure:
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=[title, "Volume", "RSI"],
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=ohlcv.index, open=ohlcv["open"], high=ohlcv["high"],
        low=ohlcv["low"], close=ohlcv["close"], name="Price",
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
    ), row=1, col=1)

    # Overlays from indicators
    if indicators is not None:
        for col, color in [("ema_20", "#ff9800"), ("ema_50", "#2196f3"), ("ema_200", "#9c27b0")]:
            if col in indicators.columns:
                fig.add_trace(go.Scatter(
                    x=indicators.index, y=indicators[col], name=col.upper(),
                    line=dict(width=1, color=color), opacity=0.7,
                ), row=1, col=1)

        if "bb_upper" in indicators.columns:
            fig.add_trace(go.Scatter(
                x=indicators.index, y=indicators["bb_upper"], name="BB Upper",
                line=dict(width=1, dash="dot", color="#90a4ae"), showlegend=False,
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=indicators.index, y=indicators["bb_lower"], name="BB Lower",
                line=dict(width=1, dash="dot", color="#90a4ae"),
                fill="tonexty", fillcolor="rgba(144,164,174,0.1)", showlegend=False,
            ), row=1, col=1)

    # Buy/Sell markers
    if signals:
        buys = [(s.timestamp, s) for s in signals if s.action == "BUY"]
        sells = [(s.timestamp, s) for s in signals if s.action == "SELL"]
        if buys:
            fig.add_trace(go.Scatter(
                x=[b[0] for b in buys],
                y=[ohlcv["low"].iloc[-len(signals) + i] * 0.998 for i, b in enumerate(buys)],
                mode="markers", name="BUY",
                marker=dict(symbol="triangle-up", size=12, color="#26a69a"),
            ), row=1, col=1)
        if sells:
            fig.add_trace(go.Scatter(
                x=[s[0] for s in sells],
                y=[ohlcv["high"].iloc[-len(signals) + i] * 1.002 for i, s in enumerate(sells)],
                mode="markers", name="SELL",
                marker=dict(symbol="triangle-down", size=12, color="#ef5350"),
            ), row=1, col=1)

    # Volume
    colors = ["#26a69a" if c >= o else "#ef5350" for c, o in zip(ohlcv["close"], ohlcv["open"], strict=False)]
    fig.add_trace(go.Bar(
        x=ohlcv.index, y=ohlcv["volume"], name="Volume",
        marker_color=colors, opacity=0.5, showlegend=False,
    ), row=2, col=1)

    # RSI
    if indicators is not None and "rsi" in indicators.columns:
        fig.add_trace(go.Scatter(
            x=indicators.index, y=indicators["rsi"], name="RSI",
            line=dict(color="#ff9800", width=1),
        ), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=3, col=1)

    fig.update_layout(
        height=700, xaxis_rangeslider_visible=False,
        template="plotly_dark", margin=dict(t=40, b=20, l=50, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def equity_curve_chart(equity: list[float], dates: list, title: str = "Equity Curve") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=equity, mode="lines", name="Equity",
        line=dict(color="#26a69a", width=2),
        fill="tozeroy", fillcolor="rgba(38,166,154,0.1)",
    ))
    fig.update_layout(
        title=title, height=350, template="plotly_dark",
        margin=dict(t=40, b=20, l=50, r=20),
        yaxis_title="Capital",
    )
    return fig


def sentiment_timeline(articles_with_sentiment: list[dict]) -> go.Figure:
    """Plot sentiment scores over time. Expects list of {published_utc, score, title}."""
    if not articles_with_sentiment:
        fig = go.Figure()
        fig.update_layout(title="No sentiment data", height=250, template="plotly_dark")
        return fig

    dates = [a["published_utc"] for a in articles_with_sentiment]
    scores = [a["score"] for a in articles_with_sentiment]
    titles = [a.get("title", "")[:60] for a in articles_with_sentiment]
    colors = ["#26a69a" if s > 0 else "#ef5350" if s < 0 else "#90a4ae" for s in scores]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=dates, y=scores, name="Sentiment",
        marker_color=colors, text=titles, hoverinfo="text+y",
    ))
    fig.add_hline(y=0, line_color="white", opacity=0.3)
    fig.update_layout(
        title="News Sentiment Timeline", height=250, template="plotly_dark",
        margin=dict(t=40, b=20, l=50, r=20), yaxis_title="Score",
    )
    return fig


def prediction_gauge(up_prob: float, down_prob: float, direction: str, confidence: float) -> go.Figure:
    color = "#26a69a" if direction == "BUY" else "#ef5350" if direction == "SELL" else "#ff9800"
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=up_prob * 100,
        title={"text": f"Signal: {direction} ({confidence:.0%})"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 40], "color": "rgba(239,83,80,0.2)"},
                {"range": [40, 60], "color": "rgba(255,152,0,0.2)"},
                {"range": [60, 100], "color": "rgba(38,166,154,0.2)"},
            ],
            "threshold": {"line": {"color": "white", "width": 2}, "thickness": 0.75, "value": 50},
        },
        number={"suffix": "% UP"},
    ))
    fig.update_layout(height=250, template="plotly_dark", margin=dict(t=60, b=20))
    return fig
