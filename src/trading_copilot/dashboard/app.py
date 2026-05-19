"""Streamlit dashboard — Trading Copilot main UI.

Features:
- Auto-refresh with configurable interval
- Live price fetch from yfinance
- Walk-forward (out-of-sample) predictions — train on past, predict TODAY
- Forward-looking prediction overlay on chart
- Custom ticker input — analyze any stock, not just the watchlist
"""

from __future__ import annotations

import sys
from pathlib import Path

_src = str(Path(__file__).resolve().parents[2])
if _src not in sys.path:
    sys.path.insert(0, _src)

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from trading_copilot.backtest.engine import run_backtest
from trading_copilot.backtest.report import summary_text, trade_log_text
from trading_copilot.backtest.types import BacktestConfig
from trading_copilot.config import get_settings
from trading_copilot.dashboard.charts import (
    candlestick_chart,
    equity_curve_chart,
    prediction_gauge,
    sentiment_timeline,
)
from trading_copilot.data.cache import OHLCVCache
from trading_copilot.data.fetcher import fetch_ohlcv
from trading_copilot.features.pipeline import build_feature_matrix
from trading_copilot.indicators.engine import compute_all
from trading_copilot.logging_setup import get_logger
from trading_copilot.models.engine import Prediction, predict_ensemble, train_ensemble
from trading_copilot.models.trainers import ensemble_predict_proba, get_model
from trading_copilot.models.validation import walk_forward_splits
from trading_copilot.news.store import NewsStore
from trading_copilot.nlp.engine import analyze_articles
from trading_copilot.signals.generator import generate_signals

log = get_logger("dashboard")

st.set_page_config(page_title="Trading Copilot", layout="wide", page_icon="$")

# ---------------------------------------------------------------------------
# Expanded ticker universe
# ---------------------------------------------------------------------------

NSE_POPULAR = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "BHARTIARTL.NS", "ITC.NS", "SBIN.NS", "KOTAKBANK.NS",
    "LT.NS", "BAJFINANCE.NS", "AXISBANK.NS", "MARUTI.NS", "TITAN.NS",
    "SUNPHARMA.NS", "WIPRO.NS", "HCLTECH.NS", "ULTRACEMCO.NS", "ASIANPAINT.NS",
    "TATAMOTORS.NS", "TATASTEEL.NS", "POWERGRID.NS", "NTPC.NS", "ADANIGREEN.NS",
    "ADANIENT.NS", "BAJAJFINSV.NS", "TECHM.NS", "NESTLEIND.NS", "JSWSTEEL.NS",
    "DRREDDY.NS", "DIVISLAB.NS", "CIPLA.NS", "GRASIM.NS", "ONGC.NS",
    "COALINDIA.NS", "BPCL.NS", "HEROMOTOCO.NS", "EICHERMOT.NS", "INDUSINDBK.NS",
]

US_POPULAR = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "BRK-B",
    "JPM", "V", "UNH", "MA", "HD", "PG", "JNJ", "COST", "ABBV",
    "CRM", "NFLX", "AMD", "ORCL", "ADBE", "PEP", "KO", "TMO",
    "INTC", "CSCO", "DIS", "NKE", "BA", "PYPL", "UBER", "SQ",
    "COIN", "RIVN", "PLTR", "SOFI", "SNOW", "NET", "SHOP",
]


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _fetch_live_ohlcv(ticker: str, days: int) -> pd.DataFrame | None:
    """Fetch fresh data from yfinance, fall back to cache."""
    cache = OHLCVCache()
    start = datetime.now(UTC) - timedelta(days=days)
    df = fetch_ohlcv(ticker, start=start.strftime("%Y-%m-%d"))
    if df is not None and not df.empty:
        cache.save(ticker, df)
        return df
    cached = cache.load(ticker)
    if cached is not None and len(cached) > 50:
        return cached
    return None


@st.cache_data(ttl=300)
def _load_news(hours: int = 72) -> list[dict]:
    try:
        store = NewsStore()
        since = datetime.now(UTC) - timedelta(hours=hours)
        articles = store.query(since=since, limit=200)
        return [
            {
                "title": a.title, "source": a.source, "url": a.url,
                "published_utc": a.published_utc, "summary": a.summary,
            }
            for a in articles
        ]
    except Exception:
        return []


@st.cache_data(ttl=300)
def _run_sentiment(articles_raw: list[dict]) -> list[dict]:
    if not articles_raw:
        return []
    from trading_copilot.news.models import Article
    articles = [
        Article(
            title=a["title"], source=a["source"], url=a["url"],
            published_utc=a["published_utc"], summary=a.get("summary", ""),
        )
        for a in articles_raw
    ]
    results = analyze_articles(articles, use_finbert=False)
    return [
        {
            "title": a["title"], "published_utc": a["published_utc"],
            "score": r.score, "label": r.sentiment, "source": a["source"],
        }
        for a, r in zip(articles_raw, results, strict=False)
    ]


# ---------------------------------------------------------------------------
# Walk-forward out-of-sample prediction
# ---------------------------------------------------------------------------

def _walk_forward_predict(
    ohlcv: pd.DataFrame, ticker: str, min_bars: int = 300,
) -> tuple[Prediction, pd.Series, dict] | None:
    """Train on all data except the last bar, predict TODAY out-of-sample.

    Returns (prediction_for_today, latest_feature_row, cv_metrics) or None.
    """
    if len(ohlcv) < min_bars:
        return None

    features, targets = build_feature_matrix(ohlcv, ticker=ticker, dropna=True)
    target_col = "target_5d_dir"
    if target_col not in targets.columns or len(features) < 250:
        return None
    y = targets[target_col]
    if y.nunique() < 2:
        return None

    try:
        train_features = features.iloc[:-1]
        train_y = y.iloc[:-1]
        today_features = features.iloc[[-1]]

        model_names = ["random_forest", "xgboost", "lightgbm"]
        models = []
        weights = []
        cv_metrics: dict[str, float] = {}

        for name in model_names:
            splits = walk_forward_splits(
                len(train_features), n_splits=4, min_train_size=180,
            )
            from sklearn.metrics import roc_auc_score
            fold_aucs = []
            for split in splits:
                X_tr = train_features.values[split.train_idx]
                y_tr = train_y.values[split.train_idx]
                X_te = train_features.values[split.test_idx]
                y_te = train_y.values[split.test_idx]
                m = get_model(name)
                m.fit(X_tr, y_tr)
                proba = m.predict_proba(X_te)[:, 1]
                try:
                    fold_aucs.append(roc_auc_score(y_te, proba))
                except ValueError:
                    fold_aucs.append(0.5)

            avg_auc = float(np.mean(fold_aucs))
            cv_metrics[name] = avg_auc

            final_model = get_model(name)
            final_model.fit(train_features.values, train_y.values)
            models.append(final_model)
            weights.append(avg_auc)

        total_w = sum(weights)
        weights = [w / total_w for w in weights]

        probas = ensemble_predict_proba(models, today_features.values, weights)
        proba = probas[0]
        down_prob, up_prob = float(proba[0]), float(proba[1])
        confidence = max(up_prob, down_prob)
        direction = "HOLD" if confidence < 0.55 else ("UP" if up_prob > down_prob else "DOWN")

        pred = Prediction(
            up_prob=round(up_prob, 4),
            down_prob=round(down_prob, 4),
            direction=direction,
            confidence=round(confidence, 4),
            model_name="ensemble",
        )

        cv_metrics["ensemble_weighted_auc"] = sum(
            cv_metrics[n] * w for n, w in zip(model_names, weights, strict=False)
        )

        return pred, features.iloc[-1], cv_metrics

    except Exception as e:
        log.warning("walk_forward_predict_failed", error=str(e))
        return None


def _build_backtest_predictions(
    ohlcv: pd.DataFrame, ticker: str,
) -> tuple[list[Prediction], pd.DataFrame, pd.DataFrame] | None:
    """In-sample predictions for backtesting only."""
    if len(ohlcv) < 300:
        return None
    features, targets = build_feature_matrix(ohlcv, ticker=ticker, dropna=True)
    target_col = "target_5d_dir"
    if target_col not in targets.columns:
        return None
    y = targets[target_col]
    if len(features) < 250 or y.nunique() < 2:
        return None
    try:
        models, weights = train_ensemble(features, y)
        preds = predict_ensemble(models, weights, features)
        return preds, features, targets
    except Exception as e:
        log.warning("prediction_failed", error=str(e))
        return None


# ---------------------------------------------------------------------------
# Prediction overlay on chart
# ---------------------------------------------------------------------------

def _add_prediction_overlay(fig, ohlcv: pd.DataFrame, latest_pred: Prediction, horizon_days: int = 5):
    last_close = ohlcv["close"].iloc[-1]
    last_date = ohlcv.index[-1]

    future_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=horizon_days, tz="UTC")
    if len(future_dates) == 0:
        return fig

    conf = latest_pred.confidence
    atr = ohlcv["close"].pct_change().std() * last_close

    if latest_pred.direction == "UP":
        mid_move = atr * conf * np.linspace(0.2, 1.0, horizon_days)
        mid_line = last_close + mid_move
    elif latest_pred.direction == "DOWN":
        mid_move = atr * conf * np.linspace(0.2, 1.0, horizon_days)
        mid_line = last_close - mid_move
    else:
        mid_line = np.full(horizon_days, last_close)

    band_width = atr * np.linspace(0.5, 1.5, horizon_days)
    upper = mid_line + band_width
    lower = mid_line - band_width

    color = "#26a69a" if latest_pred.direction == "UP" else "#ef5350" if latest_pred.direction == "DOWN" else "#ff9800"
    fill_color = "rgba(38,166,154,0.15)" if latest_pred.direction == "UP" else "rgba(239,83,80,0.15)" if latest_pred.direction == "DOWN" else "rgba(255,152,0,0.15)"

    conn_dates = [last_date, *list(future_dates)]
    conn_mid = [last_close, *list(mid_line)]
    conn_upper = [last_close, *list(upper)]
    conn_lower = [last_close, *list(lower)]

    fig.add_trace(go.Scatter(
        x=conn_dates, y=conn_upper, mode="lines", name="Pred Upper",
        line=dict(width=1, dash="dot", color=color), showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=conn_dates, y=conn_lower, mode="lines", name="Pred Lower",
        line=dict(width=1, dash="dot", color=color),
        fill="tonexty", fillcolor=fill_color, showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=conn_dates, y=conn_mid, mode="lines+markers", name=f"Pred {latest_pred.direction}",
        line=dict(width=2, dash="dash", color=color),
        marker=dict(size=6, color=color),
    ), row=1, col=1)

    fig.add_annotation(
        x=future_dates[-1], y=mid_line[-1],
        text=f"{latest_pred.direction} {latest_pred.confidence:.0%}",
        showarrow=True, arrowhead=2, arrowcolor=color,
        font=dict(color=color, size=12, family="monospace"),
        bgcolor="rgba(0,0,0,0.7)", bordercolor=color,
    )
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    settings = get_settings()

    # Build full ticker universe: watchlist + popular + custom
    watchlist_tickers = []
    for market_tickers in settings.markets.watchlist.values():
        watchlist_tickers.extend(market_tickers)

    all_known = list(dict.fromkeys(watchlist_tickers + NSE_POPULAR + US_POPULAR))

    # --- Sidebar ---
    st.sidebar.title("Trading Copilot")

    st.sidebar.subheader("Ticker Selection")
    custom_ticker = st.sidebar.text_input(
        "Enter any ticker (e.g. TATAMOTORS.NS, TSLA, BAJAJ-AUTO.NS)",
        value="",
        help="Type any yfinance-compatible ticker. NSE stocks end in .NS, BSE in .BO",
    )

    market_filter = st.sidebar.radio("Market", ["All", "NSE", "US"], horizontal=True)
    if market_filter == "NSE":
        filtered = [t for t in all_known if t.endswith((".NS", ".BO"))]
    elif market_filter == "US":
        filtered = [t for t in all_known if not t.endswith((".NS", ".BO"))]
    else:
        filtered = all_known

    selected_from_list = st.sidebar.selectbox("Or pick from list", filtered, index=0)

    ticker = custom_ticker.strip().upper() if custom_ticker.strip() else selected_from_list

    st.sidebar.markdown("---")
    st.sidebar.subheader("Settings")
    lookback = st.sidebar.slider("Lookback (days)", 60, 730, 730)

    auto_refresh = st.sidebar.checkbox("Auto-refresh", value=False)
    refresh_interval = st.sidebar.select_slider(
        "Refresh interval",
        options=[30, 60, 120, 300],
        value=60,
        format_func=lambda x: f"{x}s" if x < 60 else f"{x // 60}m",
    )

    run_ml = st.sidebar.checkbox("Run ML Predictions", value=True)
    run_bt = st.sidebar.checkbox("Run Backtest", value=False)

    st.sidebar.markdown("---")
    last_updated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    st.sidebar.caption(f"Last updated: {last_updated}")

    # --- Auto refresh ---
    if auto_refresh:
        import time as _time
        if "last_refresh" not in st.session_state:
            st.session_state.last_refresh = _time.time()
        elapsed = _time.time() - st.session_state.last_refresh
        remaining = max(0, refresh_interval - elapsed)
        if remaining == 0:
            st.session_state.last_refresh = _time.time()
            st.rerun()
        else:
            st.sidebar.progress(1 - remaining / refresh_interval, text=f"Next refresh in {int(remaining)}s")

    # --- Load live data ---
    with st.spinner(f"Fetching latest data for {ticker}..."):
        ohlcv = _fetch_live_ohlcv(ticker, lookback)

    if ohlcv is None or ohlcv.empty:
        st.error(f"No data for **{ticker}**. Check the ticker symbol is valid (NSE: add .NS suffix, BSE: add .BO).")
        return

    indicators = compute_all(ohlcv)

    # --- Walk-forward out-of-sample prediction ---
    latest_pred = None
    latest_signal = None
    cv_metrics = None

    if run_ml:
        with st.spinner("Running walk-forward prediction (out-of-sample)..."):
            wf_result = _walk_forward_predict(ohlcv, ticker)

        if wf_result is not None:
            latest_pred, latest_row, cv_metrics = wf_result
            signals = generate_signals(ticker, [latest_pred], pd.DataFrame([latest_row]))
            latest_signal = signals[0] if signals else None

    # --- TODAY'S PREDICTION banner ---
    if latest_pred and latest_signal:
        action = latest_signal.action
        bg = {"BUY": "#1b5e20", "SELL": "#b71c1c", "HOLD": "#e65100"}.get(action, "#37474f")
        arrow = {"BUY": "^", "SELL": "v", "HOLD": "-"}.get(action, "?")

        st.markdown(
            f"""<div style="background:{bg}; padding:16px 24px; border-radius:8px; margin-bottom:16px;">
            <h2 style="color:white; margin:0;">
                {arrow} {action} {ticker} &mdash; {latest_signal.confidence:.0%} confidence
            </h2>
            <p style="color:#ccc; margin:4px 0 0 0;">
                5-day outlook: {latest_pred.direction} (UP {latest_pred.up_prob:.1%} / DOWN {latest_pred.down_prob:.1%})
                &bull; Risk: {latest_signal.risk_level}
                &bull; Out-of-sample prediction
            </p>
            </div>""",
            unsafe_allow_html=True,
        )

        if cv_metrics:
            st.caption(
                "Model validation (walk-forward AUC): "
                + " | ".join(f"{k}: {v:.3f}" for k, v in cv_metrics.items())
            )
    elif run_ml:
        st.warning(f"Not enough data for predictions on {ticker} (need 300+ daily bars). Try a different ticker or increase lookback.")

    # --- Price Chart with prediction overlay ---
    fig = candlestick_chart(ohlcv, indicators, title=f"{ticker} — Live")
    if latest_pred:
        fig = _add_prediction_overlay(fig, ohlcv, latest_pred, horizon_days=5)
    st.plotly_chart(fig, use_container_width=True)

    # --- Key metrics row ---
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    last_close = ohlcv["close"].iloc[-1]
    prev_close = ohlcv["close"].iloc[-2] if len(ohlcv) > 1 else last_close
    change_pct = (last_close - prev_close) / prev_close

    col1.metric("Last Close", f"{last_close:,.2f}", f"{change_pct:+.2%}")
    col2.metric("RSI (14)", f"{indicators['rsi'].iloc[-1]:.1f}" if "rsi" in indicators.columns else "N/A")
    col3.metric("ATR %", f"{indicators['atr_pct'].iloc[-1]:.2%}" if "atr_pct" in indicators.columns else "N/A")
    col4.metric("Volume Ratio", f"{indicators['volume_ratio'].iloc[-1]:.2f}" if "volume_ratio" in indicators.columns else "N/A")
    if latest_pred:
        col5.metric("UP Prob", f"{latest_pred.up_prob:.1%}")
        col6.metric("DOWN Prob", f"{latest_pred.down_prob:.1%}")
    else:
        col5.metric("UP Prob", "N/A")
        col6.metric("DOWN Prob", "N/A")

    # --- Signal detail ---
    if latest_signal and latest_signal.action != "HOLD":
        with st.expander("Signal Reasoning & Factors", expanded=True):
            st.write(latest_signal.reasoning)
            if latest_signal.factors:
                st.markdown("**Contributing factors:**")
                for f in latest_signal.factors:
                    st.markdown(f"- {f}")

    # --- Prediction gauge ---
    if latest_pred:
        st.plotly_chart(
            prediction_gauge(latest_pred.up_prob, latest_pred.down_prob,
                             latest_signal.action if latest_signal else latest_pred.direction,
                             latest_pred.confidence),
            use_container_width=True,
        )

    # --- News & Sentiment ---
    st.markdown("---")
    news_col, sent_col = st.columns([1, 1])

    articles_raw = _load_news()
    sentiment_data = _run_sentiment(articles_raw) if articles_raw else []

    with news_col:
        st.subheader("Latest Headlines")
        if articles_raw:
            for a in articles_raw[:15]:
                pub = a["published_utc"]
                if isinstance(pub, str):
                    pub = pub[:16]
                st.markdown(f"**{a['source']}** ({pub})  \n{a['title']}")
        else:
            st.info("No recent news. Run `scripts/fetch_news.py` to populate.")

    with sent_col:
        st.subheader("Sentiment Timeline")
        if sentiment_data:
            st.plotly_chart(sentiment_timeline(sentiment_data), use_container_width=True)
        else:
            st.info("No sentiment data available.")

    # --- Backtest ---
    if run_bt:
        st.markdown("---")
        st.subheader("Backtest (Walk-Forward)")

        bt_col1, bt_col2, bt_col3, bt_col4 = st.columns(4)
        initial_capital = bt_col1.number_input("Capital", value=100_000, step=10_000)
        stop_loss = bt_col2.slider("Stop Loss %", 1.0, 10.0, 3.0) / 100
        take_profit = bt_col3.slider("Take Profit %", 2.0, 20.0, 6.0) / 100
        max_hold = bt_col4.slider("Max Hold Days", 3, 30, 10)

        bt_config = BacktestConfig(
            initial_capital=initial_capital,
            stop_loss_pct=stop_loss,
            take_profit_pct=take_profit,
            max_hold_days=max_hold,
        )

        with st.spinner("Building walk-forward backtest signals..."):
            bt_pred_result = _build_backtest_predictions(ohlcv, ticker)

        if bt_pred_result is None:
            st.warning("Not enough data for backtesting.")
        else:
            preds, pred_features, _targets = bt_pred_result
            bt_signals = generate_signals(ticker, preds, pred_features)

            with st.spinner("Running backtest..."):
                bt_result = run_backtest(bt_signals, ohlcv.loc[pred_features.index], bt_config)

            metric_cols = st.columns(5)
            metric_cols[0].metric("Total Return", f"{bt_result.total_return_pct:.2%}")
            metric_cols[1].metric("Sharpe Ratio", f"{bt_result.sharpe_ratio:.2f}")
            metric_cols[2].metric("Win Rate", f"{bt_result.win_rate:.1%}")
            metric_cols[3].metric("Max Drawdown", f"{bt_result.max_drawdown_pct:.2%}")
            metric_cols[4].metric("Total Trades", bt_result.total_trades)

            eq_col, log_col = st.columns([1, 1])
            with eq_col:
                if bt_result.equity_curve and bt_result.equity_dates:
                    st.plotly_chart(
                        equity_curve_chart(bt_result.equity_curve, bt_result.equity_dates),
                        use_container_width=True,
                    )

            with log_col, st.expander("Trade Log", expanded=True):
                st.code(trade_log_text(bt_result, max_trades=30))

            with st.expander("Full Backtest Report"):
                st.code(summary_text(bt_result))

            st.info(
                "Note: Backtest uses in-sample signals. The live prediction above "
                "uses walk-forward (out-of-sample) methodology for realistic estimates."
            )


if __name__ == "__main__":
    main()
