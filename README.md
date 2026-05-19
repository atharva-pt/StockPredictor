# Trading Copilot

AI market-intelligence and **probabilistic** directional-prediction copilot for NSE/BSE/US markets.
Swing-trading focus. Free tools only. Runs locally on macOS.

> ⚠️ This system does **not** predict exact prices. It estimates probabilities of directional
> movement with explicit confidence and reasoning. Realistic out-of-sample accuracy ceiling for
> liquid large-caps is roughly **53–58%**. Anything higher should make you suspicious of leakage.

---

## Status: Phase 1 — Project Skeleton

Phase 1 ships only the foundation: config, logging, tests, and folder layout. No data fetching,
no indicators, no models yet. Each subsequent phase is added deliberately on its own branch.

| Phase | Scope | Status |
|------:|-------|--------|
| 1 | Project skeleton, config, logging | ✅ this commit |
| 2 | Market data pipeline (yfinance + SQLite/parquet cache) | ⏳ |
| 3 | Technical indicators | ⏳ |
| 4 | News + RSS ingestion | ⏳ |
| 5 | Sentiment (FinBERT + VADER) | ⏳ |
| 6 | Feature engineering (leakage-safe) | ⏳ |
| 7 | ML models (RF / XGBoost / LightGBM, walk-forward CV) | ⏳ |
| 8 | Signal generation | ⏳ |
| 9 | Backtesting | ⏳ |
| 10 | Streamlit dashboard | ⏳ |
| 11 | Telegram alerts | ⏳ |
| 12 | Explainability | ⏳ |

---

## Setup

### 1. Install `uv` (one time)

```bash
brew install uv
```

### 2. Create env and install deps

From the project root:

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### 3. Copy env template

```bash
cp .env.example .env
```

Phase 1 needs no secrets. Telegram tokens come in Phase 11.

### 4. Run the test suite

```bash
pytest
```

Expected: 5 passed.

### 5. Lint

```bash
ruff check .
```

---

## Layout

```
trading-copilot/
├── config/                     # settings.yaml — runtime config (NO secrets)
├── src/trading_copilot/
│   ├── config.py               # pydantic-typed config loader
│   ├── logging_setup.py        # structlog + rotating JSON file handler
│   ├── data/                   # Phase 2
│   ├── indicators/             # Phase 3
│   ├── news/                   # Phase 4
│   ├── nlp/                    # Phase 5
│   ├── features/               # Phase 6
│   ├── models/                 # Phase 7
│   ├── signals/                # Phase 8
│   ├── backtest/               # Phase 9
│   ├── dashboard/              # Phase 10
│   ├── alerts/                 # Phase 11
│   └── explain/                # Phase 12
├── data/                       # gitignored — caches, DB, logs
├── tests/                      # mirrors src/ layout
├── notebooks/                  # exploration only (never production logic)
└── scripts/                    # one-off CLIs (added in later phases)
```

## Design rules baked in from day one

- **All timestamps are UTC internally.** Conversion to `Asia/Kolkata` happens only at display edges.
- **Secrets in `.env`, config in YAML.** Never mixed.
- **Subpackages are empty until their phase lands.** No speculative code.
- **Logs are structured.** Rotating JSON file + colored console.
- **No third-party data libs in Phase 1.** Pandas/yfinance/etc. arrive with the phase that needs them.

## What this project explicitly will NOT do

- Claim accuracy above well-validated empirical reality
- Use tick-level / sub-minute intraday data (not available free)
- Send real orders to a broker
- Be a substitute for risk management or your own judgement
