# OandaFX — Institutional-Grade Forex Trading Bot

> A self-learning, multi-account, multi-pair forex trading system built on OANDA's REST API v20, deployable on Vultr VPS with GPU-assisted model training via Google Colab.

---

## Project Overview

OandaFX is a modular, production-ready algorithmic trading system that combines classical technical analysis with modern machine learning. It is designed from the ground up to operate within OANDA's regulatory and API constraints while scaling across multiple accounts, multiple API keys, and unlimited currency pairs and timeframes.

The system learns how to trade — not just execute rules. It uses reinforcement learning to discover when to enter, where to place stop losses and take profits, how to size positions, and when to stay flat. All training data comes directly from OANDA's historical candle API, ensuring the model trains on the same pricing it will trade live.

---

## Repository Structure

```
oandafx/
├── README.md                        ← You are here
├── docs/
│   ├── ARCHITECTURE.md              ← System design & module map
│   ├── DATA_PIPELINE.md             ← OANDA data collection & storage
│   ├── FEATURE_ENGINEERING.md       ← Indicators, patterns, MTF features
│   ├── MODEL_SPEC.md                ← ML/RL model architecture
│   ├── RISK_MANAGEMENT.md           ← SL/TP, position sizing, drawdown
│   ├── OANDA_COMPLIANCE.md          ← Broker rules, leverage, spreads
│   ├── MULTI_ACCOUNT.md             ← Multi-API & multi-account setup
│   ├── DEPLOYMENT.md                ← Vultr VPS deployment guide
│   ├── TRAINING.md                  ← Colab GPU training workflow
│   └── MONITORING.md                ← Logging, alerts, dashboards
├── config/
│   ├── accounts.yaml.example        ← Account/API key template
│   └── bot_config.yaml.example      ← Bot strategy config template
├── src/
│   ├── core/
│   │   ├── account_manager.py       ← Multi-account orchestrator
│   │   ├── api_client.py            ← OANDA REST v20 wrapper
│   │   └── session.py               ← Trading session controller
│   ├── data/
│   │   ├── downloader.py            ← Historical data fetcher (paginated)
│   │   ├── stream.py                ← Live price streaming
│   │   └── storage.py               ← Parquet/SQLite store
│   ├── features/
│   │   ├── indicators.py            ← Technical indicators (TA-Lib)
│   │   ├── patterns.py              ← Chart pattern detection
│   │   ├── mtf.py                   ← Multi-timeframe alignment
│   │   └── engineer.py              ← Feature pipeline orchestrator
│   ├── models/
│   │   ├── transformer.py           ← Sequence model (PyTorch)
│   │   ├── rl_agent.py              ← PPO reinforcement learning agent
│   │   ├── sl_tp_module.py          ← Dynamic SL/TP predictor
│   │   └── ensemble.py              ← Model ensemble & voting
│   ├── risk/
│   │   ├── position_sizer.py        ← Kelly / fixed-fraction sizing
│   │   ├── circuit_breaker.py       ← Daily loss halt, max DD
│   │   └── validator.py             ← Pre-trade compliance check
│   ├── execution/
│   │   ├── order_manager.py         ← Order placement & modification
│   │   ├── trade_monitor.py         ← Live trade tracker
│   │   └── reconciler.py            ← Account reconciliation
│   ├── backtest/
│   │   ├── engine.py                ← Walk-forward backtest engine
│   │   ├── metrics.py               ← Sharpe, sortino, MFE, MAE
│   │   └── reporter.py              ← HTML/CSV report generator
│   └── monitoring/
│       ├── logger.py                ← Structured JSON logging
│       ├── alerts.py                ← Telegram / email alerting
│       └── dashboard.py             ← FastAPI metrics endpoint
├── scripts/
│   ├── setup_vps.sh                 ← Vultr VPS bootstrap script
│   ├── download_history.py          ← One-time data download CLI
│   ├── run_backtest.py              ← Backtest runner CLI
│   └── start_bot.py                 ← Live trading launcher
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_analysis.ipynb
│   ├── 03_model_training.ipynb      ← Run on Google Colab (GPU)
│   └── 04_backtest_analysis.ipynb
├── tests/
│   ├── test_api_client.py
│   ├── test_features.py
│   ├── test_risk.py
│   └── test_execution.py
├── requirements.txt
├── requirements-gpu.txt             ← Colab/GPU training dependencies
├── Dockerfile
├── docker-compose.yml
└── .env.example                     ← Environment variable template
```

---

## Quickstart

### 1. Clone and configure

```bash
git clone https://github.com/your-org/oandafx.git
cd oandafx
cp .env.example .env
cp config/accounts.yaml.example config/accounts.yaml
# Edit both files with your OANDA credentials
```

### 2. Install dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Download historical data

```bash
python scripts/download_history.py \
  --pairs EUR_USD GBP_USD USD_JPY \
  --granularities M15 H1 H4 D \
  --start 2015-01-01
```

### 4. Train the model (Colab)

Open `notebooks/03_model_training.ipynb` in Google Colab with a GPU runtime. Upload your downloaded data to Drive or Colab storage, then run all cells.

### 5. Run backtest

```bash
python scripts/run_backtest.py \
  --model models/latest.pt \
  --start 2023-01-01 \
  --end 2024-12-31
```

### 6. Deploy to Vultr VPS

```bash
# On your Vultr instance (Ubuntu 22.04)
bash scripts/setup_vps.sh
python scripts/start_bot.py --config config/accounts.yaml
```

---

## Core Design Principles

| Principle | Implementation |
|-----------|---------------|
| **OANDA-native data** | All training and live data sourced from OANDA API — no third-party data mismatches |
| **Multi-account** | Single bot instance manages N OANDA accounts simultaneously via `account_manager.py` |
| **No hardcoded pairs** | Pair list is fully configurable; the model is pair-agnostic |
| **No hardcoded timeframes** | MTF alignment engine dynamically composes any combination |
| **Adaptive SL/TP** | Stop loss and take profit are model outputs, not fixed rules |
| **Regulation-aware** | Leverage, margin, and NFA rules enforced in `validator.py` before every order |
| **Resilient** | Exponential backoff, reconnect logic, and circuit breakers handle API failures |
| **Auditable** | Every decision, order, and error is logged as structured JSON |

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11+ |
| OANDA API | `oandapyV20` (REST v20) |
| Data storage | Parquet (historical), SQLite (live trades) |
| ML framework | PyTorch 2.x |
| RL framework | Stable-Baselines3 (PPO) |
| Technical indicators | TA-Lib, pandas-ta |
| Web framework | FastAPI (monitoring dashboard) |
| Deployment | Vultr VPS, Ubuntu 22.04, Docker |
| GPU training | Google Colab (T4/A100) |
| Version control | GitHub |
| IDE | VS Code + Python extension |
| Alerting | Telegram Bot API |

---

## Multi-Account Architecture

The system supports adding unlimited OANDA accounts. Each account is defined in `config/accounts.yaml` and identified by a unique alias. Accounts can share the same trading strategy or run independent strategies.

```yaml
accounts:
  - alias: "primary"
    api_key: "your-api-key-1"
    account_id: "001-001-XXXXXXX-001"
    environment: "live"           # or "practice"
    strategy: "default"
    max_risk_per_trade: 0.01      # 1% per trade

  - alias: "secondary"
    api_key: "your-api-key-2"
    account_id: "001-001-XXXXXXX-002"
    environment: "practice"
    strategy: "conservative"
    max_risk_per_trade: 0.005
```

See [docs/MULTI_ACCOUNT.md](docs/MULTI_ACCOUNT.md) for full configuration reference.

---

## OANDA API Key Facts

| Constraint | Value | Impact |
|------------|-------|--------|
| Max candles per request | 5,000 | Pagination required for long histories |
| Historical data depth | 2005 to present | ~20 years of training data available |
| Supported granularities | S5, S10, S15, S30, M1, M2, M4, M5, M10, M15, M30, H1, H2, H3, H4, H6, H8, H12, D, W, M | Full MTF coverage |
| Max leverage (US majors) | 50:1 | Enforced in position sizer |
| Instruments available | 90+ | FX pairs, metals, indices, commodities |
| API maintenance window | Friday ~5pm–6pm ET | Bot pauses automatically |
| Streaming endpoint | Yes (pricing + transactions) | Used for live tick data |

---

## Disclaimer

This software is for educational and research purposes. Forex trading involves substantial risk of loss and is not appropriate for all investors. Past performance of any model or backtest is not indicative of future results. Always test on a demo account before trading live funds. The authors accept no responsibility for financial losses incurred through use of this software.

---

## Documentation Index

| Document | Purpose |
|----------|---------|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Full system design, module interactions, data flow |
| [DATA_PIPELINE.md](docs/DATA_PIPELINE.md) | How historical and live data is collected, stored, paginated |
| [FEATURE_ENGINEERING.md](docs/FEATURE_ENGINEERING.md) | All indicators, patterns, and MTF features |
| [MODEL_SPEC.md](docs/MODEL_SPEC.md) | Neural network and RL agent design |
| [RISK_MANAGEMENT.md](docs/RISK_MANAGEMENT.md) | Position sizing, SL/TP, circuit breakers |
| [OANDA_COMPLIANCE.md](docs/OANDA_COMPLIANCE.md) | Regulatory constraints, spread modeling |
| [MULTI_ACCOUNT.md](docs/MULTI_ACCOUNT.md) | Multi-API and multi-account configuration |
| [DEPLOYMENT.md](docs/DEPLOYMENT.md) | Vultr VPS setup, Docker, process management |
| [TRAINING.md](docs/TRAINING.md) | Google Colab GPU training guide |
| [MONITORING.md](docs/MONITORING.md) | Logging, alerting, dashboard |
