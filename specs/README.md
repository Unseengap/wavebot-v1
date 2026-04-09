# WaveBot — OANDA Wave-Based Forex Trading System

> A precision forex trading bot built exclusively on OANDA's REST API v20.  
> Architecture is 100% wave-native — no bolted-on indicators, no black-box ML as a first layer.  
> The market is waves. The model speaks waves.

---

## Philosophy

Price does not move randomly. It moves in waves — impulse legs followed by corrections, repeating across every timeframe simultaneously. This bot is built on one core belief:

**If you can accurately map the wave state of every timeframe in real time, and understand which timeframes are aligned, you have an edge — before any machine learning is applied.**

Traditional bots apply indicators (RSI, MACD, Bollinger Bands) on top of flat OANDA candle data. Those indicators are derivatives of price. This system works directly with price structure — the peaks, troughs, impulse legs, and corrections that constitute the actual geometry of the market.

The strength hierarchy is the foundation:

```
M1 < M5 < M15 < M30 < H1 < H4 < D < W < M
```

A wave signal on a higher timeframe is not "more important" arbitrarily — it is stronger because more capital has committed to it, more time has been spent establishing it, and more participants are required to reverse it. Smaller timeframes do not oppose higher timeframes — they map inside them, providing precision entry into larger moves.

---

## Repository Structure

```
forex-wave-bot/
│
├── README.md                        ← You are here
├── SPECS.md                         ← Full technical specification
├── WAVE_ENGINE.md                   ← Wave detection algorithm spec
├── STRENGTH_HIERARCHY.md            ← Timeframe weighting system
├── CONFLUENCE_ENGINE.md             ← Multi-timeframe signal fusion
├── OANDA_INTEGRATION.md             ← OANDA API rules, limits, compliance
├── RISK_MANAGEMENT.md               ← SL/TP, position sizing, drawdown
├── BACKTESTING.md                   ← Walk-forward validation methodology
├── DEPLOYMENT.md                    ← VS Code, GitHub, Colab, live rollout
│
├── src/
│   ├── data/
│   │   ├── oanda_client.py          ← OANDA API wrapper (REST v20)
│   │   ├── data_collector.py        ← Paginated historical downloader
│   │   └── stream_handler.py        ← Live price streaming
│   │
│   ├── wave/
│   │   ├── swing_detector.py        ← Peak/trough identification
│   │   ├── wave_labeler.py          ← Wave state labeling per candle
│   │   ├── wave_scorer.py           ← Direction + conviction scoring
│   │   └── amplitude_tracker.py    ← Historical wave amplitude stats
│   │
│   ├── confluence/
│   │   ├── hierarchy_weights.py     ← Timeframe strength coefficients
│   │   ├── alignment_engine.py      ← Cross-timeframe confluence scoring
│   │   └── entry_filter.py          ← Entry conditions from confluence
│   │
│   ├── risk/
│   │   ├── position_sizer.py        ← Kelly / fixed-fraction sizing
│   │   ├── sl_tp_engine.py          ← Wave-origin SL, amplitude-based TP
│   │   └── circuit_breaker.py       ← Daily drawdown halt, max loss
│   │
│   ├── execution/
│   │   ├── order_manager.py         ← OANDA order placement / modification
│   │   ├── trade_monitor.py         ← Live trade tracking
│   │   └── spread_filter.py         ← Entry block on excessive spread
│   │
│   ├── backtest/
│   │   ├── engine.py                ← Walk-forward backtest runner
│   │   ├── spread_simulator.py      ← Variable spread simulation
│   │   └── metrics.py               ← Win rate, Sharpe, MFE, MAE
│   │
│   └── utils/
│       ├── logger.py                ← Structured trade + system logging
│       ├── config.py                ← Central config loader
│       └── timezone_handler.py     ← Session timing (London/NY/Asia)
│
├── config/
│   ├── pairs.yaml                   ← Active instruments list
│   ├── timeframes.yaml              ← Enabled TFs + hierarchy weights
│   └── risk.yaml                   ← Risk parameters
│
├── notebooks/
│   ├── 01_data_collection.ipynb    ← Pull + store OANDA history
│   ├── 02_wave_labeling.ipynb      ← Visualize wave detection
│   ├── 03_confluence_analysis.ipynb ← MTF alignment exploration
│   └── 04_backtest_analysis.ipynb  ← Results + tuning
│
├── tests/
│   ├── test_swing_detector.py
│   ├── test_wave_scorer.py
│   └── test_confluence_engine.py
│
└── requirements.txt
```

---

## Quickstart

### 1. Prerequisites

- Python 3.10+
- OANDA fxTrade or fxTrade Practice account (v20)
- OANDA API access token (generated from Account Management Portal)
- VS Code with Python extension
- Google Colab access (for GPU training of RL layer, Phase 2)

### 2. Installation

```bash
git clone https://github.com/YOUR_USERNAME/forex-wave-bot.git
cd forex-wave-bot
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configuration

```bash
cp config/example.env .env
# Edit .env and add:
# OANDA_API_TOKEN=your_token_here
# OANDA_ACCOUNT_ID=your_account_id
# OANDA_ENVIRONMENT=practice  # or 'live'
```

### 4. Pull Historical Data

```bash
python src/data/data_collector.py --pair EUR_USD --start 2015-01-01 --timeframes M1,M5,M15,H1,H4,D
```

### 5. Run Wave Labeling (Visual Check)

```bash
jupyter notebook notebooks/02_wave_labeling.ipynb
```

### 6. Backtest

```bash
python src/backtest/engine.py --pair EUR_USD --start 2020-01-01 --end 2023-12-31
```

### 7. Paper Trade

```bash
python main.py --mode paper --pairs EUR_USD,GBP_USD,USD_JPY
```

---

## Core Principles

### Accuracy Over Frequency

This bot does not trade for volume. It trades for accuracy. A setup is only valid when multiple timeframes are wave-aligned. The model sits in cash most of the time. When it acts, it acts with high conviction.

### No Entry Against a Higher-Frame Wave

A lower timeframe can only generate an entry signal in the direction of the nearest higher timeframe wave with conviction above threshold. A counter-trend entry is only permitted when the lower timeframe is showing clear exhaustion of the higher frame's wave (wave maturity > 80%) combined with a structural reversal signal.

### Wave Origin is the Stop Loss

Stop losses are not arbitrary pip distances or ATR multiples. They sit at the origin of the wave that generated the entry signal. If the wave's origin is breached, the wave thesis is invalidated. The trade is wrong. Exit.

### Spread is Real Cost

All backtests and live entries account for OANDA's variable bid/ask spread. No entry is placed when the spread exceeds a pair-specific maximum threshold. Spread cost is modeled explicitly in every backtest run.

---

## OANDA Compliance

| Rule | Implementation |
|---|---|
| Max leverage (US) | 50:1 majors, 20:1 minors — hardcoded ceiling |
| 5,000 candle API limit | Paginated downloader with auto-advance |
| Variable spreads | Real-time spread filter + backtest simulation |
| API maintenance windows | Friday ~5pm NY — reconnect with backoff |
| Transaction logging | All orders/trades written to structured log |
| Demo before live | Paper trading mode mirrors live API exactly |

---

## Accuracy Targets

| Metric | Target |
|---|---|
| Win rate | ≥ 60% |
| Risk:Reward | ≥ 1:2 minimum per trade |
| Max daily drawdown | 2% account |
| Max total drawdown | 8% account |
| Sharpe ratio (backtest) | ≥ 1.5 |
| Trades per day (per pair) | 0–5 (quality over quantity) |

---

## Development Roadmap

| Phase | Focus | Status |
|---|---|---|
| Phase 1 | OANDA data pipeline + storage | Build first |
| Phase 2 | Wave detector + labeling engine | Core of the system |
| Phase 3 | Strength hierarchy + confluence scoring | Standalone edge |
| Phase 4 | Risk engine + OANDA compliance | Non-negotiable |
| Phase 5 | Backtesting + walk-forward validation | Before any live capital |
| Phase 6 | Paper trading on OANDA practice | 4–6 weeks minimum |
| Phase 7 | Live deployment (micro lots first) | After Phase 6 proves out |
| Phase 8 | RL agent on top of wave confluence | Only if Phase 3 needs it |

> Phase 8 is deliberately last. If the pure wave confluence system achieves accuracy targets, the RL layer may never be needed.

---

## Important Disclaimer

This system is built for educational and research purposes. Forex trading carries significant risk. Past backtest performance does not guarantee future results. Always start on OANDA's practice account. Never risk capital you cannot afford to lose.
