# ARCHITECTURE.md — System Design & Module Map

## Overview

OandaFX is built as a pipeline of loosely coupled modules. Each module has a single responsibility and communicates through well-defined interfaces. This design allows individual components to be upgraded, replaced, or tested in isolation without affecting the rest of the system.

The system runs as a persistent process on a Vultr VPS, polling OANDA's live pricing stream and executing decisions from the trained model. A separate training pipeline runs offline on Google Colab and produces model artifacts that are pushed to the VPS via GitHub.

---

## High-Level Data Flow

```
OANDA API (Historical)
        │
        ▼
  Data Downloader ──────────────► Parquet Storage
  (paginated, all TFs)                  │
                                        ▼
                                Feature Engineer
                                (indicators, patterns, MTF)
                                        │
                                        ▼
                              Google Colab (GPU Training)
                                        │
                                  Model Artifact (.pt)
                                        │
                                    GitHub Push
                                        │
                                        ▼
                                  Vultr VPS ◄──── OANDA Live Stream
                                        │
                                  Account Manager
                                   /    |    \
                              Acct1  Acct2  AcctN
                                   \    |    /
                                  Risk Validator
                                        │
                                  Order Manager
                                        │
                                  OANDA API (Execution)
                                        │
                                  Trade Monitor
                                        │
                                  Reconciler + Logger
```

---

## Module Specifications

### 1. `core/account_manager.py`

**Purpose:** Orchestrates all OANDA accounts simultaneously. Each account runs in its own thread. The manager handles startup, shutdown, health checks, and inter-account coordination (e.g., preventing correlated overexposure across accounts).

**Responsibilities:**
- Load and validate all accounts from `config/accounts.yaml`
- Instantiate one `api_client.APIClient` per account
- Spin up one `session.TradingSession` per account in a dedicated thread
- Expose a unified status interface for the monitoring dashboard
- Propagate global circuit breaker signals (e.g., OANDA maintenance window) to all sessions

**Key interfaces:**
```python
class AccountManager:
    def __init__(self, config_path: str): ...
    def start_all(self): ...
    def stop_all(self): ...
    def get_status(self) -> dict: ...
    def add_account(self, account_config: dict): ...    # hot-add without restart
    def remove_account(self, alias: str): ...           # graceful shutdown of one account
```

---

### 2. `core/api_client.py`

**Purpose:** Thin wrapper around `oandapyV20` that adds retry logic, rate limiting, error classification, and structured logging for every API call.

**Responsibilities:**
- Authenticate with a specific OANDA API key and account ID
- Wrap all REST v20 endpoints used by the system (candles, orders, trades, pricing, account summary)
- Implement exponential backoff on `503`, `429`, and connection errors
- Detect and handle the Friday maintenance window gracefully
- Log every request and response at DEBUG level

**Retry policy:**
```
Attempt 1: immediate
Attempt 2: wait 2s
Attempt 3: wait 4s
Attempt 4: wait 8s
Attempt 5: wait 16s → raise after 5 failures
```

**Key interfaces:**
```python
class APIClient:
    def __init__(self, api_key: str, account_id: str, environment: str): ...
    def get_candles(self, instrument: str, granularity: str,
                    from_dt: datetime, to_dt: datetime) -> pd.DataFrame: ...
    def stream_prices(self, instruments: list[str]) -> Generator: ...
    def place_order(self, order: dict) -> dict: ...
    def modify_trade(self, trade_id: str, sl: float, tp: float) -> dict: ...
    def close_trade(self, trade_id: str) -> dict: ...
    def get_account_summary(self) -> dict: ...
    def get_open_trades(self) -> list[dict]: ...
```

---

### 3. `data/downloader.py`

**Purpose:** Downloads complete historical OANDA candle data for any instrument and granularity combination. Handles pagination transparently — OANDA caps responses at 5,000 candles per request, so long histories require many sequential requests.

**Pagination logic:**
```
1. Set cursor = start_date
2. Request 5,000 candles from cursor
3. Advance cursor to last returned candle timestamp
4. Deduplicate overlap (first candle of new batch = last of previous)
5. Repeat until cursor >= end_date or "now"
6. Concatenate all batches into a single DataFrame
7. Save to Parquet partitioned by instrument + granularity
```

**Data schema per candle:**
```
time          datetime64[ns, UTC]
open          float64
high          float64
low           float64
close         float64
volume        int64
complete      bool
spread_bid    float64    (from separate pricing endpoint)
spread_ask    float64
```

**Storage:** Each instrument + granularity combination is stored as a separate Parquet file:
```
data/
  EUR_USD/
    M15.parquet
    H1.parquet
    H4.parquet
    D.parquet
  GBP_USD/
    M15.parquet
    ...
```

---

### 4. `data/stream.py`

**Purpose:** Maintains a persistent WebSocket connection to OANDA's pricing stream endpoint. Distributes incoming ticks to all active trading sessions via an internal pub/sub queue.

**Key behavior:**
- Subscribes to all instruments being actively traded across all accounts
- Reconnects automatically on disconnection
- Does NOT construct OHLC candles from the stream (stream candles are unreliable per OANDA documentation) — instead, closed candles are fetched from the candle endpoint on a timer
- Passes raw bid/ask ticks to the execution layer for spread-aware order placement

---

### 5. `features/engineer.py`

**Purpose:** Transforms raw OANDA candle data into a rich feature matrix ready for model input. Orchestrates indicator calculation, pattern detection, and multi-timeframe alignment.

**Feature vector per bar (example, M15 base timeframe):**

| Category | Features |
|----------|---------|
| Price action | Returns (1, 3, 5, 10, 20 bars), log returns, bar range, body ratio |
| Trend indicators | EMA(8,21,50,200), SMA(20,50,200), VWAP, ADX, Aroon |
| Momentum | RSI(14), Stochastic(14,3), MACD(12,26,9), CCI(20), Williams %R |
| Volatility | ATR(14), Bollinger Bands(20,2), Keltner Channels, Historical Vol(20) |
| Volume | OBV, Volume ratio (current/20-bar avg), Chaikin Money Flow |
| Support/Resistance | Distance to nearest swing high/low (20, 50 bars), pivot levels |
| Patterns | 15 candlestick patterns (doji, engulfing, hammer, etc.) as binary flags |
| Chart patterns | HnS probability score, wedge detected, triangle detected |
| Multi-timeframe | H1 trend direction, H4 trend direction, D momentum |
| Session | London open flag, NY open flag, overlap flag, Asia flag, time of day |
| Spread | Current spread in pips, spread vs 20-bar avg ratio |
| Account state | Current P&L (unrealized), open trade count, margin utilization |

Total feature dimensions: approximately 85–120 depending on configuration.

---

### 6. `models/` — Model Ensemble

The system uses three cooperating models:

**A. Transformer sequence model** (`transformer.py`)
- Input: 128-bar lookback window, full feature vector per bar
- Output: Directional signal (long / short / flat), confidence score
- Architecture: 4 encoder layers, 8 attention heads, 256-dim hidden
- Trained: supervised on historical OANDA data

**B. PPO reinforcement learning agent** (`rl_agent.py`)
- Input: current market state + account state
- Output: action (buy / sell / hold / close), SL distance (in pips), TP distance (in pips), position size (fraction of balance)
- Reward: risk-adjusted return, penalized for drawdown and overtrading
- Trained: in a simulated environment using historical OANDA data

**C. SL/TP dynamic module** (`sl_tp_module.py`)
- Input: current ATR, recent volatility regime, support/resistance distances, account risk budget
- Output: ATR-adjusted SL distance, TP distance, risk:reward ratio
- This module overrides the RL agent's raw SL/TP when market volatility is extreme

**Ensemble voting:**
```
Final signal = weighted vote(Transformer, RL agent)
  weights updated weekly based on recent live performance
  minimum confidence threshold: 0.65 to trigger entry
```

---

### 7. `risk/` — Risk Management Layer

All signals from the model pass through the risk layer before reaching the execution layer. The risk layer can veto any trade.

**Checks performed before every order:**

1. Daily loss limit not exceeded (`circuit_breaker.py`)
2. Max concurrent trades not exceeded (configurable per account)
3. Correlated pair exposure within limits (e.g., EUR_USD + GBP_USD both long = high correlation)
4. Leverage utilization below max margin requirement
5. Stop loss present and >= 1× ATR from entry
6. Take profit present and risk:reward >= 1.2:1
7. Position size within `max_risk_per_trade` % of account balance
8. OANDA leverage rules not violated (50:1 for US majors)
9. Not within 10 minutes of OANDA maintenance window

If any check fails, the trade is rejected and logged with the rejection reason.

---

### 8. `execution/order_manager.py`

**Purpose:** Translates validated trading decisions into OANDA API order requests. Handles market orders, limit orders, stop entries, and OCO bracket orders.

**Order types supported:**
- Market order with attached SL and TP (primary method)
- Limit entry order (for pending setups)
- Stop entry order (for breakout setups)
- Trailing stop modification on open trades

**Slippage handling:**
- All market orders include a `priceBound` set to entry price ± 1.5× spread
- Orders rejected by OANDA due to slippage are re-evaluated by the model before retry

---

### 9. `backtest/engine.py`

**Purpose:** Walk-forward backtesting engine that simulates live trading conditions as closely as possible.

**Simulation features:**
- Variable bid/ask spread (not mid-price) using stored OANDA spread data
- Commission simulation (none for OANDA spot FX — spread is the cost)
- Slippage model (random draw from historical spread distribution)
- No look-ahead bias (features computed using only data available at bar close)
- Walk-forward splits: train on N years, test on following M months, roll forward

**Output metrics:**
- Total return, annualized return
- Sharpe ratio, Sortino ratio
- Maximum drawdown, average drawdown
- Win rate, profit factor
- Average MFE (Maximum Favorable Excursion)
- Average MAE (Maximum Adverse Excursion)
- Trade frequency, average trade duration

---

## Thread & Process Model (VPS)

```
Main process
├── AccountManager (main thread)
│   ├── TradingSession:primary   (thread 1)
│   │   ├── PriceStream listener
│   │   ├── Feature computation
│   │   ├── Model inference
│   │   └── Order execution
│   ├── TradingSession:secondary (thread 2)
│   │   └── ...
│   └── TradingSession:N         (thread N)
├── Monitoring dashboard (FastAPI, port 8080)  (process 2)
└── Log aggregator                              (process 3)
```

Each trading session thread is independent. A crash in one session does not affect others. The AccountManager monitors thread health and restarts crashed sessions after a backoff delay.

---

## Configuration Files

### `config/accounts.yaml`
Defines all OANDA API credentials and per-account strategy settings. See [MULTI_ACCOUNT.md](MULTI_ACCOUNT.md).

### `config/bot_config.yaml`
Global strategy parameters — pairs to trade, timeframes to use, model path, feature set, risk parameters. This file is the single source of truth for what the bot does.

```yaml
strategy:
  name: "default"
  pairs:
    - EUR_USD
    - GBP_USD
    - USD_JPY
    - AUD_USD
  base_timeframe: "M15"
  higher_timeframes:
    - H1
    - H4
    - D
  model_path: "models/latest.pt"
  min_confidence: 0.65

features:
  lookback_bars: 128
  include_volume: true
  include_sentiment: false    # set true when news API connected

risk:
  max_risk_per_trade: 0.01    # 1% of account balance
  max_daily_loss: 0.03        # 3% halt circuit breaker
  max_open_trades: 5
  max_correlated_pairs: 2
  min_rr_ratio: 1.2
```
