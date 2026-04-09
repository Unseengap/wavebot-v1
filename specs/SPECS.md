# SPECS.md — WaveBot Full Technical Specification

## System Overview

WaveBot is a wave-native forex trading system that identifies, scores, and trades confluent wave setups across multiple timeframes using OANDA's REST API v20 as the exclusive data and execution layer.

The system has no external data dependencies. Every price, every candle, every spread value, and every order flows through OANDA. This is intentional — training and trading on the same broker's data eliminates pricing discrepancies that destroy live performance.

---

## Technology Stack

| Layer | Technology | Reason |
|---|---|---|
| Language | Python 3.10+ | oandapyV20 library, NumPy, pandas ecosystem |
| API | OANDA REST v20 | Exclusive broker + data source |
| Data storage | SQLite (local) → Parquet (archive) | Fast local reads, efficient long-term storage |
| Wave computation | NumPy + custom algorithms | Speed, no ML overhead in wave detection |
| Backtesting | Custom engine (no external framework) | Full control over spread simulation and wave-aware logic |
| Visualization | Matplotlib + Plotly (notebooks) | Wave labeling inspection |
| Deployment | Python process + systemd / Colab scheduled | Lightweight, no cloud required initially |
| Version control | GitHub | CI for backtest regression tests |
| GPU training | Google Colab (Phase 8 only) | RL agent if needed later |

---

## OANDA API Constraints — Hardcoded Rules

These are not configurable. They reflect OANDA's actual regulatory and technical constraints.

### Data Constraints

```python
OANDA_MAX_CANDLES_PER_REQUEST = 5000
OANDA_HISTORICAL_START = "2005-01-01"  # Earliest available data

OANDA_GRANULARITIES = [
    "S5", "S10", "S15", "S30",           # Seconds
    "M1", "M2", "M3", "M4", "M5",        # Minutes (1–5)
    "M10", "M15", "M30",                  # Minutes (10–30)
    "H1", "H2", "H3", "H4", "H6", "H8", "H12",  # Hours
    "D", "W", "M"                         # Day, Week, Month
]

# API maintenance window — OANDA goes offline
OANDA_MAINTENANCE_WINDOW = "Friday 17:00 - Sunday 17:00 ET"
```

### Leverage Constraints (US Accounts — NFA Regulated)

```python
LEVERAGE_LIMITS = {
    "majors":  50,   # EUR/USD, GBP/USD, USD/JPY, USD/CAD, AUD/USD, USD/CHF, NZD/USD
    "minors":  20,   # Cross pairs (EUR/GBP, GBP/JPY, etc.)
    "exotics": 10,   # Less liquid pairs
    "metals":  20,   # XAU/USD, XAG/USD
}
```

### Order Types Available

```python
SUPPORTED_ORDER_TYPES = [
    "MARKET",           # Immediate execution at current price
    "LIMIT",            # Execute at specified price or better
    "STOP",             # Execute when price reaches stop level
    "MARKET_IF_TOUCHED",# MIT orders
    "TAKE_PROFIT",      # Attached TP on open trade
    "STOP_LOSS",        # Attached SL on open trade
    "TRAILING_STOP_LOSS"# Dynamic trailing stop
]
```

---

## Supported Instruments

### Priority Tier 1 — Major Pairs (tightest spreads, highest liquidity)

```yaml
- EUR_USD
- GBP_USD
- USD_JPY
- USD_CAD
- AUD_USD
- USD_CHF
- NZD_USD
```

### Priority Tier 2 — Minor Pairs (active when majors show no setup)

```yaml
- EUR_GBP
- EUR_JPY
- GBP_JPY
- AUD_JPY
- EUR_AUD
- GBP_CAD
```

### Priority Tier 3 — Metals (treated as FX pairs)

```yaml
- XAU_USD   # Gold
- XAG_USD   # Silver
```

The bot trades Tier 1 first. Tier 2 and 3 only activate when Tier 1 shows insufficient confluence. This preserves spread efficiency — major pairs have the lowest transaction cost.

---

## Active Timeframes

The system uses six core timeframes. Monthly and weekly are loaded as context only — they do not generate entries.

```yaml
timeframes:
  entry_frames:         # Generate actual entry signals
    - M1
    - M5
    - M15

  confirmation_frames:  # Must align with entry direction
    - H1
    - H4

  context_frames:       # Directional bias only, no entries
    - D
    - W                 # Loaded but not traded against directly
```

---

## Data Pipeline Specification

### Historical Data Collection

```python
class DataCollector:
    """
    Pulls OANDA candle data with automatic pagination.
    Handles the 5,000 candle-per-request limit transparently.
    Stores to SQLite with Parquet archiving.
    """

    def collect(
        self,
        instrument: str,        # e.g. "EUR_USD"
        granularity: str,       # e.g. "M15"
        start_date: str,        # ISO 8601 "2018-01-01T00:00:00Z"
        end_date: str = None,   # Defaults to now
        price: str = "MBA"      # Mid, Bid, Ask — we want all three
    ) -> pd.DataFrame:
        """
        Returns DataFrame with columns:
        time, open_mid, high_mid, low_mid, close_mid,
        open_bid, high_bid, low_bid, close_bid,
        open_ask, high_ask, low_ask, close_ask,
        volume, complete
        """
```

**Critical**: Always collect bid AND ask, not just mid. Spread simulation in backtesting requires both. Most implementations only pull mid price — this is a fatal flaw for accurate backtesting.

### Live Data Streaming

```python
class StreamHandler:
    """
    Maintains a live price stream via OANDA streaming endpoint.
    On each tick: updates current candle state for all active timeframes.
    On candle close: triggers wave detection pipeline.
    """

    # OANDA streaming endpoint
    STREAM_URL = "https://stream-fxtrade.oanda.com/v3/accounts/{}/pricing/stream"

    def on_candle_close(self, instrument: str, granularity: str, candle: dict):
        """Called when a candle closes. Triggers wave detection."""
        self.wave_engine.process_closed_candle(instrument, granularity, candle)
```

### Data Storage Schema

```sql
-- SQLite schema
CREATE TABLE candles (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    instrument  TEXT NOT NULL,
    granularity TEXT NOT NULL,
    time        TEXT NOT NULL,
    open_mid    REAL,
    high_mid    REAL,
    low_mid     REAL,
    close_mid   REAL,
    open_bid    REAL,
    high_bid    REAL,
    low_bid     REAL,
    close_bid   REAL,
    open_ask    REAL,
    high_ask    REAL,
    low_ask     REAL,
    close_ask   REAL,
    volume      INTEGER,
    complete    INTEGER,
    UNIQUE(instrument, granularity, time)
);

CREATE INDEX idx_candles_lookup
ON candles(instrument, granularity, time);
```

---

## Configuration Files

### config/pairs.yaml

```yaml
pairs:
  tier_1:
    - instrument: EUR_USD
      pip_location: -4
      spread_max_pips: 1.5      # Don't enter if spread exceeds this
      min_wave_pips: 10         # Minimum wave size to be valid on M1

    - instrument: GBP_USD
      pip_location: -4
      spread_max_pips: 2.0
      min_wave_pips: 12

    - instrument: USD_JPY
      pip_location: -2
      spread_max_pips: 1.5
      min_wave_pips: 10

  tier_2:
    - instrument: EUR_GBP
      pip_location: -4
      spread_max_pips: 2.5
      min_wave_pips: 8

    - instrument: GBP_JPY
      pip_location: -2
      spread_max_pips: 3.0
      min_wave_pips: 15
```

### config/timeframes.yaml

```yaml
# Two-Stage Scoring Architecture:
# Stage 1 (Directional Gate): D and W set a binary bias filter (BULLISH/BEARISH/NEUTRAL)
# Stage 2 (Entry Quality):    M1–H4 compute the actual confluence score
# D and W do NOT contribute to the confluence score — they are gate-only.

hierarchy:
  # --- Entry Frames (Stage 2 — contribute to confluence score) ---
  M1:
    entry_weight: 1.0
    label: "1 Minute"
    role: entry

  M5:
    entry_weight: 2.0
    label: "5 Minute"
    role: entry

  M15:
    entry_weight: 3.5
    label: "15 Minute"
    role: entry

  # --- Confirmation Frames (Stage 2 — contribute to confluence score) ---
  H1:
    entry_weight: 5.0       # Was 6.0 — flattened to give entry frames more influence
    label: "1 Hour"
    role: confirmation

  H4:
    entry_weight: 7.0       # Was 10.0 — flattened to prevent H4 domination
    label: "4 Hour"
    role: confirmation

  # --- Context Frames (Stage 1 — directional gate only, NOT in score) ---
  D:
    label: "Daily"
    role: context_gate       # Binary gate: BULLISH / BEARISH / NEUTRAL

  W:
    label: "Weekly"
    role: context_gate       # Overrides D when weekly conviction > 0.5

confluence:
  min_score_to_enter: 0.72    # 0.0–1.0. Pure wave alignment threshold — never modified by session
  min_frames_aligned: 3       # At least 3 entry/confirmation frames must agree
  require_h1_alignment: true  # H1 must be in agreement for any entry
  require_h4_alignment: false # H4 agreement increases score but not required
  counter_gate_override: 0.85 # Score required to trade against the directional gate
```

### config/risk.yaml

```yaml
risk:
  max_risk_per_trade: 0.01      # 1% of account per trade
  max_daily_drawdown: 0.02      # 2% — halt all trading if hit
  max_total_drawdown: 0.08      # 8% — emergency stop, manual review
  max_open_trades: 3            # Across all pairs simultaneously
  max_trades_per_pair: 1        # One position per instrument at a time

position_sizing:
  method: "fixed_fraction"      # "fixed_fraction" or "kelly"
  fixed_fraction: 0.01          # Risk 1% of balance per trade

sl_tp:
  sl_method: "wave_origin"      # Stop at origin of signal wave
  tp_method: "wave_amplitude"   # TP based on historical wave amplitude
  min_rr_ratio: 2.0             # Minimum 1:2 risk:reward to enter
  sl_buffer_pips: 2             # Extra pips beyond wave origin for SL

session_sizing:
  # Session multiplier affects POSITION SIZE, not confluence score.
  # The entry threshold (0.72) is never modified by session.
  London_Open: 1.10             # 08:00–10:00 UTC
  London_NY_Overlap: 1.25       # 13:00–17:00 UTC — peak volume
  NY_Session: 1.00              # 13:00–22:00 UTC — baseline
  Asia_Session: 0.60            # 00:00–08:00 UTC — reduced size
  Dead_Zone: 0.00               # 22:00–00:00 UTC — no entries

spread:
  apply_to_backtest: true       # Always true — no exception
  spread_model: "variable"      # Use actual bid/ask from OANDA data

slippage:
  apply_to_backtest: true       # Always true — applied on top of spread
  volatility_threshold: 2.0     # Candle range > 2× ATR triggers elevated slippage
  volatility_multiplier: 2.5    # Slippage up to 2.5× base during fast moves
```

---

## Execution Flow

This is the exact sequence of operations the bot executes on every closed candle:

```
1. CANDLE CLOSES (OANDA stream event)
       │
       ▼
2. WAVE DETECTOR runs on closed candle
   → Identifies new swing high/low if formed
   → Updates wave state: IMPULSE / CORRECTION / RANGING
   → Calculates wave maturity (0–100%)
       │
       ▼
3. WAVE SCORER updates timeframe score
   → Direction: +1.0 (bullish) to -1.0 (bearish)
   → Conviction: 0.0 (none) to 1.0 (max)
   → Maturity: used to flag exhaustion
       │
       ▼
4. DIRECTIONAL GATE (Stage 1) checks D/W bias
   → D and W wave scores → BULLISH / BEARISH / NEUTRAL gate
   → If NEUTRAL: no entries in either direction (wait)
   → If BULLISH/BEARISH: only entries in gate direction allowed
       │
       ▼
5. CONFLUENCE ENGINE (Stage 2) scores entry/confirmation frames
   → Uses flattened entry weights (M1–H4 only, D/W excluded)
   → Calculates alignment score (-1.0 to +1.0)
   → Score reflects wave alignment quality, unmodified by session
       │
       ▼
6. ENTRY FILTER checks pre-conditions
   → Score above 0.72 threshold?
   → Direction matches Stage 1 gate?
   → Spread within limits? (OANDA live bid/ask)
   → No existing position on this pair?
   → Min R:R achievable at current wave amplitude?
       │
       ▼
7. RISK ENGINE calculates trade parameters
   → Position size (1% risk on account)
   → Session multiplier applied to position size (not score)
   → Stop loss (wave origin + buffer)
   → Take profit (wave amplitude projection)
       │
       ▼
8. ORDER MANAGER submits to OANDA
   → Market order with attached SL + TP
   → Trade ID logged
   → All parameters written to trade log
       │
       ▼
9. TRADE MONITOR tracks open position
   → Checks for wave reversal signal
   → Optional: partial close at 1:1 R:R
   → Hard SL and TP managed by OANDA (not local)
```

---

## Error Handling Requirements

All OANDA API calls must implement:

```python
RETRY_CONFIG = {
    "max_retries": 5,
    "backoff_base": 2,          # Exponential: 2, 4, 8, 16, 32 seconds
    "retry_on_status": [429, 500, 502, 503, 504],
    "no_retry_on_status": [400, 401, 403, 404],  # Fatal errors — stop + alert
}

# OANDA maintenance detection
MAINTENANCE_INDICATORS = [
    "401 Unauthorized during known maintenance window",
    "Connection refused on stream endpoint",
    "HTTP 503 Service Unavailable"
]
# On maintenance detection: close stream, wait, reconnect — do NOT place orders
```

---

## Testing Requirements

Every module has a corresponding test. The CI pipeline (GitHub Actions) runs the full test suite on every push.

```bash
# Run all tests
pytest tests/ -v

# Run specific module
pytest tests/test_swing_detector.py -v

# Run backtest regression (confirms no performance degradation)
pytest tests/test_backtest_regression.py -v --pair EUR_USD
```

Minimum test coverage: **80%** on wave engine modules. The wave detector is the most critical component — it must be tested exhaustively against known wave patterns.

---

## Logging Specification

All trades and system events are written to structured JSON logs.

```json
{
  "timestamp": "2024-03-15T14:32:01.443Z",
  "event": "TRADE_OPENED",
  "instrument": "EUR_USD",
  "direction": "LONG",
  "entry_price": 1.08543,
  "stop_loss": 1.08421,
  "take_profit": 1.08787,
  "position_size": 10000,
  "risk_amount": 12.20,
  "risk_pct": 0.01,
  "confluence_score": 0.81,
  "frames_aligned": ["M15", "H1", "H4"],
  "signal_wave_origin": 1.08421,
  "signal_wave_tf": "M15",
  "spread_at_entry": 0.8,
  "oanda_trade_id": "6381947201",
  "session": "London-NY_Overlap"
}
```

---

## Performance Benchmarks

These benchmarks must be met in backtesting before any live capital is deployed.

| Metric | Minimum | Target |
|---|---|---|
| Win rate | 55% | 65%+ |
| Average R:R | 1:2 | 1:2.5+ |
| Profit factor | 1.5 | 2.0+ |
| Max drawdown | < 10% | < 6% |
| Sharpe ratio | 1.2 | 1.8+ |
| Calmar ratio | 1.0 | 1.5+ |
| Monthly trades (per pair) | 10+ | 20–50 |

If backtesting on EUR_USD from 2018–2023 does not meet the minimum benchmarks, do not proceed to paper trading. Diagnose the wave detector first.
