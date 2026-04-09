# DATA_PIPELINE.md — OANDA Data Collection & Storage

## Overview

All training and live trading data is sourced exclusively from OANDA's REST API v20. This guarantees that the model trains on the same bid/ask pricing structure it will encounter in live trading, including OANDA-specific spread behavior, tick volume, and session timing.

---

## OANDA API — Data Constraints & Engineering Solutions

### Constraint 1: 5,000 candle limit per request

OANDA caps every candle API response at 5,000 results. For longer histories this requires paginated fetching.

**Solution — pagination loop:**

```python
def fetch_full_history(client, instrument, granularity, start, end):
    all_candles = []
    cursor = start

    while cursor < end:
        params = {
            "from": cursor.isoformat() + "Z",
            "count": 5000,
            "granularity": granularity,
            "price": "BA"        # bid + ask (not mid)
        }
        response = client.request(InstrumentsCandles(instrument=instrument, params=params))
        candles = response["candles"]

        if not candles:
            break

        # Drop incomplete (in-progress) candles
        complete = [c for c in candles if c["complete"]]
        all_candles.extend(complete)

        # Advance cursor to last returned timestamp
        cursor = parse_datetime(candles[-1]["time"])

    return build_dataframe(all_candles)
```

**Estimated candle counts for common lookback periods:**

| Granularity | 1 Year | 5 Years | 10 Years | Requests needed (10yr) |
|-------------|--------|---------|----------|------------------------|
| M1 | ~260,000 | ~1.3M | ~2.6M | ~520 |
| M15 | ~17,500 | ~87,500 | ~175,000 | ~35 |
| H1 | ~4,380 | ~21,900 | ~43,800 | ~9 |
| H4 | ~1,095 | ~5,475 | ~10,950 | ~3 |
| D | ~260 | ~1,300 | ~2,600 | ~1 |

**Recommendation:** Download M15, H1, H4, and D from 2010 to present as the core training dataset. M1 data should be downloaded only for pairs being scalped.

---

### Constraint 2: Base price vs. account price spread

The candle endpoint (`/v3/instruments/{instrument}/candles`) returns base-group pricing, not your account's specific spread tier. Live spreads may differ slightly.

**Solution:** Store both bid and ask via `price=BA` parameter. When simulating in backtest, use the actual spread data stored. For live trading, always fetch real-time bid/ask from the pricing endpoint before placing orders.

---

### Constraint 3: No streaming OHLC

OANDA's streaming endpoint cannot be used to construct accurate OHLC candles — not all ticks are guaranteed to be delivered.

**Solution:** Use a polling timer to fetch the most recently completed candle from the candle endpoint on each new bar. For M15 trading, poll every 15 minutes at bar close + 2 seconds.

---

### Constraint 4: API maintenance windows

OANDA typically takes the API offline for ~30–60 minutes after the weekly close (Friday ~5pm ET).

**Solution:** The `api_client.py` detects `503` responses and activates a "maintenance hold" that suspends all trading activity and retries connectivity every 5 minutes. All open trades are left in place with their existing SL/TP during maintenance.

---

## Supported Granularities

The following granularities are available from OANDA and can be requested in any combination:

| Code | Description | Recommended use |
|------|-------------|----------------|
| S5 | 5 seconds | Tick-level analysis only |
| S10 | 10 seconds | — |
| S15 | 15 seconds | — |
| S30 | 30 seconds | — |
| M1 | 1 minute | Scalping entry refinement |
| M2 | 2 minutes | — |
| M4 | 4 minutes | — |
| M5 | 5 minutes | Short-term entry |
| M10 | 10 minutes | — |
| **M15** | **15 minutes** | **Primary trading timeframe** |
| M30 | 30 minutes | Trend confirmation |
| **H1** | **1 hour** | **Structure & trend** |
| H2 | 2 hours | — |
| H3 | 3 hours | — |
| **H4** | **4 hours** | **Higher-timeframe bias** |
| H6 | 6 hours | — |
| H8 | 8 hours | — |
| H12 | 12 hours | Session structure |
| **D** | **Daily** | **Macro bias** |
| W | Weekly | Long-term regime |
| M | Monthly | Macro regime |

Bold = recommended for initial training dataset.

---

## Data Storage Architecture

### Directory layout

```
data/
├── raw/                          # Raw Parquet files from OANDA
│   ├── EUR_USD/
│   │   ├── M15.parquet
│   │   ├── H1.parquet
│   │   ├── H4.parquet
│   │   └── D.parquet
│   ├── GBP_USD/
│   │   └── ...
│   └── [28 major/minor pairs]/
│       └── ...
├── processed/                    # Feature-engineered files
│   ├── EUR_USD_M15_features.parquet
│   └── ...
├── models/                       # Trained model artifacts
│   ├── latest.pt
│   ├── v1.0.0.pt
│   └── metadata.json
└── logs/                         # Live trade logs
    ├── trades_2025.jsonl
    └── errors_2025.jsonl
```

### Parquet schema

Each raw Parquet file contains the following columns:

```
time          datetime64[ns, UTC]    Bar open timestamp
open          float64                Opening price (mid)
high          float64                High price (mid)
low           float64                Low price (mid)
close         float64                Closing price (mid)
bid_open      float64                Opening bid price
bid_close     float64                Closing bid price
ask_open      float64                Opening ask price
ask_close     float64                Closing ask price
volume        int64                  Tick volume
spread_pips   float64                (ask_close - bid_close) / pip_size
complete      bool                   False = bar was in-progress when fetched
```

---

## Live Data Collection

### Candle polling (bar-by-bar)

For each active trading pair, a timer fires at every bar close + 2 seconds:

```python
def on_bar_close(instrument, granularity):
    candle = api_client.get_candles(
        instrument=instrument,
        granularity=granularity,
        count=2,          # last 2 bars (1 complete + 1 forming)
        price="BA"
    )
    latest_complete = candle[candle["complete"] == True].iloc[-1]
    feature_buffer.append(instrument, granularity, latest_complete)
    signal = model.predict(feature_buffer.get_window(instrument))
    risk_layer.evaluate(signal, account_state)
```

### Price streaming (tick-by-tick)

For real-time bid/ask spread tracking and execution timing:

```python
stream = api_client.stream_prices(instruments=["EUR_USD", "GBP_USD"])
for tick in stream:
    spread_tracker.update(tick["instrument"], tick["asks"][0]["price"],
                                              tick["bids"][0]["price"])
    execution_layer.check_pending_orders(tick)
```

---

## Data Quality Controls

| Check | Implementation |
|-------|---------------|
| Duplicate timestamps | Deduplicate on ingest; log count of duplicates removed |
| Gaps in data | Gap detection on load; flag gaps > 3 bars for non-weekend periods |
| Incomplete bars | Filter `complete == False` before training; allow in live inference only |
| Price anomalies | Flag bars where range > 5× ATR(20) for manual review |
| Stale data | Alert if latest stored bar is > 2× granularity period old |
| Weekend gaps | Expected gap from Friday close to Sunday open; exclude from gap detection |

---

## Incremental Updates

After the initial full history download, the system performs incremental updates:

```bash
# Run daily via cron on VPS
python scripts/download_history.py --mode incremental --all-pairs
```

Incremental mode fetches only candles newer than the last stored timestamp for each instrument + granularity combination, then appends to the existing Parquet files.

---

## Recommended Initial Download

For a production-ready training dataset, download the following on first setup:

```bash
python scripts/download_history.py \
  --pairs EUR_USD GBP_USD USD_JPY AUD_USD USD_CHF USD_CAD NZD_USD EUR_GBP EUR_JPY GBP_JPY \
  --granularities M15 H1 H4 D \
  --start 2010-01-01 \
  --output-dir data/raw
```

Estimated download time: 20–40 minutes depending on connection speed.
Estimated storage: ~2–4 GB for the above pairs and timeframes.

---

## Instruments Available on OANDA

OANDA provides 90+ instruments. The system supports all of them — just add the instrument code to `config/bot_config.yaml`.

**Major FX pairs (lowest spreads, highest liquidity):**
EUR_USD, GBP_USD, USD_JPY, USD_CHF, USD_CAD, AUD_USD, NZD_USD

**Minor FX pairs:**
EUR_GBP, EUR_JPY, EUR_CHF, EUR_CAD, EUR_AUD, GBP_JPY, GBP_CHF, GBP_CAD, AUD_JPY, CHF_JPY, CAD_JPY, NZD_JPY

**Metals:**
XAU_USD (Gold), XAG_USD (Silver), XPT_USD (Platinum)

**Commodities:**
WTICO_USD (WTI Oil), BCOCOUSD (Brent Oil), NATGAS_USD

**Indices:**
US30_USD (Dow Jones), SPX500_USD (S&P 500), NAS100_USD (NASDAQ), UK100_GBP (FTSE), DE30_EUR (DAX)

> **Note:** Leverage limits vary significantly by instrument type. Always check [OANDA_COMPLIANCE.md](OANDA_COMPLIANCE.md) before adding new instruments to the bot.
