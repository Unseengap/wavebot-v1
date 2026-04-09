# OANDA_INTEGRATION.md — OANDA API Rules, Limits & Compliance

## Overview

OANDA is the exclusive data and execution layer for WaveBot. No external data feeds, no alternative brokers, no supplementary price sources. This is a design decision — training and trading on the same broker's data eliminates the pricing discrepancies that destroy live performance when bots are built on one data source and traded on another.

---

## Account Setup

### Required Account Type

- **OANDA v20 account** (not legacy v1)
- v20 accounts work exclusively with the REST API v20
- Practice account: `fxTrade Practice` — identical API behavior to live
- Live account: `fxTrade`

### API Authentication

```python
# Generate token at: https://www.oanda.com/account/tpa/personal_token
# Token works across both practice and live endpoints

OANDA_ENVIRONMENTS = {
    "practice": {
        "api":    "https://api-fxpractice.oanda.com",
        "stream": "https://stream-fxpractice.oanda.com",
    },
    "live": {
        "api":    "https://api-fxtrade.oanda.com",
        "stream": "https://stream-fxtrade.oanda.com",
    }
}

headers = {
    "Authorization": f"Bearer {OANDA_API_TOKEN}",
    "Content-Type": "application/json",
}
```

### Environment Variable Configuration

```bash
# .env file (never commit to GitHub)
OANDA_API_TOKEN=your_64_character_token_here
OANDA_ACCOUNT_ID=your_account_id_here
OANDA_ENVIRONMENT=practice    # Switch to 'live' only after Phase 6 complete
```

---

## Historical Data API

### Candle Endpoint

```
GET /v3/instruments/{instrument}/candles
```

**Parameters:**

| Parameter | Values | Notes |
|---|---|---|
| `price` | `M`, `B`, `A`, `BA`, `MBA` | Always use `MBA` — mid, bid, AND ask |
| `granularity` | See list below | Timeframe |
| `from` | ISO 8601 datetime | Start of range |
| `to` | ISO 8601 datetime | End of range (defaults to now) |
| `count` | 1–5000 | Max candles per request |
| `includeFirst` | true/false | Include candle at `from` time |

**Critical constraint: 5,000 candle maximum per request.**

This means to download 5 years of M15 data (≈175,200 candles), you need 36 paginated requests minimum. The `DataCollector` handles this automatically.

### Paginated Downloader

```python
class DataCollector:

    def collect_range(
        self,
        instrument: str,
        granularity: str,
        start: str,            # "2018-01-01T00:00:00Z"
        end: str = None,       # Defaults to now
    ) -> pd.DataFrame:

        all_candles = []
        current_start = start
        page_size = 5000

        while True:
            params = {
                "price": "MBA",              # Mid, Bid, Ask — all three
                "granularity": granularity,
                "from": current_start,
                "count": page_size,
                "includeFirst": "true",
            }

            response = self._request(
                f"/v3/instruments/{instrument}/candles",
                params=params
            )

            candles = response.get("candles", [])
            if not candles:
                break

            # Filter to completed candles only
            complete = [c for c in candles if c.get("complete", False)]
            all_candles.extend(complete)

            if len(candles) < page_size:
                break  # Got fewer than requested — we're at the end

            # Advance to next page
            last_time = candles[-1]["time"]
            current_start = last_time  # includeFirst=true handles the overlap

            time.sleep(0.2)  # Rate limit courtesy — OANDA allows ~100 req/sec

        return self._to_dataframe(all_candles)
```

### Available Granularities

```python
GRANULARITIES = {
    # Seconds — only use for micro-analysis, not primary trading
    "S5":  5,
    "S10": 10,
    "S15": 15,
    "S30": 30,

    # Minutes
    "M1":  60,
    "M2":  120,
    "M4":  240,
    "M5":  300,
    "M10": 600,
    "M15": 900,
    "M30": 1800,

    # Hours
    "H1":  3600,
    "H2":  7200,
    "H3":  10800,
    "H4":  14400,
    "H6":  21600,
    "H8":  28800,
    "H12": 43200,

    # Day / Week / Month
    "D":   86400,
    "W":   604800,
    "M":   2592000,   # Approximate
}
```

### Note on Candle Alignment

OANDA aligns candles to a daily boundary. The default daily alignment is 17:00 New York time (the standard forex day close). This is the correct setting — do not change it. Weekly candles open Monday and close Friday at 17:00 NY.

---

## Live Streaming API

### Pricing Stream Endpoint

```
GET /v3/accounts/{accountID}/pricing/stream?instruments=EUR_USD,GBP_USD
```

The stream returns a continuous JSON stream of price ticks. Each tick contains bid and ask prices. The stream does NOT return OHLC candle data — that must be constructed locally from ticks, or pulled via the candle endpoint after each bar closes.

**Recommended approach:** Use the stream for real-time price awareness (spread checking, stop monitoring). Build candle state from the candle endpoint on close.

```python
class StreamHandler:

    def run(self, instruments: list[str]):
        url = f"{BASE_URL}/v3/accounts/{ACCOUNT_ID}/pricing/stream"
        params = {"instruments": ",".join(instruments)}

        with requests.get(url, headers=HEADERS, params=params, stream=True) as resp:
            for line in resp.iter_lines():
                if line:
                    tick = json.loads(line)

                    if tick.get("type") == "PRICE":
                        self._on_price_tick(tick)
                    elif tick.get("type") == "HEARTBEAT":
                        self._on_heartbeat(tick)  # Confirm stream alive

    def _on_price_tick(self, tick: dict):
        instrument = tick["instrument"]
        bid = float(tick["bids"][0]["price"])
        ask = float(tick["asks"][0]["price"])
        spread_pips = (ask - bid) / self.pip_size(instrument)

        self.current_spreads[instrument] = spread_pips
        self.current_prices[instrument] = {"bid": bid, "ask": ask}

        # Check if any open trade SL/TP is hit (backup check — OANDA handles this)
        self.trade_monitor.check_levels(instrument, bid, ask)
```

---

## Order Management

### Place Market Order with SL and TP

```python
def place_market_order(
    instrument: str,
    units: int,              # Positive = long, negative = short
    stop_loss_price: float,
    take_profit_price: float,
    client_trade_id: str,    # Your internal ID for logging
) -> dict:

    order_body = {
        "order": {
            "type": "MARKET",
            "instrument": instrument,
            "units": str(units),
            "timeInForce": "FOK",    # Fill or Kill — no partial fills
            "stopLossOnFill": {
                "price": str(round(stop_loss_price, 5)),
                "timeInForce": "GTC"  # Good till cancelled
            },
            "takeProfitOnFill": {
                "price": str(round(take_profit_price, 5)),
                "timeInForce": "GTC"
            },
            "clientExtensions": {
                "id": client_trade_id,
                "comment": "WaveBot auto entry"
            }
        }
    }

    response = requests.post(
        f"{BASE_URL}/v3/accounts/{ACCOUNT_ID}/orders",
        headers=HEADERS,
        json=order_body,
    )

    return response.json()
```

### Modify Open Trade (Move SL/TP)

```python
def modify_trade(
    trade_id: str,
    new_stop_loss: float = None,
    new_take_profit: float = None,
) -> dict:

    body = {}
    if new_stop_loss:
        body["stopLoss"] = {"price": str(round(new_stop_loss, 5))}
    if new_take_profit:
        body["takeProfit"] = {"price": str(round(new_take_profit, 5))}

    response = requests.put(
        f"{BASE_URL}/v3/accounts/{ACCOUNT_ID}/trades/{trade_id}/orders",
        headers=HEADERS,
        json=body,
    )

    return response.json()
```

### Close Trade (Emergency / Circuit Breaker)

```python
def close_trade(trade_id: str, units: str = "ALL") -> dict:
    response = requests.put(
        f"{BASE_URL}/v3/accounts/{ACCOUNT_ID}/trades/{trade_id}/close",
        headers=HEADERS,
        json={"units": units},
    )
    return response.json()
```

### Close All Open Trades (Emergency Stop)

```python
def emergency_close_all() -> list[dict]:
    """Called by circuit breaker when daily drawdown limit is hit."""
    trades = self.get_open_trades()
    results = []
    for trade in trades:
        result = self.close_trade(trade["id"])
        results.append(result)
        log.critical(f"EMERGENCY CLOSE: {trade['instrument']} trade {trade['id']}")
        time.sleep(0.1)  # Small delay between closes
    return results
```

---

## Account State Monitoring

```python
def get_account_summary() -> dict:
    """
    Returns current account state including:
    - balance
    - unrealizedPL
    - marginUsed
    - marginAvailable
    - openTradeCount
    - openPositionCount
    """
    response = requests.get(
        f"{BASE_URL}/v3/accounts/{ACCOUNT_ID}/summary",
        headers=HEADERS,
    )
    return response.json()["account"]

def get_open_trades() -> list[dict]:
    response = requests.get(
        f"{BASE_URL}/v3/accounts/{ACCOUNT_ID}/openTrades",
        headers=HEADERS,
    )
    return response.json().get("trades", [])
```

---

## Transaction Logging (Compliance)

OANDA logs every transaction server-side. WaveBot additionally logs all transactions locally for analysis and compliance.

```python
def get_transaction_history(from_time: str, to_time: str) -> list[dict]:
    """
    Retrieves all account transactions from OANDA.
    Transaction types include: ORDER_FILL, STOP_LOSS_ORDER, TAKE_PROFIT_ORDER,
    MARKET_ORDER, TRADE_CLOSE, etc.
    """
    params = {
        "from": from_time,
        "to": to_time,
        "type": "ORDER_FILL,STOP_LOSS_ORDER,TAKE_PROFIT_ORDER,TRADE_CLOSE"
    }
    response = requests.get(
        f"{BASE_URL}/v3/accounts/{ACCOUNT_ID}/transactions",
        headers=HEADERS,
        params=params,
    )
    return response.json().get("transactions", [])
```

All transactions are synced nightly and stored in SQLite alongside wave analysis data. This creates a complete audit trail.

---

## Error Handling & Resilience

### OANDA Error Codes

```python
OANDA_ERRORS = {
    # 4xx — Client errors (your problem, do not retry blindly)
    400: "BAD_REQUEST — Invalid parameters. Log and fix.",
    401: "UNAUTHORIZED — Token invalid or expired. Stop bot, alert.",
    403: "FORBIDDEN — Account restriction. Stop bot, investigate.",
    404: "NOT_FOUND — Instrument or trade ID doesn't exist.",
    429: "RATE_LIMITED — Too many requests. Back off and retry.",

    # 5xx — Server errors (OANDA's problem, retry with backoff)
    500: "INTERNAL_SERVER_ERROR — Retry with backoff.",
    502: "BAD_GATEWAY — Retry with backoff.",
    503: "SERVICE_UNAVAILABLE — Likely maintenance. Wait and retry.",
    504: "GATEWAY_TIMEOUT — Retry with backoff.",
}

def handle_api_response(response: requests.Response) -> dict:
    if response.status_code == 200 or response.status_code == 201:
        return response.json()

    elif response.status_code == 401:
        log.critical("OANDA AUTH FAILURE — stopping bot")
        raise AuthenticationError("Token invalid")

    elif response.status_code == 429:
        retry_after = int(response.headers.get("Retry-After", 5))
        log.warning(f"Rate limited. Waiting {retry_after}s")
        time.sleep(retry_after)
        return None  # Caller retries

    elif response.status_code in (500, 502, 503, 504):
        log.warning(f"OANDA server error {response.status_code}")
        return None  # Caller applies exponential backoff

    else:
        log.error(f"OANDA error {response.status_code}: {response.text}")
        raise OandaAPIError(response.status_code, response.text)
```

### Maintenance Window Detection

```python
def is_maintenance_window() -> bool:
    """
    OANDA typically takes maintenance Friday 17:00 to Sunday 17:00 ET.
    The API returns 503 or 401 during maintenance even with valid token.
    """
    now_et = datetime.now(pytz.timezone("America/New_York"))
    # Friday after 17:00 ET or Saturday any time
    if now_et.weekday() == 4 and now_et.hour >= 17:
        return True
    if now_et.weekday() == 5:
        return True
    # Sunday before 17:00 ET
    if now_et.weekday() == 6 and now_et.hour < 17:
        return True
    return False

def wait_for_maintenance_end():
    """
    Polls the OANDA pricing endpoint every 5 minutes until it responds.
    Does not attempt any trades during this period.
    """
    while True:
        try:
            response = requests.get(
                f"{BASE_URL}/v3/accounts/{ACCOUNT_ID}/summary",
                headers=HEADERS,
                timeout=10,
            )
            if response.status_code == 200:
                log.info("OANDA maintenance ended — resuming")
                return
        except Exception:
            pass
        log.info("OANDA maintenance in progress — checking again in 5 minutes")
        time.sleep(300)
```

---

## Leverage & Margin Compliance

```python
def calculate_max_units(
    instrument: str,
    account_balance: float,
    risk_fraction: float = 0.01,
    sl_distance_pips: float = 20,
    pip_value: float = 10,        # USD value per pip per standard lot
) -> int:
    """
    Calculate maximum position size respecting:
    1. Risk fraction (1% of balance)
    2. OANDA leverage limits
    3. Available margin

    Returns integer units (standard: 100,000 units = 1 lot)
    """

    # Risk-based sizing
    risk_amount = account_balance * risk_fraction
    pip_risk = sl_distance_pips * pip_value
    risk_based_lots = risk_amount / pip_risk
    risk_based_units = int(risk_based_lots * 100_000)

    # Leverage check
    max_leverage = LEVERAGE_LIMITS.get(
        get_pair_tier(instrument), 20
    )
    max_notional = account_balance * max_leverage
    max_units_leverage = int(max_notional / get_current_price(instrument))

    # Use the more conservative of the two
    return min(risk_based_units, max_units_leverage)
```

---

## Pip Value Reference

```python
PIP_LOCATIONS = {
    # 4-decimal pairs — 1 pip = 0.0001
    "EUR_USD": -4, "GBP_USD": -4, "AUD_USD": -4,
    "NZD_USD": -4, "USD_CAD": -4, "USD_CHF": -4,
    "EUR_GBP": -4, "EUR_AUD": -4, "EUR_CAD": -4,
    "GBP_CAD": -4, "GBP_AUD": -4,

    # 2-decimal pairs — 1 pip = 0.01 (JPY pairs)
    "USD_JPY": -2, "EUR_JPY": -2, "GBP_JPY": -2,
    "AUD_JPY": -2, "CAD_JPY": -2, "CHF_JPY": -2,

    # Metals — treated differently
    "XAU_USD": -2,  # Gold, pip = $0.01
    "XAG_USD": -4,  # Silver
}

def get_pip_size(instrument: str) -> float:
    location = PIP_LOCATIONS.get(instrument, -4)
    return 10 ** location
```
