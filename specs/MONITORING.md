# MONITORING.md — Logging, Alerts & Dashboards

## Overview

OandaFX includes a comprehensive monitoring stack that provides real-time visibility into bot health, trade execution, account performance, and system errors. The stack consists of three components:

1. **Structured JSON logging** (`src/monitoring/logger.py`) — every action is logged with machine-parseable context
2. **Alerting** (`src/monitoring/alerts.py`) — Telegram and email notifications for critical events
3. **Dashboard** (`src/monitoring/dashboard.py`) — FastAPI web interface exposing metrics and account status

All monitoring runs on the Vultr VPS alongside the trading bot.

---

## Architecture

```
Trading Sessions (per account)
        │
        ├── Trade events ──────────► Structured Logger ──► JSONL files
        ├── Model signals ─────────►       │                  │
        ├── Risk rejections ───────►       │                  ▼
        └── Errors ────────────────►       │            Log rotation
                                           │            (daily, 90-day retention)
                                           │
                                           ├──────────► Alert Engine
                                           │            ├── Telegram bot
                                           │            └── Email (optional)
                                           │
                                           └──────────► FastAPI Dashboard
                                                        ├── /api/accounts
                                                        ├── /api/trades
                                                        ├── /api/performance
                                                        ├── /api/signals
                                                        └── /api/health
```

---

## Component 1: Structured JSON Logging

### Design

All log output uses structured JSON format (one JSON object per line). This makes logs machine-parseable for analysis, searchable with `jq`, and compatible with external log aggregation tools.

### Log levels

| Level | Use case |
|-------|----------|
| `DEBUG` | API request/response details, feature computation steps, model inference internals |
| `INFO` | Trade opened/closed, signal generated, data downloaded, account status |
| `WARNING` | Spread wider than expected, partial fill, approaching daily loss limit |
| `ERROR` | API failure after retries, order rejection, data gap detected |
| `CRITICAL` | Circuit breaker triggered, account connection lost, unhandled exception |

### Log format

Every log entry includes a base set of fields plus event-specific context:

```json
{
  "timestamp": "2026-04-09T14:30:02.451Z",
  "level": "INFO",
  "logger": "execution.order_manager",
  "account": "primary_live",
  "event": "trade_opened",
  "data": {
    "instrument": "EUR_USD",
    "direction": "long",
    "units": 10000,
    "entry_price": 1.08542,
    "sl_price": 1.08312,
    "tp_price": 1.08887,
    "sl_pips": 23.0,
    "tp_pips": 34.5,
    "rr_ratio": 1.5,
    "risk_pct": 0.01,
    "model_confidence": 0.73,
    "transformer_signal": "long",
    "rl_signal": "long",
    "spread_at_entry": 1.1,
    "trade_id": "12345"
  }
}
```

### Logger configuration

```python
# src/monitoring/logger.py

import json
import logging
from datetime import datetime, timezone
from pathlib import Path


class JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects."""

    def format(self, record):
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
        }

        # Add account context if present
        if hasattr(record, "account"):
            log_entry["account"] = record.account

        # Add event type and structured data
        if hasattr(record, "event"):
            log_entry["event"] = record.event
        if hasattr(record, "data"):
            log_entry["data"] = record.data

        # Fallback to message string
        if record.msg:
            log_entry["message"] = record.getMessage()

        # Include exception info
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, default=str)


def setup_logging(log_dir="data/logs", level="INFO"):
    """Configure root logger with JSON file handler and console handler."""

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # File handler — daily rotation
    from logging.handlers import TimedRotatingFileHandler

    file_handler = TimedRotatingFileHandler(
        filename=log_path / "oandafx.jsonl",
        when="midnight",
        interval=1,
        backupCount=90,        # Retain 90 days of logs
        utc=True,
    )
    file_handler.setFormatter(JSONFormatter())

    # Console handler — human-readable for live debugging
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level))
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    return root_logger
```

### Log file layout

```
data/logs/
├── oandafx.jsonl                     # Current day's logs
├── oandafx.jsonl.2026-04-08          # Yesterday
├── oandafx.jsonl.2026-04-07          # Day before
├── ...                               # Up to 90 days retained
├── trades_2026.jsonl                 # Dedicated trade log (append-only)
└── errors_2026.jsonl                 # Errors-only log for fast review
```

### Event types logged

| Event | Level | Trigger |
|-------|-------|---------|
| `bot_started` | INFO | Bot process begins |
| `bot_stopped` | INFO | Graceful shutdown |
| `account_connected` | INFO | OANDA API authentication successful |
| `account_disconnected` | ERROR | OANDA connection lost |
| `data_downloaded` | INFO | Historical candle batch fetched |
| `data_gap_detected` | WARNING | Missing bars in candle stream |
| `feature_computed` | DEBUG | Feature vector built for a bar |
| `signal_generated` | INFO | Model produces directional signal |
| `signal_flat` | DEBUG | Model outputs flat / no trade |
| `risk_approved` | DEBUG | Trade passes all risk checks |
| `risk_rejected` | WARNING | Trade blocked by risk layer |
| `trade_opened` | INFO | Order filled by OANDA |
| `trade_modified` | INFO | SL/TP updated on open position |
| `trade_closed` | INFO | Position closed (TP, SL, or manual) |
| `order_rejected` | ERROR | OANDA rejected the order |
| `circuit_breaker_triggered` | CRITICAL | Daily loss or drawdown limit hit |
| `maintenance_detected` | WARNING | OANDA API returns 503 |
| `api_error` | ERROR | API call failed after max retries |
| `reconciliation_mismatch` | ERROR | Local state differs from OANDA |

### Querying logs with jq

```bash
# All trades opened today
cat data/logs/oandafx.jsonl | jq 'select(.event == "trade_opened")'

# All errors in the last 24 hours
cat data/logs/oandafx.jsonl | jq 'select(.level == "ERROR" or .level == "CRITICAL")'

# P&L summary for a specific account
cat data/logs/oandafx.jsonl | jq 'select(.event == "trade_closed" and .data.account == "primary_live") | .data.pnl'

# Count signals by direction
cat data/logs/oandafx.jsonl | jq 'select(.event == "signal_generated") | .data.direction' | sort | uniq -c

# Risk rejections and their reasons
cat data/logs/oandafx.jsonl | jq 'select(.event == "risk_rejected") | {account: .account, reason: .data.reason}'
```

---

## Component 2: Alerting

### Telegram Bot Integration

The primary alerting channel is a Telegram bot that sends real-time notifications to a private chat or group.

#### Setup

1. Create a Telegram bot via [@BotFather](https://t.me/BotFather):
   - Send `/newbot` and follow prompts
   - Save the bot token (e.g., `7123456789:AAH...`)
2. Create a private group or channel and add the bot
3. Get the chat ID:
   ```bash
   curl https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates
   # Look for "chat": {"id": -100XXXXXXXXXX}
   ```
4. Add credentials to `.env`:
   ```bash
   TELEGRAM_BOT_TOKEN=7123456789:AAH...
   TELEGRAM_CHAT_ID=-100XXXXXXXXXX
   ```

#### Alert levels & messages

| Alert level | Events | Notification |
|-------------|--------|-------------|
| **Trade** | Trade opened, closed | Instrument, direction, entry/exit price, P&L |
| **Warning** | Risk rejection, wide spread, approaching loss limit | Reason and current account state |
| **Critical** | Circuit breaker, connection lost, unhandled error | Full context, requires manual review |
| **Daily summary** | End of trading day | Day P&L, win/loss count, open positions |
| **Weekly report** | End of trading week | Week P&L, Sharpe, drawdown, model accuracy |

#### Message formatting

```python
# src/monitoring/alerts.py

import os
import asyncio
import aiohttp
import logging

logger = logging.getLogger("monitoring.alerts")


class TelegramAlerter:
    """Sends formatted alerts to Telegram."""

    def __init__(self):
        self.token = os.environ["TELEGRAM_BOT_TOKEN"]
        self.chat_id = os.environ["TELEGRAM_CHAT_ID"]
        self.base_url = f"https://api.telegram.org/bot{self.token}"
        self._session = None

    async def _get_session(self):
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def send(self, message: str, parse_mode: str = "HTML"):
        """Send a message to Telegram. Silently logs failures."""
        try:
            session = await self._get_session()
            async with session.post(
                f"{self.base_url}/sendMessage",
                json={
                    "chat_id": self.chat_id,
                    "text": message,
                    "parse_mode": parse_mode,
                },
            ) as resp:
                if resp.status != 200:
                    logger.error(f"Telegram send failed: {resp.status}")
        except Exception as e:
            logger.error(f"Telegram send error: {e}")

    async def trade_opened(self, trade_data: dict):
        direction_emoji = "🟢" if trade_data["direction"] == "long" else "🔴"
        msg = (
            f"{direction_emoji} <b>Trade Opened</b>\n"
            f"Account: <code>{trade_data['account']}</code>\n"
            f"Pair: <b>{trade_data['instrument']}</b>\n"
            f"Direction: {trade_data['direction'].upper()}\n"
            f"Entry: {trade_data['entry_price']}\n"
            f"SL: {trade_data['sl_price']} ({trade_data['sl_pips']:.1f} pips)\n"
            f"TP: {trade_data['tp_price']} ({trade_data['tp_pips']:.1f} pips)\n"
            f"RR: {trade_data['rr_ratio']:.1f}:1\n"
            f"Risk: {trade_data['risk_pct']:.1%}\n"
            f"Confidence: {trade_data['model_confidence']:.0%}"
        )
        await self.send(msg)

    async def trade_closed(self, trade_data: dict):
        pnl = trade_data["pnl"]
        emoji = "✅" if pnl >= 0 else "❌"
        msg = (
            f"{emoji} <b>Trade Closed</b>\n"
            f"Account: <code>{trade_data['account']}</code>\n"
            f"Pair: <b>{trade_data['instrument']}</b>\n"
            f"Direction: {trade_data['direction'].upper()}\n"
            f"Entry: {trade_data['entry_price']} → Exit: {trade_data['exit_price']}\n"
            f"P&L: <b>{'+' if pnl >= 0 else ''}{pnl:.2f} USD</b>\n"
            f"Duration: {trade_data['duration_bars']} bars\n"
            f"Closed by: {trade_data['close_reason']}"
        )
        await self.send(msg)

    async def circuit_breaker(self, account: str, reason: str, details: dict):
        msg = (
            f"🚨 <b>CIRCUIT BREAKER TRIGGERED</b> 🚨\n\n"
            f"Account: <code>{account}</code>\n"
            f"Reason: <b>{reason}</b>\n"
            f"Daily P&L: {details.get('daily_pnl', 'N/A')}\n"
            f"Drawdown: {details.get('drawdown', 'N/A')}\n"
            f"Open trades: {details.get('open_trades', 'N/A')}\n\n"
            f"⚠️ Trading halted. Manual review required."
        )
        await self.send(msg)

    async def daily_summary(self, summary: dict):
        emoji = "📈" if summary["day_pnl"] >= 0 else "📉"
        msg = (
            f"{emoji} <b>Daily Summary — {summary['date']}</b>\n\n"
            f"Account: <code>{summary['account']}</code>\n"
            f"Day P&L: <b>{'+' if summary['day_pnl'] >= 0 else ''}"
            f"{summary['day_pnl']:.2f} USD ({summary['day_pnl_pct']:+.2%})</b>\n"
            f"Trades: {summary['trades_opened']} opened, "
            f"{summary['trades_closed']} closed\n"
            f"Wins: {summary['wins']} | Losses: {summary['losses']}\n"
            f"Win rate: {summary['win_rate']:.0%}\n"
            f"Open positions: {summary['open_positions']}\n"
            f"Balance: {summary['balance']:.2f} USD\n"
            f"Drawdown: {summary['drawdown']:.2%}"
        )
        await self.send(msg)

    async def weekly_report(self, report: dict):
        msg = (
            f"📊 <b>Weekly Report — {report['week_start']} to {report['week_end']}</b>\n\n"
            f"Account: <code>{report['account']}</code>\n"
            f"Week P&L: <b>{'+' if report['week_pnl'] >= 0 else ''}"
            f"{report['week_pnl']:.2f} USD ({report['week_pnl_pct']:+.2%})</b>\n"
            f"Total trades: {report['total_trades']}\n"
            f"Win rate: {report['win_rate']:.0%}\n"
            f"Profit factor: {report['profit_factor']:.2f}\n"
            f"Sharpe (week): {report['sharpe']:.2f}\n"
            f"Max drawdown: {report['max_drawdown']:.2%}\n"
            f"Model accuracy: {report['model_accuracy']:.0%}\n"
            f"Balance: {report['balance']:.2f} USD"
        )
        await self.send(msg)

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
```

### Email Alerts (Optional)

For environments where Telegram is not available, email alerting can be enabled:

```bash
# .env
EMAIL_ALERTS_ENABLED=true
EMAIL_SMTP_HOST=smtp.gmail.com
EMAIL_SMTP_PORT=587
EMAIL_USERNAME=your-email@gmail.com
EMAIL_APP_PASSWORD=xxxx-xxxx-xxxx-xxxx    # Google App Password
EMAIL_RECIPIENT=alerts@your-domain.com
```

Email alerts fire only for `CRITICAL`-level events (circuit breaker, connection loss). Trade-level notifications remain Telegram-only to avoid inbox overload.

---

## Component 3: FastAPI Dashboard

### Overview

The monitoring dashboard is a FastAPI application that runs as a separate process on port 8080 of the Vultr VPS. It exposes both a JSON API and a minimal HTML dashboard for browser-based monitoring.

### Startup

The dashboard is launched automatically by the bot process or can be started independently:

```bash
# Started by the bot
python scripts/start_bot.py --config config/accounts.yaml
# Dashboard auto-starts on port 8080

# Or run standalone (for development / debugging)
uvicorn src.monitoring.dashboard:app --host 0.0.0.0 --port 8080
```

### API Endpoints

#### `GET /api/health`

Bot health check — used by uptime monitors.

```json
{
  "status": "healthy",
  "uptime_seconds": 86421,
  "uptime_human": "1d 0h 0m",
  "accounts_active": 2,
  "accounts_paused": 0,
  "last_heartbeat": "2026-04-09T14:30:00Z",
  "version": "1.1.0",
  "model_version": "v1.1.0"
}
```

#### `GET /api/accounts`

Status of all registered accounts.

```json
{
  "accounts": [
    {
      "alias": "primary_live",
      "environment": "live",
      "status": "active",
      "balance": 12450.32,
      "unrealized_pnl": 45.20,
      "nav": 12495.52,
      "margin_used": 1250.00,
      "margin_available": 11245.52,
      "margin_utilization": 0.10,
      "open_trades": 3,
      "day_pnl": 125.40,
      "day_pnl_pct": 0.0101,
      "drawdown_from_peak": 0.022,
      "peak_balance": 12730.00,
      "pairs_active": ["EUR_USD", "GBP_USD", "USD_JPY"],
      "last_trade_time": "2026-04-09T13:15:00Z",
      "last_signal_time": "2026-04-09T14:30:00Z"
    },
    {
      "alias": "shadow_practice",
      "environment": "practice",
      "status": "active",
      "balance": 98500.00,
      "unrealized_pnl": -120.50,
      "open_trades": 5,
      "day_pnl": -85.20,
      "drawdown_from_peak": 0.035
    }
  ]
}
```

#### `GET /api/trades`

All open trades across all accounts.

```json
{
  "open_trades": [
    {
      "account": "primary_live",
      "trade_id": "12345",
      "instrument": "EUR_USD",
      "direction": "long",
      "units": 10000,
      "entry_price": 1.08542,
      "current_price": 1.08610,
      "sl_price": 1.08312,
      "tp_price": 1.08887,
      "unrealized_pnl": 6.80,
      "unrealized_pips": 6.8,
      "duration_bars": 12,
      "opened_at": "2026-04-09T11:30:00Z",
      "model_confidence": 0.73
    }
  ],
  "total_open": 3
}
```

#### `GET /api/trades/history?days=7`

Closed trade history with optional date filter.

```json
{
  "closed_trades": [
    {
      "account": "primary_live",
      "trade_id": "12340",
      "instrument": "GBP_USD",
      "direction": "short",
      "entry_price": 1.26450,
      "exit_price": 1.26210,
      "pnl": 24.00,
      "pnl_pips": 24.0,
      "rr_achieved": 1.6,
      "close_reason": "tp_hit",
      "opened_at": "2026-04-08T09:00:00Z",
      "closed_at": "2026-04-08T16:45:00Z",
      "duration_bars": 31
    }
  ],
  "total_closed": 45,
  "period_pnl": 342.50,
  "win_rate": 0.53
}
```

#### `GET /api/performance`

Aggregated performance metrics.

```json
{
  "primary_live": {
    "period": "2026-01-01 to 2026-04-09",
    "total_return": 0.245,
    "annualized_return": 0.72,
    "sharpe_ratio": 1.38,
    "sortino_ratio": 1.92,
    "max_drawdown": 0.062,
    "current_drawdown": 0.022,
    "win_rate": 0.53,
    "profit_factor": 1.45,
    "total_trades": 312,
    "avg_trade_duration_bars": 18,
    "avg_rr_achieved": 1.42,
    "expectancy_per_trade": 1.10,
    "best_trade_pnl": 145.20,
    "worst_trade_pnl": -98.50,
    "consecutive_wins_max": 8,
    "consecutive_losses_max": 4,
    "by_pair": {
      "EUR_USD": {"trades": 95, "pnl": 520.30, "win_rate": 0.55},
      "GBP_USD": {"trades": 82, "pnl": 310.20, "win_rate": 0.51},
      "USD_JPY": {"trades": 78, "pnl": 280.10, "win_rate": 0.52},
      "AUD_USD": {"trades": 57, "pnl": 120.40, "win_rate": 0.49}
    },
    "by_day_of_week": {
      "Monday": {"trades": 58, "pnl": 180.20},
      "Tuesday": {"trades": 72, "pnl": 290.30},
      "Wednesday": {"trades": 68, "pnl": 210.10},
      "Thursday": {"trades": 65, "pnl": 260.40},
      "Friday": {"trades": 49, "pnl": 90.00}
    }
  }
}
```

#### `GET /api/signals?limit=20`

Recent model signals (whether acted on or not).

```json
{
  "recent_signals": [
    {
      "timestamp": "2026-04-09T14:30:00Z",
      "instrument": "EUR_USD",
      "direction": "long",
      "transformer_confidence": 0.72,
      "rl_confidence": 0.68,
      "ensemble_confidence": 0.70,
      "action_taken": "trade_opened",
      "risk_approved": true,
      "trade_id": "12345"
    },
    {
      "timestamp": "2026-04-09T14:30:00Z",
      "instrument": "GBP_USD",
      "direction": "short",
      "transformer_confidence": 0.58,
      "rl_confidence": 0.61,
      "ensemble_confidence": 0.59,
      "action_taken": "no_trade",
      "risk_approved": false,
      "rejection_reason": "confidence_below_threshold"
    }
  ]
}
```

#### `POST /api/pause/{alias}`

Pause trading on a specific account. Open trades remain with existing SL/TP.

```json
{ "status": "paused", "alias": "primary_live", "open_trades_preserved": 3 }
```

#### `POST /api/resume/{alias}`

Resume a paused account.

```json
{ "status": "active", "alias": "primary_live" }
```

### Authentication

The dashboard requires basic authentication when deployed. Credentials are set in `.env`:

```bash
DASHBOARD_USERNAME=admin
DASHBOARD_PASSWORD=your-secure-password-here
```

```python
# src/monitoring/dashboard.py (authentication middleware)

import secrets
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

security = HTTPBasic()

def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    expected_user = os.environ.get("DASHBOARD_USERNAME", "admin")
    expected_pass = os.environ.get("DASHBOARD_PASSWORD", "")

    if not expected_pass:
        return  # Auth disabled if no password set

    user_ok = secrets.compare_digest(credentials.username.encode(), expected_user.encode())
    pass_ok = secrets.compare_digest(credentials.password.encode(), expected_pass.encode())

    if not (user_ok and pass_ok):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
```

### HTML Dashboard

The root endpoint (`GET /`) serves a minimal HTML dashboard with auto-refreshing status panels:

```
┌─────────────────────────────────────────────────────────────────┐
│  OandaFX Dashboard                          v1.1.0 | Uptime 1d │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─── primary_live ──────┐  ┌─── shadow_practice ────────────┐ │
│  │ Status: ● Active      │  │ Status: ● Active               │ │
│  │ Balance: $12,450.32   │  │ Balance: $98,500.00            │ │
│  │ Day P&L: +$125.40     │  │ Day P&L: -$85.20              │ │
│  │ Open: 3 trades        │  │ Open: 5 trades                │ │
│  │ DD: 2.2%              │  │ DD: 3.5%                      │ │
│  └────────────────────── ┘  └────────────────────────────────┘ │
│                                                                 │
│  Open Trades                                                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ EUR_USD  LONG   +6.8 pips   SL: 1.08312  TP: 1.08887   │  │
│  │ GBP_USD  SHORT  -2.1 pips   SL: 1.26650  TP: 1.26100   │  │
│  │ USD_JPY  LONG   +4.2 pips   SL: 149.120  TP: 149.680   │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  Recent Signals                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 14:30  EUR_USD  LONG   conf=0.70  → trade_opened        │  │
│  │ 14:30  GBP_USD  SHORT  conf=0.59  → no_trade (low conf) │  │
│  │ 14:15  AUD_USD  FLAT   conf=0.42  → no_trade            │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  Auto-refresh: 30s                          [Pause] [Resume]   │
└─────────────────────────────────────────────────────────────────┘
```

The HTML page uses vanilla JavaScript to poll `/api/accounts`, `/api/trades`, and `/api/signals` every 30 seconds.

---

## Alert Configuration

### Per-account notification overrides

Each account can designate its own Telegram chat ID in `config/accounts.yaml`:

```yaml
accounts:
  - alias: "primary_live"
    notifications:
      telegram_chat_id: "-100XXXXXXXXXX"    # Dedicated channel for live alerts
  - alias: "shadow_practice"
    notifications:
      telegram_chat_id: ""                  # Uses global default
```

### Alert throttling

To prevent notification fatigue, alerts are throttled:

| Alert type | Cooldown | Max per hour |
|------------|----------|-------------|
| Trade opened | None | Unlimited |
| Trade closed | None | Unlimited |
| Risk rejection | 5 minutes per pair | 12 |
| Wide spread warning | 15 minutes per pair | 4 |
| API error | 5 minutes | 12 |
| Circuit breaker | None (always fires) | Unlimited |
| Daily summary | Once per day | 1 |
| Weekly report | Once per week | 1 |

```python
# Throttle implementation
from collections import defaultdict
from datetime import datetime, timedelta

class AlertThrottle:
    def __init__(self):
        self._last_sent = defaultdict(lambda: datetime.min)

    def should_send(self, alert_key: str, cooldown_seconds: int) -> bool:
        now = datetime.utcnow()
        if now - self._last_sent[alert_key] >= timedelta(seconds=cooldown_seconds):
            self._last_sent[alert_key] = now
            return True
        return False
```

---

## Scheduled Reports

### Daily summary (sent at 22:00 UTC each weekday)

Generated by a cron job that queries the day's trade log:

```bash
# crontab entry
0 22 * * 1-5 /home/botuser/oandafx/venv/bin/python /home/botuser/oandafx/scripts/send_daily_summary.py
```

Content:
- Total P&L for the day (USD and %)
- Number of trades opened and closed
- Win rate and average RR
- Current open positions
- Account balance and drawdown
- Any risk rejections or errors

### Weekly report (sent Sunday at 06:00 UTC)

```bash
# crontab entry
0 6 * * 0 /home/botuser/oandafx/venv/bin/python /home/botuser/oandafx/scripts/send_weekly_report.py
```

Content:
- Weekly P&L breakdown by day
- Sharpe ratio and profit factor for the week
- Model signal accuracy (predicted direction vs. actual outcome)
- Best and worst trades
- Per-pair performance breakdown
- Drawdown analysis
- Comparison against previous week

---

## Log Analysis & Maintenance

### Disk space management

With structured JSON logging, logs grow at approximately:

| Trading activity | Log size per day | Per month |
|-----------------|------------------|-----------|
| Light (1–3 trades/day) | ~5 MB | ~150 MB |
| Moderate (5–10 trades/day) | ~15 MB | ~450 MB |
| Heavy (20+ trades/day) | ~40 MB | ~1.2 GB |

The `TimedRotatingFileHandler` automatically rotates and retains logs for 90 days. Old logs are deleted automatically.

### Log compression (optional)

To reduce disk usage, compress rotated logs:

```bash
# crontab — compress logs older than 1 day
0 1 * * * find /home/botuser/oandafx/data/logs/ -name "*.jsonl.2*" -mtime +1 ! -name "*.gz" -exec gzip {} \;
```

### Quick analysis commands

```bash
# Today's P&L across all accounts
cat data/logs/oandafx.jsonl | jq -r 'select(.event=="trade_closed") | .data.pnl' | awk '{sum+=$1} END {printf "Total P&L: $%.2f\n", sum}'

# Trades per pair this week
cat data/logs/oandafx.jsonl* | jq -r 'select(.event=="trade_opened") | .data.instrument' | sort | uniq -c | sort -rn

# Average model confidence on winning trades
cat data/logs/oandafx.jsonl | jq 'select(.event=="trade_closed" and .data.pnl > 0) | .data.model_confidence' | awk '{sum+=$1; n++} END {printf "Avg confidence (wins): %.2f\n", sum/n}'

# Circuit breaker activations this month
cat data/logs/oandafx.jsonl* | jq 'select(.event=="circuit_breaker_triggered")' | wc -l

# Hourly trade distribution
cat data/logs/oandafx.jsonl* | jq -r 'select(.event=="trade_opened") | .timestamp[:13]' | cut -c12-13 | sort | uniq -c

# Risk rejection reasons
cat data/logs/oandafx.jsonl | jq -r 'select(.event=="risk_rejected") | .data.reason' | sort | uniq -c | sort -rn
```

---

## External Monitoring Integration

### Uptime monitoring

Use an external uptime service (e.g., UptimeRobot, Healthchecks.io) to ping the health endpoint:

```
URL: http://YOUR_VPS_IP:8080/api/health
Method: GET
Interval: 5 minutes
Expected response: 200 OK with "status": "healthy"
Alert on: non-200 response or timeout > 10 seconds
```

### Healthchecks.io heartbeat

The bot can send periodic heartbeats to [Healthchecks.io](https://healthchecks.io/) for dead-man-switch monitoring. If the heartbeat stops, you receive an alert:

```python
import aiohttp

HEALTHCHECK_URL = os.environ.get("HEALTHCHECK_PING_URL", "")

async def send_heartbeat():
    """Called every 5 minutes by the main loop."""
    if HEALTHCHECK_URL:
        async with aiohttp.ClientSession() as session:
            await session.get(HEALTHCHECK_URL)
```

```bash
# .env
HEALTHCHECK_PING_URL=https://hc-ping.com/your-uuid-here
```

### Prometheus metrics (advanced)

For integration with Prometheus + Grafana, the dashboard can expose a `/metrics` endpoint:

```
# HELP oandafx_balance_usd Current account balance
# TYPE oandafx_balance_usd gauge
oandafx_balance_usd{account="primary_live"} 12450.32
oandafx_balance_usd{account="shadow_practice"} 98500.00

# HELP oandafx_open_trades Number of open trades
# TYPE oandafx_open_trades gauge
oandafx_open_trades{account="primary_live"} 3

# HELP oandafx_day_pnl_usd P&L for current trading day
# TYPE oandafx_day_pnl_usd gauge
oandafx_day_pnl_usd{account="primary_live"} 125.40

# HELP oandafx_drawdown_ratio Current drawdown from peak
# TYPE oandafx_drawdown_ratio gauge
oandafx_drawdown_ratio{account="primary_live"} 0.022

# HELP oandafx_trades_total Total trades executed
# TYPE oandafx_trades_total counter
oandafx_trades_total{account="primary_live",direction="long"} 162
oandafx_trades_total{account="primary_live",direction="short"} 150

# HELP oandafx_model_signals_total Model signals generated
# TYPE oandafx_model_signals_total counter
oandafx_model_signals_total{direction="long"} 450
oandafx_model_signals_total{direction="short"} 420
oandafx_model_signals_total{direction="flat"} 1200

# HELP oandafx_api_errors_total OANDA API errors
# TYPE oandafx_api_errors_total counter
oandafx_api_errors_total{type="timeout"} 12
oandafx_api_errors_total{type="503"} 8
oandafx_api_errors_total{type="429"} 2
```

To enable Prometheus metrics:

```bash
pip install prometheus-fastapi-instrumentator
```

```python
# In dashboard.py
from prometheus_fastapi_instrumentator import Instrumentator
Instrumentator().instrument(app).expose(app, endpoint="/metrics")
```

---

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| No Telegram alerts | Bad bot token or chat ID | Verify with `curl https://api.telegram.org/bot<TOKEN>/getMe` |
| Dashboard not accessible | Firewall blocking port 8080 | Run `sudo ufw allow 8080/tcp` |
| Dashboard returns 401 | Wrong credentials | Check `DASHBOARD_USERNAME` and `DASHBOARD_PASSWORD` in `.env` |
| Logs not rotating | File permissions | Ensure `botuser` owns `data/logs/` directory |
| Disk full from logs | No compression or retention | Add gzip cron job, verify `backupCount=90` |
| Stale dashboard data | Bot process crashed | Check `sudo systemctl status oandafx` |
| Duplicate alerts | Throttle not working | Verify `AlertThrottle` is shared across account sessions |
| Email alerts not sending | SMTP auth failure | Use Google App Password, not account password |
| `/api/health` reports unhealthy | OANDA API down or maintenance | Check `maintenance_detected` events in logs |
| Metrics show 0 trades | Bot started recently | Wait for first signal at next bar close |
