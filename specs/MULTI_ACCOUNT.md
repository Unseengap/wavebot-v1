# MULTI_ACCOUNT.md — Multi-API & Multi-Account Configuration

## Overview

OandaFX is built to manage multiple OANDA accounts simultaneously from a single bot instance on the Vultr VPS. Each account is defined independently in `config/accounts.yaml` with its own API key, account ID, strategy profile, and risk parameters. Accounts can be added or removed without restarting the bot.

---

## Why Multi-Account Support Matters

- **Risk isolation:** Run a conservative strategy on one account and a more aggressive strategy on another
- **Demo + live simultaneously:** Validate new model versions on a practice account before promoting to live
- **Scaling:** As performance is proven, additional live accounts can be added with different capital bases
- **Sub-accounts:** OANDA supports multiple sub-accounts under one login — each gets its own account ID
- **Multiple API keys:** Rotate or replace compromised API keys without downtime

---

## Account Configuration File

File: `config/accounts.yaml`

```yaml
# OandaFX — Account Configuration
# Copy this file from accounts.yaml.example and fill in your credentials
# NEVER commit this file to Git — it is in .gitignore by default

accounts:

  # ─── Account 1: Primary Live Account ─────────────────────────────────────
  - alias: "primary_live"
    enabled: true
    environment: "live"                        # "live" or "practice"
    api_key: "YOUR_OANDA_API_KEY_1"            # From OANDA AMP portal
    account_id: "001-001-XXXXXXX-001"          # Your OANDA account ID
    strategy: "default"                        # References bot_config.yaml strategy name
    base_currency: "USD"                       # Your account base currency

    # Risk parameters (override global defaults for this account)
    risk:
      max_risk_per_trade: 0.01                 # 1% of account balance per trade
      max_daily_loss_pct: 0.03                 # Halt if down 3% in a day
      max_drawdown_pct: 0.08                   # Halt if down 8% from peak
      max_open_trades: 5                       # Max simultaneous positions
      max_correlated_exposure: 2               # Max trades in same currency (e.g. EUR)

    # Pair whitelist (leave empty to use global config)
    pairs:
      - EUR_USD
      - GBP_USD
      - USD_JPY
      - AUD_USD
      - USD_CHF

    # Notification settings for this account
    notifications:
      telegram_chat_id: "-100XXXXXXXXXX"       # Leave blank to use global

  # ─── Account 2: Practice / Shadow Account ────────────────────────────────
  - alias: "shadow_practice"
    enabled: true
    environment: "practice"
    api_key: "YOUR_OANDA_API_KEY_2"
    account_id: "101-001-XXXXXXX-001"
    strategy: "aggressive"
    base_currency: "USD"

    risk:
      max_risk_per_trade: 0.02
      max_daily_loss_pct: 0.05
      max_drawdown_pct: 0.15
      max_open_trades: 8
      max_correlated_exposure: 3

    pairs: []                                   # Empty = use global pair list

    notifications:
      telegram_chat_id: ""                      # Uses global default

  # ─── Account 3: Conservative Long-Term Account ───────────────────────────
  - alias: "conservative_longterm"
    enabled: false                              # Set true to activate
    environment: "live"
    api_key: "YOUR_OANDA_API_KEY_3"
    account_id: "001-001-XXXXXXX-003"
    strategy: "conservative"
    base_currency: "USD"

    risk:
      max_risk_per_trade: 0.005                # 0.5% per trade — very conservative
      max_daily_loss_pct: 0.02
      max_drawdown_pct: 0.05
      max_open_trades: 3
      max_correlated_exposure: 1

    pairs:
      - EUR_USD
      - USD_JPY
      - XAU_USD                                # Gold as portfolio hedge

    notifications:
      telegram_chat_id: ""

# ─── Global Notification Settings ─────────────────────────────────────────
notifications:
  telegram_bot_token: "YOUR_TELEGRAM_BOT_TOKEN"
  telegram_chat_id: "-100XXXXXXXXXX"           # Default chat for all accounts
  email_alerts: false
  email_address: ""

# ─── Global API Rate Limiting ──────────────────────────────────────────────
rate_limits:
  max_requests_per_second: 20                  # OANDA recommended limit
  candle_requests_per_minute: 60
  order_requests_per_minute: 30
```

---

## API Key Management

### Obtaining OANDA API Keys

1. Log into your OANDA account at [fxtrade.oanda.com](https://fxtrade.oanda.com) or [fxpractice.oanda.com](https://fxpractice.oanda.com)
2. Navigate to: **My Account → My Services → Manage API Access**
3. Click **Generate** to create a personal access token
4. Copy the token — it is only shown once

**Important:** Practice environment API keys only work with the practice API endpoint (`api-fxpractice.oanda.com`). Live environment keys only work with the live endpoint (`api-fxtrade.oanda.com`). The bot handles this automatically based on the `environment` field.

### API Key Rotation

To rotate an API key without downtime:

1. Generate a new API key in the OANDA portal
2. Update `config/accounts.yaml` with the new key
3. Run: `python scripts/reload_accounts.py --alias primary_live`
4. The bot reloads credentials for that account without stopping other accounts
5. Revoke the old key in the OANDA portal

### Security

- `config/accounts.yaml` is listed in `.gitignore` — it will never be committed to GitHub
- On the VPS, the file is readable only by the bot process user: `chmod 600 config/accounts.yaml`
- API keys can alternatively be stored as environment variables and referenced in the config:
  ```yaml
  api_key: "${OANDA_API_KEY_1}"    # Reads from environment variable
  ```
- On Vultr, use user data scripts or a secrets manager to inject environment variables at boot

---

## Environment Variables Alternative

For CI/CD pipelines or Docker deployments, API credentials can be passed entirely as environment variables instead of a config file:

```bash
# .env file (never commit this to Git)
OANDA_ACCOUNT_1_ALIAS=primary_live
OANDA_ACCOUNT_1_API_KEY=your-api-key-here
OANDA_ACCOUNT_1_ACCOUNT_ID=001-001-XXXXXXX-001
OANDA_ACCOUNT_1_ENVIRONMENT=live
OANDA_ACCOUNT_1_STRATEGY=default

OANDA_ACCOUNT_2_ALIAS=shadow_practice
OANDA_ACCOUNT_2_API_KEY=your-api-key-here
OANDA_ACCOUNT_2_ACCOUNT_ID=101-001-XXXXXXX-001
OANDA_ACCOUNT_2_ENVIRONMENT=practice
OANDA_ACCOUNT_2_STRATEGY=aggressive
```

The `AccountManager` automatically detects whether to use the YAML file or environment variables.

---

## Hot-Adding Accounts at Runtime

New accounts can be added while the bot is running — no restart required:

```bash
# Add a new account via CLI
python scripts/manage_accounts.py add \
  --alias "new_live_account" \
  --api-key "your-new-api-key" \
  --account-id "001-001-XXXXXXX-004" \
  --environment live \
  --strategy default

# List all active accounts
python scripts/manage_accounts.py list

# Pause trading on one account (keeps it registered but stops new trades)
python scripts/manage_accounts.py pause --alias "shadow_practice"

# Resume a paused account
python scripts/manage_accounts.py resume --alias "shadow_practice"

# Remove an account (closes all open trades first)
python scripts/manage_accounts.py remove --alias "shadow_practice"
```

---

## Per-Account Strategy Profiles

Each account references a strategy by name. Strategies are defined in `config/bot_config.yaml`:

```yaml
strategies:

  default:
    base_timeframe: "M15"
    higher_timeframes: ["H1", "H4", "D"]
    model_path: "data/models/latest/transformer.pt"
    rl_agent_path: "data/models/latest/ppo_agent.zip"
    min_confidence: 0.65
    pairs: ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD"]

  conservative:
    base_timeframe: "H1"                # Slower timeframe = fewer, higher-quality trades
    higher_timeframes: ["H4", "D", "W"]
    model_path: "data/models/latest/transformer.pt"
    rl_agent_path: "data/models/latest/ppo_agent.zip"
    min_confidence: 0.75               # Higher confidence threshold
    pairs: ["EUR_USD", "USD_JPY"]      # Majors only

  aggressive:
    base_timeframe: "M5"               # Faster timeframe
    higher_timeframes: ["M15", "H1", "H4"]
    model_path: "data/models/latest/transformer.pt"
    rl_agent_path: "data/models/latest/ppo_agent.zip"
    min_confidence: 0.60               # Lower threshold = more trades
    pairs: ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CHF", "EUR_JPY", "GBP_JPY"]
```

---

## Correlated Exposure Limits

To prevent all accounts from piling into the same trade, the `AccountManager` tracks total system-wide exposure per currency:

```python
def check_system_exposure(currency, direction):
    """
    Example: if EUR_USD long + EUR_GBP long + EUR_CHF long are all open,
    the system has heavy EUR long exposure.
    Refuse new EUR longs if system_exposure["EUR"] > max_system_exposure.
    """
    exposure = sum(
        trade.units for account in all_accounts
        for trade in account.open_trades
        if currency in trade.instrument and trade.direction == direction
    )
    return exposure
```

This is configurable via `max_system_currency_exposure` in `bot_config.yaml`.

---

## Account Status Dashboard

All account statuses are visible via the monitoring dashboard at `http://your-vps-ip:8080`:

| Account | Status | Balance | Today P&L | Open Trades | DD from Peak |
|---------|--------|---------|-----------|-------------|-------------|
| primary_live | ✅ Active | $10,420 | +$82 (+0.79%) | 2 | 1.2% |
| shadow_practice | ✅ Active | $50,000 | -$120 (-0.24%) | 4 | 0.8% |
| conservative_longterm | ⏸ Disabled | — | — | 0 | — |

The dashboard exposes a REST endpoint at `/api/accounts` for programmatic status queries.
