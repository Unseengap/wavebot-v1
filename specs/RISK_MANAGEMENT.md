# RISK_MANAGEMENT.md — Stop Loss, Take Profit, Position Sizing & Drawdown

## Philosophy

Risk management in WaveBot is not a safety net bolted on after signal generation. It is built from the wave architecture itself:

- **Stop losses** are placed at wave origins — the point where the wave thesis is structurally invalidated
- **Take profits** are projected from historical wave amplitude data — the point where similar waves have historically ended
- **Position size** is derived from the distance to the stop loss — risk is constant, size is variable
- **Drawdown halts** are absolute — no override, no exceptions

---

## Stop Loss Placement

### Wave Origin Rule

The stop loss is always placed at the origin of the signal wave — the swing low that preceded the current bullish impulse (for longs), or the swing high that preceded the current bearish impulse (for shorts).

```python
def calculate_stop_loss(
    direction: str,
    signal_wave_origin: float,   # Price of wave origin (from WaveScore)
    pip_size: float,
    buffer_pips: float = 2.0,    # Extra buffer beyond origin
) -> float:
    """
    LONG trade:
        SL = wave_origin - buffer
        (Stop below the swing low that started the bullish wave)

    SHORT trade:
        SL = wave_origin + buffer
        (Stop above the swing high that started the bearish wave)

    If price breaches the wave origin, the wave thesis is wrong.
    The wave that generated the signal no longer exists.
    There is no reason to stay in the trade.
    """
    buffer = buffer_pips * pip_size

    if direction == "LONG":
        return round(signal_wave_origin - buffer, 5)
    else:  # SHORT
        return round(signal_wave_origin + buffer, 5)
```

### Why Wave Origin, Not ATR

ATR (Average True Range) multiples are common in algorithmic trading. WaveBot does not use them for primary SL placement for three reasons:

1. ATR is an average — it does not reflect the actual structural level that would invalidate the trade
2. ATR-based stops often sit in "noise" — close enough to current price that they get clipped by normal volatility
3. Wave origin stops are self-consistent with the signal — if the wave is invalid, the stop fires. If the wave is valid, the stop is rarely tested.

ATR is only used as a sanity check: if the wave origin SL would require more than 3× ATR in distance, the setup is rejected (the wave is too large for the current account size to risk efficiently).

```python
def validate_sl_distance(
    sl_distance_pips: float,
    atr_pips: float,
    max_atr_multiple: float = 3.0,
) -> bool:
    if sl_distance_pips > atr_pips * max_atr_multiple:
        log.warning(f"SL distance {sl_distance_pips:.1f}p > {max_atr_multiple}x ATR {atr_pips:.1f}p — rejecting setup")
        return False
    return True
```

---

## Take Profit Placement

### Wave Amplitude Projection

The take profit is placed at the distance from the entry equal to the 75th percentile of historical wave amplitudes on the confirmation timeframe (H1 or H4).

```python
def calculate_take_profit(
    direction: str,
    entry_price: float,
    amplitude_stats: dict,          # From AmplitudeTracker
    confirmation_tf: str,           # "H1" or "H4"
    tp_percentile: str = "p75",     # Conservative. Options: "p50", "p75", "p90"
    pip_size: float = 0.0001,
    nearby_zones: list = None,      # Structural resistance/support levels
) -> float:
    """
    Default: Use 75th percentile amplitude of the confirmation timeframe.
    This means 75% of historical waves from this TF have traveled at least
    this far. It is achievable without being greedy.

    Structural level adjustment:
    If a known resistance/support zone falls between entry and TP,
    the TP is pulled back to just before that zone rather than
    projecting through it.
    """
    target_pips = amplitude_stats.get(tp_percentile, 30)
    target_distance = target_pips * pip_size

    if direction == "LONG":
        raw_tp = entry_price + target_distance

        # Check for resistance zones between entry and TP
        if nearby_zones:
            for zone in nearby_zones:
                if zone["type"] == "RESISTANCE":
                    if entry_price < zone["price"] < raw_tp:
                        # Pull TP back to just below the zone
                        return round(zone["price"] - (2 * pip_size), 5)

        return round(raw_tp, 5)

    else:  # SHORT
        raw_tp = entry_price - target_distance

        if nearby_zones:
            for zone in nearby_zones:
                if zone["type"] == "SUPPORT":
                    if raw_tp < zone["price"] < entry_price:
                        return round(zone["price"] + (2 * pip_size), 5)

        return round(raw_tp, 5)
```

### Minimum R:R Enforcement

No trade is entered unless the calculated TP gives at least 1:2 risk:reward.

```python
def calculate_rr(
    entry: float,
    stop_loss: float,
    take_profit: float,
    direction: str,
) -> float:
    risk = abs(entry - stop_loss)
    reward = abs(take_profit - entry)

    if risk == 0:
        return 0.0

    return round(reward / risk, 2)

# Minimum: 2.0 (1:2 R:R)
# If TP would need to be closer than 2× the risk distance: SKIP setup
```

---

## Position Sizing

### Fixed Fraction Method (Default)

Risk a fixed percentage of the current account balance on every trade. As the account grows, position sizes grow proportionally. As the account shrinks, position sizes shrink protectively.

```python
def calculate_position_size(
    account_balance: float,
    risk_fraction: float,        # Default: 0.01 (1%)
    sl_distance_pips: float,
    pip_value_per_unit: float,   # From OANDA instrument details
) -> int:
    """
    Risk amount = balance × risk_fraction
    Units = risk_amount / (sl_distance_pips × pip_value_per_unit)

    Returns integer units. OANDA accepts fractional lots via units.

    Example:
      Balance: $10,000
      Risk fraction: 1% = $100 risk
      SL distance: 20 pips
      EUR/USD pip value: $0.0001 per unit
      Units = 100 / (20 × 0.0001) = 100 / 0.002 = 50,000 units (0.5 lots)
    """
    risk_amount = account_balance * risk_fraction
    risk_per_unit = sl_distance_pips * pip_value_per_unit

    if risk_per_unit == 0:
        return 0

    units = int(risk_amount / risk_per_unit)

    # Minimum trade size — OANDA minimum is 1 unit
    # Practical minimum for meaningful trades
    MIN_UNITS = 1000  # 0.01 mini lot
    return max(MIN_UNITS, units)
```

### Getting Pip Value from OANDA

```python
def get_pip_value_per_unit(instrument: str, account_currency: str = "USD") -> float:
    """
    For USD-quoted pairs (EUR/USD, GBP/USD, AUD/USD):
        pip_value = pip_size (e.g. 0.0001)
        per unit

    For USD-base pairs (USD/JPY, USD/CAD):
        pip_value = pip_size / current_price

    For cross pairs (EUR/GBP):
        pip_value = pip_size × GBP/USD rate (quote to USD conversion)

    OANDA's instrument details endpoint provides the pip location.
    Use: GET /v3/instruments/{instrument}
    """
```

---

## Circuit Breakers

Circuit breakers are non-negotiable. They cannot be disabled, overridden, or bypassed by any configuration.

```python
class CircuitBreaker:

    def __init__(self, config: dict):
        self.max_daily_drawdown = config["max_daily_drawdown"]   # 0.02 (2%)
        self.max_total_drawdown = config["max_total_drawdown"]   # 0.08 (8%)
        self.max_open_trades = config["max_open_trades"]         # 3
        self.daily_start_balance = None
        self.peak_balance = None
        self.trading_halted_today = False

    def check(self, account: dict) -> tuple[bool, str]:
        """
        Returns (trading_allowed: bool, reason: str)
        Called before every potential entry.
        """
        balance = float(account["balance"])

        # Initialize tracking
        if self.daily_start_balance is None:
            self.daily_start_balance = balance
        if self.peak_balance is None:
            self.peak_balance = balance
        self.peak_balance = max(self.peak_balance, balance)

        # Daily drawdown check
        daily_dd = (self.daily_start_balance - balance) / self.daily_start_balance
        if daily_dd >= self.max_daily_drawdown:
            self.trading_halted_today = True
            return False, f"DAILY DRAWDOWN {daily_dd*100:.1f}% >= {self.max_daily_drawdown*100:.0f}% LIMIT"

        # Total drawdown from peak check
        total_dd = (self.peak_balance - balance) / self.peak_balance
        if total_dd >= self.max_total_drawdown:
            return False, f"TOTAL DRAWDOWN {total_dd*100:.1f}% >= {self.max_total_drawdown*100:.0f}% LIMIT — MANUAL REVIEW REQUIRED"

        # Same-day halt persists until midnight
        if self.trading_halted_today:
            return False, "TRADING HALTED FOR TODAY — daily drawdown limit was hit"

        # Max open trades
        open_count = int(account.get("openTradeCount", 0))
        if open_count >= self.max_open_trades:
            return False, f"MAX OPEN TRADES ({self.max_open_trades}) REACHED"

        return True, "OK"

    def reset_daily(self):
        """Called at midnight ET to reset daily tracking."""
        self.daily_start_balance = None
        self.trading_halted_today = False
```

### On Total Drawdown Breach

When total drawdown from peak exceeds 8%, the bot:
1. Closes all open trades immediately (emergency close)
2. Stops all trading (no new orders)
3. Sends alert (log + optional notification)
4. **Does not restart automatically** — requires manual review and restart

This is a hard stop. A system that has lost 8% from its peak has a problem that automation cannot fix. Human review is required.

---

## Spread Filter

```python
MAX_SPREAD_PIPS = {
    "EUR_USD": 1.5,
    "GBP_USD": 2.0,
    "USD_JPY": 1.5,
    "USD_CAD": 2.0,
    "AUD_USD": 1.8,
    "USD_CHF": 2.0,
    "NZD_USD": 2.5,
    "EUR_GBP": 2.5,
    "GBP_JPY": 3.5,
    "EUR_JPY": 2.5,
    "XAU_USD": 30.0,   # Gold spread in pips (pip = $0.01)
}

def is_spread_acceptable(instrument: str, current_spread_pips: float) -> bool:
    max_allowed = MAX_SPREAD_PIPS.get(instrument, 3.0)
    return current_spread_pips <= max_allowed
```

The spread filter prevents entries during news events when spreads widen dramatically. A 5-pip spread on EUR/USD instantly makes a 20-pip stop loss setup unprofitable at 1:2 R:R — the spread alone is 25% of the risk.

---

## Trade Lifecycle Management

```python
class TradeLifecycle:
    """
    Manages an open trade from entry to close.
    Monitors for wave reversal signals that suggest early exit.
    """

    def on_candle_close(self, trade: dict, wave_scores: dict):
        """Called on every candle close while a trade is open."""

        # Check for wave reversal on signal frame
        signal_tf = trade["signal_frame"]
        ws = wave_scores.get(signal_tf)

        if ws:
            if trade["direction"] == "LONG":
                if ws.state == "BEARISH_IMPULSE" and ws.conviction > 0.6:
                    log.warning(f"Wave reversal on {signal_tf} — considering early close")
                    self._consider_early_close(trade, reason="WAVE_REVERSAL")

            elif trade["direction"] == "SHORT":
                if ws.state == "BULLISH_IMPULSE" and ws.conviction > 0.6:
                    self._consider_early_close(trade, reason="WAVE_REVERSAL")

    def _consider_early_close(self, trade: dict, reason: str):
        """
        Only closes early if:
        1. Trade is in profit (positive unrealizedPL)
        2. The reversal signal has minimum conviction 0.6

        Never close at a loss due to an early signal — let the SL do its job.
        """
        unrealized_pl = float(trade.get("unrealizedPL", 0))
        if unrealized_pl > 0:
            log.info(f"Early close: {trade['instrument']} +{unrealized_pl:.2f} — {reason}")
            self.order_manager.close_trade(trade["id"])
```

---

## Risk Summary Table

| Parameter | Value | Notes |
|---|---|---|
| Risk per trade | 1% of balance | Fixed fraction |
| Min R:R ratio | 1:2 | Hard minimum — no exceptions |
| Max open trades | 3 | Across all pairs |
| Max per pair | 1 | No pyramiding |
| Daily drawdown halt | 2% | Resets midnight ET |
| Total drawdown halt | 8% | Manual restart required |
| Max SL distance | 3× ATR | Sanity check on wave origin SL |
| Spread filter | Pair-specific | See MAX_SPREAD_PIPS |
| Leverage cap (majors) | 50:1 | OANDA/NFA regulation |
| Leverage cap (minors) | 20:1 | OANDA/NFA regulation |
