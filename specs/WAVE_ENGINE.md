# WAVE_ENGINE.md — Wave Detection Algorithm Specification

## Core Concept

The wave engine answers one question at every closed candle:

> "What is the current wave doing on this timeframe, and how confident are we?"

It does not predict. It does not use indicators. It reads the geometry of price directly — where did the last significant peak form, where did the last significant trough form, and where is price relative to those structural points right now.

---

## The Problem of Real-Time Wave Detection

Waves are trivially clear in hindsight. A peak is obvious when price has moved 50 pips below it. In real time, you do not know if the current high is a peak or just a pause before continuation.

This is the core engineering challenge the swing detector must solve. We solve it with **confirmation lag** — a deliberate trade-off between signal speed and signal accuracy.

**Confirmation rule:** A swing high is confirmed when price closes `N` candles below the candidate high without making a new high. `N` is the swing lookback parameter, configurable per timeframe.

```python
SWING_LOOKBACK = {
    "M1":  3,    # 3 candles each side (3 mins)
    "M5":  3,    # 3 candles each side (15 mins)
    "M15": 3,    # 3 candles each side (45 mins)
    "H1":  3,    # 3 candles each side (3 hours)
    "H4":  3,    # 3 candles each side (12 hours)
    "D":   3,    # 3 candles each side (3 days)
    "W":   2,    # 2 candles each side (2 weeks)
}
```

This means every swing signal is slightly lagged — you will never catch the exact peak or trough. This is intentional and correct. A bot that tries to catch the exact turn will have catastrophic false positives. A bot that waits for confirmation has lag but reliability.

---

## Swing Detector — Algorithm

### Swing High Detection

```python
def detect_swing_high(highs: np.ndarray, lookback: int = 3) -> np.ndarray:
    """
    A swing high at index i is confirmed when:
    - highs[i] is the highest value in the window [i-lookback : i+lookback+1]
    - If tied with another candle in the window, the EARLIEST occurrence wins

    The first-occurrence tiebreaker resolves a known gap: on lower timeframes
    (M1, M5) with tight-spread major pairs, equal highs are common. The original
    strict-inequality rule (np.sum(window == highs[i]) == 1) meant neither candle
    registered as a swing, creating silent gaps in wave detection.

    First-occurrence is the correct tiebreaker because the market established
    that level first — later touches are retests, not new structure.

    Returns boolean array. True = confirmed swing high at that index.
    Note: The last `lookback` candles cannot be confirmed yet (lag).
    """
    n = len(highs)
    is_swing_high = np.zeros(n, dtype=bool)

    for i in range(lookback, n - lookback):
        window = highs[i - lookback : i + lookback + 1]
        if highs[i] == np.max(window):
            # First-occurrence tiebreaker: if multiple candles share the max,
            # only the earliest index in the window qualifies
            first_max_idx = i - lookback + np.argmax(window)
            if first_max_idx == i:
                is_swing_high[i] = True

    return is_swing_high
```

### Swing Low Detection

```python
def detect_swing_low(lows: np.ndarray, lookback: int = 3) -> np.ndarray:
    """
    Symmetric to swing high. A swing low at index i when:
    - lows[i] is the lowest value in the window [i-lookback : i+lookback+1]
    - If tied, the EARLIEST occurrence wins (first-occurrence tiebreaker)

    Same rationale: the market established the support level first,
    later touches are retests.
    """
    n = len(lows)
    is_swing_low = np.zeros(n, dtype=bool)

    for i in range(lookback, n - lookback):
        window = lows[i - lookback : i + lookback + 1]
        if lows[i] == np.min(window):
            # First-occurrence tiebreaker for equal lows
            first_min_idx = i - lookback + np.argmin(window)
            if first_min_idx == i:
                is_swing_low[i] = True

    return is_swing_low
```

---

## Wave State Machine

Each timeframe maintains a wave state machine. The state machine has four states:

```
BULLISH_IMPULSE     — Price making higher highs and higher lows
BEARISH_IMPULSE     — Price making lower lows and lower highs
BULLISH_CORRECTION  — Pullback within a bullish impulse (temporary counter-move)
BEARISH_CORRECTION  — Pullback within a bearish impulse (temporary counter-move)
RANGING             — No clear impulse structure (avoid trading)
```

### State Transition Rules

```python
class WaveStateMachine:
    """
    Maintains wave state for a single instrument + timeframe.
    Updates on every closed candle.
    """

    def update(self, new_swing_high: float | None, new_swing_low: float | None):
        """
        Transition logic:

        BULLISH_IMPULSE:
          → If new swing low > last swing low: remain BULLISH_IMPULSE (HH, HL structure)
          → If new swing low < last swing low: → RANGING (structure broken)
          → If no new swings forming: → BULLISH_CORRECTION (pullback developing)

        BEARISH_IMPULSE:
          → If new swing high < last swing high: remain BEARISH_IMPULSE (LL, LH structure)
          → If new swing high > last swing high: → RANGING (structure broken)
          → If no new swings forming: → BEARISH_CORRECTION (pullback developing)

        BULLISH_CORRECTION:
          → If price makes new swing high above last impulse high: → BULLISH_IMPULSE
          → If price breaks below last impulse low: → BEARISH_IMPULSE (reversal)
          → After N candles without resolution: → RANGING

        BEARISH_CORRECTION:
          → Symmetric to BULLISH_CORRECTION

        RANGING:
          → If 2 consecutive higher lows AND 1 higher high form: → BULLISH_IMPULSE
          → If 2 consecutive lower highs AND 1 lower low form: → BEARISH_IMPULSE
          → After max_ranging_candles without resolution: reset swing baseline and restart detection
        """

# --- RANGING State Exit Rules (Explicit) ---
#
# The RANGING state is the most dangerous for a trading bot — it can go
# permanently dormant in choppy markets if exit criteria are vague.
# These rules are precise and non-negotiable.
#
# Exit to BULLISH_IMPULSE requires ALL of:
#   1. At least 2 confirmed swing lows where each is higher than the previous
#   2. At least 1 confirmed swing high that is higher than the swing high
#      that preceded or caused the RANGING state
#   3. The higher high must occur AFTER the 2 higher lows (sequence matters)
#
# Exit to BEARISH_IMPULSE requires ALL of:
#   1. At least 2 confirmed swing highs where each is lower than the previous
#   2. At least 1 confirmed swing low that is lower than the swing low
#      that preceded or caused the RANGING state
#   3. The lower low must occur AFTER the 2 lower highs (sequence matters)
#
# Forced reset after max_ranging_candles:
#   If neither exit condition is met within max_ranging_candles, the state machine
#   resets its swing baseline (clears the reference highs/lows that define the
#   range) and begins fresh detection. This prevents permanent dormancy.
#
# max_ranging_candles is timeframe-dependent and is part of the walk-forward
# parameter space:

RANGING_EXIT_CONFIG = {
    "min_higher_lows_for_bull":  2,      # Consecutive higher lows required
    "min_higher_highs_for_bull": 1,      # Higher high required after the HLs
    "min_lower_highs_for_bear":  2,      # Consecutive lower highs required
    "min_lower_lows_for_bear":   1,      # Lower low required after the LHs
    "max_ranging_candles": {
        "M1":  75,    # 75 minutes — reset if no structure after ~1 hour
        "M5":  50,    # 250 minutes (~4 hours)
        "M15": 40,    # 600 minutes (~10 hours)
        "H1":  30,    # 30 hours (~1.5 trading days)
        "H4":  20,    # 80 hours (~3.5 trading days)
        "D":   15,    # 15 trading days (~3 weeks)
        "W":   10,    # 10 weeks (~2.5 months)
    },
}
```

---

## Wave Scorer — Output Format

The wave scorer converts wave state into a numerical signal used by the confluence engine.

```python
@dataclass
class WaveScore:
    instrument: str
    granularity: str
    timestamp: datetime

    # Direction: +1.0 fully bullish, -1.0 fully bearish, 0.0 neutral
    direction: float

    # Conviction: 0.0 (uncertain) to 1.0 (maximum confidence)
    conviction: float

    # Maturity: 0.0 (fresh wave just started) to 1.0 (wave likely exhausted)
    maturity: float

    # State label for logging and debugging
    state: str  # "BULLISH_IMPULSE", "BEARISH_IMPULSE", etc.

    # Current wave origin price (used for stop loss placement)
    wave_origin: float

    # Number of candles since wave origin formed
    wave_age_candles: int

    # Current wave distance in pips from origin to current price
    wave_pips: float
```

### Direction Calculation

```python
def calculate_direction(state: str) -> float:
    return {
        "BULLISH_IMPULSE":    1.0,
        "BULLISH_CORRECTION": 0.5,   # Still bullish but with less force
        "RANGING":            0.0,
        "BEARISH_CORRECTION": -0.5,
        "BEARISH_IMPULSE":    -1.0,
    }[state]
```

### Conviction Calculation

Conviction is highest at the start of an impulse (fresh structure) and declines as the wave matures. It also reflects how cleanly the swing structure is defined.

```python
def calculate_conviction(
    state: str,
    maturity: float,          # 0.0–1.0
    swing_clarity: float,     # How clean were the swing points? 0.0–1.0
    consecutive_swings: int,  # How many swings confirm the trend?
) -> float:
    if state in ("RANGING",):
        return 0.0

    base = 0.5

    # Fresh impulse waves have higher conviction
    maturity_factor = 1.0 - (maturity * 0.4)  # Max 40% reduction at full maturity

    # Clean swings = more reliable structure
    clarity_factor = 0.7 + (swing_clarity * 0.3)

    # More consecutive confirming swings = more reliable trend
    swing_factor = min(1.0, 0.6 + (consecutive_swings * 0.1))

    conviction = base * maturity_factor * clarity_factor * swing_factor
    return round(min(1.0, conviction), 4)
```

### Maturity Calculation

Wave maturity estimates what percentage of the typical wave has been traveled, using **both price distance and time elapsed**. This is a critical entry filter — entries are rejected above 0.75 maturity to avoid catching exhausted waves.

Maturity is defined as the **maximum** of price maturity and time maturity. A wave is considered mature if **either** price or time says it is extended — this catches both "the wave has gone far enough" and "the wave has been running too long."

```python
def calculate_maturity(
    current_wave_pips: float,
    wave_age_candles: int,
    amplitude_stats: dict,         # From AmplitudeTracker (per instrument + TF)
    duration_stats: dict,          # From AmplitudeTracker (candle count stats)
) -> float:
    """
    Maturity = max(price_maturity, time_maturity)

    Price maturity:
      Based on median (p50) historical amplitude for this instrument + timeframe.
      Using median avoids skew from outlier moves (news spikes, flash crashes).

      At 0 pips traveled:   price_maturity = 0.0  (fresh)
      At p50 amplitude:     price_maturity = 0.6  (normal territory)
      At p75 amplitude:     price_maturity = 0.8  (extended)
      At p100 (max) or beyond: price_maturity = 1.0  (exhausted)

    Time maturity:
      Based on median historical wave duration (candle count) for this TF.
      A wave that has been running longer than typical is mature regardless
      of how far it has traveled in price.

      At 0 candles:         time_maturity = 0.0
      At median duration:   time_maturity = 0.6
      At 1.5× median:      time_maturity = 1.0

    Taking the max ensures either condition triggers maturity:
    - A fast spike that travels far in few candles → price_maturity catches it
    - A slow grind that takes many candles but little distance → time_maturity catches it
    """
    median_amplitude = amplitude_stats.get("p50", 0)
    median_duration = duration_stats.get("p50_candles", 0)

    # Price maturity
    if median_amplitude > 0:
        price_maturity = min(1.0, (current_wave_pips / median_amplitude) * 0.6)
    else:
        price_maturity = 0.5  # Unknown history — assume mid-maturity

    # Time maturity
    if median_duration > 0:
        time_maturity = min(1.0, (wave_age_candles / median_duration) * 0.6)
    else:
        time_maturity = 0.5  # Unknown history — assume mid-maturity

    return round(max(price_maturity, time_maturity), 4)
```

### AmplitudeTracker Duration Stats

The AmplitudeTracker must collect not just wave amplitude (pips) but also wave duration (candle count) for every completed wave. Both are required for the dual-axis maturity calculation.

```python
@dataclass
class WaveAmplitudeRecord:
    instrument: str
    granularity: str
    direction: str          # "BULLISH" or "BEARISH"
    amplitude_pips: float   # Distance from origin to peak
    duration_candles: int   # Number of candles from origin to peak
    timestamp: datetime     # When the wave completed

# Statistics returned by AmplitudeTracker.get_statistics():
# {
#     "p25": 12.3,          # 25th percentile amplitude
#     "p50": 21.4,          # Median amplitude (used for maturity)
#     "p75": 34.1,          # 75th percentile (used for TP)
#     "p90": 52.8,          # 90th percentile
#     "mean_pips": 25.7,
#     "p50_candles": 18,    # Median wave duration in candles (NEW)
#     "p75_candles": 28,    # 75th percentile duration
#     "mean_candles": 22.1,
# }
```

## Amplitude Tracker

The amplitude tracker maintains a rolling history of completed wave amplitudes per instrument per timeframe. This is the data source for maturity calculation and take profit projection.

```python
class AmplitudeTracker:
    """
    Stores the pip distance of every completed wave (impulse leg).
    Used to:
      1. Calculate wave maturity (how far has current wave traveled vs history)
      2. Project take profit (where does a typical wave end from here)
    """

    def record_completed_wave(
        self,
        instrument: str,
        granularity: str,
        amplitude_pips: float,
        direction: str,        # "BULLISH" or "BEARISH"
        session: str,          # "London", "NY", "Asia", "Overlap"
    ):
        """Stores completed wave to SQLite. Rolling 500-wave window per TF."""

    def get_statistics(
        self,
        instrument: str,
        granularity: str,
        direction: str = None,
        session: str = None,
    ) -> dict:
        """
        Returns:
        {
            "mean_pips": 32.4,
            "median_pips": 28.1,
            "p75_pips": 41.2,      # 75th percentile — conservative TP
            "p90_pips": 58.7,      # 90th percentile — aggressive TP
            "max_pips": 112.0,
            "sample_count": 347
        }
        """
```

**Take Profit Default:** Use the 75th percentile amplitude as the default TP projection. This means 75% of historical waves from this timeframe have traveled at least this far — a conservative but achievable target.

---

## Minimum Wave Size Filter

Not all swing points are meaningful. A swing high 2 pips above the surrounding candles on M1 is noise. The minimum wave size filter rejects swings that don't meet a meaningful pip threshold.

```python
MIN_WAVE_PIPS = {
    # Minimum pip distance between swing high and swing low
    # to be considered a valid wave
    "M1":  5,
    "M5":  10,
    "M15": 15,
    "H1":  25,
    "H4":  50,
    "D":   100,
    "W":   200,
}
```

A swing high/low pair that spans less than the minimum for its timeframe is ignored. This prevents the system from trading micro-noise.

---

## All-Time High / Structural Resistance Mapping

Per the system design philosophy: when smaller timeframe waves repeatedly fail at the same price level, the bot should recognize that these smaller waves are mapping out a larger structure.

```python
class StructuralLevelDetector:
    """
    Scans swing highs across all timeframes for the instrument.
    Identifies price zones where multiple timeframes have produced
    swing failures (wave reversals at approximately the same level).

    A zone is significant when:
    - 3+ swing highs cluster within 0.2% of each other
    - At least one swing is from H1 or higher timeframe
    - The zone has held for at least 10 candles on H1

    These zones are:
    1. Flagged as resistance/support in the confluence engine
    2. Used to reduce TP targets if price is approaching a zone
    3. Used to increase conviction on a breakout above a zone
    """

    def get_active_zones(self, instrument: str, current_price: float) -> list[dict]:
        """
        Returns list of nearby structural zones:
        [
            {
                "price": 1.09500,
                "type": "RESISTANCE",
                "strength": 0.85,      # 0.0–1.0
                "source_timeframes": ["M15", "H1", "H4"],
                "distance_pips": 12.3,
                "is_all_time_high": False
            }
        ]
        """
```

---

## Wave Detection — Practical Example

Consider EUR/USD M15 candles over 4 hours:

```
Time    Open     High     Low      Close    Wave State
08:00   1.08500  1.08620  1.08480  1.08590  RANGING
08:15   1.08590  1.08700  1.08560  1.08680  RANGING
08:30   1.08680  1.08820  1.08650  1.08800  RANGING → BULLISH_IMPULSE
         ← Swing low confirmed at 1.08480 (3 candles back)
         ← Price making new highs
08:45   1.08800  1.08850  1.08760  1.08830  BULLISH_IMPULSE
09:00   1.08830  1.08910  1.08800  1.08890  BULLISH_IMPULSE
         ← New swing high in progress
09:15   1.08890  1.08950  1.08830  1.08850  BULLISH_IMPULSE
         ← Possible correction starting
09:30   1.08850  1.08870  1.08780  1.08800  BULLISH_CORRECTION
         ← Swing high at 1.08950 being confirmed
09:45   1.08800  1.08830  1.08750  1.08780  BULLISH_CORRECTION
10:00   1.08780  1.08850  1.08760  1.08840  BULLISH_CORRECTION
         ← Swing low at 1.08750 forming (HL = bullish structure)
10:15   1.08840  1.08980  1.08820  1.08960  BULLISH_IMPULSE ← ENTRY ZONE
         ← Price breaks above last swing high (1.08950)
         ← Correction confirmed as HL
         ← New impulse leg starting
         ← ENTRY SIGNAL: LONG on M15
         ← SL at wave origin (1.08480) + buffer
         ← TP at historical 75th percentile amplitude
```

This is the exact sequence the wave engine runs on every closed candle, per instrument, per timeframe.

---

## What the Wave Engine Does NOT Do

- Does not use RSI, MACD, moving averages, or any derivative indicator
- Does not predict price targets using Fibonacci retracements (amplitude history is used instead — same concept, data-driven rather than ratio-based)
- Does not use volume analysis (OANDA provides tick volume, not real volume — unreliable for FX)
- Does not look at order book or DOM data
- Does not connect to any external data source

Everything is derived from OANDA candle OHLC data alone.
