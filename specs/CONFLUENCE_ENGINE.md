# CONFLUENCE_ENGINE.md — Multi-Timeframe Signal Fusion Specification

## The Hierarchy Principle

Every timeframe has a natural authority over the timeframes below it. This is not an arbitrary rule — it reflects market reality:

- A daily wave represents the aggregate positioning of thousands of institutional traders over 24 hours
- A 1-minute wave represents a few seconds of order flow, often noise
- When these conflict, the daily wave wins — the institutional positioning absorbs the noise

The hierarchy does not mean smaller timeframes are useless. It means they serve a different purpose: **precision entry into moves that higher timeframes have already confirmed.**

---

## Two-Stage Scoring Architecture

### Why Two Stages?

The original single-score model with weights D=18.0, W=30.0 vs. M1=1.0 over-concentrated authority in context frames. A strong daily signal (direction 1.0, conviction 0.85) contributed 18 × 0.85 = 15.3 to the weighted sum. All three entry frames at maximum conviction combined contributed only 1.0 + 2.0 + 3.5 = 6.5. The system was overwhelmingly "trade in the direction of the daily trend" — lower timeframes were cosmetic in the score.

The two-stage architecture separates the roles cleanly:
- **Stage 1 (Directional Gate):** D and W set a binary directional filter — bullish, bearish, or neutral. This is a pass/fail gate, not a weighted contribution.
- **Stage 2 (Entry Quality Score):** Only entry and confirmation frames (M1 through H4) compute the confluence score using a flatter weight curve where every frame meaningfully contributes.

### Stage 1 — Directional Gate (D, W)

```python
def get_directional_gate(wave_scores: dict) -> str:
    """
    Uses D and W wave scores as a binary directional filter.
    This is a GATE, not a score contributor.

    Returns: "BULLISH", "BEARISH", or "NEUTRAL"

    Rules:
    - Only entries in the gate direction are allowed
    - NEUTRAL = no entries in either direction (wait for clarity)
    - Counter-gate entries require score > 0.85 (exceptional override)
    """
    d_score = wave_scores.get("D")
    w_score = wave_scores.get("W")

    if d_score is None:
        return "NEUTRAL"

    # Weekly overrides daily if available and strong
    if w_score and abs(w_score.direction * w_score.conviction) > 0.5:
        bias_direction = w_score.direction
    else:
        bias_direction = d_score.direction * d_score.conviction

    if bias_direction > 0.3:
        return "BULLISH"
    elif bias_direction < -0.3:
        return "BEARISH"
    else:
        return "NEUTRAL"
```

### Stage 2 — Entry Quality Score (M1 through H4)

```python
def calculate_confluence_score(wave_scores: dict[str, WaveScore]) -> float:
    """
    Computes confluence using ONLY entry and confirmation frames.
    D and W are excluded — they serve as the directional gate (Stage 1).

    The weight curve is flatter than the original:
    - H4 is 7× M1 (was 10×)
    - H1 is 5× M1 (was 6×)
    - Entry frames collectively contribute meaningful weight

    Returns float in range [-1.0, +1.0]
    """

    # Flatter weights — entry frames actually matter
    ENTRY_WEIGHTS = {
        "M1":  1.0,
        "M5":  2.0,
        "M15": 3.5,
        "H1":  5.0,   # Was 6.0 — reduced to give entry frames more influence
        "H4":  7.0,   # Was 10.0 — reduced to prevent H4 domination
    }

    # D and W are EXCLUDED from scoring — they serve as Stage 1 gate only
    # This ensures lower timeframes actually contribute to entry decisions

    weighted_sum = 0.0
    total_weight = 0.0

    for granularity, ws in wave_scores.items():
        if ws is None:
            continue

        weight = ENTRY_WEIGHTS.get(granularity)
        if weight is None:
            continue  # Skip D, W, M — not part of entry scoring

        effective_signal = ws.direction * ws.conviction
        weighted_sum += effective_signal * weight
        total_weight += weight

    if total_weight == 0:
        return 0.0

    raw_score = weighted_sum / total_weight

    # Apply maturity penalty:
    # If the primary signal frame (lowest TF showing signal) is highly mature,
    # reduce the score — the wave may be exhausted
    primary_maturity = get_primary_frame_maturity(wave_scores)
    maturity_penalty = max(0.0, primary_maturity - 0.6) * 0.5
    # Penalty only applies above 60% maturity, max 20% reduction

    adjusted_score = raw_score * (1.0 - maturity_penalty)

    return round(max(-1.0, min(1.0, adjusted_score)), 4)
```

---

## Alignment Patterns

Not all confluence is equal. These are the named patterns the confluence engine recognizes:

### Pattern 1: Full Stack Alignment (highest accuracy)

All active timeframes point in the same direction.

```
D:   BULLISH_IMPULSE  (direction: +1.0, conviction: 0.82)
H4:  BULLISH_IMPULSE  (direction: +1.0, conviction: 0.74)
H1:  BULLISH_IMPULSE  (direction: +1.0, conviction: 0.68)
M15: BULLISH_CORRECTION → IMPULSE resumption (entry signal)
M5:  BULLISH_IMPULSE  (direction: +1.0, conviction: 0.71)

Confluence Score: +0.89 → STRONG LONG ENTRY
```

This pattern has the highest accuracy. The M15 correction into a resuming bullish structure, while all higher timeframes are bullish, is the highest-quality setup the system trades.

### Pattern 2: Higher-Frame Trend + Lower-Frame Entry

Higher timeframes define direction; smaller timeframes provide precise entry.

```
D:   BULLISH_IMPULSE  (direction: +1.0, conviction: 0.85)
H4:  BULLISH_CORRECTION (direction: +0.5, conviction: 0.60)
H1:  BULLISH_CORRECTION ending → new BULLISH_IMPULSE
M15: BULLISH_IMPULSE  (fresh, maturity 0.12)

Interpretation:
  - D says: trend is up
  - H4 says: we're in a normal pullback
  - H1 says: pullback is ending
  - M15 says: new upward move starting NOW

Confluence Score: +0.76 → VALID LONG ENTRY
Entry: Buy on M15 impulse confirmation
SL:   At M15 wave origin (the pullback low)
TP:   H4 amplitude projection (the size of a typical H4 impulse leg)
```

This is the most common and most elegant setup — entering a smaller timeframe impulse that is the resumption of a larger timeframe trend.

### Pattern 3: Smaller TF Mapping Structural Level

Repeated small-timeframe wave failures at the same price.

```
D:   BULLISH_IMPULSE  (direction: +1.0, conviction: 0.80)
H4:  BULLISH_IMPULSE  (approaching prior resistance)
H1:  BEARISH_CORRECTION (wave failing at resistance repeatedly)
M15: RANGING (multiple attempts at same level)
M5:  BEARISH_IMPULSE (rejections getting stronger)

Interpretation:
  - D and H4 say: trend is up
  - H1, M15, M5 say: price is coiling at resistance
  - This is NOT a short signal
  - This is a BREAKOUT SETUP — if M5 produces a bullish impulse above resistance,
    the confluence of all frames aligning produces an extremely high-conviction long

Wait for: M5 bullish impulse ABOVE the resistance level
When confirmed: Full Stack Alignment pattern triggers
```

### Pattern 4: Conflicted — No Trade

```
D:   BULLISH_IMPULSE  (direction: +1.0)
H4:  RANGING          (direction: 0.0)
H1:  BEARISH_IMPULSE  (direction: -1.0)
M15: BEARISH_IMPULSE  (direction: -1.0)

Confluence Score: +0.12 → NO TRADE
```

The D frame is bullish but the intermediate frames are not confirming. The score is near zero. The system sits in cash. This is not a failure — this is the system protecting capital. The setup will resolve one way or another, and a clear pattern will emerge.

---

## Entry Conditions Checklist

The confluence engine calls `check_entry_conditions()` after every confluence score update. All conditions must pass for an order to be submitted.

```python
def check_entry_conditions(
    instrument: str,
    direction: str,           # "LONG" or "SHORT"
    confluence_score: float,
    wave_scores: dict,
    account_state: dict,
    market_state: dict,
) -> tuple[bool, list[str]]:
    """
    Returns (can_enter: bool, failed_reasons: list[str])
    """

    checks = []

    # 1. Confluence threshold
    threshold = 0.72
    if direction == "LONG" and confluence_score < threshold:
        checks.append(f"FAIL: Confluence {confluence_score:.3f} < {threshold}")
    elif direction == "SHORT" and confluence_score > -threshold:
        checks.append(f"FAIL: Confluence {confluence_score:.3f} > -{threshold}")

    # 2. H1 must be aligned (non-negotiable)
    h1_score = wave_scores.get("H1")
    if h1_score:
        if direction == "LONG" and h1_score.direction <= 0:
            checks.append("FAIL: H1 not bullish — H1 alignment required")
        elif direction == "SHORT" and h1_score.direction >= 0:
            checks.append("FAIL: H1 not bearish — H1 alignment required")

    # 3. Minimum 3 timeframes aligned
    aligned_count = sum(
        1 for ws in wave_scores.values()
        if ws and (
            (direction == "LONG" and ws.direction > 0) or
            (direction == "SHORT" and ws.direction < 0)
        )
    )
    if aligned_count < 3:
        checks.append(f"FAIL: Only {aligned_count} frames aligned (need 3+)")

    # 4. No existing position on this instrument
    if account_state.get("open_positions", {}).get(instrument):
        checks.append(f"FAIL: Already in position on {instrument}")

    # 5. Max open trades not exceeded
    open_trade_count = len(account_state.get("open_positions", {}))
    if open_trade_count >= 3:
        checks.append(f"FAIL: Max open trades (3) reached")

    # 6. Spread within limits
    current_spread = market_state.get("spread_pips", 999)
    max_spread = get_max_spread(instrument)
    if current_spread > max_spread:
        checks.append(f"FAIL: Spread {current_spread:.1f} > max {max_spread:.1f}")

    # 7. Signal wave not too mature (exhausted)
    signal_frame = get_signal_frame(wave_scores)
    if signal_frame and signal_frame.maturity > 0.75:
        checks.append(f"FAIL: Signal wave maturity {signal_frame.maturity:.2f} > 0.75")

    # 8. Minimum R:R achievable
    rr = calculate_rr(instrument, direction, wave_scores, market_state)
    if rr < 2.0:
        checks.append(f"FAIL: R:R {rr:.2f} < 2.0 minimum")

    # 9. Daily drawdown limit not hit
    if account_state.get("daily_drawdown_pct", 0) >= 0.02:
        checks.append("FAIL: Daily drawdown limit reached — no new trades today")

    # 10. Session filter (optional but recommended)
    session = get_current_session()
    if session == "DEAD_ZONE":  # 22:00–00:00 UTC and Asian thin periods
        checks.append("SKIP: Low liquidity session — reduced entry probability")
        # Note: This is a soft skip, not a hard fail. Can be overridden.

    passed = len(checks) == 0
    return passed, checks
```

---

## The Signal Frame Concept

The "signal frame" is the timeframe that triggered the entry. It is always the lowest active timeframe that has just transitioned from CORRECTION to IMPULSE in the direction of the confluence.

```python
def get_signal_frame(wave_scores: dict) -> WaveScore | None:
    """
    Returns the signal frame — the entry trigger.

    Priority: M1 > M5 > M15 > H1 > H4
    (lowest timeframe that just started a new impulse in the trade direction)
    """
    ENTRY_ORDER = ["M1", "M5", "M15", "H1", "H4"]

    for tf in ENTRY_ORDER:
        ws = wave_scores.get(tf)
        if ws and ws.state in ("BULLISH_IMPULSE", "BEARISH_IMPULSE"):
            if ws.maturity < 0.3:  # Fresh — just started
                return ws

    return None
```

The signal frame determines:
- **Stop loss placement**: wave_origin of signal frame - buffer
- **Position sizing**: risk in pips = entry_price - SL
- **Partial close level**: when trade reaches 1:1 R:R (optional)

The **confirmation frames** (H1, H4) determine:
- **Take profit placement**: based on H4 or H1 amplitude history

This separation is critical: the signal frame gets you in precisely. The confirmation frames tell you how far the move should go.

---

## Session Weighting → Position Sizing (Not Score Modification)

### Why Session Does Not Modify the Score

The previous design applied a session multiplier directly to the confluence score. This created a problem: a 0.61 score during London/NY overlap (× 1.20 = 0.732) crossed the 0.72 entry threshold, while the same 0.61 score during Asia (× 0.85 = 0.519) did not. The session multiplier was not adjusting confidence — it was **creating entries that wouldn't otherwise exist**. This introduces regime-sensitivity and makes the entry threshold meaningless.

The fix: the confluence score is computed purely from wave alignment. Session quality influences **position size**, not the entry/no-entry decision. A 0.69 score is a 0.69 score in every session — you just trade it smaller (or not at all) during low-quality sessions.

### Session Position Size Multipliers

```python
SESSION_SIZE_MULTIPLIERS = {
    "London_Open":       1.10,  # 08:00–10:00 UTC — high volatility, reliable waves
    "London_NY_Overlap": 1.25,  # 13:00–17:00 UTC — peak volume, full position + 25%
    "NY_Session":        1.00,  # 13:00–22:00 UTC — baseline
    "Asia_Session":      0.60,  # 00:00–08:00 UTC — reduced size, thinner waves
    "Dead_Zone":         0.00,  # 22:00–00:00 UTC — no entries regardless of score
}

def apply_session_to_position_size(base_units: int, session: str) -> int:
    """
    Scales position size by session quality.
    Called AFTER the entry decision is made (score already above threshold).
    The entry threshold (0.72) is never modified.

    Dead_Zone returns 0 units → effectively blocks entry without
    corrupting the confluence score.
    """
    multiplier = SESSION_SIZE_MULTIPLIERS.get(session, 1.0)
    adjusted_units = int(base_units * multiplier)

    # Enforce minimum viable trade size
    MIN_UNITS = 1000  # 0.01 mini lot
    if multiplier == 0.0:
        return 0  # Dead zone — no trade
    return max(MIN_UNITS, adjusted_units)
```

### Examples

```
Score: 0.78 during London/NY Overlap
  → Entry: YES (0.78 > 0.72 threshold)
  → Base units: 50,000
  → Session multiplier: 1.25
  → Final units: 62,500

Score: 0.78 during Asia Session
  → Entry: YES (0.78 > 0.72 threshold — same score, same decision)
  → Base units: 50,000
  → Session multiplier: 0.60
  → Final units: 30,000

Score: 0.69 during London/NY Overlap
  → Entry: NO (0.69 < 0.72 threshold — session does NOT push it above)
  → No trade. The wave alignment is insufficient regardless of session.

Score: 0.85 during Dead Zone
  → Entry: NO (Dead Zone multiplier = 0.0 → 0 units)
  → Even strong signals are blocked during Dead Zone.
```

This separation preserves signal integrity. The entry threshold reflects wave alignment quality and nothing else.

---

## Output to Risk Engine

When entry conditions are met, the confluence engine produces a trade signal:

```python
@dataclass
class TradeSignal:
    instrument: str
    direction: str              # "LONG" or "SHORT"
    confluence_score: float
    signal_frame: str           # e.g. "M15"
    signal_wave_origin: float   # Price of wave origin (SL anchor)
    confirmation_frames: list   # e.g. ["H1", "H4"]
    pattern: str                # "FULL_STACK", "HTF_TREND_LTF_ENTRY", etc.
    session: str
    timestamp: datetime

    # Amplitude data for TP projection
    amplitude_stats: dict       # From AmplitudeTracker

    # Structural levels nearby
    nearby_zones: list[dict]    # From StructuralLevelDetector
```

This signal is passed to the risk engine, which calculates exact position size, SL price, and TP price before submitting to OANDA.
