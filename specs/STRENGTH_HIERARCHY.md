# STRENGTH_HIERARCHY.md — Timeframe Authority & Wave Weighting System

## The Hierarchy is Not Arbitrary

Every timeframe in the hierarchy represents a different class of market participant:

| Timeframe | Primary Participant | Decision Horizon | Reversal Cost |
|---|---|---|---|
| M1 | Scalpers, HFT noise | Seconds | Near zero |
| M5 | Short-term speculators | Minutes | Very low |
| M15 | Intraday traders | Hours | Low |
| H1 | Day traders, swing entry | 1 day | Moderate |
| H4 | Swing traders, fund positioning | Days | High |
| Daily | Institutional, fund managers | Weeks | Very high |
| Weekly | Macro funds, central bank flows | Months | Extremely high |

A weekly wave cannot be reversed by retail scalpers. It requires the same class of participant that created it to unwind. This is why higher timeframe waves are assigned greater authority — they represent a deeper and more committed form of capital.

---

## Hierarchy Weight Derivation

### Two-Stage Weight Architecture

The weights are split into two tiers reflecting the two-stage scoring model (see CONFLUENCE_ENGINE.md):

- **Stage 1 (Directional Gate):** D and W are used as a binary directional filter only. They are **not included** in the confluence score calculation. Their role is to set a pass/fail gate: BULLISH, BEARISH, or NEUTRAL.
- **Stage 2 (Entry Quality Score):** M1 through H4 compute the actual confluence score using a flattened weight curve where every frame meaningfully contributes.

This separation exists because the original combined weights (D=18.0, W=30.0) made lower timeframes cosmetic in the score — a strong daily signal overwhelmed all entry frames combined.

### Stage 2 Entry Weights (Used in Confluence Scoring)

```python
# Candle ratios (relative to M1)
CANDLE_RATIOS = {
    "M1":  1,
    "M5":  5,
    "M15": 15,
    "H1":  60,
    "H4":  240,
}

# Authority multipliers (empirical — not purely mathematical)
# Higher multiplier = disproportionate weight relative to raw candle ratio
# This reflects the human reality: participants on higher frames are more committed
AUTHORITY_MULTIPLIERS = {
    "M1":  1.0,
    "M5":  1.0,
    "M15": 1.05,
    "H1":  1.15,    # Reduced from 1.2 to flatten the curve
    "H4":  1.25,    # Reduced from 1.4 to prevent H4 domination
}

# Final entry weights (log-scaled ratio × authority multiplier)
# These are the values used in Stage 2 confluence scoring
ENTRY_WEIGHTS = {
    "M1":  1.0,    # Baseline
    "M5":  2.0,    # 5× ratio × 1.0 authority, scaled
    "M15": 3.5,    # 15× ratio × 1.05, scaled
    "H1":  5.0,    # 60× ratio × 1.15, log-scaled (was 6.0)
    "H4":  7.0,    # 240× ratio × 1.25, log-scaled (was 10.0)
}
```

The flatter curve gives entry frames a meaningful proportion of the total weight:
- Entry frames (M1 + M5 + M15) = 6.5 out of 18.5 total = **35%** of score
- Confirmation frames (H1 + H4) = 12.0 out of 18.5 total = **65%** of score
- Compare to original: entry frames were 6.5 out of 94.5 = **7%** of score

This means lower timeframe signals now actually influence entry decisions.

### Stage 1 Gate Weights (D, W — Not Used in Score)

D and W are analyzed by the wave engine but their output feeds a **binary directional gate** rather than the weighted confluence score.

```python
# D and W do NOT contribute to the confluence score.
# They answer one question: "What direction should we be looking?"
#
# BULLISH gate  → only long entries allowed (unless counter-score > 0.85)
# BEARISH gate  → only short entries allowed (unless counter-score > 0.85)
# NEUTRAL gate  → no entries (wait for directional clarity)
#
# Weekly overrides daily when weekly conviction > 0.5
# See CONFLUENCE_ENGINE.md Stage 1 for implementation.
```

---

## How Timeframes Interact

### The Parent-Child Relationship

Every timeframe is both a child of the timeframe above it and a parent to the timeframe below. This creates a natural filter:

```
D  ──→ sets the trend bias for the week
  H4 ──→ sets the swing structure for the day
    H1 ──→ sets the intraday structure
      M15 ──→ provides the setup (confirmation frame)
        M5  ──→ provides the entry timing
          M1  ──→ provides the precise entry candle
```

A trade entered on M5 that is aligned with M15, H1, H4, and D is essentially a D-level trade with M5 precision. This is the ideal setup — the conviction of a higher frame with the accuracy of a lower frame.

### The Fractal Property

The same wave patterns appear on every timeframe. A bullish impulse on M15 looks structurally identical to a bullish impulse on H4. The patterns are self-similar (fractal). This is why the wave engine can use identical detection logic across all timeframes — only the parameters (minimum pip size, lookback) change.

```
D:   ▁▁▂▃▄▅▆▇█▇▆▅▄▃▂▁▁   (one wave over weeks)
H4:  ▁▂▄▆█▆▄▂▁▁▂▄▆█▆▄▂▁  (multiple waves within D wave)
H1:  ▁▂▃▄▅▆▅▄▃▂▁▂▃▄▅▆▅▄  (multiple waves within each H4 wave)
M15: ▁▂▃▄▃▂▁▂▃▄▃▂▁▂▃▄▃▂  (multiple waves within each H1 wave)
```

When zoomed out, the M15 waves become the texture of the H4 wave. When zoomed in, each M15 wave has its own M5 waves forming inside it.

---

## Timeframe Role Definitions

### Context Frames (D, W) — Stage 1 Directional Gate

Context frames are loaded and analyzed but serve as a **binary directional gate** rather than contributing to the confluence score. They answer one question: **What direction should we be looking?**

The directional gate has three states:
- **BULLISH** → only long entries allowed
- **BEARISH** → only short entries allowed
- **NEUTRAL** → no entries in either direction (wait for clarity)

Counter-gate trades (e.g., shorting when gate is BULLISH) require an exceptionally high Stage 2 confluence score (> 0.85) — this is a safety valve, not a normal entry path.

```python
def get_directional_gate(wave_scores: dict) -> str:
    """
    Uses D and W wave scores as a binary directional filter.
    D and W do NOT contribute to the Stage 2 confluence score.

    Weekly overrides daily when weekly conviction > 0.5.
    """
    d_score  = wave_scores.get("D")
    w_score  = wave_scores.get("W")

    if d_score is None:
        return "NEUTRAL"

    # Weekly frame overrides daily if available and strong
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

### Confirmation Frames (H1, H4)

Confirmation frames must be aligned with the entry direction for a trade to be valid. They answer: **Is the intermediate-term structure supportive?**

H4 alignment increases the confluence score significantly. H1 alignment is required (configurable but strongly recommended). A long trade without H1 being bullish requires a manual override and is flagged in the log.

### Entry Frames (M5, M15, M1)

Entry frames generate the actual trade signal. The signal is a transition from CORRECTION to IMPULSE in the direction of the confluence. The entry frame also determines:
- Stop loss price (origin of the signal wave on this frame)
- Signal quality score (how clean was the swing structure?)
- Wave maturity at entry (is this a fresh wave or a late entry?)

### Priority Entry Frame Logic

When multiple entry frames simultaneously show valid impulse transitions, the system uses the following priority:

```python
ENTRY_FRAME_PRIORITY = ["M15", "M5", "M1"]
# M15 preferred: larger waves, more reliable
# M5 second: still meaningful, faster response
# M1 last: only if M15 and M5 are not signaling

def select_entry_frame(wave_scores: dict) -> str | None:
    for tf in ENTRY_FRAME_PRIORITY:
        ws = wave_scores.get(tf)
        if ws and ws.state in ("BULLISH_IMPULSE", "BEARISH_IMPULSE"):
            if ws.maturity < 0.3:     # Fresh wave only
                return tf
    return None   # No valid entry frame — wait
```

---

## Cross-Timeframe Conflict Resolution

### Scenario: M15 Bullish but H1 Bearish

This is the most common conflict. M15 is showing a bullish impulse but H1 is in a bearish impulse. Options:

**Option A: No trade** (default behavior when H1 alignment is required)
The M15 bullish impulse is likely a correction within the H1 bearish impulse. Trading it would mean going against a stronger frame. Skip.

**Option B: Short at M15 resistance** (advanced)
If H1 is bearish and M15 rallies (bullish correction), that M15 high could be the entry for a short trade — shorting into the M15 correction within the H1 bearish impulse. This requires:
- H4 to be bearish or neutral
- M15 wave showing exhaustion (maturity > 0.70) at the top
- Confluence score for SHORT direction still above threshold

This is a valid setup but requires the bot to recognize it as a counter-M15 entry in the direction of H1. The signal frame is technically M15 (exhausted bullish = reversal signal) but the trade direction is bearish.

### Scenario: H4 Strongly Bullish but D Strongly Bearish

The system treats this as **no trade in either direction.** The conflict between H4 and D represents a structural disagreement between swing traders and positional traders. These periods are often precursors to significant moves but the direction is uncertain. The system waits.

```python
def has_structural_conflict(wave_scores: dict) -> bool:
    """
    Returns True if higher frames are in direct opposition.
    This is a veto — no entries during structural conflict.
    """
    d_score  = wave_scores.get("D")
    h4_score = wave_scores.get("H4")

    if d_score and h4_score:
        # Both have strong conviction but opposite directions
        d_conviction  = d_score.direction * d_score.conviction
        h4_conviction = h4_score.direction * h4_score.conviction

        if d_conviction > 0.5 and h4_conviction < -0.5:
            return True  # D bullish, H4 bearish — conflict
        if d_conviction < -0.5 and h4_conviction > 0.5:
            return True  # D bearish, H4 bullish — conflict

    return False
```

---

## Smaller Timeframes as All-Time High Detectors

This is the structural level detection capability described in the philosophy.

When M5 waves are repeatedly failing at the same price — creating a pattern of lower highs that don't quite reach a previous peak — the amplitude tracker will record progressively smaller wave amplitudes. This is a fingerprint of price approaching a significant structural level.

```python
def detect_compression_at_level(
    instrument: str,
    granularity: str,
    lookback_waves: int = 10,
) -> dict | None:
    """
    Detects when recent waves are getting smaller (price is compressing).

    Returns:
    {
        "detected": True,
        "compression_level": 1.09450,   # Price where waves are failing
        "compression_type": "RESISTANCE" or "SUPPORT",
        "avg_recent_amplitude": 8.3,    # pips — getting smaller
        "avg_historical_amplitude": 21.4, # pips — normal size
        "amplitude_ratio": 0.39,         # 39% of normal = strong compression
        "significance": "HIGH"
    }
    """
    recent_amplitudes = self.amplitude_tracker.get_last_n(instrument, granularity, n=lookback_waves)
    historical_avg = self.amplitude_tracker.get_statistics(instrument, granularity)["mean_pips"]

    if len(recent_amplitudes) < lookback_waves:
        return None

    recent_avg = np.mean(recent_amplitudes)
    ratio = recent_avg / historical_avg

    if ratio < 0.5:  # Recent waves are less than half normal size
        return {
            "detected": True,
            "amplitude_ratio": ratio,
            "significance": "HIGH" if ratio < 0.35 else "MODERATE"
        }

    return None
```

When compression is detected, the system flags the current price level as a potential breakout zone. The next bullish or bearish impulse that breaks through the compression level is treated as a high-conviction breakout — the pent-up energy of the compression fueling the move.
