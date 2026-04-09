"""Two-stage confluence scoring: directional gate (D/W) + entry quality (M1-H4)."""

# Stage 2 flattened entry weights — D and W excluded from scoring
ENTRY_WEIGHTS = {
    "M1": 1.0, "M5": 2.0, "M15": 3.5,
    "H1": 5.0, "H4": 7.0,
}

SESSION_SIZE_MULTIPLIERS = {
    "London_Open":       1.10,
    "London_NY_Overlap": 1.25,
    "NY_Session":        1.00,
    "Asia_Session":      0.60,
    "Dead_Zone":         0.00,
}


def get_directional_gate(wave_scores: dict) -> str:
    """
    Stage 1: Binary directional filter from D and W.
    Returns "BULLISH", "BEARISH", or "NEUTRAL".
    """
    d_score = wave_scores.get("D")
    w_score = wave_scores.get("W")

    if d_score is None:
        return "NEUTRAL"

    # Weekly overrides daily if strong
    if w_score and abs(w_score.direction * w_score.conviction) > 0.5:
        bias = w_score.direction
    else:
        bias = d_score.direction * d_score.conviction

    if bias > 0.15:
        return "BULLISH"
    elif bias < -0.15:
        return "BEARISH"
    return "NEUTRAL"


def calculate_confluence_score(wave_scores: dict) -> float:
    """
    Stage 2: Entry quality score using only M1-H4.
    Returns float in [-1.0, +1.0].
    """
    weighted_sum = 0.0
    total_weight = 0.0

    for tf, ws in wave_scores.items():
        if ws is None:
            continue
        weight = ENTRY_WEIGHTS.get(tf)
        if weight is None:
            continue  # Skip D, W
        effective = ws.direction * ws.conviction
        weighted_sum += effective * weight
        total_weight += weight

    if total_weight == 0:
        return 0.0

    raw = weighted_sum / total_weight

    # Maturity penalty on primary signal frame
    primary_mat = _get_primary_maturity(wave_scores)
    penalty = max(0.0, primary_mat - 0.6) * 0.5
    adjusted = raw * (1.0 - penalty)

    return round(max(-1.0, min(1.0, adjusted)), 4)


def _get_primary_maturity(wave_scores: dict) -> float:
    """Get maturity of the lowest active entry frame with signal."""
    for tf in ["M5", "M15", "M1", "H1", "H4"]:
        ws = wave_scores.get(tf)
        if ws and ws.state in ("BULLISH_IMPULSE", "BEARISH_IMPULSE"):
            return ws.maturity
    return 0.0


def get_signal_frame(wave_scores: dict, direction: str):
    """
    Find the signal frame — lowest TF with a fresh impulse
    in the trade direction.
    """
    target_states = {
        "LONG": "BULLISH_IMPULSE",
        "SHORT": "BEARISH_IMPULSE",
    }
    target = target_states.get(direction)
    if not target:
        return None

    for tf in ["M5", "M15", "M1"]:
        ws = wave_scores.get(tf)
        if ws and ws.state == target and ws.maturity < 0.60:
            return ws
    return None


def get_session(hour_utc: int) -> str:
    """Map UTC hour to trading session name."""
    if 0 <= hour_utc < 8:
        return "Asia_Session"
    elif 8 <= hour_utc < 13:
        return "London_Open"
    elif 13 <= hour_utc < 17:
        return "London_NY_Overlap"
    elif 17 <= hour_utc < 22:
        return "NY_Session"
    else:
        return "Dead_Zone"


def get_session_size_multiplier(session: str) -> float:
    return SESSION_SIZE_MULTIPLIERS.get(session, 1.0)
