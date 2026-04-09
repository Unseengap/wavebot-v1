"""Wave scoring: direction, conviction, and dual-axis maturity."""
from dataclasses import dataclass


@dataclass
class WaveScore:
    instrument: str
    granularity: str
    timestamp: str
    direction: float       # +1.0 fully bullish to -1.0 fully bearish
    conviction: float      # 0.0 to 1.0
    maturity: float        # 0.0 (fresh) to 1.0 (exhausted)
    state: str             # WaveState string
    wave_origin: float     # Price — SL anchor
    wave_age_candles: int
    wave_pips: float


DIRECTION_MAP = {
    "BULLISH_IMPULSE":    1.0,
    "BULLISH_CORRECTION": 0.5,
    "RANGING":            0.0,
    "BEARISH_CORRECTION": -0.5,
    "BEARISH_IMPULSE":    -1.0,
}


def calculate_direction(state: str) -> float:
    return DIRECTION_MAP.get(state, 0.0)


def calculate_conviction(state: str, maturity: float,
                         swing_clarity: float,
                         consecutive_swings: int) -> float:
    if state == "RANGING":
        return 0.0

    base = 1.0
    maturity_factor = 1.0 - (maturity * 0.4)
    clarity_factor = 0.7 + (swing_clarity * 0.3)
    swing_factor = min(1.0, 0.6 + (consecutive_swings * 0.1))

    conviction = base * maturity_factor * clarity_factor * swing_factor
    return round(min(1.0, max(0.0, conviction)), 4)


def calculate_maturity(wave_pips: float, wave_age_candles: int,
                       amplitude_stats: dict, duration_stats: dict) -> float:
    """
    Dual-axis maturity: max(price_maturity, time_maturity).
    Uses median (p50) to avoid outlier skew.
    """
    median_amp = amplitude_stats.get("p50", 0)
    median_dur = duration_stats.get("p50_candles", 0)

    if median_amp > 0:
        price_mat = min(1.0, (wave_pips / median_amp) * 0.6)
    else:
        price_mat = 0.5

    if median_dur > 0:
        time_mat = min(1.0, (wave_age_candles / median_dur) * 0.6)
    else:
        time_mat = 0.5

    return round(max(price_mat, time_mat), 4)


def calculate_swing_clarity(last_swing_high, prev_swing_high,
                            last_swing_low, prev_swing_low,
                            state: str) -> float:
    """
    Measures how cleanly the swing structure is defined.
    Higher = cleaner swings = more reliable detection.
    """
    if state in ("BULLISH_IMPULSE", "BULLISH_CORRECTION"):
        if (last_swing_high and prev_swing_high and
                last_swing_low and prev_swing_low):
            hh_clear = last_swing_high > prev_swing_high
            hl_clear = last_swing_low > prev_swing_low
            return 1.0 if (hh_clear and hl_clear) else 0.6
        return 0.5
    elif state in ("BEARISH_IMPULSE", "BEARISH_CORRECTION"):
        if (last_swing_high and prev_swing_high and
                last_swing_low and prev_swing_low):
            lh_clear = last_swing_high < prev_swing_high
            ll_clear = last_swing_low < prev_swing_low
            return 1.0 if (lh_clear and ll_clear) else 0.6
        return 0.5
    return 0.3


def score_wave(instrument: str, granularity: str, timestamp: str,
               machine, amplitude_stats: dict,
               duration_stats: dict, pip_size: float) -> WaveScore:
    """Build a WaveScore from the current state machine."""
    state = machine.state
    direction = calculate_direction(state)

    wave_origin = machine.wave_origin or 0.0
    wave_age = machine.candle_idx - machine.wave_start_idx

    # Calculate wave distance in pips
    if state in ("BULLISH_IMPULSE", "BULLISH_CORRECTION"):
        extent = machine.wave_peak or 0.0
        wave_pips = abs(extent - wave_origin) / pip_size if wave_origin else 0.0
    elif state in ("BEARISH_IMPULSE", "BEARISH_CORRECTION"):
        extent = machine.wave_trough or 0.0
        wave_pips = abs(wave_origin - extent) / pip_size if wave_origin else 0.0
    else:
        wave_pips = 0.0

    maturity = calculate_maturity(wave_pips, wave_age,
                                  amplitude_stats, duration_stats)

    clarity = calculate_swing_clarity(
        machine.last_swing_high, machine.prev_swing_high,
        machine.last_swing_low, machine.prev_swing_low,
        state
    )
    conviction = calculate_conviction(state, maturity, clarity,
                                      machine.consecutive_swings)

    return WaveScore(
        instrument=instrument,
        granularity=granularity,
        timestamp=timestamp,
        direction=direction,
        conviction=conviction,
        maturity=maturity,
        state=state,
        wave_origin=wave_origin,
        wave_age_candles=wave_age,
        wave_pips=wave_pips,
    )
