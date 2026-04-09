"""Stop loss, take profit, and risk:reward calculation."""
import numpy as np


def calculate_stop_loss(direction: str, wave_origin: float,
                        pip_size: float, buffer_pips: float = 2.0) -> float:
    buffer = buffer_pips * pip_size
    if direction == "LONG":
        return round(wave_origin - buffer, 5)
    else:
        return round(wave_origin + buffer, 5)


def calculate_take_profit(direction: str, entry_price: float,
                          amplitude_stats: dict, pip_size: float,
                          tp_percentile: str = "p75") -> float:
    target_pips = amplitude_stats.get(tp_percentile, 30)
    target_dist = target_pips * pip_size

    if direction == "LONG":
        return round(entry_price + target_dist, 5)
    else:
        return round(entry_price - target_dist, 5)


def calculate_rr(entry: float, stop_loss: float,
                 take_profit: float, direction: str) -> float:
    risk = abs(entry - stop_loss)
    reward = abs(take_profit - entry)
    if risk == 0:
        return 0.0
    return round(reward / risk, 2)


def validate_sl_distance(sl_distance_pips: float, atr_pips: float,
                         max_atr_multiple: float = 3.0) -> bool:
    if atr_pips <= 0:
        return True
    return sl_distance_pips <= atr_pips * max_atr_multiple


def calculate_atr(highs: np.ndarray, lows: np.ndarray,
                  closes: np.ndarray, period: int = 14) -> float:
    """Simple ATR from the last `period` candles."""
    if len(highs) < period + 1:
        return float(np.mean(highs[-period:] - lows[-period:]))

    tr = np.maximum(
        highs[-period:] - lows[-period:],
        np.maximum(
            np.abs(highs[-period:] - closes[-period - 1:-1]),
            np.abs(lows[-period:] - closes[-period - 1:-1])
        )
    )
    return float(np.mean(tr))
