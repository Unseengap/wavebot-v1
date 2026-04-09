"""ATR-based trailing stop loss for Swing Reversal V3."""
import numpy as np


def calculate_atr(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 14,
) -> float:
    """
    Standard Average True Range over the last `period` candles.
    Returns 0.0 if insufficient data.
    """
    if len(highs) < 2 or len(highs) < period:
        return 0.0

    true_ranges = np.empty(len(highs) - 1)
    for i in range(1, len(highs)):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        true_ranges[i - 1] = max(hl, hc, lc)

    if len(true_ranges) < period:
        return float(np.mean(true_ranges))

    return float(np.mean(true_ranges[-period:]))


class TrailingStopManager:
    """Manages ATR-based trailing stop that never moves against the trade."""

    def __init__(self, multiplier: float = 2.0):
        self.multiplier = multiplier

    def initial_stop(
        self,
        direction: str,
        pattern_extreme: float,
        buffer_pips: float,
        pip_size: float,
    ) -> float:
        """
        Calculate initial stop loss from the pattern extreme.
        LONG: pattern_low - buffer  |  SHORT: pattern_high + buffer
        """
        buffer = buffer_pips * pip_size
        if direction == "LONG":
            return pattern_extreme - buffer
        else:
            return pattern_extreme + buffer

    def update(
        self,
        direction: str,
        current_price: float,
        current_atr: float,
        current_stop: float,
    ) -> float:
        """
        Update trailing stop. Never moves backward (against the trade).

        LONG:  new_stop = max(current_stop, price - multiplier * ATR)
        SHORT: new_stop = min(current_stop, price + multiplier * ATR)
        """
        if current_atr <= 0:
            return current_stop

        trail_distance = self.multiplier * current_atr

        if direction == "LONG":
            candidate = current_price - trail_distance
            return max(current_stop, candidate)
        else:
            candidate = current_price + trail_distance
            return min(current_stop, candidate)
