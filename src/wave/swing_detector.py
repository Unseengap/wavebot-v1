"""Swing detection with first-occurrence tiebreaker for equal values."""
import numpy as np


def detect_swing_high(highs: np.ndarray, lookback: int = 3) -> np.ndarray:
    """
    A swing high at index i is confirmed when highs[i] is the highest
    in window [i-lookback : i+lookback+1]. Ties go to earliest occurrence.
    The swing at index i is only KNOWN at index i+lookback (confirmation lag).
    """
    n = len(highs)
    is_swing = np.zeros(n, dtype=bool)
    for i in range(lookback, n - lookback):
        window = highs[i - lookback: i + lookback + 1]
        if highs[i] == np.max(window):
            first_max = i - lookback + int(np.argmax(window))
            if first_max == i:
                is_swing[i] = True
    return is_swing


def detect_swing_low(lows: np.ndarray, lookback: int = 3) -> np.ndarray:
    """Symmetric to swing high. Ties go to earliest occurrence."""
    n = len(lows)
    is_swing = np.zeros(n, dtype=bool)
    for i in range(lookback, n - lookback):
        window = lows[i - lookback: i + lookback + 1]
        if lows[i] == np.min(window):
            first_min = i - lookback + int(np.argmin(window))
            if first_min == i:
                is_swing[i] = True
    return is_swing


def detect_swings(highs: np.ndarray, lows: np.ndarray, lookback: int = 3):
    """Returns (swing_highs_bool, swing_lows_bool) arrays."""
    return detect_swing_high(highs, lookback), detect_swing_low(lows, lookback)
