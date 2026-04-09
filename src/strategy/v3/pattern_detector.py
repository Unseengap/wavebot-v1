"""4H 3-candle reversal pattern detector for Swing Reversal V3."""
from typing import Optional

from src.strategy.v3.types import CandleSnapshot, ReversalPattern


class CandleBuffer:
    """Rolling window of the last N completed candles."""

    def __init__(self, size: int = 3):
        self._size = size
        self._candles: list[CandleSnapshot] = []

    def push(self, candle: CandleSnapshot) -> None:
        self._candles.append(candle)
        if len(self._candles) > self._size:
            self._candles.pop(0)

    def is_ready(self) -> bool:
        return len(self._candles) == self._size

    @property
    def candles(self) -> list[CandleSnapshot]:
        return list(self._candles)

    def clear(self) -> None:
        self._candles.clear()


def detect_bullish_reversal(
    c1: CandleSnapshot,
    c2: CandleSnapshot,
    c3: CandleSnapshot,
    min_body_ratio: float = 0.3,
) -> Optional[ReversalPattern]:
    """
    Detect a bullish 3-candle reversal.

    Rules:
      - C1 must be bearish with meaningful body (not a doji)
      - C2 must close above C1's close (first reversal sign)
      - C3 must close >= C2's close (confirmation)
        - CLEAN: C3 low >= C2 close (no dip below)
        - REJECTION: C3 low < C2 close but C3 close >= C2 close (dipped then recovered)
    """
    # C1 must be bearish with a real body
    if not c1.is_bearish:
        return None
    if c1.body_ratio < min_body_ratio:
        return None

    # C2 must close above C1's close
    if c2.close <= c1.close:
        return None

    # C3 must close at or above C2's close
    if c3.close < c2.close:
        return None

    # Determine confirmation type
    if c3.low >= c2.close:
        conf_type = "CLEAN"
    else:
        conf_type = "REJECTION"

    pattern_low = min(c1.low, c2.low, c3.low)
    pattern_high = max(c1.high, c2.high, c3.high)

    return ReversalPattern(
        direction="BULLISH",
        candle_1=c1,
        candle_2=c2,
        candle_3=c3,
        pattern_low=pattern_low,
        pattern_high=pattern_high,
        confirmation_type=conf_type,
        detected_time=c3.time,
    )


def detect_bearish_reversal(
    c1: CandleSnapshot,
    c2: CandleSnapshot,
    c3: CandleSnapshot,
    min_body_ratio: float = 0.3,
) -> Optional[ReversalPattern]:
    """
    Detect a bearish 3-candle reversal.

    Rules:
      - C1 must be bullish with meaningful body
      - C2 must close below C1's close (first reversal sign)
      - C3 must close <= C2's close (confirmation)
        - CLEAN: C3 high <= C2 close (no wick above)
        - REJECTION: C3 high > C2 close but C3 close <= C2 close (spiked then rejected)
    """
    # C1 must be bullish with a real body
    if not c1.is_bullish:
        return None
    if c1.body_ratio < min_body_ratio:
        return None

    # C2 must close below C1's close
    if c2.close >= c1.close:
        return None

    # C3 must close at or below C2's close
    if c3.close > c2.close:
        return None

    # Determine confirmation type
    if c3.high <= c2.close:
        conf_type = "CLEAN"
    else:
        conf_type = "REJECTION"

    pattern_low = min(c1.low, c2.low, c3.low)
    pattern_high = max(c1.high, c2.high, c3.high)

    return ReversalPattern(
        direction="BEARISH",
        candle_1=c1,
        candle_2=c2,
        candle_3=c3,
        pattern_low=pattern_low,
        pattern_high=pattern_high,
        confirmation_type=conf_type,
        detected_time=c3.time,
    )


def detect_pattern(
    buffer: CandleBuffer,
    min_body_ratio: float = 0.3,
) -> Optional[ReversalPattern]:
    """
    Attempt to detect a reversal pattern from a 3-candle buffer.
    Checks bullish first, then bearish. Returns the first match or None.
    """
    if not buffer.is_ready():
        return None

    c1, c2, c3 = buffer.candles

    # Try bullish reversal
    pattern = detect_bullish_reversal(c1, c2, c3, min_body_ratio)
    if pattern is not None:
        return pattern

    # Try bearish reversal
    pattern = detect_bearish_reversal(c1, c2, c3, min_body_ratio)
    if pattern is not None:
        return pattern

    return None
