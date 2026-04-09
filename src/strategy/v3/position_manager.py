"""Position management: double-down on failed reversals, flip on confirmed reversals."""
from typing import Optional

from src.strategy.v3.types import ReversalPattern, V3Trade


# Action constants
HOLD = "HOLD"
DOUBLE_DOWN = "DOUBLE_DOWN"
FLIP = "FLIP"


class V3PositionManager:
    """
    Manages V3 trade lifecycle including the double-down / flip mechanism.

    When a position is open and an opposing 3-candle pattern begins forming:
    - If the opposing pattern FAILS (never closes beyond reference candle) →
      DOUBLE_DOWN on next same-direction signal
    - If the opposing pattern SUCCEEDS (closes beyond reference) →
      FLIP: close existing + open opposite
    """

    def __init__(self, max_double_downs: int = 1, double_down_enabled: bool = True):
        self.max_double_downs = max_double_downs
        self.double_down_enabled = double_down_enabled
        # Track pending double-down eligibility per instrument
        self._pending_double_down: dict[str, bool] = {}

    def check_opposing_candle(
        self,
        trade: V3Trade,
        candle_close: float,
    ) -> str:
        """
        Check each new 4H candle against the open trade's reference close.

        For a LONG trade: if a candle closes below reference_candle_close,
        the opposing move is confirmed → FLIP.

        For a SHORT trade: if a candle closes above reference_candle_close,
        the opposing move is confirmed → FLIP.

        If candles form an opposing pattern but never break the reference level,
        mark as eligible for double-down on the next same-direction signal.

        Returns: HOLD, FLIP
        """
        ref = trade.reference_candle_close

        if trade.direction == "LONG":
            if candle_close < ref:
                # Opposing move confirmed — signal to flip
                self._pending_double_down.pop(trade.instrument, None)
                return FLIP
        elif trade.direction == "SHORT":
            if candle_close > ref:
                self._pending_double_down.pop(trade.instrument, None)
                return FLIP

        return HOLD

    def mark_failed_reversal(self, instrument: str) -> None:
        """
        Called when a 3-candle opposing pattern completes but never broke the
        reference level. The next same-direction signal can double down.
        """
        self._pending_double_down[instrument] = True

    def should_double_down(
        self, trade: V3Trade, new_pattern: ReversalPattern
    ) -> bool:
        """
        Check if we should double down on a new same-direction signal.

        Conditions:
        - Double-down is enabled
        - A failed reversal was detected (pending flag set)
        - The new pattern direction matches the open trade direction
        - Max double-downs not exceeded
        """
        if not self.double_down_enabled:
            return False

        if not self._pending_double_down.get(trade.instrument, False):
            return False

        expected_dir = "BULLISH" if trade.direction == "LONG" else "BEARISH"
        if new_pattern.direction != expected_dir:
            return False

        if trade.double_down_count >= self.max_double_downs:
            return False

        return True

    def consume_double_down(self, instrument: str) -> None:
        """Clear the pending double-down flag after it's been used."""
        self._pending_double_down.pop(instrument, None)

    def should_close_on_signal(
        self,
        trade: V3Trade,
        new_pattern: Optional[ReversalPattern],
    ) -> bool:
        """
        Check if an open trade should be closed because the opposite
        3-candle reversal pattern has completed.
        """
        if new_pattern is None:
            return False

        if trade.direction == "LONG" and new_pattern.direction == "BEARISH":
            return True
        if trade.direction == "SHORT" and new_pattern.direction == "BULLISH":
            return True

        return False

    def reset(self, instrument: str) -> None:
        """Clear state for an instrument (e.g., after position fully closed)."""
        self._pending_double_down.pop(instrument, None)
