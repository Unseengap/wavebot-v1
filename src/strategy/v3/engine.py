"""Swing Reversal V3 strategy engine — orchestrates pattern detection, trade management, and exits."""
from typing import Optional

from src.strategy.v3.types import CandleSnapshot, ReversalPattern, V3Trade
from src.strategy.v3.pattern_detector import CandleBuffer, detect_pattern
from src.strategy.v3.position_manager import V3PositionManager, HOLD, FLIP, DOUBLE_DOWN


# Action types returned by the engine
class Action:
    def __init__(self, action_type: str, **kwargs):
        self.type = action_type
        self.data = kwargs

    def __repr__(self):
        return f"Action({self.type}, {self.data})"


class SwingReversalV3:
    """
    Main V3 strategy orchestrator.

    On each 4H candle close:
      1. Push candle to buffer
      2. Detect 3-candle reversal patterns
      3. Check for exit signals (opposite pattern) or double-down opportunities
      4. Return list of actions to execute

    No stop loss or trailing stop — exits only via opposite signal or flip.
    """

    def __init__(
        self,
        instrument: str,
        pip_size: float,
        pip_value: float = 0.0001,
        min_body_ratio: float = 0.3,
        max_double_downs: int = 1,
        double_down_enabled: bool = True,
    ):
        self.instrument = instrument
        self.pip_size = pip_size
        self.pip_value = pip_value
        self.min_body_ratio = min_body_ratio

        self.buffer = CandleBuffer(size=3)
        self.position_mgr = V3PositionManager(
            max_double_downs=max_double_downs,
            double_down_enabled=double_down_enabled,
        )

        # Track the last confirmed pattern to avoid re-signaling
        self.last_pattern: Optional[ReversalPattern] = None
        # Previous daily candle for context
        self._prev_daily_close: Optional[float] = None

    def on_h4_candle_close(
        self,
        candle: CandleSnapshot,
        open_trades: list[V3Trade],
    ) -> list[Action]:
        """
        Process a new completed 4H candle. Returns a list of actions.

        Action types:
          - OPEN_LONG / OPEN_SHORT: open a new trade
          - CLOSE: close a specific trade (with reason)
          - DOUBLE_DOWN: add to existing position
          - FLIP: close existing + open opposite
        """
        actions = []
        self.buffer.push(candle)

        # Detect pattern from the last 3 candles
        pattern = detect_pattern(self.buffer, self.min_body_ratio)

        # Get trades for this instrument
        my_trades = [t for t in open_trades if t.instrument == self.instrument]

        # --- Handle open positions first ---
        for trade in my_trades:
            # Check if each new candle breaks the reference level → potential FLIP
            flip_check = self.position_mgr.check_opposing_candle(
                trade, candle.close
            )

            if flip_check == FLIP:
                # Close existing trade, open opposite
                actions.append(Action(
                    "CLOSE",
                    trade_id=trade.id,
                    reason="FLIP_SIGNAL",
                ))
                # Open new trade in opposite direction using current candle
                actions.append(Action(
                    f"OPEN_{'SHORT' if trade.direction == 'LONG' else 'LONG'}",
                    pattern=None,
                    entry_price=candle.close,
                    reference_close=candle.close,
                ))
                continue

            # Check if an opposite pattern has completed → signal exit
            if pattern and self.position_mgr.should_close_on_signal(trade, pattern):
                actions.append(Action(
                    "CLOSE",
                    trade_id=trade.id,
                    reason="OPPOSITE_PATTERN",
                ))
                # Open new trade in the pattern direction
                direction = "LONG" if pattern.direction == "BULLISH" else "SHORT"
                actions.append(Action(
                    f"OPEN_{direction}",
                    pattern=pattern,
                    entry_price=pattern.entry_price,
                    reference_close=pattern.candle_3.close,
                ))
                self.last_pattern = pattern
                self.position_mgr.reset(self.instrument)
                return actions  # Pattern consumed

            # Check for double-down opportunity
            if pattern and self.position_mgr.should_double_down(trade, pattern):
                direction = "LONG" if pattern.direction == "BULLISH" else "SHORT"
                actions.append(Action(
                    "DOUBLE_DOWN",
                    trade_id=trade.id,
                    pattern=pattern,
                    entry_price=pattern.entry_price,
                    reference_close=pattern.candle_3.close,
                ))
                self.position_mgr.consume_double_down(self.instrument)
                self.last_pattern = pattern
                return actions

        # --- No open position: check for new entry ---
        if not my_trades and pattern and pattern != self.last_pattern:
            direction = "LONG" if pattern.direction == "BULLISH" else "SHORT"
            actions.append(Action(
                f"OPEN_{direction}",
                pattern=pattern,
                entry_price=pattern.entry_price,
                reference_close=pattern.candle_3.close,
            ))
            self.last_pattern = pattern

        return actions

    def get_daily_context(
        self, daily_candle: CandleSnapshot
    ) -> str:
        """
        Determine daily trend context from the current daily candle.
        Compares close to previous daily close.
        """
        if self._prev_daily_close is None:
            self._prev_daily_close = daily_candle.close
            return "NEUTRAL"

        if daily_candle.close > self._prev_daily_close:
            context = "BULLISH"
        elif daily_candle.close < self._prev_daily_close:
            context = "BEARISH"
        else:
            context = "NEUTRAL"

        self._prev_daily_close = daily_candle.close
        return context
