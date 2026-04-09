"""Swing Reversal V3 strategy engine — orchestrates pattern detection, trade management, and exits."""
from typing import Optional

from src.strategy.v3.types import CandleSnapshot, ReversalPattern, V3Trade
from src.strategy.v3.pattern_detector import CandleBuffer, detect_pattern
from src.strategy.v3.trailing_stop import TrailingStopManager
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
      3. Check open positions for trailing stop updates
      4. Check for exit signals (opposite pattern) or double-down opportunities
      5. Return list of actions to execute
    """

    def __init__(
        self,
        instrument: str,
        pip_size: float,
        pip_value: float = 0.0001,
        atr_period: int = 14,
        atr_multiplier: float = 2.0,
        sl_buffer_pips: float = 2.0,
        min_body_ratio: float = 0.3,
        max_double_downs: int = 1,
        double_down_enabled: bool = True,
    ):
        self.instrument = instrument
        self.pip_size = pip_size
        self.pip_value = pip_value
        self.atr_period = atr_period
        self.sl_buffer_pips = sl_buffer_pips
        self.min_body_ratio = min_body_ratio

        self.buffer = CandleBuffer(size=3)
        self.trailing = TrailingStopManager(multiplier=atr_multiplier)
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
        current_atr: float,
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
                new_dir = "BEARISH" if trade.direction == "LONG" else "BULLISH"
                actions.append(Action(
                    "CLOSE",
                    trade_id=trade.id,
                    reason="FLIP_SIGNAL",
                ))
                # Open new trade in opposite direction using current candle
                sl = self._calc_initial_sl(new_dir, candle, current_atr)
                actions.append(Action(
                    f"OPEN_{'SHORT' if trade.direction == 'LONG' else 'LONG'}",
                    pattern=None,
                    entry_price=candle.close,
                    stop_loss=sl,
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
                sl = self._calc_initial_sl_from_pattern(pattern, current_atr)
                direction = "LONG" if pattern.direction == "BULLISH" else "SHORT"
                actions.append(Action(
                    f"OPEN_{direction}",
                    pattern=pattern,
                    entry_price=pattern.entry_price,
                    stop_loss=sl,
                    reference_close=pattern.candle_3.close,
                ))
                self.last_pattern = pattern
                self.position_mgr.reset(self.instrument)
                return actions  # Pattern consumed

            # Check for double-down opportunity
            if pattern and self.position_mgr.should_double_down(trade, pattern):
                sl = self._calc_initial_sl_from_pattern(pattern, current_atr)
                direction = "LONG" if pattern.direction == "BULLISH" else "SHORT"
                actions.append(Action(
                    "DOUBLE_DOWN",
                    trade_id=trade.id,
                    pattern=pattern,
                    entry_price=pattern.entry_price,
                    stop_loss=sl,
                    reference_close=pattern.candle_3.close,
                ))
                self.position_mgr.consume_double_down(self.instrument)
                self.last_pattern = pattern
                return actions

        # --- No open position: check for new entry ---
        if not my_trades and pattern and pattern != self.last_pattern:
            sl = self._calc_initial_sl_from_pattern(pattern, current_atr)
            direction = "LONG" if pattern.direction == "BULLISH" else "SHORT"
            actions.append(Action(
                f"OPEN_{direction}",
                pattern=pattern,
                entry_price=pattern.entry_price,
                stop_loss=sl,
                reference_close=pattern.candle_3.close,
            ))
            self.last_pattern = pattern

        return actions

    def update_trailing_stops(
        self,
        current_price: float,
        current_atr: float,
        open_trades: list[V3Trade],
    ) -> dict[int, float]:
        """
        Update trailing stops for all open V3 trades on this instrument.
        Called on every H1 candle (or more frequently in live).
        Returns {trade_id: new_stop_loss}.
        """
        updates = {}
        for trade in open_trades:
            if trade.instrument != self.instrument:
                continue
            new_sl = self.trailing.update(
                direction=trade.direction,
                current_price=current_price,
                current_atr=current_atr,
                current_stop=trade.trailing_sl if trade.trailing_sl else trade.stop_loss,
            )
            if new_sl != trade.trailing_sl:
                updates[trade.id] = new_sl
        return updates

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

    def _calc_initial_sl_from_pattern(
        self, pattern: ReversalPattern, current_atr: float
    ) -> float:
        """Calculate initial SL from pattern extreme."""
        if pattern.direction == "BULLISH":
            return self.trailing.initial_stop(
                "LONG", pattern.pattern_low, self.sl_buffer_pips, self.pip_size
            )
        else:
            return self.trailing.initial_stop(
                "SHORT", pattern.pattern_high, self.sl_buffer_pips, self.pip_size
            )

    def _calc_initial_sl(
        self, direction: str, candle: CandleSnapshot, current_atr: float
    ) -> float:
        """Calculate initial SL from a single candle (used in flip without full pattern)."""
        if direction == "BULLISH":
            return self.trailing.initial_stop(
                "LONG", candle.low, self.sl_buffer_pips, self.pip_size
            )
        else:
            return self.trailing.initial_stop(
                "SHORT", candle.high, self.sl_buffer_pips, self.pip_size
            )
