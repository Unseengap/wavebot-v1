"""V3 backtest engine — runs Swing Reversal V3 strategy against historical 4H/D candle data."""
import pandas as pd
from typing import Optional

from src.strategy.v3.types import CandleSnapshot, V3Trade
from src.strategy.v3.engine import SwingReversalV3
from src.risk.position_sizer import calculate_position_size
from src.risk.circuit_breaker import CircuitBreaker


class V3BacktestEngine:
    """
    Iterate on 4H candles as the primary loop.
    Track daily context from D candles.
    No stop loss or trailing stop — exits only via opposite signal or flip.
    Output trade dicts compatible with src/backtest/metrics.calculate_metrics().
    """

    def __init__(
        self,
        instrument: str = "EUR_USD",
        pip_size: float = 0.0001,
        pip_value_per_unit: float = 0.0001,
        initial_balance: float = 10000.0,
        risk_fraction: float = 0.01,
        min_body_ratio: float = 0.3,
        max_open_v3_trades: int = 2,
        max_double_downs: int = 1,
        double_down_enabled: bool = True,
        max_daily_drawdown: float = 0.02,
        max_total_drawdown: float = 0.08,
        default_sl_pips: float = 50.0,
    ):
        self.instrument = instrument
        self.pip_size = pip_size
        self.pip_value_per_unit = pip_value_per_unit
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.risk_fraction = risk_fraction
        self.max_open = max_open_v3_trades
        self.default_sl_pips = default_sl_pips

        self.strategy = SwingReversalV3(
            instrument=instrument,
            pip_size=pip_size,
            pip_value=pip_value_per_unit,
            min_body_ratio=min_body_ratio,
            max_double_downs=max_double_downs,
            double_down_enabled=double_down_enabled,
        )

        self.circuit_breaker = CircuitBreaker(
            max_daily_drawdown=max_daily_drawdown,
            max_total_drawdown=max_total_drawdown,
            max_open_trades=max_open_v3_trades,
        )

        self.open_trades: list[V3Trade] = []
        self.closed_trades: list[V3Trade] = []
        self.trade_counter = 0
        self._current_day: Optional[str] = None

    def run(
        self,
        h4_candles: pd.DataFrame,
        d_candles: Optional[pd.DataFrame] = None,
    ) -> list[dict]:
        """
        Run the V3 backtest.

        Args:
            h4_candles: DataFrame with columns [time, open, high, low, close]
            d_candles:  Optional DataFrame with same columns for daily context

        Returns:
            List of closed trade dicts (compatible with metrics.calculate_metrics)
        """
        # Build daily lookup: date_str → CandleSnapshot
        daily_map: dict[str, CandleSnapshot] = {}
        if d_candles is not None and not d_candles.empty:
            for _, row in d_candles.iterrows():
                t = str(row["time"])
                date_key = t[:10]
                daily_map[date_key] = CandleSnapshot(
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    time=t,
                    timeframe="D",
                )

        # Main loop: iterate over 4H candles
        for idx in range(len(h4_candles)):
            row = h4_candles.iloc[idx]
            timestamp = str(row["time"])
            candle = CandleSnapshot(
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                time=timestamp,
                timeframe="H4",
            )

            # Daily reset
            day_str = timestamp[:10]
            if day_str != self._current_day:
                self._current_day = day_str
                self.circuit_breaker.reset_daily(self.balance)
                # Update daily context
                if day_str in daily_map:
                    self.strategy.get_daily_context(daily_map[day_str])

            # 1. Update open trades: track MFE/MAE
            self._update_open_trades(candle)

            # 2. Get strategy actions
            actions = self.strategy.on_h4_candle_close(
                candle, self.open_trades
            )

            # 3. Execute actions
            daily_context = self._get_context_for_day(day_str, daily_map)
            for action in actions:
                self._execute_action(action, candle, candle_idx=idx, daily_context=daily_context)

        # Close remaining trades at end of test
        for trade in list(self.open_trades):
            last_row = h4_candles.iloc[-1]
            self._close_trade(
                trade,
                exit_price=float(last_row["close"]),
                exit_time=str(last_row["time"]),
                reason="END_OF_TEST",
            )

        return [t.to_dict() for t in self.closed_trades]

    def _update_open_trades(
        self, candle: CandleSnapshot,
    ) -> None:
        """Track MFE/MAE for open trades."""
        for trade in list(self.open_trades):
            trade.bars_in_trade += 1

            # Track MFE/MAE
            if trade.direction == "LONG":
                favorable = (candle.high - trade.entry_price) / self.pip_size
                adverse = (trade.entry_price - candle.low) / self.pip_size
            else:
                favorable = (trade.entry_price - candle.low) / self.pip_size
                adverse = (candle.high - trade.entry_price) / self.pip_size

            trade.max_favorable_pips = max(trade.max_favorable_pips, favorable)
            trade.max_adverse_pips = max(trade.max_adverse_pips, adverse)

    def _execute_action(
        self,
        action,
        candle: CandleSnapshot,
        candle_idx: int,
        daily_context: str,
    ) -> None:
        """Execute a strategy action (open/close/double-down/flip)."""
        # Circuit breaker check for new entries
        if action.type.startswith("OPEN_") or action.type == "DOUBLE_DOWN":
            allowed, reason = self.circuit_breaker.check(
                self.balance, len(self.open_trades)
            )
            if not allowed:
                return

        if action.type == "CLOSE":
            trade = self._find_trade(action.data.get("trade_id"))
            if trade:
                self._close_trade(
                    trade, candle.close, candle.time,
                    action.data.get("reason", "SIGNAL_EXIT")
                )

        elif action.type in ("OPEN_LONG", "OPEN_SHORT"):
            direction = "LONG" if action.type == "OPEN_LONG" else "SHORT"
            entry = action.data["entry_price"]
            sl_dist_pips = self.default_sl_pips

            units = calculate_position_size(
                self.balance, self.risk_fraction, sl_dist_pips, self.pip_value_per_unit
            )
            if units <= 0:
                return

            self.trade_counter += 1
            pattern = action.data.get("pattern")
            trade = V3Trade(
                id=self.trade_counter,
                instrument=self.instrument,
                direction=direction,
                entry_price=entry,
                entry_time=candle.time,
                units=units,
                stop_loss=None,
                take_profit=None,
                signal_frame="H4",
                confluence_score=0.0,
                session="",
                sl_distance_pips=sl_dist_pips,
                rr_target=0.0,
                entry_candle_idx=candle_idx,
                pattern=pattern,
                confirmation_type=pattern.confirmation_type if pattern else "",
                daily_context=daily_context,
                reference_candle_close=action.data.get("reference_close", entry),
            )
            self.open_trades.append(trade)

        elif action.type == "DOUBLE_DOWN":
            original = self._find_trade(action.data.get("trade_id"))
            if not original:
                return

            entry = action.data["entry_price"]
            sl_dist_pips = self.default_sl_pips

            units = calculate_position_size(
                self.balance, self.risk_fraction, sl_dist_pips, self.pip_value_per_unit
            )
            if units <= 0:
                return

            self.trade_counter += 1
            pattern = action.data.get("pattern")
            trade = V3Trade(
                id=self.trade_counter,
                instrument=self.instrument,
                direction=original.direction,
                entry_price=entry,
                entry_time=candle.time,
                units=units,
                stop_loss=None,
                take_profit=None,
                signal_frame="H4",
                confluence_score=0.0,
                session="",
                sl_distance_pips=sl_dist_pips,
                rr_target=0.0,
                entry_candle_idx=candle_idx,
                pattern=pattern,
                confirmation_type=pattern.confirmation_type if pattern else "",
                daily_context=daily_context,
                reference_candle_close=action.data.get("reference_close", entry),
                double_down_count=original.double_down_count + 1,
            )
            original.double_down_count += 1
            self.open_trades.append(trade)

    def _close_trade(
        self, trade: V3Trade, exit_price: float, exit_time: str, reason: str
    ) -> None:
        """Close a trade and calculate P&L."""
        trade.exit_price = exit_price
        trade.exit_time = exit_time
        trade.exit_reason = reason

        if trade.direction == "LONG":
            trade.pnl_pips = (exit_price - trade.entry_price) / self.pip_size
        else:
            trade.pnl_pips = (trade.entry_price - exit_price) / self.pip_size

        trade.pnl_dollars = trade.pnl_pips * self.pip_value_per_unit * trade.units
        if trade.sl_distance_pips > 0:
            trade.rr_achieved = trade.pnl_pips / trade.sl_distance_pips

        self.balance += trade.pnl_dollars

        if trade in self.open_trades:
            self.open_trades.remove(trade)
        self.closed_trades.append(trade)

    def _find_trade(self, trade_id: Optional[int]) -> Optional[V3Trade]:
        if trade_id is None:
            return None
        for t in self.open_trades:
            if t.id == trade_id:
                return t
        return None

    def _get_context_for_day(
        self, day_str: str, daily_map: dict[str, CandleSnapshot]
    ) -> str:
        """Get the daily context string for a given day."""
        if day_str in daily_map:
            return self.strategy.get_daily_context(daily_map[day_str])
        return "NEUTRAL"
