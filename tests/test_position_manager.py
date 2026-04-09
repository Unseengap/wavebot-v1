"""Tests for V3 position manager — double-down and flip logic."""
import pytest

from src.strategy.v3.types import CandleSnapshot, ReversalPattern, V3Trade
from src.strategy.v3.position_manager import V3PositionManager, HOLD, FLIP, DOUBLE_DOWN


def _candle(o, h, l, c, t="2025-01-01T00:00:00Z"):
    return CandleSnapshot(open=o, high=h, low=l, close=c, time=t, timeframe="H4")


def _pattern(direction, c3_close=1.1000):
    c1 = _candle(1.1000, 1.1050, 1.0900, 1.0920)
    c2 = _candle(1.0910, 1.0980, 1.0890, 1.0960)
    c3 = _candle(1.0965, 1.1020, 1.0960, c3_close)
    return ReversalPattern(
        direction=direction, candle_1=c1, candle_2=c2, candle_3=c3,
        pattern_low=1.0890, pattern_high=1.1050,
        confirmation_type="CLEAN", detected_time=c3.time,
    )


def _trade(direction="LONG", instrument="EUR_USD", ref_close=1.0960, dd_count=0):
    return V3Trade(
        id=1, instrument=instrument, direction=direction,
        entry_price=1.0960, entry_time="2025-01-01T00:00:00Z",
        units=10000, stop_loss=1.0890, take_profit=None,
        signal_frame="H4", confluence_score=0.0, session="",
        sl_distance_pips=70.0, rr_target=0.0, entry_candle_idx=0,
        reference_candle_close=ref_close, double_down_count=dd_count,
    )


class TestCheckOpposingCandle:
    def test_long_candle_closes_below_reference_returns_flip(self):
        pm = V3PositionManager()
        trade = _trade("LONG", ref_close=1.0960)
        result = pm.check_opposing_candle(trade, candle_close=1.0940)
        assert result == FLIP

    def test_long_candle_stays_above_reference_returns_hold(self):
        pm = V3PositionManager()
        trade = _trade("LONG", ref_close=1.0960)
        result = pm.check_opposing_candle(trade, candle_close=1.0980)
        assert result == HOLD

    def test_short_candle_closes_above_reference_returns_flip(self):
        pm = V3PositionManager()
        trade = _trade("SHORT", ref_close=1.0960)
        result = pm.check_opposing_candle(trade, candle_close=1.0980)
        assert result == FLIP

    def test_short_candle_stays_below_reference_returns_hold(self):
        pm = V3PositionManager()
        trade = _trade("SHORT", ref_close=1.0960)
        result = pm.check_opposing_candle(trade, candle_close=1.0940)
        assert result == HOLD


class TestDoubleDown:
    def test_should_double_down_after_failed_reversal(self):
        pm = V3PositionManager(max_double_downs=1, double_down_enabled=True)
        trade = _trade("LONG")
        pm.mark_failed_reversal("EUR_USD")

        pattern = _pattern("BULLISH")
        assert pm.should_double_down(trade, pattern) is True

    def test_no_double_down_without_failed_reversal(self):
        pm = V3PositionManager(max_double_downs=1, double_down_enabled=True)
        trade = _trade("LONG")

        pattern = _pattern("BULLISH")
        assert pm.should_double_down(trade, pattern) is False

    def test_no_double_down_wrong_direction(self):
        pm = V3PositionManager(max_double_downs=1, double_down_enabled=True)
        trade = _trade("LONG")
        pm.mark_failed_reversal("EUR_USD")

        pattern = _pattern("BEARISH")  # Wrong direction for LONG trade
        assert pm.should_double_down(trade, pattern) is False

    def test_max_double_downs_respected(self):
        pm = V3PositionManager(max_double_downs=1, double_down_enabled=True)
        trade = _trade("LONG", dd_count=1)  # Already doubled down once
        pm.mark_failed_reversal("EUR_USD")

        pattern = _pattern("BULLISH")
        assert pm.should_double_down(trade, pattern) is False

    def test_double_down_disabled(self):
        pm = V3PositionManager(max_double_downs=1, double_down_enabled=False)
        trade = _trade("LONG")
        pm.mark_failed_reversal("EUR_USD")

        pattern = _pattern("BULLISH")
        assert pm.should_double_down(trade, pattern) is False

    def test_consume_clears_flag(self):
        pm = V3PositionManager(max_double_downs=1, double_down_enabled=True)
        trade = _trade("LONG")
        pm.mark_failed_reversal("EUR_USD")
        pm.consume_double_down("EUR_USD")

        pattern = _pattern("BULLISH")
        assert pm.should_double_down(trade, pattern) is False


class TestShouldCloseOnSignal:
    def test_close_long_on_bearish_signal(self):
        pm = V3PositionManager()
        trade = _trade("LONG")
        pattern = _pattern("BEARISH")
        assert pm.should_close_on_signal(trade, pattern) is True

    def test_close_short_on_bullish_signal(self):
        pm = V3PositionManager()
        trade = _trade("SHORT")
        pattern = _pattern("BULLISH")
        assert pm.should_close_on_signal(trade, pattern) is True

    def test_no_close_same_direction(self):
        pm = V3PositionManager()
        trade = _trade("LONG")
        pattern = _pattern("BULLISH")
        assert pm.should_close_on_signal(trade, pattern) is False

    def test_no_close_on_none_pattern(self):
        pm = V3PositionManager()
        trade = _trade("LONG")
        assert pm.should_close_on_signal(trade, None) is False


class TestReset:
    def test_reset_clears_pending(self):
        pm = V3PositionManager()
        pm.mark_failed_reversal("EUR_USD")
        pm.reset("EUR_USD")
        trade = _trade("LONG")
        pattern = _pattern("BULLISH")
        assert pm.should_double_down(trade, pattern) is False

    def test_reset_different_instrument_unaffected(self):
        pm = V3PositionManager(max_double_downs=1, double_down_enabled=True)
        pm.mark_failed_reversal("EUR_USD")
        pm.reset("GBP_USD")
        trade = _trade("LONG", instrument="EUR_USD")
        pattern = _pattern("BULLISH")
        assert pm.should_double_down(trade, pattern) is True
