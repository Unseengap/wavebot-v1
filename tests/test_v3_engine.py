"""Tests for V3 strategy engine — SwingReversalV3 orchestrator."""
import pytest

from src.strategy.v3.types import CandleSnapshot, V3Trade
from src.strategy.v3.engine import SwingReversalV3


def _candle(o, h, l, c, t="2025-01-01T00:00:00Z", tf="H4"):
    return CandleSnapshot(open=o, high=h, low=l, close=c, time=t, timeframe=tf)


def _trade(trade_id=1, direction="LONG", instrument="EUR_USD", ref_close=1.0960):
    return V3Trade(
        id=trade_id, instrument=instrument, direction=direction,
        entry_price=1.0960, entry_time="2025-01-01T00:00:00Z",
        units=10000, stop_loss=None, take_profit=None,
        signal_frame="H4", confluence_score=0.0, session="",
        sl_distance_pips=50.0, rr_target=0.0, entry_candle_idx=0,
        reference_candle_close=ref_close,
    )


class TestSwingReversalV3:
    def setup_method(self):
        self.engine = SwingReversalV3(
            instrument="EUR_USD",
            pip_size=0.0001,
            min_body_ratio=0.3,
        )

    def test_bullish_pattern_opens_long(self):
        """Full bullish 3-candle reversal → OPEN_LONG action."""
        c1 = _candle(o=1.1000, h=1.1050, l=1.0900, c=1.0920, t="T1")
        c2 = _candle(o=1.0910, h=1.0980, l=1.0890, c=1.0960, t="T2")
        c3 = _candle(o=1.0965, h=1.1020, l=1.0960, c=1.1010, t="T3")

        self.engine.on_h4_candle_close(c1, [])
        self.engine.on_h4_candle_close(c2, [])
        actions = self.engine.on_h4_candle_close(c3, [])

        assert len(actions) == 1
        assert actions[0].type == "OPEN_LONG"
        assert actions[0].data["pattern"].direction == "BULLISH"
        assert actions[0].data["entry_price"] == 1.1010

    def test_bearish_pattern_opens_short(self):
        """Full bearish 3-candle reversal → OPEN_SHORT action."""
        c1 = _candle(o=1.0900, h=1.1050, l=1.0880, c=1.1000, t="T1")
        c2 = _candle(o=1.1010, h=1.1030, l=1.0950, c=1.0960, t="T2")
        c3 = _candle(o=1.0950, h=1.0955, l=1.0900, c=1.0910, t="T3")

        self.engine.on_h4_candle_close(c1, [])
        self.engine.on_h4_candle_close(c2, [])
        actions = self.engine.on_h4_candle_close(c3, [])

        assert len(actions) == 1
        assert actions[0].type == "OPEN_SHORT"

    def test_no_pattern_no_action(self):
        """Candles without a reversal pattern → no actions."""
        c1 = _candle(o=1.1000, h=1.1010, l=1.0990, c=1.1005, t="T1")
        c2 = _candle(o=1.1005, h=1.1015, l=1.0995, c=1.1010, t="T2")
        c3 = _candle(o=1.1010, h=1.1020, l=1.1000, c=1.1015, t="T3")

        self.engine.on_h4_candle_close(c1, [])
        self.engine.on_h4_candle_close(c2, [])
        actions = self.engine.on_h4_candle_close(c3, [])

        assert len(actions) == 0

    def test_opposite_pattern_closes_and_opens(self):
        """Open LONG → bearish pattern → CLOSE + OPEN_SHORT."""
        c1 = _candle(o=1.1000, h=1.1050, l=1.0900, c=1.0920, t="T1")
        c2 = _candle(o=1.0910, h=1.0980, l=1.0890, c=1.0960, t="T2")
        c3 = _candle(o=1.0965, h=1.1020, l=1.0960, c=1.1010, t="T3")

        self.engine.on_h4_candle_close(c1, [])
        self.engine.on_h4_candle_close(c2, [])
        self.engine.on_h4_candle_close(c3, [])

        long_trade = _trade(direction="LONG", ref_close=1.1010)

        c4 = _candle(o=1.1000, h=1.1100, l=1.0980, c=1.1080, t="T4")
        c5 = _candle(o=1.1090, h=1.1095, l=1.1020, c=1.1030, t="T5")
        c6 = _candle(o=1.1025, h=1.1030, l=1.0980, c=1.0990, t="T6")

        self.engine.on_h4_candle_close(c4, [long_trade])
        self.engine.on_h4_candle_close(c5, [long_trade])
        actions = self.engine.on_h4_candle_close(c6, [long_trade])

        action_types = [a.type for a in actions]
        assert "CLOSE" in action_types
        assert "OPEN_SHORT" in action_types

    def test_no_entry_when_already_positioned(self):
        """Same-direction pattern while position open → no duplicate entry."""
        c1 = _candle(o=1.1000, h=1.1050, l=1.0900, c=1.0920, t="T1")
        c2 = _candle(o=1.0910, h=1.0980, l=1.0890, c=1.0960, t="T2")
        c3 = _candle(o=1.0965, h=1.1020, l=1.0960, c=1.1010, t="T3")

        self.engine.on_h4_candle_close(c1, [])
        self.engine.on_h4_candle_close(c2, [])

        long_trade = _trade(direction="LONG")
        actions = self.engine.on_h4_candle_close(c3, [long_trade])

        open_actions = [a for a in actions if a.type.startswith("OPEN_")]
        assert len(open_actions) == 0

    def test_flip_on_candle_close_beyond_reference(self):
        """Candle closes below LONG reference → FLIP (close + open opposite)."""
        long_trade = _trade(direction="LONG", ref_close=1.0960)

        c = _candle(o=1.0950, h=1.0960, l=1.0900, c=1.0910, t="T1")

        actions = self.engine.on_h4_candle_close(c, [long_trade])

        action_types = [a.type for a in actions]
        assert "CLOSE" in action_types
        assert "OPEN_SHORT" in action_types

    def test_daily_context(self):
        """Daily context returns BULLISH/BEARISH/NEUTRAL."""
        d1 = _candle(o=1.0900, h=1.1000, l=1.0880, c=1.0950, t="2025-01-01", tf="D")
        d2 = _candle(o=1.0960, h=1.1050, l=1.0940, c=1.1020, t="2025-01-02", tf="D")
        d3 = _candle(o=1.1010, h=1.1030, l=1.0950, c=1.0960, t="2025-01-03", tf="D")

        ctx1 = self.engine.get_daily_context(d1)
        assert ctx1 == "NEUTRAL"

        ctx2 = self.engine.get_daily_context(d2)
        assert ctx2 == "BULLISH"

        ctx3 = self.engine.get_daily_context(d3)
        assert ctx3 == "BEARISH"

    def test_take_profit_always_none_in_action(self):
        """V3 actions never set a take profit."""
        c1 = _candle(o=1.1000, h=1.1050, l=1.0900, c=1.0920, t="T1")
        c2 = _candle(o=1.0910, h=1.0980, l=1.0890, c=1.0960, t="T2")
        c3 = _candle(o=1.0965, h=1.1020, l=1.0960, c=1.1010, t="T3")

        self.engine.on_h4_candle_close(c1, [])
        self.engine.on_h4_candle_close(c2, [])
        actions = self.engine.on_h4_candle_close(c3, [])

        assert "take_profit" not in actions[0].data
        assert "stop_loss" not in actions[0].data
