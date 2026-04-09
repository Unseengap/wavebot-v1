"""Tests for V3 4H 3-candle reversal pattern detector."""
import pytest

from src.strategy.v3.types import CandleSnapshot
from src.strategy.v3.pattern_detector import (
    CandleBuffer,
    detect_bullish_reversal,
    detect_bearish_reversal,
    detect_pattern,
)


def _candle(o, h, l, c, t="2025-01-01T00:00:00Z", tf="H4"):
    return CandleSnapshot(open=o, high=h, low=l, close=c, time=t, timeframe=tf)


# --- CandleBuffer ---

class TestCandleBuffer:
    def test_not_ready_when_empty(self):
        buf = CandleBuffer(size=3)
        assert not buf.is_ready()

    def test_ready_after_3_pushes(self):
        buf = CandleBuffer(size=3)
        for i in range(3):
            buf.push(_candle(1, 2, 0.5, 1.5))
        assert buf.is_ready()

    def test_rolling_window(self):
        buf = CandleBuffer(size=3)
        for i in range(5):
            buf.push(_candle(i, i + 1, i - 0.5, i + 0.5, t=f"T{i}"))
        assert buf.is_ready()
        assert buf.candles[0].time == "T2"
        assert buf.candles[2].time == "T4"

    def test_clear(self):
        buf = CandleBuffer(size=3)
        for i in range(3):
            buf.push(_candle(1, 2, 0.5, 1.5))
        buf.clear()
        assert not buf.is_ready()


# --- Bullish Reversal ---

class TestBullishReversal:
    def test_clean_bullish_reversal(self):
        # C1: bearish (close < open), good body
        c1 = _candle(o=1.1000, h=1.1050, l=1.0900, c=1.0920)
        # C2: closes above C1.close
        c2 = _candle(o=1.0910, h=1.0980, l=1.0890, c=1.0960)
        # C3: closes above C2.close, no dip below C2.close
        c3 = _candle(o=1.0965, h=1.1020, l=1.0960, c=1.1010)

        pat = detect_bullish_reversal(c1, c2, c3)
        assert pat is not None
        assert pat.direction == "BULLISH"
        assert pat.confirmation_type == "CLEAN"
        assert pat.pattern_low == min(c1.low, c2.low, c3.low)
        assert pat.pattern_high == max(c1.high, c2.high, c3.high)

    def test_rejection_bullish_reversal(self):
        # C1: bearish
        c1 = _candle(o=1.1000, h=1.1050, l=1.0900, c=1.0920)
        # C2: closes above C1.close
        c2 = _candle(o=1.0910, h=1.0980, l=1.0890, c=1.0960)
        # C3: dips below C2.close (low < 1.0960) but closes >= C2.close
        c3 = _candle(o=1.0940, h=1.1020, l=1.0930, c=1.0970)

        pat = detect_bullish_reversal(c1, c2, c3)
        assert pat is not None
        assert pat.confirmation_type == "REJECTION"

    def test_failed_pattern_c3_below_c2(self):
        c1 = _candle(o=1.1000, h=1.1050, l=1.0900, c=1.0920)
        c2 = _candle(o=1.0910, h=1.0980, l=1.0890, c=1.0960)
        # C3 closes below C2's close
        c3 = _candle(o=1.0940, h=1.0960, l=1.0900, c=1.0910)

        pat = detect_bullish_reversal(c1, c2, c3)
        assert pat is None

    def test_c1_must_be_bearish(self):
        # C1 is bullish → no pattern
        c1 = _candle(o=1.0900, h=1.1050, l=1.0880, c=1.1000)
        c2 = _candle(o=1.0990, h=1.1010, l=1.0980, c=1.1005)
        c3 = _candle(o=1.1010, h=1.1050, l=1.1000, c=1.1040)

        assert detect_bullish_reversal(c1, c2, c3) is None

    def test_c2_must_close_above_c1(self):
        c1 = _candle(o=1.1000, h=1.1050, l=1.0900, c=1.0920)
        # C2 closes at C1's close (not above)
        c2 = _candle(o=1.0910, h=1.0980, l=1.0890, c=1.0920)
        c3 = _candle(o=1.0940, h=1.1020, l=1.0930, c=1.0970)

        assert detect_bullish_reversal(c1, c2, c3) is None

    def test_doji_c1_rejected(self):
        # C1 is a doji (body_ratio < 0.3)
        c1 = _candle(o=1.1000, h=1.1100, l=1.0900, c=1.0990)  # body=0.001, range=0.02 → 5%
        c2 = _candle(o=1.0980, h=1.1020, l=1.0960, c=1.1010)
        c3 = _candle(o=1.1010, h=1.1050, l=1.1000, c=1.1040)

        assert detect_bullish_reversal(c1, c2, c3) is None

    def test_entry_price_is_c3_close(self):
        c1 = _candle(o=1.1000, h=1.1050, l=1.0900, c=1.0920)
        c2 = _candle(o=1.0910, h=1.0980, l=1.0890, c=1.0960)
        c3 = _candle(o=1.0965, h=1.1020, l=1.0960, c=1.1010)

        pat = detect_bullish_reversal(c1, c2, c3)
        assert pat.entry_price == 1.1010

    def test_initial_sl_is_pattern_low(self):
        c1 = _candle(o=1.1000, h=1.1050, l=1.0900, c=1.0920)
        c2 = _candle(o=1.0910, h=1.0980, l=1.0890, c=1.0960)
        c3 = _candle(o=1.0965, h=1.1020, l=1.0960, c=1.1010)

        pat = detect_bullish_reversal(c1, c2, c3)
        assert pat.initial_sl == 1.0890  # min of all 3 lows


# --- Bearish Reversal ---

class TestBearishReversal:
    def test_clean_bearish_reversal(self):
        # C1: bullish (close > open)
        c1 = _candle(o=1.0900, h=1.1050, l=1.0880, c=1.1000)
        # C2: closes below C1.close
        c2 = _candle(o=1.1010, h=1.1030, l=1.0950, c=1.0960)
        # C3: closes below C2.close, no wick above C2.close
        c3 = _candle(o=1.0950, h=1.0955, l=1.0900, c=1.0910)

        pat = detect_bearish_reversal(c1, c2, c3)
        assert pat is not None
        assert pat.direction == "BEARISH"
        assert pat.confirmation_type == "CLEAN"
        assert pat.pattern_high == max(c1.high, c2.high, c3.high)

    def test_rejection_bearish_reversal(self):
        c1 = _candle(o=1.0900, h=1.1050, l=1.0880, c=1.1000)
        c2 = _candle(o=1.1010, h=1.1030, l=1.0950, c=1.0960)
        # C3: spikes above C2.close (high > 1.0960) but closes <= C2.close
        c3 = _candle(o=1.0970, h=1.0980, l=1.0900, c=1.0950)

        pat = detect_bearish_reversal(c1, c2, c3)
        assert pat is not None
        assert pat.confirmation_type == "REJECTION"

    def test_failed_bearish_c3_above_c2(self):
        c1 = _candle(o=1.0900, h=1.1050, l=1.0880, c=1.1000)
        c2 = _candle(o=1.1010, h=1.1030, l=1.0950, c=1.0960)
        # C3 closes above C2's close
        c3 = _candle(o=1.0970, h=1.1010, l=1.0960, c=1.1000)

        assert detect_bearish_reversal(c1, c2, c3) is None

    def test_c1_must_be_bullish_for_bearish(self):
        c1 = _candle(o=1.1000, h=1.1050, l=1.0900, c=1.0920)  # bearish
        c2 = _candle(o=1.0910, h=1.0920, l=1.0850, c=1.0860)
        c3 = _candle(o=1.0870, h=1.0875, l=1.0820, c=1.0830)

        assert detect_bearish_reversal(c1, c2, c3) is None

    def test_initial_sl_is_pattern_high(self):
        c1 = _candle(o=1.0900, h=1.1050, l=1.0880, c=1.1000)
        c2 = _candle(o=1.1010, h=1.1030, l=1.0950, c=1.0960)
        c3 = _candle(o=1.0950, h=1.0955, l=1.0900, c=1.0910)

        pat = detect_bearish_reversal(c1, c2, c3)
        assert pat.initial_sl == 1.1050  # max of all 3 highs


# --- detect_pattern (combined) ---

class TestDetectPattern:
    def test_detects_bullish_from_buffer(self):
        buf = CandleBuffer(size=3)
        buf.push(_candle(o=1.1000, h=1.1050, l=1.0900, c=1.0920))
        buf.push(_candle(o=1.0910, h=1.0980, l=1.0890, c=1.0960))
        buf.push(_candle(o=1.0965, h=1.1020, l=1.0960, c=1.1010))

        pat = detect_pattern(buf)
        assert pat is not None
        assert pat.direction == "BULLISH"

    def test_detects_bearish_from_buffer(self):
        buf = CandleBuffer(size=3)
        buf.push(_candle(o=1.0900, h=1.1050, l=1.0880, c=1.1000))
        buf.push(_candle(o=1.1010, h=1.1030, l=1.0950, c=1.0960))
        buf.push(_candle(o=1.0950, h=1.0955, l=1.0900, c=1.0910))

        pat = detect_pattern(buf)
        assert pat is not None
        assert pat.direction == "BEARISH"

    def test_returns_none_when_no_pattern(self):
        buf = CandleBuffer(size=3)
        # Three flat-ish candles, no clear reversal
        buf.push(_candle(o=1.1000, h=1.1010, l=1.0990, c=1.1005))
        buf.push(_candle(o=1.1005, h=1.1015, l=1.0995, c=1.1010))
        buf.push(_candle(o=1.1010, h=1.1020, l=1.1000, c=1.1015))

        assert detect_pattern(buf) is None

    def test_returns_none_when_buffer_not_ready(self):
        buf = CandleBuffer(size=3)
        buf.push(_candle(o=1.1000, h=1.1050, l=1.0900, c=1.0920))
        assert detect_pattern(buf) is None

    def test_long_wicks_dont_break_pattern(self):
        # C1 bearish with long lower wick
        c1 = _candle(o=1.1000, h=1.1050, l=1.0800, c=1.0920)
        # C2 closes above C1
        c2 = _candle(o=1.0910, h=1.1100, l=1.0890, c=1.0960)
        # C3 clean continuation
        c3 = _candle(o=1.0965, h=1.1200, l=1.0960, c=1.1010)

        buf = CandleBuffer(size=3)
        buf.push(c1)
        buf.push(c2)
        buf.push(c3)

        pat = detect_pattern(buf)
        assert pat is not None
        assert pat.direction == "BULLISH"
        assert pat.pattern_low == 1.0800
        assert pat.pattern_high == 1.1200
