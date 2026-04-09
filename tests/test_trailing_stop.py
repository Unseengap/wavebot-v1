"""Tests for V3 ATR-based trailing stop."""
import numpy as np
import pytest

from src.strategy.v3.trailing_stop import calculate_atr, TrailingStopManager


# --- ATR Calculation ---

class TestCalculateATR:
    def test_basic_atr(self):
        highs = np.array([1.10, 1.12, 1.11, 1.13, 1.14])
        lows = np.array([1.08, 1.09, 1.09, 1.10, 1.11])
        closes = np.array([1.09, 1.11, 1.10, 1.12, 1.13])

        atr = calculate_atr(highs, lows, closes, period=3)
        assert atr > 0

    def test_atr_returns_zero_insufficient_data(self):
        assert calculate_atr(np.array([1.0]), np.array([0.9]), np.array([0.95]), period=14) == 0.0

    def test_atr_uses_last_n_periods(self):
        # 20 candles, period=5
        np.random.seed(42)
        base = 1.1000
        highs = base + np.random.uniform(0.001, 0.01, 20)
        lows = base - np.random.uniform(0.001, 0.01, 20)
        closes = (highs + lows) / 2

        atr_5 = calculate_atr(highs, lows, closes, period=5)
        atr_10 = calculate_atr(highs, lows, closes, period=10)
        # Both should be positive
        assert atr_5 > 0
        assert atr_10 > 0

    def test_atr_with_gaps(self):
        # Simulates a gap: close[i-1] far from high[i]/low[i]
        highs = np.array([1.10, 1.15, 1.14, 1.16])
        lows = np.array([1.08, 1.12, 1.11, 1.13])
        closes = np.array([1.09, 1.14, 1.13, 1.15])

        atr = calculate_atr(highs, lows, closes, period=3)
        assert atr > 0


# --- TrailingStopManager ---

class TestTrailingStopManager:
    def setup_method(self):
        self.tsm = TrailingStopManager(multiplier=2.0)

    def test_initial_stop_long(self):
        sl = self.tsm.initial_stop("LONG", 1.0900, buffer_pips=2.0, pip_size=0.0001)
        assert sl == pytest.approx(1.0898, abs=1e-6)

    def test_initial_stop_short(self):
        sl = self.tsm.initial_stop("SHORT", 1.1050, buffer_pips=2.0, pip_size=0.0001)
        assert sl == pytest.approx(1.1052, abs=1e-6)

    def test_trailing_long_moves_up(self):
        """LONG trailing stop should move up as price rises."""
        current_stop = 1.0900
        atr = 0.0020  # 20 pips

        # Price goes up → stop should follow
        new_sl = self.tsm.update("LONG", 1.1000, atr, current_stop)
        assert new_sl > current_stop
        expected = max(current_stop, 1.1000 - 2.0 * 0.0020)
        assert new_sl == pytest.approx(expected)

    def test_trailing_long_never_moves_down(self):
        """LONG stop must never decrease."""
        stops = []
        current_stop = 1.0900
        prices = [1.1000, 1.1050, 1.1020, 1.0980, 1.1100]
        atr = 0.0020

        for price in prices:
            current_stop = self.tsm.update("LONG", price, atr, current_stop)
            stops.append(current_stop)

        # Each stop must be >= the previous one
        for i in range(1, len(stops)):
            assert stops[i] >= stops[i - 1], f"Stop moved backward at index {i}"

    def test_trailing_short_moves_down(self):
        """SHORT trailing stop should move down as price falls."""
        current_stop = 1.1100
        atr = 0.0020

        new_sl = self.tsm.update("SHORT", 1.1000, atr, current_stop)
        assert new_sl < current_stop
        expected = min(current_stop, 1.1000 + 2.0 * 0.0020)
        assert new_sl == pytest.approx(expected)

    def test_trailing_short_never_moves_up(self):
        """SHORT stop must never increase."""
        stops = []
        current_stop = 1.1100
        prices = [1.1000, 1.0950, 1.0980, 1.1020, 1.0900]
        atr = 0.0020

        for price in prices:
            current_stop = self.tsm.update("SHORT", price, atr, current_stop)
            stops.append(current_stop)

        for i in range(1, len(stops)):
            assert stops[i] <= stops[i - 1], f"Stop moved upward at index {i}"

    def test_zero_atr_no_change(self):
        current_stop = 1.0900
        new_sl = self.tsm.update("LONG", 1.1000, 0.0, current_stop)
        assert new_sl == current_stop

    def test_negative_atr_no_change(self):
        current_stop = 1.0900
        new_sl = self.tsm.update("LONG", 1.1000, -0.001, current_stop)
        assert new_sl == current_stop

    def test_custom_multiplier(self):
        tsm_tight = TrailingStopManager(multiplier=1.0)
        tsm_wide = TrailingStopManager(multiplier=3.0)
        atr = 0.0020
        price = 1.1000
        stop = 1.0900

        tight = tsm_tight.update("LONG", price, atr, stop)
        wide = tsm_wide.update("LONG", price, atr, stop)

        # Tighter multiplier → stop closer to price → higher
        assert tight > wide
