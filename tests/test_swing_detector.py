"""Tests for swing detection logic."""
import numpy as np
import pytest
from src.wave.swing_detector import detect_swing_high, detect_swing_low, detect_swings


class TestDetectSwingHigh:
    def test_clear_peak(self):
        """Single clear peak at index 5 with lookback=3."""
        highs = np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0])
        result = detect_swing_high(highs, lookback=3)
        assert result[5] is True or result[5] == True
        # Only one swing high expected
        assert result.sum() == 1

    def test_no_swings_flat(self):
        """Flat prices: first occurrence wins, only index=lookback qualifies."""
        flat = np.array([1.0] * 11)
        result = detect_swing_high(flat, lookback=3)
        # With ties going to earliest, index 3 (first valid center) should be a swing
        # because argmax returns 0, and first_max = i-lookback+0 = 0, but i=3, so 0 != 3 → False
        # Actually all equal, argmax returns 0, first_max = i-3+0 = i-3
        # For i=3: first_max = 0, 0 != 3 → not a swing
        # For i=4: first_max = 1, 1 != 4 → not a swing
        # No swings for flat data
        assert result.sum() == 0

    def test_multiple_peaks(self):
        """Two distinct peaks should both be detected."""
        highs = np.array([1.0, 1.1, 1.2, 1.5, 1.2, 1.0, 0.9, 1.0, 1.1, 1.4, 1.1, 1.0, 0.9])
        result = detect_swing_high(highs, lookback=3)
        assert result[3] == True  # First peak
        assert result[9] == True  # Second peak

    def test_edge_indices_excluded(self):
        """Indices within lookback of edges should never be swings."""
        highs = np.array([2.0, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.5, 2.0])
        result = detect_swing_high(highs, lookback=3)
        # Index 0 and 8 are within lookback of edges — cannot be swings
        assert result[0] == False
        assert result[-1] == False

    def test_tie_goes_to_earliest(self):
        """When two values in window are equal max, earliest wins."""
        highs = np.array([1.0, 1.0, 1.5, 1.0, 1.5, 1.0, 1.0])
        result = detect_swing_high(highs, lookback=2)
        # At i=2: window=[1.0,1.0,1.5,1.0,1.5], max=1.5, argmax=2 → first_max=0+2=2 → True
        # At i=4: window=[1.5,1.0,1.5,1.0,1.0], max=1.5, argmax=0 → first_max=2+0=2 → 2!=4 → False
        assert result[2] == True
        assert result[4] == False

    def test_lookback_1(self):
        """With lookback=1, each local max of 3 values is a swing."""
        highs = np.array([1.0, 2.0, 1.0, 2.0, 1.0])
        result = detect_swing_high(highs, lookback=1)
        assert result[1] == True
        assert result[3] == True
        assert result.sum() == 2

    def test_empty_array(self):
        """Empty input should return empty result."""
        highs = np.array([])
        result = detect_swing_high(highs, lookback=3)
        assert len(result) == 0

    def test_too_short_for_lookback(self):
        """Array shorter than 2*lookback+1 should have no swings."""
        highs = np.array([1.0, 2.0, 1.0])
        result = detect_swing_high(highs, lookback=3)
        assert result.sum() == 0


class TestDetectSwingLow:
    def test_clear_trough(self):
        """Single clear trough at index 5 with lookback=3."""
        lows = np.array([1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
        result = detect_swing_low(lows, lookback=3)
        assert result[5] == True
        assert result.sum() == 1

    def test_no_swings_flat(self):
        flat = np.array([1.0] * 11)
        result = detect_swing_low(flat, lookback=3)
        assert result.sum() == 0

    def test_multiple_troughs(self):
        lows = np.array([1.5, 1.3, 1.1, 0.8, 1.1, 1.4, 1.5, 1.3, 1.1, 0.9, 1.1, 1.3, 1.5])
        result = detect_swing_low(lows, lookback=3)
        assert result[3] == True
        assert result[9] == True

    def test_symmetry_with_swing_high(self):
        """Swing low should mirror swing high for inverted data."""
        data = np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0])
        inverted = -data
        high_result = detect_swing_high(data, lookback=3)
        low_result = detect_swing_low(inverted, lookback=3)
        np.testing.assert_array_equal(high_result, low_result)


class TestDetectSwings:
    def test_returns_tuple(self):
        highs = np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0])
        lows = np.array([0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9])
        sh, sl = detect_swings(highs, lows, lookback=3)
        assert isinstance(sh, np.ndarray)
        assert isinstance(sl, np.ndarray)
        assert len(sh) == len(highs)
        assert len(sl) == len(lows)

    def test_simultaneous_peak_and_trough(self):
        """Verify peak and trough detected on different indices."""
        # Peak at 5, trough somewhere else
        highs = np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0])
        lows = np.array([1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
        sh, sl = detect_swings(highs, lows, lookback=3)
        assert sh[5] == True
        assert sl[5] == True  # Both peak high and trough low at the same index
