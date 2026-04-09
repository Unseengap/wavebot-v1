"""Tests for wave scoring: direction, conviction, maturity, clarity."""
import pytest
from unittest.mock import MagicMock
from src.wave.wave_scorer import (
    calculate_direction, calculate_conviction, calculate_maturity,
    calculate_swing_clarity, score_wave, WaveScore, DIRECTION_MAP,
)


class TestCalculateDirection:
    def test_all_states(self):
        assert calculate_direction("BULLISH_IMPULSE") == 1.0
        assert calculate_direction("BULLISH_CORRECTION") == 0.5
        assert calculate_direction("RANGING") == 0.0
        assert calculate_direction("BEARISH_CORRECTION") == -0.5
        assert calculate_direction("BEARISH_IMPULSE") == -1.0

    def test_unknown_state(self):
        assert calculate_direction("UNKNOWN") == 0.0

    def test_all_map_entries_covered(self):
        for state, expected in DIRECTION_MAP.items():
            assert calculate_direction(state) == expected


class TestCalculateConviction:
    def test_ranging_always_zero(self):
        assert calculate_conviction("RANGING", 0.5, 0.8, 3) == 0.0

    def test_output_bounds(self):
        """Conviction must be in [0.0, 1.0]."""
        for state in ["BULLISH_IMPULSE", "BEARISH_IMPULSE",
                       "BULLISH_CORRECTION", "BEARISH_CORRECTION"]:
            for mat in [0.0, 0.5, 1.0]:
                for clarity in [0.0, 0.5, 1.0]:
                    for swings in [0, 3, 10]:
                        c = calculate_conviction(state, mat, clarity, swings)
                        assert 0.0 <= c <= 1.0, f"Out of bounds: {c} for {state},{mat},{clarity},{swings}"

    def test_low_maturity_higher_conviction(self):
        """Fresh wave (low maturity) should have higher conviction than exhausted."""
        fresh = calculate_conviction("BULLISH_IMPULSE", 0.1, 0.8, 3)
        tired = calculate_conviction("BULLISH_IMPULSE", 0.9, 0.8, 3)
        assert fresh > tired

    def test_high_clarity_higher_conviction(self):
        c_high = calculate_conviction("BULLISH_IMPULSE", 0.3, 1.0, 3)
        c_low = calculate_conviction("BULLISH_IMPULSE", 0.3, 0.0, 3)
        assert c_high > c_low

    def test_more_swings_higher_conviction(self):
        c_many = calculate_conviction("BEARISH_IMPULSE", 0.3, 0.8, 5)
        c_few = calculate_conviction("BEARISH_IMPULSE", 0.3, 0.8, 0)
        assert c_many > c_few


class TestCalculateMaturity:
    def test_zero_median_returns_half(self):
        m = calculate_maturity(10.0, 5, {"p50": 0}, {"p50_candles": 0})
        assert m == 0.5

    def test_fresh_wave(self):
        m = calculate_maturity(5.0, 2, {"p50": 50}, {"p50_candles": 40})
        assert m < 0.2

    def test_exhausted_wave(self):
        m = calculate_maturity(100.0, 80, {"p50": 50}, {"p50_candles": 40})
        assert m == 1.0

    def test_dual_axis_takes_max(self):
        """Maturity is max(price_maturity, time_maturity)."""
        # High price maturity, low time maturity
        m = calculate_maturity(80.0, 5, {"p50": 50}, {"p50_candles": 100})
        price_mat = min(1.0, (80.0 / 50) * 0.6)
        time_mat = min(1.0, (5 / 100) * 0.6)
        assert m == round(max(price_mat, time_mat), 4)

    def test_output_bounds(self):
        for pips in [0, 10, 100, 1000]:
            for age in [0, 5, 50, 500]:
                m = calculate_maturity(pips, age, {"p50": 30}, {"p50_candles": 20})
                assert 0.0 <= m <= 1.0


class TestCalculateSwingClarity:
    def test_bullish_clear_structure(self):
        """HH and HL = clarity 1.0."""
        c = calculate_swing_clarity(1.5, 1.4, 1.1, 1.0, "BULLISH_IMPULSE")
        assert c == 1.0

    def test_bullish_unclear(self):
        """HH but no HL = 0.6."""
        c = calculate_swing_clarity(1.5, 1.4, 0.9, 1.0, "BULLISH_IMPULSE")
        assert c == 0.6

    def test_bullish_missing_data(self):
        c = calculate_swing_clarity(1.5, None, 1.1, 1.0, "BULLISH_IMPULSE")
        assert c == 0.5

    def test_bearish_clear_structure(self):
        """LH and LL = clarity 1.0."""
        c = calculate_swing_clarity(1.3, 1.4, 0.9, 1.0, "BEARISH_IMPULSE")
        assert c == 1.0

    def test_ranging_always_low(self):
        c = calculate_swing_clarity(1.5, 1.4, 1.1, 1.0, "RANGING")
        assert c == 0.3


class TestScoreWave:
    def test_returns_wave_score(self):
        machine = MagicMock()
        machine.state = "BULLISH_IMPULSE"
        machine.wave_origin = 1.1000
        machine.wave_peak = 1.1050
        machine.wave_trough = None
        machine.candle_idx = 20
        machine.wave_start_idx = 10
        machine.last_swing_high = 1.1050
        machine.prev_swing_high = 1.1030
        machine.last_swing_low = 1.1010
        machine.prev_swing_low = 1.0990
        machine.consecutive_swings = 3

        ws = score_wave(
            "EUR_USD", "M5", "2025-01-01T12:00:00Z",
            machine, {"p50": 25}, {"p50_candles": 20}, 0.0001,
        )
        assert isinstance(ws, WaveScore)
        assert ws.direction == 1.0
        assert ws.instrument == "EUR_USD"
        assert ws.granularity == "M5"
        assert 0.0 <= ws.conviction <= 1.0
        assert 0.0 <= ws.maturity <= 1.0
        assert ws.wave_pips > 0

    def test_bearish_impulse(self):
        machine = MagicMock()
        machine.state = "BEARISH_IMPULSE"
        machine.wave_origin = 1.1050
        machine.wave_peak = None
        machine.wave_trough = 1.1000
        machine.candle_idx = 15
        machine.wave_start_idx = 5
        machine.last_swing_high = 1.1040
        machine.prev_swing_high = 1.1060
        machine.last_swing_low = 1.1000
        machine.prev_swing_low = 1.1020
        machine.consecutive_swings = 2

        ws = score_wave(
            "EUR_USD", "M5", "2025-01-01T12:00:00Z",
            machine, {"p50": 25}, {"p50_candles": 20}, 0.0001,
        )
        assert ws.direction == -1.0
        assert ws.wave_pips == pytest.approx(50.0, abs=0.1)

    def test_ranging_zero_conviction(self):
        machine = MagicMock()
        machine.state = "RANGING"
        machine.wave_origin = None
        machine.wave_peak = None
        machine.wave_trough = None
        machine.candle_idx = 10
        machine.wave_start_idx = 0
        machine.last_swing_high = None
        machine.prev_swing_high = None
        machine.last_swing_low = None
        machine.prev_swing_low = None
        machine.consecutive_swings = 0

        ws = score_wave(
            "EUR_USD", "M5", "2025-01-01T12:00:00Z",
            machine, {"p50": 25}, {"p50_candles": 20}, 0.0001,
        )
        assert ws.direction == 0.0
        assert ws.conviction == 0.0
