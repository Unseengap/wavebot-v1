"""Tests for wave state machine transitions."""
import pytest
from src.wave.wave_state_machine import WaveStateMachine, WaveState


class TestInitialState:
    def test_starts_ranging(self):
        m = WaveStateMachine("M5")
        assert m.state == WaveState.RANGING

    def test_granularity_specific_thresholds(self):
        m1 = WaveStateMachine("M1")
        h4 = WaveStateMachine("H4")
        assert m1.max_ranging > h4.max_ranging
        assert m1.correction_thresh > h4.correction_thresh


class TestRangingExits:
    def test_bullish_exit_2hl_1hh(self):
        """2 higher lows + 1 higher high → BULLISH_IMPULSE."""
        m = WaveStateMachine("M5")
        # Establish baseline swings
        m.update(new_swing_low=1.0)
        m.update(new_swing_high=1.1)
        # Higher low 1
        m.update(new_swing_low=1.05)
        # Higher low 2
        m.update(new_swing_low=1.08)
        # Higher high
        m.update(new_swing_high=1.15)
        assert m.state == WaveState.BULLISH_IMPULSE

    def test_bearish_exit_2lh_1ll(self):
        """2 lower highs + 1 lower low → BEARISH_IMPULSE."""
        m = WaveStateMachine("M5")
        m.update(new_swing_high=1.5)
        m.update(new_swing_low=1.3)
        # Lower high 1
        m.update(new_swing_high=1.45)
        # Lower high 2
        m.update(new_swing_high=1.40)
        # Lower low
        m.update(new_swing_low=1.25)
        assert m.state == WaveState.BEARISH_IMPULSE

    def test_forced_reset_after_max_candles(self):
        """Ranging resets after max candles with no exit."""
        m = WaveStateMachine("M5")
        # Feed many candles with no swings
        for _ in range(m.max_ranging + 5):
            m.update()
        assert m.state == WaveState.RANGING
        # After reset, swing tracking is cleared
        assert m.last_swing_high is None


class TestBullishImpulse:
    def test_lower_low_reverts_to_ranging(self):
        """A lower swing low during bullish impulse → RANGING."""
        m = WaveStateMachine("M5")
        m.state = WaveState.BULLISH_IMPULSE
        m.last_swing_low = 1.10
        m.last_swing_high = 1.20
        # New swing low below last → structure break
        m.update(new_swing_low=1.05)
        assert m.state == WaveState.RANGING

    def test_no_swings_transitions_to_correction(self):
        """No swings for correction_thresh candles → BULLISH_CORRECTION."""
        m = WaveStateMachine("M5")
        m.state = WaveState.BULLISH_IMPULSE
        m.last_swing_low = 1.10
        m.last_swing_high = 1.20
        for _ in range(m.correction_thresh):
            m.update()
        assert m.state == WaveState.BULLISH_CORRECTION

    def test_continuation_resets_no_swing_count(self):
        m = WaveStateMachine("M5")
        m.state = WaveState.BULLISH_IMPULSE
        m.last_swing_high = 1.20
        m.last_swing_low = 1.10
        # Feed some candles without swings
        for _ in range(m.correction_thresh - 1):
            m.update()
        # Then a higher high resets counter
        m.update(new_swing_high=1.25)
        assert m.state == WaveState.BULLISH_IMPULSE
        assert m.no_swing_count == 0


class TestBearishImpulse:
    def test_higher_high_reverts_to_ranging(self):
        m = WaveStateMachine("M5")
        m.state = WaveState.BEARISH_IMPULSE
        m.last_swing_high = 1.20
        m.last_swing_low = 1.10
        m.update(new_swing_high=1.25)
        assert m.state == WaveState.RANGING

    def test_no_swings_transitions_to_correction(self):
        m = WaveStateMachine("M5")
        m.state = WaveState.BEARISH_IMPULSE
        m.last_swing_high = 1.20
        m.last_swing_low = 1.10
        for _ in range(m.correction_thresh):
            m.update()
        assert m.state == WaveState.BEARISH_CORRECTION


class TestCorrections:
    def test_bullish_correction_to_bullish_impulse(self):
        """Break above impulse high → back to BULLISH_IMPULSE."""
        m = WaveStateMachine("M5")
        m.state = WaveState.BULLISH_CORRECTION
        m.impulse_high = 1.20
        m.impulse_low = 1.10
        m.last_swing_low = 1.12
        m.update(new_swing_high=1.25)
        assert m.state == WaveState.BULLISH_IMPULSE

    def test_bullish_correction_to_bearish_impulse(self):
        """Break below impulse low → BEARISH_IMPULSE."""
        m = WaveStateMachine("M5")
        m.state = WaveState.BULLISH_CORRECTION
        m.impulse_high = 1.20
        m.impulse_low = 1.10
        m.last_swing_high = 1.18
        m.update(new_swing_low=1.05)
        assert m.state == WaveState.BEARISH_IMPULSE

    def test_correction_timeout_to_ranging(self):
        """Max correction candles → RANGING."""
        m = WaveStateMachine("M5")
        m.state = WaveState.BULLISH_CORRECTION
        m.impulse_high = 1.20
        m.impulse_low = 1.10
        for _ in range(m.max_correction + 1):
            m.update()
        assert m.state == WaveState.RANGING

    def test_bearish_correction_to_bearish_impulse(self):
        m = WaveStateMachine("M5")
        m.state = WaveState.BEARISH_CORRECTION
        m.impulse_low = 1.10
        m.impulse_high = 1.20
        m.last_swing_high = 1.18
        m.update(new_swing_low=1.05)
        assert m.state == WaveState.BEARISH_IMPULSE

    def test_bearish_correction_to_bullish_impulse(self):
        m = WaveStateMachine("M5")
        m.state = WaveState.BEARISH_CORRECTION
        m.impulse_low = 1.10
        m.impulse_high = 1.20
        m.last_swing_low = 1.12
        m.update(new_swing_high=1.25)
        assert m.state == WaveState.BULLISH_IMPULSE


class TestWaveTracking:
    def test_wave_peak_tracked(self):
        m = WaveStateMachine("M5")
        m.update(current_high=1.10, current_low=1.08)
        assert m.wave_peak == 1.10
        m.update(current_high=1.15, current_low=1.09)
        assert m.wave_peak == 1.15
        # Lower high doesn't reduce peak
        m.update(current_high=1.12, current_low=1.07)
        assert m.wave_peak == 1.15

    def test_wave_trough_tracked(self):
        m = WaveStateMachine("M5")
        m.update(current_high=1.10, current_low=1.08)
        assert m.wave_trough == 1.08
        m.update(current_high=1.12, current_low=1.05)
        assert m.wave_trough == 1.05

    def test_transition_sets_wave_origin(self):
        """On bullish exit from RANGING, wave_origin is set."""
        m = WaveStateMachine("M5")
        m.update(new_swing_low=1.0)
        m.update(new_swing_high=1.1)
        m.update(new_swing_low=1.05)
        m.update(new_swing_low=1.08)
        m.update(new_swing_high=1.15)
        assert m.state == WaveState.BULLISH_IMPULSE
        assert m.wave_origin is not None
