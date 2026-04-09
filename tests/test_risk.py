"""Tests for risk management: position sizing, SL/TP, circuit breaker."""
import pytest
import numpy as np
from src.risk.position_sizer import calculate_position_size, apply_session_to_position_size
from src.risk.sl_tp_engine import (
    calculate_stop_loss, calculate_take_profit,
    calculate_rr, validate_sl_distance, calculate_atr,
)
from src.risk.circuit_breaker import CircuitBreaker


class TestPositionSizer:
    def test_basic_calculation(self):
        """10000 balance, 1% risk, 10 pip SL, 0.0001 pip value → units."""
        units = calculate_position_size(10000, 0.01, 10.0, 0.0001)
        # risk = 100, per_unit = 10*0.0001 = 0.001
        # units = 100 / 0.001 = 100000
        assert units == 100000

    def test_minimum_units(self):
        """Very small account should still return minimum 1000 units."""
        units = calculate_position_size(100, 0.01, 50.0, 0.0001)
        assert units >= 1000

    def test_zero_sl_returns_zero(self):
        assert calculate_position_size(10000, 0.01, 0.0, 0.0001) == 0

    def test_zero_pip_value_returns_zero(self):
        assert calculate_position_size(10000, 0.01, 10.0, 0.0) == 0

    def test_higher_risk_more_units(self):
        u1 = calculate_position_size(10000, 0.01, 10.0, 0.0001)
        u2 = calculate_position_size(10000, 0.02, 10.0, 0.0001)
        assert u2 > u1

    def test_wider_sl_fewer_units(self):
        u1 = calculate_position_size(10000, 0.01, 10.0, 0.0001)
        u2 = calculate_position_size(10000, 0.01, 20.0, 0.0001)
        assert u2 < u1


class TestSessionPositionSize:
    def test_dead_zone_zero(self):
        assert apply_session_to_position_size(10000, 0.0) == 0

    def test_multiplier_applied(self):
        base = 10000
        result = apply_session_to_position_size(base, 1.25)
        assert result == 12500

    def test_asia_reduced(self):
        base = 10000
        result = apply_session_to_position_size(base, 0.6)
        assert result == 6000

    def test_minimum_after_scaling(self):
        result = apply_session_to_position_size(100, 0.6)
        assert result >= 1000 or result == 0


class TestStopLoss:
    def test_long_sl_below_origin(self):
        sl = calculate_stop_loss("LONG", 1.1000, 0.0001, buffer_pips=2.0)
        assert sl < 1.1000
        expected = round(1.1000 - 2.0 * 0.0001, 5)
        assert sl == pytest.approx(expected)

    def test_short_sl_above_origin(self):
        sl = calculate_stop_loss("SHORT", 1.1000, 0.0001, buffer_pips=2.0)
        assert sl > 1.1000
        expected = round(1.1000 + 2.0 * 0.0001, 5)
        assert sl == pytest.approx(expected)

    def test_zero_buffer(self):
        sl = calculate_stop_loss("LONG", 1.1000, 0.0001, buffer_pips=0.0)
        assert sl == pytest.approx(1.1000)


class TestTakeProfit:
    def test_long_tp_above_entry(self):
        tp = calculate_take_profit("LONG", 1.1000, {"p75": 30}, 0.0001)
        assert tp > 1.1000

    def test_short_tp_below_entry(self):
        tp = calculate_take_profit("SHORT", 1.1000, {"p75": 30}, 0.0001)
        assert tp < 1.1000

    def test_different_percentiles(self):
        stats = {"p50": 20, "p75": 30, "p90": 50}
        tp50 = calculate_take_profit("LONG", 1.1000, stats, 0.0001, "p50")
        tp90 = calculate_take_profit("LONG", 1.1000, stats, 0.0001, "p90")
        assert tp90 > tp50


class TestRiskReward:
    def test_basic_rr(self):
        rr = calculate_rr(1.1010, 1.1000, 1.1030, "LONG")
        # Risk = 10 pips, reward = 20 pips → 2.0
        assert rr == pytest.approx(2.0)

    def test_short_rr(self):
        rr = calculate_rr(1.1000, 1.1010, 1.0980, "SHORT")
        # Risk = 10 pips, reward = 20 pips → 2.0
        assert rr == pytest.approx(2.0)

    def test_zero_risk(self):
        rr = calculate_rr(1.1000, 1.1000, 1.1020, "LONG")
        assert rr == 0.0


class TestValidateSLDistance:
    def test_within_range(self):
        assert validate_sl_distance(20.0, 10.0, max_atr_multiple=3.0) is True

    def test_too_wide(self):
        assert validate_sl_distance(40.0, 10.0, max_atr_multiple=3.0) is False

    def test_zero_atr_always_valid(self):
        assert validate_sl_distance(100.0, 0.0) is True


class TestATR:
    def test_basic_atr(self):
        highs = np.array([1.105, 1.110, 1.108, 1.112, 1.115] * 3)
        lows = np.array([1.100, 1.102, 1.101, 1.105, 1.108] * 3)
        closes = np.array([1.103, 1.108, 1.105, 1.110, 1.112] * 3)
        atr = calculate_atr(highs, lows, closes, period=14)
        assert atr > 0

    def test_short_array(self):
        """Fewer than period+1 candles: falls back to mean range."""
        highs = np.array([1.105, 1.110, 1.108])
        lows = np.array([1.100, 1.102, 1.101])
        closes = np.array([1.103, 1.108, 1.105])
        atr = calculate_atr(highs, lows, closes, period=14)
        assert atr > 0


class TestCircuitBreaker:
    def test_initial_state_allows(self):
        cb = CircuitBreaker()
        ok, reason = cb.check(10000, 0)
        assert ok is True
        assert reason == "OK"

    def test_daily_drawdown_halts(self):
        cb = CircuitBreaker(max_daily_drawdown=0.02)
        cb.reset_daily(10000)
        ok, reason = cb.check(9750, 0)  # 2.5% DD > 2%
        assert ok is False
        assert "DAILY DD" in reason

    def test_total_drawdown_halts(self):
        cb = CircuitBreaker(max_daily_drawdown=1.0, max_total_drawdown=0.08)
        cb.reset_daily(10000)
        cb.check(10000, 0)  # Sets peak
        ok, reason = cb.check(9100, 0)  # 9% DD > 8%
        assert ok is False
        assert "TOTAL DD" in reason

    def test_max_open_trades(self):
        cb = CircuitBreaker(max_open_trades=3)
        ok, reason = cb.check(10000, 3)
        assert ok is False
        assert "MAX TRADES" in reason

    def test_daily_reset_clears_halt(self):
        cb = CircuitBreaker(max_daily_drawdown=0.02)
        cb.reset_daily(10000)
        cb.check(9700, 0)  # Halted
        assert cb.halted_today is True
        cb.reset_daily(9700)
        assert cb.halted_today is False
        ok, _ = cb.check(9700, 0)
        assert ok is True

    def test_halted_today_persists(self):
        cb = CircuitBreaker(max_daily_drawdown=0.02)
        cb.reset_daily(10000)
        cb.check(9700, 0)  # Triggers halt
        # Even if balance recovers, halted_today stays
        ok, reason = cb.check(10000, 0)
        assert ok is False
        assert "HALTED" in reason

    def test_get_daily_drawdown(self):
        cb = CircuitBreaker()
        cb.reset_daily(10000)
        dd = cb.get_daily_drawdown(9900)
        assert dd == pytest.approx(0.01)

    def test_get_daily_drawdown_no_start(self):
        cb = CircuitBreaker()
        assert cb.get_daily_drawdown(9900) == 0.0
