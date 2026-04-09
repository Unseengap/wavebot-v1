"""Tests for slippage simulation."""
import pytest
from src.backtest.slippage_simulator import (
    SlippageSimulator, BASE_SLIPPAGE_PIPS,
    VOLATILITY_THRESHOLD, VOLATILITY_MULTIPLIER,
)


class TestGetSlippagePips:
    def test_normal_volatility(self):
        sim = SlippageSimulator()
        slip = sim.get_slippage_pips("EUR_USD", candle_range_pips=10.0, atr_pips=8.0)
        assert slip == BASE_SLIPPAGE_PIPS["EUR_USD"]

    def test_high_volatility_multiplied(self):
        """Candle range > 2× ATR triggers multiplied slippage."""
        sim = SlippageSimulator()
        atr = 5.0
        candle_range = atr * VOLATILITY_THRESHOLD * 1.5  # Well above threshold
        slip = sim.get_slippage_pips("EUR_USD", candle_range, atr)
        assert slip == pytest.approx(BASE_SLIPPAGE_PIPS["EUR_USD"] * VOLATILITY_MULTIPLIER)

    def test_exactly_at_threshold(self):
        """At exactly 2× ATR, should NOT trigger (need to be above)."""
        sim = SlippageSimulator()
        atr = 5.0
        candle_range = atr * VOLATILITY_THRESHOLD  # Exactly at
        slip = sim.get_slippage_pips("EUR_USD", candle_range, atr)
        assert slip == BASE_SLIPPAGE_PIPS["EUR_USD"]

    def test_zero_atr(self):
        """Zero ATR → base slippage (avoids division by zero implicitly)."""
        sim = SlippageSimulator()
        slip = sim.get_slippage_pips("EUR_USD", 10.0, 0.0)
        assert slip == BASE_SLIPPAGE_PIPS["EUR_USD"]

    def test_unknown_instrument_default(self):
        sim = SlippageSimulator()
        slip = sim.get_slippage_pips("UNKNOWN_PAIR", 5.0, 5.0)
        assert slip == 0.5

    def test_gold_higher_base(self):
        sim = SlippageSimulator()
        eur = sim.get_slippage_pips("EUR_USD", 5.0, 5.0)
        xau = sim.get_slippage_pips("XAU_USD", 5.0, 5.0)
        assert xau > eur


class TestApplyToFill:
    def test_long_entry_slips_up(self):
        sim = SlippageSimulator()
        price = sim.apply_to_fill("LONG", "ENTRY", 1.1000, 0.5, 0.0001)
        assert price > 1.1000

    def test_long_exit_slips_down(self):
        sim = SlippageSimulator()
        price = sim.apply_to_fill("LONG", "EXIT", 1.1020, 0.5, 0.0001)
        assert price < 1.1020

    def test_short_entry_slips_down(self):
        sim = SlippageSimulator()
        price = sim.apply_to_fill("SHORT", "ENTRY", 1.1000, 0.5, 0.0001)
        assert price < 1.1000

    def test_short_exit_slips_up(self):
        sim = SlippageSimulator()
        price = sim.apply_to_fill("SHORT", "EXIT", 1.0980, 0.5, 0.0001)
        assert price > 1.0980

    def test_slippage_always_adverse(self):
        """Slippage should always work against the trader."""
        sim = SlippageSimulator()
        mid = 1.1000
        slip = 1.0
        pip = 0.0001

        long_entry = sim.apply_to_fill("LONG", "ENTRY", mid, slip, pip)
        long_exit = sim.apply_to_fill("LONG", "EXIT", mid, slip, pip)
        # For LONG: entry higher (worse), exit lower (worse)
        assert long_entry > mid
        assert long_exit < mid

        short_entry = sim.apply_to_fill("SHORT", "ENTRY", mid, slip, pip)
        short_exit = sim.apply_to_fill("SHORT", "EXIT", mid, slip, pip)
        # For SHORT: entry lower (worse), exit higher (worse)
        assert short_entry < mid
        assert short_exit > mid

    def test_zero_slippage_no_change(self):
        sim = SlippageSimulator()
        price = sim.apply_to_fill("LONG", "ENTRY", 1.1000, 0.0, 0.0001)
        assert price == 1.1000
