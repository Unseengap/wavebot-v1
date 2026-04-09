"""Tests for spread simulation."""
import pytest
from src.backtest.spread_simulator import SpreadSimulator, BASE_SPREADS, SESSION_SPREAD_MULT


class TestGetSpreadPips:
    def test_from_bid_ask(self):
        sim = SpreadSimulator()
        candle = {"close_ask": 1.10020, "close_bid": 1.10000}
        spread = sim.get_spread_pips(candle, "EUR_USD", "London_Open", 0.0001)
        assert spread == pytest.approx(2.0, abs=0.01)

    def test_fallback_to_model(self):
        sim = SpreadSimulator()
        candle = {}  # No bid/ask
        spread = sim.get_spread_pips(candle, "EUR_USD", "London_Open", 0.0001)
        expected = BASE_SPREADS["EUR_USD"] * SESSION_SPREAD_MULT["London_Open"]
        assert spread == pytest.approx(expected)

    def test_asia_session_wider_spread(self):
        sim = SpreadSimulator()
        candle = {}
        london = sim.get_spread_pips(candle, "EUR_USD", "London_Open", 0.0001)
        asia = sim.get_spread_pips(candle, "EUR_USD", "Asia_Session", 0.0001)
        assert asia > london

    def test_dead_zone_widest(self):
        sim = SpreadSimulator()
        candle = {}
        dz = sim.get_spread_pips(candle, "EUR_USD", "Dead_Zone", 0.0001)
        overlap = sim.get_spread_pips(candle, "EUR_USD", "London_NY_Overlap", 0.0001)
        assert dz > overlap

    def test_jpy_pair_pip_size(self):
        """JPY pairs use 0.01 pip size."""
        sim = SpreadSimulator()
        candle = {"close_ask": 150.025, "close_bid": 150.000}
        spread = sim.get_spread_pips(candle, "USD_JPY", "NY_Session", 0.01)
        assert spread == pytest.approx(2.5, abs=0.1)

    def test_none_bid_ask_falls_back(self):
        sim = SpreadSimulator()
        candle = {"close_ask": None, "close_bid": None}
        spread = sim.get_spread_pips(candle, "EUR_USD", "NY_Session", 0.0001)
        expected = BASE_SPREADS["EUR_USD"] * SESSION_SPREAD_MULT["NY_Session"]
        assert spread == pytest.approx(expected)


class TestApplyToEntry:
    def test_long_pays_ask(self):
        sim = SpreadSimulator()
        price = sim.apply_to_entry("LONG", 1.1000, 2.0, 0.0001)
        assert price > 1.1000

    def test_short_pays_bid(self):
        sim = SpreadSimulator()
        price = sim.apply_to_entry("SHORT", 1.1000, 2.0, 0.0001)
        assert price < 1.1000

    def test_spread_magnitude(self):
        sim = SpreadSimulator()
        mid = 1.1000
        spread_pips = 2.0
        pip_size = 0.0001
        long_price = sim.apply_to_entry("LONG", mid, spread_pips, pip_size)
        short_price = sim.apply_to_entry("SHORT", mid, spread_pips, pip_size)
        total_spread = (long_price - short_price) / pip_size
        assert total_spread == pytest.approx(spread_pips, abs=0.01)


class TestApplyToExit:
    def test_long_exit_at_bid(self):
        sim = SpreadSimulator()
        price = sim.apply_to_exit("LONG", 1.1020, 2.0, 0.0001)
        assert price < 1.1020

    def test_short_exit_at_ask(self):
        sim = SpreadSimulator()
        price = sim.apply_to_exit("SHORT", 1.0980, 2.0, 0.0001)
        assert price > 1.0980
