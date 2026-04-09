"""Shared fixtures for WaveBot test suite."""
import numpy as np
import pandas as pd
import pytest
from dataclasses import dataclass


@pytest.fixture
def pip_size():
    return 0.0001


@pytest.fixture
def sample_bullish_highs():
    """Highs with a clear peak at index 5."""
    return np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0])


@pytest.fixture
def sample_bullish_lows():
    """Lows with a clear trough at index 0."""
    return np.array([0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9])


@pytest.fixture
def flat_prices():
    """No swings — all prices equal."""
    return np.array([1.0] * 11)


@pytest.fixture
def m5_candle_df():
    """Minimal M5 candle DataFrame for backtest engine tests."""
    n = 50
    np.random.seed(42)
    base = 1.1000
    closes = base + np.cumsum(np.random.randn(n) * 0.0005)
    highs = closes + np.abs(np.random.randn(n) * 0.0003)
    lows = closes - np.abs(np.random.randn(n) * 0.0003)
    opens = closes + np.random.randn(n) * 0.0001

    times = pd.date_range("2025-01-01", periods=n, freq="5min")
    return pd.DataFrame({
        "time": times,
        "open_mid": opens,
        "high_mid": highs,
        "low_mid": lows,
        "close_mid": closes,
        "volume": np.random.randint(50, 500, n),
    })


@pytest.fixture
def sample_wave_score():
    """Factory for creating WaveScore objects."""
    from src.wave.wave_scorer import WaveScore

    def _make(direction=1.0, conviction=0.8, maturity=0.3,
              state="BULLISH_IMPULSE", granularity="M5"):
        return WaveScore(
            instrument="EUR_USD",
            granularity=granularity,
            timestamp="2025-01-01T12:00:00Z",
            direction=direction,
            conviction=conviction,
            maturity=maturity,
            state=state,
            wave_origin=1.1000,
            wave_age_candles=10,
            wave_pips=15.0,
        )
    return _make


@pytest.fixture
def entry_config():
    """Standard entry filter config."""
    return {
        "min_confluence_score": 0.65,
        "min_frames_aligned": 2,
        "max_open_trades": 3,
        "max_entry_maturity": 0.75,
        "min_rr_ratio": 2.0,
        "max_daily_drawdown": 0.02,
    }


@pytest.fixture
def backtest_config():
    """Minimal backtest engine config."""
    return {
        "instrument": "EUR_USD",
        "initial_balance": 10000.0,
        "max_daily_drawdown": 0.02,
        "max_total_drawdown": 0.08,
        "max_open_trades": 3,
        "risk_fraction": 0.01,
        "sl_buffer_pips": 2.0,
        "tp_percentile": "p75",
        "min_rr_ratio": 2.0,
        "min_confluence_score": 0.65,
        "min_frames_aligned": 2,
    }


@pytest.fixture
def sample_trades():
    """Sample closed trades for metrics testing."""
    return [
        {
            "pnl_pips": 25.0, "pnl_dollars": 25.0,
            "rr_achieved": 2.5, "max_adverse_pips": 5.0,
            "max_favorable_pips": 28.0, "bars_in_trade": 12,
        },
        {
            "pnl_pips": -10.0, "pnl_dollars": -10.0,
            "rr_achieved": -1.0, "max_adverse_pips": 12.0,
            "max_favorable_pips": 3.0, "bars_in_trade": 8,
        },
        {
            "pnl_pips": 18.0, "pnl_dollars": 18.0,
            "rr_achieved": 1.8, "max_adverse_pips": 7.0,
            "max_favorable_pips": 20.0, "bars_in_trade": 15,
        },
        {
            "pnl_pips": -8.0, "pnl_dollars": -8.0,
            "rr_achieved": -0.8, "max_adverse_pips": 10.0,
            "max_favorable_pips": 2.0, "bars_in_trade": 6,
        },
        {
            "pnl_pips": 30.0, "pnl_dollars": 30.0,
            "rr_achieved": 3.0, "max_adverse_pips": 4.0,
            "max_favorable_pips": 32.0, "bars_in_trade": 20,
        },
    ]
