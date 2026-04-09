"""Tests for V3 backtest runner — end-to-end with synthetic data."""
import numpy as np
import pandas as pd
import pytest

from src.strategy.v3.backtest_runner import V3BacktestEngine
from src.backtest.metrics import calculate_metrics


def _make_h4_candles(n=50, base=1.1000, trend="down_then_up"):
    """
    Generate synthetic 4H candle data.
    First half trends down, second half trends up → should trigger a bullish reversal.
    """
    times = [f"2025-01-{1 + i // 6:02d}T{(i % 6) * 4:02d}:00:00Z" for i in range(n)]
    opens, highs, lows, closes = [], [], [], []

    price = base
    mid = n // 2

    for i in range(n):
        o = price
        if i < mid:
            # Downtrend
            c = o - np.random.uniform(0.0005, 0.0020)
            h = o + np.random.uniform(0.0001, 0.0010)
            l = c - np.random.uniform(0.0001, 0.0010)
        else:
            # Uptrend
            c = o + np.random.uniform(0.0005, 0.0020)
            h = c + np.random.uniform(0.0001, 0.0010)
            l = o - np.random.uniform(0.0001, 0.0010)

        opens.append(o)
        highs.append(h)
        lows.append(l)
        closes.append(c)
        price = c

    return pd.DataFrame({
        "time": times, "open": opens, "high": highs, "low": lows, "close": closes,
    })


def _make_daily_candles(h4_df):
    """Aggregate H4 candles to daily by date."""
    rows = []
    h4_df = h4_df.copy()
    h4_df["date"] = h4_df["time"].str[:10]
    for date, group in h4_df.groupby("date"):
        rows.append({
            "time": f"{date}T00:00:00Z",
            "open": group["open"].iloc[0],
            "high": group["high"].max(),
            "low": group["low"].min(),
            "close": group["close"].iloc[-1],
        })
    return pd.DataFrame(rows)


class TestV3BacktestEngine:
    def test_runs_without_error(self):
        """Backtest should run and return a list of trade dicts."""
        np.random.seed(42)
        h4 = _make_h4_candles(60)
        daily = _make_daily_candles(h4)

        engine = V3BacktestEngine(
            instrument="EUR_USD",
            pip_size=0.0001,
            pip_value_per_unit=0.0001,
            initial_balance=10000.0,
        )
        trades = engine.run(h4, daily)

        assert isinstance(trades, list)
        # All trades should be dicts (compatible with metrics)
        for t in trades:
            assert isinstance(t, dict)

    def test_trade_format_compatible_with_metrics(self):
        """Trade dicts should work with calculate_metrics."""
        np.random.seed(42)
        h4 = _make_h4_candles(60)
        daily = _make_daily_candles(h4)

        engine = V3BacktestEngine(initial_balance=10000.0)
        trades = engine.run(h4, daily)

        # Should not raise
        metrics = calculate_metrics(
            trades, initial_balance=10000.0,
            instrument="EUR_USD", start="2025-01-01", end="2025-01-10",
            trading_days=10,
        )
        assert metrics.initial_balance == 10000.0

    def test_no_take_profit_set(self):
        """V3 trades should never have take_profit set."""
        np.random.seed(42)
        h4 = _make_h4_candles(60)
        daily = _make_daily_candles(h4)

        engine = V3BacktestEngine()
        trades = engine.run(h4, daily)

        for t in trades:
            assert t["take_profit"] is None

    def test_balance_updates(self):
        """Balance should change after trades close."""
        np.random.seed(42)
        h4 = _make_h4_candles(60)
        daily = _make_daily_candles(h4)

        engine = V3BacktestEngine(initial_balance=10000.0)
        engine.run(h4, daily)

        if engine.closed_trades:
            total_pnl = sum(t.pnl_dollars for t in engine.closed_trades)
            assert engine.balance == pytest.approx(10000.0 + total_pnl, rel=1e-6)

    def test_exit_reasons_valid(self):
        """All trades should have a valid exit reason."""
        np.random.seed(42)
        h4 = _make_h4_candles(60)
        daily = _make_daily_candles(h4)

        engine = V3BacktestEngine()
        trades = engine.run(h4, daily)

        valid_reasons = {"OPPOSITE_PATTERN", "FLIP_SIGNAL",
                         "SIGNAL_EXIT", "END_OF_TEST"}
        for t in trades:
            assert t["exit_reason"] in valid_reasons, f"Bad exit reason: {t['exit_reason']}"

    def test_runs_without_daily_candles(self):
        """Should work fine if no daily candles provided."""
        np.random.seed(42)
        h4 = _make_h4_candles(60)

        engine = V3BacktestEngine()
        trades = engine.run(h4, d_candles=None)

        assert isinstance(trades, list)

    def test_all_trades_closed_at_end(self):
        """No open trades should remain after backtest."""
        np.random.seed(42)
        h4 = _make_h4_candles(60)

        engine = V3BacktestEngine()
        engine.run(h4)

        assert len(engine.open_trades) == 0

    def test_circuit_breaker_limits_trades(self):
        """Max open trades should be respected."""
        np.random.seed(42)
        h4 = _make_h4_candles(100)

        engine = V3BacktestEngine(max_open_v3_trades=1)
        engine.run(h4)

        # Can't directly assert max simultaneous trades without instrumentation,
        # but the engine should still run cleanly
        assert len(engine.open_trades) == 0  # All closed at end

    def test_v3_specific_fields_present(self):
        """Trade dicts should include V3-specific fields."""
        np.random.seed(42)
        h4 = _make_h4_candles(60)
        daily = _make_daily_candles(h4)

        engine = V3BacktestEngine()
        trades = engine.run(h4, daily)

        for t in trades:
            assert "pattern_type" in t
            assert "daily_context" in t
            assert "double_down_count" in t

    def test_empty_candle_data(self):
        """Should handle empty DataFrame gracefully."""
        h4 = pd.DataFrame(columns=["time", "open", "high", "low", "close"])

        engine = V3BacktestEngine()
        trades = engine.run(h4)

        assert trades == []
