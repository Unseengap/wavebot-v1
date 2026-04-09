"""Tests for backtest metrics calculation."""
import pytest
import numpy as np
from src.backtest.metrics import calculate_metrics, BacktestMetrics


class TestCalculateMetrics:
    def test_empty_trades(self):
        m = calculate_metrics([], 10000, "EUR_USD", "2025-01-01", "2025-06-01")
        assert m.total_trades == 0
        assert m.final_balance == 10000
        assert m.win_rate == 0

    def test_basic_counts(self, sample_trades):
        m = calculate_metrics(sample_trades, 10000, "EUR_USD",
                              "2025-01-01", "2025-06-01", trading_days=120)
        assert m.total_trades == 5
        assert m.winning_trades == 3
        assert m.losing_trades == 2
        assert m.win_rate == pytest.approx(0.6)

    def test_pnl_calculations(self, sample_trades):
        m = calculate_metrics(sample_trades, 10000, "EUR_USD",
                              "2025-01-01", "2025-06-01", trading_days=120)
        expected_pips = 25.0 + (-10.0) + 18.0 + (-8.0) + 30.0
        assert m.total_pips == pytest.approx(expected_pips)
        expected_dollars = 25.0 + (-10.0) + 18.0 + (-8.0) + 30.0
        assert m.total_pnl_dollars == pytest.approx(expected_dollars)
        assert m.final_balance == pytest.approx(10000 + expected_dollars)

    def test_win_loss_averages(self, sample_trades):
        m = calculate_metrics(sample_trades, 10000, "EUR_USD",
                              "2025-01-01", "2025-06-01", trading_days=120)
        assert m.avg_win_pips == pytest.approx(np.mean([25.0, 18.0, 30.0]))
        assert m.avg_loss_pips == pytest.approx(np.mean([-10.0, -8.0]))

    def test_profit_factor(self, sample_trades):
        m = calculate_metrics(sample_trades, 10000, "EUR_USD",
                              "2025-01-01", "2025-06-01", trading_days=120)
        gross_profit = 25.0 + 18.0 + 30.0
        gross_loss = abs(-10.0 + -8.0)
        assert m.profit_factor == pytest.approx(gross_profit / gross_loss)

    def test_equity_curve(self, sample_trades):
        m = calculate_metrics(sample_trades, 10000, "EUR_USD",
                              "2025-01-01", "2025-06-01")
        assert len(m.equity_curve) == len(sample_trades) + 1
        assert m.equity_curve[0] == 10000
        assert m.equity_curve[-1] == pytest.approx(m.final_balance)

    def test_drawdown_calculation(self):
        """Test max drawdown with known sequence."""
        trades = [
            {"pnl_pips": 50.0, "pnl_dollars": 50.0},
            {"pnl_pips": -80.0, "pnl_dollars": -80.0},  # DD from 10050 to 9970
            {"pnl_pips": 20.0, "pnl_dollars": 20.0},
        ]
        m = calculate_metrics(trades, 10000, "EUR_USD", "", "")
        # Peak=10050, trough=9970, DD = 80/10050 = 0.796%
        assert m.max_drawdown_pct == pytest.approx(80 / 10050 * 100, abs=0.01)
        assert m.max_drawdown_dollars == pytest.approx(80.0)

    def test_return_percentage(self):
        trades = [{"pnl_pips": 100, "pnl_dollars": 500}]
        m = calculate_metrics(trades, 10000, "EUR_USD", "", "")
        assert m.total_return_pct == pytest.approx(5.0)

    def test_all_losses(self):
        trades = [
            {"pnl_pips": -10, "pnl_dollars": -10},
            {"pnl_pips": -15, "pnl_dollars": -15},
        ]
        m = calculate_metrics(trades, 10000, "EUR_USD", "", "")
        assert m.win_rate == 0.0
        assert m.profit_factor == 0.0
        assert m.final_balance == 10000 - 25

    def test_trades_per_day(self, sample_trades):
        m = calculate_metrics(sample_trades, 10000, "EUR_USD",
                              "2025-01-01", "2025-06-01", trading_days=50)
        assert m.trades_per_day == pytest.approx(5 / 50)

    def test_trade_quality_metrics(self, sample_trades):
        m = calculate_metrics(sample_trades, 10000, "EUR_USD",
                              "2025-01-01", "2025-06-01")
        assert m.avg_mae_pips > 0
        assert m.avg_mfe_pips > 0
        assert m.avg_bars_in_trade > 0
