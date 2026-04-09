"""Tests for walk-forward optimizer utilities."""
import pytest
from src.backtest.optimizer import (
    build_param_grid, expand_params, score_metrics,
    WAVE_SENSITIVITY_MAP, CONFLUENCE_STRICTNESS_MAP, RISK_PROFILE_MAP,
)
from src.backtest.metrics import BacktestMetrics


class TestBuildParamGrid:
    def test_grid_size(self):
        """3 wave × 3 confluence × 3 risk × 4 maturity = 108."""
        grid = build_param_grid()
        assert len(grid) == 108

    def test_all_combos_have_required_keys(self):
        grid = build_param_grid()
        for p in grid:
            assert "wave_sensitivity" in p
            assert "confluence_strictness" in p
            assert "risk_profile" in p
            assert "max_entry_maturity" in p

    def test_no_duplicates(self):
        grid = build_param_grid()
        tuples = [
            (p["wave_sensitivity"], p["confluence_strictness"],
             p["risk_profile"], p["max_entry_maturity"])
            for p in grid
        ]
        assert len(set(tuples)) == 108

    def test_valid_values(self):
        grid = build_param_grid()
        for p in grid:
            assert p["wave_sensitivity"] in WAVE_SENSITIVITY_MAP
            assert p["confluence_strictness"] in CONFLUENCE_STRICTNESS_MAP
            assert p["risk_profile"] in RISK_PROFILE_MAP
            assert 0.6 <= p["max_entry_maturity"] <= 0.85


class TestExpandParams:
    def test_contains_all_keys(self):
        p = expand_params("normal", "balanced", "standard", 0.70)
        assert "swing_lookback" in p
        assert "min_confluence_score" in p
        assert "min_rr_ratio" in p
        assert "max_entry_maturity" in p
        assert p["max_entry_maturity"] == 0.70

    def test_tight_has_lower_lookback(self):
        tight = expand_params("tight", "balanced", "standard", 0.70)
        loose = expand_params("loose", "balanced", "standard", 0.70)
        assert tight["swing_lookback"] < loose["swing_lookback"]

    def test_conservative_higher_threshold(self):
        agg = expand_params("normal", "aggressive", "standard", 0.70)
        con = expand_params("normal", "conservative", "standard", 0.70)
        assert agg["min_confluence_score"] < con["min_confluence_score"]
        assert agg["min_frames_aligned"] < con["min_frames_aligned"]


class TestScoreMetrics:
    def test_too_few_trades_penalized(self):
        m = BacktestMetrics()
        m.total_trades = 5
        assert score_metrics(m) == -999

    def test_good_performance_positive(self):
        m = BacktestMetrics()
        m.total_trades = 50
        m.profit_factor = 2.0
        m.sharpe_ratio = 1.5
        m.win_rate = 0.6
        m.trades_per_day = 2.0
        m.max_drawdown_pct = 5.0
        s = score_metrics(m)
        assert s > 0

    def test_high_drawdown_penalized(self):
        m1 = BacktestMetrics()
        m1.total_trades = 50
        m1.profit_factor = 2.0
        m1.sharpe_ratio = 1.5
        m1.win_rate = 0.6
        m1.trades_per_day = 2.0
        m1.max_drawdown_pct = 5.0

        m2 = BacktestMetrics()
        m2.total_trades = 50
        m2.profit_factor = 2.0
        m2.sharpe_ratio = 1.5
        m2.win_rate = 0.6
        m2.trades_per_day = 2.0
        m2.max_drawdown_pct = 25.0

        assert score_metrics(m1) > score_metrics(m2)

    def test_higher_pf_better_score(self):
        m1 = BacktestMetrics()
        m1.total_trades = 50
        m1.profit_factor = 1.2
        m1.sharpe_ratio = 1.0
        m1.win_rate = 0.5
        m1.trades_per_day = 2.0
        m1.max_drawdown_pct = 5.0

        m2 = BacktestMetrics()
        m2.total_trades = 50
        m2.profit_factor = 2.5
        m2.sharpe_ratio = 1.0
        m2.win_rate = 0.5
        m2.trades_per_day = 2.0
        m2.max_drawdown_pct = 5.0

        assert score_metrics(m2) > score_metrics(m1)
