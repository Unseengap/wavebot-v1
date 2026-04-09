"""Walk-forward optimizer with grouped parameter space."""
import copy
from src.backtest.engine import BacktestEngine
from src.backtest.metrics import calculate_metrics


# --- Grouped Parameter Space (108 combinations) ---

WAVE_SENSITIVITY_MAP = {
    "tight": {
        "swing_lookback": 2,
        "min_wave_pips_m1": 3, "min_wave_pips_m5": 8,
        "min_wave_pips_m15": 12, "min_wave_pips_h1": 20, "min_wave_pips_h4": 40,
    },
    "normal": {
        "swing_lookback": 3,
        "min_wave_pips_m1": 5, "min_wave_pips_m5": 10,
        "min_wave_pips_m15": 15, "min_wave_pips_h1": 25, "min_wave_pips_h4": 50,
    },
    "loose": {
        "swing_lookback": 4,
        "min_wave_pips_m1": 7, "min_wave_pips_m5": 12,
        "min_wave_pips_m15": 20, "min_wave_pips_h1": 30, "min_wave_pips_h4": 60,
    },
}

CONFLUENCE_STRICTNESS_MAP = {
    "aggressive": {"min_confluence_score": 0.65, "min_frames_aligned": 2},
    "balanced": {"min_confluence_score": 0.72, "min_frames_aligned": 3},
    "conservative": {"min_confluence_score": 0.80, "min_frames_aligned": 4},
}

RISK_PROFILE_MAP = {
    "tight": {"min_rr_ratio": 2.5, "tp_percentile": "p50", "sl_buffer_pips": 1},
    "standard": {"min_rr_ratio": 2.0, "tp_percentile": "p75", "sl_buffer_pips": 2},
    "wide": {"min_rr_ratio": 1.5, "tp_percentile": "p90", "sl_buffer_pips": 3},
}


def expand_params(wave_sens, conf_strict, risk_prof, max_maturity):
    """Expand grouped params into a flat config dict."""
    params = {}
    params.update(WAVE_SENSITIVITY_MAP[wave_sens])
    params.update(CONFLUENCE_STRICTNESS_MAP[conf_strict])
    params.update(RISK_PROFILE_MAP[risk_prof])
    params["max_entry_maturity"] = max_maturity
    return params


def build_param_grid():
    """Generate all 108 grouped parameter combinations."""
    grid = []
    for ws in ["tight", "normal", "loose"]:
        for cs in ["aggressive", "balanced", "conservative"]:
            for rp in ["tight", "standard", "wide"]:
                for mat in [0.65, 0.70, 0.75, 0.80]:
                    grid.append({
                        "wave_sensitivity": ws,
                        "confluence_strictness": cs,
                        "risk_profile": rp,
                        "max_entry_maturity": mat,
                    })
    return grid


def run_single_backtest(candle_data, base_config, params):
    """Run one backtest with merged config + params."""
    config = copy.deepcopy(base_config)
    expanded = expand_params(
        params["wave_sensitivity"],
        params["confluence_strictness"],
        params["risk_profile"],
        params["max_entry_maturity"],
    )
    config.update(expanded)

    engine = BacktestEngine(config)
    trades = engine.run(candle_data)

    trading_days = 1
    if trades:
        from datetime import datetime as dt
        first = trades[0]["entry_time"][:10]
        last = trades[-1]["entry_time"][:10]
        d1 = dt.strptime(first, "%Y-%m-%d")
        d2 = dt.strptime(last, "%Y-%m-%d")
        trading_days = max(1, (d2 - d1).days)

    metrics = calculate_metrics(
        trades, config["initial_balance"],
        config["instrument"], config.get("start", ""),
        config.get("end", ""), trading_days,
    )
    return metrics, trades


def score_metrics(m):
    """Single score for ranking parameter sets. Higher = better."""
    if m.total_trades < 10:
        return -999
    # Weighted: profit_factor (40%), sharpe (30%), win_rate (20%), trades/day (10%)
    pf_score = min(3.0, m.profit_factor) / 3.0
    sh_score = min(3.0, max(0, m.sharpe_ratio)) / 3.0
    wr_score = m.win_rate
    td_score = min(10.0, m.trades_per_day) / 10.0
    # Penalty for excessive drawdown
    dd_penalty = max(0, m.max_drawdown_pct - 10) * 0.05
    return (pf_score * 0.4 + sh_score * 0.3 + wr_score * 0.2 + td_score * 0.1) - dd_penalty
