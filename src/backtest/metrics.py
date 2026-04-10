"""Backtest metrics: Sharpe, Sortino, MFE/MAE, profit factor, drawdown."""

import numpy as np


def compute_metrics(trades: list, initial_balance: float = 10_000) -> dict:
    """Compute comprehensive backtest metrics from a list of BacktestTrade objects.

    Args:
        trades: List of BacktestTrade objects from WalkForwardBacktest.run().
        initial_balance: Starting account balance.

    Returns:
        Dict of metric name -> value.
    """
    if not trades:
        return _empty_metrics()

    pnls = np.array([t.pnl for t in trades])
    pnl_pips = np.array([t.pnl_pips for t in trades])
    durations = np.array([t.duration_bars for t in trades])
    mfes = np.array([t.mfe_pips for t in trades])
    maes = np.array([t.mae_pips for t in trades])

    total_pnl = pnls.sum()
    final_balance = initial_balance + total_pnl

    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]

    # Returns series (per trade, as fraction of running balance)
    returns = []
    running_balance = initial_balance
    for pnl in pnls:
        ret = pnl / running_balance if running_balance > 0 else 0
        returns.append(ret)
        running_balance += pnl
    returns = np.array(returns)

    # Equity curve for drawdown
    equity = np.cumsum(pnls) + initial_balance
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / peak
    max_drawdown = drawdown.max() if len(drawdown) > 0 else 0
    avg_drawdown = drawdown.mean() if len(drawdown) > 0 else 0

    # Sharpe ratio (annualized, assuming ~252 trading days, ~96 M15 bars/day)
    bars_per_year = 252 * 96  # for M15
    if len(returns) > 1 and returns.std() > 0:
        avg_bars = durations.mean() if durations.mean() > 0 else 1
        trades_per_year = bars_per_year / avg_bars
        sharpe = returns.mean() / returns.std() * np.sqrt(trades_per_year)
    else:
        sharpe = 0.0

    # Sortino ratio (penalizes only downside volatility)
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 1 and downside_returns.std() > 0:
        avg_bars = durations.mean() if durations.mean() > 0 else 1
        trades_per_year = bars_per_year / avg_bars
        sortino = returns.mean() / downside_returns.std() * np.sqrt(trades_per_year)
    else:
        sortino = sharpe  # fallback

    # Win rate
    win_rate = len(wins) / len(pnls) if len(pnls) > 0 else 0

    # Profit factor
    gross_profit = wins.sum() if len(wins) > 0 else 0
    gross_loss = abs(losses.sum()) if len(losses) > 0 else 1
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Annualized return
    total_bars = durations.sum()
    if total_bars > 0:
        years = total_bars / bars_per_year
        total_return = total_pnl / initial_balance
        annualized_return = (1 + total_return) ** (1 / max(years, 0.01)) - 1 if years > 0 else total_return
    else:
        total_return = 0
        annualized_return = 0

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "max_drawdown": round(-max_drawdown, 4),  # negative convention
        "avg_drawdown": round(-avg_drawdown, 4),
        "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor, 2),
        "total_trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "avg_win": round(float(wins.mean()), 2) if len(wins) > 0 else 0,
        "avg_loss": round(float(losses.mean()), 2) if len(losses) > 0 else 0,
        "avg_win_pips": round(float(pnl_pips[pnls > 0].mean()), 1) if len(wins) > 0 else 0,
        "avg_loss_pips": round(float(pnl_pips[pnls <= 0].mean()), 1) if len(losses) > 0 else 0,
        "avg_mfe": round(float(mfes.mean()), 1) if len(mfes) > 0 else 0,
        "avg_mae": round(float(maes.mean()), 1) if len(maes) > 0 else 0,
        "avg_duration_bars": round(float(durations.mean()), 0) if len(durations) > 0 else 0,
        "total_pnl": round(total_pnl, 2),
        "final_balance": round(final_balance, 2),
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(-abs(float(losses.sum())) if len(losses) > 0 else 0, 2),
    }


def _empty_metrics() -> dict:
    """Return zeroed metrics dict when there are no trades."""
    return {
        "total_return": 0, "annualized_return": 0,
        "sharpe": 0, "sortino": 0,
        "max_drawdown": 0, "avg_drawdown": 0,
        "win_rate": 0, "profit_factor": 0,
        "total_trades": 0, "wins": 0, "losses": 0,
        "avg_win": 0, "avg_loss": 0,
        "avg_win_pips": 0, "avg_loss_pips": 0,
        "avg_mfe": 0, "avg_mae": 0,
        "avg_duration_bars": 0,
        "total_pnl": 0, "final_balance": 0,
        "gross_profit": 0, "gross_loss": 0,
    }
