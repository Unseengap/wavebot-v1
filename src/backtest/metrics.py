"""Performance metrics for backtest results."""
import numpy as np
from dataclasses import dataclass, field


@dataclass
class BacktestMetrics:
    instrument: str = ""
    start: str = ""
    end: str = ""
    initial_balance: float = 10000.0

    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0

    total_pips: float = 0.0
    avg_win_pips: float = 0.0
    avg_loss_pips: float = 0.0
    profit_factor: float = 0.0

    total_pnl_dollars: float = 0.0
    final_balance: float = 0.0
    total_return_pct: float = 0.0

    max_drawdown_pct: float = 0.0
    max_drawdown_dollars: float = 0.0
    sharpe_ratio: float = 0.0
    calmar_ratio: float = 0.0

    avg_rr_achieved: float = 0.0
    avg_mae_pips: float = 0.0
    avg_mfe_pips: float = 0.0
    avg_bars_in_trade: float = 0.0
    trades_per_day: float = 0.0

    equity_curve: list = field(default_factory=list)


def calculate_metrics(trades: list, initial_balance: float,
                      instrument: str, start: str, end: str,
                      trading_days: int = 1) -> BacktestMetrics:
    m = BacktestMetrics(
        instrument=instrument, start=start, end=end,
        initial_balance=initial_balance,
    )

    if not trades:
        m.final_balance = initial_balance
        return m

    m.total_trades = len(trades)
    wins = [t for t in trades if t["pnl_pips"] > 0]
    losses = [t for t in trades if t["pnl_pips"] <= 0]
    m.winning_trades = len(wins)
    m.losing_trades = len(losses)
    m.win_rate = m.winning_trades / m.total_trades if m.total_trades > 0 else 0

    # Pips
    m.total_pips = sum(t["pnl_pips"] for t in trades)
    m.avg_win_pips = np.mean([t["pnl_pips"] for t in wins]) if wins else 0
    m.avg_loss_pips = np.mean([t["pnl_pips"] for t in losses]) if losses else 0

    # Profit factor
    gross_profit = sum(t["pnl_dollars"] for t in wins)
    gross_loss = abs(sum(t["pnl_dollars"] for t in losses))
    m.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Dollar P&L
    m.total_pnl_dollars = sum(t["pnl_dollars"] for t in trades)
    m.final_balance = initial_balance + m.total_pnl_dollars
    m.total_return_pct = (m.total_pnl_dollars / initial_balance) * 100

    # Equity curve and drawdown
    equity = [initial_balance]
    for t in trades:
        equity.append(equity[-1] + t["pnl_dollars"])
    m.equity_curve = equity

    peak = initial_balance
    max_dd = 0
    max_dd_dollars = 0
    for e in equity:
        peak = max(peak, e)
        dd = (peak - e) / peak if peak > 0 else 0
        dd_dollars = peak - e
        max_dd = max(max_dd, dd)
        max_dd_dollars = max(max_dd_dollars, dd_dollars)
    m.max_drawdown_pct = max_dd * 100
    m.max_drawdown_dollars = max_dd_dollars

    # Sharpe ratio (annualized, daily returns)
    daily_returns = []
    if len(equity) > 1:
        eq_arr = np.array(equity)
        returns = np.diff(eq_arr) / eq_arr[:-1]
        if len(returns) > 1:
            if np.std(returns) > 0:
                m.sharpe_ratio = float(np.mean(returns) / np.std(returns) * np.sqrt(252))

    # Calmar ratio
    annual_return = m.total_return_pct / max(1, trading_days / 252)
    m.calmar_ratio = annual_return / m.max_drawdown_pct if m.max_drawdown_pct > 0 else 0

    # Trade quality
    rr_list = [t.get("rr_achieved", 0) for t in wins]
    m.avg_rr_achieved = float(np.mean(rr_list)) if rr_list else 0

    mae_list = [t.get("max_adverse_pips", 0) for t in trades]
    mfe_list = [t.get("max_favorable_pips", 0) for t in trades]
    m.avg_mae_pips = float(np.mean(mae_list)) if mae_list else 0
    m.avg_mfe_pips = float(np.mean(mfe_list)) if mfe_list else 0

    bars = [t.get("bars_in_trade", 0) for t in trades]
    m.avg_bars_in_trade = float(np.mean(bars)) if bars else 0

    m.trades_per_day = m.total_trades / max(1, trading_days)

    return m


def print_metrics(m: BacktestMetrics):
    print("=" * 65)
    print(f"  WAVEBOT BACKTEST RESULTS — {m.instrument}")
    print(f"  Period: {m.start} to {m.end}")
    print("=" * 65)
    print(f"  Starting Balance:     ${m.initial_balance:,.2f}")
    print(f"  Final Balance:        ${m.final_balance:,.2f}")
    print(f"  Total P&L:            ${m.total_pnl_dollars:,.2f}  ({m.total_return_pct:+.1f}%)")
    print("-" * 65)
    print(f"  Total Trades:         {m.total_trades}")
    print(f"  Winning:              {m.winning_trades}  |  Losing: {m.losing_trades}")
    print(f"  Win Rate:             {m.win_rate:.1%}")
    print(f"  Profit Factor:        {m.profit_factor:.2f}")
    print(f"  Total Pips:           {m.total_pips:+.1f}")
    print(f"  Avg Win:              {m.avg_win_pips:+.1f} pips")
    print(f"  Avg Loss:             {m.avg_loss_pips:.1f} pips")
    print("-" * 65)
    print(f"  Max Drawdown:         {m.max_drawdown_pct:.2f}%  (${m.max_drawdown_dollars:,.2f})")
    print(f"  Sharpe Ratio:         {m.sharpe_ratio:.2f}")
    print(f"  Calmar Ratio:         {m.calmar_ratio:.2f}")
    print("-" * 65)
    print(f"  Avg R:R (winners):    {m.avg_rr_achieved:.2f}")
    print(f"  Avg MAE:              {m.avg_mae_pips:.1f} pips")
    print(f"  Avg MFE:              {m.avg_mfe_pips:.1f} pips")
    print(f"  Avg Bars in Trade:    {m.avg_bars_in_trade:.0f}")
    print(f"  Trades/Day:           {m.trades_per_day:.1f}")
    print("=" * 65)
