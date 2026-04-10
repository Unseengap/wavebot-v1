"""Backtest report generator — HTML and CSV output."""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger("backtest.reporter")


def generate_report(
    trades: list,
    metrics: dict,
    equity_curve: list[float] = None,
    output_path: str = "reports/backtest_report.html",
) -> str:
    """Generate an HTML backtest report.

    Args:
        trades: List of BacktestTrade objects.
        metrics: Dict from compute_metrics().
        equity_curve: List of equity values over time.
        output_path: Where to save the HTML file.

    Returns:
        Path to the generated report.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Build trade table
    trade_rows = []
    for t in trades:
        trade_rows.append({
            "Entry Bar": t.entry_idx,
            "Exit Bar": t.exit_idx,
            "Direction": t.direction,
            "Entry Price": f"{t.entry_price:.5f}",
            "Exit Price": f"{t.exit_price:.5f}",
            "P&L": f"${t.pnl:.2f}",
            "Pips": f"{t.pnl_pips:.1f}",
            "Duration": t.duration_bars,
            "Exit Reason": t.exit_reason,
            "MFE": f"{t.mfe_pips:.1f}",
            "MAE": f"{t.mae_pips:.1f}",
        })

    trade_df = pd.DataFrame(trade_rows)
    trade_html = trade_df.to_html(index=False, classes="trade-table") if not trade_df.empty else "<p>No trades.</p>"

    # Build metrics summary
    metrics_html = "<table class='metrics-table'>"
    for key, val in metrics.items():
        if isinstance(val, float):
            if "pct" in key or "return" in key or "rate" in key or "drawdown" in key:
                formatted = f"{val:.2%}"
            else:
                formatted = f"{val:.2f}"
        else:
            formatted = str(val)
        display_key = key.replace("_", " ").title()
        metrics_html += f"<tr><td><b>{display_key}</b></td><td>{formatted}</td></tr>"
    metrics_html += "</table>"

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>OandaFX Backtest Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; background: #1a1a2e; color: #e0e0e0; }}
        h1, h2 {{ color: #00d4ff; }}
        .metrics-table {{ border-collapse: collapse; margin: 20px 0; }}
        .metrics-table td {{ padding: 8px 16px; border-bottom: 1px solid #333; }}
        .trade-table {{ border-collapse: collapse; width: 100%; margin: 20px 0; font-size: 13px; }}
        .trade-table th {{ background: #16213e; padding: 10px; text-align: left; }}
        .trade-table td {{ padding: 8px; border-bottom: 1px solid #333; }}
        .trade-table tr:hover {{ background: #1a1a3e; }}
        .section {{ margin: 30px 0; }}
    </style>
</head>
<body>
    <h1>OandaFX Backtest Report</h1>

    <div class="section">
        <h2>Performance Summary</h2>
        {metrics_html}
    </div>

    <div class="section">
        <h2>Trade Log ({len(trades)} trades)</h2>
        {trade_html}
    </div>
</body>
</html>"""

    path.write_text(html)
    logger.info(f"Report saved to {path}")
    return str(path)


def export_trades_csv(trades: list, output_path: str = "reports/trades.csv") -> str:
    """Export trade list to CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for t in trades:
        rows.append({
            "entry_idx": t.entry_idx,
            "exit_idx": t.exit_idx,
            "direction": t.direction,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "sl_price": t.sl_price,
            "tp_price": t.tp_price,
            "units": t.units,
            "pnl": t.pnl,
            "pnl_pips": t.pnl_pips,
            "duration_bars": t.duration_bars,
            "exit_reason": t.exit_reason,
            "mfe_pips": t.mfe_pips,
            "mae_pips": t.mae_pips,
        })

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    logger.info(f"Trades CSV saved to {path}")
    return str(path)
