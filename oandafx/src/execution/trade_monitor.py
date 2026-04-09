"""Trade monitor — tracks open trades, computes unrealized P&L, MFE/MAE."""

import logging
from datetime import datetime, timezone
from typing import Optional

from src.risk.position_sizer import get_pip_size

logger = logging.getLogger("execution.trade_monitor")


class TradeMonitor:
    """Monitors open trades across an account.

    Tracks unrealized P&L, trade duration, maximum favorable excursion (MFE),
    and maximum adverse excursion (MAE).
    """

    def __init__(self, api_client):
        self.api_client = api_client
        self._trade_tracking: dict[str, dict] = {}  # trade_id -> tracking data

    def update(self) -> list[dict]:
        """Poll OANDA for current open trades and update tracking."""
        try:
            open_trades = self.api_client.get_open_trades()
        except Exception as e:
            logger.error(f"Failed to fetch open trades: {e}")
            return []

        active_ids = set()
        results = []

        for trade in open_trades:
            trade_id = trade.get("id", "")
            active_ids.add(trade_id)

            instrument = trade.get("instrument", "")
            units = int(trade.get("currentUnits", 0))
            entry_price = float(trade.get("price", 0))
            current_price = float(trade.get("unrealizedPL", 0)) + entry_price  # approximate
            unrealized_pnl = float(trade.get("unrealizedPL", 0))

            # Initialize tracking if new trade
            if trade_id not in self._trade_tracking:
                self._trade_tracking[trade_id] = {
                    "trade_id": trade_id,
                    "instrument": instrument,
                    "direction": "long" if units > 0 else "short",
                    "units": abs(units),
                    "entry_price": entry_price,
                    "opened_at": trade.get("openTime", datetime.now(timezone.utc).isoformat()),
                    "mfe_pnl": 0.0,
                    "mae_pnl": 0.0,
                    "mfe_pips": 0.0,
                    "mae_pips": 0.0,
                }

            tracking = self._trade_tracking[trade_id]

            # Update MFE/MAE
            if unrealized_pnl > tracking["mfe_pnl"]:
                tracking["mfe_pnl"] = unrealized_pnl
            if unrealized_pnl < tracking["mae_pnl"]:
                tracking["mae_pnl"] = unrealized_pnl

            pip_size = get_pip_size(instrument)
            if tracking["direction"] == "long":
                pnl_pips = (current_price - entry_price) / pip_size
            else:
                pnl_pips = (entry_price - current_price) / pip_size

            if pnl_pips > tracking["mfe_pips"]:
                tracking["mfe_pips"] = pnl_pips
            if pnl_pips < tracking["mae_pips"]:
                tracking["mae_pips"] = pnl_pips

            tracking["unrealized_pnl"] = unrealized_pnl
            tracking["current_pips"] = pnl_pips

            results.append(tracking.copy())

        # Clean up closed trades from tracking
        closed_ids = set(self._trade_tracking.keys()) - active_ids
        for tid in closed_ids:
            del self._trade_tracking[tid]

        return results

    def get_open_trade_summary(self) -> dict:
        """Return summary of all tracked open trades."""
        trades = self.update()
        return {
            "total_open": len(trades),
            "total_unrealized_pnl": sum(t.get("unrealized_pnl", 0) for t in trades),
            "trades": trades,
        }
