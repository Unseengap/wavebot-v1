"""Account reconciliation — compares local state with OANDA API state."""

import logging
from typing import Optional

logger = logging.getLogger("execution.reconciler")


class Reconciler:
    """Reconciles local trade state with OANDA's authoritative state.

    Detects mismatches between what the bot thinks is open and what OANDA reports.
    """

    def __init__(self, api_client):
        self.api_client = api_client
        self._local_trades: dict[str, dict] = {}

    def register_trade(self, trade_id: str, trade_data: dict):
        """Register a locally-opened trade for reconciliation tracking."""
        self._local_trades[trade_id] = trade_data

    def unregister_trade(self, trade_id: str):
        """Remove a closed trade from local tracking."""
        self._local_trades.pop(trade_id, None)

    def reconcile(self) -> dict:
        """Compare local state with OANDA and report mismatches."""
        try:
            oanda_trades = self.api_client.get_open_trades()
        except Exception as e:
            logger.error(f"Reconciliation failed — API error: {e}")
            return {"status": "error", "error": str(e)}

        oanda_ids = {t["id"] for t in oanda_trades}
        local_ids = set(self._local_trades.keys())

        # Trades we think are open but OANDA says they're not
        missing_on_oanda = local_ids - oanda_ids

        # Trades OANDA says are open but we don't know about
        unknown_on_local = oanda_ids - local_ids

        mismatches = []

        if missing_on_oanda:
            for tid in missing_on_oanda:
                mismatches.append({
                    "type": "local_only",
                    "trade_id": tid,
                    "detail": "Trade exists locally but not on OANDA (likely closed externally)",
                })
                logger.warning(
                    f"Reconciliation mismatch: trade {tid} missing on OANDA",
                    extra={"event": "reconciliation_mismatch", "data": {"trade_id": tid, "type": "local_only"}},
                )
                self._local_trades.pop(tid, None)

        if unknown_on_local:
            for tid in unknown_on_local:
                mismatches.append({
                    "type": "oanda_only",
                    "trade_id": tid,
                    "detail": "Trade exists on OANDA but not tracked locally (opened externally)",
                })
                logger.warning(
                    f"Reconciliation mismatch: trade {tid} unknown locally",
                    extra={"event": "reconciliation_mismatch", "data": {"trade_id": tid, "type": "oanda_only"}},
                )

        return {
            "status": "ok" if not mismatches else "mismatches_found",
            "local_count": len(local_ids),
            "oanda_count": len(oanda_ids),
            "mismatches": mismatches,
        }
