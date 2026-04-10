"""Circuit breaker — daily loss halt, max drawdown halt, maintenance detection."""

import logging
from datetime import datetime, timezone

logger = logging.getLogger("risk.circuit_breaker")


class CircuitBreaker:
    """Monitors account health and halts trading when limits are exceeded."""

    def __init__(
        self,
        max_daily_loss_pct: float = 0.03,
        max_drawdown_pct: float = 0.08,
        max_open_trades: int = 5,
        cooldown_minutes: int = 60,
    ):
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.max_open_trades = max_open_trades
        self.cooldown_minutes = cooldown_minutes

        self._tripped = False
        self._trip_reason = ""
        self._trip_time = None
        self._daily_start_balance = 0.0
        self._peak_balance = 0.0
        self._last_reset_date = None

    def update(self, balance: float, unrealized_pnl: float = 0.0):
        """Update balance tracking. Call on each new bar."""
        nav = balance + unrealized_pnl
        today = datetime.now(timezone.utc).date()

        # Reset daily tracking at day boundary
        if self._last_reset_date != today:
            self._daily_start_balance = balance
            self._last_reset_date = today

        # Track peak
        if nav > self._peak_balance:
            self._peak_balance = nav

    def check(self, balance: float, unrealized_pnl: float = 0.0, open_trade_count: int = 0) -> tuple[bool, str]:
        """Check if trading should be halted.

        Returns:
            (is_safe, reason) — is_safe=True means trading can continue.
        """
        nav = balance + unrealized_pnl

        # Check cooldown
        if self._tripped and self._trip_time:
            elapsed = (datetime.now(timezone.utc) - self._trip_time).total_seconds() / 60
            if elapsed < self.cooldown_minutes:
                return False, f"cooldown_active ({self._trip_reason})"

        # Daily loss check
        if self._daily_start_balance > 0:
            daily_loss = (self._daily_start_balance - nav) / self._daily_start_balance
            if daily_loss >= self.max_daily_loss_pct:
                self._trip("daily_loss_limit", f"Daily loss {daily_loss:.2%} >= {self.max_daily_loss_pct:.2%}")
                return False, self._trip_reason

        # Max drawdown check
        if self._peak_balance > 0:
            drawdown = (self._peak_balance - nav) / self._peak_balance
            if drawdown >= self.max_drawdown_pct:
                self._trip("max_drawdown", f"Drawdown {drawdown:.2%} >= {self.max_drawdown_pct:.2%}")
                return False, self._trip_reason

        # Max open trades
        if open_trade_count >= self.max_open_trades:
            return False, f"max_open_trades ({open_trade_count}/{self.max_open_trades})"

        # Clear trip if checks pass
        if self._tripped:
            self._tripped = False
            logger.info("Circuit breaker reset — trading resumed")

        return True, "ok"

    def _trip(self, reason: str, detail: str):
        """Activate the circuit breaker."""
        self._tripped = True
        self._trip_reason = reason
        self._trip_time = datetime.now(timezone.utc)
        logger.critical(
            f"CIRCUIT BREAKER TRIGGERED: {detail}",
            extra={"event": "circuit_breaker_triggered", "data": {"reason": reason, "detail": detail}},
        )

    @property
    def is_tripped(self) -> bool:
        return self._tripped

    def reset(self):
        """Manually reset the circuit breaker."""
        self._tripped = False
        self._trip_reason = ""
        self._trip_time = None
        logger.info("Circuit breaker manually reset")
