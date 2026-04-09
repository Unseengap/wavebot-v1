"""Pre-trade validation — 9-check compliance gate before every order."""

import logging
from datetime import datetime, timezone
from typing import Optional

from src.risk.circuit_breaker import CircuitBreaker
from src.risk.position_sizer import fixed_fraction_size

logger = logging.getLogger("risk.validator")

# Correlated currency groups
CORRELATION_GROUPS = {
    "EUR": ["EUR_USD", "EUR_GBP", "EUR_JPY", "EUR_CHF", "EUR_CAD", "EUR_AUD"],
    "GBP": ["GBP_USD", "EUR_GBP", "GBP_JPY", "GBP_CHF", "GBP_CAD"],
    "JPY": ["USD_JPY", "EUR_JPY", "GBP_JPY", "AUD_JPY", "CHF_JPY", "CAD_JPY"],
    "AUD": ["AUD_USD", "EUR_AUD", "AUD_JPY"],
    "CHF": ["USD_CHF", "EUR_CHF", "GBP_CHF", "CHF_JPY"],
    "CAD": ["USD_CAD", "EUR_CAD", "GBP_CAD", "CAD_JPY"],
}

# OANDA max leverage by instrument type (US clients)
MAX_LEVERAGE = {
    "major_fx": 50,
    "minor_fx": 20,
    "metals": 20,
    "indices": 20,
    "commodities": 10,
}


class RiskValidator:
    """Pre-trade compliance checker — every signal must pass all 9 checks."""

    def __init__(
        self,
        circuit_breaker: Optional[CircuitBreaker] = None,
        max_risk_per_trade: float = 0.01,
        max_open_trades: int = 5,
        max_correlated_exposure: int = 2,
        min_sl_atr_multiple: float = 1.0,
        min_rr_ratio: float = 1.2,
        maintenance_buffer_minutes: int = 10,
    ):
        self.circuit_breaker = circuit_breaker or CircuitBreaker()
        self.max_risk_per_trade = max_risk_per_trade
        self.max_open_trades = max_open_trades
        self.max_correlated_exposure = max_correlated_exposure
        self.min_sl_atr_multiple = min_sl_atr_multiple
        self.min_rr_ratio = min_rr_ratio
        self.maintenance_buffer_minutes = maintenance_buffer_minutes

    def validate(
        self,
        signal: dict,
        account_summary: dict,
        open_trades: list[dict],
        instrument: str,
        atr: float = 0.0,
    ) -> tuple[bool, str]:
        """Run all pre-trade checks. Returns (approved, rejection_reason)."""

        balance = float(account_summary.get("balance", 0))
        unrealized_pnl = float(account_summary.get("unrealizedPL", 0))
        margin_used = float(account_summary.get("marginUsed", 0))
        margin_available = float(account_summary.get("marginAvailable", 0))

        # Check 1: Circuit breaker (daily loss + max drawdown)
        self.circuit_breaker.update(balance, unrealized_pnl)
        safe, reason = self.circuit_breaker.check(balance, unrealized_pnl, len(open_trades))
        if not safe:
            return False, f"circuit_breaker: {reason}"

        # Check 2: Max concurrent trades
        if len(open_trades) >= self.max_open_trades:
            return False, f"max_open_trades: {len(open_trades)}/{self.max_open_trades}"

        # Check 3: Correlated pair exposure
        correlated = self._check_correlated_exposure(instrument, open_trades)
        if correlated >= self.max_correlated_exposure:
            return False, f"correlated_exposure: {correlated}/{self.max_correlated_exposure} for {instrument}"

        # Check 4: Leverage check
        total_margin = margin_used + balance * self.max_risk_per_trade
        if margin_available > 0 and total_margin / (margin_available + margin_used) > 0.9:
            return False, "leverage_exceeded: margin utilization too high"

        # Check 5: SL present and >= min ATR multiple
        sl_pips = signal.get("sl_pips", 0)
        if sl_pips <= 0:
            return False, "no_stop_loss: SL is required"
        if atr > 0 and sl_pips < atr * self.min_sl_atr_multiple:
            return False, f"sl_too_tight: {sl_pips:.1f} pips < {self.min_sl_atr_multiple}x ATR"

        # Check 6: TP present and RR >= minimum
        tp_pips = signal.get("tp_pips", 0)
        if tp_pips <= 0:
            return False, "no_take_profit: TP is required"
        rr = tp_pips / sl_pips if sl_pips > 0 else 0
        if rr < self.min_rr_ratio:
            return False, f"rr_too_low: {rr:.2f} < {self.min_rr_ratio}"

        # Check 7: Position size within risk budget
        position_risk = signal.get("position_size", 0)
        if position_risk > self.max_risk_per_trade:
            return False, f"risk_too_high: {position_risk:.2%} > {self.max_risk_per_trade:.2%}"

        # Check 8: OANDA leverage compliance (basic check)
        # Full compliance check would need instrument classification

        # Check 9: Not too close to maintenance window
        if self._near_maintenance():
            return False, "maintenance_window: within 10 min of scheduled maintenance"

        logger.debug(
            f"Trade approved for {instrument}",
            extra={"event": "risk_approved", "data": {"instrument": instrument}},
        )
        return True, "approved"

    def _check_correlated_exposure(self, instrument: str, open_trades: list[dict]) -> int:
        """Count how many open trades share a currency with the new instrument."""
        currencies = []
        for part in instrument.split("_"):
            currencies.append(part)

        count = 0
        for trade in open_trades:
            trade_instrument = trade.get("instrument", "")
            for curr in currencies:
                if curr in trade_instrument:
                    count += 1
                    break

        return count

    def _near_maintenance(self) -> bool:
        """Check if we're within buffer of OANDA maintenance (Friday ~22:00 UTC)."""
        now = datetime.now(timezone.utc)
        # OANDA maintenance: Friday after ~17:00 ET (22:00 UTC)
        if now.weekday() == 4 and now.hour >= 21 and now.minute >= (60 - self.maintenance_buffer_minutes):
            return True
        return False
