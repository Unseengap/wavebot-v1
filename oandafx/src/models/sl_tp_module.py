"""Dynamic SL/TP module — adjusts exit levels based on ATR, S/R, and volatility."""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger("models.sl_tp_module")


class DynamicSLTP:
    """Computes risk-adjusted stop loss and take profit levels.

    Provides a safety layer on top of raw RL agent SL/TP outputs by adjusting
    for real-time volatility, S/R levels, and account risk budget.
    """

    def __init__(
        self,
        min_sl_atr: float = 1.5,
        max_sl_atr: float = 3.0,
        min_rr: float = 1.5,
        target_rr: float = 2.0,
        vol_threshold: float = 1.5,
        vol_expansion: float = 1.3,
        sr_buffer_atr: float = 0.3,
    ):
        self.min_sl_atr = min_sl_atr
        self.max_sl_atr = max_sl_atr
        self.min_rr = min_rr
        self.target_rr = target_rr
        self.vol_threshold = vol_threshold
        self.vol_expansion = vol_expansion
        self.sr_buffer_atr = sr_buffer_atr

    def compute(
        self,
        direction: str,
        entry_price: float,
        atr: float,
        vol_regime: float = 1.0,
        nearest_support_dist: Optional[float] = None,
        nearest_resistance_dist: Optional[float] = None,
        account_risk_budget: Optional[float] = None,
        position_units: Optional[int] = None,
        pip_value: float = 10.0,
    ) -> dict:
        """Compute SL and TP prices.

        Args:
            direction: 'long' or 'short'
            entry_price: Entry price for the trade
            atr: Current ATR(14) value
            vol_regime: Volatility regime ratio (hist_vol / avg_vol)
            nearest_support_dist: Distance to nearest support in price units
            nearest_resistance_dist: Distance to nearest resistance in price units
            account_risk_budget: Max dollar amount to risk on this trade
            position_units: Number of units in the position
            pip_value: Dollar value per pip for the position size

        Returns:
            Dict with sl_price, tp_price, sl_pips, tp_pips, rr_ratio
        """
        if atr <= 0:
            atr = 0.0001  # fallback minimum

        # Base SL from ATR
        sl_pips = atr * self.min_sl_atr

        # Adjust for nearby S/R level (place SL just beyond it)
        if direction == "long" and nearest_support_dist is not None:
            if 0 < nearest_support_dist < atr * 2:
                sl_pips = nearest_support_dist + atr * self.sr_buffer_atr
        elif direction == "short" and nearest_resistance_dist is not None:
            if 0 < nearest_resistance_dist < atr * 2:
                sl_pips = nearest_resistance_dist + atr * self.sr_buffer_atr

        # Volatility regime override
        if vol_regime > self.vol_threshold:
            sl_pips *= self.vol_expansion

        # Clamp SL
        sl_pips = np.clip(sl_pips, atr * self.min_sl_atr, atr * self.max_sl_atr)

        # TP: aim for target RR, at minimum min_rr
        tp_pips = sl_pips * self.target_rr

        # If there's an opposing S/R level nearby, use it as TP target
        if direction == "long" and nearest_resistance_dist is not None:
            if nearest_resistance_dist > sl_pips * self.min_rr:
                tp_pips = max(tp_pips, nearest_resistance_dist * 0.9)
        elif direction == "short" and nearest_support_dist is not None:
            if nearest_support_dist > sl_pips * self.min_rr:
                tp_pips = max(tp_pips, nearest_support_dist * 0.9)

        # Ensure minimum RR
        if tp_pips < sl_pips * self.min_rr:
            tp_pips = sl_pips * self.min_rr

        # Hard cap: never risk more than account_risk_budget
        if account_risk_budget is not None and position_units is not None and position_units > 0:
            max_sl = account_risk_budget / position_units / pip_value
            if max_sl > 0:
                sl_pips = min(sl_pips, max_sl)

        # Compute prices
        if direction == "long":
            sl_price = entry_price - sl_pips
            tp_price = entry_price + tp_pips
        else:
            sl_price = entry_price + sl_pips
            tp_price = entry_price - tp_pips

        rr_ratio = tp_pips / sl_pips if sl_pips > 0 else 0

        return {
            "sl_price": round(sl_price, 5),
            "tp_price": round(tp_price, 5),
            "sl_pips": round(sl_pips / 0.0001, 1) if atr > 0.01 else round(sl_pips / 0.01, 1),
            "tp_pips": round(tp_pips / 0.0001, 1) if atr > 0.01 else round(tp_pips / 0.01, 1),
            "rr_ratio": round(rr_ratio, 2),
        }
