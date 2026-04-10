"""Position sizing — Kelly criterion and fixed-fraction methods."""

import logging
import math

logger = logging.getLogger("risk.position_sizer")

# Pip size lookup (approximate)
PIP_SIZES = {
    "JPY": 0.01,
    "HUF": 0.01,
    "DEFAULT": 0.0001,
}


def get_pip_size(instrument: str) -> float:
    """Get pip size for an instrument."""
    quote_currency = instrument.split("_")[-1] if "_" in instrument else instrument[-3:]
    return PIP_SIZES.get(quote_currency, PIP_SIZES["DEFAULT"])


def fixed_fraction_size(
    balance: float,
    risk_pct: float,
    sl_pips: float,
    pip_value: float = 10.0,
) -> int:
    """Compute position size using fixed-fraction method.

    Args:
        balance: Account balance in base currency.
        risk_pct: Max risk per trade as fraction (e.g. 0.01 = 1%).
        sl_pips: Stop loss in pips.
        pip_value: Dollar value per pip per standard lot.

    Returns:
        Position size in units (1 standard lot = 100,000 units).
    """
    if sl_pips <= 0 or pip_value <= 0 or balance <= 0:
        return 0

    risk_amount = balance * risk_pct
    pip_value_per_unit = pip_value / 100_000  # pip value per single unit
    units = int(risk_amount / (sl_pips * pip_value_per_unit))

    # Ensure minimum 1 unit and maximum reasonable position
    units = max(units, 0)
    max_units = int(balance * 50 / 1.0)  # rough 50:1 leverage cap
    units = min(units, max_units)

    return units


def kelly_fraction(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """Compute Kelly criterion optimal fraction.

    Args:
        win_rate: Historical win rate (0 to 1).
        avg_win: Average winning trade return.
        avg_loss: Average losing trade return (positive number).

    Returns:
        Optimal fraction of bankroll to risk (capped at 0.25).
    """
    if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
        return 0.01

    b = avg_win / avg_loss  # odds ratio
    kelly = (win_rate * b - (1 - win_rate)) / b

    # Half-Kelly for safety, capped at 25%
    kelly = max(0, kelly * 0.5)
    kelly = min(kelly, 0.25)

    return round(kelly, 4)
