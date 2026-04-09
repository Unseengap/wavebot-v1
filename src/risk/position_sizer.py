"""Position sizing: fixed-fraction with session scaling."""


def calculate_position_size(
    account_balance: float,
    risk_fraction: float,
    sl_distance_pips: float,
    pip_value_per_unit: float,
) -> int:
    """
    Units = risk_amount / (sl_distance * pip_value_per_unit).
    Returns integer units. OANDA minimum is 1 unit.
    """
    if sl_distance_pips <= 0 or pip_value_per_unit <= 0:
        return 0

    risk_amount = account_balance * risk_fraction
    risk_per_unit = sl_distance_pips * pip_value_per_unit
    units = int(risk_amount / risk_per_unit)

    return max(1000, units)  # Minimum 0.01 mini lot


def apply_session_to_position_size(base_units: int, session_multiplier: float) -> int:
    """Scale position size by session quality. 0.0 = no trade."""
    if session_multiplier <= 0:
        return 0
    adjusted = int(base_units * session_multiplier)
    return max(1000, adjusted) if adjusted > 0 else 0
