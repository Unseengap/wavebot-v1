"""Slippage simulation based on instrument and volatility."""


BASE_SLIPPAGE_PIPS = {
    "EUR_USD": 0.3, "GBP_USD": 0.5, "USD_JPY": 0.3, "USD_CAD": 0.4,
    "AUD_USD": 0.4, "USD_CHF": 0.4, "NZD_USD": 0.5, "EUR_GBP": 0.5,
    "GBP_JPY": 0.8, "EUR_JPY": 0.5, "XAU_USD": 1.5, "XAG_USD": 2.0,
}

VOLATILITY_THRESHOLD = 2.0     # candle_range > 2x ATR
VOLATILITY_MULTIPLIER = 2.5    # slippage up to 2.5x base


class SlippageSimulator:
    def get_slippage_pips(self, instrument: str,
                          candle_range_pips: float,
                          atr_pips: float) -> float:
        base = BASE_SLIPPAGE_PIPS.get(instrument, 0.5)
        if atr_pips > 0 and candle_range_pips > VOLATILITY_THRESHOLD * atr_pips:
            return base * VOLATILITY_MULTIPLIER
        return base

    def apply_to_fill(self, direction: str, fill_type: str,
                      price: float, slippage_pips: float,
                      pip_size: float) -> float:
        """Slippage always works AGAINST you."""
        slip = slippage_pips * pip_size
        if direction == "LONG":
            return price + slip if fill_type == "ENTRY" else price - slip
        else:
            return price - slip if fill_type == "ENTRY" else price + slip
