"""Variable spread simulation from bid/ask or session-based model."""


BASE_SPREADS = {
    "EUR_USD": 1.2, "GBP_USD": 1.6, "USD_JPY": 1.2, "USD_CAD": 1.8,
    "AUD_USD": 1.5, "USD_CHF": 1.6, "NZD_USD": 2.0, "EUR_GBP": 1.8,
    "GBP_JPY": 2.8, "EUR_JPY": 1.8, "XAU_USD": 25.0, "XAG_USD": 30.0,
}

SESSION_SPREAD_MULT = {
    "London_NY_Overlap": 0.85,   # Tightest spreads
    "London_Open":       0.90,
    "NY_Session":        1.00,
    "Asia_Session":      1.40,
    "Dead_Zone":         1.80,
}


class SpreadSimulator:
    def get_spread_pips(self, candle: dict, instrument: str,
                        session: str, pip_size: float) -> float:
        # Direct from bid/ask if available
        if "close_ask" in candle and "close_bid" in candle:
            ask = candle["close_ask"]
            bid = candle["close_bid"]
            if ask and bid and pip_size > 0:
                return (ask - bid) / pip_size

        # Model fallback
        base = BASE_SPREADS.get(instrument, 1.5)
        mult = SESSION_SPREAD_MULT.get(session, 1.0)
        return base * mult

    def apply_to_entry(self, direction: str, mid_price: float,
                       spread_pips: float, pip_size: float) -> float:
        half = (spread_pips * pip_size) / 2
        if direction == "LONG":
            return mid_price + half   # Buy at ask
        else:
            return mid_price - half   # Sell at bid

    def apply_to_exit(self, direction: str, mid_price: float,
                      spread_pips: float, pip_size: float) -> float:
        half = (spread_pips * pip_size) / 2
        if direction == "LONG":
            return mid_price - half   # Sell at bid
        else:
            return mid_price + half   # Buy at ask
