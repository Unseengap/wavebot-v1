"""Order manager — translates signals into OANDA API order requests."""

import logging
from typing import Optional

from src.risk.position_sizer import fixed_fraction_size, get_pip_size

logger = logging.getLogger("execution.order_manager")


class OrderManager:
    """Places and manages orders on OANDA.

    Supports market orders with SL/TP, limit entries, stop entries,
    and trailing stop modifications.
    """

    def __init__(self, api_client, max_slippage_spread_multiple: float = 1.5):
        self.api_client = api_client
        self.max_slippage_spread_multiple = max_slippage_spread_multiple

    def execute_signal(
        self,
        signal: dict,
        instrument: str,
        account_summary: dict,
        spread: Optional[float] = None,
    ) -> Optional[dict]:
        """Convert an approved signal into an OANDA order.

        Args:
            signal: Validated signal from the ensemble (action, sl_pips, tp_pips, position_size).
            instrument: OANDA instrument code (e.g. 'EUR_USD').
            account_summary: Current account state from OANDA.
            spread: Current bid-ask spread in price units.

        Returns:
            Order result dict or None if order failed.
        """
        action = signal.get("action", "flat")
        if action == "flat":
            return None

        balance = float(account_summary.get("balance", 0))
        risk_pct = signal.get("position_size", 0.01)
        sl_pips = signal.get("sl_pips", 0)
        tp_pips = signal.get("tp_pips", 0)

        pip_size = get_pip_size(instrument)

        # Convert pips to price distance
        sl_price_dist = sl_pips * pip_size
        tp_price_dist = tp_pips * pip_size

        # Get current price
        try:
            candle = self.api_client.get_candles(instrument=instrument, count=1, price="BA")
            if candle.empty:
                logger.error(f"No price data for {instrument}")
                return None

            current_bid = candle.iloc[-1].get("bid_close", candle.iloc[-1]["close"])
            current_ask = candle.iloc[-1].get("ask_close", candle.iloc[-1]["close"])
            current_spread = current_ask - current_bid

        except Exception as e:
            logger.error(f"Failed to get current price for {instrument}: {e}")
            return None

        # Compute position size
        pip_value = 10.0  # approximate for standard lot; TODO: compute from account currency
        units = fixed_fraction_size(balance, risk_pct, sl_pips, pip_value)
        if units == 0:
            logger.warning(f"Position size is 0 for {instrument}")
            return None

        # Determine entry price and SL/TP
        if action == "long":
            entry_price = current_ask
            sl_price = entry_price - sl_price_dist
            tp_price = entry_price + tp_price_dist
            units_signed = units
            price_bound = entry_price + current_spread * self.max_slippage_spread_multiple
        else:  # short
            entry_price = current_bid
            sl_price = entry_price + sl_price_dist
            tp_price = entry_price - tp_price_dist
            units_signed = -units
            price_bound = entry_price - current_spread * self.max_slippage_spread_multiple

        # Build OANDA order
        order_data = {
            "order": {
                "type": "MARKET",
                "instrument": instrument,
                "units": str(units_signed),
                "priceBound": f"{price_bound:.5f}",
                "stopLossOnFill": {
                    "price": f"{sl_price:.5f}",
                },
                "takeProfitOnFill": {
                    "price": f"{tp_price:.5f}",
                },
                "timeInForce": "FOK",
            }
        }

        try:
            result = self.api_client.place_order(order_data)
            order_fill = result.get("orderFillTransaction", {})

            trade_result = {
                "instrument": instrument,
                "direction": action,
                "units": abs(units_signed),
                "entry_price": float(order_fill.get("price", entry_price)),
                "sl_price": sl_price,
                "tp_price": tp_price,
                "sl_pips": sl_pips,
                "tp_pips": tp_pips,
                "rr_ratio": round(tp_pips / sl_pips, 2) if sl_pips > 0 else 0,
                "risk_pct": risk_pct,
                "spread_at_entry": round(current_spread / pip_size, 1),
                "model_confidence": signal.get("confidence", 0),
                "trade_id": order_fill.get("tradeOpened", {}).get("tradeID", ""),
            }

            logger.info(
                f"Order filled: {action} {instrument} {units} units",
                extra={"event": "trade_opened", "data": trade_result},
            )
            return trade_result

        except Exception as e:
            logger.error(
                f"Order failed for {instrument}: {e}",
                extra={"event": "order_rejected", "data": {
                    "instrument": instrument, "error": str(e),
                }},
            )
            return None

    def place_limit_order(
        self,
        instrument: str,
        direction: str,
        entry_price: float,
        sl_price: float,
        tp_price: float,
        units: int,
        expiry_seconds: int = 3600,
    ) -> Optional[dict]:
        """Place a limit entry order."""
        units_signed = units if direction == "long" else -units

        order_data = {
            "order": {
                "type": "LIMIT",
                "instrument": instrument,
                "units": str(units_signed),
                "price": f"{entry_price:.5f}",
                "stopLossOnFill": {"price": f"{sl_price:.5f}"},
                "takeProfitOnFill": {"price": f"{tp_price:.5f}"},
                "timeInForce": "GTD",
                "gtdTime": "",  # TODO: compute from expiry_seconds
            }
        }

        try:
            result = self.api_client.place_order(order_data)
            return result
        except Exception as e:
            logger.error(f"Limit order failed: {e}")
            return None

    def modify_sl_tp(self, trade_id: str, sl: Optional[float] = None, tp: Optional[float] = None) -> bool:
        """Modify SL/TP on an open trade."""
        try:
            self.api_client.modify_trade(trade_id, sl=sl, tp=tp)
            logger.info(f"Modified trade {trade_id}: SL={sl}, TP={tp}",
                        extra={"event": "trade_modified"})
            return True
        except Exception as e:
            logger.error(f"Failed to modify trade {trade_id}: {e}")
            return False

    def close_trade(self, trade_id: str) -> bool:
        """Close an open trade."""
        try:
            self.api_client.close_trade(trade_id)
            logger.info(f"Closed trade {trade_id}", extra={"event": "trade_closed"})
            return True
        except Exception as e:
            logger.error(f"Failed to close trade {trade_id}: {e}")
            return False
