"""OANDA REST API v20 wrapper with retry logic, rate limiting, and structured logging."""

import time
import logging
from datetime import datetime, timezone
from typing import Generator, Optional

import pandas as pd
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.trades as trades
import oandapyV20.endpoints.accounts as accounts
import oandapyV20.endpoints.pricing as pricing

logger = logging.getLogger("core.api_client")

OANDA_ENVIRONMENTS = {
    "live": "https://api-fxtrade.oanda.com",
    "practice": "https://api-fxpractice.oanda.com",
}

MAX_CANDLES_PER_REQUEST = 5000
MAX_RETRIES = 5
RETRY_BACKOFF_BASE = 2  # seconds


class APIClient:
    """Thin wrapper around oandapyV20 with retry, rate limiting, and logging."""

    def __init__(self, api_key: str, account_id: str, environment: str = "practice"):
        self.api_key = api_key
        self.account_id = account_id
        self.environment = environment
        hostname = OANDA_ENVIRONMENTS.get(environment, OANDA_ENVIRONMENTS["practice"])
        self.client = oandapyV20.API(access_token=api_key, environment=environment)
        self._last_request_time = 0.0
        self._min_request_interval = 0.05  # 20 req/s

    def _rate_limit(self):
        """Enforce minimum interval between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()

    def _request_with_retry(self, endpoint, max_retries: int = MAX_RETRIES):
        """Execute an API request with exponential backoff retry."""
        for attempt in range(1, max_retries + 1):
            self._rate_limit()
            try:
                response = self.client.request(endpoint)
                logger.debug(
                    "API request succeeded",
                    extra={"event": "api_request", "data": {
                        "endpoint": str(type(endpoint).__name__),
                        "attempt": attempt,
                    }},
                )
                return response
            except oandapyV20.exceptions.V20Error as e:
                status_code = getattr(e, "code", None)
                if status_code in (503, 429) and attempt < max_retries:
                    wait = RETRY_BACKOFF_BASE ** attempt
                    logger.warning(
                        f"API error {status_code}, retrying in {wait}s (attempt {attempt}/{max_retries})",
                        extra={"event": "api_retry", "data": {
                            "status_code": status_code, "attempt": attempt, "wait": wait,
                        }},
                    )
                    if status_code == 503:
                        logger.warning("Possible OANDA maintenance window",
                                       extra={"event": "maintenance_detected"})
                    time.sleep(wait)
                    continue
                logger.error(
                    f"API request failed after {attempt} attempts: {e}",
                    extra={"event": "api_error", "data": {
                        "status_code": status_code, "error": str(e),
                    }},
                )
                raise
            except Exception as e:
                if attempt < max_retries:
                    wait = RETRY_BACKOFF_BASE ** attempt
                    logger.warning(f"Connection error, retrying in {wait}s: {e}")
                    time.sleep(wait)
                    continue
                raise

    def get_candles(
        self,
        instrument: str,
        granularity: str,
        from_dt: Optional[datetime] = None,
        to_dt: Optional[datetime] = None,
        count: Optional[int] = None,
        price: str = "BA",
    ) -> pd.DataFrame:
        """Fetch candles for an instrument. Handles single request (no pagination)."""
        params = {"granularity": granularity, "price": price}
        if from_dt:
            params["from"] = from_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        if to_dt:
            params["to"] = to_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        if count:
            params["count"] = min(count, MAX_CANDLES_PER_REQUEST)

        endpoint = instruments.InstrumentsCandles(instrument=instrument, params=params)
        response = self._request_with_retry(endpoint)
        candles = response.get("candles", [])
        if not candles:
            return pd.DataFrame()

        return self._candles_to_df(candles, price)

    def _candles_to_df(self, candles: list, price: str = "BA") -> pd.DataFrame:
        """Convert OANDA candle JSON to a DataFrame."""
        rows = []
        for c in candles:
            row = {
                "time": pd.Timestamp(c["time"]),
                "volume": int(c["volume"]),
                "complete": c["complete"],
            }
            if "mid" in c:
                row.update({
                    "open": float(c["mid"]["o"]),
                    "high": float(c["mid"]["h"]),
                    "low": float(c["mid"]["l"]),
                    "close": float(c["mid"]["c"]),
                })
            if "bid" in c:
                row.update({
                    "bid_open": float(c["bid"]["o"]),
                    "bid_high": float(c["bid"]["h"]),
                    "bid_low": float(c["bid"]["l"]),
                    "bid_close": float(c["bid"]["c"]),
                })
            if "ask" in c:
                row.update({
                    "ask_open": float(c["ask"]["o"]),
                    "ask_high": float(c["ask"]["h"]),
                    "ask_low": float(c["ask"]["l"]),
                    "ask_close": float(c["ask"]["c"]),
                })
            # Compute mid from bid/ask if mid not present
            if "mid" not in c and "bid" in c and "ask" in c:
                row["open"] = (row["bid_open"] + row["ask_open"]) / 2
                row["high"] = (row["bid_high"] + row["ask_high"]) / 2
                row["low"] = (row["bid_low"] + row["ask_low"]) / 2
                row["close"] = (row["bid_close"] + row["ask_close"]) / 2

            if "bid_close" in row and "ask_close" in row:
                row["spread_pips"] = row["ask_close"] - row["bid_close"]

            rows.append(row)

        df = pd.DataFrame(rows)
        if not df.empty:
            df["time"] = pd.to_datetime(df["time"], utc=True)
            df = df.sort_values("time").reset_index(drop=True)
        return df

    def get_full_history(
        self,
        instrument: str,
        granularity: str,
        start: datetime,
        end: Optional[datetime] = None,
        price: str = "BA",
    ) -> pd.DataFrame:
        """Paginated download of full candle history (handles 5000-candle limit)."""
        if end is None:
            end = datetime.now(timezone.utc)

        all_dfs = []
        cursor = start
        request_count = 0

        while cursor < end:
            params = {
                "from": cursor.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "count": MAX_CANDLES_PER_REQUEST,
                "granularity": granularity,
                "price": price,
            }
            endpoint = instruments.InstrumentsCandles(instrument=instrument, params=params)
            response = self._request_with_retry(endpoint)
            candles = response.get("candles", [])
            request_count += 1

            if not candles:
                break

            complete = [c for c in candles if c["complete"]]
            if not complete:
                break

            df_batch = self._candles_to_df(complete, price)
            all_dfs.append(df_batch)

            last_time = pd.Timestamp(complete[-1]["time"])
            cursor = last_time.to_pydatetime().replace(tzinfo=timezone.utc)

            if len(candles) < MAX_CANDLES_PER_REQUEST:
                break

            logger.info(
                f"{instrument}/{granularity}: fetched {len(complete)} candles "
                f"(request {request_count}, cursor={cursor.isoformat()})",
                extra={"event": "data_downloaded"},
            )

        if not all_dfs:
            return pd.DataFrame()

        df = pd.concat(all_dfs, ignore_index=True)
        df = df.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)

        logger.info(
            f"{instrument}/{granularity}: total {len(df)} candles in {request_count} requests",
            extra={"event": "data_downloaded"},
        )
        return df

    def place_order(self, order: dict) -> dict:
        """Place an order on OANDA."""
        endpoint = orders.OrderCreate(accountID=self.account_id, data=order)
        return self._request_with_retry(endpoint)

    def modify_trade(self, trade_id: str, sl: Optional[float] = None, tp: Optional[float] = None) -> dict:
        """Modify SL/TP on an open trade."""
        data = {}
        if sl is not None:
            data["stopLoss"] = {"price": f"{sl:.5f}"}
        if tp is not None:
            data["takeProfit"] = {"price": f"{tp:.5f}"}
        endpoint = trades.TradeCRCDO(accountID=self.account_id, tradeID=trade_id, data=data)
        return self._request_with_retry(endpoint)

    def close_trade(self, trade_id: str) -> dict:
        """Close an open trade."""
        endpoint = trades.TradeClose(accountID=self.account_id, tradeID=trade_id)
        return self._request_with_retry(endpoint)

    def get_account_summary(self) -> dict:
        """Get account summary (balance, NAV, margin, etc.)."""
        endpoint = accounts.AccountSummary(accountID=self.account_id)
        response = self._request_with_retry(endpoint)
        return response.get("account", {})

    def get_open_trades(self) -> list[dict]:
        """Get all open trades for the account."""
        endpoint = trades.OpenTrades(accountID=self.account_id)
        response = self._request_with_retry(endpoint)
        return response.get("trades", [])

    def stream_prices(self, instrument_list: list[str]) -> Generator:
        """Stream live bid/ask prices. Yields tick dicts."""
        params = {"instruments": ",".join(instrument_list)}
        endpoint = pricing.PricingStream(accountID=self.account_id, params=params)
        try:
            for tick in self.client.request(endpoint):
                if tick.get("type") == "PRICE":
                    yield {
                        "instrument": tick["instrument"],
                        "time": tick["time"],
                        "bid": float(tick["bids"][0]["price"]),
                        "ask": float(tick["asks"][0]["price"]),
                        "spread": float(tick["asks"][0]["price"]) - float(tick["bids"][0]["price"]),
                    }
        except Exception as e:
            logger.error(f"Price stream error: {e}", extra={"event": "stream_error"})
            raise
