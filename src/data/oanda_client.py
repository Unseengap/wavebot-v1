"""OANDA REST API v20 client for historical data collection."""
import time
import requests
import pandas as pd
from datetime import datetime, timezone


class OandaClient:
    MAX_CANDLES = 5000

    def __init__(self, api_token: str, account_id: str,
                 environment: str = "practice"):
        envs = {
            "practice": "https://api-fxpractice.oanda.com",
            "live": "https://api-fxtrade.oanda.com",
        }
        self.base_url = envs.get(environment, envs["practice"])
        self.account_id = account_id
        self.headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
        }

    def get_candles(self, instrument: str, granularity: str,
                    from_time: str = None, to_time: str = None,
                    count: int = None) -> list:
        """Fetch candles from OANDA. Returns list of candle dicts."""
        url = f"{self.base_url}/v3/instruments/{instrument}/candles"
        params = {
            "price": "MBA",
            "granularity": granularity,
        }
        if from_time:
            params["from"] = from_time
            params["includeFirst"] = "true"
        if to_time:
            params["to"] = to_time
        if count and not (from_time and to_time):
            params["count"] = min(count, self.MAX_CANDLES)

        resp = requests.get(url, headers=self.headers, params=params)

        if resp.status_code == 429:
            retry = int(resp.headers.get("Retry-After", 5))
            time.sleep(retry)
            return self.get_candles(instrument, granularity,
                                    from_time, to_time, count)

        if resp.status_code not in (200, 201):
            raise Exception(f"OANDA API error {resp.status_code}: {resp.text[:200]}")

        return resp.json().get("candles", [])

    def collect_range(self, instrument: str, granularity: str,
                      start: str, end: str = None) -> pd.DataFrame:
        """
        Download full date range with automatic pagination.
        Returns DataFrame with mid, bid, ask OHLC columns.
        """
        all_candles = []
        current_start = start
        page = 0

        while True:
            page += 1
            params_from = current_start

            candles = self.get_candles(
                instrument, granularity,
                from_time=params_from,
                to_time=end,
                count=self.MAX_CANDLES,
            )

            if not candles:
                break

            complete = [c for c in candles if c.get("complete", False)]
            all_candles.extend(complete)

            if len(candles) < self.MAX_CANDLES:
                break

            last_time = candles[-1]["time"]
            if last_time == current_start:
                break
            current_start = last_time
            time.sleep(0.2)

            if page % 10 == 0:
                print(f"  Page {page}: {len(all_candles)} candles so far...")

        return self._to_dataframe(all_candles)

    def _to_dataframe(self, candles: list) -> pd.DataFrame:
        rows = []
        for c in candles:
            row = {"time": c["time"], "volume": c.get("volume", 0)}
            for price_type in ["mid", "bid", "ask"]:
                p = c.get(price_type, {})
                if p:
                    row[f"open_{price_type}"] = float(p.get("o", 0))
                    row[f"high_{price_type}"] = float(p.get("h", 0))
                    row[f"low_{price_type}"] = float(p.get("l", 0))
                    row[f"close_{price_type}"] = float(p.get("c", 0))
            rows.append(row)

        df = pd.DataFrame(rows)
        if not df.empty:
            df["time"] = pd.to_datetime(df["time"])
            df = df.sort_values("time").reset_index(drop=True)
            df = df.drop_duplicates(subset=["time"], keep="last")
        return df
