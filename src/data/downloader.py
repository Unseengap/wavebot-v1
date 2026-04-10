"""Historical data downloader with pagination — fetches full candle history from OANDA."""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from src.core.api_client import APIClient
from src.data.storage import save_candles, load_candles

logger = logging.getLogger("data.downloader")


def download_pair_history(
    client: APIClient,
    instrument: str,
    granularity: str,
    start: datetime,
    end: Optional[datetime] = None,
    output_dir: str = "data/raw",
) -> pd.DataFrame:
    """Download full candle history for one instrument+granularity and save to Parquet."""
    if end is None:
        end = datetime.now(timezone.utc)

    logger.info(f"Downloading {instrument}/{granularity}: {start.date()} → {end.date()}")

    df = client.get_full_history(
        instrument=instrument,
        granularity=granularity,
        start=start,
        end=end,
        price="BA",
    )

    if df.empty:
        logger.warning(f"No data returned for {instrument}/{granularity}")
        return df

    out_path = Path(output_dir) / instrument / f"{granularity}.parquet"
    save_candles(df, out_path)

    logger.info(
        f"Saved {len(df)} candles to {out_path}",
        extra={"event": "data_downloaded", "data": {
            "instrument": instrument, "granularity": granularity,
            "rows": len(df), "path": str(out_path),
        }},
    )
    return df


def download_incremental(
    client: APIClient,
    instrument: str,
    granularity: str,
    data_dir: str = "data/raw",
) -> pd.DataFrame:
    """Fetch only candles newer than the last stored timestamp and append."""
    path = Path(data_dir) / instrument / f"{granularity}.parquet"
    existing = load_candles(path)

    if existing.empty:
        logger.info(f"No existing data for {instrument}/{granularity}, doing full download")
        return download_pair_history(
            client, instrument, granularity,
            start=datetime(2010, 1, 1, tzinfo=timezone.utc),
            output_dir=data_dir,
        )

    last_time = existing["time"].max()
    start = last_time.to_pydatetime().replace(tzinfo=timezone.utc)

    logger.info(f"Incremental update {instrument}/{granularity} from {start.isoformat()}")

    new_df = client.get_full_history(
        instrument=instrument,
        granularity=granularity,
        start=start,
        end=datetime.now(timezone.utc),
        price="BA",
    )

    if new_df.empty:
        logger.info(f"No new data for {instrument}/{granularity}")
        return existing

    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)

    save_candles(combined, path)
    logger.info(f"Appended {len(new_df)} new candles to {path} (total: {len(combined)})")
    return combined


def download_all_pairs(
    client: APIClient,
    pairs: list[str],
    granularities: list[str],
    start: datetime,
    end: Optional[datetime] = None,
    output_dir: str = "data/raw",
    mode: str = "full",
) -> dict[str, pd.DataFrame]:
    """Download data for multiple pairs and granularities."""
    results = {}
    for pair in pairs:
        for gran in granularities:
            key = f"{pair}/{gran}"
            try:
                if mode == "incremental":
                    df = download_incremental(client, pair, gran, output_dir)
                else:
                    df = download_pair_history(client, pair, gran, start, end, output_dir)
                results[key] = df
            except Exception as e:
                logger.error(f"Failed to download {key}: {e}")
                results[key] = pd.DataFrame()
    return results
