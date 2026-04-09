"""Parquet storage helpers — save, load, append candle data with quality controls."""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger("data.storage")

REQUIRED_COLUMNS = ["time", "open", "high", "low", "close", "volume", "complete"]


def save_candles(df: pd.DataFrame, path: str | Path):
    """Save a candle DataFrame to Parquet with schema enforcement."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df = df.sort_values("time").reset_index(drop=True)
    df.to_parquet(path, index=False, engine="pyarrow")
    logger.debug(f"Saved {len(df)} rows to {path}")


def load_candles(path: str | Path) -> pd.DataFrame:
    """Load candle data from Parquet with type coercion."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_parquet(path, engine="pyarrow")
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True)
    return df


def append_candles(new_df: pd.DataFrame, path: str | Path) -> pd.DataFrame:
    """Append new candles to an existing Parquet file, deduplicating."""
    existing = load_candles(path)
    if existing.empty:
        save_candles(new_df, path)
        return new_df

    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
    save_candles(combined, path)
    return combined


def detect_gaps(df: pd.DataFrame, granularity_seconds: int, ignore_weekends: bool = True) -> pd.DataFrame:
    """Detect gaps in candle data exceeding 3x the expected bar interval.

    Returns a DataFrame of gap start/end times.
    """
    if df.empty or len(df) < 2:
        return pd.DataFrame(columns=["gap_start", "gap_end", "gap_seconds"])

    df = df.sort_values("time").reset_index(drop=True)
    time_diff = df["time"].diff().dt.total_seconds()

    threshold = granularity_seconds * 3
    gaps = time_diff[time_diff > threshold]

    gap_records = []
    for idx in gaps.index:
        gap_start = df.loc[idx - 1, "time"]
        gap_end = df.loc[idx, "time"]

        # Skip Friday 22:00 UTC -> Sunday 22:00 UTC (expected weekend gap)
        if ignore_weekends and gap_start.weekday() == 4 and gap_end.weekday() == 6:
            continue

        gap_records.append({
            "gap_start": gap_start,
            "gap_end": gap_end,
            "gap_seconds": time_diff[idx],
        })

    if gap_records:
        logger.warning(
            f"Detected {len(gap_records)} data gap(s)",
            extra={"event": "data_gap_detected", "data": {"count": len(gap_records)}},
        )

    return pd.DataFrame(gap_records)


def detect_anomalies(df: pd.DataFrame, atr_multiplier: float = 5.0) -> pd.DataFrame:
    """Flag bars where range exceeds atr_multiplier × ATR(20)."""
    if df.empty or len(df) < 20:
        return pd.DataFrame()

    df = df.copy()
    bar_range = df["high"] - df["low"]

    # Simple ATR approximation using rolling mean of bar range
    atr_20 = bar_range.rolling(20).mean()
    threshold = atr_20 * atr_multiplier

    anomalies = df[bar_range > threshold].copy()
    if not anomalies.empty:
        anomalies["bar_range"] = bar_range[anomalies.index]
        anomalies["atr_20"] = atr_20[anomalies.index]
        logger.warning(
            f"Detected {len(anomalies)} anomalous bar(s)",
            extra={"event": "price_anomaly", "data": {"count": len(anomalies)}},
        )

    return anomalies


def remove_incomplete_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Filter out bars that are not complete (still forming)."""
    if "complete" in df.columns:
        return df[df["complete"] == True].reset_index(drop=True)
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate timestamps."""
    before = len(df)
    df = df.drop_duplicates(subset=["time"]).reset_index(drop=True)
    removed = before - len(df)
    if removed > 0:
        logger.debug(f"Removed {removed} duplicate timestamp(s)")
    return df
