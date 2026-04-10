"""Multi-timeframe alignment — merges higher-TF context onto base timeframe bars."""

import numpy as np
import pandas as pd

from src.features.indicators import _ema, _sma


def compute_htf_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute trend/momentum features for a higher-timeframe DataFrame."""
    c = df["close"]
    h = df["high"]
    l = df["low"]

    result = df[["time"]].copy()

    # Trend direction: EMA(8) vs EMA(21)
    ema_8 = _ema(c, 8)
    ema_21 = _ema(c, 21)
    result["trend_dir"] = np.where(ema_8 > ema_21, 1, np.where(ema_8 < ema_21, -1, 0))

    # RSI(14) rescaled to [-1, 1]
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    result["rsi"] = (rsi - 50) / 50

    # ADX(14)
    tr = pd.concat([
        h - l,
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()

    plus_dm = pd.Series(np.where((h - h.shift(1)) > (l.shift(1) - l), np.maximum(h - h.shift(1), 0), 0), index=df.index)
    minus_dm = pd.Series(np.where((l.shift(1) - l) > (h - h.shift(1)), np.maximum(l.shift(1) - l, 0), 0), index=df.index)

    plus_di = 100 * plus_dm.rolling(14).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.rolling(14).mean() / atr.replace(0, np.nan)

    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    result["adx"] = dx.rolling(14).mean()

    # Above EMA(200)
    ema_200 = _ema(c, 200)
    result["above_ema200"] = (c > ema_200).astype(int)

    return result


def merge_higher_tf(base_df: pd.DataFrame, htf_df: pd.DataFrame, tf_label: str) -> pd.DataFrame:
    """Merge higher-timeframe features onto base timeframe using forward-fill alignment.

    Each higher TF bar applies to all base bars that fall within its time window.
    """
    htf_features = compute_htf_features(htf_df)

    # Rename columns with TF prefix
    rename_map = {
        "trend_dir": f"{tf_label}_trend_dir",
        "rsi": f"{tf_label}_rsi",
        "adx": f"{tf_label}_adx",
        "above_ema200": f"{tf_label}_above_ema200",
    }
    htf_features = htf_features.rename(columns=rename_map)

    # Merge using asof join (forward-fill higher TF values to base TF)
    base_df = base_df.sort_values("time")
    htf_features = htf_features.sort_values("time")

    merged = pd.merge_asof(
        base_df,
        htf_features,
        on="time",
        direction="backward",
    )

    return merged


def add_mtf_alignment(df: pd.DataFrame, tf_labels: list[str] = None) -> pd.DataFrame:
    """Compute MTF alignment score from existing higher-TF trend direction columns."""
    if tf_labels is None:
        tf_labels = ["h1", "h4", "d"]

    trend_cols = [f"{tf}_trend_dir" for tf in tf_labels if f"{tf}_trend_dir" in df.columns]

    if not trend_cols:
        df["mtf_alignment"] = 0
        df["mtf_agreement"] = 0
        return df

    alignment = df[trend_cols].sum(axis=1)
    df["mtf_alignment"] = alignment
    df["mtf_agreement"] = alignment.abs() / len(trend_cols)

    return df
