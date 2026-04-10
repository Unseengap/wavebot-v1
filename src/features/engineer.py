"""Feature pipeline orchestrator — transforms raw OANDA candles into model-ready features."""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.features.indicators import (
    add_price_action,
    add_trend_indicators,
    add_momentum_indicators,
    add_volatility_indicators,
    normalize_by_atr,
)
from src.features.patterns import (
    add_candlestick_patterns,
    add_chart_patterns,
    add_support_resistance,
)
from src.features.mtf import merge_higher_tf, add_mtf_alignment

logger = logging.getLogger("features.engineer")

# Timeframe label mapping for MTF column naming
TF_LABELS = {
    "M5": "m5", "M15": "m15", "M30": "m30",
    "H1": "h1", "H2": "h2", "H4": "h4",
    "H8": "h8", "H12": "h12", "D": "d", "W": "w",
}


class FeatureEngineer:
    """Transforms raw OANDA OHLCV candle data into a structured feature matrix.

    Features are grouped into: price action, trend, momentum, volatility,
    patterns, support/resistance, multi-timeframe, and session/time.
    All price-based features are ATR-normalized for pair-agnostic generalization.
    """

    def __init__(
        self,
        base_timeframe: str = "M15",
        higher_timeframes: Optional[list[str]] = None,
        lookback_bars: int = 128,
        include_volume: bool = True,
        include_patterns: bool = True,
        include_mtf: bool = True,
        chart_pattern_lookback: int = 50,
    ):
        self.base_tf = base_timeframe
        self.higher_tfs = higher_timeframes or ["H1", "H4", "D"]
        self.lookback = lookback_bars
        self.include_volume = include_volume
        self.include_patterns = include_patterns
        self.include_mtf = include_mtf
        self.chart_pattern_lookback = chart_pattern_lookback

    def build(self, raw_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Build the full feature matrix from multi-timeframe raw candle data.

        Args:
            raw_data: Dict mapping timeframe codes ('M15', 'H1', etc.) to candle DataFrames.

        Returns:
            Feature DataFrame aligned to base timeframe bars.
        """
        if self.base_tf not in raw_data:
            raise ValueError(f"Base timeframe {self.base_tf} not found in raw_data")

        base = raw_data[self.base_tf].copy()
        logger.info(f"Building features from {len(base)} {self.base_tf} bars")

        # ── Price action features ──────────────────────────────
        base = add_price_action(base)

        # ── Volatility (ATR needed for normalization) ──────────
        base = add_volatility_indicators(base)

        # ── Trend indicators ──────────────────────────────────
        base = add_trend_indicators(base)

        # ── Momentum indicators ───────────────────────────────
        base = add_momentum_indicators(base)

        # ── Candlestick patterns ─────────────────────────────
        if self.include_patterns:
            base = add_candlestick_patterns(base)
            base = add_chart_patterns(base, lookback=self.chart_pattern_lookback)
            base = add_support_resistance(base)

        # ── Multi-timeframe alignment ─────────────────────────
        if self.include_mtf:
            tf_labels_used = []
            for tf in self.higher_tfs:
                if tf in raw_data and not raw_data[tf].empty:
                    label = TF_LABELS.get(tf, tf.lower())
                    base = merge_higher_tf(base, raw_data[tf], label)
                    tf_labels_used.append(label)

            base = add_mtf_alignment(base, tf_labels_used)

        # ── Session / time features ──────────────────────────
        base = self._add_session_features(base)

        # ── ATR normalization ─────────────────────────────────
        base = normalize_by_atr(base)

        # ── Drop NaN rows from indicator warmup ──────────────
        before = len(base)
        base = base.dropna().reset_index(drop=True)
        dropped = before - len(base)
        if dropped > 0:
            logger.info(f"Dropped {dropped} NaN rows from indicator warmup")

        logger.info(f"Feature matrix: {base.shape[0]} rows × {base.shape[1]} columns")
        return base

    def _add_session_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-of-day and session features with cyclical encoding."""
        if "time" not in df.columns:
            return df

        t = pd.to_datetime(df["time"], utc=True)
        hour = t.dt.hour + t.dt.minute / 60.0

        # Cyclical encoding
        df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

        # Day of week (one-hot)
        dow = t.dt.dayofweek  # 0=Monday
        for d in range(5):
            df[f"dow_{d}"] = (dow == d).astype(int)

        # Session flags
        h = t.dt.hour
        m = t.dt.minute
        time_decimal = h + m / 60.0

        df["is_london_open"] = ((time_decimal >= 8.0) & (time_decimal <= 9.5)).astype(int)
        df["is_ny_open"] = ((time_decimal >= 13.5) & (time_decimal <= 15.0)).astype(int)
        df["is_overlap"] = ((time_decimal >= 13.5) & (time_decimal <= 16.5)).astype(int)
        df["is_asia_session"] = ((time_decimal >= 23.0) | (time_decimal <= 8.0)).astype(int)
        df["is_low_liquidity"] = (
            ((dow == 4) & (time_decimal > 16.5)) | df["is_asia_session"].astype(bool)
        ).astype(int)

        return df

    def get_feature_columns(self, df: pd.DataFrame) -> list[str]:
        """Return the list of feature columns (excluding metadata)."""
        exclude = {"time", "pair", "open", "high", "low", "close", "volume",
                   "complete", "bid_open", "bid_high", "bid_low", "bid_close",
                   "ask_open", "ask_high", "ask_low", "ask_close",
                   "spread_pips", "spread_pips_raw",
                   "label_h1", "label_h3", "label_h5"}
        return [c for c in df.columns if c not in exclude]
