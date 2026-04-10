"""Technical indicators — price action, trend, momentum, volatility.

All price-based features are ATR-normalized for pair-agnostic generalization.
"""

import numpy as np
import pandas as pd

try:
    import pandas_ta as ta
except ImportError:
    ta = None


# ═══════════════════════════════════════════════════════════════
#  PRICE ACTION FEATURES
# ═══════════════════════════════════════════════════════════════

def add_price_action(df: pd.DataFrame) -> pd.DataFrame:
    """Add price action features: returns, bar structure, spread."""
    c = df["close"]
    o = df["open"]
    h = df["high"]
    l = df["low"]

    # Log returns at multiple horizons
    for n in [1, 3, 5, 10, 20]:
        df[f"return_{n}"] = np.log(c / c.shift(n))

    # Bar structure (ATR-normalized later)
    bar_range = h - l
    bar_range_safe = bar_range.replace(0, np.nan)

    df["bar_range"] = bar_range
    df["bar_body"] = (c - o).abs() / bar_range_safe
    df["upper_wick"] = (h - pd.concat([o, c], axis=1).max(axis=1)) / bar_range_safe
    df["lower_wick"] = (pd.concat([o, c], axis=1).min(axis=1) - l) / bar_range_safe
    df["close_position"] = (c - l) / bar_range_safe

    # Gap
    df["gap_open"] = o - c.shift(1)

    # Spread features
    if "bid_close" in df.columns and "ask_close" in df.columns:
        df["spread_pips_raw"] = df["ask_close"] - df["bid_close"]
        spread_avg = df["spread_pips_raw"].rolling(20).mean()
        df["spread_ratio"] = df["spread_pips_raw"] / spread_avg.replace(0, np.nan)

    return df


# ═══════════════════════════════════════════════════════════════
#  TREND INDICATORS
# ═══════════════════════════════════════════════════════════════

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def add_trend_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add trend-following indicators: EMA, SMA distances, ADX, Aroon."""
    c = df["close"]

    # EMA distances (raw — will be ATR-normalized)
    for span in [8, 21, 50, 200]:
        ema = _ema(c, span)
        df[f"ema_{span}_dist"] = c - ema

    # EMA crosses
    df["ema_8_21_cross"] = _ema(c, 8) - _ema(c, 21)
    df["ema_21_50_cross"] = _ema(c, 21) - _ema(c, 50)

    # SMA distances
    df["sma_20_dist"] = c - _sma(c, 20)

    # SMA slopes
    sma_50 = _sma(c, 50)
    sma_200 = _sma(c, 200)
    df["sma_50_slope"] = sma_50 - sma_50.shift(5)
    df["sma_200_slope"] = sma_200 - sma_200.shift(10)

    # ADX / DI (using pandas_ta if available)
    if ta is not None:
        adx_df = ta.adx(df["high"], df["low"], c, length=14)
        if adx_df is not None and not adx_df.empty:
            df["adx_14"] = adx_df.iloc[:, 0]
            df["plus_di"] = adx_df.iloc[:, 1]
            df["minus_di"] = adx_df.iloc[:, 2]

        # Aroon
        aroon_df = ta.aroon(df["high"], df["low"], length=25)
        if aroon_df is not None and not aroon_df.empty:
            df["aroon_up"] = aroon_df.iloc[:, 0]
            df["aroon_down"] = aroon_df.iloc[:, 1]
            df["aroon_osc"] = aroon_df.iloc[:, 2] if aroon_df.shape[1] > 2 else df["aroon_up"] - df["aroon_down"]
    else:
        # Fallback: simple ADX approximation
        df["adx_14"] = np.nan
        df["plus_di"] = np.nan
        df["minus_di"] = np.nan
        df["aroon_up"] = np.nan
        df["aroon_down"] = np.nan
        df["aroon_osc"] = np.nan

    # Ichimoku cloud distance (simplified)
    tenkan = (df["high"].rolling(9).max() + df["low"].rolling(9).min()) / 2
    kijun = (df["high"].rolling(26).max() + df["low"].rolling(26).min()) / 2
    senkou_a = (tenkan + kijun) / 2
    senkou_b = (df["high"].rolling(52).max() + df["low"].rolling(52).min()) / 2
    cloud_mid = (senkou_a + senkou_b) / 2
    df["ichimoku_cloud_dist"] = c - cloud_mid
    df["price_above_cloud"] = np.where(c > pd.concat([senkou_a, senkou_b], axis=1).max(axis=1), 1,
                                np.where(c < pd.concat([senkou_a, senkou_b], axis=1).min(axis=1), -1, 0))

    return df


# ═══════════════════════════════════════════════════════════════
#  MOMENTUM INDICATORS
# ═══════════════════════════════════════════════════════════════

def add_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add momentum oscillators: RSI, Stochastic, MACD, CCI, Williams %R."""
    c = df["close"]
    h = df["high"]
    l = df["low"]

    # RSI
    for period in [7, 14]:
        delta = c.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        df[f"rsi_{period}"] = (rsi - 50) / 50  # Rescale to [-1, 1]

    # Stochastic
    low_14 = l.rolling(14).min()
    high_14 = h.rolling(14).max()
    denom = (high_14 - low_14).replace(0, np.nan)
    df["stoch_k"] = ((c - low_14) / denom - 0.5) * 2  # Rescale to [-1, 1]
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()
    df["stoch_kd_diff"] = df["stoch_k"] - df["stoch_d"]

    # MACD (raw values — ATR-normalized later)
    ema_12 = _ema(c, 12)
    ema_26 = _ema(c, 26)
    df["macd_line"] = ema_12 - ema_26
    df["macd_signal"] = _ema(df["macd_line"], 9)
    df["macd_hist"] = df["macd_line"] - df["macd_signal"]

    # CCI
    typical_price = (h + l + c) / 3
    sma_tp = typical_price.rolling(20).mean()
    mad = typical_price.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    df["cci_20"] = (typical_price - sma_tp) / (0.015 * mad.replace(0, np.nan)) / 200

    # Williams %R
    high_14_wr = h.rolling(14).max()
    low_14_wr = l.rolling(14).min()
    denom_wr = (high_14_wr - low_14_wr).replace(0, np.nan)
    df["williams_r"] = ((high_14_wr - c) / denom_wr - 0.5) * -2  # Rescale to [-1, 1]

    # Rate of Change
    df["roc_10"] = np.log(c / c.shift(10))

    # Momentum
    df["momentum_10"] = c - c.shift(10)

    # MFI (if volume available)
    if "volume" in df.columns:
        tp = (h + l + c) / 3
        mf = tp * df["volume"]
        pos_mf = pd.Series(np.where(tp > tp.shift(1), mf, 0), index=df.index).rolling(14).sum()
        neg_mf = pd.Series(np.where(tp <= tp.shift(1), mf, 0), index=df.index).rolling(14).sum()
        mfi = 100 - (100 / (1 + pos_mf / neg_mf.replace(0, np.nan)))
        df["mfi_14"] = (mfi - 50) / 50  # Rescale to [-1, 1]
    else:
        df["mfi_14"] = np.nan

    return df


# ═══════════════════════════════════════════════════════════════
#  VOLATILITY INDICATORS
# ═══════════════════════════════════════════════════════════════

def add_volatility_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add volatility indicators: ATR, Bollinger Bands, Keltner Channels, squeeze."""
    c = df["close"]
    h = df["high"]
    l = df["low"]

    # ATR calculation
    tr = pd.concat([
        h - l,
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs(),
    ], axis=1).max(axis=1)

    df["atr_14"] = tr.rolling(14).mean()
    atr_7 = tr.rolling(7).mean()
    atr_50 = tr.rolling(50).mean()

    df["atr_7_ratio"] = atr_7 / df["atr_14"].replace(0, np.nan)
    df["atr_50_ratio"] = df["atr_14"] / atr_50.replace(0, np.nan)

    # Bollinger Bands
    sma_20 = _sma(c, 20)
    std_20 = c.rolling(20).std()
    bb_upper = sma_20 + 2 * std_20
    bb_lower = sma_20 - 2 * std_20

    df["bb_upper_dist"] = c - bb_upper
    df["bb_lower_dist"] = c - bb_lower
    bb_range = (bb_upper - bb_lower).replace(0, np.nan)
    df["bb_pct_b"] = (c - bb_lower) / bb_range
    df["bb_pct_b"] = df["bb_pct_b"].clip(-0.5, 1.5)
    df["bb_bandwidth"] = bb_range

    # Keltner Channels
    ema_20 = _ema(c, 20)
    kc_upper = ema_20 + 1.5 * df["atr_14"]
    kc_lower = ema_20 - 1.5 * df["atr_14"]

    df["kc_upper_dist"] = c - kc_upper
    df["kc_lower_dist"] = c - kc_lower

    # Bollinger Squeeze (BB inside KC)
    df["bb_squeeze"] = ((bb_lower > kc_lower) & (bb_upper < kc_upper)).astype(int)

    # Historical volatility
    log_returns = np.log(c / c.shift(1))
    df["hist_vol_20"] = log_returns.rolling(20).std() * np.sqrt(252)
    vol_50_avg = df["hist_vol_20"].rolling(50).mean()
    df["vol_regime"] = df["hist_vol_20"] / vol_50_avg.replace(0, np.nan)

    return df


# ═══════════════════════════════════════════════════════════════
#  ATR NORMALIZATION
# ═══════════════════════════════════════════════════════════════

def normalize_by_atr(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize price-based features by ATR(14) for pair-agnostic generalization."""
    atr = df["atr_14"].replace(0, np.nan)

    # Columns that need ATR normalization
    atr_normalize_cols = [
        "bar_range", "gap_open",
        "ema_8_dist", "ema_21_dist", "ema_50_dist", "ema_200_dist",
        "ema_8_21_cross", "ema_21_50_cross",
        "sma_20_dist", "sma_50_slope", "sma_200_slope",
        "ichimoku_cloud_dist",
        "macd_line", "macd_signal", "macd_hist",
        "momentum_10",
        "bb_upper_dist", "bb_lower_dist", "bb_bandwidth",
        "kc_upper_dist", "kc_lower_dist",
    ]

    for col in atr_normalize_cols:
        if col in df.columns:
            df[col] = df[col] / atr

    return df
