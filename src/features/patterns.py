"""Pattern recognition — candlestick patterns, chart patterns, support/resistance."""

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════
#  CANDLESTICK PATTERNS (binary: 1=bullish, -1=bearish, 0=none)
# ═══════════════════════════════════════════════════════════════

def add_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Detect 15 candlestick patterns as binary flags."""
    o = df["open"]
    h = df["high"]
    l = df["low"]
    c = df["close"]

    body = (c - o).abs()
    bar_range = (h - l).replace(0, np.nan)
    upper_wick = h - pd.concat([o, c], axis=1).max(axis=1)
    lower_wick = pd.concat([o, c], axis=1).min(axis=1) - l

    prev_o = o.shift(1)
    prev_c = c.shift(1)
    prev_h = h.shift(1)
    prev_l = l.shift(1)
    prev_body = (prev_c - prev_o).abs()
    prev_range = (prev_h - prev_l).replace(0, np.nan)

    # 1. Doji — body < 10% of range
    df["pat_doji"] = (body / bar_range < 0.1).astype(int)

    # 2. Hammer — small body at top, long lower wick (>60%)
    is_hammer = (lower_wick / bar_range > 0.6) & (body / bar_range < 0.3)
    df["pat_hammer"] = np.where(is_hammer & (c > o), 1, np.where(is_hammer & (c < o), -1, 0))

    # 3. Engulfing
    bull_eng = (c > o) & (prev_c < prev_o) & (o <= prev_c) & (c >= prev_o)
    bear_eng = (c < o) & (prev_c > prev_o) & (o >= prev_c) & (c <= prev_o)
    df["pat_engulfing"] = np.where(bull_eng, 1, np.where(bear_eng, -1, 0))

    # 4. Morning/Evening Star (3-bar pattern)
    pp_c = c.shift(2)
    pp_o = o.shift(2)
    small_middle = body.shift(1) / bar_range.shift(1) < 0.3
    morning = (pp_c < pp_o) & small_middle & (c > o) & (c > (pp_o + pp_c) / 2)
    evening = (pp_c > pp_o) & small_middle & (c < o) & (c < (pp_o + pp_c) / 2)
    df["pat_morning_star"] = np.where(morning, 1, np.where(evening, -1, 0))

    # 5. Shooting Star — small body at bottom, long upper wick (>60%)
    is_star = (upper_wick / bar_range > 0.6) & (body / bar_range < 0.3)
    df["pat_shooting_star"] = np.where(is_star & (c < o), -1, np.where(is_star & (c > o), 1, 0))

    # 6. Inside Bar — current range entirely within previous range
    df["pat_inside_bar"] = ((h <= prev_h) & (l >= prev_l)).astype(int)

    # 7. Outside Bar — current engulfs previous range
    df["pat_outside_bar"] = ((h > prev_h) & (l < prev_l)).astype(int)

    # 8. Pin Bar — wick > 60% of range on one side, body in opposite third
    bull_pin = (lower_wick / bar_range > 0.6) & ((c - l) / bar_range > 0.7)
    bear_pin = (upper_wick / bar_range > 0.6) & ((h - c) / bar_range > 0.7)
    df["pat_pin_bar"] = np.where(bull_pin, 1, np.where(bear_pin, -1, 0))

    # 9. Tweezer — same high or same low as previous (within ATR*0.05)
    atr = df.get("atr_14", bar_range.rolling(14).mean())
    tol = atr * 0.05
    tweezer_top = ((h - prev_h).abs() < tol) & (c < o) & (prev_c > prev_o)
    tweezer_bot = ((l - prev_l).abs() < tol) & (c > o) & (prev_c < prev_o)
    df["pat_tweezer"] = np.where(tweezer_bot, 1, np.where(tweezer_top, -1, 0))

    # 10. Three White Soldiers / Three Black Crows
    c1, c2, c3 = c, c.shift(1), c.shift(2)
    o1, o2, o3 = o, o.shift(1), o.shift(2)
    soldiers = (c1 > o1) & (c2 > o2) & (c3 > o3) & (c1 > c2) & (c2 > c3)
    crows = (c1 < o1) & (c2 < o2) & (c3 < o3) & (c1 < c2) & (c2 < c3)
    df["pat_three_soldiers"] = np.where(soldiers, 1, np.where(crows, -1, 0))

    # 11. Dark Cloud Cover
    dark_cloud = (prev_c > prev_o) & (o > prev_h) & (c < (prev_o + prev_c) / 2) & (c > prev_o)
    df["pat_dark_cloud"] = np.where(dark_cloud, -1, 0)

    # 12. Piercing Line
    piercing = (prev_c < prev_o) & (o < prev_l) & (c > (prev_o + prev_c) / 2) & (c < prev_o)
    df["pat_piercing"] = np.where(piercing, 1, 0)

    # 13. Spinning Top — small body, roughly equal wicks
    small_body = body / bar_range < 0.3
    balanced_wicks = (upper_wick / lower_wick.replace(0, np.nan)).between(0.5, 2.0)
    df["pat_spinning_top"] = (small_body & balanced_wicks).astype(int)

    # 14. Marubozu — no wicks (body > 95% of range)
    df["pat_marubozu"] = np.where(
        (body / bar_range > 0.95) & (c > o), 1,
        np.where((body / bar_range > 0.95) & (c < o), -1, 0)
    )

    # 15. Harami — inside bar where body is within previous body
    harami_bull = (prev_c < prev_o) & (c > o) & (o > prev_c) & (c < prev_o)
    harami_bear = (prev_c > prev_o) & (c < o) & (o < prev_c) & (c > prev_o)
    df["pat_harami"] = np.where(harami_bull, 1, np.where(harami_bear, -1, 0))

    return df


# ═══════════════════════════════════════════════════════════════
#  CHART PATTERNS (probability scores 0.0 to 1.0)
# ═══════════════════════════════════════════════════════════════

def _find_swing_points(series: pd.Series, window: int = 10) -> tuple[pd.Series, pd.Series]:
    """Find local swing highs and lows."""
    swing_highs = series[(series == series.rolling(window * 2 + 1, center=True).max())]
    swing_lows = series[(series == series.rolling(window * 2 + 1, center=True).min())]
    return swing_highs, swing_lows


def add_chart_patterns(df: pd.DataFrame, lookback: int = 50) -> pd.DataFrame:
    """Detect chart patterns with probability scores using rolling window analysis."""
    c = df["close"].values
    h = df["high"].values
    l = df["low"].values
    n = len(df)

    # Initialize pattern columns
    pattern_cols = [
        "chart_hns_prob", "chart_inv_hns_prob",
        "chart_double_top_prob", "chart_double_bottom_prob",
        "chart_ascending_triangle", "chart_descending_triangle",
        "chart_symmetrical_triangle",
        "chart_bull_wedge", "chart_bear_wedge",
        "chart_flag_bull", "chart_flag_bear",
    ]
    for col in pattern_cols:
        df[col] = 0.0

    if n < lookback:
        return df

    # Rolling window pattern detection
    for i in range(lookback, n):
        window_h = h[i - lookback:i]
        window_l = l[i - lookback:i]
        window_c = c[i - lookback:i]

        # Double Top: two similar highs with a valley between
        max_idx = np.argmax(window_h)
        max_val = window_h[max_idx]
        # Find second highest that's not adjacent
        mask = np.ones(lookback, dtype=bool)
        mask[max(0, max_idx - 3):min(lookback, max_idx + 4)] = False
        if mask.any():
            second_max = np.max(window_h[mask])
            if max_val > 0 and abs(second_max - max_val) / max_val < 0.01:
                df.iloc[i, df.columns.get_loc("chart_double_top_prob")] = 0.7

        # Double Bottom: two similar lows with a peak between
        min_idx = np.argmin(window_l)
        min_val = window_l[min_idx]
        mask = np.ones(lookback, dtype=bool)
        mask[max(0, min_idx - 3):min(lookback, min_idx + 4)] = False
        if mask.any():
            second_min = np.min(window_l[mask])
            if min_val > 0 and abs(second_min - min_val) / min_val < 0.01:
                df.iloc[i, df.columns.get_loc("chart_double_bottom_prob")] = 0.7

        # Triangle detection using trend line convergence
        highs_slope = np.polyfit(range(lookback), window_h, 1)[0] if lookback > 1 else 0
        lows_slope = np.polyfit(range(lookback), window_l, 1)[0] if lookback > 1 else 0

        if highs_slope < 0 and lows_slope > 0:
            df.iloc[i, df.columns.get_loc("chart_symmetrical_triangle")] = 0.6
        elif highs_slope < -abs(lows_slope) * 0.5 and abs(lows_slope) < abs(highs_slope) * 0.3:
            df.iloc[i, df.columns.get_loc("chart_descending_triangle")] = 0.6
        elif lows_slope > abs(highs_slope) * 0.5 and abs(highs_slope) < abs(lows_slope) * 0.3:
            df.iloc[i, df.columns.get_loc("chart_ascending_triangle")] = 0.6

        # Wedge detection
        range_narrowing = (window_h[-1] - window_l[-1]) < (window_h[0] - window_l[0]) * 0.6
        if range_narrowing:
            if highs_slope > 0 and lows_slope > 0:
                df.iloc[i, df.columns.get_loc("chart_bear_wedge")] = 0.5  # Rising wedge = bearish
            elif highs_slope < 0 and lows_slope < 0:
                df.iloc[i, df.columns.get_loc("chart_bull_wedge")] = 0.5  # Falling wedge = bullish

        # Flag detection (consolidation after strong move)
        first_half = window_c[:lookback // 2]
        second_half = window_c[lookback // 2:]
        first_range = first_half.max() - first_half.min()
        second_range = second_half.max() - second_half.min()

        if first_range > 0 and second_range < first_range * 0.4:
            trend = window_c[lookback // 2 - 1] - window_c[0]
            if trend > 0:
                df.iloc[i, df.columns.get_loc("chart_flag_bull")] = 0.6
            elif trend < 0:
                df.iloc[i, df.columns.get_loc("chart_flag_bear")] = 0.6

    return df


# ═══════════════════════════════════════════════════════════════
#  SUPPORT & RESISTANCE
# ═══════════════════════════════════════════════════════════════

def add_support_resistance(df: pd.DataFrame) -> pd.DataFrame:
    """Compute S/R levels: swing high/low distances, Fibonacci, pivots."""
    c = df["close"]
    h = df["high"]
    l = df["low"]

    # Swing high/low (50-bar lookback)
    swing_high = h.rolling(50, center=False).max()
    swing_low = l.rolling(50, center=False).min()

    atr = df.get("atr_14", (h - l).rolling(14).mean()).replace(0, np.nan)

    df["sr_nearest_resistance_dist"] = (swing_high - c) / atr
    df["sr_nearest_support_dist"] = (c - swing_low) / atr
    df["sr_at_resistance"] = (df["sr_nearest_resistance_dist"].abs() < 0.5).astype(int)
    df["sr_at_support"] = (df["sr_nearest_support_dist"].abs() < 0.5).astype(int)

    # Fibonacci retracement levels (from 50-bar swing range)
    swing_range = swing_high - swing_low
    fib_382 = swing_low + swing_range * 0.382
    fib_500 = swing_low + swing_range * 0.500
    fib_618 = swing_low + swing_range * 0.618

    df["fib_382_dist"] = (c - fib_382) / atr
    df["fib_500_dist"] = (c - fib_500) / atr
    df["fib_618_dist"] = (c - fib_618) / atr

    # Pivot points (daily approximation using rolling)
    daily_h = h.rolling(96).max()  # ~96 M15 bars per day
    daily_l = l.rolling(96).min()
    daily_c = c.shift(1)
    pivot = (daily_h + daily_l + daily_c) / 3
    df["daily_pivot_dist"] = (c - pivot) / atr

    # Weekly pivot
    weekly_h = h.rolling(480).max()  # ~480 M15 bars per week
    weekly_l = l.rolling(480).min()
    weekly_pivot = (weekly_h + weekly_l + c.shift(1)) / 3
    df["weekly_pivot_dist"] = (c - weekly_pivot) / atr

    return df
