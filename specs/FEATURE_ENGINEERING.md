# FEATURE_ENGINEERING.md — Indicators, Patterns & Multi-Timeframe Features

## Overview

The feature engineering pipeline transforms raw OANDA OHLCV candle data into a structured feature matrix that the model can learn from. Features are grouped into six categories: price action, trend, momentum, volatility, pattern recognition, and multi-timeframe context. All features are normalized to be pair-agnostic so a single model can trade any instrument.

---

## Normalization Approach

**Problem:** Raw prices differ wildly across pairs (USD/JPY trades around 150, EUR/USD around 1.08). A model trained on raw prices cannot generalize across pairs.

**Solution — pip-relative normalization:**
All price-based features are expressed as multiples of the current ATR(14) or in pip units relative to the current price. This makes features dimensionless and pair-agnostic.

```python
# Example: EMA distance feature
ema_50 = ta.ema(close, length=50)
ema_distance_raw = close - ema_50              # in price units (not useful)
ema_distance_atr = (close - ema_50) / atr_14  # in ATR units (useful, generalizes)
```

**Session normalization:** Time-of-day features are encoded using sine/cosine to preserve cyclical structure:
```python
hour_sin = sin(2π × hour_of_day / 24)
hour_cos = cos(2π × hour_of_day / 24)
```

---

## Feature Categories

### Category 1: Price Action Features

| Feature | Description | Normalization |
|---------|-------------|---------------|
| `return_1` | 1-bar log return | Raw (already scale-free) |
| `return_3` | 3-bar log return | Raw |
| `return_5` | 5-bar log return | Raw |
| `return_10` | 10-bar log return | Raw |
| `return_20` | 20-bar log return | Raw |
| `bar_range` | High - Low | Divided by ATR(14) |
| `bar_body` | abs(Close - Open) | Divided by bar_range |
| `upper_wick` | High - max(Open, Close) | Divided by bar_range |
| `lower_wick` | min(Open, Close) - Low | Divided by bar_range |
| `close_position` | (Close - Low) / (High - Low) | Raw [0, 1] |
| `gap_open` | Open vs. previous Close | Divided by ATR(14) |
| `spread_pips` | Current bid-ask spread | In pips |
| `spread_ratio` | Current spread / 20-bar avg spread | Raw ratio |

---

### Category 2: Trend Indicators

| Feature | Description | Normalization |
|---------|-------------|---------------|
| `ema_8_dist` | Close - EMA(8) | Divided by ATR(14) |
| `ema_21_dist` | Close - EMA(21) | Divided by ATR(14) |
| `ema_50_dist` | Close - EMA(50) | Divided by ATR(14) |
| `ema_200_dist` | Close - EMA(200) | Divided by ATR(14) |
| `ema_8_21_cross` | EMA(8) - EMA(21) | Divided by ATR(14) |
| `ema_21_50_cross` | EMA(21) - EMA(50) | Divided by ATR(14) |
| `sma_20_dist` | Close - SMA(20) | Divided by ATR(14) |
| `sma_50_slope` | SMA(50) slope (5 bars) | Divided by ATR(14) |
| `sma_200_slope` | SMA(200) slope (10 bars) | Divided by ATR(14) |
| `adx_14` | Average Directional Index | Raw [0, 100] |
| `plus_di` | +DI | Raw [0, 100] |
| `minus_di` | -DI | Raw [0, 100] |
| `aroon_up` | Aroon Up(25) | Raw [0, 100] |
| `aroon_down` | Aroon Down(25) | Raw [0, 100] |
| `aroon_osc` | Aroon Oscillator | Raw [-100, 100] |
| `ichimoku_cloud_dist` | Close vs. Kumo midpoint | Divided by ATR(14) |
| `price_above_cloud` | 1 if close above cloud, -1 below | Binary |

---

### Category 3: Momentum Indicators

| Feature | Description | Normalization |
|---------|-------------|---------------|
| `rsi_14` | RSI(14) | Rescaled to [-1, 1] via (RSI-50)/50 |
| `rsi_7` | RSI(7) — faster | Rescaled |
| `stoch_k` | Stochastic %K(14,3) | Rescaled to [-1, 1] |
| `stoch_d` | Stochastic %D | Rescaled to [-1, 1] |
| `stoch_kd_diff` | %K - %D | Raw difference |
| `macd_line` | MACD(12,26) | Divided by ATR(14) |
| `macd_signal` | MACD signal(9) | Divided by ATR(14) |
| `macd_hist` | MACD histogram | Divided by ATR(14) |
| `cci_20` | CCI(20) | Divided by 200 (typical range) |
| `williams_r` | Williams %R(14) | Rescaled to [-1, 1] |
| `roc_10` | Rate of Change(10) | Raw (log scale) |
| `momentum_10` | Momentum(10) | Divided by ATR(14) |
| `mfi_14` | Money Flow Index | Rescaled to [-1, 1] |

---

### Category 4: Volatility Indicators

| Feature | Description | Normalization |
|---------|-------------|---------------|
| `atr_14` | ATR(14) in pips | Absolute (used for normalization base) |
| `atr_7_ratio` | ATR(7) / ATR(14) | Raw ratio |
| `atr_50_ratio` | ATR(14) / ATR(50) | Raw ratio — volatility regime |
| `bb_upper_dist` | Close - BB Upper | Divided by ATR(14) |
| `bb_lower_dist` | Close - BB Lower | Divided by ATR(14) |
| `bb_pct_b` | Bollinger %B | Raw [0, 1], clipped to [-0.5, 1.5] |
| `bb_bandwidth` | BB bandwidth | Divided by ATR(14) |
| `kc_upper_dist` | Close - Keltner Upper | Divided by ATR(14) |
| `kc_lower_dist` | Close - Keltner Lower | Divided by ATR(14) |
| `bb_squeeze` | BB inside KC = 1, else 0 | Binary |
| `hist_vol_20` | 20-bar realized volatility | Raw (annualized) |
| `vol_regime` | hist_vol_20 / 50-bar avg vol | Volatility regime ratio |

---

### Category 5: Pattern Recognition

#### Candlestick patterns (binary flags: 1 = bullish, -1 = bearish, 0 = none)

| Feature | Pattern |
|---------|---------|
| `pat_doji` | Doji (open ≈ close) |
| `pat_hammer` | Hammer / Hanging Man |
| `pat_engulfing` | Bullish/Bearish Engulfing |
| `pat_morning_star` | Morning Star / Evening Star |
| `pat_shooting_star` | Shooting Star / Inverted Hammer |
| `pat_inside_bar` | Inside Bar (mother candle + inside bar) |
| `pat_outside_bar` | Outside Bar (engulfs previous) |
| `pat_pin_bar` | Pin Bar (>60% wick rejection) |
| `pat_tweezer` | Tweezer Top / Bottom |
| `pat_three_soldiers` | Three White Soldiers / Black Crows |
| `pat_dark_cloud` | Dark Cloud Cover |
| `pat_piercing` | Piercing Line |
| `pat_spinning_top` | Spinning Top |
| `pat_marubozu` | Marubozu (no wicks) |
| `pat_harami` | Harami (inside bar with gap) |

#### Chart patterns (probability scores: 0.0 to 1.0)

| Feature | Pattern |
|---------|---------|
| `chart_hns_prob` | Head and Shoulders probability |
| `chart_inv_hns_prob` | Inverse Head and Shoulders |
| `chart_double_top_prob` | Double Top |
| `chart_double_bottom_prob` | Double Bottom |
| `chart_ascending_triangle` | Ascending Triangle |
| `chart_descending_triangle` | Descending Triangle |
| `chart_symmetrical_triangle` | Symmetrical Triangle |
| `chart_bull_wedge` | Rising/Falling Wedge (bullish resolution) |
| `chart_bear_wedge` | Rising/Falling Wedge (bearish resolution) |
| `chart_flag_bull` | Bull Flag |
| `chart_flag_bear` | Bear Flag |

#### Support & Resistance

| Feature | Description | Normalization |
|---------|-------------|---------------|
| `sr_nearest_support_dist` | Distance to nearest swing low (50-bar) | Divided by ATR(14) |
| `sr_nearest_resistance_dist` | Distance to nearest swing high (50-bar) | Divided by ATR(14) |
| `sr_at_support` | Within 0.5 ATR of support | Binary |
| `sr_at_resistance` | Within 0.5 ATR of resistance | Binary |
| `fib_382_dist` | Distance to 38.2% Fibonacci level | Divided by ATR(14) |
| `fib_500_dist` | Distance to 50% Fibonacci level | Divided by ATR(14) |
| `fib_618_dist` | Distance to 61.8% Fibonacci level | Divided by ATR(14) |
| `daily_pivot_dist` | Distance to daily pivot point | Divided by ATR(14) |
| `weekly_pivot_dist` | Distance to weekly pivot | Divided by ATR(14) |

---

### Category 6: Multi-Timeframe (MTF) Features

MTF features capture the context from higher timeframes. For a bot trading M15, the H1 and H4 trends act as a filter — it avoids taking short signals when H4 is strongly bullish.

**MTF Alignment logic:**

For each higher timeframe (H1, H4, D), the following are computed and merged down to the base timeframe bar:

```
[H1 at 09:00] → applies to all M15 bars in the 09:00–09:59 window
[H4 at 08:00] → applies to all M15 bars in the 08:00–11:59 window
[D at 00:00]  → applies to all M15 bars in that calendar day
```

| Feature | Description |
|---------|-------------|
| `h1_trend_dir` | H1 EMA(8) vs EMA(21): +1, -1, or 0 |
| `h1_rsi` | H1 RSI(14) rescaled |
| `h1_adx` | H1 ADX(14) |
| `h1_above_ema200` | H1 close above EMA(200): binary |
| `h4_trend_dir` | H4 EMA(8) vs EMA(21): +1, -1, or 0 |
| `h4_rsi` | H4 RSI(14) rescaled |
| `h4_adx` | H4 ADX(14) |
| `h4_above_ema200` | H4 close above EMA(200): binary |
| `d_trend_dir` | Daily EMA(21) vs EMA(50): +1, -1, or 0 |
| `d_rsi` | Daily RSI(14) |
| `d_above_ema200` | Daily close above EMA(200): binary |
| `mtf_alignment` | Sum of trend dirs (+3 = all up, -3 = all down) |
| `mtf_agreement` | abs(mtf_alignment) / 3 — [0, 1] agreement score |

---

### Category 7: Session & Time Features

| Feature | Description |
|---------|-------------|
| `hour_sin` | sin(2π × hour / 24) |
| `hour_cos` | cos(2π × hour / 24) |
| `day_of_week` | 0=Monday … 4=Friday (one-hot encoded × 5) |
| `is_london_open` | 08:00–09:30 UTC: binary |
| `is_ny_open` | 13:30–15:00 UTC: binary |
| `is_overlap` | London+NY overlap 13:30–16:30 UTC: binary |
| `is_asia_session` | 23:00–08:00 UTC: binary |
| `is_low_liquidity` | Friday post-16:30 UTC or Asia session: binary |
| `bars_to_ny_open` | Bars remaining until NY open (0 if open) |
| `bars_to_daily_close` | Bars remaining until 22:00 UTC close |

---

## Feature Pipeline Implementation

```python
class FeatureEngineer:
    def __init__(self, config: dict):
        self.base_tf = config["base_timeframe"]
        self.higher_tfs = config["higher_timeframes"]
        self.lookback = config["lookback_bars"]

    def compute(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        data = {
            "M15": df_m15,
            "H1":  df_h1,
            "H4":  df_h4,
            "D":   df_d
        }
        Returns feature DataFrame aligned to base_tf bars.
        """
        base = data[self.base_tf].copy()

        # Price action
        base = self._add_price_features(base)

        # Indicators
        base = self._add_trend_indicators(base)
        base = self._add_momentum_indicators(base)
        base = self._add_volatility_indicators(base)

        # Patterns
        base = self._add_candlestick_patterns(base)
        base = self._add_chart_patterns(base)
        base = self._add_sr_levels(base)

        # MTF
        for tf in self.higher_tfs:
            base = self._merge_higher_tf(base, data[tf], tf)

        # Session
        base = self._add_session_features(base)

        # Final normalization pass
        base = self._normalize(base)

        # Drop NaN rows caused by indicator warmup periods
        base = base.dropna()

        return base
```

---

## Feature Importance Tracking

After each training cycle, the system logs feature importance scores to `data/feature_importance.json`. Features consistently ranked low across training cycles are candidates for removal to reduce model dimensionality and training time.

Features flagged as high importance in typical forex models:
- `mtf_alignment` — direction agreement across timeframes
- `atr_50_ratio` — volatility regime detection
- `macd_hist` — momentum direction and speed
- `bb_squeeze` — pre-breakout compression
- `sr_at_support` / `sr_at_resistance` — key level proximity
- `adx_14` — trend strength filter

---

## Compute Requirements

Feature computation on the full historical dataset (10 years, 10 pairs, 4 timeframes):

| Operation | Estimated time (CPU) |
|-----------|---------------------|
| Load raw Parquet files | ~10 seconds |
| Compute all indicators | ~2–5 minutes |
| MTF alignment merge | ~1 minute |
| Pattern detection | ~3–8 minutes |
| Save processed features | ~30 seconds |
| **Total** | **~10–15 minutes** |

For live inference, feature computation on a single new bar takes < 50ms on any modern CPU.
