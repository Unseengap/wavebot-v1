# BACKTESTING.md — Walk-Forward Validation Methodology

## Why Custom Backtesting

WaveBot does not use Backtrader, Zipline, or any off-the-shelf backtesting framework. These frameworks are designed for equities and do not natively support:

- Variable bid/ask spread simulation (essential for forex accuracy)
- Wave state reconstruction on historical data (the core of this system)
- OANDA-specific candle alignment and timezone handling
- The exact execution model (FOK orders, OANDA fill mechanics)

A custom backtesting engine ensures that what works in backtesting is actually achievable in live trading.

---

## Golden Rule: No Data Leakage

Data leakage is the single most common reason backtests look great and live trading fails. The backtest engine enforces strict temporal separation.

```python
class BacktestEngine:
    """
    Processes historical candles in strict chronological order.
    At every step, the engine has access ONLY to data that would
    have been available at that exact moment in real time.

    No lookahead. No future data. No cheating.
    """

    def run(self, config: BacktestConfig):
        # Load ALL candles for the period
        candles = self.data_store.load_candles(
            config.instrument,
            config.granularity,
            config.start,
            config.end
        )

        # Process exactly as the live system would — one candle at a time
        for i, candle in enumerate(candles):
            if not candle["complete"]:
                continue

            # State at this point: candles[0:i+1] only
            # The engine literally cannot see candles[i+1:]
            self.wave_engine.process_candle(candle)
            self.confluence_engine.update()
            self.entry_filter.check_and_execute(candle["time"], candle)
            self.trade_monitor.update_open_trades(candle)
```

---

## Variable Spread Simulation

Most forex backtests use fixed spreads. Real spreads are variable — they widen during news events, thin markets, and OANDA maintenance windows. Fixed spread backtests are optimistic and misleading.

```python
class SpreadSimulator:
    """
    Reconstructs historical spread from stored bid/ask data.
    OANDA candle data with price="MBA" contains mid, bid, AND ask prices.
    The spread is directly calculable: ask_close - bid_close.

    For periods where only mid-price data is available:
    Apply instrument-specific spread model based on session + volatility.
    """

    def get_spread_pips(
        self,
        candle: dict,
        instrument: str,
        session: str,
    ) -> float:

        # Direct calculation if we have bid/ask
        if "bid" in candle and "ask" in candle:
            spread = candle["ask"]["c"] - candle["bid"]["c"]
            pip_size = get_pip_size(instrument)
            return spread / pip_size

        # Model-based fallback
        base_spread = BASE_SPREADS[instrument]
        session_multiplier = SESSION_SPREAD_MULTIPLIERS[session]
        return base_spread * session_multiplier

    def apply_to_entry(
        self,
        direction: str,
        mid_price: float,
        spread_pips: float,
        pip_size: float,
    ) -> float:
        """
        Longs are filled at the ASK (mid + half spread).
        Shorts are filled at the BID (mid - half spread).
        This is the real cost of entry that most backtests ignore.
        """
        half_spread = (spread_pips * pip_size) / 2

        if direction == "LONG":
            return mid_price + half_spread   # You pay the ask
        else:
            return mid_price - half_spread   # You sell at the bid
```

---

## Slippage Simulation

The spread simulator handles the bid/ask cost, but **slippage** — the difference between the quoted price at order time and the actual fill price — is a separate cost that most backtests ignore entirely.

Slippage is particularly dangerous for this system because the highest-conviction signals (Full Stack Alignment) tend to fire during fast moves when order books are thin. A 20-pip stop loss setup with 2 pips of slippage has already lost 10% of its risk budget before the trade begins.

### Slippage Model

```python
class SlippageSimulator:
    """
    Applies empirical slippage to backtest fills.
    Slippage is ALWAYS adverse — it works against you on both entry and exit.

    Base slippage values are derived from OANDA's typical fill behavior
    during normal market conditions. During high-volatility periods
    (candle range > 2× ATR), slippage increases by the volatility multiplier.
    """

    # Instrument-specific base slippage in pips (normal conditions)
    BASE_SLIPPAGE_PIPS = {
        "EUR_USD": 0.3,
        "GBP_USD": 0.5,
        "USD_JPY": 0.3,
        "USD_CAD": 0.4,
        "AUD_USD": 0.4,
        "USD_CHF": 0.4,
        "NZD_USD": 0.5,
        "EUR_GBP": 0.5,
        "GBP_JPY": 0.8,
        "EUR_JPY": 0.5,
        "XAU_USD": 1.5,
        "XAG_USD": 2.0,
    }

    # During high-volatility candles, slippage increases
    VOLATILITY_MULTIPLIER = 2.5    # Up to 2.5× base during fast moves
    VOLATILITY_THRESHOLD = 2.0     # Trigger: candle range > 2× ATR

    def get_slippage_pips(
        self,
        instrument: str,
        candle_range_pips: float,
        atr_pips: float,
    ) -> float:
        """
        Returns slippage in pips for a single fill (entry or exit).
        Always positive — caller applies it adversely.
        """
        base = self.BASE_SLIPPAGE_PIPS.get(instrument, 0.5)

        if atr_pips > 0 and candle_range_pips > self.VOLATILITY_THRESHOLD * atr_pips:
            return base * self.VOLATILITY_MULTIPLIER
        return base

    def apply_to_fill(
        self,
        direction: str,
        fill_type: str,         # "ENTRY" or "EXIT"
        price: float,
        slippage_pips: float,
        pip_size: float,
    ) -> float:
        """
        Slippage always works AGAINST you:
        - LONG entry: filled HIGHER (worse price)
        - LONG exit:  filled LOWER (worse price)
        - SHORT entry: filled LOWER (worse price)
        - SHORT exit:  filled HIGHER (worse price)
        """
        slip = slippage_pips * pip_size

        if direction == "LONG":
            if fill_type == "ENTRY":
                return price + slip   # Pay more to get in
            else:
                return price - slip   # Get less when closing
        else:  # SHORT
            if fill_type == "ENTRY":
                return price - slip   # Sell lower to get in
            else:
                return price + slip   # Buy back higher when closing
```

### Integration with Backtest Engine

Slippage is applied **on top of** the spread simulation. The total execution cost per trade is:

```
Total cost = spread (bid/ask) + entry slippage + exit slippage
```

For a typical EUR/USD trade during normal conditions:
- Spread: ~1.2 pips
- Entry slippage: ~0.3 pips
- Exit slippage: ~0.3 pips
- **Total: ~1.8 pips** (vs. 1.2 pips with spread only)

During Fast moves (Full Stack Alignment trigger during news):
- Spread: ~2.5 pips (widened)
- Entry slippage: ~0.75 pips (2.5× base)
- Exit slippage: ~0.75 pips
- **Total: ~4.0 pips** — a 20-pip SL trade now has 20% cost overhead

A strategy that passes the Go/No-Go matrix **after** slippage simulation is genuinely robust.

---

## Walk-Forward Validation

Walk-forward validation is the only rigorous methodology for time-series trading strategies. It mirrors how the system would actually be used: train on past data, test on future data, advance forward, repeat.

```
Walk-Forward Structure for EUR/USD 2018–2023:

Training Window (6 months)     Test Window (2 months)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Jan 2018 ─── Jun 2018          Jul 2018 ─── Aug 2018   → Fold 1
Mar 2018 ─── Aug 2018          Sep 2018 ─── Oct 2018   → Fold 2
May 2018 ─── Oct 2018          Nov 2018 ─── Dec 2018   → Fold 3
...
Jan 2023 ─── Jun 2023          Jul 2023 ─── Aug 2023   → Fold N

Each fold: train wave parameters (lookback, min wave pips, confluence threshold)
           on training window, then evaluate on test window.

Final score: Average performance across ALL test windows.
```

What is being "trained" here? The wave detection parameters — lookback periods, minimum wave sizes, confluence thresholds, hierarchy weights. These are optimized on the training window and frozen for the test window. The test window is never touched during optimization.

```python
class WalkForwardValidator:

    def run(
        self,
        instrument: str,
        full_start: str,
        full_end: str,
        train_months: int = 6,
        test_months: int = 2,
        step_months: int = 2,   # How far to advance each fold
    ) -> WalkForwardResults:

        folds = self.generate_folds(full_start, full_end, train_months, test_months, step_months)
        fold_results = []

        for fold in folds:
            # Step 1: Optimize parameters on training window
            best_params = self.optimizer.optimize(
                instrument=instrument,
                start=fold.train_start,
                end=fold.train_end,
                param_space=PARAM_SPACE,
            )

            # Step 2: Run backtest on test window with frozen params
            result = self.run_backtest(
                instrument=instrument,
                start=fold.test_start,
                end=fold.test_end,
                params=best_params,  # No further optimization allowed
            )

            fold_results.append(result)
            log.info(f"Fold {fold.id}: WR={result.win_rate:.1%}, Sharpe={result.sharpe:.2f}")

        return WalkForwardResults(folds=fold_results)
```

---

## Parameter Space

### The Overfitting Problem

The original parameter space contained ~11.9 million combinations (4×4×4×4×4×4 × 5×3 × 3×3×3 × 4). Even with Bayesian optimization, fitting this on 6 months of training data risks overfitting to each fold — the optimizer finds parameters that exploit noise in the training window rather than capturing genuine market structure.

The solution: **scaling groups**. Parameters that reflect the same underlying market property are constrained to move together. For example, `min_wave_pips_m5` and `min_wave_pips_m15` should not be optimized independently — they represent minimum meaningful move size at different zoom levels.

### Scaling Group Architecture

```python
# PHASE 1: Coarse search — 108 combinations (vs. 11.9M original)
# Each "knob" controls a group of related parameters

PARAM_SPACE_GROUPED = {

    # GROUP 1: Wave sensitivity — single knob controls all TFs
    # Controls: swing_lookback, min_wave_pips for all timeframes
    "wave_sensitivity": ["tight", "normal", "loose"],
    #
    # "tight"  → lookback=2, min_pips: M1=3,  M5=8,  M15=12, H1=20, H4=40
    # "normal" → lookback=3, min_pips: M1=5,  M5=10, M15=15, H1=25, H4=50
    # "loose"  → lookback=4, min_pips: M1=7,  M5=12, M15=20, H1=30, H4=60

    # GROUP 2: Confluence strictness — single knob
    # Controls: min_confluence_score, min_frames_aligned
    "confluence_strictness": ["aggressive", "balanced", "conservative"],
    #
    # "aggressive"   → threshold=0.65, min_frames=2
    # "balanced"     → threshold=0.72, min_frames=3
    # "conservative" → threshold=0.80, min_frames=4

    # GROUP 3: Risk profile — single knob
    # Controls: min_rr_ratio, tp_percentile, sl_buffer_pips
    "risk_profile": ["tight", "standard", "wide"],
    #
    # "tight"    → min_rr=2.5, tp_percentile=p50, sl_buffer=1
    # "standard" → min_rr=2.0, tp_percentile=p75, sl_buffer=2
    # "wide"     → min_rr=1.5, tp_percentile=p90, sl_buffer=3

    # GROUP 4: Maturity filter — independent (small search space)
    "max_entry_maturity": [0.65, 0.70, 0.75, 0.80],
}

# Total: 3 × 3 × 3 × 4 = 108 combinations
```

### Scaling Group Expansion

Each group maps to concrete parameter values:

```python
WAVE_SENSITIVITY_MAP = {
    "tight": {
        "swing_lookback": 2,
        "min_wave_pips_m1": 3,   "min_wave_pips_m5": 8,
        "min_wave_pips_m15": 12, "min_wave_pips_h1": 20,
        "min_wave_pips_h4": 40,
    },
    "normal": {
        "swing_lookback": 3,
        "min_wave_pips_m1": 5,   "min_wave_pips_m5": 10,
        "min_wave_pips_m15": 15, "min_wave_pips_h1": 25,
        "min_wave_pips_h4": 50,
    },
    "loose": {
        "swing_lookback": 4,
        "min_wave_pips_m1": 7,   "min_wave_pips_m5": 12,
        "min_wave_pips_m15": 20, "min_wave_pips_h1": 30,
        "min_wave_pips_h4": 60,
    },
}

CONFLUENCE_STRICTNESS_MAP = {
    "aggressive":   {"min_confluence_score": 0.65, "min_frames_aligned": 2},
    "balanced":     {"min_confluence_score": 0.72, "min_frames_aligned": 3},
    "conservative": {"min_confluence_score": 0.80, "min_frames_aligned": 4},
}

RISK_PROFILE_MAP = {
    "tight":    {"min_rr_ratio": 2.5, "tp_percentile": "p50", "sl_buffer_pips": 1},
    "standard": {"min_rr_ratio": 2.0, "tp_percentile": "p75", "sl_buffer_pips": 2},
    "wide":     {"min_rr_ratio": 1.5, "tp_percentile": "p90", "sl_buffer_pips": 3},
}

def expand_grouped_params(grouped: dict) -> dict:
    """Expands a grouped parameter selection into concrete values."""
    params = {}
    params.update(WAVE_SENSITIVITY_MAP[grouped["wave_sensitivity"]])
    params.update(CONFLUENCE_STRICTNESS_MAP[grouped["confluence_strictness"]])
    params.update(RISK_PROFILE_MAP[grouped["risk_profile"]])
    params["max_entry_maturity"] = grouped["max_entry_maturity"]
    return params
```

### Two-Phase Optimization

```python
class WalkForwardOptimizer:
    """
    Phase 1 (coarse): Search 108 grouped combinations.
    Phase 2 (fine):   If a group wins >60% of folds, lock it and
                      search ±1 step within that group's parameters.

    This produces robust parameters with no overfitting risk.
    """

    def optimize(self, instrument, start, end):
        # Phase 1: Coarse grid across 108 combinations
        coarse_results = self.grid_search(
            param_space=PARAM_SPACE_GROUPED,
            instrument=instrument, start=start, end=end,
        )

        winning_group = coarse_results.best_params
        fold_consistency = coarse_results.fold_win_rate(winning_group)

        if fold_consistency < 0.60:
            log.warning(f"Best group only wins {fold_consistency:.0%} of folds — no stable optimum")
            return expand_grouped_params(winning_group)

        # Phase 2: Fine-tune within the winning group
        fine_space = self.build_fine_space(winning_group)
        # Fine space: ~50 combinations max (small perturbations)
        fine_results = self.grid_search(
            param_space=fine_space,
            instrument=instrument, start=start, end=end,
        )

        return fine_results.best_params
```

### Legacy Parameter Space (Retained for Reference)

The original unconstrained space is retained for targeted research experiments only. It must **never** be used in walk-forward optimization for production parameters.

```python
# WARNING: ~11.9M combinations — overfitting risk on 6-month windows
# Use only for single-variable sensitivity analysis, not full optimization

PARAM_SPACE_UNCONSTRAINED = {
    "swing_lookback": [2, 3, 4, 5],
    "min_wave_pips_m1":  [3, 5, 7, 10],
    "min_wave_pips_m5":  [8, 10, 12, 15],
    "min_wave_pips_m15": [12, 15, 20, 25],
    "min_wave_pips_h1":  [20, 25, 30, 40],
    "min_wave_pips_h4":  [40, 50, 60, 80],
    "min_confluence_score": [0.65, 0.70, 0.72, 0.75, 0.80],
    "min_frames_aligned":   [2, 3, 4],
    "min_rr_ratio":  [1.5, 2.0, 2.5],
    "tp_percentile": ["p50", "p75", "p90"],
    "sl_buffer_pips": [1, 2, 3],
    "max_entry_maturity": [0.65, 0.70, 0.75, 0.80],
}
```

The optimizer uses a simple grid search initially. With GPU time on Colab, Bayesian optimization (optuna) provides faster convergence.

---

## Performance Metrics

```python
@dataclass
class BacktestMetrics:
    instrument: str
    start: str
    end: str

    # Core metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float              # winning / total

    # Profit metrics
    total_pips: float            # Net pips gained/lost
    avg_win_pips: float
    avg_loss_pips: float
    profit_factor: float         # gross_profit / gross_loss

    # Risk metrics
    max_drawdown_pct: float      # Peak-to-trough as % of peak balance
    max_drawdown_pips: float
    sharpe_ratio: float          # Annualized
    calmar_ratio: float          # Annual return / max drawdown

    # Trade quality
    avg_rr_achieved: float       # Actual R:R of winning trades
    avg_mae_pips: float          # Maximum Adverse Excursion (how far against before win)
    avg_mfe_pips: float          # Maximum Favorable Excursion (how far in profit)

    # Efficiency
    avg_bars_in_trade: float     # How long trades typically run
    trades_per_month: float

    def summary(self) -> str:
        return (
            f"Trades: {self.total_trades} | WR: {self.win_rate:.1%} | "
            f"PF: {self.profit_factor:.2f} | Sharpe: {self.sharpe_ratio:.2f} | "
            f"MaxDD: {self.max_drawdown_pct:.1%} | Pips: {self.total_pips:.0f}"
        )
```

---

## Go / No-Go Decision Matrix

Run backtest on EUR/USD 2018–2023 (5 years, walk-forward). These thresholds must ALL be met before paper trading begins.

| Metric | No-Go (fix the model) | Caution (proceed carefully) | Go (proceed to paper) |
|---|---|---|---|
| Win rate | < 50% | 50–58% | ≥ 58% |
| Profit factor | < 1.3 | 1.3–1.6 | ≥ 1.6 |
| Max drawdown | > 15% | 10–15% | < 10% |
| Sharpe ratio | < 1.0 | 1.0–1.4 | ≥ 1.4 |
| Trades/month | < 5 | 5–10 | > 10 |
| Walk-forward consistency | > 40% of folds losing | 20–40% losing | < 20% losing |

A strategy that only works in some market regimes (trending vs ranging) will have inconsistent fold performance. If more than 30% of walk-forward folds are losing, the system is regime-dependent and needs work before deployment.

---

## Backtest Report Output

After every backtest run, a report is written to `reports/backtest_{instrument}_{date}.json`:

```json
{
  "run_date": "2024-03-15T10:23:44Z",
  "instrument": "EUR_USD",
  "period": "2018-01-01 to 2023-12-31",
  "params": {
    "swing_lookback": 3,
    "min_confluence_score": 0.72,
    "tp_percentile": "p75"
  },
  "metrics": {
    "total_trades": 847,
    "win_rate": 0.623,
    "profit_factor": 1.89,
    "max_drawdown_pct": 0.071,
    "sharpe_ratio": 1.62,
    "total_pips": 3241
  },
  "walk_forward_folds": [
    {"fold": 1, "period": "2018-01 to 2018-08", "win_rate": 0.61, "pf": 1.72},
    {"fold": 2, "period": "2018-03 to 2018-10", "win_rate": 0.65, "pf": 2.01}
  ],
  "decision": "GO — all metrics above minimum thresholds"
}
```
