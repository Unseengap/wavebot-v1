# MODEL_SPEC.md — Machine Learning & Reinforcement Learning Architecture

## Overview

OandaFX uses a three-component model ensemble: a Transformer sequence model for directional prediction, a Proximal Policy Optimization (PPO) reinforcement learning agent for trade management decisions, and a dynamic SL/TP module for risk-adjusted exit placement. Together they produce actionable, risk-aware trade signals.

---

## Model 1: Transformer Sequence Model

### Purpose

Predicts the directional bias of the market for the next 1–5 bars (multi-horizon) based on the past 128 bars of feature data. Outputs a probability distribution over three classes: long, short, flat.

### Architecture

```
Input shape: [batch, 128 bars, ~100 features]
                    │
          ┌─────────▼─────────┐
          │  Linear projection │   100 → 256 dims
          └─────────┬─────────┘
                    │
          ┌─────────▼─────────┐
          │ Positional encoding│   Sinusoidal, learned
          └─────────┬─────────┘
                    │
          ┌─────────▼─────────┐
          │ Transformer Encoder│   4 layers
          │  8 attention heads │   
          │  256 hidden dims   │
          │  1024 FFN dims     │
          │  Dropout: 0.1      │
          └─────────┬─────────┘
                    │
          ┌─────────▼─────────┐
          │  [CLS] token pool  │   Final bar representation
          └─────────┬─────────┘
                    │
          ┌─────────▼─────────┐
          │  Output heads      │
          │  ├─ Direction (3)  │   softmax → [P_long, P_short, P_flat]
          │  ├─ Horizon 1 (3)  │   1-bar prediction
          │  ├─ Horizon 3 (3)  │   3-bar prediction
          │  └─ Horizon 5 (3)  │   5-bar prediction
          └────────────────────┘
```

### Training details

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 1e-4, cosine decay |
| Batch size | 256 |
| Epochs | 100 (early stopping patience=10) |
| Loss function | Cross-entropy (weighted by class frequency) |
| Label generation | Next N bars direction (threshold: 0.5× ATR) |
| Train/val/test split | 70% / 15% / 15% (time-ordered, no shuffling) |
| Regularization | Dropout(0.1), weight decay(1e-4), label smoothing(0.1) |
| Hardware | Google Colab T4/A100 GPU |
| Training time (est.) | 2–4 hours for full 10-year dataset |

### Label generation logic

```python
def generate_labels(df, horizon=5, threshold_atr_multiplier=0.5):
    """
    Long label:  future_high - current_close > threshold AND
                 future does not hit threshold_down first
    Short label: current_close - future_low > threshold AND
                 future does not hit threshold_up first
    Flat label:  neither condition met within horizon bars
    """
    atr = df["atr_14"]
    threshold = atr * threshold_atr_multiplier
    ...
```

### Output interpretation

```python
signal = transformer.predict(feature_window)
# signal = {"long": 0.72, "short": 0.15, "flat": 0.13}

# Only trade if confidence exceeds threshold
if signal["long"] > 0.65:
    direction = "long"
elif signal["short"] > 0.65:
    direction = "short"
else:
    direction = "flat"   # No trade
```

---

## Model 2: PPO Reinforcement Learning Agent

### Purpose

The RL agent learns how to actively manage trades — not just when to enter, but how large to size positions, where to set SL/TP, when to close early, and when to let profits run. It learns from simulated trading experience across the entire historical dataset.

### State space

The RL agent receives a state vector combining market features and account state:

```
Market state (from Transformer output + raw features):
  - Transformer direction probabilities [3]
  - Current feature vector [~100]
  - MTF alignment score [1]
  - Volatility regime [1]

Account state:
  - Current open P&L (normalized by balance) [1]
  - Number of open trades [1]
  - Margin utilization [1]
  - Current drawdown from peak [1]
  - Bars since last trade [1]

State vector total: ~110 dimensions
```

### Action space

The agent outputs a continuous action vector:

```
action[0]: entry decision     Continuous [-1, 1]
                              < -0.3 = short
                              > +0.3 = long
                              between = no trade

action[1]: position size      Continuous [0, 1]
                              Scaled to [0.005, max_risk] fraction of balance

action[2]: SL distance        Continuous [0, 1]
                              Scaled to [0.5×ATR, 3×ATR]

action[3]: TP distance        Continuous [0, 1]
                              Scaled to [1×ATR, 5×ATR]

action[4]: close now          Binary {0, 1}
                              1 = close existing trade immediately
```

### Reward function

The reward function is the critical design element. It must reward good risk-adjusted returns while penalizing undesirable behaviors:

```python
def compute_reward(trade_result, account_state, action):
    # Base reward: risk-adjusted return on this trade
    pnl_pct = trade_result.pnl / account_state.balance_at_entry
    atr_at_entry = trade_result.atr_at_entry
    reward = pnl_pct / (action.sl_distance / atr_at_entry)   # return per unit risk

    # Penalty: drawdown
    dd_penalty = -2.0 * max(0, account_state.current_drawdown - 0.02)

    # Penalty: overtrading (more than 5 trades open simultaneously)
    overtrading_penalty = -0.5 if account_state.open_trades > 5 else 0

    # Penalty: no SL (absolute veto)
    no_sl_penalty = -10.0 if action.sl_distance == 0 else 0

    # Penalty: holding losing trade too long (>3× average winning duration)
    duration_penalty = -0.3 if trade_result.loss and trade_result.duration > avg_win_duration * 3 else 0

    # Bonus: profitable trade with good RR
    rr_bonus = 0.5 if trade_result.pnl > 0 and trade_result.rr_achieved > 1.5 else 0

    return reward + dd_penalty + overtrading_penalty + no_sl_penalty + duration_penalty + rr_bonus
```

### PPO hyperparameters

| Parameter | Value |
|-----------|-------|
| Algorithm | PPO (Proximal Policy Optimization) |
| Library | Stable-Baselines3 |
| Policy network | MlpPolicy: [256, 256, 128] |
| Learning rate | 3e-4 |
| n_steps | 2048 |
| Batch size | 64 |
| n_epochs | 10 |
| gamma | 0.99 |
| gae_lambda | 0.95 |
| clip_range | 0.2 |
| ent_coef | 0.01 (encourage exploration) |
| Training environment | Custom gym.Env wrapping historical OANDA data |
| Total timesteps | 5,000,000 |

### Training environment

The PPO agent trains inside a custom Gym environment that simulates OANDA market conditions:

```python
class OandaForexEnv(gym.Env):
    def __init__(self, features_df, spread_pips, pip_value, max_leverage=50):
        # Simulates OANDA spreads, leverage limits, and margin rules
        # Uses bid/ask prices (not mid) for realistic fill simulation
        # Episode = one calendar year of data
        # Reset = random start within training period
        ...
```

---

## Model 3: Dynamic SL/TP Module

### Purpose

Provides a safety layer on top of the RL agent's raw SL/TP outputs. Adjusts exit levels based on real-time market volatility, key S/R levels, and risk budget.

### Logic

```python
def compute_sl_tp(direction, entry_price, atr, sr_levels, account_risk_budget):

    # Base SL from ATR
    base_sl_pips = atr * 1.5       # 1.5× ATR minimum

    # Adjust for nearby S/R level (place SL just beyond it)
    nearest_level = find_nearest_sr(entry_price, direction, sr_levels)
    if nearest_level exists within 2×ATR:
        sl_pips = distance_to_level + 0.3×ATR    # just beyond the level

    # Volatility regime override
    if vol_regime > 1.5:           # high volatility period
        sl_pips *= 1.3             # wider SL in choppy markets

    # TP: aim for minimum 1.5:1 RR, prefer 2:1 or better
    tp_pips = max(sl_pips * 1.5, nearest_opposing_sr_level_distance)

    # Hard cap: never risk more than account_risk_budget on this trade
    max_sl_pips = account_risk_budget / position_size_in_units / pip_value
    sl_pips = min(sl_pips, max_sl_pips)

    return sl_pips, tp_pips
```

---

## Model Ensemble Voting

The Transformer and RL agent votes are combined before final signal emission:

```python
def ensemble_signal(transformer_output, rl_output, weights):
    """
    weights = {"transformer": 0.6, "rl": 0.4}
    Updated weekly based on live performance ratio
    """
    # Only enter if both models agree on direction
    t_dir = transformer_output.dominant_class      # "long", "short", or "flat"
    r_dir = rl_output.entry_direction              # "long", "short", or "none"

    if t_dir != r_dir or t_dir == "flat":
        return Signal(action="flat", reason="models_disagree")

    # Confidence weighted average
    confidence = (
        transformer_output.confidence * weights["transformer"] +
        rl_output.confidence * weights["rl"]
    )

    if confidence < 0.65:
        return Signal(action="flat", reason="low_confidence")

    return Signal(
        action=t_dir,
        confidence=confidence,
        position_size=rl_output.position_size,
        sl_pips=sl_tp_module.sl,
        tp_pips=sl_tp_module.tp
    )
```

---

## Model Versioning & Storage

All trained models are stored in `data/models/` with semantic versioning:

```
data/models/
├── v1.0.0/
│   ├── transformer.pt
│   ├── ppo_agent.zip          (Stable-Baselines3 format)
│   ├── sl_tp_scaler.pkl
│   └── metadata.json
├── v1.1.0/
│   └── ...
└── latest -> v1.1.0/          (symlink)
```

`metadata.json` contains:
```json
{
  "version": "1.1.0",
  "training_date": "2025-04-01",
  "training_data_range": ["2010-01-01", "2024-12-31"],
  "pairs_trained_on": ["EUR_USD", "GBP_USD", "USD_JPY"],
  "base_timeframe": "M15",
  "val_sharpe": 1.42,
  "val_win_rate": 0.54,
  "features_used": 98,
  "transformer_epochs": 87,
  "ppo_timesteps": 5000000,
  "ensemble_weights": {"transformer": 0.6, "rl": 0.4}
}
```

---

## Continuous Retraining Schedule

| Trigger | Action |
|---------|--------|
| Weekly (Sunday 06:00 UTC) | Incremental fine-tune on last 2 weeks of live data |
| Monthly | Full retrain on all data including last month |
| Drawdown > 8% in 30 days | Emergency retrain + strategy review flag |
| New pair added | Fine-tune on pair-specific data for 500,000 PPO steps |

Retraining runs on Google Colab using the `notebooks/03_model_training.ipynb` notebook, triggered via GitHub Actions (Colab API or manual execution). The resulting model artifact is pushed to the GitHub repo and the VPS automatically pulls it at next session start.
