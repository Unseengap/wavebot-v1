# TRAINING.md — Google Colab GPU Training Workflow

## Overview

All model training for OandaFX runs on Google Colab using free or Pro-tier GPU instances (T4 or A100). The training pipeline is self-contained in `notebooks/03_model_training.ipynb` and produces versioned model artifacts that are pushed to GitHub and pulled by the Vultr VPS for live inference.

Training is **never** done on the VPS — the VPS has no GPU and its resources are reserved for live trading execution.

---

## Training Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                       Google Colab (GPU)                         │
│                                                                  │
│  1. Mount Google Drive                                           │
│  2. Clone oandafx repo from GitHub                               │
│  3. Install GPU dependencies (requirements-gpu.txt)              │
│  4. Load raw Parquet data from Drive or download from OANDA      │
│  5. Run feature engineering pipeline                             │
│  6. Train Transformer (supervised)                               │
│  7. Train PPO RL agent (reinforcement learning)                  │
│  8. Fit SL/TP scaler                                             │
│  9. Run validation backtest                                      │
│ 10. Save model artifacts to Drive + push to GitHub               │
└──────────────────────────────────────────────────────────────────┘
                              │
                         git push
                              │
                              ▼
                      GitHub Repository
                     data/models/vX.Y.Z/
                              │
                         git pull
                              │
                              ▼
                   Vultr VPS (live inference)
```

---

## Prerequisites

### Google Colab tier

| Tier | GPU | VRAM | Session limit | Recommended |
|------|-----|------|---------------|-------------|
| Free | Tesla T4 | 15 GB | ~12 hours | Initial experimentation |
| Pro ($9.99/mo) | T4 / A100 | 15–40 GB | ~24 hours | Regular retraining |
| Pro+ ($49.99/mo) | A100 40 GB | 40 GB | Priority queue | Full 10-year retrains |

**Minimum for full training:** Colab Pro with T4 GPU (15 GB VRAM). The Transformer model and PPO agent both fit within 15 GB with batch size 256.

### Google Drive storage

Training data and model artifacts are cached on Google Drive to persist across Colab sessions:

```
Google Drive/
└── oandafx/
    ├── data/
    │   └── raw/                   # Parquet files (synced from VPS)
    │       ├── EUR_USD/
    │       │   ├── M15.parquet
    │       │   ├── H1.parquet
    │       │   ├── H4.parquet
    │       │   └── D.parquet
    │       └── [other pairs]/
    ├── models/                    # Trained model checkpoints
    │   ├── v1.0.0/
    │   ├── v1.1.0/
    │   └── latest/
    └── logs/                      # TensorBoard training logs
        └── run_2026-04-09/
```

---

## Step-by-Step Training Workflow

### Step 1: Environment Setup (Colab Cell 1)

```python
# ─── Mount Google Drive ─────────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')

# ─── Clone the repository ──────────────────────────────────────
!git clone https://github.com/your-org/oandafx.git /content/oandafx
%cd /content/oandafx

# ─── Install GPU dependencies ──────────────────────────────────
!pip install -r requirements-gpu.txt

# ─── Verify GPU availability ───────────────────────────────────
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
```

**`requirements-gpu.txt` contents:**

```
torch>=2.1.0
stable-baselines3>=2.1.0
gymnasium>=0.29.0
pandas>=2.1.0
numpy>=1.24.0
pyarrow>=14.0.0
ta-lib>=0.4.28
scikit-learn>=1.3.0
tensorboard>=2.15.0
oandapyV20>=0.7.2
tqdm>=4.66.0
matplotlib>=3.8.0
seaborn>=0.13.0
```

### Step 2: Load and Validate Data (Colab Cell 2)

```python
import os
import pandas as pd
from pathlib import Path

# ─── Data source: Google Drive (fastest) or fresh download ─────
DRIVE_DATA = Path("/content/drive/MyDrive/oandafx/data/raw")
LOCAL_DATA = Path("/content/oandafx/data/raw")

if DRIVE_DATA.exists():
    DATA_DIR = DRIVE_DATA
    print(f"Loading data from Google Drive: {DATA_DIR}")
else:
    # Download fresh from OANDA if Drive data not available
    print("No Drive data found. Downloading from OANDA...")
    !python scripts/download_history.py \
        --pairs EUR_USD GBP_USD USD_JPY AUD_USD USD_CHF USD_CAD NZD_USD \
        --granularities M15 H1 H4 D \
        --start 2010-01-01 \
        --output-dir {LOCAL_DATA}
    DATA_DIR = LOCAL_DATA

# ─── Validate data ─────────────────────────────────────────────
pairs = [d.name for d in DATA_DIR.iterdir() if d.is_dir()]
print(f"Pairs available: {pairs}")

for pair in pairs:
    for tf in ["M15", "H1", "H4", "D"]:
        path = DATA_DIR / pair / f"{tf}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            print(f"  {pair}/{tf}: {len(df):>8,} bars | {df['time'].min()} → {df['time'].max()}")
        else:
            print(f"  {pair}/{tf}: MISSING")
```

### Step 3: Feature Engineering (Colab Cell 3)

```python
import sys
sys.path.insert(0, "/content/oandafx/src")

from features.engineer import FeatureEngineer

# ─── Configure feature pipeline ────────────────────────────────
engineer = FeatureEngineer(
    base_timeframe="M15",
    higher_timeframes=["H1", "H4", "D"],
    lookback_bars=128,
    include_volume=True,
    include_patterns=True,
    include_mtf=True,
)

# ─── Process all pairs ─────────────────────────────────────────
feature_datasets = {}
for pair in pairs:
    print(f"Processing {pair}...")
    raw_data = {
        "M15": pd.read_parquet(DATA_DIR / pair / "M15.parquet"),
        "H1": pd.read_parquet(DATA_DIR / pair / "H1.parquet"),
        "H4": pd.read_parquet(DATA_DIR / pair / "H4.parquet"),
        "D": pd.read_parquet(DATA_DIR / pair / "D.parquet"),
    }
    features_df = engineer.build(raw_data)
    feature_datasets[pair] = features_df
    print(f"  {pair}: {features_df.shape[0]:,} rows × {features_df.shape[1]} features")

# ─── Combine all pairs into one training dataset ───────────────
# Pair-agnostic: features are normalized by ATR so all pairs share the same scale
all_features = pd.concat(
    [df.assign(pair=pair) for pair, df in feature_datasets.items()],
    ignore_index=True
)
print(f"\nTotal training dataset: {all_features.shape[0]:,} rows × {all_features.shape[1]} columns")
```

### Step 4: Label Generation (Colab Cell 4)

```python
from models.transformer import generate_labels

# ─── Generate directional labels ───────────────────────────────
# Long:  future high exceeds threshold before future low hits stop
# Short: future low drops below threshold before future high hits stop
# Flat:  neither condition met within horizon
all_features = generate_labels(
    all_features,
    horizons=[1, 3, 5],
    threshold_atr_multiplier=0.5
)

# ─── Check label distribution ──────────────────────────────────
for h in [1, 3, 5]:
    col = f"label_h{h}"
    dist = all_features[col].value_counts(normalize=True)
    print(f"Horizon {h}: Long={dist.get('long', 0):.1%}  Short={dist.get('short', 0):.1%}  Flat={dist.get('flat', 0):.1%}")
```

**Expected distribution (approximately):**
```
Horizon 1: Long=25%   Short=25%   Flat=50%
Horizon 3: Long=30%   Short=30%   Flat=40%
Horizon 5: Long=35%   Short=35%   Flat=30%
```

### Step 5: Train/Validation/Test Split (Colab Cell 5)

```python
# ─── Time-ordered split (NO shuffling) ─────────────────────────
# This prevents look-ahead bias
all_features = all_features.sort_values("time").reset_index(drop=True)

n = len(all_features)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

train_df = all_features.iloc[:train_end]
val_df = all_features.iloc[train_end:val_end]
test_df = all_features.iloc[val_end:]

print(f"Train: {len(train_df):>10,} rows | {train_df['time'].min()} → {train_df['time'].max()}")
print(f"Val:   {len(val_df):>10,} rows | {val_df['time'].min()} → {val_df['time'].max()}")
print(f"Test:  {len(test_df):>10,} rows | {test_df['time'].min()} → {test_df['time'].max()}")
```

---

## Phase 1: Transformer Training

### Step 6: Prepare DataLoaders (Colab Cell 6)

```python
import torch
from torch.utils.data import Dataset, DataLoader

class ForexSequenceDataset(Dataset):
    """Sliding window dataset for Transformer training."""

    def __init__(self, df, feature_cols, label_col, lookback=128):
        self.features = df[feature_cols].values.astype("float32")
        self.labels = df[label_col].values
        self.lookback = lookback

        # Encode labels: long=0, short=1, flat=2
        label_map = {"long": 0, "short": 1, "flat": 2}
        self.encoded_labels = [label_map[l] for l in self.labels]

    def __len__(self):
        return len(self.features) - self.lookback

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx : idx + self.lookback])
        y = torch.tensor(self.encoded_labels[idx + self.lookback])
        return x, y


# ─── Feature columns (exclude metadata) ────────────────────────
exclude_cols = ["time", "pair", "label_h1", "label_h3", "label_h5",
                "open", "high", "low", "close", "volume"]
feature_cols = [c for c in all_features.columns if c not in exclude_cols]
print(f"Feature dimensions: {len(feature_cols)}")

# ─── Create datasets ───────────────────────────────────────────
train_dataset = ForexSequenceDataset(train_df, feature_cols, "label_h5", lookback=128)
val_dataset = ForexSequenceDataset(val_df, feature_cols, "label_h5", lookback=128)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

print(f"Train batches: {len(train_loader)}")
print(f"Val batches:   {len(val_loader)}")
```

### Step 7: Train Transformer Model (Colab Cell 7)

```python
from models.transformer import ForexTransformer
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import time

# ─── Initialize model ──────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ForexTransformer(
    input_dim=len(feature_cols),
    d_model=256,
    nhead=8,
    num_layers=4,
    dim_feedforward=1024,
    dropout=0.1,
    num_classes=3,
    num_horizons=3,
).to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Training on: {device}")

# ─── Optimizer & scheduler ─────────────────────────────────────
optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

# ─── Class weights (address label imbalance) ───────────────────
from sklearn.utils.class_weights import compute_class_weight
import numpy as np

class_weights = compute_class_weight(
    "balanced",
    classes=np.array([0, 1, 2]),
    y=np.array(train_dataset.encoded_labels[128:])
)
weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
criterion = torch.nn.CrossEntropyLoss(weight=weights_tensor, label_smoothing=0.1)

# ─── TensorBoard logging ───────────────────────────────────────
log_dir = "/content/drive/MyDrive/oandafx/logs/transformer_" + time.strftime("%Y%m%d_%H%M%S")
writer = SummaryWriter(log_dir)

# ─── Training loop ─────────────────────────────────────────────
best_val_loss = float("inf")
patience_counter = 0
PATIENCE = 10

for epoch in range(100):
    # Train
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        logits = model(batch_x)           # [batch, 3]
        loss = criterion(logits, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item() * batch_x.size(0)
        train_correct += (logits.argmax(dim=1) == batch_y).sum().item()
        train_total += batch_x.size(0)

    scheduler.step()

    # Validate
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            val_loss += loss.item() * batch_x.size(0)
            val_correct += (logits.argmax(dim=1) == batch_y).sum().item()
            val_total += batch_x.size(0)

    train_loss /= train_total
    val_loss /= val_total
    train_acc = train_correct / train_total
    val_acc = val_correct / val_total

    writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)
    writer.add_scalars("Accuracy", {"train": train_acc, "val": val_acc}, epoch)
    writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)

    print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f} | LR: {scheduler.get_last_lr()[0]:.2e}")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "/content/best_transformer.pt")
        print(f"  ✓ Saved best model (val_loss={val_loss:.4f})")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"  Early stopping at epoch {epoch+1}")
            break

writer.close()
print(f"\nBest validation loss: {best_val_loss:.4f}")
```

**Expected training time:**

| GPU | Full 10-year dataset | Last 2 years (fine-tune) |
|-----|---------------------|--------------------------|
| T4 (15 GB) | ~3–4 hours | ~30–45 minutes |
| A100 (40 GB) | ~1–2 hours | ~15–20 minutes |

---

## Phase 2: PPO Reinforcement Learning Training

### Step 8: Set Up RL Environment (Colab Cell 8)

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from models.rl_agent import OandaForexEnv

# ─── Load best Transformer for state augmentation ──────────────
model_transformer = ForexTransformer(
    input_dim=len(feature_cols), d_model=256, nhead=8,
    num_layers=4, dim_feedforward=1024, dropout=0.1,
    num_classes=3, num_horizons=3
).to(device)
model_transformer.load_state_dict(torch.load("/content/best_transformer.pt"))
model_transformer.eval()

# ─── Create vectorized training environments ───────────────────
def make_env(pair_data, seed):
    def _init():
        env = OandaForexEnv(
            features_df=pair_data,
            feature_cols=feature_cols,
            transformer=model_transformer,
            spread_pips=1.2,              # Average EUR_USD spread
            pip_value=10.0,               # Standard lot pip value
            max_leverage=50,              # OANDA US major leverage limit
            initial_balance=10_000,
            lookback=128,
        )
        env.reset(seed=seed)
        return env
    return _init

# ─── 4 parallel environments for faster training ───────────────
n_envs = 4
train_envs = SubprocVecEnv([
    make_env(train_df, seed=i) for i in range(n_envs)
])

# ─── Separate evaluation environment ───────────────────────────
eval_env = OandaForexEnv(
    features_df=val_df,
    feature_cols=feature_cols,
    transformer=model_transformer,
    spread_pips=1.2,
    pip_value=10.0,
    max_leverage=50,
    initial_balance=10_000,
    lookback=128,
)
```

### Step 9: Train PPO Agent (Colab Cell 9)

```python
# ─── Initialize PPO ────────────────────────────────────────────
ppo_agent = PPO(
    policy="MlpPolicy",
    env=train_envs,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    verbose=1,
    tensorboard_log="/content/drive/MyDrive/oandafx/logs/ppo_" + time.strftime("%Y%m%d_%H%M%S"),
    device=device,
    policy_kwargs=dict(
        net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]),
    ),
)

# ─── Callbacks ──────────────────────────────────────────────────
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="/content/best_ppo/",
    log_path="/content/ppo_eval_logs/",
    eval_freq=50_000,
    n_eval_episodes=10,
    deterministic=True,
)

checkpoint_callback = CheckpointCallback(
    save_freq=250_000,
    save_path="/content/ppo_checkpoints/",
    name_prefix="ppo_oandafx",
)

# ─── Train ──────────────────────────────────────────────────────
print("Starting PPO training — 5,000,000 timesteps...")
ppo_agent.learn(
    total_timesteps=5_000_000,
    callback=[eval_callback, checkpoint_callback],
    progress_bar=True,
)
print("PPO training complete.")
```

**Expected training time:**

| GPU | 5M timesteps | 1M timesteps (fine-tune) |
|-----|-------------|--------------------------|
| T4 (15 GB) | ~2–3 hours | ~25–35 minutes |
| A100 (40 GB) | ~1–1.5 hours | ~10–15 minutes |

### Step 10: Fit SL/TP Scaler (Colab Cell 10)

```python
import pickle
from sklearn.preprocessing import StandardScaler

# ─── Fit a scaler on ATR and volatility features for SL/TP module ──
sl_tp_features = ["atr_14", "atr_7_ratio", "atr_50_ratio", "vol_regime",
                   "sr_nearest_support_dist", "sr_nearest_resistance_dist"]

scaler = StandardScaler()
scaler.fit(train_df[sl_tp_features].dropna())

with open("/content/sl_tp_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("SL/TP scaler fitted and saved.")
```

---

## Phase 3: Validation & Export

### Step 11: Run Validation Backtest (Colab Cell 11)

```python
from backtest.engine import WalkForwardBacktest
from backtest.metrics import compute_metrics

# ─── Backtest on held-out test set ──────────────────────────────
backtest = WalkForwardBacktest(
    features_df=test_df,
    feature_cols=feature_cols,
    transformer=model_transformer,
    ppo_agent=ppo_agent,
    spread_pips=1.2,
    initial_balance=10_000,
)

results = backtest.run()
metrics = compute_metrics(results)

print("=" * 60)
print("VALIDATION BACKTEST RESULTS")
print("=" * 60)
print(f"  Total return:     {metrics['total_return']:.2%}")
print(f"  Annualized return:{metrics['annualized_return']:.2%}")
print(f"  Sharpe ratio:     {metrics['sharpe']:.2f}")
print(f"  Sortino ratio:    {metrics['sortino']:.2f}")
print(f"  Max drawdown:     {metrics['max_drawdown']:.2%}")
print(f"  Win rate:         {metrics['win_rate']:.2%}")
print(f"  Profit factor:    {metrics['profit_factor']:.2f}")
print(f"  Total trades:     {metrics['total_trades']}")
print(f"  Avg trade dur:    {metrics['avg_duration_bars']:.0f} bars")
print("=" * 60)

# ─── Minimum thresholds for deployment ──────────────────────────
DEPLOY_THRESHOLDS = {
    "sharpe": 1.0,
    "max_drawdown": -0.15,
    "profit_factor": 1.2,
    "win_rate": 0.45,
}

deploy_ok = (
    metrics["sharpe"] >= DEPLOY_THRESHOLDS["sharpe"]
    and metrics["max_drawdown"] >= DEPLOY_THRESHOLDS["max_drawdown"]
    and metrics["profit_factor"] >= DEPLOY_THRESHOLDS["profit_factor"]
    and metrics["win_rate"] >= DEPLOY_THRESHOLDS["win_rate"]
)

if deploy_ok:
    print("\n✅ Model PASSES deployment thresholds — safe to deploy.")
else:
    print("\n❌ Model FAILS deployment thresholds — DO NOT deploy.")
    print("   Review training data, features, or hyperparameters.")
```

**Minimum deployment thresholds:**

| Metric | Threshold | Reasoning |
|--------|-----------|-----------|
| Sharpe ratio | ≥ 1.0 | Consistent risk-adjusted returns |
| Max drawdown | ≤ 15% | Within account circuit breaker limits |
| Profit factor | ≥ 1.2 | Positive expected value after spreads |
| Win rate | ≥ 45% | Viable with 1.5:1+ RR ratio |

### Step 12: Save & Version Model Artifacts (Colab Cell 12)

```python
import json
import shutil
from datetime import datetime

# ─── Version string ─────────────────────────────────────────────
VERSION = "1.1.0"   # Increment manually or auto-generate
SAVE_DIR = Path(f"/content/drive/MyDrive/oandafx/models/v{VERSION}")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ─── Save Transformer ──────────────────────────────────────────
torch.save(model_transformer.state_dict(), SAVE_DIR / "transformer.pt")

# ─── Save PPO agent ────────────────────────────────────────────
ppo_agent.save(str(SAVE_DIR / "ppo_agent"))

# ─── Save SL/TP scaler ─────────────────────────────────────────
shutil.copy("/content/sl_tp_scaler.pkl", SAVE_DIR / "sl_tp_scaler.pkl")

# ─── Save metadata ─────────────────────────────────────────────
metadata = {
    "version": VERSION,
    "training_date": datetime.utcnow().strftime("%Y-%m-%d"),
    "training_data_range": [
        str(train_df["time"].min().date()),
        str(train_df["time"].max().date()),
    ],
    "pairs_trained_on": pairs,
    "base_timeframe": "M15",
    "feature_count": len(feature_cols),
    "transformer_params": sum(p.numel() for p in model_transformer.parameters()),
    "transformer_best_val_loss": float(best_val_loss),
    "ppo_timesteps": 5_000_000,
    "validation_metrics": {k: float(v) for k, v in metrics.items()},
    "deploy_approved": deploy_ok,
    "ensemble_weights": {"transformer": 0.6, "rl": 0.4},
    "colab_gpu": torch.cuda.get_device_name(0),
}

with open(SAVE_DIR / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

# ─── Update latest symlink ─────────────────────────────────────
latest_link = Path("/content/drive/MyDrive/oandafx/models/latest")
if latest_link.exists():
    latest_link.unlink()
latest_link.symlink_to(SAVE_DIR)

print(f"Model v{VERSION} saved to Google Drive: {SAVE_DIR}")
print(f"Files: {[f.name for f in SAVE_DIR.iterdir()]}")
```

### Step 13: Push to GitHub (Colab Cell 13)

```python
# ─── Copy artifacts to repo ─────────────────────────────────────
REPO_MODEL_DIR = Path(f"/content/oandafx/data/models/v{VERSION}")
REPO_MODEL_DIR.mkdir(parents=True, exist_ok=True)

shutil.copytree(SAVE_DIR, REPO_MODEL_DIR, dirs_exist_ok=True)

# ─── Update latest symlink in repo ──────────────────────────────
repo_latest = Path("/content/oandafx/data/models/latest")
if repo_latest.exists() or repo_latest.is_symlink():
    repo_latest.unlink()
repo_latest.symlink_to(f"v{VERSION}")

# ─── Git push ───────────────────────────────────────────────────
!cd /content/oandafx && \
    git add data/models/ && \
    git commit -m "Model v{VERSION} — Sharpe {metrics['sharpe']:.2f}, WR {metrics['win_rate']:.1%}" && \
    git push origin main

print(f"✅ Model v{VERSION} pushed to GitHub.")
print("   Run 'git pull' on VPS or wait for weekly auto-pull.")
```

---

## Retraining Schedule

| Schedule | Trigger | Scope | Estimated time |
|----------|---------|-------|----------------|
| Weekly (Sunday 06:00 UTC) | Cron / manual | Fine-tune on last 2 weeks of live data | 1–1.5 hours |
| Monthly (1st Sunday) | Cron / manual | Full retrain on all data including new month | 4–6 hours |
| Emergency | Drawdown > 8% in 30 days | Full retrain + hyperparameter review | 4–6 hours |
| New pair added | Manual | Fine-tune PPO on pair-specific data (500K steps) | 1 hour |

### Weekly Fine-Tune Workflow

For incremental weekly retraining, only update the last two weeks of data and fine-tune both models with a reduced learning rate:

```python
# Lower learning rate for fine-tuning
optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)

# Fewer epochs — just adjust to recent market conditions
for epoch in range(20):
    # ... training loop on recent data only ...

# Fine-tune PPO for 500K additional timesteps
ppo_agent.learn(total_timesteps=500_000, reset_num_timesteps=False)
```

---

## Session Management & Recovery

### Preventing Colab disconnects

Colab sessions disconnect after inactivity. Use these strategies:

1. **Keep the browser tab focused** — Colab detects idle tabs
2. **Use Colab Pro** — longer session limits and background execution
3. **Checkpoint frequently** — the training loop already saves to Google Drive on each improvement
4. **Resume from checkpoint** — if disconnected mid-training:

```python
# ─── Resume Transformer from last checkpoint ───────────────────
model.load_state_dict(torch.load("/content/drive/MyDrive/oandafx/models/latest/transformer.pt"))
# Adjust epoch counter and continue training loop

# ─── Resume PPO from last checkpoint ───────────────────────────
ppo_agent = PPO.load(
    "/content/ppo_checkpoints/ppo_oandafx_250000_steps",
    env=train_envs,
    device=device,
)
ppo_agent.learn(total_timesteps=remaining_steps, reset_num_timesteps=False)
```

### Colab runtime management

```python
# Check remaining GPU quota (Colab Pro)
!nvidia-smi

# Monitor GPU memory during training
import GPUtil
GPUtil.showUtilization()

# Force garbage collection between phases
import gc
gc.collect()
torch.cuda.empty_cache()
```

---

## Data Transfer Between VPS and Colab

### VPS → Colab (raw data)

Option A: **Via Google Drive sync** (recommended for initial setup)

```bash
# On VPS: upload raw data to Google Drive using rclone
sudo apt install rclone
rclone config   # set up Google Drive remote
rclone sync /home/botuser/oandafx/data/raw gdrive:oandafx/data/raw --progress
```

Option B: **Via GitHub** (for incremental updates)

```bash
# On VPS: commit recent data and push
cd /home/botuser/oandafx
git add data/raw/
git commit -m "Data update $(date +%Y-%m-%d)"
git push origin main

# On Colab: pull latest data
!cd /content/oandafx && git pull origin main
```

### Colab → VPS (trained models)

Models flow through GitHub (see Step 13). On the VPS, the weekly cron job or manual pull fetches the latest model:

```bash
# On VPS
cd /home/botuser/oandafx
git pull origin main
sudo systemctl restart oandafx
```

---

## TensorBoard Monitoring

View training progress in real-time from Colab:

```python
%load_ext tensorboard
%tensorboard --logdir /content/drive/MyDrive/oandafx/logs/
```

Key metrics to watch:
- **Train/Val loss convergence** — val loss should decrease and stabilize
- **Val accuracy** — should plateau around 45–55% (3-class, better than 33% random)
- **Learning rate decay** — should follow cosine curve
- **PPO episode reward** — should trend upward and stabilize
- **PPO entropy** — should gradually decrease (exploitation over exploration)

---

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| `CUDA out of memory` | Batch size too large | Reduce `batch_size` to 128 or 64 |
| Transformer overfitting | Not enough regularization | Increase dropout to 0.2, add more weight decay |
| PPO reward stays flat | Reward function too sparse | Add intermediate rewards, check spread sim |
| Val loss diverges | Learning rate too high | Reduce LR to 5e-5 |
| Colab session expires | Idle timeout | Checkpoint to Drive, resume from last save |
| `Drive quota exceeded` | Too many checkpoints | Delete old checkpoints, keep last 3 versions |
| Feature NaN values | Missing data in raw Parquet | Run data quality checks, fill gaps |
| Label imbalance severe | Threshold too tight/loose | Adjust `threshold_atr_multiplier` (try 0.3–0.7) |
| Model predicts flat always | Training data too noisy | Filter low-volatility periods, increase ATR threshold |
| Git push fails from Colab | Auth issue | Use personal access token: `git remote set-url origin https://TOKEN@github.com/...` |
