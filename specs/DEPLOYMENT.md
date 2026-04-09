# DEPLOYMENT.md — VS Code, GitHub, Colab & Live Rollout

## Development Environment

### Required Tools

```
VS Code                  — Primary IDE
Python 3.10+             — Runtime
Git + GitHub             — Version control + CI
Google Colab             — GPU for optimization (Phase 8 if needed)
OANDA fxTrade Practice   — Paper trading (before any live capital)
OANDA fxTrade Live       — Live trading (after paper trading validation)
```

### VS Code Extensions (Recommended)

```
Python (Microsoft)
Pylance (Microsoft)
Jupyter (Microsoft)     — For notebooks
GitLens                 — Git history in editor
Python Test Explorer    — Run pytest from sidebar
YAML (Red Hat)          — Config file editing
```

### Python Dependencies

```txt
# requirements.txt

# OANDA API
oandapyV20>=0.7.2

# Data processing
numpy>=1.24.0
pandas>=2.0.0

# Storage
sqlalchemy>=2.0.0

# Configuration
pyyaml>=6.0
python-dotenv>=1.0.0

# Visualization (notebooks only)
matplotlib>=3.7.0
plotly>=5.15.0
jupyter>=1.0.0

# Backtesting optimization
optuna>=3.0.0          # Bayesian parameter search (optional)

# Utilities
pytz>=2023.3
requests>=2.31.0
structlog>=23.1.0      # Structured JSON logging

# Testing
pytest>=7.4.0
pytest-cov>=4.1.0

# Type checking
mypy>=1.5.0
```

---

## GitHub Repository Setup

### Repository Structure

```bash
# Initialize
git init
git remote add origin https://github.com/YOUR_USERNAME/forex-wave-bot.git

# Branch strategy
main           — Production-ready code only
develop        — Integration branch
feature/*      — Individual feature branches
backtest/*     — Backtest experiment branches
```

### .gitignore

```gitignore
# Environment and secrets — NEVER commit these
.env
*.env
config/secrets.yaml

# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
venv/
env/

# Data — too large for GitHub
data/
*.db
*.sqlite
*.parquet
*.csv

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# Reports — local only
reports/

# IDE
.vscode/settings.json
.idea/

# OS
.DS_Store
Thumbs.db
```

### GitHub Actions CI

```yaml
# .github/workflows/test.yml
name: WaveBot Tests

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt

      - name: Run tests
        run: |
          pytest tests/ -v --cov=src --cov-report=term-missing

      - name: Check coverage
        run: |
          pytest tests/ --cov=src --cov-fail-under=80
```

---

## Google Colab Setup

### Data Collection Notebook

The data collection notebook runs on Colab free tier (no GPU needed for data pulls). Mount Google Drive to persist the SQLite database between sessions.

```python
# notebooks/01_data_collection.ipynb

# Cell 1: Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Cell 2: Install dependencies
!pip install oandapyV20 sqlalchemy python-dotenv -q

# Cell 3: Clone repo
!git clone https://github.com/YOUR_USERNAME/forex-wave-bot.git
%cd forex-wave-bot

# Cell 4: Set credentials (use Colab secrets, not .env)
import os
os.environ["OANDA_API_TOKEN"] = "your_token"
os.environ["OANDA_ACCOUNT_ID"] = "your_account"
os.environ["OANDA_ENVIRONMENT"] = "practice"

# Cell 5: Run collection
from src.data.data_collector import DataCollector

collector = DataCollector(db_path="/content/drive/MyDrive/wavebot/data.db")

pairs = ["EUR_USD", "GBP_USD", "USD_JPY", "USD_CAD", "AUD_USD"]
timeframes = ["M1", "M5", "M15", "H1", "H4", "D"]

for pair in pairs:
    for tf in timeframes:
        print(f"Collecting {pair} {tf}...")
        collector.collect_range(pair, tf, start="2018-01-01")
        print(f"  ✓ Done")
```

### Walk-Forward Optimization on GPU

GPU acceleration is useful for Bayesian parameter optimization (Phase 8 RL training, or if optuna search space is large).

```python
# notebooks/04_backtest_analysis.ipynb

# Colab GPU setup
!nvidia-smi  # Confirm GPU available

# Run walk-forward with GPU-accelerated optuna
from src.backtest.engine import WalkForwardValidator
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

validator = WalkForwardValidator(
    data_path="/content/drive/MyDrive/wavebot/data.db"
)

results = validator.run(
    instrument="EUR_USD",
    full_start="2018-01-01",
    full_end="2023-12-31",
    train_months=6,
    test_months=2,
)

print(results.summary())
results.save("/content/drive/MyDrive/wavebot/reports/")
```

---

## Deployment Phases

### Phase 1: Practice API — Paper Trading

Run the bot against OANDA's practice environment. The practice API is **identical** to live in behavior — same endpoints, same latency, same order types. The only difference is the money is simulated.

```bash
# Set environment to practice
OANDA_ENVIRONMENT=practice

# Run bot in paper mode
python main.py \
  --mode paper \
  --pairs EUR_USD,GBP_USD,USD_JPY \
  --timeframes M5,M15,H1,H4 \
  --log-level INFO
```

**Paper trading duration: Minimum 4 weeks, ideally 6–8 weeks.**

During this period, verify:
- Wave detection matches visual chart analysis (spot-check manually)
- Orders place and fill as expected on practice account
- Risk calculations match theoretical values
- Circuit breakers fire correctly in test scenarios
- Spread filter works (manually widen spread in config to test rejection)
- No memory leaks (run `top` or Activity Monitor while bot runs)

### Phase 2: Live — Micro Lots Only

After passing paper trading validation:

```bash
OANDA_ENVIRONMENT=live
```

Start with 0.01 lots (1,000 units) per trade regardless of account size. This is not about the money — it is about verifying that the live API behavior matches the practice API behavior. Run at micro lot size for two weeks.

Verify:
- Fills occur at expected prices (compare to practice fills)
- Spreads match expectations
- SL and TP orders are attached correctly
- Transaction logs match OANDA's transaction history

### Phase 3: Live — Full Risk

After two weeks of verified micro lot behavior, scale to normal position sizing (1% risk per trade).

```python
# config/risk.yaml
risk:
  max_risk_per_trade: 0.01   # 1% — now active at full size
```

---

## Running the Bot

### main.py Entry Point

```python
# main.py
import sys
import signal
import argparse
from src.bot import WaveBot
from src.utils.logger import setup_logger
from src.utils.config import load_config

def main():
    parser = argparse.ArgumentParser(description="WaveBot — OANDA Wave Trading System")
    parser.add_argument("--mode", choices=["paper", "live", "backtest"], default="paper")
    parser.add_argument("--pairs", default="EUR_USD,GBP_USD,USD_JPY")
    parser.add_argument("--timeframes", default="M5,M15,H1,H4")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    log = setup_logger(args.log_level)
    config = load_config()

    bot = WaveBot(
        mode=args.mode,
        pairs=args.pairs.split(","),
        timeframes=args.timeframes.split(","),
        config=config,
    )

    # Graceful shutdown on Ctrl+C
    def shutdown(sig, frame):
        log.info("Shutdown signal received — closing gracefully")
        bot.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    log.info(f"WaveBot starting — mode={args.mode}, pairs={args.pairs}")
    bot.run()

if __name__ == "__main__":
    main()
```

### Keeping the Bot Running (Linux / Mac)

```bash
# Run as background process with logging
nohup python main.py --mode live > logs/wavebot.log 2>&1 &

# Check if running
ps aux | grep main.py

# View live logs
tail -f logs/wavebot.log

# Stop gracefully
kill -SIGTERM $(cat wavebot.pid)
```

---

## Monitoring

### Daily Review Checklist

Run this every morning before market open:

```bash
# Check last 24h trade log
python scripts/daily_review.py --date yesterday

# Output:
# Trades placed: 3
# Trades closed: 2 (1 TP, 1 SL)
# Open trades: 1 (EUR_USD LONG, +12 pips)
# Daily P&L: +$47.20
# Drawdown today: 0.23%
# Circuit breaker status: ACTIVE (no limits hit)
# API errors last 24h: 0
# Maintenance events: 0
```

### Performance Dashboard

```bash
# Run weekly performance report
python scripts/performance_report.py --period 30d

# Output metrics:
# Win rate (30d): 61.4%
# Profit factor: 1.78
# Total pips: +312
# Sharpe (annualized): 1.54
# Max drawdown: 3.2%
# Best pair: GBP_USD (+142 pips)
# Worst pair: USD_JPY (-23 pips)
```

---

## Safety Checklist Before Going Live

- [ ] Walk-forward backtest passes all minimum thresholds
- [ ] Paper traded for minimum 4 weeks
- [ ] Paper trade metrics match backtest expectations (within 15%)
- [ ] Circuit breakers tested — confirmed they fire correctly
- [ ] Emergency close tested — all trades close on breaker trigger
- [ ] API token is from practice account (practice) or live account (live)
- [ ] `.env` file is NOT committed to GitHub
- [ ] `OANDA_ENVIRONMENT` is set correctly
- [ ] Max position size calculated correctly for account balance
- [ ] Bot ran continuously for 48h without crash or memory leak
- [ ] Maintenance window behavior tested — bot resumes correctly
- [ ] Spread filter tested — entries rejected when spread is high
- [ ] Trade log writing correctly to SQLite
- [ ] Transaction sync with OANDA history matches local logs
