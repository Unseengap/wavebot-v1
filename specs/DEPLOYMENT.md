# DEPLOYMENT.md — Vultr VPS Deployment Guide

## Overview

OandaFX runs as a persistent process on a Vultr Cloud Compute instance. The bot is managed by `systemd` for automatic restarts, containerized with Docker for reproducibility, and monitored via a FastAPI dashboard. This guide covers full server setup from a fresh Vultr instance.

---

## Recommended Vultr Configuration

| Setting | Recommendation |
|---------|---------------|
| Plan | Cloud Compute — Regular Performance |
| vCPU | 2 vCPU (minimum), 4 vCPU (recommended for 5+ accounts) |
| RAM | 4 GB (minimum), 8 GB (recommended) |
| Storage | 80 GB NVMe SSD |
| OS | Ubuntu 22.04 LTS x64 |
| Region | New York (NJ) — closest to OANDA's US servers |
| IPv4 | Yes (static IP for firewall whitelisting) |
| Backups | Enable Vultr automatic backups |
| DDoS Protection | Enable |

**Estimated cost:** $24–$48/month depending on plan selected.

**Why Vultr over AWS/GCP:** Lower latency to OANDA's New York trading servers, predictable pricing, and simpler networking setup for a single-purpose trading server.

---

## Step 1: Initial Server Setup

SSH into your new Vultr instance as root:

```bash
ssh root@YOUR_VPS_IP
```

### Create a non-root user

```bash
adduser botuser
usermod -aG sudo botuser
# Switch to the new user for all subsequent steps
su - botuser
```

### System updates

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y git curl wget htop ufw fail2ban
```

### Configure firewall

```bash
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh                    # Port 22
sudo ufw allow 8080/tcp               # Bot monitoring dashboard
sudo ufw enable
sudo ufw status
```

### Configure fail2ban (brute force protection)

```bash
sudo systemctl enable fail2ban
sudo systemctl start fail2ban
```

---

## Step 2: Install Python & Dependencies

```bash
# Install Python 3.11
sudo apt install -y python3.11 python3.11-venv python3.11-dev python3-pip

# Verify
python3.11 --version

# Install system dependencies for TA-Lib
sudo apt install -y build-essential libta-lib-dev libhdf5-dev libssl-dev
```

### Install Docker (optional but recommended)

```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker botuser
newgrp docker
docker --version
```

---

## Step 3: Clone the Repository

```bash
cd /home/botuser
git clone https://github.com/your-org/oandafx.git
cd oandafx
```

### Set up Python virtual environment

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Verify OANDA API connectivity

```bash
python scripts/verify_connection.py
# Should print: "✅ All accounts connected successfully"
```

---

## Step 4: Configure Credentials

```bash
# Copy example configs
cp config/accounts.yaml.example config/accounts.yaml
cp config/bot_config.yaml.example config/bot_config.yaml
cp .env.example .env

# Edit with your real credentials
nano config/accounts.yaml     # Add your OANDA API keys and account IDs
nano config/bot_config.yaml   # Set pairs, timeframes, strategy
nano .env                     # Set Telegram bot token, email, etc.

# Secure the credentials file
chmod 600 config/accounts.yaml
chmod 600 .env
```

---

## Step 5: Download Historical Data

This only needs to be done once. Subsequent updates are incremental.

```bash
source venv/bin/activate

python scripts/download_history.py \
  --pairs EUR_USD GBP_USD USD_JPY AUD_USD USD_CHF USD_CAD NZD_USD \
  --granularities M15 H1 H4 D \
  --start 2010-01-01 \
  --output-dir data/raw \
  --verbose

# Expected output: ~2-4 GB of Parquet files, takes 20-40 minutes
```

---

## Step 6: Upload Trained Model

After training on Google Colab (see [TRAINING.md](TRAINING.md)), push the model to GitHub:

```bash
# On Colab after training
git add data/models/
git commit -m "New model v1.0.0 trained on 2010-2024"
git push origin main
```

Then on the VPS:

```bash
cd /home/botuser/oandafx
git pull origin main
```

---

## Step 7: Run Backtests (Validation)

Before going live, confirm the model produces expected results on the VPS environment:

```bash
source venv/bin/activate

python scripts/run_backtest.py \
  --model data/models/latest/transformer.pt \
  --start 2023-01-01 \
  --end 2024-12-31 \
  --output reports/backtest_validation.html

# Review the report
cat reports/backtest_validation.html | python -c "import sys; print(sys.stdin.read()[:500])"
```

---

## Step 8: Set Up systemd Service

This ensures the bot automatically restarts on crash or server reboot.

```bash
sudo nano /etc/systemd/system/oandafx.service
```

Paste the following:

```ini
[Unit]
Description=OandaFX Trading Bot
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=botuser
WorkingDirectory=/home/botuser/oandafx
ExecStart=/home/botuser/oandafx/venv/bin/python scripts/start_bot.py --config config/accounts.yaml
Restart=on-failure
RestartSec=30
StartLimitInterval=200
StartLimitBurst=5

# Environment
EnvironmentFile=/home/botuser/oandafx/.env

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=oandafx

# Security
NoNewPrivileges=yes
PrivateTmp=yes

[Install]
WantedBy=multi-user.target
```

Enable and start the service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable oandafx
sudo systemctl start oandafx
sudo systemctl status oandafx
```

View live logs:

```bash
sudo journalctl -u oandafx -f
```

---

## Step 9: Set Up Monitoring Dashboard

The bot exposes a FastAPI monitoring dashboard on port 8080.

Access it at: `http://YOUR_VPS_IP:8080`

Available endpoints:

| Endpoint | Description |
|----------|-------------|
| `GET /` | Dashboard HTML page |
| `GET /api/accounts` | All account statuses (JSON) |
| `GET /api/trades` | All open trades across accounts |
| `GET /api/performance` | Performance metrics |
| `GET /api/signals` | Recent model signals |
| `GET /api/health` | Bot health check |
| `POST /api/pause/{alias}` | Pause a specific account |
| `POST /api/resume/{alias}` | Resume a paused account |

To add basic auth to the dashboard:

```bash
# In .env
DASHBOARD_USERNAME=admin
DASHBOARD_PASSWORD=your-secure-password-here
```

---

## Step 10: Configure Daily Data Updates (Cron)

```bash
crontab -e
```

Add the following lines:

```cron
# Incremental data update every day at 22:30 UTC (after market close)
30 22 * * 1-5 /home/botuser/oandafx/venv/bin/python /home/botuser/oandafx/scripts/download_history.py --mode incremental >> /home/botuser/oandafx/data/logs/data_update.log 2>&1

# Daily account reconciliation at 23:00 UTC
0 23 * * 1-5 /home/botuser/oandafx/venv/bin/python /home/botuser/oandafx/scripts/reconcile.py >> /home/botuser/oandafx/data/logs/reconcile.log 2>&1

# Weekly git pull to get latest model
0 5 * * 0 cd /home/botuser/oandafx && git pull origin main && sudo systemctl restart oandafx >> /home/botuser/oandafx/data/logs/update.log 2>&1
```

---

## Docker Deployment (Alternative)

If you prefer Docker over bare-metal systemd:

```bash
# Build the image
docker build -t oandafx:latest .

# Run with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f oandafx

# Restart
docker-compose restart oandafx
```

`docker-compose.yml` mounts `config/` and `data/` as volumes so credentials and data persist across container restarts.

---

## VPS Maintenance

### Updating the bot

```bash
sudo systemctl stop oandafx
cd /home/botuser/oandafx
git pull origin main
source venv/bin/activate
pip install -r requirements.txt
sudo systemctl start oandafx
```

### Server resource monitoring

```bash
# CPU and memory
htop

# Disk usage
df -h
du -sh data/

# Bot process
ps aux | grep start_bot
```

### Backup strategy

Vultr automatic backups run daily and retain 7 days. Additionally, push critical data to GitHub:

```bash
# Exclude raw data (too large) but backup configs and models
git add config/ data/models/ data/logs/
git commit -m "Backup $(date +%Y-%m-%d)"
git push origin main
```

---

## Troubleshooting

| Symptom | Likely cause | Solution |
|---------|-------------|----------|
| Bot not starting | Config error | Check `sudo journalctl -u oandafx -n 50` |
| `401 Unauthorized` | Bad API key | Verify key in OANDA portal, check `accounts.yaml` |
| `503 Service Unavailable` | OANDA maintenance | Normal — bot retries automatically |
| `Insufficient margin` | Position too large | Reduce `max_risk_per_trade` in config |
| High memory usage | Large feature buffer | Reduce `lookback_bars` in `bot_config.yaml` |
| Dashboard not accessible | Firewall | Run `sudo ufw allow 8080/tcp` |
| Duplicate orders | Clock drift | Run `sudo ntpdate pool.ntp.org` |

### Check VPS clock sync (critical for order timing)

```bash
timedatectl status
# Should show: System clock synchronized: yes

# If not synced:
sudo apt install -y ntp
sudo systemctl enable ntp
sudo systemctl start ntp
```
