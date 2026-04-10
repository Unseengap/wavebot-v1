"""FastAPI monitoring dashboard — exposes account status, trades, and metrics."""

import os
import logging
import secrets
from datetime import datetime, timezone

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import HTMLResponse

logger = logging.getLogger("monitoring.dashboard")

app = FastAPI(title="OandaFX Dashboard", version="1.0.0")
security = HTTPBasic()

# Global references set by the bot on startup
_account_manager = None
_start_time = datetime.now(timezone.utc)


def set_account_manager(manager):
    """Register the AccountManager instance for the dashboard to query."""
    global _account_manager
    _account_manager = manager


def _verify_auth(credentials: HTTPBasicCredentials = Depends(security)):
    """Basic HTTP auth using env vars."""
    expected_user = os.environ.get("DASHBOARD_USERNAME", "admin")
    expected_pass = os.environ.get("DASHBOARD_PASSWORD", "changeme")

    user_ok = secrets.compare_digest(credentials.username.encode(), expected_user.encode())
    pass_ok = secrets.compare_digest(credentials.password.encode(), expected_pass.encode())

    if not (user_ok and pass_ok):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials


@app.get("/api/health")
def health_check():
    """Bot health check endpoint."""
    uptime = (datetime.now(timezone.utc) - _start_time).total_seconds()
    hours = int(uptime // 3600)
    minutes = int((uptime % 3600) // 60)

    mgr_status = _account_manager.get_status() if _account_manager else {}

    return {
        "status": "healthy",
        "uptime_seconds": int(uptime),
        "uptime_human": f"{hours}h {minutes}m",
        "accounts_active": mgr_status.get("active_sessions", 0),
        "last_heartbeat": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/accounts", dependencies=[Depends(_verify_auth)])
def get_accounts():
    """All account statuses."""
    if _account_manager is None:
        return {"accounts": [], "error": "AccountManager not initialized"}
    return _account_manager.get_status()


@app.get("/api/trades", dependencies=[Depends(_verify_auth)])
def get_trades():
    """All open trades across accounts."""
    if _account_manager is None:
        return {"open_trades": [], "total_open": 0}

    all_trades = []
    for alias, client in _account_manager.clients.items():
        try:
            trades = client.get_open_trades()
            for t in trades:
                t["account"] = alias
            all_trades.extend(trades)
        except Exception as e:
            logger.error(f"Failed to fetch trades for {alias}: {e}")

    return {"open_trades": all_trades, "total_open": len(all_trades)}


@app.get("/api/performance", dependencies=[Depends(_verify_auth)])
def get_performance():
    """Performance metrics summary."""
    if _account_manager is None:
        return {"error": "AccountManager not initialized"}

    performance = {}
    for alias, client in _account_manager.clients.items():
        try:
            summary = client.get_account_summary()
            performance[alias] = {
                "balance": float(summary.get("balance", 0)),
                "nav": float(summary.get("NAV", 0)),
                "unrealized_pnl": float(summary.get("unrealizedPL", 0)),
                "margin_used": float(summary.get("marginUsed", 0)),
                "margin_available": float(summary.get("marginAvailable", 0)),
            }
        except Exception as e:
            performance[alias] = {"error": str(e)}

    return performance


@app.get("/api/signals", dependencies=[Depends(_verify_auth)])
def get_recent_signals():
    """Recent model signals."""
    if _account_manager is None:
        return {"signals": []}

    signals = []
    for alias, session in _account_manager.sessions.items():
        status = session.get_status()
        signals.append({
            "account": alias,
            "last_signal_time": status.get("last_signal_time"),
            "last_trade_time": status.get("last_trade_time"),
        })
    return {"signals": signals}


@app.post("/api/pause/{alias}", dependencies=[Depends(_verify_auth)])
def pause_account(alias: str):
    """Pause trading on a specific account."""
    if _account_manager is None:
        raise HTTPException(400, "AccountManager not initialized")
    _account_manager.pause_account(alias)
    return {"status": "paused", "account": alias}


@app.post("/api/resume/{alias}", dependencies=[Depends(_verify_auth)])
def resume_account(alias: str):
    """Resume a paused account."""
    if _account_manager is None:
        raise HTTPException(400, "AccountManager not initialized")
    _account_manager.resume_account(alias)
    return {"status": "resumed", "account": alias}


@app.get("/", response_class=HTMLResponse)
def dashboard_page(creds: HTTPBasicCredentials = Depends(_verify_auth)):
    """Minimal HTML dashboard."""
    return """<!DOCTYPE html>
<html>
<head><title>OandaFX Dashboard</title>
<style>
    body { font-family: 'Segoe UI', sans-serif; margin: 40px; background: #1a1a2e; color: #e0e0e0; }
    h1 { color: #00d4ff; }
    .card { background: #16213e; padding: 20px; border-radius: 8px; margin: 10px 0; }
    a { color: #00d4ff; }
</style>
</head>
<body>
    <h1>OandaFX Dashboard</h1>
    <div class="card">
        <h3>API Endpoints</h3>
        <ul>
            <li><a href="/api/health">/api/health</a> — Bot health check</li>
            <li><a href="/api/accounts">/api/accounts</a> — Account statuses</li>
            <li><a href="/api/trades">/api/trades</a> — Open trades</li>
            <li><a href="/api/performance">/api/performance</a> — Performance metrics</li>
            <li><a href="/api/signals">/api/signals</a> — Recent signals</li>
        </ul>
    </div>
</body>
</html>"""
