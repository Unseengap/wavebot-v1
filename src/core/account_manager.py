"""Multi-account orchestrator — manages all OANDA accounts from a single process."""

import logging
import threading
from pathlib import Path
from typing import Optional

import yaml

from src.core.api_client import APIClient
from src.core.session import TradingSession

logger = logging.getLogger("core.account_manager")


class AccountManager:
    """Orchestrates all OANDA trading accounts.

    Loads account configs, instantiates one APIClient + TradingSession per account,
    manages startup/shutdown, and tracks cross-account exposure.
    """

    def __init__(self, config_path: str = "config/accounts.yaml", bot_config_path: str = "config/bot_config.yaml"):
        self.config_path = config_path
        self.bot_config_path = bot_config_path
        self.accounts: dict[str, dict] = {}
        self.sessions: dict[str, TradingSession] = {}
        self.clients: dict[str, APIClient] = {}
        self._lock = threading.Lock()
        self._strategies: dict = {}

        self._load_configs()

    def _load_configs(self):
        """Load account and bot configuration from YAML files."""
        config_file = Path(self.config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Account config not found: {self.config_path}")

        with open(config_file, "r") as f:
            config = yaml.safe_load(f)

        self._raw_config = config

        bot_config_file = Path(self.bot_config_path)
        if bot_config_file.exists():
            with open(bot_config_file, "r") as f:
                bot_config = yaml.safe_load(f)
            self._strategies = bot_config.get("strategies", {})

        for acct in config.get("accounts", []):
            alias = acct["alias"]
            self.accounts[alias] = acct

    def start_all(self):
        """Start trading sessions for all enabled accounts."""
        for alias, acct_config in self.accounts.items():
            if not acct_config.get("enabled", True):
                logger.info(f"Account '{alias}' is disabled, skipping")
                continue
            self._start_account(alias, acct_config)

        logger.info(
            f"Started {len(self.sessions)} of {len(self.accounts)} accounts",
            extra={"event": "bot_started"},
        )

    def stop_all(self):
        """Gracefully stop all trading sessions."""
        for alias, session in self.sessions.items():
            try:
                session.stop()
            except Exception as e:
                logger.error(f"Error stopping session '{alias}': {e}")

        logger.info("All sessions stopped", extra={"event": "bot_stopped"})

    def _start_account(self, alias: str, acct_config: dict):
        """Initialize and start a single account."""
        client = APIClient(
            api_key=acct_config["api_key"],
            account_id=acct_config["account_id"],
            environment=acct_config.get("environment", "practice"),
        )
        self.clients[alias] = client

        strategy_name = acct_config.get("strategy", "default")
        strategy_config = self._strategies.get(strategy_name, {})

        # Merge account-level pair override into strategy
        if acct_config.get("pairs"):
            strategy_config["pairs"] = acct_config["pairs"]

        session = TradingSession(
            alias=alias,
            api_client=client,
            config=strategy_config,
        )
        self.sessions[alias] = session
        session.start()

        logger.info(
            f"Account '{alias}' started ({acct_config.get('environment', 'practice')})",
            extra={"event": "account_connected", "account": alias},
        )

    def add_account(self, account_config: dict):
        """Hot-add an account without restarting other sessions."""
        alias = account_config["alias"]
        with self._lock:
            if alias in self.sessions:
                raise ValueError(f"Account '{alias}' already exists")
            self.accounts[alias] = account_config
            self._start_account(alias, account_config)

    def remove_account(self, alias: str):
        """Gracefully stop and remove an account."""
        with self._lock:
            if alias in self.sessions:
                self.sessions[alias].stop()
                del self.sessions[alias]
            if alias in self.clients:
                del self.clients[alias]
            if alias in self.accounts:
                del self.accounts[alias]
        logger.info(f"Account '{alias}' removed", extra={"account": alias})

    def pause_account(self, alias: str):
        """Pause trading on an account (keeps it registered)."""
        if alias in self.sessions:
            self.sessions[alias].stop()

    def resume_account(self, alias: str):
        """Resume a paused account."""
        if alias in self.accounts and alias in self.sessions:
            self.sessions[alias].start()

    def get_status(self) -> dict:
        """Return status of all accounts."""
        statuses = {}
        for alias, session in self.sessions.items():
            statuses[alias] = session.get_status()
            statuses[alias]["environment"] = self.accounts[alias].get("environment", "practice")
        return {
            "total_accounts": len(self.accounts),
            "active_sessions": sum(1 for s in self.sessions.values() if s.is_running),
            "accounts": statuses,
        }

    def get_system_exposure(self, currency: str, direction: str) -> int:
        """Track system-wide exposure to a currency across all accounts."""
        exposure = 0
        for alias, client in self.clients.items():
            try:
                open_trades = client.get_open_trades()
                for trade in open_trades:
                    instrument = trade.get("instrument", "")
                    trade_dir = "long" if int(trade.get("currentUnits", 0)) > 0 else "short"
                    if currency in instrument and trade_dir == direction:
                        exposure += abs(int(trade.get("currentUnits", 0)))
            except Exception:
                pass
        return exposure
