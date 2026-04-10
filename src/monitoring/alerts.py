"""Telegram alerting — sends trade notifications and critical alerts."""

import os
import logging
import asyncio
from typing import Optional

import aiohttp

logger = logging.getLogger("monitoring.alerts")


class TelegramAlerter:
    """Sends formatted HTML alerts to a Telegram bot."""

    def __init__(self, token: str = None, chat_id: str = None):
        self.token = token or os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")
        self.base_url = f"https://api.telegram.org/bot{self.token}"
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def send(self, message: str, parse_mode: str = "HTML"):
        """Send a message to Telegram. Logs errors silently."""
        if not self.token or not self.chat_id:
            return
        try:
            session = await self._get_session()
            async with session.post(
                f"{self.base_url}/sendMessage",
                json={"chat_id": self.chat_id, "text": message, "parse_mode": parse_mode},
            ) as resp:
                if resp.status != 200:
                    logger.error(f"Telegram send failed: {resp.status}")
        except Exception as e:
            logger.error(f"Telegram send error: {e}")

    def send_sync(self, message: str):
        """Synchronous wrapper for sending alerts."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(self.send(message))
            else:
                loop.run_until_complete(self.send(message))
        except RuntimeError:
            asyncio.run(self.send(message))

    async def trade_opened(self, data: dict):
        emoji = "\U0001f7e2" if data.get("direction") == "long" else "\U0001f534"
        msg = (
            f"{emoji} <b>Trade Opened</b>\n"
            f"Account: <code>{data.get('account', '')}</code>\n"
            f"Pair: <b>{data.get('instrument', '')}</b>\n"
            f"Direction: {data.get('direction', '').upper()}\n"
            f"Entry: {data.get('entry_price', '')}\n"
            f"SL: {data.get('sl_price', '')} ({data.get('sl_pips', 0):.1f} pips)\n"
            f"TP: {data.get('tp_price', '')} ({data.get('tp_pips', 0):.1f} pips)\n"
            f"RR: {data.get('rr_ratio', 0):.1f}:1\n"
            f"Risk: {data.get('risk_pct', 0):.1%}\n"
            f"Confidence: {data.get('model_confidence', 0):.0%}"
        )
        await self.send(msg)

    async def trade_closed(self, data: dict):
        pnl = data.get("pnl", 0)
        emoji = "\u2705" if pnl >= 0 else "\u274c"
        msg = (
            f"{emoji} <b>Trade Closed</b>\n"
            f"Account: <code>{data.get('account', '')}</code>\n"
            f"Pair: <b>{data.get('instrument', '')}</b>\n"
            f"P&L: <b>{'+' if pnl >= 0 else ''}{pnl:.2f} USD</b>\n"
            f"Duration: {data.get('duration_bars', 0)} bars\n"
            f"Closed by: {data.get('close_reason', '')}"
        )
        await self.send(msg)

    async def circuit_breaker(self, account: str, reason: str, details: dict):
        msg = (
            f"\U0001f6a8 <b>CIRCUIT BREAKER TRIGGERED</b> \U0001f6a8\n\n"
            f"Account: <code>{account}</code>\n"
            f"Reason: <b>{reason}</b>\n"
            f"Daily P&L: {details.get('daily_pnl', 'N/A')}\n"
            f"Drawdown: {details.get('drawdown', 'N/A')}\n\n"
            f"\u26a0\ufe0f Trading halted. Manual review required."
        )
        await self.send(msg)

    async def daily_summary(self, data: dict):
        emoji = "\U0001f4c8" if data.get("day_pnl", 0) >= 0 else "\U0001f4c9"
        msg = (
            f"{emoji} <b>Daily Summary — {data.get('date', '')}</b>\n\n"
            f"Account: <code>{data.get('account', '')}</code>\n"
            f"Day P&L: <b>{data.get('day_pnl', 0):+.2f} USD ({data.get('day_pnl_pct', 0):+.2%})</b>\n"
            f"Trades: {data.get('trades_opened', 0)} opened, {data.get('trades_closed', 0)} closed\n"
            f"Wins: {data.get('wins', 0)} | Losses: {data.get('losses', 0)}\n"
            f"Win rate: {data.get('win_rate', 0):.0%}\n"
            f"Balance: {data.get('balance', 0):.2f} USD\n"
            f"Drawdown: {data.get('drawdown', 0):.2%}"
        )
        await self.send(msg)

    async def weekly_report(self, data: dict):
        msg = (
            f"\U0001f4ca <b>Weekly Report — {data.get('week_start', '')} to {data.get('week_end', '')}</b>\n\n"
            f"Account: <code>{data.get('account', '')}</code>\n"
            f"Week P&L: <b>{data.get('week_pnl', 0):+.2f} USD ({data.get('week_pnl_pct', 0):+.2%})</b>\n"
            f"Total trades: {data.get('total_trades', 0)}\n"
            f"Win rate: {data.get('win_rate', 0):.0%}\n"
            f"Profit factor: {data.get('profit_factor', 0):.2f}\n"
            f"Sharpe (week): {data.get('sharpe', 0):.2f}\n"
            f"Max drawdown: {data.get('max_drawdown', 0):.2%}\n"
            f"Balance: {data.get('balance', 0):.2f} USD"
        )
        await self.send(msg)

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
