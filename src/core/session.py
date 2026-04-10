"""Trading session controller — one per OANDA account."""

import logging
import time
import threading
from datetime import datetime, timezone, timedelta
from typing import Optional

import pandas as pd

logger = logging.getLogger("core.session")

# Granularity to seconds mapping
GRANULARITY_SECONDS = {
    "S5": 5, "S10": 10, "S15": 15, "S30": 30,
    "M1": 60, "M2": 120, "M4": 240, "M5": 300, "M10": 600, "M15": 900,
    "M30": 1800, "H1": 3600, "H2": 7200, "H3": 10800, "H4": 14400,
    "H6": 21600, "H8": 28800, "H12": 43200, "D": 86400, "W": 604800,
}


class TradingSession:
    """Manages the trading loop for a single OANDA account.

    Polls for new bar closes, computes features, runs model inference,
    passes signals through risk validation, and executes approved trades.
    """

    def __init__(
        self,
        alias: str,
        api_client,
        feature_engineer=None,
        model_ensemble=None,
        risk_validator=None,
        order_manager=None,
        trade_monitor=None,
        config: Optional[dict] = None,
    ):
        self.alias = alias
        self.api_client = api_client
        self.feature_engineer = feature_engineer
        self.model_ensemble = model_ensemble
        self.risk_validator = risk_validator
        self.order_manager = order_manager
        self.trade_monitor = trade_monitor
        self.config = config or {}

        self.base_tf = self.config.get("base_timeframe", "M15")
        self.higher_tfs = self.config.get("higher_timeframes", ["H1", "H4", "D"])
        self.pairs = self.config.get("pairs", ["EUR_USD"])
        self.lookback = self.config.get("lookback_bars", 128)
        self.bar_delay = self.config.get("bar_close_delay_seconds", 2)

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._feature_buffers: dict[str, pd.DataFrame] = {}

        self.last_signal_time: Optional[datetime] = None
        self.last_trade_time: Optional[datetime] = None

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self):
        """Start the trading session in a background thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, name=f"session-{self.alias}", daemon=True)
        self._thread.start()
        logger.info(f"Session '{self.alias}' started", extra={
            "event": "session_started", "account": self.alias,
        })

    def stop(self):
        """Gracefully stop the trading session."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=30)
        logger.info(f"Session '{self.alias}' stopped", extra={
            "event": "session_stopped", "account": self.alias,
        })

    def _run_loop(self):
        """Main polling loop — fires on each bar close."""
        bar_seconds = GRANULARITY_SECONDS.get(self.base_tf, 900)

        while self._running:
            try:
                now = datetime.now(timezone.utc)
                seconds_into_bar = now.timestamp() % bar_seconds
                sleep_until_close = bar_seconds - seconds_into_bar + self.bar_delay

                if sleep_until_close > bar_seconds:
                    sleep_until_close -= bar_seconds

                time.sleep(max(sleep_until_close, 1))

                if not self._running:
                    break

                self._on_bar_close()

            except Exception as e:
                logger.error(
                    f"Session '{self.alias}' error: {e}",
                    extra={"event": "session_error", "account": self.alias},
                )
                time.sleep(5)

    def _on_bar_close(self):
        """Process a new bar close: fetch data, compute features, generate signal."""
        for instrument in self.pairs:
            try:
                self._process_instrument(instrument)
            except Exception as e:
                logger.error(
                    f"Error processing {instrument}: {e}",
                    extra={"event": "instrument_error", "account": self.alias,
                           "data": {"instrument": instrument}},
                )

    def _process_instrument(self, instrument: str):
        """Full pipeline for one instrument on bar close."""
        # 1. Fetch latest candles for all timeframes
        raw_data = {}
        for tf in [self.base_tf] + self.higher_tfs:
            candles = self.api_client.get_candles(
                instrument=instrument,
                granularity=tf,
                count=self.lookback + 50,  # extra bars for indicator warmup
                price="BA",
            )
            if candles.empty:
                logger.warning(f"No candle data for {instrument}/{tf}")
                return
            raw_data[tf] = candles[candles["complete"] == True]

        # 2. Compute features
        if self.feature_engineer is None:
            return

        features = self.feature_engineer.build(raw_data)
        if features.empty or len(features) < self.lookback:
            return

        # 3. Model inference
        if self.model_ensemble is None:
            return

        signal = self.model_ensemble.predict(features, instrument)
        self.last_signal_time = datetime.now(timezone.utc)

        logger.info(
            f"Signal for {instrument}: {signal}",
            extra={"event": "signal_generated", "account": self.alias,
                   "data": {"instrument": instrument, "signal": str(signal)}},
        )

        if signal.get("action") == "flat":
            logger.debug(f"Flat signal for {instrument}, no trade",
                         extra={"event": "signal_flat"})
            return

        # 4. Risk validation
        if self.risk_validator is None:
            return

        account_summary = self.api_client.get_account_summary()
        open_trades = self.api_client.get_open_trades()

        approved, reason = self.risk_validator.validate(
            signal=signal,
            account_summary=account_summary,
            open_trades=open_trades,
            instrument=instrument,
        )

        if not approved:
            logger.warning(
                f"Trade rejected for {instrument}: {reason}",
                extra={"event": "risk_rejected", "account": self.alias,
                       "data": {"instrument": instrument, "reason": reason}},
            )
            return

        # 5. Execute order
        if self.order_manager is None:
            return

        result = self.order_manager.execute_signal(
            signal=signal,
            instrument=instrument,
            account_summary=account_summary,
        )

        if result:
            self.last_trade_time = datetime.now(timezone.utc)
            logger.info(
                f"Trade opened for {instrument}",
                extra={"event": "trade_opened", "account": self.alias,
                       "data": result},
            )

    def get_status(self) -> dict:
        """Return current session status."""
        return {
            "alias": self.alias,
            "running": self._running,
            "pairs": self.pairs,
            "base_timeframe": self.base_tf,
            "last_signal_time": self.last_signal_time.isoformat() if self.last_signal_time else None,
            "last_trade_time": self.last_trade_time.isoformat() if self.last_trade_time else None,
        }
