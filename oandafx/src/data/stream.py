"""Live price streaming from OANDA with auto-reconnect and pub/sub distribution."""

import logging
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger("data.stream")


class PriceStream:
    """Maintains a persistent connection to OANDA's pricing stream.

    Distributes incoming ticks to registered listeners via callbacks.
    Auto-reconnects on disconnection with exponential backoff.
    """

    def __init__(self, api_client, instruments: list[str]):
        self.api_client = api_client
        self.instruments = instruments
        self._listeners: list[Callable] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._reconnect_delay = 1

    def add_listener(self, callback: Callable):
        """Register a callback to receive tick data."""
        self._listeners.append(callback)

    def remove_listener(self, callback: Callable):
        """Unregister a tick listener."""
        self._listeners = [l for l in self._listeners if l is not callback]

    def start(self):
        """Start the price stream in a background thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._stream_loop, daemon=True)
        self._thread.start()
        logger.info(f"Price stream started for {self.instruments}")

    def stop(self):
        """Stop the price stream."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10)
        logger.info("Price stream stopped")

    def _stream_loop(self):
        """Main loop with auto-reconnect."""
        while self._running:
            try:
                for tick in self.api_client.stream_prices(self.instruments):
                    if not self._running:
                        break
                    self._distribute_tick(tick)
                    self._reconnect_delay = 1  # reset on successful data
            except Exception as e:
                if not self._running:
                    break
                logger.warning(
                    f"Stream disconnected: {e}. Reconnecting in {self._reconnect_delay}s",
                    extra={"event": "stream_reconnect"},
                )
                time.sleep(self._reconnect_delay)
                self._reconnect_delay = min(self._reconnect_delay * 2, 60)

    def _distribute_tick(self, tick: dict):
        """Send tick data to all registered listeners."""
        for listener in self._listeners:
            try:
                listener(tick)
            except Exception as e:
                logger.error(f"Listener error: {e}")

    @property
    def is_running(self) -> bool:
        return self._running
