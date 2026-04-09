"""Structured JSON logging with daily rotation."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from logging.handlers import TimedRotatingFileHandler


class JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects."""

    def format(self, record):
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
        }

        if hasattr(record, "account"):
            log_entry["account"] = record.account
        if hasattr(record, "event"):
            log_entry["event"] = record.event
        if hasattr(record, "data"):
            log_entry["data"] = record.data
        if record.msg:
            log_entry["message"] = record.getMessage()
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, default=str)


def setup_logging(log_dir: str = "data/logs", level: str = "INFO") -> logging.Logger:
    """Configure root logger with JSON file handler and console handler."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # File handler — daily rotation, 90-day retention
    file_handler = TimedRotatingFileHandler(
        filename=log_path / "oandafx.jsonl",
        when="midnight",
        interval=1,
        backupCount=90,
        utc=True,
    )
    file_handler.setFormatter(JSONFormatter())

    # Console handler — human-readable
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level))

    # Clear existing handlers to avoid duplicates
    root_logger.handlers.clear()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    return root_logger
