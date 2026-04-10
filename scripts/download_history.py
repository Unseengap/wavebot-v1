#!/usr/bin/env python3
"""CLI for downloading historical OANDA candle data.

Usage:
    python scripts/download_history.py \
        --pairs EUR_USD GBP_USD USD_JPY \
        --granularities M15 H1 H4 D \
        --start 2010-01-01 \
        --output-dir data/raw \
        --mode full \
        --verbose
"""

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.api_client import APIClient
from src.data.downloader import download_all_pairs
from src.monitoring.logger import setup_logging

DEFAULT_PAIRS = [
    "EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD",
    "USD_CHF", "USD_CAD", "NZD_USD",
]

DEFAULT_GRANULARITIES = ["M15", "H1", "H4", "D"]


def main():
    parser = argparse.ArgumentParser(description="Download historical OANDA candle data")
    parser.add_argument("--pairs", nargs="+", default=DEFAULT_PAIRS, help="Currency pairs to download")
    parser.add_argument("--granularities", nargs="+", default=DEFAULT_GRANULARITIES, help="Timeframes")
    parser.add_argument("--start", default="2010-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD), defaults to now")
    parser.add_argument("--output-dir", default="data/raw", help="Output directory for Parquet files")
    parser.add_argument("--mode", choices=["full", "incremental"], default="full", help="Download mode")
    parser.add_argument("--api-key", default=None, help="OANDA API key (or set OANDA_API_KEY_1 env var)")
    parser.add_argument("--account-id", default=None, help="OANDA account ID (or set OANDA_ACCOUNT_ID_1)")
    parser.add_argument("--environment", default="practice", choices=["live", "practice"])
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument("--all-pairs", action="store_true", help="Download all major and minor pairs")

    args = parser.parse_args()

    # Setup logging
    level = "DEBUG" if args.verbose else "INFO"
    setup_logging(level=level)

    # Get credentials
    api_key = args.api_key or os.environ.get("OANDA_API_KEY_1", "")
    account_id = args.account_id or os.environ.get("OANDA_ACCOUNT_ID_1", "")

    if not api_key or not account_id:
        print("ERROR: OANDA API key and account ID required.")
        print("  Set OANDA_API_KEY_1 and OANDA_ACCOUNT_ID_1 environment variables")
        print("  or pass --api-key and --account-id flags.")
        sys.exit(1)

    # Parse dates
    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = None
    if args.end:
        end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    # Expanded pair list
    if args.all_pairs:
        pairs = [
            "EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CHF", "USD_CAD", "NZD_USD",
            "EUR_GBP", "EUR_JPY", "EUR_CHF", "EUR_CAD", "EUR_AUD",
            "GBP_JPY", "GBP_CHF", "GBP_CAD",
            "AUD_JPY", "CHF_JPY", "CAD_JPY", "NZD_JPY",
        ]
    else:
        pairs = args.pairs

    # Initialize client
    client = APIClient(api_key=api_key, account_id=account_id, environment=args.environment)

    print(f"Downloading {len(pairs)} pairs × {len(args.granularities)} timeframes")
    print(f"Mode: {args.mode} | Start: {args.start} | Output: {args.output_dir}")
    print("-" * 60)

    # Download
    results = download_all_pairs(
        client=client,
        pairs=pairs,
        granularities=args.granularities,
        start=start,
        end=end,
        output_dir=args.output_dir,
        mode=args.mode,
    )

    # Summary
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    total_rows = 0
    for key, df in results.items():
        rows = len(df)
        total_rows += rows
        status = f"{rows:>10,} bars" if rows > 0 else "  FAILED"
        print(f"  {key:<20} {status}")

    print(f"\nTotal: {total_rows:,} candles saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
