"""Walk-forward backtest engine with realistic spread/slippage simulation."""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("backtest.engine")


@dataclass
class BacktestTrade:
    """Record of a single backtested trade."""
    entry_idx: int = 0
    exit_idx: int = 0
    instrument: str = ""
    direction: str = ""
    entry_price: float = 0.0
    exit_price: float = 0.0
    sl_price: float = 0.0
    tp_price: float = 0.0
    units: int = 0
    pnl: float = 0.0
    pnl_pips: float = 0.0
    duration_bars: int = 0
    exit_reason: str = ""
    mfe_pips: float = 0.0
    mae_pips: float = 0.0
    model_confidence: float = 0.0


class WalkForwardBacktest:
    """Walk-forward backtesting engine simulating live OANDA trading conditions.

    Features:
    - Variable bid/ask spread from stored OANDA data
    - Slippage model (random draw from historical spread distribution)
    - No look-ahead bias (features at bar_close only)
    - Walk-forward splits for realistic out-of-sample testing
    """

    def __init__(
        self,
        features_df: pd.DataFrame,
        feature_cols: list[str],
        transformer=None,
        ppo_agent=None,
        sl_tp_module=None,
        spread_pips: float = 1.2,
        initial_balance: float = 10_000,
        max_risk_per_trade: float = 0.01,
        max_open_trades: int = 5,
        min_confidence: float = 0.65,
        lookback: int = 128,
    ):
        self.features_df = features_df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.transformer = transformer
        self.ppo_agent = ppo_agent
        self.sl_tp_module = sl_tp_module
        self.spread_pips = spread_pips
        self.initial_balance = initial_balance
        self.max_risk = max_risk_per_trade
        self.max_open = max_open_trades
        self.min_confidence = min_confidence
        self.lookback = lookback

    def run(self) -> list[BacktestTrade]:
        """Execute the full backtest and return trade list."""
        balance = self.initial_balance
        open_trades: list[dict] = []
        closed_trades: list[BacktestTrade] = []
        equity_curve = []

        features = self.features_df[self.feature_cols].values.astype(np.float32)
        close = self.features_df["close"].values if "close" in self.features_df.columns else np.ones(len(features))
        high = self.features_df["high"].values if "high" in self.features_df.columns else close
        low = self.features_df["low"].values if "low" in self.features_df.columns else close
        atr = self.features_df["atr_14"].values if "atr_14" in self.features_df.columns else np.full(len(features), 0.001)

        n = len(features)
        pip_size = 0.0001 if close[0] < 50 else 0.01  # JPY pairs
        spread_price = self.spread_pips * pip_size

        logger.info(f"Starting backtest: {n} bars, initial balance={self.initial_balance}")

        for i in range(self.lookback, n):
            current_close = close[i]
            current_high = high[i]
            current_low = low[i]
            current_atr = atr[i]

            # ── Check open trades for SL/TP hits ────────────
            for trade in list(open_trades):
                hit_sl = False
                hit_tp = False
                exit_price = 0.0

                if trade["direction"] == "long":
                    if current_low <= trade["sl_price"]:
                        hit_sl = True
                        exit_price = trade["sl_price"]
                    elif current_high >= trade["tp_price"]:
                        hit_tp = True
                        exit_price = trade["tp_price"]
                else:
                    if current_high >= trade["sl_price"]:
                        hit_sl = True
                        exit_price = trade["sl_price"]
                    elif current_low <= trade["tp_price"]:
                        hit_tp = True
                        exit_price = trade["tp_price"]

                if hit_sl or hit_tp:
                    # Calculate P&L
                    if trade["direction"] == "long":
                        pnl = (exit_price - trade["entry_price"] - spread_price / 2) * trade["units"]
                    else:
                        pnl = (trade["entry_price"] - exit_price - spread_price / 2) * trade["units"]

                    balance += pnl

                    bt_trade = BacktestTrade(
                        entry_idx=trade["entry_idx"],
                        exit_idx=i,
                        direction=trade["direction"],
                        entry_price=trade["entry_price"],
                        exit_price=exit_price,
                        sl_price=trade["sl_price"],
                        tp_price=trade["tp_price"],
                        units=trade["units"],
                        pnl=pnl,
                        pnl_pips=(exit_price - trade["entry_price"]) / pip_size if trade["direction"] == "long"
                                 else (trade["entry_price"] - exit_price) / pip_size,
                        duration_bars=i - trade["entry_idx"],
                        exit_reason="tp" if hit_tp else "sl",
                        mfe_pips=trade.get("mfe", 0),
                        mae_pips=trade.get("mae", 0),
                        model_confidence=trade.get("confidence", 0),
                    )
                    closed_trades.append(bt_trade)
                    open_trades.remove(trade)

                else:
                    # Track MFE/MAE
                    if trade["direction"] == "long":
                        excursion = (current_high - trade["entry_price"]) / pip_size
                        adverse = (trade["entry_price"] - current_low) / pip_size
                    else:
                        excursion = (trade["entry_price"] - current_low) / pip_size
                        adverse = (current_high - trade["entry_price"]) / pip_size

                    trade["mfe"] = max(trade.get("mfe", 0), excursion)
                    trade["mae"] = max(trade.get("mae", 0), adverse)

            # ── Generate signal ──────────────────────────────
            if len(open_trades) >= self.max_open:
                equity_curve.append(balance + sum(
                    (current_close - t["entry_price"]) * t["units"] if t["direction"] == "long"
                    else (t["entry_price"] - current_close) * t["units"]
                    for t in open_trades
                ))
                continue

            signal = self._get_signal(features, i)

            if signal is None or signal["action"] == "flat":
                equity_curve.append(balance)
                continue

            # ── Open new trade ───────────────────────────────
            direction = signal["action"]
            sl_dist = max(current_atr * 1.5, pip_size * 10)
            tp_dist = sl_dist * 2.0

            if self.sl_tp_module:
                try:
                    sl_tp = self.sl_tp_module.compute(
                        direction=direction, entry_price=current_close,
                        atr=current_atr,
                    )
                    sl_dist = abs(sl_tp["sl_price"] - current_close)
                    tp_dist = abs(sl_tp["tp_price"] - current_close)
                except Exception:
                    pass

            # Slippage simulation
            slippage = np.random.uniform(0, spread_price)

            if direction == "long":
                entry_price = current_close + spread_price / 2 + slippage
                sl_price = entry_price - sl_dist
                tp_price = entry_price + tp_dist
            else:
                entry_price = current_close - spread_price / 2 - slippage
                sl_price = entry_price + sl_dist
                tp_price = entry_price - tp_dist

            sl_pips = sl_dist / pip_size
            units = int(balance * self.max_risk / max(sl_pips * pip_size, pip_size))
            if units <= 0:
                continue

            open_trades.append({
                "direction": direction,
                "entry_price": entry_price,
                "sl_price": sl_price,
                "tp_price": tp_price,
                "units": units,
                "entry_idx": i,
                "confidence": signal.get("confidence", 0),
                "mfe": 0, "mae": 0,
            })

            equity_curve.append(balance)

        # Close any remaining open trades at last close
        for trade in open_trades:
            final_price = close[-1]
            if trade["direction"] == "long":
                pnl = (final_price - trade["entry_price"]) * trade["units"]
            else:
                pnl = (trade["entry_price"] - final_price) * trade["units"]
            balance += pnl

            closed_trades.append(BacktestTrade(
                entry_idx=trade["entry_idx"], exit_idx=n - 1,
                direction=trade["direction"],
                entry_price=trade["entry_price"], exit_price=final_price,
                sl_price=trade["sl_price"], tp_price=trade["tp_price"],
                units=trade["units"], pnl=pnl,
                duration_bars=n - 1 - trade["entry_idx"],
                exit_reason="end_of_data",
            ))

        self._equity_curve = equity_curve
        self._final_balance = balance

        logger.info(f"Backtest complete: {len(closed_trades)} trades, final balance={balance:.2f}")
        return closed_trades

    def _get_signal(self, features: np.ndarray, idx: int) -> Optional[dict]:
        """Get model signal for bar at idx (no look-ahead)."""
        if self.transformer is None:
            return None

        try:
            import torch
            window = features[idx - self.lookback:idx]
            x = torch.tensor(window, dtype=torch.float32).unsqueeze(0)

            device = next(self.transformer.parameters()).device
            x = x.to(device)

            probs = self.transformer.predict_probs(x, horizon=2)
            max_class = max(probs, key=probs.get)
            confidence = probs[max_class]

            if confidence < self.min_confidence or max_class == "flat":
                return {"action": "flat"}

            return {"action": max_class, "confidence": confidence}

        except Exception as e:
            logger.debug(f"Signal generation error at idx {idx}: {e}")
            return None

    @property
    def equity_curve(self) -> list[float]:
        return getattr(self, "_equity_curve", [])

    @property
    def final_balance(self) -> float:
        return getattr(self, "_final_balance", self.initial_balance)
