"""PPO reinforcement learning agent and custom Gym environment for forex trading."""

import logging
from typing import Optional

import numpy as np
import pandas as pd

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

logger = logging.getLogger("models.rl_agent")


class OandaForexEnv(gym.Env):
    """Custom Gym environment that simulates OANDA forex trading.

    State:  ~110 dimensions (market features + Transformer output + account state)
    Action: 5-dim continuous (entry decision, position size, SL dist, TP dist, close now)
    Reward: Risk-adjusted return with penalties for drawdown, overtrading, missing SL.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        features_df: pd.DataFrame,
        feature_cols: list[str],
        transformer=None,
        spread_pips: float = 1.2,
        pip_value: float = 10.0,
        max_leverage: int = 50,
        initial_balance: float = 10_000,
        lookback: int = 128,
        max_open_trades: int = 5,
    ):
        super().__init__()

        self.features = features_df[feature_cols].values.astype(np.float32)
        self.close_prices = features_df["close"].values if "close" in features_df.columns else None
        self.atr_values = features_df["atr_14"].values if "atr_14" in features_df.columns else None
        self.n_bars = len(self.features)
        self.n_features = len(feature_cols)
        self.transformer = transformer
        self.spread_pips = spread_pips
        self.pip_value = pip_value
        self.max_leverage = max_leverage
        self.initial_balance = initial_balance
        self.lookback = lookback
        self.max_open_trades = max_open_trades

        # State: market features + transformer probs (3) + account state (5)
        state_dim = self.n_features + 3 + 5
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )

        # Action: [entry_decision, position_size, sl_distance, tp_distance, close_now]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(5,), dtype=np.float32
        )

        self._reset_state()

    def _reset_state(self):
        """Initialize/reset account tracking state."""
        self.balance = self.initial_balance
        self.peak_balance = self.initial_balance
        self.current_step = self.lookback
        self.open_trades: list[dict] = []
        self.closed_trades: list[dict] = []
        self.total_trades = 0
        self.daily_pnl = 0.0

    def reset(self, seed=None, options=None):
        """Reset environment to a random starting point in the data."""
        super().reset(seed=seed)
        self._reset_state()

        if self.n_bars > self.lookback + 100:
            max_start = self.n_bars - self.lookback - 100
            self.current_step = self.np_random.integers(self.lookback, max_start)

        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        """Build observation vector: market features + transformer probs + account state."""
        # Current bar features
        if self.current_step >= self.n_bars:
            market_features = np.zeros(self.n_features, dtype=np.float32)
        else:
            market_features = self.features[self.current_step]

        # Transformer output probabilities (if available)
        transformer_probs = np.array([0.33, 0.33, 0.34], dtype=np.float32)
        if self.transformer is not None and self.current_step >= self.lookback:
            try:
                import torch
                window = self.features[self.current_step - self.lookback:self.current_step]
                x = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
                if next(self.transformer.parameters()).is_cuda:
                    x = x.cuda()
                with torch.no_grad():
                    logits = self.transformer(x)
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                transformer_probs = probs.astype(np.float32)
            except Exception:
                pass

        # Account state
        unrealized_pnl = sum(self._trade_pnl(t) for t in self.open_trades)
        account_state = np.array([
            unrealized_pnl / max(self.balance, 1),
            len(self.open_trades) / self.max_open_trades,
            0.0,  # margin utilization placeholder
            (self.peak_balance - self.balance) / max(self.peak_balance, 1),
            min(self.current_step - (self.closed_trades[-1]["close_step"] if self.closed_trades else 0), 100) / 100,
        ], dtype=np.float32)

        return np.concatenate([market_features, transformer_probs, account_state])

    def step(self, action: np.ndarray):
        """Execute one step in the environment."""
        entry_decision = float(action[0])
        position_size_frac = (float(action[1]) + 1) / 2  # map [-1,1] to [0,1]
        sl_frac = (float(action[2]) + 1) / 2
        tp_frac = (float(action[3]) + 1) / 2
        close_now = float(action[4]) > 0.5

        reward = 0.0
        current_price = self.close_prices[self.current_step] if self.close_prices is not None else 1.0
        current_atr = self.atr_values[self.current_step] if self.atr_values is not None else 0.001

        # Close existing trades if signaled
        if close_now and self.open_trades:
            for trade in list(self.open_trades):
                pnl = self._close_trade(trade, current_price)
                reward += self._compute_trade_reward(pnl, trade, current_atr)

        # Open new trade if signaled
        if entry_decision > 0.3 and len(self.open_trades) < self.max_open_trades:
            direction = "long"
            sl_dist = current_atr * (0.5 + sl_frac * 2.5)  # 0.5 to 3.0 ATR
            tp_dist = current_atr * (1.0 + tp_frac * 4.0)   # 1.0 to 5.0 ATR
            risk_pct = 0.005 + position_size_frac * 0.015    # 0.5% to 2%
            self._open_trade(direction, current_price, sl_dist, tp_dist, risk_pct, current_atr)

        elif entry_decision < -0.3 and len(self.open_trades) < self.max_open_trades:
            direction = "short"
            sl_dist = current_atr * (0.5 + sl_frac * 2.5)
            tp_dist = current_atr * (1.0 + tp_frac * 4.0)
            risk_pct = 0.005 + position_size_frac * 0.015
            self._open_trade(direction, current_price, sl_dist, tp_dist, risk_pct, current_atr)

        # Check existing trades for SL/TP hits
        self._check_open_trades(reward)

        # Penalties
        dd = (self.peak_balance - self.balance) / max(self.peak_balance, 1)
        reward -= 2.0 * max(0, dd - 0.02)  # drawdown penalty
        if len(self.open_trades) > self.max_open_trades:
            reward -= 0.5

        # Advance step
        self.current_step += 1
        terminated = self.current_step >= self.n_bars - 1
        truncated = self.balance <= self.initial_balance * 0.5  # margin call

        obs = self._get_obs()
        return obs, reward, terminated, truncated, {}

    def _open_trade(self, direction, price, sl_dist, tp_dist, risk_pct, atr):
        """Record opening a new trade."""
        entry_price = price + self.spread_pips * 0.0001 / 2 if direction == "long" else price - self.spread_pips * 0.0001 / 2

        trade = {
            "direction": direction,
            "entry_price": entry_price,
            "sl_price": entry_price - sl_dist if direction == "long" else entry_price + sl_dist,
            "tp_price": entry_price + tp_dist if direction == "long" else entry_price - tp_dist,
            "sl_dist": sl_dist,
            "tp_dist": tp_dist,
            "risk_pct": risk_pct,
            "atr_at_entry": atr,
            "open_step": self.current_step,
            "units": int(self.balance * risk_pct / max(sl_dist, 1e-8)),
        }
        self.open_trades.append(trade)
        self.total_trades += 1

    def _close_trade(self, trade, exit_price) -> float:
        """Close a trade and return P&L."""
        if trade["direction"] == "long":
            pnl = (exit_price - trade["entry_price"]) * trade["units"]
        else:
            pnl = (trade["entry_price"] - exit_price) * trade["units"]

        # Account for spread on exit
        pnl -= self.spread_pips * 0.0001 * trade["units"] / 2

        self.balance += pnl
        self.peak_balance = max(self.peak_balance, self.balance)

        trade["exit_price"] = exit_price
        trade["pnl"] = pnl
        trade["close_step"] = self.current_step
        trade["duration"] = self.current_step - trade["open_step"]
        self.closed_trades.append(trade)

        if trade in self.open_trades:
            self.open_trades.remove(trade)

        return pnl

    def _check_open_trades(self, reward_accumulator):
        """Check if any open trades hit SL or TP."""
        if self.close_prices is None:
            return
        if self.current_step >= self.n_bars:
            return

        # Use high/low if available for more realistic fill simulation
        for trade in list(self.open_trades):
            price = self.close_prices[self.current_step]

            if trade["direction"] == "long":
                if price <= trade["sl_price"]:
                    self._close_trade(trade, trade["sl_price"])
                elif price >= trade["tp_price"]:
                    self._close_trade(trade, trade["tp_price"])
            else:
                if price >= trade["sl_price"]:
                    self._close_trade(trade, trade["sl_price"])
                elif price <= trade["tp_price"]:
                    self._close_trade(trade, trade["tp_price"])

    def _trade_pnl(self, trade) -> float:
        """Calculate unrealized P&L for an open trade."""
        if self.close_prices is None or self.current_step >= self.n_bars:
            return 0.0
        price = self.close_prices[self.current_step]
        if trade["direction"] == "long":
            return (price - trade["entry_price"]) * trade["units"]
        else:
            return (trade["entry_price"] - price) * trade["units"]

    def _compute_trade_reward(self, pnl, trade, current_atr) -> float:
        """Compute reward for a completed trade."""
        if trade.get("atr_at_entry", 0) == 0:
            return 0.0

        # Risk-adjusted return
        risk_units = trade.get("sl_dist", current_atr) / trade["atr_at_entry"]
        reward = (pnl / max(self.balance, 1)) / max(risk_units, 0.01)

        # Bonus for good RR
        if pnl > 0 and trade.get("tp_dist", 0) > 0 and trade.get("sl_dist", 0) > 0:
            rr = trade["tp_dist"] / trade["sl_dist"]
            if rr >= 1.5:
                reward += 0.5

        # Penalty for no SL
        if trade.get("sl_dist", 0) == 0:
            reward -= 10.0

        # Penalty for holding losers too long
        avg_win_dur = 20  # baseline
        if pnl < 0 and trade.get("duration", 0) > avg_win_dur * 3:
            reward -= 0.3

        return reward
