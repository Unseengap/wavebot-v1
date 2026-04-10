"""Model ensemble — weighted voting between Transformer and RL agent."""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger("models.ensemble")


@dataclass
class Signal:
    """Trading signal output from the ensemble."""
    action: str = "flat"           # "long", "short", or "flat"
    confidence: float = 0.0
    position_size: float = 0.0     # fraction of balance
    sl_pips: float = 0.0
    tp_pips: float = 0.0
    reason: str = ""
    transformer_probs: dict = field(default_factory=dict)
    rl_action: dict = field(default_factory=dict)


class EnsembleSignal:
    """Weighted voting between Transformer and RL agent.

    Both models must agree on direction. Confidence must exceed threshold.
    Weights are updated weekly based on live performance.
    """

    def __init__(
        self,
        transformer=None,
        rl_agent=None,
        sl_tp_module=None,
        transformer_weight: float = 0.6,
        rl_weight: float = 0.4,
        min_confidence: float = 0.65,
    ):
        self.transformer = transformer
        self.rl_agent = rl_agent
        self.sl_tp_module = sl_tp_module
        self.transformer_weight = transformer_weight
        self.rl_weight = rl_weight
        self.min_confidence = min_confidence

    def predict(self, features_df, instrument: str = "") -> dict:
        """Generate a trading signal from the model ensemble.

        Args:
            features_df: Feature DataFrame (needs at least `lookback` rows).
            instrument: Instrument being traded (for logging).

        Returns:
            Signal dict with action, confidence, position_size, sl_pips, tp_pips.
        """
        signal = Signal()

        # Get Transformer prediction
        t_dir = "flat"
        t_confidence = 0.0
        if self.transformer is not None:
            try:
                import torch
                feature_cols = [c for c in features_df.columns if c not in {
                    "time", "pair", "open", "high", "low", "close", "volume",
                    "complete", "bid_open", "bid_high", "bid_low", "bid_close",
                    "ask_open", "ask_high", "ask_low", "ask_close",
                    "spread_pips", "spread_pips_raw",
                    "label_h1", "label_h3", "label_h5",
                }]
                window = features_df[feature_cols].values[-128:].astype(np.float32)
                x = torch.tensor(window, dtype=torch.float32).unsqueeze(0)

                device = next(self.transformer.parameters()).device
                x = x.to(device)

                probs = self.transformer.predict_probs(x, horizon=2)
                signal.transformer_probs = probs

                max_class = max(probs, key=probs.get)
                t_confidence = probs[max_class]
                t_dir = max_class

            except Exception as e:
                logger.error(f"Transformer prediction error: {e}")

        # Get RL agent prediction
        r_dir = "flat"
        r_confidence = 0.0
        r_position_size = 0.0
        r_sl = 0.0
        r_tp = 0.0

        if self.rl_agent is not None:
            try:
                obs = features_df.iloc[-1:].values.astype(np.float32).flatten()
                action, _ = self.rl_agent.predict(obs, deterministic=True)

                entry_dec = float(action[0])
                if entry_dec > 0.3:
                    r_dir = "long"
                    r_confidence = min(abs(entry_dec), 1.0)
                elif entry_dec < -0.3:
                    r_dir = "short"
                    r_confidence = min(abs(entry_dec), 1.0)

                r_position_size = (float(action[1]) + 1) / 2 * 0.02
                r_sl = (float(action[2]) + 1) / 2
                r_tp = (float(action[3]) + 1) / 2

                signal.rl_action = {
                    "direction": r_dir,
                    "confidence": r_confidence,
                    "position_size": r_position_size,
                    "raw_action": action.tolist() if hasattr(action, "tolist") else list(action),
                }
            except Exception as e:
                logger.error(f"RL agent prediction error: {e}")

        # Ensemble voting
        if t_dir != r_dir or t_dir == "flat":
            signal.action = "flat"
            signal.reason = "models_disagree" if t_dir != r_dir else "both_flat"
            return signal.__dict__

        # Weighted confidence
        confidence = (
            t_confidence * self.transformer_weight +
            r_confidence * self.rl_weight
        )

        if confidence < self.min_confidence:
            signal.action = "flat"
            signal.confidence = confidence
            signal.reason = "low_confidence"
            return signal.__dict__

        signal.action = t_dir
        signal.confidence = confidence
        signal.position_size = r_position_size
        signal.sl_pips = r_sl
        signal.tp_pips = r_tp
        signal.reason = "ensemble_agree"

        # Apply SL/TP module override if available
        if self.sl_tp_module is not None:
            try:
                entry_price = float(features_df["close"].iloc[-1])
                atr = float(features_df["atr_14"].iloc[-1]) if "atr_14" in features_df.columns else 0.001
                vol_regime = float(features_df["vol_regime"].iloc[-1]) if "vol_regime" in features_df.columns else 1.0

                sl_tp = self.sl_tp_module.compute(
                    direction=t_dir,
                    entry_price=entry_price,
                    atr=atr,
                    vol_regime=vol_regime,
                )
                signal.sl_pips = sl_tp["sl_pips"]
                signal.tp_pips = sl_tp["tp_pips"]
            except Exception as e:
                logger.error(f"SL/TP module error: {e}")

        logger.info(
            f"Ensemble signal: {signal.action} (conf={signal.confidence:.2f})",
            extra={"event": "ensemble_signal", "data": {
                "instrument": instrument,
                "action": signal.action,
                "confidence": signal.confidence,
            }},
        )

        return signal.__dict__

    def update_weights(self, transformer_performance: float, rl_performance: float):
        """Update ensemble weights based on recent live performance."""
        total = transformer_performance + rl_performance
        if total > 0:
            self.transformer_weight = transformer_performance / total
            self.rl_weight = rl_performance / total
        logger.info(
            f"Updated ensemble weights: Transformer={self.transformer_weight:.2f}, RL={self.rl_weight:.2f}"
        )
