"""
WaveBot Backtest Engine — processes candles chronologically with zero lookahead.
Primary entry on M5 candles. Multi-timeframe wave state sync.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.wave.swing_detector import detect_swing_high, detect_swing_low
from src.wave.wave_state_machine import WaveStateMachine, WaveState
from src.wave.wave_scorer import score_wave, WaveScore
from src.wave.amplitude_tracker import AmplitudeTracker
from src.confluence.alignment_engine import (
    get_directional_gate, calculate_confluence_score,
    get_signal_frame, get_session, get_session_size_multiplier,
)
from src.confluence.entry_filter import check_entry_conditions
from src.risk.position_sizer import calculate_position_size, apply_session_to_position_size
from src.risk.sl_tp_engine import (
    calculate_stop_loss, calculate_take_profit,
    calculate_rr, validate_sl_distance, calculate_atr,
)
from src.risk.circuit_breaker import CircuitBreaker
from src.backtest.spread_simulator import SpreadSimulator
from src.backtest.slippage_simulator import SlippageSimulator


# Pip size per instrument
PIP_SIZES = {
    "EUR_USD": 0.0001, "GBP_USD": 0.0001, "USD_CHF": 0.0001,
    "AUD_USD": 0.0001, "NZD_USD": 0.0001, "USD_CAD": 0.0001,
    "EUR_GBP": 0.0001, "EUR_JPY": 0.01, "GBP_JPY": 0.01,
    "USD_JPY": 0.01, "AUD_JPY": 0.01,
    "XAU_USD": 0.01, "XAG_USD": 0.001,
}

# Max spread per instrument
MAX_SPREAD = {
    "EUR_USD": 2.0, "GBP_USD": 2.5, "USD_JPY": 2.0, "USD_CAD": 2.5,
    "AUD_USD": 2.0, "USD_CHF": 2.5, "NZD_USD": 3.0, "EUR_GBP": 3.0,
    "GBP_JPY": 4.0, "EUR_JPY": 3.0, "XAU_USD": 40.0,
}

SWING_LOOKBACK = {
    "M1": 3, "M5": 3, "M15": 3, "H1": 3, "H4": 3, "D": 3, "W": 2,
}


class TimeframeState:
    """Holds pre-processed wave states for a single timeframe."""
    def __init__(self, granularity: str, candles: pd.DataFrame,
                 instrument: str, pip_size: float, amplitude_tracker):
        self.granularity = granularity
        self.candles = candles
        self.instrument = instrument
        self.pip_size = pip_size
        self.amplitude_tracker = amplitude_tracker

        lookback = SWING_LOOKBACK.get(granularity, 3)
        highs = candles["high_mid"].values
        lows = candles["low_mid"].values
        closes = candles["close_mid"].values
        times = candles["time"].values

        # Detect swings on full array
        self.swing_highs = detect_swing_high(highs, lookback)
        self.swing_lows = detect_swing_low(lows, lookback)
        self.lookback = lookback
        self.highs = highs
        self.lows = lows
        self.closes = closes
        self.times = times

        # Wave state machine
        self.machine = WaveStateMachine(granularity=granularity)

        # Pre-process: run state machine candle by candle
        self.scores = []   # List of WaveScore, one per candle
        self.score_times = []

        prev_state = None
        prev_wave_origin = None

        for i in range(len(candles)):
            # Swing confirmed at i-lookback becomes known at candle i
            check_idx = i - lookback
            new_sh = None
            new_sl = None
            if check_idx >= 0:
                if self.swing_highs[check_idx]:
                    new_sh = highs[check_idx]
                if self.swing_lows[check_idx]:
                    new_sl = lows[check_idx]

            self.machine.update(
                new_swing_high=new_sh,
                new_swing_low=new_sl,
                current_high=highs[i],
                current_low=lows[i],
            )

            # Detect wave completion for amplitude tracking
            cur_state = self.machine.state
            if prev_state in (WaveState.BULLISH_IMPULSE, WaveState.BEARISH_IMPULSE):
                if cur_state != prev_state and prev_wave_origin:
                    if prev_state == WaveState.BULLISH_IMPULSE and self.machine.wave_peak:
                        amp = abs(self.machine.wave_peak - prev_wave_origin) / pip_size
                        dur = i - self.machine.wave_start_idx
                        amplitude_tracker.record_wave(instrument, granularity, amp, max(1, dur))
                    elif prev_state == WaveState.BEARISH_IMPULSE and self.machine.wave_trough:
                        amp = abs(prev_wave_origin - self.machine.wave_trough) / pip_size
                        dur = i - self.machine.wave_start_idx
                        amplitude_tracker.record_wave(instrument, granularity, amp, max(1, dur))

            prev_state = cur_state
            prev_wave_origin = self.machine.wave_origin

            # Score
            amp_stats = amplitude_tracker.get_amplitude_stats(instrument, granularity)
            dur_stats = amplitude_tracker.get_duration_stats(instrument, granularity)
            ws = score_wave(instrument, granularity, str(times[i]),
                           self.machine, amp_stats, dur_stats, pip_size)
            self.scores.append(ws)
            self.score_times.append(times[i])

        self.score_times_arr = np.array(self.score_times)

    def get_score_at(self, timestamp) -> WaveScore:
        """Get the latest wave score at or before the given timestamp."""
        idx = np.searchsorted(self.score_times_arr, timestamp, side="right") - 1
        if idx < 0:
            return None
        return self.scores[idx]


class BacktestEngine:
    def __init__(self, config: dict):
        self.config = config
        self.instrument = config["instrument"]
        self.pip_size = PIP_SIZES.get(self.instrument, 0.0001)
        self.pip_value = self.pip_size  # For USD-quoted pairs
        self.max_spread = MAX_SPREAD.get(self.instrument, 2.0)

        self.initial_balance = config.get("initial_balance", 10000.0)
        self.balance = self.initial_balance
        self.peak_balance = self.balance

        self.spread_sim = SpreadSimulator()
        self.slip_sim = SlippageSimulator()
        self.circuit = CircuitBreaker(
            max_daily_drawdown=config.get("max_daily_drawdown", 0.02),
            max_total_drawdown=config.get("max_total_drawdown", 0.08),
            max_open_trades=config.get("max_open_trades", 3),
        )
        self.amplitude_tracker = AmplitudeTracker()

        self.open_trades = []
        self.closed_trades = []
        self.trade_counter = 0
        self.current_day = None

    def run(self, candle_data: dict) -> list:
        """
        candle_data: dict of granularity -> DataFrame
        Must include at least M5 + one or more of M15, H1, H4, D.
        Returns list of closed trade dicts.
        """
        print(f"Pre-processing timeframes...")
        tf_states = {}
        for tf, df in candle_data.items():
            print(f"  {tf}: {len(df)} candles")
            tf_states[tf] = TimeframeState(
                tf, df, self.instrument, self.pip_size, self.amplitude_tracker
            )

        # Primary loop on M5 candles
        m5 = candle_data["M5"]
        m5_state = tf_states["M5"]
        timeframes = list(candle_data.keys())

        total = len(m5)
        print(f"\nRunning backtest: {total} M5 candles...")

        for i in range(len(m5)):
            row = m5.iloc[i]
            ts = row["time"]
            ts_np = m5_state.times[i]

            # Daily reset
            day = ts.date() if hasattr(ts, 'date') else pd.Timestamp(ts).date()
            if day != self.current_day:
                self.current_day = day
                self.circuit.reset_daily(self.balance)

            # Get current candle data
            candle = {
                "high_mid": row["high_mid"],
                "low_mid": row["low_mid"],
                "close_mid": row["close_mid"],
                "open_mid": row["open_mid"],
            }
            if "close_ask" in row and "close_bid" in row:
                candle["close_ask"] = row.get("close_ask")
                candle["close_bid"] = row.get("close_bid")

            # 1. Update open trades (check SL/TP)
            self._update_trades(candle, ts)

            # 2. Get wave scores from all timeframes
            wave_scores = {}
            for tf in timeframes:
                wave_scores[tf] = tf_states[tf].get_score_at(ts_np)

            # 3. Check for entry
            self._check_entry(wave_scores, candle, ts, m5_state, i)

            # Progress
            if (i + 1) % 50000 == 0:
                pct = (i + 1) / total * 100
                print(f"  {pct:.0f}% complete | Trades: {len(self.closed_trades)} closed, "
                      f"{len(self.open_trades)} open | Balance: ${self.balance:,.2f}")

        # Close any remaining open trades at last price
        if self.open_trades:
            last_close = m5.iloc[-1]["close_mid"]
            for t in list(self.open_trades):
                self._close_trade(t, last_close, str(m5.iloc[-1]["time"]), "END_OF_TEST")
            self.open_trades.clear()

        print(f"\nBacktest complete: {len(self.closed_trades)} trades")
        return self.closed_trades

    def _check_entry(self, wave_scores, candle, timestamp, m5_state, candle_idx):
        # Circuit breaker
        allowed, reason = self.circuit.check(self.balance, len(self.open_trades))
        if not allowed:
            return

        # Stage 1: Directional gate
        gate = get_directional_gate(wave_scores)
        if gate == "NEUTRAL":
            return

        # Stage 2: Confluence score
        score = calculate_confluence_score(wave_scores)

        # Determine direction
        direction = "LONG" if score > 0 else "SHORT" if score < 0 else None
        if direction is None:
            return

        # Enforce gate alignment
        if direction == "LONG" and gate == "BEARISH" and abs(score) < 0.85:
            return
        if direction == "SHORT" and gate == "BULLISH" and abs(score) < 0.85:
            return

        # Signal frame
        sig = get_signal_frame(wave_scores, direction)
        if sig is None:
            return

        # Session
        hour = pd.Timestamp(timestamp).hour if not isinstance(timestamp, datetime) else timestamp.hour
        session = get_session(hour)
        session_mult = get_session_size_multiplier(session)
        if session_mult <= 0:
            return

        # Spread check
        spread = self.spread_sim.get_spread_pips(candle, self.instrument,
                                                  session, self.pip_size)

        # Entry price with spread + slippage
        entry_mid = candle["close_mid"]
        entry_with_spread = self.spread_sim.apply_to_entry(
            direction, entry_mid, spread, self.pip_size)

        # ATR for slippage and SL validation
        start_idx = max(0, candle_idx - 14)
        atr = calculate_atr(
            m5_state.highs[start_idx:candle_idx + 1],
            m5_state.lows[start_idx:candle_idx + 1],
            m5_state.closes[start_idx:candle_idx + 1],
        )
        atr_pips = atr / self.pip_size

        candle_range = (candle["high_mid"] - candle["low_mid"]) / self.pip_size
        slip = self.slip_sim.get_slippage_pips(self.instrument, candle_range, atr_pips)
        entry_price = self.slip_sim.apply_to_fill(
            direction, "ENTRY", entry_with_spread, slip, self.pip_size)

        # SL from wave origin
        sl = calculate_stop_loss(direction, sig.wave_origin, self.pip_size,
                                  self.config.get("sl_buffer_pips", 2.0))
        sl_dist = abs(entry_price - sl) / self.pip_size

        # Validate SL distance
        if sl_dist < 3:
            return  # Too tight
        if not validate_sl_distance(sl_dist, atr_pips):
            return

        # TP from amplitude projection
        confirm_tf = "H4" if "H4" in wave_scores else "H1"
        amp_stats = self.amplitude_tracker.get_amplitude_stats(
            self.instrument, confirm_tf)
        tp = calculate_take_profit(direction, entry_price, amp_stats,
                                    self.pip_size,
                                    self.config.get("tp_percentile", "p75"))

        # R:R check
        rr = calculate_rr(entry_price, sl, tp, direction)

        # Entry conditions
        daily_dd = self.circuit.get_daily_drawdown(self.balance)
        can_enter, fails = check_entry_conditions(
            self.instrument, direction, score, wave_scores, gate,
            self.open_trades, spread, self.max_spread, sig, rr,
            daily_dd, session, self.config,
        )

        if not can_enter:
            return

        # Position size
        base_units = calculate_position_size(
            self.balance, self.config.get("risk_fraction", 0.01),
            sl_dist, self.pip_value,
        )
        units = apply_session_to_position_size(base_units, session_mult)
        if units <= 0:
            return

        # Open trade
        self.trade_counter += 1
        trade = {
            "id": self.trade_counter,
            "instrument": self.instrument,
            "direction": direction,
            "entry_price": entry_price,
            "entry_time": str(timestamp),
            "units": units,
            "stop_loss": sl,
            "take_profit": tp,
            "signal_frame": sig.granularity,
            "confluence_score": abs(score),
            "session": session,
            "sl_distance_pips": sl_dist,
            "rr_target": rr,
            "entry_candle_idx": candle_idx,
            # Tracking
            "max_favorable_pips": 0.0,
            "max_adverse_pips": 0.0,
        }
        self.open_trades.append(trade)

    def _update_trades(self, candle, timestamp):
        high = candle["high_mid"]
        low = candle["low_mid"]

        for trade in list(self.open_trades):
            d = trade["direction"]
            sl = trade["stop_loss"]
            tp = trade["take_profit"]
            entry = trade["entry_price"]

            # Track MFE/MAE
            if d == "LONG":
                favorable = (high - entry) / self.pip_size
                adverse = (entry - low) / self.pip_size
            else:
                favorable = (entry - low) / self.pip_size
                adverse = (high - entry) / self.pip_size

            trade["max_favorable_pips"] = max(trade["max_favorable_pips"], favorable)
            trade["max_adverse_pips"] = max(trade["max_adverse_pips"], adverse)

            # Check SL and TP
            sl_hit = False
            tp_hit = False

            if d == "LONG":
                if low <= sl:
                    sl_hit = True
                if high >= tp:
                    tp_hit = True
            else:
                if high >= sl:
                    sl_hit = True
                if low <= tp:
                    tp_hit = True

            # If both hit same candle, assume SL first (conservative)
            if sl_hit:
                self._close_trade(trade, sl, str(timestamp), "SL")
                self.open_trades.remove(trade)
            elif tp_hit:
                self._close_trade(trade, tp, str(timestamp), "TP")
                self.open_trades.remove(trade)

    def _close_trade(self, trade, exit_price, exit_time, reason):
        d = trade["direction"]

        # Apply spread + slippage to exit
        session = get_session(pd.Timestamp(exit_time).hour)
        candle_dict = {}  # Simplified — use model spread for exit
        spread = self.spread_sim.get_spread_pips(candle_dict, self.instrument,
                                                  session, self.pip_size)
        exit_after_spread = self.spread_sim.apply_to_exit(
            d, exit_price, spread * 0.5, self.pip_size)  # Half spread on exit
        exit_final = self.slip_sim.apply_to_fill(
            d, "EXIT", exit_after_spread, 0.2, self.pip_size)  # Minimal exit slippage

        if d == "LONG":
            pnl_pips = (exit_final - trade["entry_price"]) / self.pip_size
        else:
            pnl_pips = (trade["entry_price"] - exit_final) / self.pip_size

        pnl_dollars = pnl_pips * self.pip_value * trade["units"]

        trade["exit_price"] = exit_final
        trade["exit_time"] = exit_time
        trade["exit_reason"] = reason
        trade["pnl_pips"] = round(pnl_pips, 1)
        trade["pnl_dollars"] = round(pnl_dollars, 2)

        # R:R achieved (for winners)
        if pnl_pips > 0:
            trade["rr_achieved"] = round(pnl_pips / trade["sl_distance_pips"], 2)
        else:
            trade["rr_achieved"] = round(pnl_pips / trade["sl_distance_pips"], 2)

        # Bars in trade
        trade["bars_in_trade"] = 0  # Simplified

        self.balance += pnl_dollars
        self.peak_balance = max(self.peak_balance, self.balance)
        self.closed_trades.append(trade)
