"""Wave state machine with explicit RANGING exit criteria."""


class WaveState:
    BULLISH_IMPULSE = "BULLISH_IMPULSE"
    BEARISH_IMPULSE = "BEARISH_IMPULSE"
    BULLISH_CORRECTION = "BULLISH_CORRECTION"
    BEARISH_CORRECTION = "BEARISH_CORRECTION"
    RANGING = "RANGING"


MAX_RANGING_CANDLES = {
    "M1": 75, "M5": 50, "M15": 40, "H1": 30,
    "H4": 20, "D": 15, "W": 10,
}

# Candles with no new swings before transition to CORRECTION
CORRECTION_THRESHOLD = {
    "M1": 8, "M5": 6, "M15": 5, "H1": 5,
    "H4": 4, "D": 4, "W": 3,
}

# Max correction candles before RANGING
MAX_CORRECTION_CANDLES = {
    "M1": 30, "M5": 25, "M15": 20, "H1": 15,
    "H4": 12, "D": 10, "W": 8,
}


class WaveStateMachine:
    def __init__(self, granularity="M5"):
        self.state = WaveState.RANGING
        self.granularity = granularity
        self.max_ranging = MAX_RANGING_CANDLES.get(granularity, 50)
        self.correction_thresh = CORRECTION_THRESHOLD.get(granularity, 6)
        self.max_correction = MAX_CORRECTION_CANDLES.get(granularity, 20)

        # Swing tracking
        self.last_swing_high = None
        self.last_swing_low = None
        self.prev_swing_high = None
        self.prev_swing_low = None

        # Wave tracking
        self.wave_origin = None
        self.wave_start_idx = 0
        self.wave_peak = None  # Highest point during bullish wave
        self.wave_trough = None  # Lowest point during bearish wave
        self.candle_idx = 0

        # State counters
        self.no_swing_count = 0  # Candles since last swing
        self.ranging_count = 0
        self.correction_count = 0
        self.consecutive_swings = 0

        # RANGING exit tracking
        self.r_higher_lows = 0
        self.r_higher_highs = 0
        self.r_lower_highs = 0
        self.r_lower_lows = 0

        # Correction anchors
        self.impulse_high = None
        self.impulse_low = None

    def update(self, new_swing_high=None, new_swing_low=None,
               current_high=None, current_low=None):
        """
        Call on every candle. Pass confirmed swings (lagged by lookback).
        current_high/low are the actual candle high/low for wave extent tracking.
        """
        self.candle_idx += 1
        had_swing = new_swing_high is not None or new_swing_low is not None

        # Track wave extent
        if current_high is not None:
            if self.wave_peak is None or current_high > self.wave_peak:
                self.wave_peak = current_high
        if current_low is not None:
            if self.wave_trough is None or current_low < self.wave_trough:
                self.wave_trough = current_low

        # Dispatch to state handler
        if self.state == WaveState.BULLISH_IMPULSE:
            self._bullish_impulse(new_swing_high, new_swing_low, had_swing)
        elif self.state == WaveState.BEARISH_IMPULSE:
            self._bearish_impulse(new_swing_high, new_swing_low, had_swing)
        elif self.state == WaveState.BULLISH_CORRECTION:
            self._bullish_correction(new_swing_high, new_swing_low)
        elif self.state == WaveState.BEARISH_CORRECTION:
            self._bearish_correction(new_swing_high, new_swing_low)
        elif self.state == WaveState.RANGING:
            self._ranging(new_swing_high, new_swing_low)

        # Update swing history AFTER state handler
        if new_swing_high is not None:
            self.prev_swing_high = self.last_swing_high
            self.last_swing_high = new_swing_high
        if new_swing_low is not None:
            self.prev_swing_low = self.last_swing_low
            self.last_swing_low = new_swing_low

    # --- State Handlers ---

    def _bullish_impulse(self, sh, sl, had_swing):
        if sl is not None and self.last_swing_low is not None:
            if sl < self.last_swing_low:
                self._to(WaveState.RANGING)
                return
            else:
                self.consecutive_swings += 1
        if sh is not None and self.last_swing_high is not None:
            if sh > self.last_swing_high:
                self.consecutive_swings += 1

        if not had_swing:
            self.no_swing_count += 1
            if self.no_swing_count >= self.correction_thresh:
                self.impulse_high = self.last_swing_high
                self.impulse_low = self.last_swing_low
                self._to(WaveState.BULLISH_CORRECTION)
        else:
            self.no_swing_count = 0

    def _bearish_impulse(self, sh, sl, had_swing):
        if sh is not None and self.last_swing_high is not None:
            if sh > self.last_swing_high:
                self._to(WaveState.RANGING)
                return
            else:
                self.consecutive_swings += 1
        if sl is not None and self.last_swing_low is not None:
            if sl < self.last_swing_low:
                self.consecutive_swings += 1

        if not had_swing:
            self.no_swing_count += 1
            if self.no_swing_count >= self.correction_thresh:
                self.impulse_high = self.last_swing_high
                self.impulse_low = self.last_swing_low
                self._to(WaveState.BEARISH_CORRECTION)
        else:
            self.no_swing_count = 0

    def _bullish_correction(self, sh, sl):
        self.correction_count += 1
        if sh is not None and self.impulse_high is not None:
            if sh > self.impulse_high:
                self.wave_origin = self.last_swing_low
                self.wave_start_idx = self.candle_idx
                self.wave_peak = sh
                self.wave_trough = None
                self._to(WaveState.BULLISH_IMPULSE)
                return
        if sl is not None and self.impulse_low is not None:
            if sl < self.impulse_low:
                self.wave_origin = self.last_swing_high
                self.wave_start_idx = self.candle_idx
                self.wave_trough = sl
                self.wave_peak = None
                self._to(WaveState.BEARISH_IMPULSE)
                return
        if self.correction_count >= self.max_correction:
            self._to(WaveState.RANGING)

    def _bearish_correction(self, sh, sl):
        self.correction_count += 1
        if sl is not None and self.impulse_low is not None:
            if sl < self.impulse_low:
                self.wave_origin = self.last_swing_high
                self.wave_start_idx = self.candle_idx
                self.wave_trough = sl
                self.wave_peak = None
                self._to(WaveState.BEARISH_IMPULSE)
                return
        if sh is not None and self.impulse_high is not None:
            if sh > self.impulse_high:
                self.wave_origin = self.last_swing_low
                self.wave_start_idx = self.candle_idx
                self.wave_peak = sh
                self.wave_trough = None
                self._to(WaveState.BULLISH_IMPULSE)
                return
        if self.correction_count >= self.max_correction:
            self._to(WaveState.RANGING)

    def _ranging(self, sh, sl):
        self.ranging_count += 1

        if sl is not None and self.last_swing_low is not None:
            if sl > self.last_swing_low:
                self.r_higher_lows += 1
                self.r_lower_lows = 0
            elif sl < self.last_swing_low:
                self.r_lower_lows += 1
                self.r_higher_lows = 0

        if sh is not None and self.last_swing_high is not None:
            if sh > self.last_swing_high:
                self.r_higher_highs += 1
                self.r_lower_highs = 0
            elif sh < self.last_swing_high:
                self.r_lower_highs += 1
                self.r_higher_highs = 0

        # Bullish exit: 2 higher lows + 1 higher high
        if self.r_higher_lows >= 2 and self.r_higher_highs >= 1:
            self.wave_origin = self.last_swing_low if sl is None else sl
            if self.wave_origin is None:
                self.wave_origin = self.last_swing_low
            self.wave_start_idx = self.candle_idx
            self.wave_peak = self.last_swing_high
            self.wave_trough = None
            self._to(WaveState.BULLISH_IMPULSE)
            return

        # Bearish exit: 2 lower highs + 1 lower low
        if self.r_lower_highs >= 2 and self.r_lower_lows >= 1:
            self.wave_origin = self.last_swing_high if sh is None else sh
            if self.wave_origin is None:
                self.wave_origin = self.last_swing_high
            self.wave_start_idx = self.candle_idx
            self.wave_trough = self.last_swing_low
            self.wave_peak = None
            self._to(WaveState.BEARISH_IMPULSE)
            return

        # Forced reset
        if self.ranging_count >= self.max_ranging:
            self._reset_ranging()

    # --- Transitions ---

    def _to(self, new_state):
        self.state = new_state
        self.no_swing_count = 0
        self.correction_count = 0
        if new_state == WaveState.RANGING:
            self.ranging_count = 0
            self.r_higher_lows = 0
            self.r_higher_highs = 0
            self.r_lower_highs = 0
            self.r_lower_lows = 0
            self.consecutive_swings = 0
        elif new_state in (WaveState.BULLISH_IMPULSE, WaveState.BEARISH_IMPULSE):
            self.consecutive_swings = 1
            self.ranging_count = 0

    def _reset_ranging(self):
        """Hard reset after max ranging candles — clear swing baseline."""
        self.ranging_count = 0
        self.r_higher_lows = 0
        self.r_higher_highs = 0
        self.r_lower_highs = 0
        self.r_lower_lows = 0
        self.last_swing_high = None
        self.last_swing_low = None
        self.prev_swing_high = None
        self.prev_swing_low = None
