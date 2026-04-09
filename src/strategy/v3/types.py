"""V3 data types for the Swing Reversal strategy."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CandleSnapshot:
    open: float
    high: float
    low: float
    close: float
    time: str
    timeframe: str

    @property
    def is_bullish(self) -> bool:
        return self.close > self.open

    @property
    def is_bearish(self) -> bool:
        return self.close < self.open

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def body_ratio(self) -> float:
        return self.body / self.range if self.range > 0 else 0.0


@dataclass
class ReversalPattern:
    direction: str  # "BULLISH" or "BEARISH"
    candle_1: CandleSnapshot
    candle_2: CandleSnapshot
    candle_3: CandleSnapshot
    pattern_low: float
    pattern_high: float
    confirmation_type: str  # "CLEAN" or "REJECTION"
    detected_time: str

    @property
    def entry_price(self) -> float:
        return self.candle_3.close

    @property
    def initial_sl(self) -> float:
        if self.direction == "BULLISH":
            return self.pattern_low
        return self.pattern_high


@dataclass
class V3Trade:
    id: int
    instrument: str
    direction: str  # "LONG" or "SHORT"
    entry_price: float
    entry_time: str
    units: int
    stop_loss: float
    take_profit: Optional[float]  # Always None for V3
    signal_frame: str
    confluence_score: float
    session: str
    sl_distance_pips: float
    rr_target: float
    entry_candle_idx: int
    max_favorable_pips: float = 0.0
    max_adverse_pips: float = 0.0
    bars_in_trade: int = 0
    # V3-specific fields
    pattern: Optional[ReversalPattern] = None
    confirmation_type: str = ""
    daily_context: str = ""
    reference_candle_close: float = 0.0
    double_down_count: int = 0
    trailing_sl: float = 0.0
    trailing_sl_history: list = field(default_factory=list)
    # Set on close
    exit_price: Optional[float] = None
    exit_time: Optional[str] = None
    exit_reason: Optional[str] = None
    pnl_pips: float = 0.0
    pnl_dollars: float = 0.0
    rr_achieved: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dict matching V1/V2 trade format for metrics compatibility."""
        return {
            "id": self.id,
            "instrument": self.instrument,
            "direction": self.direction,
            "entry_price": self.entry_price,
            "entry_time": self.entry_time,
            "units": self.units,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "signal_frame": self.signal_frame,
            "confluence_score": self.confluence_score,
            "session": self.session,
            "sl_distance_pips": self.sl_distance_pips,
            "rr_target": self.rr_target,
            "entry_candle_idx": self.entry_candle_idx,
            "max_favorable_pips": self.max_favorable_pips,
            "max_adverse_pips": self.max_adverse_pips,
            "bars_in_trade": self.bars_in_trade,
            "exit_price": self.exit_price,
            "exit_time": self.exit_time,
            "exit_reason": self.exit_reason,
            "pnl_pips": round(self.pnl_pips, 1),
            "pnl_dollars": round(self.pnl_dollars, 2),
            "rr_achieved": round(self.rr_achieved, 2),
            # V3-specific
            "pattern_type": self.confirmation_type,
            "daily_context": self.daily_context,
            "double_down_count": self.double_down_count,
        }
