"""Tests for confluence alignment engine and entry filter."""
import pytest
from src.confluence.alignment_engine import (
    get_directional_gate, calculate_confluence_score,
    get_signal_frame, get_session, get_session_size_multiplier,
    _get_primary_maturity,
)
from src.confluence.entry_filter import check_entry_conditions
from src.wave.wave_scorer import WaveScore


def _ws(direction=1.0, conviction=0.8, maturity=0.3,
        state="BULLISH_IMPULSE", granularity="M5"):
    return WaveScore(
        instrument="EUR_USD", granularity=granularity,
        timestamp="2025-01-01T12:00:00Z",
        direction=direction, conviction=conviction,
        maturity=maturity, state=state,
        wave_origin=1.1000, wave_age_candles=10, wave_pips=15.0,
    )


class TestDirectionalGate:
    def test_bullish_daily(self):
        scores = {"D": _ws(direction=1.0, conviction=0.8, granularity="D")}
        assert get_directional_gate(scores) == "BULLISH"

    def test_bearish_daily(self):
        scores = {"D": _ws(direction=-1.0, conviction=0.8, granularity="D")}
        assert get_directional_gate(scores) == "BEARISH"

    def test_neutral_low_conviction(self):
        scores = {"D": _ws(direction=0.5, conviction=0.2, granularity="D")}
        assert get_directional_gate(scores) == "NEUTRAL"

    def test_no_daily_is_neutral(self):
        scores = {"M5": _ws(granularity="M5")}
        assert get_directional_gate(scores) == "NEUTRAL"

    def test_weekly_overrides_when_strong(self):
        scores = {
            "D": _ws(direction=1.0, conviction=0.8, granularity="D"),
            "W": _ws(direction=-1.0, conviction=0.8, granularity="W"),
        }
        # W direction*conviction = -0.8, abs > 0.5 → W overrides
        assert get_directional_gate(scores) == "BEARISH"

    def test_weekly_weak_doesnt_override(self):
        scores = {
            "D": _ws(direction=1.0, conviction=0.8, granularity="D"),
            "W": _ws(direction=-0.5, conviction=0.3, granularity="W"),
        }
        # W direction*conviction = -0.15, abs < 0.5 → D is used
        assert get_directional_gate(scores) == "BULLISH"


class TestConfluenceScore:
    def test_all_bullish_positive_score(self):
        scores = {
            "M5": _ws(direction=1.0, conviction=0.8, granularity="M5"),
            "M15": _ws(direction=1.0, conviction=0.7, granularity="M15"),
            "H1": _ws(direction=1.0, conviction=0.6, granularity="H1"),
        }
        s = calculate_confluence_score(scores)
        assert s > 0

    def test_all_bearish_negative_score(self):
        scores = {
            "M5": _ws(direction=-1.0, conviction=0.8, state="BEARISH_IMPULSE", granularity="M5"),
            "H1": _ws(direction=-1.0, conviction=0.7, state="BEARISH_IMPULSE", granularity="H1"),
        }
        s = calculate_confluence_score(scores)
        assert s < 0

    def test_output_bounds(self):
        scores = {
            "M1": _ws(direction=1.0, conviction=1.0, granularity="M1"),
            "M5": _ws(direction=1.0, conviction=1.0, granularity="M5"),
            "M15": _ws(direction=1.0, conviction=1.0, granularity="M15"),
            "H1": _ws(direction=1.0, conviction=1.0, granularity="H1"),
            "H4": _ws(direction=1.0, conviction=1.0, granularity="H4"),
        }
        s = calculate_confluence_score(scores)
        assert -1.0 <= s <= 1.0

    def test_empty_scores(self):
        assert calculate_confluence_score({}) == 0.0

    def test_dw_excluded_from_entry_score(self):
        """D and W should not contribute to entry scoring."""
        scores_with_dw = {
            "M5": _ws(direction=1.0, conviction=0.8, granularity="M5"),
            "D": _ws(direction=-1.0, conviction=1.0, granularity="D"),
            "W": _ws(direction=-1.0, conviction=1.0, granularity="W"),
        }
        scores_without_dw = {
            "M5": _ws(direction=1.0, conviction=0.8, granularity="M5"),
        }
        assert calculate_confluence_score(scores_with_dw) == calculate_confluence_score(scores_without_dw)

    def test_maturity_penalty(self):
        """High maturity on signal frame should reduce score."""
        fresh = {
            "M5": _ws(direction=1.0, conviction=0.8, maturity=0.2,
                       state="BULLISH_IMPULSE", granularity="M5"),
        }
        tired = {
            "M5": _ws(direction=1.0, conviction=0.8, maturity=0.9,
                       state="BULLISH_IMPULSE", granularity="M5"),
        }
        assert calculate_confluence_score(fresh) > calculate_confluence_score(tired)


class TestGetSignalFrame:
    def test_finds_fresh_bullish_m5(self):
        scores = {
            "M5": _ws(direction=1.0, maturity=0.2, state="BULLISH_IMPULSE", granularity="M5"),
        }
        sig = get_signal_frame(scores, "LONG")
        assert sig is not None
        assert sig.granularity == "M5"

    def test_skips_exhausted(self):
        scores = {
            "M5": _ws(direction=1.0, maturity=0.8, state="BULLISH_IMPULSE", granularity="M5"),
        }
        sig = get_signal_frame(scores, "LONG")
        assert sig is None

    def test_returns_none_for_wrong_direction(self):
        scores = {
            "M5": _ws(direction=1.0, maturity=0.2, state="BULLISH_IMPULSE", granularity="M5"),
        }
        sig = get_signal_frame(scores, "SHORT")
        assert sig is None

    def test_invalid_direction(self):
        scores = {"M5": _ws(granularity="M5")}
        assert get_signal_frame(scores, "SIDEWAYS") is None


class TestSession:
    def test_asia(self):
        assert get_session(3) == "Asia_Session"

    def test_london_open(self):
        assert get_session(10) == "London_Open"

    def test_overlap(self):
        assert get_session(14) == "London_NY_Overlap"

    def test_ny(self):
        assert get_session(18) == "NY_Session"

    def test_dead_zone(self):
        assert get_session(23) == "Dead_Zone"

    def test_session_multipliers(self):
        assert get_session_size_multiplier("Dead_Zone") == 0.0
        assert get_session_size_multiplier("London_NY_Overlap") == 1.25
        assert get_session_size_multiplier("Unknown") == 1.0


class TestEntryFilter:
    @pytest.fixture
    def base_args(self, entry_config):
        """All-passing entry conditions."""
        sig = _ws(direction=1.0, conviction=0.8, maturity=0.3,
                  state="BULLISH_IMPULSE", granularity="M5")
        return {
            "instrument": "EUR_USD",
            "direction": "LONG",
            "confluence_score": 0.75,
            "wave_scores": {
                "M5": _ws(direction=1.0, conviction=0.8, granularity="M5"),
                "M15": _ws(direction=1.0, conviction=0.7, granularity="M15"),
                "H1": _ws(direction=1.0, conviction=0.6, granularity="H1"),
            },
            "gate": "BULLISH",
            "open_trades": [],
            "current_spread_pips": 1.5,
            "max_spread_pips": 2.0,
            "signal_frame_ws": sig,
            "rr_ratio": 2.5,
            "daily_drawdown_pct": 0.005,
            "session": "London_Open",
            "config": entry_config,
        }

    def test_all_pass(self, base_args):
        ok, fails = check_entry_conditions(**base_args)
        assert ok is True
        assert fails == []

    def test_low_confluence_fails(self, base_args):
        base_args["confluence_score"] = 0.4
        ok, fails = check_entry_conditions(**base_args)
        assert ok is False
        assert any("Score" in f for f in fails)

    def test_neutral_gate_fails(self, base_args):
        base_args["gate"] = "NEUTRAL"
        ok, fails = check_entry_conditions(**base_args)
        assert ok is False
        assert any("NEUTRAL" in f for f in fails)

    def test_counter_gate_low_score_fails(self, base_args):
        base_args["gate"] = "BEARISH"
        base_args["confluence_score"] = 0.75  # < 0.85
        ok, fails = check_entry_conditions(**base_args)
        assert ok is False

    def test_counter_gate_high_score_passes(self, base_args):
        base_args["gate"] = "BEARISH"
        base_args["confluence_score"] = 0.90  # > 0.85
        ok, fails = check_entry_conditions(**base_args)
        # score check still passes since 0.90 > 0.65
        # gate check passes since abs(0.90) > 0.85
        assert not any("gate" in f.lower() for f in fails)

    def test_duplicate_instrument_fails(self, base_args):
        base_args["open_trades"] = [{"instrument": "EUR_USD"}]
        ok, fails = check_entry_conditions(**base_args)
        assert ok is False
        assert any("position" in f.lower() for f in fails)

    def test_max_trades_fails(self, base_args):
        base_args["open_trades"] = [
            {"instrument": "GBP_USD"},
            {"instrument": "USD_JPY"},
            {"instrument": "AUD_USD"},
        ]
        ok, fails = check_entry_conditions(**base_args)
        assert ok is False
        assert any("Max trades" in f for f in fails)

    def test_high_spread_fails(self, base_args):
        base_args["current_spread_pips"] = 5.0
        ok, fails = check_entry_conditions(**base_args)
        assert ok is False
        assert any("Spread" in f for f in fails)

    def test_high_maturity_fails(self, base_args):
        base_args["signal_frame_ws"] = _ws(maturity=0.9)
        ok, fails = check_entry_conditions(**base_args)
        assert ok is False
        assert any("Maturity" in f for f in fails)

    def test_low_rr_fails(self, base_args):
        base_args["rr_ratio"] = 1.2
        ok, fails = check_entry_conditions(**base_args)
        assert ok is False
        assert any("R:R" in f for f in fails)

    def test_drawdown_limit_fails(self, base_args):
        base_args["daily_drawdown_pct"] = 0.03
        ok, fails = check_entry_conditions(**base_args)
        assert ok is False
        assert any("drawdown" in f.lower() for f in fails)

    def test_dead_zone_fails(self, base_args):
        base_args["session"] = "Dead_Zone"
        ok, fails = check_entry_conditions(**base_args)
        assert ok is False
        assert any("Dead Zone" in f for f in fails)

    def test_short_direction(self, entry_config):
        """SHORT entry requires negative confluence."""
        sig = _ws(direction=-1.0, conviction=0.8, maturity=0.3,
                  state="BEARISH_IMPULSE", granularity="M5")
        ok, fails = check_entry_conditions(
            instrument="EUR_USD", direction="SHORT",
            confluence_score=-0.75,
            wave_scores={
                "M5": _ws(direction=-1.0, conviction=0.8,
                          state="BEARISH_IMPULSE", granularity="M5"),
                "H1": _ws(direction=-1.0, conviction=0.7,
                          state="BEARISH_IMPULSE", granularity="H1"),
            },
            gate="BEARISH", open_trades=[], current_spread_pips=1.5,
            max_spread_pips=2.0, signal_frame_ws=sig,
            rr_ratio=2.5, daily_drawdown_pct=0.005,
            session="London_Open", config=entry_config,
        )
        assert ok is True
