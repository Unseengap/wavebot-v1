"""Tracks historical wave amplitudes and durations for maturity + TP."""
import numpy as np
from collections import defaultdict


class AmplitudeTracker:
    def __init__(self):
        # Key: (instrument, granularity) -> list of (amplitude_pips, duration_candles)
        self._records = defaultdict(list)

    def record_wave(self, instrument: str, granularity: str,
                    amplitude_pips: float, duration_candles: int):
        if amplitude_pips > 0 and duration_candles > 0:
            self._records[(instrument, granularity)].append(
                (amplitude_pips, duration_candles)
            )

    def get_amplitude_stats(self, instrument: str, granularity: str) -> dict:
        """Returns percentile statistics for wave amplitudes."""
        records = self._records.get((instrument, granularity), [])
        if len(records) < 5:
            return {"p25": 15, "p50": 25, "p75": 40, "p90": 60, "mean_pips": 30}

        amps = np.array([r[0] for r in records])
        return {
            "p25": float(np.percentile(amps, 25)),
            "p50": float(np.percentile(amps, 50)),
            "p75": float(np.percentile(amps, 75)),
            "p90": float(np.percentile(amps, 90)),
            "mean_pips": float(np.mean(amps)),
        }

    def get_duration_stats(self, instrument: str, granularity: str) -> dict:
        """Returns percentile statistics for wave durations (candle count)."""
        records = self._records.get((instrument, granularity), [])
        if len(records) < 5:
            return {"p50_candles": 20, "p75_candles": 35, "mean_candles": 25}

        durs = np.array([r[1] for r in records])
        return {
            "p50_candles": int(np.percentile(durs, 50)),
            "p75_candles": int(np.percentile(durs, 75)),
            "mean_candles": float(np.mean(durs)),
        }

    def get_count(self, instrument: str, granularity: str) -> int:
        return len(self._records.get((instrument, granularity), []))
