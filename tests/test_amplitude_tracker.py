"""Tests for amplitude tracker."""
import pytest
from src.wave.amplitude_tracker import AmplitudeTracker


class TestAmplitudeTracker:
    def test_defaults_with_few_records(self):
        t = AmplitudeTracker()
        stats = t.get_amplitude_stats("EUR_USD", "M5")
        assert stats["p50"] == 25  # Default
        assert stats["mean_pips"] == 30

    def test_duration_defaults(self):
        t = AmplitudeTracker()
        stats = t.get_duration_stats("EUR_USD", "M5")
        assert stats["p50_candles"] == 20

    def test_record_and_retrieve(self):
        t = AmplitudeTracker()
        for i in range(10):
            t.record_wave("EUR_USD", "M5", 10.0 + i, 5 + i)
        stats = t.get_amplitude_stats("EUR_USD", "M5")
        assert "p50" in stats
        assert stats["p50"] > 0
        assert t.get_count("EUR_USD", "M5") == 10

    def test_ignores_zero_amplitude(self):
        t = AmplitudeTracker()
        t.record_wave("EUR_USD", "M5", 0.0, 10)
        assert t.get_count("EUR_USD", "M5") == 0

    def test_ignores_zero_duration(self):
        t = AmplitudeTracker()
        t.record_wave("EUR_USD", "M5", 10.0, 0)
        assert t.get_count("EUR_USD", "M5") == 0

    def test_separate_instruments(self):
        t = AmplitudeTracker()
        for i in range(6):
            t.record_wave("EUR_USD", "M5", 10.0 + i, 5 + i)
            t.record_wave("GBP_USD", "M5", 20.0 + i, 8 + i)
        eur_stats = t.get_amplitude_stats("EUR_USD", "M5")
        gbp_stats = t.get_amplitude_stats("GBP_USD", "M5")
        assert gbp_stats["p50"] > eur_stats["p50"]

    def test_separate_granularities(self):
        t = AmplitudeTracker()
        for i in range(6):
            t.record_wave("EUR_USD", "M5", 10.0, 5)
            t.record_wave("EUR_USD", "H1", 50.0, 20)
        assert t.get_count("EUR_USD", "M5") == 6
        assert t.get_count("EUR_USD", "H1") == 6

    def test_duration_stats_with_data(self):
        t = AmplitudeTracker()
        for i in range(10):
            t.record_wave("EUR_USD", "M5", 15.0, 10 + i * 2)
        stats = t.get_duration_stats("EUR_USD", "M5")
        assert stats["p50_candles"] > 0
        assert stats["mean_candles"] > 0
