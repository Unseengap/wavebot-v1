"""Tests for OANDA client data parsing (no live API calls)."""
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from src.data.oanda_client import OandaClient


@pytest.fixture
def client():
    return OandaClient(api_token="test-token", account_id="test-account")


@pytest.fixture
def sample_candle_response():
    """Minimal OANDA candle response."""
    return {
        "candles": [
            {
                "time": "2025-01-01T00:00:00.000000000Z",
                "complete": True,
                "volume": 100,
                "mid": {"o": "1.10000", "h": "1.10150", "l": "1.09900", "c": "1.10100"},
                "bid": {"o": "1.09990", "h": "1.10140", "l": "1.09890", "c": "1.10090"},
                "ask": {"o": "1.10010", "h": "1.10160", "l": "1.09910", "c": "1.10110"},
            },
            {
                "time": "2025-01-01T00:05:00.000000000Z",
                "complete": True,
                "volume": 150,
                "mid": {"o": "1.10100", "h": "1.10200", "l": "1.10050", "c": "1.10180"},
                "bid": {"o": "1.10090", "h": "1.10190", "l": "1.10040", "c": "1.10170"},
                "ask": {"o": "1.10110", "h": "1.10210", "l": "1.10060", "c": "1.10190"},
            },
            {
                "time": "2025-01-01T00:10:00.000000000Z",
                "complete": False,  # Incomplete — should be excluded from collect_range
                "volume": 80,
                "mid": {"o": "1.10180", "h": "1.10250", "l": "1.10150", "c": "1.10220"},
            },
        ]
    }


class TestOandaClientInit:
    def test_practice_url(self):
        c = OandaClient("token", "acct", "practice")
        assert "fxpractice" in c.base_url

    def test_live_url(self):
        c = OandaClient("token", "acct", "live")
        assert "fxtrade" in c.base_url

    def test_default_is_practice(self):
        c = OandaClient("token", "acct")
        assert "fxpractice" in c.base_url

    def test_auth_header(self, client):
        assert "Authorization" in client.headers
        assert "Bearer test-token" in client.headers["Authorization"]


class TestToDataframe:
    def test_basic_parsing(self, client, sample_candle_response):
        candles = sample_candle_response["candles"]
        df = client._to_dataframe(candles)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "open_mid" in df.columns
        assert "close_bid" in df.columns
        assert "high_ask" in df.columns

    def test_correct_values(self, client, sample_candle_response):
        df = client._to_dataframe(sample_candle_response["candles"])
        assert df.iloc[0]["open_mid"] == pytest.approx(1.10000)
        assert df.iloc[0]["close_mid"] == pytest.approx(1.10100)
        assert df.iloc[0]["close_bid"] == pytest.approx(1.10090)
        assert df.iloc[0]["close_ask"] == pytest.approx(1.10110)

    def test_sorted_by_time(self, client):
        candles = [
            {"time": "2025-01-01T00:10:00Z", "volume": 50,
             "mid": {"o": "1.1", "h": "1.1", "l": "1.1", "c": "1.1"}},
            {"time": "2025-01-01T00:00:00Z", "volume": 50,
             "mid": {"o": "1.0", "h": "1.0", "l": "1.0", "c": "1.0"}},
        ]
        df = client._to_dataframe(candles)
        assert df.iloc[0]["open_mid"] == pytest.approx(1.0)
        assert df.iloc[1]["open_mid"] == pytest.approx(1.1)

    def test_deduplicates_by_time(self, client):
        candles = [
            {"time": "2025-01-01T00:00:00Z", "volume": 50,
             "mid": {"o": "1.0", "h": "1.0", "l": "1.0", "c": "1.0"}},
            {"time": "2025-01-01T00:00:00Z", "volume": 80,
             "mid": {"o": "1.1", "h": "1.1", "l": "1.1", "c": "1.1"}},
        ]
        df = client._to_dataframe(candles)
        assert len(df) == 1
        # keeps last
        assert df.iloc[0]["open_mid"] == pytest.approx(1.1)

    def test_empty_candles(self, client):
        df = client._to_dataframe([])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


class TestGetCandles:
    @patch("src.data.oanda_client.requests.get")
    def test_successful_fetch(self, mock_get, client, sample_candle_response):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = sample_candle_response
        mock_get.return_value = mock_resp

        candles = client.get_candles("EUR_USD", "M5", count=10)
        assert len(candles) == 3
        mock_get.assert_called_once()

    @patch("src.data.oanda_client.requests.get")
    def test_api_error_raises(self, mock_get, client):
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.text = "Bad request"
        mock_get.return_value = mock_resp

        with pytest.raises(Exception, match="OANDA API error 400"):
            client.get_candles("EUR_USD", "M5")

    @patch("src.data.oanda_client.requests.get")
    def test_count_capped_at_max(self, mock_get, client, sample_candle_response):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = sample_candle_response
        mock_get.return_value = mock_resp

        client.get_candles("EUR_USD", "M5", count=99999)
        call_params = mock_get.call_args[1]["params"]
        assert call_params["count"] <= OandaClient.MAX_CANDLES

    @patch("src.data.oanda_client.requests.get")
    def test_mba_price_requested(self, mock_get, client, sample_candle_response):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = sample_candle_response
        mock_get.return_value = mock_resp

        client.get_candles("EUR_USD", "M5")
        call_params = mock_get.call_args[1]["params"]
        assert call_params["price"] == "MBA"


class TestCollectRange:
    @patch("src.data.oanda_client.requests.get")
    def test_filters_incomplete_candles(self, mock_get, client, sample_candle_response):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = sample_candle_response
        mock_get.return_value = mock_resp

        df = client.collect_range("EUR_USD", "M5", "2025-01-01T00:00:00Z")
        # 3 candles, 1 incomplete → 2 in DataFrame
        assert len(df) == 2
