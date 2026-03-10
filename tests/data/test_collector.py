"""Tests for data collector with mocked Bybit API."""

from datetime import datetime, timezone
from data.collector import BybitCollector, _interval_to_ms


class MockBybitClient:
    """Mock Bybit API client for testing."""

    def __init__(self, candle_data=None, funding_data=None, oi_data=None, ticker_data=None):
        self._candle_data = candle_data or []
        self._funding_data = funding_data or []
        self._oi_data = oi_data or []
        self._ticker_data = ticker_data or []

    def get_kline(self, **kwargs):
        return {"result": {"list": self._candle_data}}

    def get_funding_rate_history(self, **kwargs):
        return {"result": {"list": self._funding_data}}

    def get_open_interest(self, **kwargs):
        return {"result": {"list": self._oi_data}}

    def get_tickers(self, **kwargs):
        return {"result": {"list": self._ticker_data}}


class TestBybitCollectorCandles:
    def test_parse_candles(self):
        mock_data = [
            # [timestamp_ms, open, high, low, close, volume, turnover]
            ["1704067200000", "42000.0", "42500.0", "41800.0", "42200.0", "100.5", "4200000"],
            ["1704063600000", "41500.0", "42100.0", "41400.0", "42000.0", "95.3", "3950000"],
        ]
        client = MockBybitClient(candle_data=mock_data)
        collector = BybitCollector(client=client)

        candles = collector.get_candles(
            symbol="BTCUSDT",
            interval="60",
            start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_time=datetime(2024, 1, 1, 2, 0, tzinfo=timezone.utc),
        )

        assert len(candles) == 2
        # Should be sorted ascending by timestamp
        assert candles[0].timestamp < candles[1].timestamp
        assert candles[0].symbol == "BTCUSDT"
        assert candles[0].open == 41500.0
        assert candles[0].close == 42000.0
        assert candles[1].close == 42200.0

    def test_empty_response(self):
        client = MockBybitClient(candle_data=[])
        collector = BybitCollector(client=client)

        candles = collector.get_candles(
            symbol="BTCUSDT",
            interval="60",
            start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        assert candles == []


class TestBybitCollectorFunding:
    def test_parse_funding_rates(self):
        mock_data = [
            {"fundingRateTimestamp": "1704067200000", "fundingRate": "0.0001", "symbol": "BTCUSDT"},
            {"fundingRateTimestamp": "1704038400000", "fundingRate": "-0.0002", "symbol": "BTCUSDT"},
        ]
        client = MockBybitClient(funding_data=mock_data)
        collector = BybitCollector(client=client)

        rates = collector.get_funding_rates(
            symbol="BTCUSDT",
            start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_time=datetime(2024, 1, 2, tzinfo=timezone.utc),
        )

        assert len(rates) == 2
        assert rates[0].timestamp < rates[1].timestamp
        assert rates[0].rate == -0.0002
        assert rates[1].rate == 0.0001


class TestBybitCollectorOpenInterest:
    def test_parse_open_interest(self):
        mock_data = [
            {"timestamp": "1704067200000", "openInterest": "50000.5"},
            {"timestamp": "1704063600000", "openInterest": "49500.3"},
        ]
        client = MockBybitClient(oi_data=mock_data)
        collector = BybitCollector(client=client)

        oi = collector.get_open_interest_history(
            symbol="BTCUSDT",
            start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )

        assert len(oi) == 2
        assert oi[0].timestamp < oi[1].timestamp
        assert oi[0].value == 49500.3


class TestBybitCollectorTickers:
    def test_get_usdt_perpetuals(self):
        mock_data = [
            {"symbol": "BTCUSDT", "turnover24h": "500000000", "lastPrice": "42000"},
            {"symbol": "ETHUSDT", "turnover24h": "200000000", "lastPrice": "2200"},
            {"symbol": "BTCPERP", "turnover24h": "100000000", "lastPrice": "42000"},  # non-USDT
            {"symbol": "SOLUSDT", "turnover24h": "50000000", "lastPrice": "100"},
        ]
        client = MockBybitClient(ticker_data=mock_data)
        collector = BybitCollector(client=client)

        perps = collector.get_all_usdt_perpetuals()

        # Should only include USDT pairs, sorted by volume desc
        assert len(perps) == 3
        assert perps[0]["symbol"] == "BTCUSDT"
        assert perps[1]["symbol"] == "ETHUSDT"
        assert perps[2]["symbol"] == "SOLUSDT"

    def test_empty_tickers(self):
        client = MockBybitClient(ticker_data=[])
        collector = BybitCollector(client=client)
        perps = collector.get_all_usdt_perpetuals()
        assert perps == []


class TestIntervalToMs:
    def test_one_hour(self):
        assert _interval_to_ms("60") == 3_600_000

    def test_four_hours(self):
        assert _interval_to_ms("240") == 14_400_000

    def test_fifteen_minutes(self):
        assert _interval_to_ms("15") == 900_000

    def test_unknown_defaults_to_one_hour(self):
        assert _interval_to_ms("unknown") == 3_600_000
