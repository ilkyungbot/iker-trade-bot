# tests/service/test_briefing_service.py
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime, timezone, timedelta
from service.briefing_service import BriefingService
from service.coin_analyzer import CoinAnalyzer


class TestBriefingService:
    def setup_method(self):
        self.collector = MagicMock()
        self.config = MagicMock()
        self.config.signal.primary_interval = "240"
        self.config.signal.max_pairs = 5
        self.analyzer = CoinAnalyzer(collector=self.collector, config=self.config)
        self.service = BriefingService(
            collector=self.collector,
            config=self.config,
            coin_analyzer=self.analyzer,
        )

    @pytest.mark.anyio
    async def test_generate_briefing_empty_tickers(self):
        """When no tickers available, return minimal briefing."""
        self.collector.get_all_usdt_perpetuals = MagicMock(return_value=[])
        result = await self.service.generate_briefing()
        assert "time" in result
        assert result["scored_coins"] == []
        assert result["funding_alerts"] == []

    @pytest.mark.anyio
    async def test_generate_briefing_with_tickers(self):
        """When tickers available, return populated briefing."""
        ticker = {
            "symbol": "BTCUSDT",
            "mark_price": 67500.0,
            "last_price": 67500.0,
            "volume_24h": 5e9,
            "high_24h": 68000.0,
            "low_24h": 65000.0,
            "prev_price_1h": 67000.0,
            "price_24h_pct": 3.0,
            "funding_rate": 0.0001,
        }
        self.collector.get_all_usdt_perpetuals = MagicMock(return_value=[ticker])
        # get_candles returns empty so scoring is skipped
        self.collector.get_candles = MagicMock(return_value=[])
        self.collector.get_funding_rates = MagicMock(return_value=[])

        result = await self.service.generate_briefing()
        assert "time" in result
        assert "market_summary" in result
        top_coins = result["market_summary"].get("top_coins", [])
        assert len(top_coins) == 1
        assert top_coins[0]["symbol"] == "BTCUSDT"

    @pytest.mark.anyio
    async def test_generate_briefing_funding_alert(self):
        """High funding rate triggers funding alert."""
        from core.types import FundingRate

        ticker = {
            "symbol": "BTCUSDT",
            "mark_price": 67500.0,
            "last_price": 67500.0,
            "volume_24h": 5e9,
            "high_24h": 68000.0,
            "low_24h": 65000.0,
            "prev_price_1h": 67000.0,
            "price_24h_pct": 3.0,
            "funding_rate": 0.001,
        }
        self.collector.get_all_usdt_perpetuals = MagicMock(return_value=[ticker])
        self.collector.get_candles = MagicMock(return_value=[])

        # Funding rate above threshold (0.0005)
        now = datetime.now(timezone.utc)
        high_rate = FundingRate(timestamp=now, symbol="BTCUSDT", rate=0.001)
        self.collector.get_funding_rates = MagicMock(return_value=[high_rate])

        result = await self.service.generate_briefing()
        assert len(result["funding_alerts"]) == 1
        assert result["funding_alerts"][0]["symbol"] == "BTCUSDT"
        assert result["funding_alerts"][0]["rate"] == 0.001

    @pytest.mark.anyio
    async def test_generate_briefing_funding_below_threshold(self):
        """Low funding rate does NOT trigger funding alert."""
        from core.types import FundingRate

        ticker = {
            "symbol": "BTCUSDT",
            "mark_price": 67500.0,
            "last_price": 67500.0,
            "volume_24h": 5e9,
            "high_24h": 68000.0,
            "low_24h": 65000.0,
            "prev_price_1h": 67000.0,
            "price_24h_pct": 3.0,
            "funding_rate": 0.0001,
        }
        self.collector.get_all_usdt_perpetuals = MagicMock(return_value=[ticker])
        self.collector.get_candles = MagicMock(return_value=[])

        now = datetime.now(timezone.utc)
        low_rate = FundingRate(timestamp=now, symbol="BTCUSDT", rate=0.0001)
        self.collector.get_funding_rates = MagicMock(return_value=[low_rate])

        result = await self.service.generate_briefing()
        assert result["funding_alerts"] == []

    @pytest.mark.anyio
    async def test_generate_briefing_price_sanity_fallback(self):
        """When mark_price is way above high_24h, falls back to last_price."""
        ticker = {
            "symbol": "BTCUSDT",
            "mark_price": 80000.0,   # > high_24h * 1.05
            "last_price": 67500.0,
            "volume_24h": 5e9,
            "high_24h": 68000.0,
            "low_24h": 65000.0,
            "prev_price_1h": 67000.0,
            "price_24h_pct": 3.0,
            "funding_rate": 0.0001,
        }
        self.collector.get_all_usdt_perpetuals = MagicMock(return_value=[ticker])
        self.collector.get_candles = MagicMock(return_value=[])
        self.collector.get_funding_rates = MagicMock(return_value=[])

        result = await self.service.generate_briefing()
        top_coins = result["market_summary"].get("top_coins", [])
        assert len(top_coins) == 1
        # Should use last_price (67500) as fallback
        assert top_coins[0]["price"] == 67500.0

    @pytest.mark.anyio
    async def test_generate_briefing_funding_error_silenced(self):
        """Funding rate fetch errors are silently swallowed."""
        ticker = {
            "symbol": "BTCUSDT",
            "mark_price": 67500.0,
            "last_price": 67500.0,
            "volume_24h": 5e9,
            "high_24h": 68000.0,
            "low_24h": 65000.0,
            "prev_price_1h": 67000.0,
            "price_24h_pct": 3.0,
            "funding_rate": 0.0001,
        }
        self.collector.get_all_usdt_perpetuals = MagicMock(return_value=[ticker])
        self.collector.get_candles = MagicMock(return_value=[])
        self.collector.get_funding_rates = MagicMock(side_effect=Exception("network error"))

        # Should not raise
        result = await self.service.generate_briefing()
        assert "time" in result
        assert result["funding_alerts"] == []

    @pytest.mark.anyio
    async def test_generate_briefing_scored_coins_sorted(self):
        """Scored coins are sorted by score descending."""
        from core.types import Candle
        import pandas as pd

        # Make 3 tickers with known symbols
        tickers = [
            {
                "symbol": "BTCUSDT",
                "mark_price": 67500.0,
                "last_price": 67500.0,
                "volume_24h": 5e9,
                "high_24h": 68000.0,
                "low_24h": 65000.0,
                "prev_price_1h": 67000.0,
                "price_24h_pct": 3.0,
                "funding_rate": 0.0001,
            },
            {
                "symbol": "ETHUSDT",
                "mark_price": 3500.0,
                "last_price": 3500.0,
                "volume_24h": 2e9,
                "high_24h": 3600.0,
                "low_24h": 3400.0,
                "prev_price_1h": 3450.0,
                "price_24h_pct": 1.5,
                "funding_rate": 0.0002,
            },
        ]
        self.collector.get_all_usdt_perpetuals = MagicMock(return_value=tickers)
        # Return empty candles so scoring is skipped for all
        self.collector.get_candles = MagicMock(return_value=[])
        self.collector.get_funding_rates = MagicMock(return_value=[])

        result = await self.service.generate_briefing()
        # scored_coins should be empty (no candles)
        assert result["scored_coins"] == []
        # But top_coins should have both
        top_coins = result["market_summary"]["top_coins"]
        symbols = [c["symbol"] for c in top_coins]
        assert "BTCUSDT" in symbols
        assert "ETHUSDT" in symbols

    @pytest.mark.anyio
    async def test_generate_briefing_has_change_1h(self):
        """Each top_coin entry has 1h change computed from prev_price_1h."""
        ticker = {
            "symbol": "BTCUSDT",
            "mark_price": 67500.0,
            "last_price": 67500.0,
            "volume_24h": 5e9,
            "high_24h": 68000.0,
            "low_24h": 65000.0,
            "prev_price_1h": 67000.0,
            "price_24h_pct": 3.0,
            "funding_rate": 0.0001,
        }
        self.collector.get_all_usdt_perpetuals = MagicMock(return_value=[ticker])
        self.collector.get_candles = MagicMock(return_value=[])
        self.collector.get_funding_rates = MagicMock(return_value=[])

        result = await self.service.generate_briefing()
        coin = result["market_summary"]["top_coins"][0]
        expected_1h = round((67500.0 - 67000.0) / 67000.0 * 100, 2)
        assert coin["change_1h"] == pytest.approx(expected_1h, abs=0.01)
