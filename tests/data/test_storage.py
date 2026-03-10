"""Tests for data storage using SQLite in-memory."""

from datetime import datetime, timezone
import pytest
from data.storage import Storage
from core.types import Candle, Trade, Side, StrategyName, FundingRate


@pytest.fixture
def storage():
    return Storage(db_url="sqlite:///:memory:")


class TestCandleStorage:
    def test_save_and_retrieve(self, storage):
        candles = [
            Candle(datetime(2024, 1, 1, i, tzinfo=timezone.utc),
                   100, 105, 95, 102, 1000, "BTCUSDT", "60")
            for i in range(5)
        ]
        saved = storage.save_candles(candles)
        assert saved == 5

        retrieved = storage.get_candles("BTCUSDT", "60")
        assert len(retrieved) == 5
        assert retrieved[0].symbol == "BTCUSDT"

    def test_no_duplicates(self, storage):
        candle = Candle(datetime(2024, 1, 1, tzinfo=timezone.utc),
                        100, 105, 95, 102, 1000, "BTCUSDT", "60")
        storage.save_candles([candle])
        storage.save_candles([candle])  # same candle again
        retrieved = storage.get_candles("BTCUSDT", "60")
        assert len(retrieved) == 1

    def test_filter_by_time_range(self, storage):
        candles = [
            Candle(datetime(2024, 1, 1, i, tzinfo=timezone.utc),
                   100, 105, 95, 102, 1000, "BTCUSDT", "60")
            for i in range(10)
        ]
        storage.save_candles(candles)

        retrieved = storage.get_candles(
            "BTCUSDT", "60",
            start=datetime(2024, 1, 1, 3, tzinfo=timezone.utc),
            end=datetime(2024, 1, 1, 7, tzinfo=timezone.utc),
        )
        assert len(retrieved) == 5  # hours 3,4,5,6,7

    def test_limit(self, storage):
        candles = [
            Candle(datetime(2024, 1, 1, i, tzinfo=timezone.utc),
                   100, 105, 95, 102, 1000, "BTCUSDT", "60")
            for i in range(10)
        ]
        storage.save_candles(candles)
        retrieved = storage.get_candles("BTCUSDT", "60", limit=3)
        assert len(retrieved) == 3

    def test_empty_result(self, storage):
        retrieved = storage.get_candles("ETHUSDT", "60")
        assert retrieved == []


class TestTradeStorage:
    def test_save_and_retrieve(self, storage):
        trade = Trade(
            symbol="BTCUSDT", side=Side.LONG,
            strategy=StrategyName.TREND_FOLLOWING,
            entry_price=50000.0, exit_price=51000.0,
            quantity=0.1, leverage=5.0,
            entry_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
            exit_time=datetime(2024, 1, 2, tzinfo=timezone.utc),
            pnl=100.0, pnl_percent=2.0,
            fees=5.0, slippage=1.0,
            stop_loss_hit=False, trailing_stop_hit=True,
        )
        storage.save_trade(trade)

        trades = storage.get_trades(symbol="BTCUSDT")
        assert len(trades) == 1
        assert trades[0].pnl == 100.0
        assert trades[0].side == Side.LONG
        assert trades[0].strategy == StrategyName.TREND_FOLLOWING

    def test_filter_by_strategy(self, storage):
        for strat in [StrategyName.TREND_FOLLOWING, StrategyName.FUNDING_RATE]:
            storage.save_trade(Trade(
                symbol="BTCUSDT", side=Side.LONG, strategy=strat,
                entry_price=50000, exit_price=51000, quantity=0.1, leverage=5,
                entry_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
                exit_time=datetime(2024, 1, 2, tzinfo=timezone.utc),
                pnl=100, pnl_percent=2, fees=5, slippage=1,
                stop_loss_hit=False, trailing_stop_hit=False,
            ))

        tf_trades = storage.get_trades(strategy=StrategyName.TREND_FOLLOWING)
        assert len(tf_trades) == 1

    def test_filter_by_time(self, storage):
        for day in range(1, 4):
            storage.save_trade(Trade(
                symbol="BTCUSDT", side=Side.LONG,
                strategy=StrategyName.TREND_FOLLOWING,
                entry_price=50000, exit_price=51000, quantity=0.1, leverage=5,
                entry_time=datetime(2024, 1, day, tzinfo=timezone.utc),
                exit_time=datetime(2024, 1, day, 12, tzinfo=timezone.utc),
                pnl=100, pnl_percent=2, fees=5, slippage=1,
                stop_loss_hit=False, trailing_stop_hit=False,
            ))

        trades = storage.get_trades(
            start=datetime(2024, 1, 2, tzinfo=timezone.utc),
            end=datetime(2024, 1, 3, tzinfo=timezone.utc),
        )
        assert len(trades) == 2


class TestPerformanceStorage:
    def test_save_performance(self, storage):
        storage.save_performance(
            date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            total_capital=1000000.0,
            daily_pnl=5000.0,
            daily_pnl_percent=0.5,
            cumulative_pnl=5000.0,
            max_drawdown=0.01,
            open_positions=2,
            trades_today=3,
            win_rate=0.6,
        )
        # No assertion needed — just verify it doesn't raise


class TestFundingRateStorage:
    def test_save_and_no_duplicates(self, storage):
        rates = [
            FundingRate(datetime(2024, 1, 1, tzinfo=timezone.utc), "BTCUSDT", 0.0001),
            FundingRate(datetime(2024, 1, 1, 8, tzinfo=timezone.utc), "BTCUSDT", -0.0002),
        ]
        saved = storage.save_funding_rates(rates)
        assert saved == 2

        saved_again = storage.save_funding_rates(rates)
        assert saved_again == 0
