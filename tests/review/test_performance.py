"""Tests for performance analysis."""

from datetime import datetime, timezone, timedelta
from review.performance import (
    calculate_metrics,
    calculate_strategy_attribution,
    calculate_pair_attribution,
)
from core.types import Trade, Side, StrategyName


def _make_trade(
    pnl: float = 100.0,
    symbol: str = "BTCUSDT",
    strategy: StrategyName = StrategyName.TREND_FOLLOWING,
    fees: float = 5.0,
    hours: int = 4,
) -> Trade:
    return Trade(
        symbol=symbol, side=Side.LONG, strategy=strategy,
        entry_price=50000, exit_price=50000 + pnl * 10,
        quantity=0.1, leverage=5,
        entry_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
        exit_time=datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(hours=hours),
        pnl=pnl, pnl_percent=pnl / 100,
        fees=fees, slippage=0,
        stop_loss_hit=pnl < 0, trailing_stop_hit=False,
    )


class TestMetrics:
    def test_empty_trades(self):
        m = calculate_metrics([])
        assert m.total_trades == 0
        assert m.win_rate == 0

    def test_all_winners(self):
        trades = [_make_trade(pnl=100) for _ in range(5)]
        m = calculate_metrics(trades)
        assert m.win_rate == 1.0
        assert m.total_pnl == 525  # gross: 500 pnl + 25 fees
        assert m.net_pnl == 500    # net: 525 - 25 fees
        assert m.winning_trades == 5

    def test_all_losers(self):
        trades = [_make_trade(pnl=-50) for _ in range(5)]
        m = calculate_metrics(trades)
        assert m.win_rate == 0.0
        assert m.total_pnl == -225  # gross: -250 + 25 fees
        assert m.net_pnl == -250    # net: -225 - 25 fees

    def test_mixed(self):
        trades = [
            _make_trade(pnl=200),
            _make_trade(pnl=-100),
            _make_trade(pnl=150),
            _make_trade(pnl=-80),
        ]
        m = calculate_metrics(trades)
        assert m.total_trades == 4
        assert m.winning_trades == 2
        assert m.win_rate == 0.5
        assert m.avg_win == 175.0  # (200+150)/2
        assert m.avg_loss == 90.0  # (100+80)/2

    def test_max_drawdown(self):
        trades = [
            _make_trade(pnl=100),
            _make_trade(pnl=-200),
            _make_trade(pnl=-100),
            _make_trade(pnl=500),
        ]
        m = calculate_metrics(trades)
        # Peak at 100, then drops to -200 (cumulative at 100, then -100, then -200, then 300)
        # Drawdown = 100 - (-200) = 300
        assert m.max_drawdown == 300

    def test_profit_factor(self):
        trades = [
            _make_trade(pnl=300),
            _make_trade(pnl=-100),
        ]
        m = calculate_metrics(trades)
        assert m.profit_factor == 3.0

    def test_avg_duration(self):
        trades = [_make_trade(hours=4), _make_trade(hours=8)]
        m = calculate_metrics(trades)
        assert m.avg_trade_duration_hours == 6.0


class TestStrategyAttribution:
    def test_separates_strategies(self):
        trades = [
            _make_trade(pnl=100, strategy=StrategyName.TREND_FOLLOWING),
            _make_trade(pnl=50, strategy=StrategyName.FUNDING_RATE),
            _make_trade(pnl=-30, strategy=StrategyName.TREND_FOLLOWING),
        ]
        attrs = calculate_strategy_attribution(trades)

        assert "trend_following" in attrs
        assert "funding_rate" in attrs
        assert attrs["trend_following"].total_trades == 2
        assert attrs["funding_rate"].total_trades == 1


class TestPairAttribution:
    def test_separates_pairs(self):
        trades = [
            _make_trade(pnl=100, symbol="BTCUSDT"),
            _make_trade(pnl=50, symbol="ETHUSDT"),
        ]
        attrs = calculate_pair_attribution(trades)
        assert "BTCUSDT" in attrs
        assert "ETHUSDT" in attrs
