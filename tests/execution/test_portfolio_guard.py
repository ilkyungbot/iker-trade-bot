"""Tests for PortfolioGuard — portfolio-level risk controls."""

from datetime import datetime, timezone

import pytest

from core.types import ManualPosition, Side
from execution.portfolio_guard import GuardResult, PortfolioGuard


def _make_position(symbol: str = "BTCUSDT", side: Side = Side.LONG) -> ManualPosition:
    return ManualPosition(
        id=1,
        chat_id="test",
        symbol=symbol,
        side=side,
        entry_price=50000.0,
        leverage=5.0,
        created_at=datetime.now(timezone.utc),
    )


class TestPortfolioGuard:
    def test_allows_when_under_limit(self):
        guard = PortfolioGuard(max_positions=3, account_balance_usdt=10000.0)
        positions = [_make_position("BTCUSDT")]
        result = guard.check_can_open(positions, "ETHUSDT")
        assert result.allowed is True
        assert result.reason == "진입 가능"

    def test_blocks_max_positions(self):
        guard = PortfolioGuard(max_positions=2, account_balance_usdt=10000.0)
        positions = [_make_position("BTCUSDT"), _make_position("ETHUSDT")]
        result = guard.check_can_open(positions, "SOLUSDT")
        assert result.allowed is False
        assert "최대 동시 포지션" in result.reason
        assert "2개" in result.reason

    def test_blocks_daily_loss_limit(self):
        guard = PortfolioGuard(daily_loss_limit_pct=5.0, account_balance_usdt=1000.0)
        # Record loss: 60 USDT on 1000 USDT balance = 6%
        guard.record_realized_loss(60.0)
        result = guard.check_can_open([], "BTCUSDT")
        assert result.allowed is False
        assert "일일 손실 한도" in result.reason

    def test_blocks_monthly_drawdown(self):
        guard = PortfolioGuard(
            daily_loss_limit_pct=99.0,
            monthly_drawdown_limit_pct=20.0,
            account_balance_usdt=1000.0,
        )
        # Record loss: 250 USDT on 1000 USDT balance = 25%
        guard.record_realized_loss(250.0)
        result = guard.check_can_open([], "BTCUSDT")
        assert result.allowed is False
        assert "월간 드로다운" in result.reason

    def test_record_loss_accumulates(self):
        guard = PortfolioGuard(account_balance_usdt=10000.0)
        guard.record_realized_loss(10.0)
        guard.record_realized_loss(20.0)
        assert guard._daily_realized_loss == 30.0
        assert guard._monthly_realized_loss == 30.0

    def test_record_profit_reduces_loss(self):
        guard = PortfolioGuard(account_balance_usdt=10000.0)
        guard.record_realized_loss(50.0)
        guard.record_realized_loss(-20.0)  # profit
        assert guard._daily_realized_loss == 30.0
        assert guard._monthly_realized_loss == 30.0

    def test_reset_daily(self):
        guard = PortfolioGuard(account_balance_usdt=10000.0)
        guard.record_realized_loss(50.0)
        guard.reset_daily()
        assert guard._daily_realized_loss == 0.0
        # Monthly should remain
        assert guard._monthly_realized_loss == 50.0

    def test_get_daily_status(self):
        guard = PortfolioGuard(daily_loss_limit_pct=5.0, account_balance_usdt=1000.0)
        guard.record_realized_loss(30.0)
        status = guard.get_daily_status()
        assert status["realized_loss_usdt"] == 30.0
        assert status["account_balance_usdt"] == 1000.0
        assert status["loss_pct"] == 3.0
        assert status["limit_pct"] == 5.0
        assert status["remaining_pct"] == 2.0

    def test_set_account_balance(self):
        guard = PortfolioGuard(account_balance_usdt=1000.0)
        guard.set_account_balance(5000.0)
        assert guard.account_balance_usdt == 5000.0

    def test_no_balance_skips_loss_check(self):
        guard = PortfolioGuard(daily_loss_limit_pct=5.0, account_balance_usdt=0.0)
        guard.record_realized_loss(99999.0)
        result = guard.check_can_open([], "BTCUSDT")
        assert result.allowed is True
