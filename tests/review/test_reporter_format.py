"""Tests for Reporter format_* methods (Task 10)."""

import pytest
from datetime import datetime, timezone
from core.types import (
    Signal, SignalMessage, SignalQuality, SignalAction,
    StrategyName, Side, ManualPosition,
)
from review.reporter import Reporter, _format_price
from execution.exit_manager import ExitSignal


@pytest.fixture
def reporter():
    return Reporter(sender=None, chat_id="test")


@pytest.fixture
def sample_signal_message():
    signal = Signal(
        timestamp=datetime.now(timezone.utc),
        symbol="BTCUSDT",
        action=SignalAction.ENTER_LONG,
        strategy=StrategyName.TREND_FOLLOWING,
        entry_price=67500,
        stop_loss=66000,
        take_profit=70000,
        confidence=0.8,
        metadata={"score": 5},
    )
    return SignalMessage(
        signal=signal,
        quality=SignalQuality.STRONG,
        explanation=["EMA 골든크로스", "RSI 상향돌파"],
        indicators={"rsi": 55},
        risk_reward_ratio=1.67,
    )


class TestFormatPrice:
    def test_large_price(self):
        assert _format_price(67500.0) == "67,500.00"

    def test_medium_price(self):
        assert _format_price(1.5) == "1.5000"

    def test_small_price(self):
        assert _format_price(0.00045) == "0.000450"


class TestFormatSignalMessage:
    def test_contains_symbol(self, reporter, sample_signal_message):
        text = reporter.format_signal_message(sample_signal_message)
        assert "BTCUSDT" in text

    def test_contains_direction(self, reporter, sample_signal_message):
        text = reporter.format_signal_message(sample_signal_message)
        assert "롱" in text

    def test_contains_rr(self, reporter, sample_signal_message):
        text = reporter.format_signal_message(sample_signal_message)
        assert "1.67" in text


class TestFormatWeeklyAccuracy:
    def test_no_signals(self, reporter):
        assert "없습니다" in reporter.format_weekly_accuracy({"total": 0})

    def test_with_signals(self, reporter):
        text = reporter.format_weekly_accuracy({"total": 10, "tp_rate": 0.6, "sl_rate": 0.2, "by_quality": {}})
        assert "10개" in text


class TestFormatPositionDashboard:
    def test_no_positions(self, reporter):
        assert "없습니다" in reporter.format_position_dashboard([], {})

    def test_with_position(self, reporter, make_position):
        pos = make_position()
        text = reporter.format_position_dashboard(
            [pos],
            {pos.symbol: {"current_price": 52000, "pnl_pct": 20.0, "pnl_usdt": 100.0}},
        )
        assert "BTCUSDT" in text and "+20.0%" in text


class TestFormatCoinAnalysis:
    def test_basic(self, reporter):
        analysis = {
            "symbol": "BTCUSDT", "price": 67500.0,
            "change_1h": 1.5, "change_24h": 3.2,
            "high_24h": 68000.0, "low_24h": 65000.0,
            "volume_24h": 5e9, "funding_rate": 0.0001,
            "indicators": {
                "rsi": 55.0, "adx": 30.0, "macd_hist": 0.002,
                "bb_width": 0.05, "ema_20": 67000.0, "ema_50": 66000.0,
                "atr": 500.0, "volume_ratio": 1.5, "is_sideways": False,
            },
            "ema_position": "강세 정배열",
            "direction": "long", "score": 4,
            "reasons": ["EMA 골든크로스", "RSI 상향돌파", "MACD 양전환", "ADX 30"],
            "verdict": "적극 매수", "verdict_reason": "4/7점",
            "entry": 67500.0, "sl": 66750.0, "tp": 69000.0,
        }
        text = reporter.format_coin_analysis(analysis)
        assert "BTC" in text and "적극 매수" in text


class TestFormatExitSignalV2:
    def test_sl_warning(self, reporter, make_position):
        sig = ExitSignal(
            signal_type="sl_warning",
            position_id=1,
            symbol="BTCUSDT",
            message="손절가까지 2.5% 남음",
            severity="warning",
            pnl_percent=-8.0,
            suggested_action="즉시 청산하세요.",
        )
        text = reporter.format_exit_signal_v2(sig, position=make_position())
        assert "손절 접근" in text and "BTCUSDT" in text


class TestFormatHourlyBriefing:
    def test_basic(self, reporter):
        briefing = {
            "time": "03/23 12:00 UTC",
            "market_summary": {"top_coins": [
                {
                    "symbol": "BTCUSDT", "price": 67500,
                    "change_1h": 1.0, "change_24h": 3.0,
                    "volume_24h": 5e9, "high_24h": 68000, "low_24h": 65000,
                }
            ]},
            "scored_coins": [], "funding_alerts": [], "watched_pairs": ["BTCUSDT"],
        }
        text = reporter.format_hourly_briefing(briefing)
        assert "BTC" in text and "브리핑" in text


class TestFormatJournalReport:
    def test_no_trades(self, reporter):
        assert "없습니다" in reporter.format_journal_report({"total_trades": 0})

    def test_with_trades(self, reporter):
        report = {
            "total_trades": 5, "wins": 3, "win_rate": 60,
            "total_pnl_usdt": 150.0, "avg_pnl_pct": 5.0,
            "max_consecutive_loss": 1, "by_regime": {},
        }
        text = reporter.format_journal_report(report)
        assert "5건" in text and "60%" in text
