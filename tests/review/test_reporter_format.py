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


class TestFormatMonitoringUpdate:
    def test_long_profit(self, reporter):
        text = reporter.format_monitoring_update(
            symbol="BTCUSDT",
            direction="long",
            entry_price=50000.0,
            current_price=52000.0,
            stop_loss=48000.0,
            take_profit=55000.0,
        )
        assert "BTCUSDT" in text
        assert "롱" in text
        assert "홀딩" in text
        assert "+4.00%" in text

    def test_short_profit(self, reporter):
        # Short: profit when price goes down
        text = reporter.format_monitoring_update(
            symbol="ETHUSDT",
            direction="short",
            entry_price=3000.0,
            current_price=2900.0,
            stop_loss=3200.0,
            take_profit=2700.0,
        )
        assert "숏" in text
        assert "ETHUSDT" in text

    def test_long_loss(self, reporter):
        text = reporter.format_monitoring_update(
            symbol="SOLUSDT",
            direction="long",
            entry_price=100.0,
            current_price=95.0,
            stop_loss=90.0,
            take_profit=115.0,
        )
        assert "SOLUSDT" in text
        assert "-5.00%" in text


class TestFormatExitSignal:
    def test_long_exit(self, reporter):
        text = reporter.format_exit_signal("BTCUSDT", "long", "EMA 데드크로스")
        assert "BTCUSDT" in text
        assert "롱" in text
        assert "EMA 데드크로스" in text
        assert "청산" in text

    def test_short_exit(self, reporter):
        text = reporter.format_exit_signal("ETHUSDT", "short", "RSI 과매수")
        assert "숏" in text
        assert "RSI 과매수" in text


class TestFormatPositionRegistered:
    def test_with_sl_tp(self, reporter, make_position):
        pos = make_position(stop_loss=48000.0, take_profit=55000.0)
        text = reporter.format_position_registered(pos)
        assert "등록" in text
        assert "BTCUSDT" in text
        assert "롱" in text
        assert "50,000.00" in text  # entry_price formatted
        assert "모니터링" in text

    def test_short_position(self, reporter, make_position):
        from core.types import Side
        pos = make_position(side=Side.SHORT, entry_price=3000.0)
        text = reporter.format_position_registered(pos)
        assert "숏" in text

    def test_without_sl_tp(self, reporter, make_position):
        pos = make_position(stop_loss=None, take_profit=None)
        text = reporter.format_position_registered(pos)
        assert "등록" in text


class TestFormatPositionClosed:
    def test_with_pnl(self, reporter, make_position):
        pos = make_position()
        text = reporter.format_position_closed(pos, final_pnl=15.5)
        assert "청산" in text
        assert "BTCUSDT" in text
        assert "+15.5%" in text
        assert "모니터링을 종료" in text

    def test_without_pnl(self, reporter, make_position):
        pos = make_position()
        text = reporter.format_position_closed(pos, final_pnl=None)
        assert "청산" in text
        assert "모니터링을 종료" in text
        # PnL line should NOT appear
        assert "PnL" not in text

    def test_negative_pnl(self, reporter, make_position):
        pos = make_position()
        text = reporter.format_position_closed(pos, final_pnl=-8.2)
        assert "-8.2%" in text

    def test_nan_pnl_omitted(self, reporter, make_position):
        import math
        pos = make_position()
        text = reporter.format_position_closed(pos, final_pnl=float("nan"))
        # NaN pnl should be omitted
        assert "PnL" not in text


class TestFormatEdgeAlert:
    def _make_edge_signal(self, direction="bearish"):
        from strategy.edge_detector import EdgeSignal
        return EdgeSignal(
            signal_type="funding_extreme",
            direction=direction,
            strength="strong",
            message="펀딩레이트 극단적 양수 (0.1%)",
            data={"rate": 0.001},
        )

    def test_bearish_signal_long_position(self, reporter, make_position):
        """Bearish signal against long position = unfavorable."""
        edge = self._make_edge_signal(direction="bearish")
        pos = make_position()  # LONG by default
        text = reporter.format_edge_alert(edge, pos)
        assert "불리" in text
        assert "BTCUSDT" in text

    def test_bullish_signal_short_position(self, reporter, make_position):
        """Bullish signal against short position = unfavorable."""
        from core.types import Side
        edge = self._make_edge_signal(direction="bullish")
        pos = make_position(side=Side.SHORT)
        text = reporter.format_edge_alert(edge, pos)
        assert "불리" in text

    def test_bearish_signal_short_position(self, reporter, make_position):
        """Bearish signal with short position = favorable."""
        from core.types import Side
        edge = self._make_edge_signal(direction="bearish")
        pos = make_position(side=Side.SHORT)
        text = reporter.format_edge_alert(edge, pos)
        assert "유리" in text

    def test_bullish_signal_long_position(self, reporter, make_position):
        """Bullish signal with long position = favorable."""
        edge = self._make_edge_signal(direction="bullish")
        pos = make_position()  # LONG by default
        text = reporter.format_edge_alert(edge, pos)
        assert "유리" in text

    def test_neutral_direction(self, reporter, make_position):
        """Neutral direction gives neutral message."""
        edge = self._make_edge_signal(direction="neutral")
        pos = make_position()
        text = reporter.format_edge_alert(edge, pos)
        assert "중립" in text


class TestFormatRegimeChange:
    def _make_regime_state(self, regime_value="high_vol_trend", atr_ratio=1.5, btc_above=True):
        from strategy.market_regime import RegimeState, Regime
        regime_map = {
            "high_vol_trend": Regime.HIGH_VOL_TREND,
            "low_vol_range": Regime.LOW_VOL_RANGE,
            "transition": Regime.TRANSITION,
        }
        return RegimeState(
            regime=regime_map[regime_value],
            atr_ratio=atr_ratio,
            btc_above_200ema=btc_above,
            message="고변동 추세장 — 추세추종 유리",
        )

    def test_high_vol_trend_no_position(self, reporter):
        regime = self._make_regime_state("high_vol_trend")
        text = reporter.format_regime_change(regime, position=None)
        assert "레짐" in text
        assert "트레일링" in text

    def test_low_vol_range_no_position(self, reporter):
        regime = self._make_regime_state("low_vol_range", atr_ratio=0.5)
        text = reporter.format_regime_change(regime, position=None)
        assert "신규 진입 자제" in text

    def test_transition_no_position(self, reporter):
        regime = self._make_regime_state("transition", atr_ratio=1.0)
        text = reporter.format_regime_change(regime, position=None)
        assert "50%" in text

    def test_high_vol_with_long_position(self, reporter, make_position):
        regime = self._make_regime_state("high_vol_trend")
        pos = make_position()
        text = reporter.format_regime_change(regime, position=pos)
        assert "롱" in text
        assert "변동성 확대" in text

    def test_low_vol_with_short_position(self, reporter, make_position):
        from core.types import Side
        regime = self._make_regime_state("low_vol_range")
        pos = make_position(side=Side.SHORT)
        text = reporter.format_regime_change(regime, position=pos)
        assert "숏" in text
        assert "횡보장" in text

    def test_transition_with_position(self, reporter, make_position):
        regime = self._make_regime_state("transition")
        pos = make_position()
        text = reporter.format_regime_change(regime, position=pos)
        assert "전환" in text

    def test_atr_ratio_included(self, reporter):
        regime = self._make_regime_state("high_vol_trend", atr_ratio=1.75)
        text = reporter.format_regime_change(regime)
        assert "1.75" in text

    def test_btc_above_200ema_shown(self, reporter):
        regime = self._make_regime_state(btc_above=True)
        text = reporter.format_regime_change(regime)
        assert "위" in text

    def test_btc_below_200ema_shown(self, reporter):
        regime = self._make_regime_state(btc_above=False)
        text = reporter.format_regime_change(regime)
        assert "아래" in text


class TestFormatPositionEvent:
    def _make_event(self, event_type="bullish_signal", pnl_percent=5.0, severity="info"):
        from strategy.position_monitor import PositionEvent
        return PositionEvent(
            event_type=event_type,
            position_id=1,
            symbol="BTCUSDT",
            message="상승 신호 감지",
            severity=severity,
            pnl_percent=pnl_percent,
        )

    def test_bullish_signal_event(self, reporter, make_position):
        event = self._make_event(event_type="bullish_signal", pnl_percent=5.0)
        pos = make_position()
        text = reporter.format_position_event(event, pos)
        assert "BTCUSDT" in text
        assert "상승 신호" in text

    def test_bearish_signal_event(self, reporter, make_position):
        event = self._make_event(event_type="bearish_signal", pnl_percent=-5.0)
        pos = make_position()
        text = reporter.format_position_event(event, pos)
        assert "하락 신호" in text

    def test_critical_severity_shows_warning(self, reporter, make_position):
        event = self._make_event(severity="critical", pnl_percent=-15.0)
        pos = make_position()
        text = reporter.format_position_event(event, pos)
        assert "즉시 확인" in text

    def test_sell_recommendation(self, reporter, make_position):
        event = self._make_event(event_type="sell_recommendation", pnl_percent=10.0)
        pos = make_position()
        text = reporter.format_position_event(event, pos)
        assert "매도 추천" in text

    def test_hold_recommendation(self, reporter, make_position):
        event = self._make_event(event_type="hold_recommendation", pnl_percent=2.0)
        pos = make_position()
        text = reporter.format_position_event(event, pos)
        assert "홀딩 추천" in text

    def test_pnl_percent_displayed(self, reporter, make_position):
        event = self._make_event(pnl_percent=12.5)
        pos = make_position()
        text = reporter.format_position_event(event, pos)
        assert "+12.5%" in text


class TestFormatExitSignalV2Extended:
    """Extended tests for format_exit_signal_v2."""

    def _make_exit_signal(self, signal_type="sl_warning", pnl=-8.0, severity="warning"):
        from execution.exit_manager import ExitSignal
        return ExitSignal(
            signal_type=signal_type,
            position_id=1,
            symbol="BTCUSDT",
            message="손절가까지 2.5% 남음",
            severity=severity,
            pnl_percent=pnl,
            suggested_action="즉시 청산하세요.",
        )

    def test_partial_tp(self, reporter, make_position):
        sig = self._make_exit_signal(signal_type="partial_tp", pnl=5.0, severity="info")
        text = reporter.format_exit_signal_v2(sig, position=make_position())
        assert "1차 익절" in text

    def test_trailing_stop(self, reporter, make_position):
        sig = self._make_exit_signal(signal_type="trailing_stop", pnl=3.0, severity="warning")
        text = reporter.format_exit_signal_v2(sig, position=make_position())
        assert "트레일링" in text

    def test_time_stop(self, reporter, make_position):
        sig = self._make_exit_signal(signal_type="time_stop", pnl=-1.0, severity="info")
        text = reporter.format_exit_signal_v2(sig, position=make_position())
        assert "시간 스탑" in text

    def test_critical_shows_urgent_warning(self, reporter, make_position):
        sig = self._make_exit_signal(signal_type="sl_warning", pnl=-15.0, severity="critical")
        text = reporter.format_exit_signal_v2(sig, position=make_position())
        assert "즉시 확인" in text

    def test_no_position(self, reporter):
        sig = self._make_exit_signal()
        text = reporter.format_exit_signal_v2(sig, position=None)
        assert "BTCUSDT" in text
        # position section should not be rendered
        assert "마진" not in text

    def test_with_margin_shows_usdt_pnl(self, reporter, make_position):
        sig = self._make_exit_signal(pnl=-8.0)
        pos = make_position(margin_usdt=500.0)
        text = reporter.format_exit_signal_v2(sig, position=pos)
        # 500 * -8% = -40 USDT
        assert "-40.0 USDT" in text
