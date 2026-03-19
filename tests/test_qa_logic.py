"""
QA edge-case tests for ManualPosition monitoring logic.
Targets: position_monitor.py, position_manager.py, types.py
"""
import pytest
import pandas as pd
from datetime import datetime, timezone
from core.types import ManualPosition, Side
from strategy.position_monitor import PositionMonitor
from conversation.position_manager import PositionManager


# ── helpers ──────────────────────────────────────────────────────────────

def _pos(side=Side.LONG, entry=67500.0, leverage=10.0, pid=1):
    return ManualPosition(
        id=pid, chat_id="123", symbol="BTCUSDT", side=side,
        entry_price=entry, leverage=leverage,
        created_at=datetime.now(timezone.utc),
    )


def _df(close=68000, open_price=67600, rsi=50, adx=25, volume_ratio=1.0,
        ema_20=67800, ema_50=67000, **overrides):
    """Minimal 2-row DataFrame."""
    data = {
        "open": [67000, open_price],
        "close": [67500, close],
        "rsi": [50, rsi],
        "adx": [20, adx],
        "macd_hist": [-0.01, 0.01],
        "volume_ratio": [1.0, volume_ratio],
        "ema_20": [67700, ema_20],
        "ema_50": [67000, ema_50],
        "atr": [500, 500],
        "ema_golden_cross": [False, False],
        "ema_death_cross": [False, False],
        "rsi_cross_up": [False, False],
        "rsi_cross_down": [False, False],
        "macd_hist_cross_up": [False, False],
        "macd_hist_cross_down": [False, False],
        "bb_lower": [66000, 66000],
        "bb_upper": [69000, 69000],
    }
    for k, v in overrides.items():
        if k in data:
            data[k][1] = v
    return pd.DataFrame(data)


# ════════════════════════════════════════════════════════════════════════
# 1. PnL 계산
# ════════════════════════════════════════════════════════════════════════

class TestPnLCalculation:
    def setup_method(self):
        self.m = PositionMonitor()

    def test_long_profit(self):
        pos = _pos(Side.LONG, entry=100, leverage=10)
        # price went up 5% → leveraged PnL = 50%
        assert self.m._calc_pnl_pct(pos, 105) == pytest.approx(50.0)

    def test_long_loss(self):
        pos = _pos(Side.LONG, entry=100, leverage=10)
        assert self.m._calc_pnl_pct(pos, 95) == pytest.approx(-50.0)

    def test_short_profit(self):
        pos = _pos(Side.SHORT, entry=100, leverage=10)
        # price went down 5% → SHORT profit = 50%
        assert self.m._calc_pnl_pct(pos, 95) == pytest.approx(50.0)

    def test_short_loss(self):
        pos = _pos(Side.SHORT, entry=100, leverage=10)
        assert self.m._calc_pnl_pct(pos, 105) == pytest.approx(-50.0)

    def test_leverage_1(self):
        pos = _pos(Side.LONG, entry=100, leverage=1)
        assert self.m._calc_pnl_pct(pos, 110) == pytest.approx(10.0)

    def test_entry_price_zero_division(self):
        """FIXED: entry_price=0 → returns 0.0 instead of ZeroDivisionError."""
        pos = _pos(Side.LONG, entry=0, leverage=10)
        assert self.m._calc_pnl_pct(pos, 100) == 0.0


# ════════════════════════════════════════════════════════════════════════
# 2. Liquidation distance
# ════════════════════════════════════════════════════════════════════════

class TestLiquidationDistance:
    def setup_method(self):
        self.m = PositionMonitor()

    def test_long_leverage_1_liq_price_is_zero(self):
        """leverage=1 → liq_price=0 for LONG.
        Distance denominator = entry - liq = entry - 0 = entry.
        Should NOT divide by zero, should return a large value.
        """
        pos = _pos(Side.LONG, entry=100, leverage=1)
        dist = self.m._calc_liquidation_distance(pos, 100)
        # liq_price = 100*(1-1/1) = 0.  dist = (100-0)/(100-0)*100 = 100
        assert dist == pytest.approx(100.0)

    def test_short_leverage_1_liq_price_is_2x_entry(self):
        """leverage=1 → liq_price = entry*(1+1) = 2*entry for SHORT."""
        pos = _pos(Side.SHORT, entry=100, leverage=1)
        dist = self.m._calc_liquidation_distance(pos, 100)
        # liq_price = 200. dist = (200-100)/(200-100)*100 = 100
        assert dist == pytest.approx(100.0)

    def test_long_high_leverage(self):
        """leverage=125 → liq_price = entry*(1-1/125) very close to entry."""
        pos = _pos(Side.LONG, entry=100, leverage=125)
        dist = self.m._calc_liquidation_distance(pos, 100)
        # liq = 100*(1-0.008) = 99.2.  dist = (100-99.2)/(100-99.2)*100 = 100
        assert dist == pytest.approx(100.0)

    def test_short_high_leverage(self):
        pos = _pos(Side.SHORT, entry=100, leverage=125)
        dist = self.m._calc_liquidation_distance(pos, 100)
        assert dist == pytest.approx(100.0)

    def test_long_price_below_liq(self):
        """If current_price drops below liq, should return 0."""
        pos = _pos(Side.LONG, entry=100, leverage=10)
        # liq = 100*(1-0.1) = 90
        dist = self.m._calc_liquidation_distance(pos, 85)
        assert dist == 0

    def test_short_price_above_liq(self):
        pos = _pos(Side.SHORT, entry=100, leverage=10)
        # liq = 100*(1+0.1) = 110
        dist = self.m._calc_liquidation_distance(pos, 115)
        assert dist == 0

    def test_long_liq_distance_meaningful_value(self):
        """LONG entry=100, leverage=10 → liq=90. price=95 → half margin consumed."""
        pos = _pos(Side.LONG, entry=100, leverage=10)
        # liq=90, price=95, dist=(95-90)/(100-90)*100 = 50
        dist = self.m._calc_liquidation_distance(pos, 95)
        assert dist == pytest.approx(50.0)

    def test_entry_price_zero_liq_long(self):
        """FIXED: entry_price=0 → returns 0.0 instead of ZeroDivisionError."""
        pos = _pos(Side.LONG, entry=0, leverage=10)
        assert self.m._calc_liquidation_distance(pos, 100) == 0.0

    def test_entry_price_zero_liq_short(self):
        """BUG: entry_price=0, SHORT → liq_price=0, current(100)>=0 → returns 0.
        Silently claims 'already liquidated' instead of raising an error.
        This is a silent data corruption bug — no crash, wrong result.
        """
        pos = _pos(Side.SHORT, entry=0, leverage=10)
        # Does NOT raise — returns 0 silently (thinks liquidated)
        dist = self.m._calc_liquidation_distance(pos, 100)
        assert dist == 0  # wrong but current behavior


# ════════════════════════════════════════════════════════════════════════
# 3. _check_sell: SHORT position with pnl >= 30
# ════════════════════════════════════════════════════════════════════════

class TestCheckSell:
    def setup_method(self):
        self.m = PositionMonitor()

    def test_sell_recommendation_fires_for_short_in_profit(self):
        """
        BUG CHECK: pnl_pct >= 30 triggers sell for ANY side.
        For SHORT in profit (price dropped), pnl_pct = +50%.
        This correctly means "take profit" — NOT a bug, it's "매도 추천" meaning "close position".
        """
        pos = _pos(Side.SHORT, entry=100, leverage=10)
        current = _df(close=95).iloc[-1]
        pnl_pct = self.m._calc_pnl_pct(pos, 95)  # +50%
        reasons = self.m._check_sell(pos, current, pnl_pct, 0.0)
        assert len(reasons) > 0
        assert "PnL" in reasons[0]

    def test_sell_recommendation_does_not_fire_for_short_in_loss(self):
        """SHORT loss pnl = -50%, should NOT trigger sell."""
        pos = _pos(Side.SHORT, entry=100, leverage=10)
        current = _df(close=105).iloc[-1]
        pnl_pct = self.m._calc_pnl_pct(pos, 105)  # -50%
        reasons = self.m._check_sell(pos, current, pnl_pct, 0.0)
        assert len(reasons) == 0


# ════════════════════════════════════════════════════════════════════════
# 4. _check_buy: leveraged PnL threshold
# ════════════════════════════════════════════════════════════════════════

class TestCheckBuy:
    def setup_method(self):
        self.m = PositionMonitor()

    def test_buy_threshold_is_leveraged_pnl(self):
        """
        BUG CHECK: pnl_pct < -5 is checked.
        With leverage=10 and a 0.6% price drop, leveraged PnL = -6%.
        This means -5% threshold is on leveraged PnL, which for high leverage
        triggers on very small price moves. Verify this is intentional.
        """
        pos = _pos(Side.LONG, entry=100, leverage=10)
        pnl = self.m._calc_pnl_pct(pos, 99.4)  # -6% leveraged
        assert pnl < -5
        # With RSI oversold + MACD, buy should trigger
        df = _df(close=99.4, rsi=28)
        df.at[1, "macd_hist_cross_up"] = True
        current = df.iloc[-1]
        reasons = self.m._check_buy(pos, current, df, pnl)
        assert len(reasons) > 0

    def test_buy_not_triggered_for_short_loss_with_wrong_indicators(self):
        """SHORT loss + bullish indicators (RSI low) should NOT trigger buy."""
        pos = _pos(Side.SHORT, entry=100, leverage=10)
        pnl = self.m._calc_pnl_pct(pos, 105)  # -50%
        df = _df(close=105, rsi=28)  # RSI low = bullish, wrong for SHORT averaging
        df.at[1, "macd_hist_cross_up"] = True
        current = df.iloc[-1]
        reasons = self.m._check_buy(pos, current, df, pnl)
        # SHORT averaging needs RSI >= 70, not <= 30
        assert len(reasons) == 0

    def test_buy_triggered_for_short_loss_correct_indicators(self):
        """SHORT loss + bearish indicators (RSI high) SHOULD trigger buy."""
        pos = _pos(Side.SHORT, entry=100, leverage=10)
        pnl = self.m._calc_pnl_pct(pos, 105)  # -50%
        df = _df(close=105, rsi=72)
        df.at[1, "macd_hist_cross_down"] = True
        current = df.iloc[-1]
        reasons = self.m._check_buy(pos, current, df, pnl)
        assert len(reasons) > 0


# ════════════════════════════════════════════════════════════════════════
# 5. detect_events with empty/short DataFrame
# ════════════════════════════════════════════════════════════════════════

class TestEmptyDataFrame:
    def test_empty_df_returns_no_events(self):
        m = PositionMonitor()
        pos = _pos()
        events = m.detect_events(pos, pd.DataFrame(), current_price=68000, funding_rate=0)
        assert events == []

    def test_single_row_df_returns_no_events(self):
        m = PositionMonitor()
        pos = _pos()
        df = _df().iloc[:1]  # only 1 row
        events = m.detect_events(pos, df, current_price=68000, funding_rate=0)
        assert events == []


# ════════════════════════════════════════════════════════════════════════
# 6. Liquidation distance formula correctness deep test
# ════════════════════════════════════════════════════════════════════════

class TestLiqDistanceFormula:
    """
    The formula normalizes distance as percentage of margin, not price.
    At entry price, distance should always be 100% (full margin remaining).
    At liq price, distance should be 0%.
    """
    def setup_method(self):
        self.m = PositionMonitor()

    def test_long_at_entry_distance_is_100(self):
        pos = _pos(Side.LONG, entry=100, leverage=10)
        assert self.m._calc_liquidation_distance(pos, 100) == pytest.approx(100.0)

    def test_short_at_entry_distance_is_100(self):
        pos = _pos(Side.SHORT, entry=100, leverage=10)
        assert self.m._calc_liquidation_distance(pos, 100) == pytest.approx(100.0)

    def test_long_at_liq_price_distance_is_0(self):
        pos = _pos(Side.LONG, entry=100, leverage=10)
        # liq = 90
        assert self.m._calc_liquidation_distance(pos, 90) == pytest.approx(0.0)

    def test_short_at_liq_price_distance_is_0(self):
        pos = _pos(Side.SHORT, entry=100, leverage=10)
        # liq = 110
        assert self.m._calc_liquidation_distance(pos, 110) == pytest.approx(0.0)

    def test_long_price_above_entry_distance_over_100(self):
        """BUG CHECK: If price moves favorably, distance > 100%. Is that intended?"""
        pos = _pos(Side.LONG, entry=100, leverage=10)
        # liq=90, price=110 → dist = (110-90)/(100-90)*100 = 200%
        dist = self.m._calc_liquidation_distance(pos, 110)
        assert dist == pytest.approx(200.0)
        # This is not capped. Values >100 mean margin is more than initial.

    def test_short_price_below_entry_distance_over_100(self):
        pos = _pos(Side.SHORT, entry=100, leverage=10)
        # liq=110, price=90 → dist = (110-90)/(110-100)*100 = 200%
        dist = self.m._calc_liquidation_distance(pos, 90)
        assert dist == pytest.approx(200.0)


# ════════════════════════════════════════════════════════════════════════
# 7. PositionManager edge cases
# ════════════════════════════════════════════════════════════════════════

class TestPositionManagerEdgeCases:
    @pytest.fixture(autouse=True)
    def setup_pm(self, tmp_path):
        self.pm = PositionManager(db_path=str(tmp_path / "test.db"))

    def test_duplicate_symbol_allowed(self):
        """Same coin can be registered twice — verify this is the behavior."""
        self.pm.open_position("123", "BTCUSDT", Side.LONG, 67500, 10, stop_loss=66000)
        self.pm.open_position("123", "BTCUSDT", Side.SHORT, 68000, 5, stop_loss=69000)
        positions = self.pm.get_active_positions("123")
        assert len(positions) == 2

    def test_close_by_symbol_closes_only_first_duplicate(self):
        """
        FIXED: close_position_by_symbol now closes only the FIRST active position.
        If user has 2 BTC positions (LONG + SHORT), only the first gets closed.
        """
        self.pm.open_position("123", "BTCUSDT", Side.LONG, 67500, 10, stop_loss=66000)
        self.pm.open_position("123", "BTCUSDT", Side.SHORT, 68000, 5, stop_loss=69000)
        result = self.pm.close_position_by_symbol("123", "BTCUSDT")
        assert result is True
        positions = self.pm.get_active_positions("123")
        # Only first position closed, second remains
        assert len(positions) == 1

    def test_close_nonexistent_position(self):
        result = self.pm.close_position(999, "123")
        assert result is False

    def test_close_already_closed_position(self):
        pos = self.pm.open_position("123", "BTCUSDT", Side.LONG, 67500, 10, stop_loss=66000)
        self.pm.close_position(pos.id, "123")
        # Close again
        result = self.pm.close_position(pos.id, "123")
        assert result is False

    def test_close_wrong_chat_id(self):
        """Cannot close another user's position."""
        pos = self.pm.open_position("123", "BTCUSDT", Side.LONG, 67500, 10, stop_loss=66000)
        result = self.pm.close_position(pos.id, "456")
        assert result is False
        # Position still active for original user
        assert len(self.pm.get_active_positions("123")) == 1

    def test_zero_entry_price_rejected(self):
        """FIXED: entry_price=0 now raises ValueError."""
        with pytest.raises(ValueError, match="진입가는 0보다 커야 합니다"):
            self.pm.open_position("123", "BTCUSDT", Side.LONG, 0.0, 10, stop_loss=66000)

    def test_zero_leverage_rejected(self):
        """FIXED: leverage=0 now raises ValueError."""
        with pytest.raises(ValueError, match="레버리지는 1~125 사이여야 합니다"):
            self.pm.open_position("123", "BTCUSDT", Side.LONG, 67500, 0.0, stop_loss=66000)

    def test_negative_leverage_rejected(self):
        """FIXED: negative leverage now raises ValueError."""
        with pytest.raises(ValueError, match="레버리지는 1~125 사이여야 합니다"):
            self.pm.open_position("123", "BTCUSDT", Side.LONG, 67500, -5, stop_loss=66000)


# ════════════════════════════════════════════════════════════════════════
# 8. Zero/negative leverage in monitor calculations
# ════════════════════════════════════════════════════════════════════════

class TestZeroLeverage:
    def setup_method(self):
        self.m = PositionMonitor()

    def test_leverage_zero_pnl(self):
        """leverage=0 → pnl = 0 always (no crash, but nonsensical)."""
        pos = _pos(Side.LONG, entry=100, leverage=0)
        pnl = self.m._calc_pnl_pct(pos, 150)
        assert pnl == 0.0  # Not a crash, but misleading

    def test_leverage_zero_liq_distance_returns_zero(self):
        """FIXED: leverage=0 → returns 0.0 instead of ZeroDivisionError."""
        pos = _pos(Side.LONG, entry=100, leverage=0)
        assert self.m._calc_liquidation_distance(pos, 100) == 0.0

    def test_negative_leverage_liq_distance_nonsense(self):
        """Negative leverage → liq_price > entry for LONG (nonsense)."""
        pos = _pos(Side.LONG, entry=100, leverage=-5)
        # liq = 100*(1 - 1/(-5)) = 100*(1+0.2) = 120
        # current=100, 100 <= 120 → returns 0 (thinks already liquidated)
        dist = self.m._calc_liquidation_distance(pos, 100)
        assert dist == 0  # Nonsensical but no crash


# ════════════════════════════════════════════════════════════════════════
# 9. _check_sell funding rate edge case
# ════════════════════════════════════════════════════════════════════════

class TestFundingRate:
    def setup_method(self):
        self.m = PositionMonitor()

    def test_short_positive_funding_no_sell(self):
        """SHORT benefits from positive funding — should NOT trigger sell."""
        pos = _pos(Side.SHORT, entry=100, leverage=10)
        current = _df(close=100).iloc[-1]
        reasons = self.m._check_sell(pos, current, 5.0, 0.005)
        # Only triggers for SHORT if funding < -0.001
        assert not any("펀딩비" in r for r in reasons)

    def test_long_negative_funding_no_sell(self):
        """LONG benefits from negative funding — should NOT trigger sell."""
        pos = _pos(Side.LONG, entry=100, leverage=10)
        current = _df(close=100).iloc[-1]
        reasons = self.m._check_sell(pos, current, 5.0, -0.005)
        assert not any("펀딩비" in r for r in reasons)


# ════════════════════════════════════════════════════════════════════════
# 10. Full integration: detect_events with current_price=0
# ════════════════════════════════════════════════════════════════════════

class TestCurrentPriceZero:
    def test_current_price_zero_long(self):
        """current_price=0 → LONG pnl = -100%*leverage, liq check may break."""
        m = PositionMonitor()
        pos = _pos(Side.LONG, entry=100, leverage=10)
        df = _df(close=0)
        # Should not crash — price=0 means total loss
        events = m.detect_events(pos, df, current_price=0, funding_rate=0)
        # liq_price = 90, current=0 < 90 → liq_distance = 0 → position_check fires
        types = [e.event_type for e in events]
        assert "position_check" in types
