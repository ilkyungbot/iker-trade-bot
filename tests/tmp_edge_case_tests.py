"""Edge case tests for QA review of v2 commits."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from core.types import Side, ConversationState
from execution.risk_calculator import (
    suggest_stop_loss,
    suggest_take_profit,
    calculate_rr_ratio,
    validate_position,
    add_slippage_buffer,
)
from strategy.market_regime import MarketRegimeClassifier, Regime
from conversation.position_manager import PositionManager
from conversation.state_machine import ConversationStateMachine


# ══════════════════════════════════════════════════════════════════════
# risk_calculator edge cases
# ══════════════════════════════════════════════════════════════════════

class TestRiskCalculatorEdgeCases:
    def test_suggest_sl_zero_atr(self):
        """ATR=0 → SL == entry (+ round offset only)."""
        sl = suggest_stop_loss(entry_price=100.0, side=Side.LONG, atr=0.0, leverage=5)
        # distance=0, base=100, offset=-0.12 → 99.88
        assert sl == pytest.approx(99.88)

    def test_suggest_sl_zero_price(self):
        """entry_price=0 → base goes negative for LONG."""
        sl = suggest_stop_loss(entry_price=0.0, side=Side.LONG, atr=1.0, leverage=5)
        # base = 0 - 1.5 = -1.5, offset → -1.62
        assert sl < 0

    def test_suggest_tp_sl_equals_entry(self):
        """SL == entry → risk=0 → TP == entry."""
        tp = suggest_take_profit(entry_price=100.0, side=Side.LONG, stop_loss=100.0)
        assert tp == 100.0

    def test_rr_ratio_zero_risk(self):
        """SL == entry → risk=0 → RR should be 0 (not crash)."""
        rr = calculate_rr_ratio(100.0, 100.0, 110.0, Side.LONG)
        assert rr == 0.0

    def test_validate_zero_margin(self):
        """margin_usdt=0 → max_loss_pct should be 0, not ZeroDivisionError."""
        result = validate_position(
            entry_price=100.0, stop_loss=97.0, take_profit=106.0,
            leverage=5, margin_usdt=0.0, side=Side.LONG,
        )
        assert result["max_loss_pct"] == 0.0
        assert result["max_loss_usdt"] == 0.0

    def test_validate_zero_entry_price(self):
        """entry_price=0 → sl_pct calculation should not crash (ZeroDivisionError)."""
        # entry_price=0 causes division by zero in sl_pct = abs(0 - sl) / 0
        with pytest.raises(ZeroDivisionError):
            validate_position(
                entry_price=0.0, stop_loss=-1.0, take_profit=1.0,
                leverage=5, margin_usdt=1000.0, side=Side.LONG,
            )

    def test_slippage_buffer_zero_sl(self):
        """SL=0 → buffer should return 0."""
        adj = add_slippage_buffer(0.0, Side.LONG)
        assert adj == 0.0

    def test_round_offset_is_constant_not_percentage(self):
        """_ROUND_OFFSET=0.12 is constant, so for BTC at 60000 it's negligible.
        This is a design concern — 0.12 on a 60000 price is meaningless."""
        sl = suggest_stop_loss(entry_price=60000.0, side=Side.LONG, atr=500.0, leverage=5)
        base = 60000.0 - 1.5 * 500.0  # 59250.0
        # offset is just -0.12 → 59249.88 — negligible on BTC price scale
        assert sl == pytest.approx(59249.88)


# ══════════════════════════════════════════════════════════════════════
# market_regime edge cases
# ══════════════════════════════════════════════════════════════════════

class TestMarketRegimeEdgeCases:
    @pytest.fixture
    def classifier(self):
        return MarketRegimeClassifier()

    def test_single_row_df(self, classifier):
        """Only 1 row of data → should not crash."""
        df = pd.DataFrame({"close": [100.0], "atr": [1.5]})
        result = classifier.classify(df)
        # short=1 row, long=1 row → ratio=1.0 → TRANSITION
        assert result.regime == Regime.TRANSITION

    def test_all_zero_atr(self, classifier):
        """All ATR=0 → atr_long=0 → should return ratio 1.0 (fallback)."""
        df = pd.DataFrame({"close": [100.0] * 60, "atr": [0.0] * 60})
        result = classifier.classify(df)
        assert result.atr_ratio == 1.0
        assert result.regime == Regime.TRANSITION

    def test_btc_df_insufficient_length(self, classifier):
        """BTC DataFrame < 200 rows → default to btc_above_200ema=True."""
        df = pd.DataFrame({"close": [100.0] * 60, "atr": [1.0] * 60})
        btc_df = pd.DataFrame({"close": [50.0] * 100})  # < 200
        result = classifier.classify(df, btc_df=btc_df)
        assert result.btc_above_200ema is True

    def test_empty_btc_df(self, classifier):
        """Empty BTC DataFrame → default to True."""
        df = pd.DataFrame({"close": [100.0] * 60, "atr": [1.0] * 60})
        btc_df = pd.DataFrame({"close": []})
        result = classifier.classify(df, btc_df=btc_df)
        assert result.btc_above_200ema is True

    def test_missing_atr_column_raises(self, classifier):
        """DataFrame without 'atr' → should raise KeyError."""
        df = pd.DataFrame({"close": [100.0] * 60})
        with pytest.raises(KeyError):
            classifier.classify(df)


# ══════════════════════════════════════════════════════════════════════
# position_manager edge cases
# ══════════════════════════════════════════════════════════════════════

class TestPositionManagerEdgeCases:
    @pytest.fixture
    def pm(self, tmp_path):
        db = str(tmp_path / "test.db")
        return PositionManager(db_path=db)

    def test_open_position_without_sl(self, pm):
        """기획: SL 없으면 등록 거부 → 실제: SL=None이어도 등록 성공 (기획 불일치)."""
        pos = pm.open_position("u1", "BTCUSDT", Side.LONG, 60000.0, 5.0)
        assert pos.id is not None
        assert pos.stop_loss is None  # 등록됨!

    def test_open_position_zero_price_rejected(self, pm):
        with pytest.raises(ValueError):
            pm.open_position("u1", "BTCUSDT", Side.LONG, 0.0, 5.0)

    def test_open_position_negative_price_rejected(self, pm):
        with pytest.raises(ValueError):
            pm.open_position("u1", "BTCUSDT", Side.LONG, -100.0, 5.0)

    def test_open_position_leverage_126_rejected(self, pm):
        with pytest.raises(ValueError):
            pm.open_position("u1", "BTCUSDT", Side.LONG, 60000.0, 126.0)

    def test_open_position_leverage_zero_rejected(self, pm):
        with pytest.raises(ValueError):
            pm.open_position("u1", "BTCUSDT", Side.LONG, 60000.0, 0.0)

    def test_new_fields_roundtrip(self, pm):
        """새 필드(SL/TP/margin/reason)가 DB에 저장되고 올바르게 조회되는지."""
        pos = pm.open_position(
            "u1", "ETHUSDT", Side.SHORT, 3000.0, 10.0,
            stop_loss=3100.0, take_profit=2800.0,
            margin_usdt=500.0, entry_reason="펀딩비 음수",
        )
        loaded = pm.get_active_positions("u1")
        assert len(loaded) == 1
        p = loaded[0]
        assert p.stop_loss == 3100.0
        assert p.take_profit == 2800.0
        assert p.margin_usdt == 500.0
        assert p.entry_reason == "펀딩비 음수"

    def test_close_by_symbol(self, pm):
        pm.open_position("u1", "BTCUSDT", Side.LONG, 60000.0, 5.0)
        assert pm.close_position_by_symbol("u1", "BTCUSDT") is True
        assert len(pm.get_active_positions("u1")) == 0

    def test_close_nonexistent(self, pm):
        assert pm.close_position_by_symbol("u1", "BTCUSDT") is False


# ══════════════════════════════════════════════════════════════════════
# state_machine edge cases
# ══════════════════════════════════════════════════════════════════════

class TestStateMachineEdgeCases:
    @pytest.fixture
    def sm(self, tmp_path):
        db = str(tmp_path / "test.db")
        return ConversationStateMachine(db_path=db)

    def test_user_exited_from_idle_returns_false(self, sm):
        """IDLE 상태에서 user_exited → False (전이 거부)."""
        result = sm.user_exited("u1")
        assert result is False

    def test_user_exited_from_monitoring_returns_true(self, sm):
        """MONITORING → IDLE 전이 허용."""
        session = sm.get_session("u1")
        session.state = ConversationState.MONITORING
        sm._save_session(session)

        result = sm.user_exited("u1")
        assert result is True
        assert sm.get_session("u1").state == ConversationState.IDLE

    def test_force_idle_from_any_state(self, sm):
        session = sm.get_session("u1")
        session.state = ConversationState.MONITORING
        sm._save_session(session)

        sm.force_idle("u1")
        assert sm.get_session("u1").state == ConversationState.IDLE

    def test_legacy_state_migrated_to_idle(self, sm):
        """DB에 'idle'/'monitoring' 외 상태가 있으면 IDLE로 마이그레이션."""
        import sqlite3
        with sqlite3.connect(sm._db_path) as conn:
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc).isoformat()
            conn.execute(
                "INSERT INTO user_sessions (chat_id, state, updated_at) VALUES (?, ?, ?)",
                ("u2", "awaiting_entry", now),
            )
        session = sm.get_session("u2")
        assert session.state == ConversationState.IDLE
