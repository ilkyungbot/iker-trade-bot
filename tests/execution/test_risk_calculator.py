"""Tests for risk_calculator module — written before implementation (TDD)."""

import pytest
from src.core.types import Side
from src.execution.risk_calculator import (
    add_slippage_buffer,
    calculate_rr_ratio,
    suggest_stop_loss,
    suggest_take_profit,
    validate_position,
)


# ── suggest_stop_loss ────────────────────────────────────────────────

class TestSuggestStopLoss:
    def test_long_stop_loss_below_entry(self):
        sl = suggest_stop_loss(entry_price=100.0, side=Side.LONG, atr=2.0, leverage=5)
        assert sl < 100.0

    def test_long_stop_loss_uses_atr_multiplier(self):
        sl = suggest_stop_loss(entry_price=100.0, side=Side.LONG, atr=2.0, leverage=5)
        # Base: 100 - 1.5*2 = 97.0, then round-number offset applied
        assert 96.0 < sl < 97.5

    def test_short_stop_loss_above_entry(self):
        sl = suggest_stop_loss(entry_price=100.0, side=Side.SHORT, atr=2.0, leverage=5)
        assert sl > 100.0

    def test_short_stop_loss_uses_atr_multiplier(self):
        sl = suggest_stop_loss(entry_price=100.0, side=Side.SHORT, atr=2.0, leverage=5)
        # Base: 100 + 1.5*2 = 103.0, then round-number offset applied
        assert 102.5 < sl < 104.0

    def test_round_number_offset_avoids_whole_numbers(self):
        # ATR chosen so base SL lands exactly on .000
        sl = suggest_stop_loss(entry_price=100.0, side=Side.LONG, atr=2.0, leverage=5)
        # Should NOT be exactly 97.000
        frac = sl % 1
        assert frac != 0.0


# ── suggest_take_profit ──────────────────────────────────────────────

class TestSuggestTakeProfit:
    def test_long_tp_above_entry(self):
        tp = suggest_take_profit(entry_price=100.0, side=Side.LONG, stop_loss=97.0)
        assert tp > 100.0

    def test_long_tp_rr_2_to_1(self):
        tp = suggest_take_profit(entry_price=100.0, side=Side.LONG, stop_loss=97.0)
        # risk=3, reward=6 → tp=106
        assert tp == pytest.approx(106.0)

    def test_short_tp_below_entry(self):
        tp = suggest_take_profit(entry_price=100.0, side=Side.SHORT, stop_loss=103.0)
        assert tp < 100.0

    def test_short_tp_rr_2_to_1(self):
        tp = suggest_take_profit(entry_price=100.0, side=Side.SHORT, stop_loss=103.0)
        # risk=3, reward=6 → tp=94
        assert tp == pytest.approx(94.0)


# ── calculate_rr_ratio ──────────────────────────────────────────────

class TestCalculateRRRatio:
    def test_basic_long_2_to_1(self):
        rr = calculate_rr_ratio(
            entry_price=100.0, stop_loss=97.0, take_profit=106.0, side=Side.LONG,
        )
        assert rr == pytest.approx(2.0)

    def test_basic_short_2_to_1(self):
        rr = calculate_rr_ratio(
            entry_price=100.0, stop_loss=103.0, take_profit=94.0, side=Side.SHORT,
        )
        assert rr == pytest.approx(2.0)

    def test_asymmetric_rr(self):
        rr = calculate_rr_ratio(
            entry_price=100.0, stop_loss=98.0, take_profit=110.0, side=Side.LONG,
        )
        assert rr == pytest.approx(5.0)


# ── validate_position ───────────────────────────────────────────────

class TestValidatePosition:
    def test_valid_position(self):
        result = validate_position(
            entry_price=100.0, stop_loss=97.0, take_profit=106.0,
            leverage=5, margin_usdt=1000.0, side=Side.LONG,
        )
        assert result["valid"] is True
        assert result["rr_ratio"] == pytest.approx(2.0)
        assert len(result["warnings"]) == 0

    def test_rr_too_low_warning(self):
        result = validate_position(
            entry_price=100.0, stop_loss=99.0, take_profit=101.0,
            leverage=5, margin_usdt=1000.0, side=Side.LONG,
        )
        assert any("R:R" in w for w in result["warnings"])

    def test_leveraged_loss_too_high_warning(self):
        # SL 10% away, leverage 5 → leveraged loss 50%
        result = validate_position(
            entry_price=100.0, stop_loss=90.0, take_profit=120.0,
            leverage=5, margin_usdt=1000.0, side=Side.LONG,
        )
        assert any("20%" in w for w in result["warnings"])

    def test_sl_too_close_warning(self):
        result = validate_position(
            entry_price=100.0, stop_loss=99.8, take_profit=100.6,
            leverage=5, margin_usdt=1000.0, side=Side.LONG,
        )
        assert any("too close" in w.lower() for w in result["warnings"])

    def test_sl_wrong_side_warning(self):
        # SL above entry for LONG → wrong side
        result = validate_position(
            entry_price=100.0, stop_loss=102.0, take_profit=106.0,
            leverage=5, margin_usdt=1000.0, side=Side.LONG,
        )
        assert result["valid"] is False
        assert any("wrong side" in w.lower() for w in result["warnings"])

    def test_max_loss_usdt(self):
        result = validate_position(
            entry_price=100.0, stop_loss=97.0, take_profit=106.0,
            leverage=5, margin_usdt=1000.0, side=Side.LONG,
        )
        # position_size = 1000 * 5 = 5000 USDT notional
        # loss_pct = 3% → max_loss = 5000 * 0.03 = 150
        assert result["max_loss_usdt"] == pytest.approx(150.0)

    def test_max_loss_pct(self):
        result = validate_position(
            entry_price=100.0, stop_loss=97.0, take_profit=106.0,
            leverage=5, margin_usdt=1000.0, side=Side.LONG,
        )
        # 150 / 1000 = 15%
        assert result["max_loss_pct"] == pytest.approx(15.0)


# ── add_slippage_buffer ─────────────────────────────────────────────

class TestAddSlippageBuffer:
    def test_long_buffer_lowers_sl(self):
        adjusted = add_slippage_buffer(stop_loss=97.0, side=Side.LONG)
        assert adjusted < 97.0

    def test_long_default_buffer(self):
        adjusted = add_slippage_buffer(stop_loss=97.0, side=Side.LONG, buffer_pct=0.005)
        assert adjusted == pytest.approx(97.0 * (1 - 0.005))

    def test_short_buffer_raises_sl(self):
        adjusted = add_slippage_buffer(stop_loss=103.0, side=Side.SHORT)
        assert adjusted > 103.0

    def test_short_default_buffer(self):
        adjusted = add_slippage_buffer(stop_loss=103.0, side=Side.SHORT, buffer_pct=0.005)
        assert adjusted == pytest.approx(103.0 * (1 + 0.005))
