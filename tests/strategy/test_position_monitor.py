import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from core.types import ManualPosition, Side
from strategy.position_monitor import PositionMonitor, PositionEvent


def _make_position(symbol="BTCUSDT", side=Side.LONG, entry=67500, leverage=10):
    return ManualPosition(
        id=1, chat_id="123", symbol=symbol, side=side,
        entry_price=entry, leverage=leverage,
        created_at=datetime.now(timezone.utc),
    )


def _make_df(close=68000, prev_close=67500, rsi=50, adx=25, macd_hist=0.01,
             prev_macd_hist=-0.01, volume_ratio=1.0, ema_20=67800, ema_50=67000,
             funding_rate=0.0001, atr=500, open_price=67600):
    """2행 DataFrame (prev + current)."""
    data = {
        "open": [prev_close - 100, open_price],
        "close": [prev_close, close],
        "rsi": [50, rsi],
        "adx": [20, adx],
        "macd_hist": [prev_macd_hist, macd_hist],
        "volume_ratio": [1.0, volume_ratio],
        "ema_20": [67700, ema_20],
        "ema_50": [67000, ema_50],
        "atr": [500, atr],
        "ema_golden_cross": [False, False],
        "ema_death_cross": [False, False],
        "rsi_cross_up": [False, False],
        "rsi_cross_down": [False, False],
        "macd_hist_cross_up": [False, False],
        "macd_hist_cross_down": [False, False],
        "bb_lower": [66000, 66000],
        "bb_upper": [69000, 69000],
    }
    return pd.DataFrame(data)


def test_no_events_normal_conditions():
    monitor = PositionMonitor()
    pos = _make_position()
    # ema_20 < ema_50 이면 LONG 정배열 아님 → hold 추천 안 뜸, adx=15 → 추세 약함
    df = _make_df(ema_20=66500, ema_50=67000, adx=15)
    events = monitor.detect_events(pos, df, current_price=67600, funding_rate=0.0001)
    assert len(events) == 0


def test_liquidation_warning():
    monitor = PositionMonitor()
    pos = _make_position(leverage=20, entry=67500)
    events = monitor.detect_events(pos, _make_df(close=64500), current_price=64500, funding_rate=0)
    event_types = [e.event_type for e in events]
    assert "position_check" in event_types


def test_bullish_signal_detected():
    monitor = PositionMonitor()
    pos = _make_position()
    df = _make_df(rsi=35, volume_ratio=2.5)
    df.at[1, "ema_golden_cross"] = True
    df.at[1, "macd_hist_cross_up"] = True
    events = monitor.detect_events(pos, df, current_price=68000, funding_rate=0)
    event_types = [e.event_type for e in events]
    assert "bullish_signal" in event_types


def test_bearish_signal_detected():
    monitor = PositionMonitor()
    pos = _make_position()
    df = _make_df(rsi=72)
    df.at[1, "ema_death_cross"] = True
    df.at[1, "macd_hist_cross_down"] = True
    events = monitor.detect_events(pos, df, current_price=67000, funding_rate=0)
    event_types = [e.event_type for e in events]
    assert "bearish_signal" in event_types


def test_sell_recommendation_long_high_pnl():
    monitor = PositionMonitor()
    pos = _make_position(leverage=10, entry=67500)
    df = _make_df(close=70875, rsi=75)
    events = monitor.detect_events(pos, df, current_price=70875, funding_rate=0)
    event_types = [e.event_type for e in events]
    assert "sell_recommendation" in event_types


def test_buy_recommendation_long_dip():
    monitor = PositionMonitor()
    pos = _make_position(side=Side.LONG, leverage=5, entry=67500)
    df = _make_df(close=66000, rsi=28)
    df.at[1, "macd_hist_cross_up"] = True
    events = monitor.detect_events(pos, df, current_price=66000, funding_rate=0)
    event_types = [e.event_type for e in events]
    assert "buy_recommendation" in event_types


def test_cooldown_prevents_duplicate():
    monitor = PositionMonitor()
    pos = _make_position(leverage=20, entry=67500)
    df = _make_df(close=64500)
    # 첫 번째: 이벤트 발생
    events1 = monitor.detect_events(pos, df, current_price=64500, funding_rate=0)
    assert len(events1) > 0
    # 두 번째: 쿨다운으로 차단
    events2 = monitor.detect_events(pos, df, current_price=64500, funding_rate=0)
    assert len(events2) == 0


def test_clear_position_resets_cooldown():
    monitor = PositionMonitor()
    pos = _make_position(leverage=20, entry=67500)
    df = _make_df(close=64500)
    events1 = monitor.detect_events(pos, df, current_price=64500, funding_rate=0)
    assert len(events1) > 0
    # clear 후 다시 이벤트 발생
    monitor.clear_position(pos.id)
    events2 = monitor.detect_events(pos, df, current_price=64500, funding_rate=0)
    assert len(events2) > 0
