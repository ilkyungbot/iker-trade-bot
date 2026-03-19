import pytest
import pandas as pd
from datetime import datetime, timezone
from core.types import ManualPosition, Side, FundingRate, OpenInterest
from strategy.position_monitor import PositionMonitorV2

def _make_position(sl=66000, tp=70000):
    return ManualPosition(
        id=1, chat_id="123", symbol="BTCUSDT", side=Side.LONG,
        entry_price=67500, leverage=5, created_at=datetime.now(timezone.utc),
        stop_loss=sl, take_profit=tp, margin_usdt=500,
    )

def _make_df(n=60):
    data = {"close": [67500 + i*10 for i in range(n)], "atr": [500]*n,
            "ema_20": [67000]*n, "ema_50": [66500]*n, "adx": [25]*n,
            "rsi": [50]*n, "macd_hist": [0.01]*n, "volume_ratio": [1.0]*n,
            "ema_golden_cross": [False]*n, "ema_death_cross": [False]*n,
            "rsi_cross_up": [False]*n, "rsi_cross_down": [False]*n,
            "macd_hist_cross_up": [False]*n, "macd_hist_cross_down": [False]*n,
            "bb_lower": [66000]*n, "bb_upper": [69000]*n, "open": [67400]*n}
    return pd.DataFrame(data)

def test_check_position_returns_expected_keys():
    monitor = PositionMonitorV2()
    pos = _make_position()
    df = _make_df()
    result = monitor.check_position(pos, df, current_price=68000, atr=500)
    assert "exit_signals" in result
    assert "edge_signals" in result
    assert "regime" in result
    assert "pnl_pct" in result

def test_check_position_with_funding_and_oi():
    monitor = PositionMonitorV2()
    pos = _make_position()
    df = _make_df()
    now = datetime.now(timezone.utc)
    funding = [FundingRate(now, "BTCUSDT", 0.001)]  # extreme
    oi = [OpenInterest(now, "BTCUSDT", 1000000), OpenInterest(now, "BTCUSDT", 900000)]  # -10% drop
    result = monitor.check_position(pos, df, current_price=68000, atr=500, funding_rates=funding, oi_data=oi)
    assert len(result["edge_signals"]) > 0

def test_clear_position():
    monitor = PositionMonitorV2()
    pos = _make_position()
    df = _make_df()
    monitor.check_position(pos, df, current_price=68000, atr=500)
    monitor.clear_position(1)
    # Should not crash
