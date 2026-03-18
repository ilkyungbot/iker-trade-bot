import pytest
from datetime import datetime, timezone
from conversation.position_manager import PositionManager
from core.types import Side


@pytest.fixture
def pm(tmp_path):
    return PositionManager(db_path=str(tmp_path / "test.db"))


def test_open_position(pm):
    pos = pm.open_position("123", "BTCUSDT", Side.LONG, 67500.0, 10.0)
    assert pos.id is not None
    assert pos.symbol == "BTCUSDT"
    assert pos.side == Side.LONG
    assert pos.entry_price == 67500.0
    assert pos.leverage == 10.0
    assert pos.is_active is True


def test_get_active_positions(pm):
    pm.open_position("123", "BTCUSDT", Side.LONG, 67500.0, 10.0)
    pm.open_position("123", "ETHUSDT", Side.SHORT, 3800.0, 5.0)
    positions = pm.get_active_positions("123")
    assert len(positions) == 2


def test_close_position(pm):
    pos = pm.open_position("123", "BTCUSDT", Side.LONG, 67500.0, 10.0)
    result = pm.close_position(pos.id, "123")
    assert result is True
    assert len(pm.get_active_positions("123")) == 0


def test_close_by_symbol(pm):
    pm.open_position("123", "BTCUSDT", Side.LONG, 67500.0, 10.0)
    result = pm.close_position_by_symbol("123", "BTCUSDT")
    assert result is True
    assert len(pm.get_active_positions("123")) == 0


def test_get_all_active_positions(pm):
    pm.open_position("123", "BTCUSDT", Side.LONG, 67500.0, 10.0)
    pm.open_position("456", "ETHUSDT", Side.SHORT, 3800.0, 5.0)
    all_positions = pm.get_all_active_positions()
    assert len(all_positions) == 2
