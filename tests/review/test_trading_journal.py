"""TDD tests for TradingJournal."""

import sqlite3
import tempfile
import os
from datetime import datetime, timezone, timedelta

import pytest

from core.types import ManualPosition, Side
from review.trading_journal import TradingJournal, JournalEntry


@pytest.fixture
def journal(tmp_path):
    db_path = str(tmp_path / "test_journal.db")
    return TradingJournal(db_path=db_path)


def _make_position(
    id_=1,
    chat_id="chat_123",
    symbol="BTCUSDT",
    side=Side.LONG,
    entry_price=50000.0,
    leverage=10.0,
    stop_loss=49000.0,
    take_profit=52000.0,
    margin_usdt=100.0,
    entry_reason="breakout",
) -> ManualPosition:
    return ManualPosition(
        id=id_,
        chat_id=chat_id,
        symbol=symbol,
        side=side,
        entry_price=entry_price,
        leverage=leverage,
        created_at=datetime.now(timezone.utc),
        stop_loss=stop_loss,
        take_profit=take_profit,
        margin_usdt=margin_usdt,
        entry_reason=entry_reason,
    )


class TestRecordEntry:
    def test_record_entry(self, journal):
        pos = _make_position()
        entry_id = journal.record_entry(pos, regime="high_vol_trend")

        assert entry_id is not None
        assert entry_id >= 1

        # Verify data persisted
        with sqlite3.connect(journal._db_path) as conn:
            row = conn.execute(
                "SELECT position_id, chat_id, symbol, side, leverage, entry_price, stop_loss, take_profit, margin_usdt, entry_reason, regime FROM trading_journal WHERE id = ?",
                (entry_id,),
            ).fetchone()

        assert row is not None
        assert row[0] == 1  # position_id
        assert row[1] == "chat_123"
        assert row[2] == "BTCUSDT"
        assert row[3] == "long"
        assert row[4] == 10.0
        assert row[5] == 50000.0
        assert row[6] == 49000.0
        assert row[7] == 52000.0
        assert row[8] == 100.0
        assert row[9] == "breakout"
        assert row[10] == "high_vol_trend"


class TestRecordExit:
    def test_record_exit_long_profit(self, journal):
        """Long position exits higher -> positive PnL."""
        pos = _make_position(side=Side.LONG, entry_price=50000.0, leverage=10.0, margin_usdt=100.0)
        journal.record_entry(pos)

        journal.record_exit(position_id=1, exit_price=51000.0, exit_reason="take_profit")

        with sqlite3.connect(journal._db_path) as conn:
            row = conn.execute(
                "SELECT exit_price, realized_pnl_pct, realized_pnl_usdt, exit_reason FROM trading_journal WHERE position_id = 1",
            ).fetchone()

        assert row[0] == 51000.0
        assert row[1] == 20.0  # (51000-50000)/50000 * 100 * 10 = 20%
        assert row[2] == 20.0  # 100 * 20 / 100 = 20 USDT
        assert row[3] == "take_profit"

    def test_record_exit_short_profit(self, journal):
        """Short position exits lower -> positive PnL."""
        pos = _make_position(id_=2, side=Side.SHORT, entry_price=50000.0, leverage=5.0, margin_usdt=200.0)
        journal.record_entry(pos)

        journal.record_exit(position_id=2, exit_price=48000.0, exit_reason="take_profit")

        with sqlite3.connect(journal._db_path) as conn:
            row = conn.execute(
                "SELECT realized_pnl_pct, realized_pnl_usdt FROM trading_journal WHERE position_id = 2",
            ).fetchone()

        # short: -(48000-50000)/50000 * 100 * 5 = 20%
        assert row[0] == 20.0
        assert row[1] == 40.0  # 200 * 20/100

    def test_record_exit_with_margin_usdt(self, journal):
        """PnL USDT is calculated from margin."""
        pos = _make_position(id_=3, side=Side.LONG, entry_price=40000.0, leverage=20.0, margin_usdt=50.0)
        journal.record_entry(pos)

        # 5% price move * 20x = 100% PnL
        journal.record_exit(position_id=3, exit_price=42000.0, exit_reason="manual")

        with sqlite3.connect(journal._db_path) as conn:
            row = conn.execute(
                "SELECT realized_pnl_pct, realized_pnl_usdt FROM trading_journal WHERE position_id = 3",
            ).fetchone()

        assert row[0] == 100.0  # (42000-40000)/40000 * 100 * 20
        assert row[1] == 50.0   # 50 * 100 / 100

    def test_record_exit_nonexistent_position(self, journal):
        """Exit for non-existent position should be a no-op."""
        journal.record_exit(position_id=999, exit_price=50000.0)
        # Should not raise


class TestWeeklyReport:
    def test_weekly_report_empty(self, journal):
        report = journal.weekly_report("chat_123")
        assert report["total_trades"] == 0
        assert report["win_rate"] == 0
        assert report["total_pnl_usdt"] == 0
        assert report["max_consecutive_loss"] == 0
        assert report["by_regime"] == {}

    def test_weekly_report_with_data(self, journal):
        # Insert 3 closed trades directly
        now = datetime.now(timezone.utc)
        with sqlite3.connect(journal._db_path) as conn:
            for i, (pnl_pct, pnl_usdt) in enumerate([(10.0, 10.0), (-5.0, -5.0), (15.0, 15.0)]):
                conn.execute(
                    "INSERT INTO trading_journal (position_id, chat_id, symbol, side, leverage, entry_price, entry_time, exit_price, exit_time, realized_pnl_pct, realized_pnl_usdt, exit_reason, regime) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (i + 1, "chat_123", "BTCUSDT", "long", 10.0, 50000.0,
                     (now - timedelta(days=2)).isoformat(), 51000.0,
                     (now - timedelta(days=1)).isoformat(),
                     pnl_pct, pnl_usdt, "manual", "high_vol_trend"),
                )

        report = journal.weekly_report("chat_123")
        assert report["total_trades"] == 3
        assert report["win_rate"] == 66.7  # 2/3
        assert report["total_pnl_usdt"] == 20.0
        assert report["max_consecutive_loss"] == 1

    def test_weekly_report_by_regime(self, journal):
        now = datetime.now(timezone.utc)
        with sqlite3.connect(journal._db_path) as conn:
            # 2 trades in high_vol_trend (1 win, 1 loss)
            conn.execute(
                "INSERT INTO trading_journal (position_id, chat_id, symbol, side, leverage, entry_price, entry_time, exit_price, exit_time, realized_pnl_pct, realized_pnl_usdt, exit_reason, regime) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (1, "chat_123", "BTCUSDT", "long", 10.0, 50000.0,
                 (now - timedelta(days=3)).isoformat(), 51000.0,
                 (now - timedelta(days=2)).isoformat(), 20.0, 20.0, "tp", "high_vol_trend"),
            )
            conn.execute(
                "INSERT INTO trading_journal (position_id, chat_id, symbol, side, leverage, entry_price, entry_time, exit_price, exit_time, realized_pnl_pct, realized_pnl_usdt, exit_reason, regime) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (2, "chat_123", "BTCUSDT", "short", 5.0, 50000.0,
                 (now - timedelta(days=3)).isoformat(), 51000.0,
                 (now - timedelta(days=2)).isoformat(), -10.0, -10.0, "sl", "high_vol_trend"),
            )
            # 1 trade in low_vol_range
            conn.execute(
                "INSERT INTO trading_journal (position_id, chat_id, symbol, side, leverage, entry_price, entry_time, exit_price, exit_time, realized_pnl_pct, realized_pnl_usdt, exit_reason, regime) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (3, "chat_123", "ETHUSDT", "long", 10.0, 3000.0,
                 (now - timedelta(days=1)).isoformat(), 3100.0,
                 now.isoformat(), 5.0, 5.0, "manual", "low_vol_range"),
            )

        report = journal.weekly_report("chat_123")
        assert "high_vol_trend" in report["by_regime"]
        assert "low_vol_range" in report["by_regime"]

        hvt = report["by_regime"]["high_vol_trend"]
        assert hvt["trades"] == 2
        assert hvt["wins"] == 1
        assert hvt["pnl"] == 10.0

        lvr = report["by_regime"]["low_vol_range"]
        assert lvr["trades"] == 1
        assert lvr["wins"] == 1
        assert lvr["pnl"] == 5.0

    def test_weekly_report_excludes_old_trades(self, journal):
        """Trades older than the window should be excluded."""
        old_time = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        with sqlite3.connect(journal._db_path) as conn:
            conn.execute(
                "INSERT INTO trading_journal (position_id, chat_id, symbol, side, leverage, entry_price, entry_time, exit_price, exit_time, realized_pnl_pct, realized_pnl_usdt, exit_reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (1, "chat_123", "BTCUSDT", "long", 10.0, 50000.0,
                 old_time, 51000.0, old_time, 10.0, 10.0, "manual"),
            )

        report = journal.weekly_report("chat_123", days=7)
        assert report["total_trades"] == 0
