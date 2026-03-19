import sqlite3
import json
import logging
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from core.types import ManualPosition, Side

logger = logging.getLogger(__name__)

@dataclass
class JournalEntry:
    id: int | None
    position_id: int
    chat_id: str
    symbol: str
    side: str  # "long" or "short"
    leverage: float
    entry_price: float
    stop_loss: float | None
    take_profit: float | None
    margin_usdt: float | None
    entry_reason: str
    entry_time: datetime
    exit_price: float | None = None
    exit_time: datetime | None = None
    realized_pnl_pct: float | None = None
    realized_pnl_usdt: float | None = None
    exit_reason: str = ""  # "stop_loss", "take_profit", "manual", "time_stop"
    regime: str = ""  # "high_vol_trend", "low_vol_range", "transition"

class TradingJournal:
    def __init__(self, db_path: str = "signal_bot.db"):
        self._db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trading_journal (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    position_id INTEGER NOT NULL,
                    chat_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    leverage REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    stop_loss REAL,
                    take_profit REAL,
                    margin_usdt REAL,
                    entry_reason TEXT DEFAULT '',
                    entry_time TEXT NOT NULL,
                    exit_price REAL,
                    exit_time TEXT,
                    realized_pnl_pct REAL,
                    realized_pnl_usdt REAL,
                    exit_reason TEXT DEFAULT '',
                    regime TEXT DEFAULT ''
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_journal_chat_time ON trading_journal (chat_id, entry_time)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_journal_position ON trading_journal (position_id)")

    def record_entry(self, position: ManualPosition, regime: str = "") -> int:
        """Record a new position entry. Returns journal entry ID."""
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute(
                "INSERT INTO trading_journal (position_id, chat_id, symbol, side, leverage, entry_price, stop_loss, take_profit, margin_usdt, entry_reason, entry_time, regime) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (position.id, position.chat_id, position.symbol, position.side.value, position.leverage,
                 position.entry_price, position.stop_loss, position.take_profit, position.margin_usdt,
                 position.entry_reason, position.created_at.isoformat(), regime)
            )
            return cursor.lastrowid

    def record_exit(self, position_id: int, exit_price: float, exit_reason: str = "manual") -> None:
        """Record position exit with PnL calculation."""
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT entry_price, side, leverage, margin_usdt FROM trading_journal WHERE position_id = ? ORDER BY id DESC LIMIT 1",
                (position_id,)
            ).fetchone()
            if not row:
                return

            entry_price, side, leverage, margin_usdt = row
            pnl_pct = (exit_price - entry_price) / entry_price * 100
            if side == "short":
                pnl_pct = -pnl_pct
            pnl_pct *= leverage
            pnl_usdt = (margin_usdt * pnl_pct / 100) if margin_usdt else None

            conn.execute(
                "UPDATE trading_journal SET exit_price = ?, exit_time = ?, realized_pnl_pct = ?, realized_pnl_usdt = ?, exit_reason = ? WHERE position_id = ? AND exit_price IS NULL",
                (exit_price, datetime.now(timezone.utc).isoformat(), round(pnl_pct, 2), round(pnl_usdt, 2) if pnl_usdt else None, exit_reason, position_id)
            )

    def weekly_report(self, chat_id: str, days: int = 7) -> dict:
        """Generate weekly performance report."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                "SELECT side, leverage, realized_pnl_pct, realized_pnl_usdt, exit_reason, regime FROM trading_journal WHERE chat_id = ? AND exit_time IS NOT NULL AND entry_time > ?",
                (chat_id, cutoff)
            ).fetchall()

        if not rows:
            return {"total_trades": 0, "win_rate": 0, "total_pnl_usdt": 0, "avg_rr": 0, "by_regime": {}, "max_consecutive_loss": 0}

        wins = sum(1 for r in rows if r[2] and r[2] > 0)
        total = len(rows)
        total_pnl = sum(r[3] or 0 for r in rows)

        # By regime
        by_regime = {}
        for r in rows:
            regime = r[5] or "unknown"
            if regime not in by_regime:
                by_regime[regime] = {"trades": 0, "wins": 0, "pnl": 0}
            by_regime[regime]["trades"] += 1
            if r[2] and r[2] > 0:
                by_regime[regime]["wins"] += 1
            by_regime[regime]["pnl"] += r[3] or 0

        # Max consecutive loss
        max_consec = 0
        current = 0
        for r in rows:
            if r[2] and r[2] < 0:
                current += 1
                max_consec = max(max_consec, current)
            else:
                current = 0

        return {
            "total_trades": total,
            "win_rate": round(wins / total * 100, 1) if total > 0 else 0,
            "total_pnl_usdt": round(total_pnl, 2),
            "avg_pnl_pct": round(sum(r[2] or 0 for r in rows) / total, 1) if total > 0 else 0,
            "by_regime": by_regime,
            "max_consecutive_loss": max_consec,
        }
