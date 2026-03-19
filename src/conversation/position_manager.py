"""
수동 포지션 관리자.
사용자가 Telegram에서 등록한 포지션의 CRUD를 SQLite로 관리.
"""
import logging
import sqlite3
from datetime import datetime, timezone

from core.types import ManualPosition, Side

logger = logging.getLogger(__name__)


class PositionManager:
    """복수 수동 포지션 관리. SQLite 저장."""

    def __init__(self, db_path: str = "signal_bot.db"):
        self._db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS manual_positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    leverage REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    is_active INTEGER NOT NULL DEFAULT 1,
                    stop_loss REAL,
                    take_profit REAL,
                    margin_usdt REAL,
                    entry_reason TEXT DEFAULT ''
                )
            """)
            conn.execute("PRAGMA journal_mode=WAL")

    def open_position(
        self,
        chat_id: str,
        symbol: str,
        side: Side,
        entry_price: float,
        leverage: float,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        margin_usdt: float | None = None,
        entry_reason: str = "",
    ) -> ManualPosition:
        if entry_price <= 0:
            raise ValueError("진입가는 0보다 커야 합니다.")
        if leverage <= 0 or leverage > 125:
            raise ValueError("레버리지는 1~125 사이여야 합니다.")
        now = datetime.now(timezone.utc)
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute(
                "INSERT INTO manual_positions "
                "(chat_id, symbol, side, entry_price, leverage, created_at, is_active, "
                "stop_loss, take_profit, margin_usdt, entry_reason) "
                "VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?)",
                (chat_id, symbol, side.value, entry_price, leverage, now.isoformat(),
                 stop_loss, take_profit, margin_usdt, entry_reason),
            )
            pos_id = cursor.lastrowid
        return ManualPosition(
            id=pos_id,
            chat_id=chat_id,
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            leverage=leverage,
            created_at=now,
            stop_loss=stop_loss,
            take_profit=take_profit,
            margin_usdt=margin_usdt,
            entry_reason=entry_reason,
        )

    def get_active_positions(self, chat_id: str) -> list[ManualPosition]:
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                "SELECT id, chat_id, symbol, side, entry_price, leverage, created_at, "
                "is_active, stop_loss, take_profit, margin_usdt, entry_reason "
                "FROM manual_positions WHERE chat_id = ? AND is_active = 1",
                (chat_id,),
            ).fetchall()
        return [self._row_to_position(r) for r in rows]

    def get_all_active_positions(self) -> list[ManualPosition]:
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                "SELECT id, chat_id, symbol, side, entry_price, leverage, created_at, "
                "is_active, stop_loss, take_profit, margin_usdt, entry_reason "
                "FROM manual_positions WHERE is_active = 1",
            ).fetchall()
        return [self._row_to_position(r) for r in rows]

    def close_position(self, position_id: int, chat_id: str) -> bool:
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute(
                "UPDATE manual_positions SET is_active = 0 "
                "WHERE id = ? AND chat_id = ? AND is_active = 1",
                (position_id, chat_id),
            )
            return cursor.rowcount > 0

    def close_position_by_symbol(self, chat_id: str, symbol: str) -> bool:
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT id FROM manual_positions "
                "WHERE chat_id = ? AND symbol = ? AND is_active = 1 "
                "ORDER BY id LIMIT 1",
                (chat_id, symbol),
            ).fetchone()
            if row is None:
                return False
            conn.execute(
                "UPDATE manual_positions SET is_active = 0 WHERE id = ?",
                (row[0],),
            )
            return True

    @staticmethod
    def _row_to_position(row: tuple) -> ManualPosition:
        return ManualPosition(
            id=row[0],
            chat_id=row[1],
            symbol=row[2],
            side=Side(row[3]),
            entry_price=row[4],
            leverage=row[5],
            created_at=datetime.fromisoformat(row[6]),
            is_active=bool(row[7]),
            stop_loss=row[8],
            take_profit=row[9],
            margin_usdt=row[10],
            entry_reason=row[11] or "",
        )
