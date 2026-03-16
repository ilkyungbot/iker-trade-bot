"""
시그널 정확도 추적.

시그널 발송 후 4H/8H/24H 가격 체크, TP/SL 도달 여부 기록.
주간 정확도 리포트 생성.
"""

import logging
import sqlite3
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)


class SignalTracker:
    """시그널 결과 추적 및 정확도 리포트."""

    def __init__(self, db_path: str = "signal_bot.db"):
        self._db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS signal_outcomes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    quality TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    stop_loss REAL NOT NULL,
                    take_profit REAL NOT NULL,
                    signal_time TEXT NOT NULL,
                    price_4h REAL,
                    price_8h REAL,
                    price_24h REAL,
                    tp_hit INTEGER DEFAULT 0,
                    sl_hit INTEGER DEFAULT 0,
                    checked_at TEXT
                )
            """)

    def record_signal(
        self,
        symbol: str,
        direction: str,
        strategy: str,
        quality: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        signal_time: datetime,
    ) -> int:
        """발송된 시그널 기록. ID 반환."""
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute(
                """INSERT INTO signal_outcomes
                (symbol, direction, strategy, quality, entry_price, stop_loss, take_profit, signal_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (symbol, direction, strategy, quality, entry_price, stop_loss, take_profit, signal_time.isoformat()),
            )
            return cursor.lastrowid or 0

    def update_outcome(
        self,
        signal_id: int,
        price_4h: float | None = None,
        price_8h: float | None = None,
        price_24h: float | None = None,
    ) -> None:
        """시그널 결과 업데이트 (시간별 가격 체크)."""
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT entry_price, stop_loss, take_profit, direction FROM signal_outcomes WHERE id = ?",
                (signal_id,),
            ).fetchone()
            if not row:
                return

            entry, sl, tp, direction = row
            tp_hit = 0
            sl_hit = 0

            for price in [price_4h, price_8h, price_24h]:
                if price is None:
                    continue
                if direction == "long":
                    if price >= tp:
                        tp_hit = 1
                    if price <= sl:
                        sl_hit = 1
                else:
                    if price <= tp:
                        tp_hit = 1
                    if price >= sl:
                        sl_hit = 1

            updates = []
            params = []
            if price_4h is not None:
                updates.append("price_4h = ?")
                params.append(price_4h)
            if price_8h is not None:
                updates.append("price_8h = ?")
                params.append(price_8h)
            if price_24h is not None:
                updates.append("price_24h = ?")
                params.append(price_24h)
            if tp_hit:
                updates.append("tp_hit = 1")
            if sl_hit:
                updates.append("sl_hit = 1")
            updates.append("checked_at = ?")
            params.append(datetime.now(timezone.utc).isoformat())
            params.append(signal_id)

            conn.execute(
                f"UPDATE signal_outcomes SET {', '.join(updates)} WHERE id = ?",
                params,
            )

    def get_unchecked_signals(self, hours_ago: int = 24) -> list[dict]:
        """아직 결과가 체크되지 않은 시그널 목록."""
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT id, symbol, direction, entry_price, stop_loss, take_profit, signal_time
                FROM signal_outcomes
                WHERE (price_24h IS NULL) AND signal_time <= ?""",
                (cutoff,),
            ).fetchall()
            return [dict(r) for r in rows]

    def weekly_report(self, days: int = 7) -> dict:
        """주간 정확도 리포트."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                """SELECT quality, tp_hit, sl_hit, direction, strategy
                FROM signal_outcomes
                WHERE signal_time >= ? AND checked_at IS NOT NULL""",
                (cutoff,),
            ).fetchall()

        total = len(rows)
        if total == 0:
            return {"total": 0, "tp_rate": 0, "sl_rate": 0, "by_quality": {}}

        tp_count = sum(1 for r in rows if r[1])
        sl_count = sum(1 for r in rows if r[2])

        by_quality: dict = {}
        for r in rows:
            q = r[0]
            if q not in by_quality:
                by_quality[q] = {"total": 0, "tp": 0, "sl": 0}
            by_quality[q]["total"] += 1
            if r[1]:
                by_quality[q]["tp"] += 1
            if r[2]:
                by_quality[q]["sl"] += 1

        return {
            "total": total,
            "tp_rate": tp_count / total if total > 0 else 0,
            "sl_rate": sl_count / total if total > 0 else 0,
            "by_quality": by_quality,
        }
