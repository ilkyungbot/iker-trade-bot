"""
대화 상태 머신 (v2 — 단순화).

상태 전이:
  IDLE ──(수동 포지션 등록)──→ MONITORING
  MONITORING ──(청산)──→ IDLE
"""

import logging
import sqlite3
import json
from datetime import datetime, timezone
from pathlib import Path

from core.types import (
    ConversationState,
    SignalMessage,
    SignalQuality,
    Signal,
    SignalAction,
    StrategyName,
    UserSession,
)

logger = logging.getLogger(__name__)


class ConversationStateMachine:
    """한 번에 1개 시그널만 활성. 상태는 SQLite에 저장."""

    def __init__(self, db_path: str = "signal_bot.db"):
        self._db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_sessions (
                    chat_id TEXT PRIMARY KEY,
                    state TEXT NOT NULL DEFAULT 'idle',
                    active_signal_json TEXT,
                    entry_confirmed_at TEXT,
                    user_entry_price REAL,
                    updated_at TEXT NOT NULL
                )
            """)

    def get_session(self, chat_id: str) -> UserSession:
        """현재 세션을 로드하거나 새로 생성."""
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT state, active_signal_json, entry_confirmed_at, user_entry_price "
                "FROM user_sessions WHERE chat_id = ?",
                (chat_id,),
            ).fetchone()

        if row is None:
            return UserSession(chat_id=chat_id, state=ConversationState.IDLE)

        # Migrate legacy states to IDLE
        state_value = row[0]
        if state_value not in ("idle", "monitoring"):
            state_value = "idle"

        session = UserSession(
            chat_id=chat_id,
            state=ConversationState(state_value),
            active_signal=_deserialize_signal_message(row[1]) if row[1] else None,
            entry_confirmed_at=datetime.fromisoformat(row[2]) if row[2] else None,
            user_entry_price=row[3],
        )
        return session

    def _save_session(self, session: UserSession) -> None:
        now = datetime.now(timezone.utc).isoformat()
        signal_json = _serialize_signal_message(session.active_signal) if session.active_signal else None
        entry_at = session.entry_confirmed_at.isoformat() if session.entry_confirmed_at else None

        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """INSERT INTO user_sessions (chat_id, state, active_signal_json, entry_confirmed_at, user_entry_price, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(chat_id) DO UPDATE SET
                    state=excluded.state,
                    active_signal_json=excluded.active_signal_json,
                    entry_confirmed_at=excluded.entry_confirmed_at,
                    user_entry_price=excluded.user_entry_price,
                    updated_at=excluded.updated_at""",
                (session.chat_id, session.state.value, signal_json, entry_at, session.user_entry_price, now),
            )

    # --- 상태 전이 ---

    def user_exited(self, chat_id: str) -> bool:
        """사용자 청산 확인. MONITORING → IDLE."""
        session = self.get_session(chat_id)
        if session.state != ConversationState.MONITORING:
            return False

        session.state = ConversationState.IDLE
        session.active_signal = None
        session.entry_confirmed_at = None
        session.user_entry_price = None
        self._save_session(session)
        return True

    def force_idle(self, chat_id: str) -> None:
        """강제로 IDLE 상태로 리셋."""
        session = self.get_session(chat_id)
        session.state = ConversationState.IDLE
        session.active_signal = None
        session.entry_confirmed_at = None
        session.user_entry_price = None
        self._save_session(session)


def _serialize_signal_message(msg: SignalMessage) -> str:
    """SignalMessage → JSON string."""
    s = msg.signal
    return json.dumps({
        "signal": {
            "timestamp": s.timestamp.isoformat(),
            "symbol": s.symbol,
            "action": s.action.value,
            "strategy": s.strategy.value,
            "entry_price": s.entry_price,
            "stop_loss": s.stop_loss,
            "take_profit": s.take_profit,
            "confidence": s.confidence,
            "metadata": s.metadata,
        },
        "quality": msg.quality.value,
        "explanation": msg.explanation,
        "indicators": msg.indicators,
        "risk_reward_ratio": msg.risk_reward_ratio,
    })


def _deserialize_signal_message(json_str: str) -> SignalMessage:
    """JSON string → SignalMessage."""
    d = json.loads(json_str)
    sd = d["signal"]
    signal = Signal(
        timestamp=datetime.fromisoformat(sd["timestamp"]),
        symbol=sd["symbol"],
        action=SignalAction(sd["action"]),
        strategy=StrategyName(sd["strategy"]),
        entry_price=sd["entry_price"],
        stop_loss=sd["stop_loss"],
        take_profit=sd["take_profit"],
        confidence=sd["confidence"],
        metadata=sd.get("metadata", {}),
    )
    return SignalMessage(
        signal=signal,
        quality=SignalQuality(d["quality"]),
        explanation=d["explanation"],
        indicators=d["indicators"],
        risk_reward_ratio=d["risk_reward_ratio"],
    )
