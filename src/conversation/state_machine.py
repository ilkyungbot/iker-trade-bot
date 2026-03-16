"""
대화 상태 머신.

상태 전이:
  IDLE ──(시그널 생성)──→ SIGNAL_SENT
  SIGNAL_SENT ──("잡았다")──→ MONITORING
  SIGNAL_SENT ──("패스"/1시간 타임아웃)──→ IDLE
  MONITORING ──(청산 조건)──→ EXIT_SIGNAL_SENT
  MONITORING ──("팔았다" 조기 청산)──→ IDLE
  EXIT_SIGNAL_SENT ──("팔았다")──→ IDLE
  EXIT_SIGNAL_SENT ──("홀딩")──→ MONITORING
"""

import logging
import sqlite3
import json
from datetime import datetime, timezone, timedelta
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

        session = UserSession(
            chat_id=chat_id,
            state=ConversationState(row[0]),
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

    def send_signal(self, chat_id: str, signal_msg: SignalMessage) -> bool:
        """시그널 발송. IDLE → SIGNAL_SENT."""
        session = self.get_session(chat_id)
        if session.state != ConversationState.IDLE:
            logger.warning(f"Cannot send signal: state is {session.state.value}")
            return False

        session.state = ConversationState.SIGNAL_SENT
        session.active_signal = signal_msg
        self._save_session(session)
        return True

    def user_entered(self, chat_id: str, entry_price: float | None = None) -> bool:
        """사용자 진입 확인 ('잡았다'). SIGNAL_SENT → MONITORING."""
        session = self.get_session(chat_id)
        if session.state != ConversationState.SIGNAL_SENT:
            return False

        session.state = ConversationState.MONITORING
        session.entry_confirmed_at = datetime.now(timezone.utc)
        session.user_entry_price = entry_price or (
            session.active_signal.signal.entry_price if session.active_signal else None
        )
        self._save_session(session)
        return True

    def user_passed(self, chat_id: str) -> bool:
        """사용자 패스 ('패스'). SIGNAL_SENT → IDLE."""
        session = self.get_session(chat_id)
        if session.state != ConversationState.SIGNAL_SENT:
            return False

        session.state = ConversationState.IDLE
        session.active_signal = None
        self._save_session(session)
        return True

    def send_exit_signal(self, chat_id: str) -> bool:
        """청산 시그널 발송. MONITORING → EXIT_SIGNAL_SENT."""
        session = self.get_session(chat_id)
        if session.state != ConversationState.MONITORING:
            return False

        session.state = ConversationState.EXIT_SIGNAL_SENT
        self._save_session(session)
        return True

    def user_exited(self, chat_id: str) -> bool:
        """사용자 청산 확인 ('팔았다'). MONITORING|EXIT_SIGNAL_SENT → IDLE."""
        session = self.get_session(chat_id)
        if session.state not in (ConversationState.MONITORING, ConversationState.EXIT_SIGNAL_SENT):
            return False

        session.state = ConversationState.IDLE
        session.active_signal = None
        session.entry_confirmed_at = None
        session.user_entry_price = None
        self._save_session(session)
        return True

    def user_hold(self, chat_id: str) -> bool:
        """사용자 홀딩 ('홀딩'). EXIT_SIGNAL_SENT → MONITORING."""
        session = self.get_session(chat_id)
        if session.state != ConversationState.EXIT_SIGNAL_SENT:
            return False

        session.state = ConversationState.MONITORING
        self._save_session(session)
        return True

    def check_expiry(self, chat_id: str, expiry_minutes: int = 60) -> bool:
        """SIGNAL_SENT 상태에서 타임아웃 체크. 만료 시 IDLE로 전이."""
        session = self.get_session(chat_id)
        if session.state != ConversationState.SIGNAL_SENT:
            return False

        # active_signal의 timestamp 기반 만료 체크
        if session.active_signal:
            signal_time = session.active_signal.signal.timestamp
            if signal_time.tzinfo is None:
                signal_time = signal_time.replace(tzinfo=timezone.utc)
            elapsed = datetime.now(timezone.utc) - signal_time
            if elapsed > timedelta(minutes=expiry_minutes):
                session.state = ConversationState.IDLE
                session.active_signal = None
                self._save_session(session)
                return True
        return False

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
