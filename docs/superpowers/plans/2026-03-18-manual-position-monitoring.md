# Manual Position Monitoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 사용자가 시그널과 무관하게 수동 포지션을 등록하면, 해당 코인의 이벤트(상승/하락 신호, 포지션 체크, 매도/홀딩/매수 추천)를 레버리지 맥락 고려하여 실시간 푸시하는 기능 추가

**Architecture:** 기존 단일 시그널 상태머신과 별도로, 복수 수동 포지션을 관리하는 `PositionManager`(SQLite)를 추가. 대화 흐름은 `TelegramCommandHandler`에서 user_data 기반 멀티스텝 질문으로 처리. 5분 주기 스케줄러가 활성 포지션별 이벤트를 감지하여 Telegram 푸시.

**Tech Stack:** Python 3.12+ / SQLite / python-telegram-bot (ConversationHandler 대신 user_data 기반) / pandas / APScheduler

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `src/conversation/position_manager.py` | 수동 포지션 CRUD (SQLite) |
| Create | `src/strategy/position_monitor.py` | 포지션별 이벤트 감지 (지표 기반 6종 이벤트) |
| Modify | `src/core/types.py` | `ManualPosition` dataclass 추가 |
| Modify | `src/review/telegram_commands.py` | "신규 포지션"/"청산" 대화 흐름 + 이벤트 라우팅 |
| Modify | `src/review/reporter.py` | 포지션 이벤트 메시지 포맷 |
| Modify | `src/main.py` | PositionManager + PositionMonitor 초기화, 5분 스케줄러 등록 |
| Create | `tests/conversation/test_position_manager.py` | PositionManager 단위 테스트 |
| Create | `tests/strategy/test_position_monitor.py` | PositionMonitor 단위 테스트 |
| Create | `tests/review/test_telegram_position_flow.py` | 대화 흐름 테스트 |

---

### Task 1: ManualPosition 타입 추가

**Files:**
- Modify: `src/core/types.py:125` (PairInfo 아래, 파일 끝)

- [ ] **Step 1: types.py에 ManualPosition dataclass 추가**

```python
# src/core/types.py 끝에 추가 (PairInfo 아래)

@dataclass
class ManualPosition:
    """사용자가 수동 등록한 포지션."""
    id: int | None  # DB auto-increment
    chat_id: str
    symbol: str  # e.g., "BTCUSDT"
    side: Side  # LONG or SHORT
    entry_price: float
    leverage: float
    created_at: datetime
    is_active: bool = True
```

- [ ] **Step 2: 커밋**

```bash
git add src/core/types.py
git commit -m "feat: add ManualPosition dataclass for user-registered positions"
```

---

### Task 2: PositionManager (포지션 CRUD)

**Files:**
- Create: `src/conversation/position_manager.py`
- Create: `tests/conversation/test_position_manager.py`

- [ ] **Step 1: 테스트 작성**

```python
# tests/conversation/test_position_manager.py
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
```

- [ ] **Step 2: 테스트 실행 → 실패 확인**

Run: `cd /Users/gowid/iker-trade-bot && python -m pytest tests/conversation/test_position_manager.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: PositionManager 구현**

```python
# src/conversation/position_manager.py
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
                    is_active INTEGER NOT NULL DEFAULT 1
                )
            """)

    def open_position(
        self, chat_id: str, symbol: str, side: Side,
        entry_price: float, leverage: float,
    ) -> ManualPosition:
        now = datetime.now(timezone.utc)
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute(
                "INSERT INTO manual_positions (chat_id, symbol, side, entry_price, leverage, created_at, is_active) "
                "VALUES (?, ?, ?, ?, ?, ?, 1)",
                (chat_id, symbol, side.value, entry_price, leverage, now.isoformat()),
            )
            pos_id = cursor.lastrowid
        return ManualPosition(
            id=pos_id, chat_id=chat_id, symbol=symbol, side=side,
            entry_price=entry_price, leverage=leverage, created_at=now,
        )

    def get_active_positions(self, chat_id: str) -> list[ManualPosition]:
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                "SELECT id, chat_id, symbol, side, entry_price, leverage, created_at "
                "FROM manual_positions WHERE chat_id = ? AND is_active = 1",
                (chat_id,),
            ).fetchall()
        return [self._row_to_position(r) for r in rows]

    def get_all_active_positions(self) -> list[ManualPosition]:
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                "SELECT id, chat_id, symbol, side, entry_price, leverage, created_at "
                "FROM manual_positions WHERE is_active = 1",
            ).fetchall()
        return [self._row_to_position(r) for r in rows]

    def close_position(self, position_id: int, chat_id: str) -> bool:
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute(
                "UPDATE manual_positions SET is_active = 0 WHERE id = ? AND chat_id = ? AND is_active = 1",
                (position_id, chat_id),
            )
            return cursor.rowcount > 0

    def close_position_by_symbol(self, chat_id: str, symbol: str) -> bool:
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute(
                "UPDATE manual_positions SET is_active = 0 WHERE chat_id = ? AND symbol = ? AND is_active = 1",
                (chat_id, symbol),
            )
            return cursor.rowcount > 0

    @staticmethod
    def _row_to_position(row) -> ManualPosition:
        return ManualPosition(
            id=row[0], chat_id=row[1], symbol=row[2],
            side=Side(row[3]), entry_price=row[4], leverage=row[5],
            created_at=datetime.fromisoformat(row[6]),
        )
```

- [ ] **Step 4: 테스트 실행 → 통과 확인**

Run: `cd /Users/gowid/iker-trade-bot && python -m pytest tests/conversation/test_position_manager.py -v`
Expected: all PASS

- [ ] **Step 5: 커밋**

```bash
git add src/conversation/position_manager.py tests/conversation/test_position_manager.py
git commit -m "feat: add PositionManager for manual position CRUD"
```

---

### Task 3: PositionMonitor (이벤트 감지 엔진)

**Files:**
- Create: `src/strategy/position_monitor.py`
- Create: `tests/strategy/test_position_monitor.py`

- [ ] **Step 1: 테스트 작성**

```python
# tests/strategy/test_position_monitor.py
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
    df = _make_df()
    events = monitor.detect_events(pos, df, current_price=67600, funding_rate=0.0001)
    # 가격 변화 0.15% at 10x = 1.5% PnL — 평온한 상태면 이벤트 없어야 함
    assert len(events) == 0


def test_liquidation_warning():
    monitor = PositionMonitor()
    pos = _make_position(leverage=20, entry=67500)
    # 20x 롱이면 청산가 약 67500 * (1 - 1/20) = 64125 근방
    # 현재가 64500이면 청산가 접근
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
    # 10x 롱, 가격 70875 = +5% 가격변동 = +50% PnL
    df = _make_df(close=70875, rsi=75)
    events = monitor.detect_events(pos, df, current_price=70875, funding_rate=0)
    event_types = [e.event_type for e in events]
    assert "sell_recommendation" in event_types


def test_buy_recommendation_long_dip():
    monitor = PositionMonitor()
    pos = _make_position(side=Side.LONG, leverage=5, entry=67500)
    # 가격 하락 but RSI 과매도 + MACD 반등 → 물타기 기회
    df = _make_df(close=66000, rsi=28)
    df.at[1, "macd_hist_cross_up"] = True
    events = monitor.detect_events(pos, df, current_price=66000, funding_rate=0)
    event_types = [e.event_type for e in events]
    assert "buy_recommendation" in event_types
```

- [ ] **Step 2: 테스트 실행 → 실패 확인**

Run: `cd /Users/gowid/iker-trade-bot && python -m pytest tests/strategy/test_position_monitor.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: PositionMonitor 구현**

```python
# src/strategy/position_monitor.py
"""
포지션 모니터링 — 이벤트 감지 엔진.

활성 포지션별로 기술 지표 + 레버리지 맥락을 분석하여
6종 이벤트(상승/하락 신호, 포지션 체크, 매도/홀딩/매수 추천)를 감지.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

from core.types import ManualPosition, Side

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PositionEvent:
    """감지된 포지션 이벤트."""
    event_type: str  # bullish_signal, bearish_signal, position_check, sell_recommendation, hold_recommendation, buy_recommendation
    position_id: int
    symbol: str
    message: str  # 한국어 설명
    severity: str  # info, warning, critical
    pnl_percent: float  # 현재 레버리지 반영 PnL%


class PositionMonitor:
    """포지션별 이벤트 감지."""

    def __init__(self):
        # 이벤트 중복 방지: {position_id: {event_type: last_trigger_time}}
        self._last_events: dict[int, dict[str, datetime]] = {}

    def detect_events(
        self,
        position: ManualPosition,
        df: pd.DataFrame,
        current_price: float,
        funding_rate: float,
    ) -> list[PositionEvent]:
        """포지션에 대한 이벤트 감지. df는 최근 지표가 포함된 DataFrame."""
        events: list[PositionEvent] = []

        if len(df) < 2:
            return events

        current = df.iloc[-1]
        pnl_pct = self._calc_pnl_pct(position, current_price)
        liq_distance = self._calc_liquidation_distance(position, current_price)

        # 1. 포지션 체크 (청산가 접근)
        if liq_distance < 30:  # 청산가까지 마진의 30% 이내
            severity = "critical" if liq_distance < 15 else "warning"
            events.append(PositionEvent(
                event_type="position_check",
                position_id=position.id,
                symbol=position.symbol,
                message=f"청산가 접근 주의! 잔여 마진 약 {liq_distance:.1f}% (PnL: {pnl_pct:+.1f}%)",
                severity=severity,
                pnl_percent=pnl_pct,
            ))

        # 2. 상승 신호
        bullish_reasons = self._check_bullish(current, df)
        if len(bullish_reasons) >= 2:
            events.append(PositionEvent(
                event_type="bullish_signal",
                position_id=position.id,
                symbol=position.symbol,
                message="상승 신호: " + " / ".join(bullish_reasons),
                severity="info",
                pnl_percent=pnl_pct,
            ))

        # 3. 하락 신호
        bearish_reasons = self._check_bearish(current, df)
        if len(bearish_reasons) >= 2:
            events.append(PositionEvent(
                event_type="bearish_signal",
                position_id=position.id,
                symbol=position.symbol,
                message="하락 신호: " + " / ".join(bearish_reasons),
                severity="warning" if position.side == Side.LONG else "info",
                pnl_percent=pnl_pct,
            ))

        # 4. 매도 추천
        sell_reasons = self._check_sell(position, current, pnl_pct, funding_rate)
        if sell_reasons:
            events.append(PositionEvent(
                event_type="sell_recommendation",
                position_id=position.id,
                symbol=position.symbol,
                message="매도 추천: " + " / ".join(sell_reasons),
                severity="warning",
                pnl_percent=pnl_pct,
            ))

        # 5. 매수 추천 (물타기)
        buy_reasons = self._check_buy(position, current, df, pnl_pct)
        if buy_reasons:
            events.append(PositionEvent(
                event_type="buy_recommendation",
                position_id=position.id,
                symbol=position.symbol,
                message="매수(물타기) 추천: " + " / ".join(buy_reasons),
                severity="info",
                pnl_percent=pnl_pct,
            ))

        # 6. 홀딩 추천 (추세 유지 중)
        if not sell_reasons and not buy_reasons and abs(pnl_pct) < 30:
            hold_reasons = self._check_hold(position, current, pnl_pct)
            if hold_reasons:
                events.append(PositionEvent(
                    event_type="hold_recommendation",
                    position_id=position.id,
                    symbol=position.symbol,
                    message="홀딩 추천: " + " / ".join(hold_reasons),
                    severity="info",
                    pnl_percent=pnl_pct,
                ))

        return events

    def _calc_pnl_pct(self, pos: ManualPosition, current_price: float) -> float:
        """레버리지 반영 PnL%."""
        price_change_pct = (current_price - pos.entry_price) / pos.entry_price * 100
        if pos.side == Side.SHORT:
            price_change_pct = -price_change_pct
        return price_change_pct * pos.leverage

    def _calc_liquidation_distance(self, pos: ManualPosition, current_price: float) -> float:
        """청산가까지 남은 마진 비율 (%). 100이면 안전, 0이면 청산."""
        # 단순화: 청산가 ≈ entry × (1 ∓ 1/leverage)
        if pos.side == Side.LONG:
            liq_price = pos.entry_price * (1 - 1 / pos.leverage)
            if current_price <= liq_price:
                return 0
            return (current_price - liq_price) / (pos.entry_price - liq_price) * 100
        else:
            liq_price = pos.entry_price * (1 + 1 / pos.leverage)
            if current_price >= liq_price:
                return 0
            return (liq_price - current_price) / (liq_price - pos.entry_price) * 100

    def _check_bullish(self, current, df) -> list[str]:
        reasons = []
        if current.get("ema_golden_cross", False):
            reasons.append("EMA 골든크로스")
        if current.get("rsi_cross_up", False):
            reasons.append("RSI 상향돌파")
        if current.get("macd_hist_cross_up", False):
            reasons.append("MACD 양전환")
        vol_ratio = current.get("volume_ratio", 1.0)
        if vol_ratio >= 2.0 and current.get("close", 0) > current.get("open", 0):
            reasons.append(f"거래량 급증 {vol_ratio:.1f}배")
        rsi = current.get("rsi", 50)
        if rsi <= 30:
            reasons.append(f"RSI 과매도 ({rsi:.0f})")
        return reasons

    def _check_bearish(self, current, df) -> list[str]:
        reasons = []
        if current.get("ema_death_cross", False):
            reasons.append("EMA 데드크로스")
        if current.get("rsi_cross_down", False):
            reasons.append("RSI 하향돌파")
        if current.get("macd_hist_cross_down", False):
            reasons.append("MACD 음전환")
        vol_ratio = current.get("volume_ratio", 1.0)
        if vol_ratio >= 2.0 and current.get("close", 0) < current.get("open", 0):
            reasons.append(f"거래량 급증 {vol_ratio:.1f}배 (하락)")
        rsi = current.get("rsi", 50)
        if rsi >= 70:
            reasons.append(f"RSI 과매수 ({rsi:.0f})")
        return reasons

    def _check_sell(self, pos, current, pnl_pct, funding_rate) -> list[str]:
        reasons = []
        # 고수익 구간 도달
        if pnl_pct >= 30:
            reasons.append(f"PnL {pnl_pct:+.1f}% 도달")
        # 롱인데 RSI 과매수 + 높은 PnL
        if pos.side == Side.LONG and current.get("rsi", 50) >= 72 and pnl_pct > 10:
            reasons.append(f"RSI 과매수({current.get('rsi', 0):.0f}) + 수익 구간")
        # 숏인데 RSI 과매도 + 높은 PnL
        if pos.side == Side.SHORT and current.get("rsi", 50) <= 28 and pnl_pct > 10:
            reasons.append(f"RSI 과매도({current.get('rsi', 0):.0f}) + 수익 구간")
        # 불리한 펀딩비 누적
        if pos.side == Side.LONG and funding_rate > 0.001:
            reasons.append(f"높은 펀딩비 ({funding_rate*100:.3f}%)")
        if pos.side == Side.SHORT and funding_rate < -0.001:
            reasons.append(f"높은 역펀딩비 ({funding_rate*100:.3f}%)")
        return reasons

    def _check_buy(self, pos, current, df, pnl_pct) -> list[str]:
        """물타기 기회 감지 (현 포지션 방향과 동일한 추가 매수)."""
        reasons = []
        if pos.side == Side.LONG and pnl_pct < -5:
            # 롱인데 손실 중 + 반등 신호
            if current.get("rsi", 50) <= 30:
                reasons.append(f"RSI 과매도({current.get('rsi', 0):.0f})")
            if current.get("macd_hist_cross_up", False):
                reasons.append("MACD 반등 신호")
        elif pos.side == Side.SHORT and pnl_pct < -5:
            if current.get("rsi", 50) >= 70:
                reasons.append(f"RSI 과매수({current.get('rsi', 0):.0f})")
            if current.get("macd_hist_cross_down", False):
                reasons.append("MACD 하락 신호")
        return reasons

    def _check_hold(self, pos, current, pnl_pct) -> list[str]:
        """홀딩 권유 조건."""
        reasons = []
        adx = current.get("adx", 0)
        ema_20 = current.get("ema_20", 0)
        ema_50 = current.get("ema_50", 0)
        close = current.get("close", 0)

        if pos.side == Side.LONG:
            if close > ema_20 > ema_50 and adx > 20:
                reasons.append("강세 정배열 유지")
            if 0 < pnl_pct < 20 and adx > 25:
                reasons.append(f"추세 강도 양호 (ADX {adx:.0f})")
        else:
            if close < ema_20 < ema_50 and adx > 20:
                reasons.append("약세 역배열 유지")
            if 0 < pnl_pct < 20 and adx > 25:
                reasons.append(f"추세 강도 양호 (ADX {adx:.0f})")
        return reasons
```

- [ ] **Step 4: 테스트 실행 → 통과 확인**

Run: `cd /Users/gowid/iker-trade-bot && python -m pytest tests/strategy/test_position_monitor.py -v`
Expected: all PASS

- [ ] **Step 5: 커밋**

```bash
git add src/strategy/position_monitor.py tests/strategy/test_position_monitor.py
git commit -m "feat: add PositionMonitor for event detection on manual positions"
```

---

### Task 4: Reporter 포지션 이벤트 메시지 포맷 추가

**Files:**
- Modify: `src/review/reporter.py`

- [ ] **Step 1: reporter.py에 포지션 이벤트 포맷 메서드 추가**

`Reporter` 클래스에 다음 메서드 추가:

```python
# src/review/reporter.py — 모듈 레벨에 추가 (기존 _QUALITY_ICON 아래)

_EVENT_ICON = {
    "bullish_signal": "\U0001f4c8",      # 📈
    "bearish_signal": "\U0001f4c9",      # 📉
    "position_check": "\U0001f6a8",      # 🚨
    "sell_recommendation": "\U0001f4b0",  # 💰
    "hold_recommendation": "\u23f3",      # ⏳
    "buy_recommendation": "\U0001f6d2",   # 🛒
}

_EVENT_KR = {
    "bullish_signal": "상승 신호",
    "bearish_signal": "하락 신호",
    "position_check": "포지션 체크",
    "sell_recommendation": "매도 추천",
    "hold_recommendation": "홀딩 추천",
    "buy_recommendation": "매수 추천",
}

def format_position_event(self, event, position) -> str:
    """포지션 이벤트 Telegram 메시지."""
    from strategy.position_monitor import PositionEvent
    from core.types import ManualPosition, Side

    icon = _EVENT_ICON.get(event.event_type, "\u2139\ufe0f")
    event_kr = _EVENT_KR.get(event.event_type, event.event_type)
    side_kr = "롱" if position.side == Side.LONG else "숏"
    pnl_icon = "\u2705" if event.pnl_percent > 0 else "\u26a0\ufe0f" if event.pnl_percent < -10 else "\u2796"

    lines = [
        f"<b>{icon} {event_kr} | {position.symbol}</b>",
        "",
        f"방향: {side_kr} {position.leverage}x",
        f"평단: {position.entry_price:,.2f}",
        f"PnL: {pnl_icon} {event.pnl_percent:+.1f}%",
        "",
        f"{event.message}",
    ]

    if event.severity == "critical":
        lines.append("")
        lines.append("<b>\u26a0\ufe0f 즉시 확인이 필요합니다!</b>")

    return "\n".join(lines)

def format_position_registered(self, position) -> str:
    """포지션 등록 확인 메시지."""
    from core.types import ManualPosition, Side
    side_kr = "롱" if position.side == Side.LONG else "숏"
    return (
        f"<b>\u2705 포지션 등록 완료</b>\n\n"
        f"종목: {position.symbol}\n"
        f"방향: {side_kr}\n"
        f"평단: {position.entry_price:,.2f}\n"
        f"레버리지: {position.leverage}x\n\n"
        f"모니터링을 시작합니다. 청산 시 '청산'을 입력하세요."
    )

def format_position_closed(self, position, final_pnl: float | None = None) -> str:
    """포지션 청산 메시지."""
    from core.types import ManualPosition, Side
    side_kr = "롱" if position.side == Side.LONG else "숏"
    msg = (
        f"<b>\U0001f6aa 포지션 청산</b>\n\n"
        f"종목: {position.symbol} ({side_kr} {position.leverage}x)\n"
        f"평단: {position.entry_price:,.2f}\n"
    )
    if final_pnl is not None:
        msg += f"최종 PnL: {final_pnl:+.1f}%\n"
    msg += "\n모니터링을 종료합니다."
    return msg

async def send_position_event(self, event, position) -> None:
    text = self.format_position_event(event, position)
    await self._send(text)

async def send_position_registered(self, position) -> None:
    text = self.format_position_registered(position)
    await self._send(text)

async def send_position_closed(self, position, final_pnl=None) -> None:
    text = self.format_position_closed(position, final_pnl)
    await self._send(text)
```

- [ ] **Step 2: 커밋**

```bash
git add src/review/reporter.py
git commit -m "feat: add position event message formats to Reporter"
```

---

### Task 5: Telegram 대화 흐름 (신규 포지션 / 청산)

**Files:**
- Modify: `src/review/telegram_commands.py`
- Create: `tests/review/test_telegram_position_flow.py`

- [ ] **Step 1: 테스트 작성**

```python
# tests/review/test_telegram_position_flow.py
"""
신규 포지션 대화 흐름 테스트.

user_data 기반 멀티스텝:
  "신규 포지션" → (코인명 질문) → "BTCUSDT" → (롱/숏 질문) → "롱"
  → (평단 질문) → "67500" → (레버리지 질문) → "10" → 등록 완료
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# 대화 흐름 상태 상수 테스트
from review.telegram_commands import _POSITION_FLOW_STEPS


def test_position_flow_steps_defined():
    assert "ask_symbol" in _POSITION_FLOW_STEPS
    assert "ask_side" in _POSITION_FLOW_STEPS
    assert "ask_entry" in _POSITION_FLOW_STEPS
    assert "ask_leverage" in _POSITION_FLOW_STEPS
```

- [ ] **Step 2: telegram_commands.py 수정**

`_KR_COMMANDS`에 "신규 포지션"과 "청산" 추가, 대화 흐름 처리 로직 추가:

```python
# src/review/telegram_commands.py 수정사항

# 파일 상단 임포트에 추가
from datetime import datetime, timezone, timedelta
from core.types import ConversationState, Side

# _KR_COMMANDS 딕셔너리에 추가
_KR_COMMANDS: dict[str, str] = {
    "잡았다": "_cmd_entered",
    "팔았다": "_cmd_exited",
    "패스": "_cmd_pass",
    "홀딩": "_cmd_hold",
    "상태": "_cmd_status",
    "현황": "_cmd_briefing",
    "성과": "_cmd_performance",
    "도움말": "_cmd_help",
    "신규 포지션": "_cmd_new_position",
    "청산": "_cmd_close_position",
}

# 대화 흐름 상태
_POSITION_FLOW_STEPS = {
    "ask_symbol": "어떤 코인인가요? (예: BTC, ETH, SOL)",
    "ask_side": "롱인가요, 숏인가요? (롱/숏)",
    "ask_entry": "평단가를 입력해주세요. (숫자만)",
    "ask_leverage": "레버리지 배율을 입력해주세요. (예: 10)",
}
```

`TelegramCommandHandler` 클래스에 다음 메서드 추가:

```python
async def _cmd_new_position(self, update, context):
    """신규 포지션 등록 시작."""
    if not self._check_auth(update):
        return
    context.user_data["position_flow"] = "ask_symbol"
    context.user_data["position_data"] = {}
    await update.message.reply_text(
        "\U0001f4dd <b>신규 포지션 등록</b>\n\n" + _POSITION_FLOW_STEPS["ask_symbol"],
        parse_mode="HTML",
    )

async def _cmd_close_position(self, update, context):
    """포지션 청산."""
    if not self._check_auth(update):
        return
    bot = self._bot_ref
    if not bot or not hasattr(bot, "position_manager"):
        await update.message.reply_text("봇이 초기화되지 않았습니다.")
        return

    positions = bot.position_manager.get_active_positions(self.chat_id)
    if not positions:
        await update.message.reply_text("활성 포지션이 없습니다.")
        return

    if len(positions) == 1:
        pos = positions[0]
        # 현재가 조회하여 PnL 계산
        final_pnl = None
        try:
            now = datetime.now(timezone.utc)
            candles = await bot._run_sync(
                bot.collector.get_candles, pos.symbol, "1",
                start_time=now - timedelta(minutes=5),
            )
            if candles:
                current_price = candles[-1].close
                price_change = (current_price - pos.entry_price) / pos.entry_price * 100
                if pos.side == Side.SHORT:
                    price_change = -price_change
                final_pnl = price_change * pos.leverage
        except Exception:
            pass

        bot.position_manager.close_position(pos.id, self.chat_id)
        if hasattr(bot, "position_monitor"):
            bot.position_monitor.clear_position(pos.id)
        text = bot.reporter.format_position_closed(pos, final_pnl)
        await update.message.reply_text(text, parse_mode="HTML")
    else:
        # 복수 포지션 → 어떤 것 청산할지 질문
        context.user_data["position_flow"] = "ask_close_which"
        lines = ["\U0001f4dd <b>어떤 포지션을 청산할까요?</b>\n"]
        for p in positions:
            side_kr = "롱" if p.side == Side.LONG else "숏"
            lines.append(f"• {p.symbol} ({side_kr} {p.leverage}x) — 평단 {p.entry_price:,.2f}")
        lines.append("\n코인명을 입력해주세요. (예: BTC)")
        await update.message.reply_text("\n".join(lines), parse_mode="HTML")
```

`_route_korean` 메서드를 수정하여 대화 흐름 처리:

```python
async def _route_korean(self, update, context):
    if not self._check_auth(update):
        return
    text = (update.message.text or "").strip()

    # 대화 흐름 중이면 흐름 처리 우선
    flow_step = context.user_data.get("position_flow")
    if flow_step:
        await self._handle_position_flow(update, context, text)
        return

    method_name = _KR_COMMANDS.get(text)
    if method_name:
        handler = getattr(self, method_name)
        await handler(update, context)
        return

    # 티커 심볼 입력 감지
    upper = text.upper()
    if upper.isalpha() and 1 <= len(upper) <= 10:
        await self._cmd_analyze_coin(update, context, upper)

async def _handle_position_flow(self, update, context, text):
    """신규 포지션 대화 흐름 단계별 처리."""
    step = context.user_data.get("position_flow")
    data = context.user_data.get("position_data", {})
    bot = self._bot_ref

    if step == "ask_symbol":
        symbol = text.strip().upper()
        if not symbol.endswith("USDT"):
            symbol += "USDT"
        data["symbol"] = symbol
        context.user_data["position_data"] = data
        context.user_data["position_flow"] = "ask_side"
        await update.message.reply_text(_POSITION_FLOW_STEPS["ask_side"])

    elif step == "ask_side":
        text_lower = text.strip()
        if text_lower in ("롱", "long", "매수"):
            data["side"] = Side.LONG
        elif text_lower in ("숏", "short", "매도"):
            data["side"] = Side.SHORT
        else:
            await update.message.reply_text("'롱' 또는 '숏'으로 입력해주세요.")
            return
        context.user_data["position_data"] = data
        context.user_data["position_flow"] = "ask_entry"
        await update.message.reply_text(_POSITION_FLOW_STEPS["ask_entry"])

    elif step == "ask_entry":
        try:
            entry_price = float(text.strip().replace(",", ""))
            if entry_price <= 0:
                raise ValueError
        except ValueError:
            await update.message.reply_text("올바른 숫자를 입력해주세요.")
            return
        data["entry_price"] = entry_price
        context.user_data["position_data"] = data
        context.user_data["position_flow"] = "ask_leverage"
        await update.message.reply_text(_POSITION_FLOW_STEPS["ask_leverage"])

    elif step == "ask_leverage":
        try:
            leverage = float(text.strip().replace("배", "").replace("x", ""))
            if leverage <= 0 or leverage > 125:
                raise ValueError
        except ValueError:
            await update.message.reply_text("1~125 사이 숫자를 입력해주세요.")
            return
        data["leverage"] = leverage

        # 등록 완료
        if bot and hasattr(bot, "position_manager"):
            pos = bot.position_manager.open_position(
                self.chat_id, data["symbol"], data["side"],
                data["entry_price"], data["leverage"],
            )
            text_msg = bot.reporter.format_position_registered(pos)
            await update.message.reply_text(text_msg, parse_mode="HTML")
        else:
            await update.message.reply_text("포지션 매니저가 초기화되지 않았습니다.")

        # 흐름 종료
        context.user_data.pop("position_flow", None)
        context.user_data.pop("position_data", None)

    elif step == "ask_close_which":
        symbol = text.strip().upper()
        if not symbol.endswith("USDT"):
            symbol += "USDT"
        if bot and hasattr(bot, "position_manager"):
            positions = bot.position_manager.get_active_positions(self.chat_id)
            target = next((p for p in positions if p.symbol == symbol), None)
            if target:
                bot.position_manager.close_position(target.id, self.chat_id)
                if hasattr(bot, "position_monitor"):
                    bot.position_monitor.clear_position(target.id)
                text_msg = bot.reporter.format_position_closed(target)
                await update.message.reply_text(text_msg, parse_mode="HTML")
            else:
                await update.message.reply_text(f"{symbol} 포지션을 찾을 수 없습니다.")
        context.user_data.pop("position_flow", None)
        context.user_data.pop("position_data", None)
```

도움말 메시지에 신규 포지션/청산 추가:

```python
# _cmd_help 메서드의 msg 문자열에 추가
"신규 포지션 \u2014 수동 포지션 등록\n"
"청산 \u2014 포지션 청산\n"
```

- [ ] **Step 3: 테스트 실행 → 통과 확인**

Run: `cd /Users/gowid/iker-trade-bot && python -m pytest tests/review/test_telegram_position_flow.py -v`
Expected: PASS

- [ ] **Step 4: 커밋**

```bash
git add src/review/telegram_commands.py tests/review/test_telegram_position_flow.py
git commit -m "feat: add manual position registration and close conversation flow"
```

---

### Task 6: main.py 통합 (PositionManager + 이벤트 모니터링 스케줄러)

**Files:**
- Modify: `src/main.py:49-91` (SignalBot.__init__)
- Modify: `src/main.py:93-137` (SignalBot.start)

- [ ] **Step 1: SignalBot.__init__에 PositionManager, PositionMonitor 초기화 추가**

```python
# src/main.py — import 추가
from conversation.position_manager import PositionManager
from strategy.position_monitor import PositionMonitor

# SignalBot.__init__ 내부에 추가 (self.signal_tracker = ... 다음)
self.position_manager = PositionManager()
self.position_monitor = PositionMonitor()
```

- [ ] **Step 2: start()에 5분 주기 포지션 이벤트 스케줄러 추가**

```python
# SignalBot.start() — scheduler.add_job 추가
scheduler.add_job(self.monitor_manual_positions, "cron", minute="*/5")
```

- [ ] **Step 3: monitor_manual_positions 메서드 구현**

```python
async def monitor_manual_positions(self) -> None:
    """수동 포지션 이벤트 감지 + 푸시."""
    positions = self.position_manager.get_all_active_positions()
    if not positions:
        return

    now = datetime.now(timezone.utc)

    for pos in positions:
        try:
            # 4H 캔들 + 지표
            candles = await self._run_sync(
                self.collector.get_candles,
                pos.symbol, self.config.signal.primary_interval,
                start_time=now - timedelta(days=30),
            )
            if not candles or len(candles) < 10:
                continue

            df = candles_to_dataframe(candles)
            df = add_all_features(df)

            # 현재가
            recent = await self._run_sync(
                self.collector.get_candles,
                pos.symbol, "1",
                start_time=now - timedelta(minutes=5),
            )
            if not recent:
                continue
            current_price = recent[-1].close

            # 펀딩비
            funding_rate = 0.0
            try:
                rates = await self._run_sync(
                    self.collector.get_funding_rates,
                    pos.symbol, start_time=now - timedelta(hours=8),
                )
                if rates:
                    funding_rate = rates[-1].rate
            except Exception:
                pass

            # 이벤트 감지
            events = self.position_monitor.detect_events(pos, df, current_price, funding_rate)

            # 이벤트 발송
            for event in events:
                await self.reporter.send_position_event(event, pos)

        except Exception as e:
            logger.error(f"Manual position monitor error for {pos.symbol}: {e}", exc_info=True)
```

- [ ] **Step 4: _cmd_status에 활성 수동 포지션 표시 추가**

`telegram_commands.py`의 `_cmd_status` 메서드에서 수동 포지션도 표시하도록 수정:

```python
# _cmd_status 끝부분에 추가
if hasattr(bot, "position_manager"):
    positions = bot.position_manager.get_active_positions(self.chat_id)
    if positions:
        msg += "\n<b>\U0001f4cb 수동 포지션</b>\n"
        for p in positions:
            side_kr = "롱" if p.side == Side.LONG else "숏"
            msg += f"• {p.symbol} ({side_kr} {p.leverage}x) 평단 {p.entry_price:,.2f}\n"
```

- [ ] **Step 5: 전체 테스트 실행**

Run: `cd /Users/gowid/iker-trade-bot && python -m pytest tests/ -v --tb=short`
Expected: all PASS

- [ ] **Step 6: 커밋**

```bash
git add src/main.py src/review/telegram_commands.py
git commit -m "feat: integrate manual position monitoring into SignalBot scheduler"
```

---

### Task 7: 이벤트 중복 방지 + 쿨다운

**Files:**
- Modify: `src/strategy/position_monitor.py`

- [ ] **Step 1: PositionMonitor에 이벤트 쿨다운 로직 추가**

`detect_events` 반환 전에 중복 이벤트를 필터링:

```python
# PositionMonitor.detect_events 끝에 추가 (return events 전)
events = self._filter_cooldown(position.id, events)
return events

def _filter_cooldown(self, position_id: int, events: list[PositionEvent], cooldown_minutes: int = 30) -> list[PositionEvent]:
    """동일 이벤트 타입에 대해 쿨다운 적용."""
    now = datetime.now(timezone.utc)
    if position_id not in self._last_events:
        self._last_events[position_id] = {}

    filtered = []
    for event in events:
        last_time = self._last_events[position_id].get(event.event_type)
        # critical은 쿨다운 10분, 나머지 30분
        cd = 10 if event.severity == "critical" else cooldown_minutes
        if last_time is None or (now - last_time).total_seconds() >= cd * 60:
            filtered.append(event)
            self._last_events[position_id][event.event_type] = now

    return filtered

def clear_position(self, position_id: int) -> None:
    """포지션 청산 시 쿨다운 기록 제거."""
    self._last_events.pop(position_id, None)
```

- [ ] **Step 2: 테스트 추가**

```python
# tests/strategy/test_position_monitor.py에 추가

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
```

- [ ] **Step 3: 테스트 실행 → 통과 확인**

Run: `cd /Users/gowid/iker-trade-bot && python -m pytest tests/strategy/test_position_monitor.py -v`
Expected: all PASS

- [ ] **Step 4: 커밋**

```bash
git add src/strategy/position_monitor.py tests/strategy/test_position_monitor.py
git commit -m "feat: add event cooldown to prevent duplicate notifications"
```

---

### Task 8: 최종 통합 테스트 + 도움말 업데이트

**Files:**
- Modify: `src/review/telegram_commands.py` (도움말)

- [ ] **Step 1: 도움말 메시지 업데이트 확인**

_cmd_help의 msg에 "신규 포지션"과 "청산" 명령어가 포함되었는지 확인.

- [ ] **Step 2: 전체 테스트 실행**

Run: `cd /Users/gowid/iker-trade-bot && python -m pytest tests/ -v --tb=short`
Expected: all PASS

- [ ] **Step 3: 최종 커밋**

```bash
git add -A
git commit -m "feat: complete manual position monitoring with event push notifications"
```
