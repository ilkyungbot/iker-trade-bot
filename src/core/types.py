"""
Core type definitions used across all layers.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class Side(Enum):
    LONG = "long"
    SHORT = "short"


class SignalAction(Enum):
    ENTER_LONG = "enter_long"
    ENTER_SHORT = "enter_short"
    EXIT = "exit"
    HOLD = "hold"


class StrategyName(Enum):
    TREND_FOLLOWING = "trend_following"
    FUNDING_RATE = "funding_rate"


class ConversationState(Enum):
    IDLE = "idle"
    MONITORING = "monitoring"


class SignalQuality(Enum):
    STRONG = "strong"      # 3+ 지표 일치
    MODERATE = "moderate"  # 2개 지표 일치
    WEAK = "weak"          # 1개만


@dataclass(frozen=True)
class Candle:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str
    interval: str  # "1h", "4h", "15m"


@dataclass(frozen=True)
class FundingRate:
    timestamp: datetime
    symbol: str
    rate: float  # e.g., 0.0001 = 0.01%


@dataclass(frozen=True)
class OpenInterest:
    timestamp: datetime
    symbol: str
    value: float  # USD value


@dataclass(frozen=True)
class Signal:
    timestamp: datetime
    symbol: str
    action: SignalAction
    strategy: StrategyName
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float  # 0.0 to 1.0
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class SignalMessage:
    signal: Signal
    quality: SignalQuality
    explanation: list[str]       # 한국어 사유 목록
    indicators: dict             # 지표 스냅샷
    risk_reward_ratio: float


@dataclass
class UserSession:
    chat_id: str
    state: ConversationState
    active_signal: SignalMessage | None = None
    entry_confirmed_at: datetime | None = None
    user_entry_price: float | None = None


@dataclass(frozen=True)
class Trade:
    """A completed trade (position opened and closed)."""
    symbol: str
    side: Side
    strategy: StrategyName
    entry_price: float
    exit_price: float
    quantity: float
    leverage: float
    entry_time: datetime
    exit_time: datetime
    pnl: float  # realized PnL in USDT
    pnl_percent: float  # PnL as % of margin
    fees: float
    slippage: float
    stop_loss_hit: bool
    trailing_stop_hit: bool
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class PairInfo:
    symbol: str
    volume_24h: float
    atr_percent: float  # ATR / price
    correlation_to_btc: float
    score: float  # ranking score


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
    stop_loss: float | None = None
    take_profit: float | None = None
    margin_usdt: float | None = None
    entry_reason: str = ""
