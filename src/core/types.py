"""
Core type definitions used across all layers.
"""

from dataclasses import dataclass, field
from datetime import datetime
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


class OrderType(Enum):
    LIMIT = "limit"
    MARKET = "market"


class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class TradingMode(Enum):
    PAPER = "paper"
    LIVE = "live"


class CircuitBreakerState(Enum):
    NORMAL = "normal"
    DAILY_HALT = "daily_halt"
    WEEKLY_HALT = "weekly_halt"
    SIZE_REDUCED = "size_reduced"
    FULL_STOP = "full_stop"


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


@dataclass
class Position:
    symbol: str
    side: Side
    entry_price: float
    quantity: float
    leverage: float
    stop_loss: float
    trailing_stop: float
    strategy: StrategyName
    entry_time: datetime
    unrealized_pnl: float = 0.0
    highest_pnl: float = 0.0  # for trailing stop tracking
    exchange_sl_order_id: str | None = None

    @property
    def notional_value(self) -> float:
        return self.entry_price * self.quantity

    @property
    def margin_used(self) -> float:
        return self.notional_value / self.leverage


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


@dataclass
class Order:
    symbol: str
    side: Side
    order_type: OrderType
    price: float
    quantity: float
    leverage: float
    status: OrderStatus = OrderStatus.PENDING
    order_id: str | None = None
    exchange_order_id: str | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    filled_at: datetime | None = None
    cancel_after_candles: int = 2  # cancel limit order after N candles


@dataclass
class PortfolioState:
    """Current state of the entire portfolio."""
    total_capital: float
    available_capital: float
    positions: list[Position] = field(default_factory=list)
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    peak_capital: float = 0.0
    current_mdd: float = 0.0
    consecutive_losses: int = 0
    consecutive_wins: int = 0
    circuit_breaker_state: CircuitBreakerState = CircuitBreakerState.NORMAL

    @property
    def total_margin_used(self) -> float:
        return sum(p.margin_used for p in self.positions)

    @property
    def position_count(self) -> int:
        return len(self.positions)


@dataclass(frozen=True)
class PairInfo:
    symbol: str
    volume_24h: float
    atr_percent: float  # ATR / price
    correlation_to_btc: float
    score: float  # ranking score
