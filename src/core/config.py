"""
Adjustable configuration loaded from environment variables.

Unlike safety.py, these values CAN be changed via .env or environment.
They are operational settings, not safety limits.
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

from core.types import TradingMode

load_dotenv()


@dataclass(frozen=True)
class BybitConfig:
    api_key: str
    api_secret: str
    testnet: bool

    @classmethod
    def from_env(cls) -> "BybitConfig":
        return cls(
            api_key=os.getenv("BYBIT_API_KEY", ""),
            api_secret=os.getenv("BYBIT_API_SECRET", ""),
            testnet=os.getenv("BYBIT_TESTNET", "true").lower() == "true",
        )


@dataclass(frozen=True)
class DatabaseConfig:
    url: str

    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        return cls(
            url=os.getenv("DATABASE_URL", "sqlite:///futures_bot.db"),
        )


@dataclass(frozen=True)
class TelegramConfig:
    bot_token: str
    chat_id: str

    @classmethod
    def from_env(cls) -> "TelegramConfig":
        return cls(
            bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
            chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
        )


@dataclass(frozen=True)
class TradingConfig:
    mode: TradingMode
    default_leverage: float  # default leverage when model has no opinion
    strategy_a_allocation: float  # fraction of capital for trend following
    strategy_b_allocation: float  # fraction of capital for funding rate
    max_pairs: int
    pair_rebalance_days: int  # how often to rebalance pair selection
    candle_intervals: tuple[str, ...]  # timeframes to collect
    primary_interval: str  # main decision timeframe
    trend_interval: str  # higher timeframe for trend context

    @classmethod
    def from_env(cls) -> "TradingConfig":
        paper = os.getenv("PAPER_TRADING", "true").lower() == "true"
        return cls(
            mode=TradingMode.PAPER if paper else TradingMode.LIVE,
            default_leverage=5.0,
            strategy_a_allocation=0.70,
            strategy_b_allocation=0.30,
            max_pairs=5,
            pair_rebalance_days=14,
            candle_intervals=("15", "60", "240"),  # 15m, 1H, 4H (Bybit format)
            primary_interval="60",
            trend_interval="240",
        )


@dataclass(frozen=True)
class AppConfig:
    bybit: BybitConfig
    database: DatabaseConfig
    telegram: TelegramConfig
    trading: TradingConfig

    @classmethod
    def from_env(cls) -> "AppConfig":
        return cls(
            bybit=BybitConfig.from_env(),
            database=DatabaseConfig.from_env(),
            telegram=TelegramConfig.from_env(),
            trading=TradingConfig.from_env(),
        )
