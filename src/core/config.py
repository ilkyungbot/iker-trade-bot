"""
Adjustable configuration loaded from environment variables.

Unlike safety.py, these values CAN be changed via .env or environment.
They are operational settings, not safety limits.
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

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
            testnet=False,  # 항상 메인넷 사용 (시세 조회 전용)
        )


@dataclass(frozen=True)
class DatabaseConfig:
    url: str

    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        return cls(
            url=os.getenv("DATABASE_URL", "sqlite:///signal_bot.db"),
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
class SignalConfig:
    signal_cooldown_minutes: int
    monitoring_interval_minutes: int
    signal_expiry_minutes: int
    min_signal_quality: str
    primary_interval: str         # 4H 봉 기본
    candle_intervals: tuple[str, ...]
    max_pairs: int
    pair_rebalance_days: int

    @classmethod
    def from_env(cls) -> "SignalConfig":
        return cls(
            signal_cooldown_minutes=int(os.getenv("SIGNAL_COOLDOWN_MINUTES", "30")),
            monitoring_interval_minutes=int(os.getenv("MONITORING_INTERVAL_MINUTES", "15")),
            signal_expiry_minutes=int(os.getenv("SIGNAL_EXPIRY_MINUTES", "60")),
            min_signal_quality=os.getenv("MIN_SIGNAL_QUALITY", "moderate"),
            primary_interval=os.getenv("PRIMARY_INTERVAL", "240"),
            candle_intervals=("15", "60", "240"),
            max_pairs=int(os.getenv("MAX_PAIRS", "5")),
            pair_rebalance_days=int(os.getenv("PAIR_REBALANCE_DAYS", "14")),
        )


@dataclass(frozen=True)
class AppConfig:
    bybit: BybitConfig
    database: DatabaseConfig
    telegram: TelegramConfig
    signal: SignalConfig

    @classmethod
    def from_env(cls) -> "AppConfig":
        return cls(
            bybit=BybitConfig.from_env(),
            database=DatabaseConfig.from_env(),
            telegram=TelegramConfig.from_env(),
            signal=SignalConfig.from_env(),
        )
