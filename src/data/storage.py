"""
Layer 1: Data storage.

Stores and retrieves candles, trades, and performance metrics.
Uses SQLAlchemy for compatibility with both SQLite (testing) and PostgreSQL (production).
"""

import logging
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    create_engine,
    Column,
    String,
    Float,
    Integer,
    Boolean,
    DateTime,
    JSON,
    text,
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session

from core.types import Candle, Trade, Side, StrategyName

logger = logging.getLogger(__name__)

Base = declarative_base()


class CandleRow(Base):
    __tablename__ = "candles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    interval = Column(String(10), nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)


class FundingRateRow(Base):
    __tablename__ = "funding_rates"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    rate = Column(Float, nullable=False)


class TradeRow(Base):
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(String(10), nullable=False)
    strategy = Column(String(30), nullable=False)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=False)
    quantity = Column(Float, nullable=False)
    leverage = Column(Float, nullable=False)
    entry_time = Column(DateTime, nullable=False, index=True)
    exit_time = Column(DateTime, nullable=False)
    pnl = Column(Float, nullable=False)
    pnl_percent = Column(Float, nullable=False)
    fees = Column(Float, nullable=False)
    slippage = Column(Float, nullable=False)
    stop_loss_hit = Column(Boolean, nullable=False)
    trailing_stop_hit = Column(Boolean, nullable=False)
    metadata_json = Column(JSON, default=dict)


class PerformanceRow(Base):
    __tablename__ = "performance"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(DateTime, nullable=False, index=True)
    total_capital = Column(Float, nullable=False)
    daily_pnl = Column(Float, nullable=False)
    daily_pnl_percent = Column(Float, nullable=False)
    cumulative_pnl = Column(Float, nullable=False)
    max_drawdown = Column(Float, nullable=False)
    open_positions = Column(Integer, nullable=False)
    trades_today = Column(Integer, nullable=False)
    win_rate = Column(Float, nullable=True)


class Storage:
    """Database interface for all persistent data."""

    def __init__(self, db_url: str = "sqlite:///futures_bot.db"):
        self.engine = create_engine(db_url, echo=False)
        Base.metadata.create_all(self.engine)
        self._Session = sessionmaker(bind=self.engine)

    def _get_session(self) -> Session:
        return self._Session()

    # --- Candles ---

    def save_candles(self, candles: list[Candle]) -> int:
        """Save candles, skipping duplicates (same symbol+interval+timestamp)."""
        if not candles:
            return 0

        session = self._get_session()
        try:
            saved = 0
            for c in candles:
                exists = (
                    session.query(CandleRow)
                    .filter_by(symbol=c.symbol, interval=c.interval, timestamp=c.timestamp)
                    .first()
                )
                if not exists:
                    session.add(CandleRow(
                        timestamp=c.timestamp, symbol=c.symbol, interval=c.interval,
                        open=c.open, high=c.high, low=c.low, close=c.close, volume=c.volume,
                    ))
                    saved += 1
            session.commit()
            return saved
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get_candles(
        self,
        symbol: str,
        interval: str,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = None,
    ) -> list[Candle]:
        """Retrieve candles from database."""
        session = self._get_session()
        try:
            q = (
                session.query(CandleRow)
                .filter_by(symbol=symbol, interval=interval)
                .order_by(CandleRow.timestamp.asc())
            )
            if start:
                q = q.filter(CandleRow.timestamp >= start)
            if end:
                q = q.filter(CandleRow.timestamp <= end)
            if limit:
                q = q.limit(limit)

            return [
                Candle(
                    timestamp=row.timestamp,
                    open=row.open, high=row.high, low=row.low,
                    close=row.close, volume=row.volume,
                    symbol=row.symbol, interval=row.interval,
                )
                for row in q.all()
            ]
        finally:
            session.close()

    # --- Trades ---

    def save_trade(self, trade: Trade) -> None:
        """Save a completed trade."""
        session = self._get_session()
        try:
            session.add(TradeRow(
                symbol=trade.symbol,
                side=trade.side.value,
                strategy=trade.strategy.value,
                entry_price=trade.entry_price,
                exit_price=trade.exit_price,
                quantity=trade.quantity,
                leverage=trade.leverage,
                entry_time=trade.entry_time,
                exit_time=trade.exit_time,
                pnl=trade.pnl,
                pnl_percent=trade.pnl_percent,
                fees=trade.fees,
                slippage=trade.slippage,
                stop_loss_hit=trade.stop_loss_hit,
                trailing_stop_hit=trade.trailing_stop_hit,
                metadata_json=trade.metadata,
            ))
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get_trades(
        self,
        symbol: str | None = None,
        strategy: StrategyName | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[Trade]:
        """Retrieve trades with optional filters."""
        session = self._get_session()
        try:
            q = session.query(TradeRow).order_by(TradeRow.entry_time.asc())
            if symbol:
                q = q.filter_by(symbol=symbol)
            if strategy:
                q = q.filter_by(strategy=strategy.value)
            if start:
                q = q.filter(TradeRow.entry_time >= start)
            if end:
                q = q.filter(TradeRow.entry_time <= end)

            return [
                Trade(
                    symbol=row.symbol,
                    side=Side(row.side),
                    strategy=StrategyName(row.strategy),
                    entry_price=row.entry_price,
                    exit_price=row.exit_price,
                    quantity=row.quantity,
                    leverage=row.leverage,
                    entry_time=row.entry_time,
                    exit_time=row.exit_time,
                    pnl=row.pnl,
                    pnl_percent=row.pnl_percent,
                    fees=row.fees,
                    slippage=row.slippage,
                    stop_loss_hit=row.stop_loss_hit,
                    trailing_stop_hit=row.trailing_stop_hit,
                    metadata=row.metadata_json or {},
                )
                for row in q.all()
            ]
        finally:
            session.close()

    # --- Performance ---

    def save_performance(
        self,
        date: datetime,
        total_capital: float,
        daily_pnl: float,
        daily_pnl_percent: float,
        cumulative_pnl: float,
        max_drawdown: float,
        open_positions: int,
        trades_today: int,
        win_rate: float | None = None,
    ) -> None:
        """Save daily performance snapshot."""
        session = self._get_session()
        try:
            session.add(PerformanceRow(
                date=date,
                total_capital=total_capital,
                daily_pnl=daily_pnl,
                daily_pnl_percent=daily_pnl_percent,
                cumulative_pnl=cumulative_pnl,
                max_drawdown=max_drawdown,
                open_positions=open_positions,
                trades_today=trades_today,
                win_rate=win_rate,
            ))
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # --- Funding Rates ---

    def save_funding_rates(self, rates: list) -> int:
        """Save funding rates, skipping duplicates."""
        if not rates:
            return 0

        session = self._get_session()
        try:
            saved = 0
            for r in rates:
                exists = (
                    session.query(FundingRateRow)
                    .filter_by(symbol=r.symbol, timestamp=r.timestamp)
                    .first()
                )
                if not exists:
                    session.add(FundingRateRow(
                        timestamp=r.timestamp, symbol=r.symbol, rate=r.rate,
                    ))
                    saved += 1
            session.commit()
            return saved
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
