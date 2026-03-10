"""
Backtesting engine.

Simulates trading strategies on historical candle data.
Walks through candles bar-by-bar, generating signals and tracking positions.
No lookahead bias — only uses data up to the current bar.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta

import pandas as pd
import numpy as np

from core.types import (
    Trade, Position, Side, Signal, SignalAction, StrategyName,
)
from core.safety import MAX_LEVERAGE, MAX_RISK_PER_TRADE
from data.features import add_all_features, add_ema
from strategy.trend_following import TrendFollowingStrategy
from strategy.funding_rate import FundingRateStrategy
from sizing.kelly import kelly_fraction, calculate_position_size
from sizing.adjustments import apply_all_adjustments

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for a backtest run."""
    initial_capital: float = 100_000.0
    strategy_a_allocation: float = 0.7
    strategy_b_allocation: float = 0.3
    default_leverage: float = 5.0
    fee_rate: float = 0.0004  # 0.04% per side (0.08% round trip)
    slippage_rate: float = 0.0001  # 0.01% slippage per trade
    max_positions: int = 5
    # Simulated funding rates (list of floats or None to disable)
    funding_rates: list[float] | None = None


@dataclass
class BacktestResult:
    """Results of a completed backtest."""
    trades: list[Trade]
    equity_curve: list[float]  # capital at each bar
    timestamps: list[datetime]
    initial_capital: float
    final_capital: float
    total_bars: int

    @property
    def total_return(self) -> float:
        if self.initial_capital <= 0:
            return 0.0
        return (self.final_capital - self.initial_capital) / self.initial_capital

    @property
    def total_return_pct(self) -> float:
        return self.total_return * 100


@dataclass
class _OpenPosition:
    """Internal position tracking during backtest."""
    symbol: str
    side: Side
    entry_price: float
    quantity: float
    stop_loss: float
    take_profit: float
    trailing_stop: float
    strategy: StrategyName
    entry_time: datetime
    entry_bar: int
    highest_pnl: float = 0.0


class BacktestRunner:
    """
    Bar-by-bar backtesting engine.

    Usage:
        runner = BacktestRunner(config)
        result = runner.run(df_1h, df_4h, symbol)
    """

    def __init__(self, config: BacktestConfig | None = None):
        self.config = config or BacktestConfig()
        self.trend_strategy = TrendFollowingStrategy()
        self.funding_strategy = FundingRateStrategy()

    def run(
        self,
        df_1h: pd.DataFrame,
        df_4h: pd.DataFrame,
        symbol: str = "BTCUSDT",
    ) -> BacktestResult:
        """
        Run backtest on historical data.

        Args:
            df_1h: 1H candle DataFrame with OHLCV columns (features will be added)
            df_4h: 4H candle DataFrame with OHLCV columns
            symbol: symbol name for logging

        Returns:
            BacktestResult with trades, equity curve, and summary stats
        """
        # Add features if not present
        if "atr" not in df_1h.columns:
            df_1h = add_all_features(df_1h)
        if "ema_50" not in df_4h.columns:
            df_4h = add_ema(df_4h, 50)

        capital = self.config.initial_capital
        positions: list[_OpenPosition] = []
        trades: list[Trade] = []
        equity_curve: list[float] = []
        timestamps: list[datetime] = []

        # Need enough warmup bars for indicators
        warmup = 55  # enough for EMA 50 + 5
        if len(df_1h) <= warmup:
            return BacktestResult(
                trades=[], equity_curve=[capital], timestamps=[],
                initial_capital=capital, final_capital=capital, total_bars=0,
            )

        # Build a mapping from 1h timestamp to closest 4h bar index
        timestamps_4h = df_4h["timestamp"].values if "timestamp" in df_4h.columns else None

        for i in range(warmup, len(df_1h)):
            bar = df_1h.iloc[i]
            bar_time = bar.get("timestamp", datetime(2024, 1, 1, tzinfo=timezone.utc))
            if not isinstance(bar_time, datetime):
                bar_time = datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i)

            close = bar["close"]
            high = bar["high"]
            low = bar["low"]

            # 1. Check stops for open positions
            closed_symbols = set()
            for pos in list(positions):
                hit_sl = False
                hit_tp = False

                if pos.side == Side.LONG:
                    if low <= pos.stop_loss:
                        hit_sl = True
                    if high >= pos.trailing_stop:
                        hit_tp = True
                else:  # SHORT
                    if high >= pos.stop_loss:
                        hit_sl = True
                    if low <= pos.trailing_stop:
                        hit_tp = True

                if hit_sl or hit_tp:
                    exit_price = pos.stop_loss if hit_sl else pos.trailing_stop
                    trade = self._close_position(pos, exit_price, bar_time, hit_sl, hit_tp)
                    capital += trade.pnl
                    trades.append(trade)
                    positions.remove(pos)
                    closed_symbols.add(pos.symbol)

            # 2. Update trailing stops for remaining positions
            for pos in positions:
                atr_val = bar.get("atr", 0)
                if pd.isna(atr_val) or atr_val <= 0:
                    continue
                if pos.side == Side.LONG:
                    unrealized = (close - pos.entry_price) * pos.quantity
                    new_trail = close - atr_val * 2.0
                    if unrealized > pos.highest_pnl:
                        pos.highest_pnl = unrealized
                        if new_trail > pos.trailing_stop:
                            pos.trailing_stop = new_trail
                else:
                    unrealized = (pos.entry_price - close) * pos.quantity
                    new_trail = close + atr_val * 2.0
                    if unrealized > pos.highest_pnl:
                        pos.highest_pnl = unrealized
                        if new_trail < pos.trailing_stop:
                            pos.trailing_stop = new_trail

            # 3. Generate signals (only if not at max positions)
            if len(positions) < self.config.max_positions:
                # Current position side for this symbol
                current_side = None
                for p in positions:
                    if p.symbol == symbol:
                        current_side = p.side.value

                # Slice data up to current bar (no lookahead)
                df_1h_slice = df_1h.iloc[:i + 1]

                # Find matching 4H slice
                df_4h_slice = df_4h
                if timestamps_4h is not None and "timestamp" in df_1h.columns:
                    mask = df_4h["timestamp"] <= bar_time
                    if mask.any():
                        df_4h_slice = df_4h[mask]

                # Strategy A: Trend Following
                signal_a = self.trend_strategy.generate_signal(
                    df_1h_slice, df_4h_slice, symbol, current_side,
                )

                # Strategy B: Funding Rate
                funding_rate = None
                if self.config.funding_rates:
                    fr_idx = min(i - warmup, len(self.config.funding_rates) - 1)
                    funding_rate = self.config.funding_rates[fr_idx]

                signal_b = self.funding_strategy.generate_signal(
                    df_1h_slice, df_4h_slice, symbol, current_side,
                    latest_funding_rate=funding_rate,
                )

                # Process signals
                for signal, allocation in [
                    (signal_a, self.config.strategy_a_allocation),
                    (signal_b, self.config.strategy_b_allocation),
                ]:
                    if signal is None:
                        continue
                    if signal.action == SignalAction.EXIT:
                        # Find and close position
                        for pos in list(positions):
                            if pos.symbol == symbol:
                                trade = self._close_position(pos, close, bar_time, False, False)
                                capital += trade.pnl
                                trades.append(trade)
                                positions.remove(pos)
                                break
                    elif signal.action in (SignalAction.ENTER_LONG, SignalAction.ENTER_SHORT):
                        # Skip if already have a position in this symbol
                        if any(p.symbol == symbol for p in positions):
                            continue

                        pos = self._open_position(
                            signal, capital, allocation, bar_time, i,
                        )
                        if pos is not None:
                            positions.append(pos)

            # Track equity (capital + unrealized PnL)
            unrealized = 0.0
            for pos in positions:
                if pos.side == Side.LONG:
                    unrealized += (close - pos.entry_price) * pos.quantity
                else:
                    unrealized += (pos.entry_price - close) * pos.quantity
            equity_curve.append(capital + unrealized)
            timestamps.append(bar_time)

        # Close any remaining positions at last bar
        last_close = df_1h.iloc[-1]["close"]
        last_time = df_1h.iloc[-1].get("timestamp", datetime.now(timezone.utc))
        if not isinstance(last_time, datetime):
            last_time = datetime.now(timezone.utc)

        for pos in positions:
            trade = self._close_position(pos, last_close, last_time, False, False)
            capital += trade.pnl
            trades.append(trade)

        return BacktestResult(
            trades=trades,
            equity_curve=equity_curve,
            timestamps=timestamps,
            initial_capital=self.config.initial_capital,
            final_capital=capital,
            total_bars=len(df_1h) - warmup,
        )

    def _open_position(
        self, signal: Signal, capital: float, allocation: float,
        bar_time: datetime, bar_idx: int,
    ) -> _OpenPosition | None:
        """Create a new position from a signal."""
        risk_fraction = min(MAX_RISK_PER_TRADE, 0.01)  # conservative for backtest
        capital_for_strategy = capital * allocation

        quantity = calculate_position_size(
            capital=capital_for_strategy,
            risk_fraction=risk_fraction,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            leverage=self.config.default_leverage,
        )

        if quantity <= 0:
            return None

        # Apply slippage to entry
        slippage = signal.entry_price * self.config.slippage_rate
        if signal.action == SignalAction.ENTER_LONG:
            entry_price = signal.entry_price + slippage
        else:
            entry_price = signal.entry_price - slippage

        return _OpenPosition(
            symbol=signal.symbol,
            side=Side.LONG if signal.action == SignalAction.ENTER_LONG else Side.SHORT,
            entry_price=entry_price,
            quantity=quantity,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            trailing_stop=signal.take_profit,
            strategy=signal.strategy,
            entry_time=bar_time,
            entry_bar=bar_idx,
        )

    def _close_position(
        self, pos: _OpenPosition, exit_price: float,
        exit_time: datetime, stop_loss_hit: bool, trailing_stop_hit: bool,
    ) -> Trade:
        """Close a position and return the resulting Trade."""
        # Apply slippage to exit
        slippage = exit_price * self.config.slippage_rate
        if pos.side == Side.LONG:
            actual_exit = exit_price - slippage
            pnl = (actual_exit - pos.entry_price) * pos.quantity
        else:
            actual_exit = exit_price + slippage
            pnl = (pos.entry_price - actual_exit) * pos.quantity

        # Fees
        entry_fee = pos.entry_price * pos.quantity * self.config.fee_rate
        exit_fee = actual_exit * pos.quantity * self.config.fee_rate
        total_fees = entry_fee + exit_fee
        pnl -= total_fees

        # PnL percent (relative to margin used)
        margin = pos.entry_price * pos.quantity / self.config.default_leverage
        pnl_pct = pnl / margin if margin > 0 else 0

        return Trade(
            symbol=pos.symbol,
            side=pos.side,
            strategy=pos.strategy,
            entry_price=pos.entry_price,
            exit_price=actual_exit,
            quantity=pos.quantity,
            leverage=self.config.default_leverage,
            entry_time=pos.entry_time,
            exit_time=exit_time,
            pnl=pnl,
            pnl_percent=pnl_pct,
            fees=total_fees,
            slippage=slippage * pos.quantity * 2,
            stop_loss_hit=stop_loss_hit,
            trailing_stop_hit=trailing_stop_hit,
        )
