"""
Layer 6: Performance analysis.

Calculates key metrics from trade history for self-improvement decisions.
"""

import logging
from dataclasses import dataclass
from datetime import datetime

from core.types import Trade, StrategyName

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    payoff_ratio: float  # avg_win / avg_loss
    total_pnl: float
    total_fees: float
    net_pnl: float
    max_drawdown: float
    sharpe_ratio: float
    profit_factor: float  # gross profit / gross loss
    avg_trade_duration_hours: float


def calculate_metrics(trades: list[Trade]) -> PerformanceMetrics:
    """Calculate comprehensive performance metrics from trade list."""
    if not trades:
        return PerformanceMetrics(
            total_trades=0, winning_trades=0, losing_trades=0,
            win_rate=0, avg_win=0, avg_loss=0, payoff_ratio=0,
            total_pnl=0, total_fees=0, net_pnl=0,
            max_drawdown=0, sharpe_ratio=0, profit_factor=0,
            avg_trade_duration_hours=0,
        )

    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]

    total_trades = len(trades)
    winning_trades = len(wins)
    losing_trades = len(losses)
    win_rate = winning_trades / total_trades if total_trades > 0 else 0

    avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0
    avg_loss = abs(sum(t.pnl for t in losses) / len(losses)) if losses else 0
    payoff_ratio = avg_win / avg_loss if avg_loss > 0 else float("inf")

    total_pnl = sum(t.pnl for t in trades)
    total_fees = sum(t.fees for t in trades)
    net_pnl = total_pnl

    # Max drawdown from cumulative PnL
    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    for t in trades:
        cumulative += t.pnl
        if cumulative > peak:
            peak = cumulative
        dd = peak - cumulative
        if dd > max_dd:
            max_dd = dd
    max_drawdown = max_dd

    # Sharpe ratio (simplified: mean return / std return)
    returns = [t.pnl_percent for t in trades]
    if len(returns) > 1:
        import numpy as np
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        sharpe_ratio = float(mean_ret / std_ret) if std_ret > 0 else 0
    else:
        sharpe_ratio = 0

    # Profit factor
    gross_profit = sum(t.pnl for t in wins)
    gross_loss = abs(sum(t.pnl for t in losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Average trade duration
    durations = []
    for t in trades:
        dur = (t.exit_time - t.entry_time).total_seconds() / 3600
        durations.append(dur)
    avg_duration = sum(durations) / len(durations) if durations else 0

    return PerformanceMetrics(
        total_trades=total_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        payoff_ratio=payoff_ratio,
        total_pnl=total_pnl,
        total_fees=total_fees,
        net_pnl=net_pnl,
        max_drawdown=max_drawdown,
        sharpe_ratio=sharpe_ratio,
        profit_factor=profit_factor,
        avg_trade_duration_hours=avg_duration,
    )


def calculate_strategy_attribution(
    trades: list[Trade],
) -> dict[str, PerformanceMetrics]:
    """Calculate metrics per strategy."""
    by_strategy: dict[str, list[Trade]] = {}
    for t in trades:
        key = t.strategy.value
        if key not in by_strategy:
            by_strategy[key] = []
        by_strategy[key].append(t)

    return {name: calculate_metrics(strat_trades) for name, strat_trades in by_strategy.items()}


def calculate_pair_attribution(
    trades: list[Trade],
) -> dict[str, PerformanceMetrics]:
    """Calculate metrics per trading pair."""
    by_pair: dict[str, list[Trade]] = {}
    for t in trades:
        if t.symbol not in by_pair:
            by_pair[t.symbol] = []
        by_pair[t.symbol].append(t)

    return {symbol: calculate_metrics(pair_trades) for symbol, pair_trades in by_pair.items()}
