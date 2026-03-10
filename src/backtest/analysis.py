"""
Backtest analysis utilities.

Compute performance metrics and summary statistics from backtest results.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from backtest.runner import BacktestResult
from review.performance import calculate_metrics, PerformanceMetrics

logger = logging.getLogger(__name__)


@dataclass
class BacktestAnalysis:
    """Extended analysis of backtest results."""
    metrics: PerformanceMetrics
    total_return_pct: float
    annualized_return_pct: float
    max_drawdown_pct: float
    calmar_ratio: float  # annualized return / max drawdown
    num_bars: int
    avg_bars_per_trade: float
    win_streak: int
    loss_streak: int


def analyze(result: BacktestResult, bars_per_year: float = 8760) -> BacktestAnalysis:
    """
    Analyze backtest results.

    Args:
        result: BacktestResult from runner
        bars_per_year: number of bars in a year (8760 for 1H candles)
    """
    metrics = calculate_metrics(result.trades)

    total_return_pct = result.total_return_pct

    # Annualize
    if result.total_bars > 0:
        years = result.total_bars / bars_per_year
        if years > 0 and result.final_capital > 0 and result.initial_capital > 0:
            annualized = (result.final_capital / result.initial_capital) ** (1 / years) - 1
            annualized_return_pct = annualized * 100
        else:
            annualized_return_pct = 0.0
    else:
        annualized_return_pct = 0.0

    # Max drawdown from equity curve
    if result.equity_curve:
        curve = np.array(result.equity_curve)
        peak = np.maximum.accumulate(curve)
        drawdowns = (peak - curve) / np.where(peak > 0, peak, 1)
        max_dd_pct = float(np.max(drawdowns)) * 100 if len(drawdowns) > 0 else 0.0
    else:
        max_dd_pct = 0.0

    # Calmar ratio
    calmar = annualized_return_pct / max_dd_pct if max_dd_pct > 0 else 0.0

    # Average bars per trade
    avg_bars = result.total_bars / len(result.trades) if result.trades else 0

    # Streaks
    win_streak = 0
    loss_streak = 0
    current_win = 0
    current_loss = 0
    for t in result.trades:
        if t.pnl > 0:
            current_win += 1
            current_loss = 0
            win_streak = max(win_streak, current_win)
        else:
            current_loss += 1
            current_win = 0
            loss_streak = max(loss_streak, current_loss)

    return BacktestAnalysis(
        metrics=metrics,
        total_return_pct=total_return_pct,
        annualized_return_pct=annualized_return_pct,
        max_drawdown_pct=max_dd_pct,
        calmar_ratio=calmar,
        num_bars=result.total_bars,
        avg_bars_per_trade=avg_bars,
        win_streak=win_streak,
        loss_streak=loss_streak,
    )


def format_report(analysis: BacktestAnalysis) -> str:
    """Format backtest analysis as a readable report string."""
    m = analysis.metrics
    lines = [
        "=" * 50,
        "BACKTEST REPORT",
        "=" * 50,
        f"Total Return:       {analysis.total_return_pct:+.2f}%",
        f"Annualized Return:  {analysis.annualized_return_pct:+.2f}%",
        f"Max Drawdown:       {analysis.max_drawdown_pct:.2f}%",
        f"Calmar Ratio:       {analysis.calmar_ratio:.2f}",
        "",
        f"Total Trades:       {m.total_trades}",
        f"Win Rate:           {m.win_rate:.1%}",
        f"Profit Factor:      {m.profit_factor:.2f}",
        f"Sharpe Ratio:       {m.sharpe_ratio:.2f}",
        f"Payoff Ratio:       {m.payoff_ratio:.2f}",
        "",
        f"Avg Win:            {m.avg_win:+.2f}",
        f"Avg Loss:           {m.avg_loss:.2f}",
        f"Total PnL:          {m.total_pnl:+.2f}",
        f"Total Fees:         {m.total_fees:.2f}",
        "",
        f"Win Streak:         {analysis.win_streak}",
        f"Loss Streak:        {analysis.loss_streak}",
        f"Bars:               {analysis.num_bars}",
        "=" * 50,
    ]
    return "\n".join(lines)
