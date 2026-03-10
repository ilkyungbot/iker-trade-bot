"""
Layer 6: Trade logging.

Records every trade with full context for later analysis.
"""

import logging
from core.types import Trade
from data.storage import Storage

logger = logging.getLogger(__name__)


class TradeLogger:
    """Logs trades to persistent storage."""

    def __init__(self, storage: Storage):
        self.storage = storage

    def log_trade(self, trade: Trade) -> None:
        """Save a completed trade to database."""
        self.storage.save_trade(trade)
        logger.info(
            f"Logged trade: {trade.side.value} {trade.symbol} "
            f"PnL={trade.pnl:.2f} ({trade.pnl_percent:.1f}%) "
            f"Strategy={trade.strategy.value}"
        )
