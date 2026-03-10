"""
Layer 3: Strategy base interface.
"""

from abc import ABC, abstractmethod
import pandas as pd
from core.types import Signal


class Strategy(ABC):
    """Abstract base for all trading strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy identifier."""
        ...

    @abstractmethod
    def generate_signal(
        self,
        df_1h: pd.DataFrame,
        df_4h: pd.DataFrame,
        symbol: str,
        current_position_side: str | None = None,
    ) -> Signal | None:
        """
        Generate a trading signal based on candle data with features.

        Args:
            df_1h: 1H candle DataFrame with all features added
            df_4h: 4H candle DataFrame with EMA features
            symbol: trading pair symbol
            current_position_side: "long", "short", or None if no position

        Returns:
            Signal if action needed, None if HOLD
        """
        ...
