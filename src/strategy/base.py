"""
Layer 3: Strategy base interface.
"""

from abc import ABC, abstractmethod
import pandas as pd
from core.types import SignalMessage


class Strategy(ABC):
    """Abstract base for all signal strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy identifier."""
        ...

    @abstractmethod
    def generate_signal(
        self,
        df: pd.DataFrame,
        symbol: str,
        **kwargs,
    ) -> SignalMessage | None:
        """
        Generate a signal based on 4H candle data with all features.

        Args:
            df: 4H candle DataFrame with all features added
            symbol: trading pair symbol
            **kwargs: strategy-specific parameters (e.g., latest_funding_rate)

        Returns:
            SignalMessage if action needed, None if no signal
        """
        ...
