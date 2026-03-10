"""
Layer 2: Automatic pair selection.

Selects top trading pairs based on volume, volatility (ATR%), and correlation constraints.
Rebalances every 2 weeks.
"""

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from core.safety import MIN_PAIR_VOLUME_24H
from core.types import PairInfo

logger = logging.getLogger(__name__)

# Minimum months of 1H data required
MIN_HISTORY_MONTHS = 6

# Maximum pairs with high correlation to avoid concentration
MAX_HIGH_CORR_PAIRS = 3
HIGH_CORR_THRESHOLD = 0.85


class PairSelector:
    """Selects and ranks trading pairs."""

    def __init__(self, max_pairs: int = 5, rebalance_days: int = 14):
        self.max_pairs = max_pairs
        self.rebalance_days = rebalance_days
        self._last_rebalance: datetime | None = None
        self._current_pairs: list[PairInfo] = []

    def select_pairs(
        self,
        tickers: list[dict],
        candle_data: dict[str, pd.DataFrame],
        now: datetime | None = None,
    ) -> list[PairInfo]:
        """
        Select top pairs from available tickers.

        Args:
            tickers: list of {"symbol": str, "volume_24h": float, "last_price": float}
            candle_data: dict of symbol -> DataFrame with OHLCV + features (must include "atr_pct")
            now: current time for rebalance check

        Returns:
            list of PairInfo, sorted by score descending
        """
        now = now or datetime.now(tz=None)

        # Check if rebalance is needed
        if self._should_skip_rebalance(now):
            return self._current_pairs

        # Step 1: Filter by minimum volume
        eligible = [t for t in tickers if t["volume_24h"] >= MIN_PAIR_VOLUME_24H]
        logger.info(f"Pairs with volume >= ${MIN_PAIR_VOLUME_24H:,.0f}: {len(eligible)}")

        # Step 2: Filter by minimum history
        eligible = [
            t for t in eligible
            if t["symbol"] in candle_data and len(candle_data[t["symbol"]]) >= MIN_HISTORY_MONTHS * 30 * 24
        ]
        logger.info(f"Pairs with sufficient history: {len(eligible)}")

        if not eligible:
            logger.warning("No eligible pairs found!")
            return []

        # Step 3: Calculate scores
        pair_infos = []
        for t in eligible:
            symbol = t["symbol"]
            df = candle_data[symbol]

            atr_pct = df["atr_pct"].iloc[-1] if "atr_pct" in df.columns else 0.0
            if pd.isna(atr_pct) or atr_pct <= 0:
                continue

            # Normalize volume to 0-1 scale relative to max
            max_vol = max(x["volume_24h"] for x in eligible)
            vol_score = t["volume_24h"] / max_vol if max_vol > 0 else 0

            # Score = ATR% × volume_score (higher volatility + higher liquidity = better)
            score = atr_pct * vol_score

            # Calculate correlation to BTC
            btc_corr = 1.0  # default if BTC data not available
            if "BTCUSDT" in candle_data and symbol != "BTCUSDT":
                btc_corr = self._calculate_correlation(
                    candle_data["BTCUSDT"]["close"],
                    df["close"],
                )

            pair_infos.append(PairInfo(
                symbol=symbol,
                volume_24h=t["volume_24h"],
                atr_percent=atr_pct,
                correlation_to_btc=btc_corr,
                score=score,
            ))

        # Step 4: Sort by score
        pair_infos.sort(key=lambda p: p.score, reverse=True)

        # Step 5: Apply correlation constraint
        selected = self._apply_correlation_constraint(pair_infos)

        self._current_pairs = selected
        self._last_rebalance = now
        logger.info(f"Selected pairs: {[p.symbol for p in selected]}")

        return selected

    def _should_skip_rebalance(self, now: datetime) -> bool:
        """Check if we should skip rebalance (not enough time passed)."""
        if not self._last_rebalance or not self._current_pairs:
            return False
        elapsed = (now - self._last_rebalance).days
        return elapsed < self.rebalance_days

    def _apply_correlation_constraint(
        self, ranked_pairs: list[PairInfo]
    ) -> list[PairInfo]:
        """Select pairs while respecting correlation limits."""
        selected: list[PairInfo] = []
        high_corr_count = 0  # pairs with correlation > threshold to BTC

        for pair in ranked_pairs:
            if len(selected) >= self.max_pairs:
                break

            is_high_corr = abs(pair.correlation_to_btc) > HIGH_CORR_THRESHOLD
            if is_high_corr:
                if high_corr_count >= MAX_HIGH_CORR_PAIRS:
                    continue
                high_corr_count += 1

            selected.append(pair)

        return selected

    @staticmethod
    def _calculate_correlation(
        series_a: pd.Series, series_b: pd.Series, window: int = 168
    ) -> float:
        """Calculate rolling correlation between two price series (last `window` candles)."""
        # Align by taking the shorter length
        min_len = min(len(series_a), len(series_b))
        if min_len < window:
            window = max(min_len, 2)

        a = series_a.iloc[-window:].reset_index(drop=True)
        b = series_b.iloc[-window:].reset_index(drop=True)

        # Use returns instead of raw prices for meaningful correlation
        a_ret = a.pct_change().dropna()
        b_ret = b.pct_change().dropna()

        if len(a_ret) < 2 or len(b_ret) < 2:
            return 1.0

        min_len = min(len(a_ret), len(b_ret))
        corr = np.corrcoef(a_ret.iloc[-min_len:], b_ret.iloc[-min_len:])[0, 1]
        return float(corr) if not np.isnan(corr) else 1.0
