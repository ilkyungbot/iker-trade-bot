"""Market regime classifier for 3-7x leverage trading.

ATR ratio 기반 시장 국면 분류.
- HIGH_VOL_TREND: 고변동 추세장 (ATR ratio > 1.3)
- LOW_VOL_RANGE: 저변동 횡보장 (ATR ratio < 0.7)
- TRANSITION: 변동성 전환 구간 (0.7 ~ 1.3)
"""

import logging
from dataclasses import dataclass
from enum import Enum

import pandas as pd

logger = logging.getLogger(__name__)

ATR_SHORT_WINDOW = 14
ATR_LONG_WINDOW = 50
HIGH_VOL_THRESHOLD = 1.3
LOW_VOL_THRESHOLD = 0.7
BTC_EMA_PERIOD = 200


class Regime(Enum):
    HIGH_VOL_TREND = "high_vol_trend"
    LOW_VOL_RANGE = "low_vol_range"
    TRANSITION = "transition"


_REGIME_MESSAGES = {
    Regime.HIGH_VOL_TREND: "고변동 추세장 — 추세추종 유리, 넓은 스탑 권장",
    Regime.LOW_VOL_RANGE: "저변동 횡보장 — 진입 자제 권장, 좁은 스탑",
    Regime.TRANSITION: "변동성 전환 구간 — 포지션 축소 권장",
}


@dataclass(frozen=True)
class RegimeState:
    regime: Regime
    atr_ratio: float
    btc_above_200ema: bool
    message: str


class MarketRegimeClassifier:
    """ATR ratio 기반 시장 국면 분류기."""

    def classify(
        self, df: pd.DataFrame, btc_df: pd.DataFrame | None = None
    ) -> RegimeState:
        """Classify market regime from ATR data.

        Args:
            df: DataFrame with 'atr' column (14-period ATR from features.py).
            btc_df: Optional BTC DataFrame with 'close' for 200 EMA filter.

        Returns:
            RegimeState with regime, atr_ratio, btc filter, and Korean message.
        """
        atr_ratio = self._compute_atr_ratio(df)
        btc_above = self._check_btc_200ema(btc_df)
        regime = self._classify_regime(atr_ratio)

        return RegimeState(
            regime=regime,
            atr_ratio=round(atr_ratio, 4),
            btc_above_200ema=btc_above,
            message=_REGIME_MESSAGES[regime],
        )

    def get_regime_advice(self, regime_state: RegimeState, side: str) -> dict:
        """Return regime-specific trading advice.

        Returns:
            dict with position_size_factor, stop_buffer_multiplier, warnings.
        """
        warnings: list[str] = []

        if regime_state.regime == Regime.HIGH_VOL_TREND:
            position_size_factor = 1.0
            stop_buffer_multiplier = 1.5
        elif regime_state.regime == Regime.LOW_VOL_RANGE:
            position_size_factor = 0.5
            stop_buffer_multiplier = 0.8
            warnings.append("횡보장 — 진입 자제 권장")
        else:  # TRANSITION
            position_size_factor = 0.5
            stop_buffer_multiplier = 1.0
            warnings.append("변동성 전환 구간 — 모니터링 강화")

        if not regime_state.btc_above_200ema and side == "long":
            warnings.append("BTC가 200 EMA 하회 중 — 롱 진입 주의")

        return {
            "position_size_factor": position_size_factor,
            "stop_buffer_multiplier": stop_buffer_multiplier,
            "warnings": warnings,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_atr_ratio(df: pd.DataFrame) -> float:
        """ATR short / ATR long ratio."""
        atr = df["atr"]
        n = len(atr)

        short_window = min(ATR_SHORT_WINDOW, n)
        long_window = min(ATR_LONG_WINDOW, n)

        atr_short = atr.iloc[-short_window:].mean()
        atr_long = atr.iloc[-long_window:].mean()

        if atr_long == 0:
            logger.warning("ATR long mean is 0, returning ratio 1.0")
            return 1.0

        return atr_short / atr_long

    @staticmethod
    def _check_btc_200ema(btc_df: pd.DataFrame | None) -> bool:
        """Check if BTC price is above 200 EMA. Defaults True if no data."""
        if btc_df is None or btc_df.empty:
            return True

        if len(btc_df) < BTC_EMA_PERIOD:
            # Not enough data for reliable EMA; default to True
            return True

        ema_200 = btc_df["close"].ewm(span=BTC_EMA_PERIOD, adjust=False).mean()
        return bool(btc_df["close"].iloc[-1] > ema_200.iloc[-1])

    @staticmethod
    def _classify_regime(atr_ratio: float) -> Regime:
        if atr_ratio > HIGH_VOL_THRESHOLD:
            return Regime.HIGH_VOL_TREND
        elif atr_ratio < LOW_VOL_THRESHOLD:
            return Regime.LOW_VOL_RANGE
        else:
            return Regime.TRANSITION
