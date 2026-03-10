"""
Layer 1: Data validation.

Checks candle gaps, timestamp continuity, price anomalies, and data staleness.
Returns validation results that determine whether trading should continue.
"""

import logging
from datetime import datetime, timezone
from dataclasses import dataclass

from core.types import Candle
from core.safety import MAX_DATA_STALENESS_SECONDS
from data.collector import _interval_to_ms

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    is_valid: bool
    gaps: list[tuple[datetime, datetime]]  # (expected_time, next_actual_time)
    anomalies: list[tuple[datetime, str]]  # (timestamp, description)
    is_stale: bool
    stale_seconds: float


class DataValidator:
    """Validates market data quality before it's used by strategies."""

    PRICE_ANOMALY_THRESHOLD = 0.20  # 20% price change in one candle

    def validate_candles(
        self,
        candles: list[Candle],
        interval: str,
        now: datetime | None = None,
    ) -> ValidationResult:
        """Validate a list of candles for gaps, anomalies, and staleness."""
        if not candles:
            return ValidationResult(
                is_valid=False,
                gaps=[],
                anomalies=[(datetime.now(timezone.utc), "No candles provided")],
                is_stale=True,
                stale_seconds=float("inf"),
            )

        gaps = self._find_gaps(candles, interval)
        anomalies = self._find_anomalies(candles)
        is_stale, stale_seconds = self._check_staleness(candles, now)

        is_valid = len(gaps) == 0 and len(anomalies) == 0 and not is_stale

        if gaps:
            logger.warning(f"Found {len(gaps)} gaps in {candles[0].symbol} {interval} data")
        if anomalies:
            logger.warning(f"Found {len(anomalies)} anomalies in {candles[0].symbol} data")
        if is_stale:
            logger.warning(f"Data is stale by {stale_seconds:.0f}s for {candles[0].symbol}")

        return ValidationResult(
            is_valid=is_valid,
            gaps=gaps,
            anomalies=anomalies,
            is_stale=is_stale,
            stale_seconds=stale_seconds,
        )

    def _find_gaps(
        self, candles: list[Candle], interval: str
    ) -> list[tuple[datetime, datetime]]:
        """Find missing candles in the sequence."""
        if len(candles) < 2:
            return []

        interval_ms = _interval_to_ms(interval)
        # Allow 10% tolerance for timestamp alignment
        tolerance_ms = interval_ms * 0.1
        gaps = []

        for i in range(1, len(candles)):
            prev_ts = candles[i - 1].timestamp.timestamp() * 1000
            curr_ts = candles[i].timestamp.timestamp() * 1000
            expected_ts = prev_ts + interval_ms

            if curr_ts - expected_ts > tolerance_ms:
                gaps.append((
                    candles[i - 1].timestamp,
                    candles[i].timestamp,
                ))

        return gaps

    def _find_anomalies(
        self, candles: list[Candle]
    ) -> list[tuple[datetime, str]]:
        """Find price anomalies (extreme moves in single candle)."""
        anomalies = []

        for i, candle in enumerate(candles):
            # Check for zero or negative prices
            if candle.close <= 0 or candle.open <= 0:
                anomalies.append((
                    candle.timestamp,
                    f"Non-positive price: open={candle.open}, close={candle.close}",
                ))
                continue

            # Check for extreme price change within candle
            max_price = max(candle.open, candle.close)
            min_price = min(candle.open, candle.close)
            if min_price > 0:
                change = (max_price - min_price) / min_price
                if change > self.PRICE_ANOMALY_THRESHOLD:
                    anomalies.append((
                        candle.timestamp,
                        f"Extreme candle body: {change:.1%} change",
                    ))

            # Check high/low consistency
            if candle.high < candle.low:
                anomalies.append((
                    candle.timestamp,
                    f"High ({candle.high}) < Low ({candle.low})",
                ))

            if candle.high < max(candle.open, candle.close):
                anomalies.append((
                    candle.timestamp,
                    f"High ({candle.high}) < max(open, close)",
                ))

            if candle.low > min(candle.open, candle.close):
                anomalies.append((
                    candle.timestamp,
                    f"Low ({candle.low}) > min(open, close)",
                ))

            # Check for inter-candle extreme move
            if i > 0:
                prev_close = candles[i - 1].close
                if prev_close > 0:
                    gap_change = abs(candle.open - prev_close) / prev_close
                    if gap_change > self.PRICE_ANOMALY_THRESHOLD:
                        anomalies.append((
                            candle.timestamp,
                            f"Inter-candle gap: {gap_change:.1%} from prev close",
                        ))

        return anomalies

    def _check_staleness(
        self,
        candles: list[Candle],
        now: datetime | None = None,
    ) -> tuple[bool, float]:
        """Check if the most recent candle is too old."""
        if now is None:
            now = datetime.now(timezone.utc)

        latest = candles[-1].timestamp
        if latest.tzinfo is None:
            latest = latest.replace(tzinfo=timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        diff_seconds = (now - latest).total_seconds()
        is_stale = diff_seconds > MAX_DATA_STALENESS_SECONDS

        return is_stale, diff_seconds
