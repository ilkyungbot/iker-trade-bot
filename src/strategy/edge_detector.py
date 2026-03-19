"""Detects trading edges from funding rate extremes and OI changes."""

from dataclasses import dataclass

from core.types import FundingRate, OpenInterest


@dataclass(frozen=True)
class EdgeSignal:
    signal_type: str   # "funding_extreme", "oi_anomaly", "funding_trend"
    direction: str     # "bullish", "bearish", "neutral"
    strength: str      # "strong", "moderate", "weak"
    message: str       # Korean explanation
    data: dict         # raw values for display


class EdgeDetector:
    """Detect trading edges from funding rate extremes and OI changes."""

    # Thresholds expressed as raw rate values (not percentages).
    _EXTREME_MODERATE = 0.0005  # 0.05%
    _EXTREME_STRONG = 0.001    # 0.1%

    def detect_funding_extreme(
        self, funding_rates: list[FundingRate]
    ) -> EdgeSignal | None:
        """Check if latest funding rate is extreme (>= +/-0.05%)."""
        if not funding_rates:
            return None

        latest = funding_rates[-1]
        rate = latest.rate
        abs_rate = abs(rate)

        if abs_rate < self._EXTREME_MODERATE:
            return None

        strength = "strong" if abs_rate >= self._EXTREME_STRONG else "moderate"
        rate_pct = round(rate * 100, 4)

        if rate > 0:
            return EdgeSignal(
                signal_type="funding_extreme",
                direction="bearish",
                strength=strength,
                message=f"펀딩레이트 극단적 양수 ({rate_pct}%) — 롱 과열, 하락 반전 주의",
                data={"rate": rate, "rate_pct": rate_pct},
            )
        else:
            return EdgeSignal(
                signal_type="funding_extreme",
                direction="bullish",
                strength=strength,
                message=f"펀딩레이트 극단적 음수 ({rate_pct}%) — 숏 과열, 상승 반전 주의",
                data={"rate": rate, "rate_pct": rate_pct},
            )

    def detect_funding_trend(
        self, funding_rates: list[FundingRate]
    ) -> EdgeSignal | None:
        """Check if last 3 funding rates are trending in same direction."""
        if len(funding_rates) < 3:
            return None

        last3 = funding_rates[-3:]
        diffs = [last3[i + 1].rate - last3[i].rate for i in range(2)]

        if all(d > 0 for d in diffs):
            return EdgeSignal(
                signal_type="funding_trend",
                direction="bearish",
                strength="moderate",
                message="펀딩레이트 3회 연속 상승 — 롱 과열 가속",
                data={"rates": [fr.rate for fr in last3]},
            )
        elif all(d < 0 for d in diffs):
            return EdgeSignal(
                signal_type="funding_trend",
                direction="bullish",
                strength="moderate",
                message="펀딩레이트 3회 연속 하락 — 숏 과열 가속",
                data={"rates": [fr.rate for fr in last3]},
            )
        return None

    def detect_oi_anomaly(
        self,
        oi_data: list[OpenInterest],
        threshold_pct: float = 3.0,
    ) -> EdgeSignal | None:
        """Check if OI changed significantly between first and last data points."""
        if len(oi_data) < 2:
            return None

        old, new = oi_data[0], oi_data[-1]
        if old.value == 0:
            return None

        change_pct = ((new.value - old.value) / old.value) * 100

        if abs(change_pct) < threshold_pct:
            return None

        change_str = f"{change_pct:+.1f}"

        if change_pct < 0:
            return EdgeSignal(
                signal_type="oi_anomaly",
                direction="neutral",
                strength="strong" if abs(change_pct) >= 5.0 else "moderate",
                message=f"OI {change_str}% 급감 — 대량 청산 발생",
                data={"change_pct": round(change_pct, 2), "old": old.value, "new": new.value},
            )
        else:
            return EdgeSignal(
                signal_type="oi_anomaly",
                direction="neutral",
                strength="strong" if change_pct >= 5.0 else "moderate",
                message=f"OI {change_str}% 급증 — 신규 자금 유입",
                data={"change_pct": round(change_pct, 2), "old": old.value, "new": new.value},
            )

    def detect_all(
        self,
        funding_rates: list[FundingRate],
        oi_data: list[OpenInterest],
    ) -> list[EdgeSignal]:
        """Run all detections and return list of signals."""
        signals: list[EdgeSignal] = []

        for detector_fn, args in [
            (self.detect_funding_extreme, (funding_rates,)),
            (self.detect_funding_trend, (funding_rates,)),
            (self.detect_oi_anomaly, (oi_data,)),
        ]:
            result = detector_fn(*args)
            if result is not None:
                signals.append(result)

        return signals
