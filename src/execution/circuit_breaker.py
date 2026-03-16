"""
시그널 쿨다운 관리.

기존 circuit breaker에서 시그널 발송 쿨다운과 API 에러 추적만 남김.
"""

import logging
from datetime import datetime, timezone, timedelta

from core.safety import MAX_API_ERRORS_PER_HOUR, MAX_DATA_STALENESS_SECONDS

logger = logging.getLogger(__name__)


class SignalCooldown:
    """시그널 쿨다운 및 API 에러 추적."""

    def __init__(self, cooldown_minutes: int = 30):
        self._cooldown = timedelta(minutes=cooldown_minutes)
        self._api_errors: list[datetime] = []
        self._halted = False

    @property
    def is_halted(self) -> bool:
        return self._halted

    def can_send_signal(self, last_signal_time: datetime | None) -> bool:
        """쿨다운 이후 시그널 발송 가능 여부."""
        if self._halted:
            return False
        if last_signal_time is None:
            return True
        return datetime.now(timezone.utc) - last_signal_time >= self._cooldown

    def record_api_error(self) -> bool:
        """API 에러 기록. 임계값 초과 시 True 반환."""
        now = datetime.now(timezone.utc)
        self._api_errors.append(now)
        cutoff = now - timedelta(hours=1)
        self._api_errors = [e for e in self._api_errors if e > cutoff]

        if len(self._api_errors) >= MAX_API_ERRORS_PER_HOUR:
            self._halted = True
            logger.error(f"API error threshold breached: {len(self._api_errors)} errors/hour")
            return True
        return False

    def check_data_staleness(self, last_data_time: datetime) -> bool:
        """데이터 신선도 체크. 문제 시 True."""
        now = datetime.now(timezone.utc)
        if last_data_time.tzinfo is None:
            last_data_time = last_data_time.replace(tzinfo=timezone.utc)
        staleness = (now - last_data_time).total_seconds()
        if staleness > MAX_DATA_STALENESS_SECONDS:
            logger.warning(f"Data stale: {staleness:.0f}s ago")
            return True
        return False

    def resume(self) -> None:
        self._halted = False
        self._api_errors = []
