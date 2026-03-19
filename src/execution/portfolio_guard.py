"""Portfolio-level risk controls for 3-7x leverage trading."""

from dataclasses import dataclass

from core.types import ManualPosition, Side


@dataclass(frozen=True)
class GuardResult:
    allowed: bool
    reason: str  # Korean message


class PortfolioGuard:
    def __init__(
        self,
        max_positions: int = 3,
        daily_loss_limit_pct: float = 5.0,
        monthly_drawdown_limit_pct: float = 20.0,
        btc_correlation_threshold: float = 0.8,
        account_balance_usdt: float = 0.0,
    ):
        self.max_positions = max_positions
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.monthly_drawdown_limit_pct = monthly_drawdown_limit_pct
        self.btc_correlation_threshold = btc_correlation_threshold
        self.account_balance_usdt = account_balance_usdt
        self._daily_realized_loss: float = 0.0
        self._monthly_realized_loss: float = 0.0

    def set_account_balance(self, balance: float) -> None:
        """Update the account balance used as denominator for loss % calculation."""
        self.account_balance_usdt = balance

    def check_can_open(
        self, active_positions: list[ManualPosition], new_symbol: str
    ) -> GuardResult:
        """Check if a new position can be opened."""
        # 1. Max positions check
        if len(active_positions) >= self.max_positions:
            return GuardResult(
                False,
                f"최대 동시 포지션 {self.max_positions}개 초과.",
            )

        # 2. Daily loss limit check
        if self.account_balance_usdt > 0:
            daily_pct = (self._daily_realized_loss / self.account_balance_usdt) * 100
            if daily_pct >= self.daily_loss_limit_pct:
                return GuardResult(
                    False,
                    f"일일 손실 한도 {self.daily_loss_limit_pct}% 도달.",
                )

            # 3. Monthly drawdown check
            monthly_pct = (self._monthly_realized_loss / self.account_balance_usdt) * 100
            if monthly_pct >= self.monthly_drawdown_limit_pct:
                return GuardResult(
                    False,
                    f"월간 드로다운 {self.monthly_drawdown_limit_pct}% 도달.",
                )

        return GuardResult(True, "진입 가능")

    def record_realized_loss(self, loss_usdt: float) -> None:
        """Record loss (positive = loss, negative = profit). Net basis."""
        self._daily_realized_loss += loss_usdt
        self._monthly_realized_loss += loss_usdt

    def reset_daily(self) -> None:
        """Reset daily counters (call at UTC midnight)."""
        self._daily_realized_loss = 0.0

    def reset_monthly(self) -> None:
        """Reset monthly counters (call at month start)."""
        self._monthly_realized_loss = 0.0

    def get_daily_status(self) -> dict:
        """Return current daily loss status."""
        pct = (
            (self._daily_realized_loss / self.account_balance_usdt * 100)
            if self.account_balance_usdt > 0
            else 0
        )
        return {
            "realized_loss_usdt": self._daily_realized_loss,
            "account_balance_usdt": self.account_balance_usdt,
            "loss_pct": round(pct, 1),
            "limit_pct": self.daily_loss_limit_pct,
            "remaining_pct": round(max(0, self.daily_loss_limit_pct - pct), 1),
        }
