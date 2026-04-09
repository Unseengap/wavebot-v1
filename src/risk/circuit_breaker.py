"""Circuit breaker: non-negotiable drawdown and trade count limits."""


class CircuitBreaker:
    def __init__(self, max_daily_drawdown=0.02, max_total_drawdown=0.08,
                 max_open_trades=3):
        self.max_daily_dd = max_daily_drawdown
        self.max_total_dd = max_total_drawdown
        self.max_open = max_open_trades
        self.daily_start_balance = None
        self.peak_balance = None
        self.halted_today = False

    def check(self, balance: float, open_trade_count: int) -> tuple:
        """Returns (allowed: bool, reason: str)."""
        if self.daily_start_balance is None:
            self.daily_start_balance = balance
        if self.peak_balance is None:
            self.peak_balance = balance
        self.peak_balance = max(self.peak_balance, balance)

        # Daily drawdown
        if self.daily_start_balance > 0:
            daily_dd = (self.daily_start_balance - balance) / self.daily_start_balance
            if daily_dd >= self.max_daily_dd:
                self.halted_today = True
                return False, f"DAILY DD {daily_dd*100:.1f}% >= {self.max_daily_dd*100:.0f}%"

        # Total drawdown from peak
        if self.peak_balance > 0:
            total_dd = (self.peak_balance - balance) / self.peak_balance
            if total_dd >= self.max_total_dd:
                return False, f"TOTAL DD {total_dd*100:.1f}% >= {self.max_total_dd*100:.0f}%"

        if self.halted_today:
            return False, "HALTED for today"

        if open_trade_count >= self.max_open:
            return False, f"MAX TRADES ({self.max_open})"

        return True, "OK"

    def reset_daily(self, balance: float):
        self.daily_start_balance = balance
        self.halted_today = False

    def get_daily_drawdown(self, balance: float) -> float:
        if self.daily_start_balance is None or self.daily_start_balance <= 0:
            return 0.0
        return max(0, (self.daily_start_balance - balance) / self.daily_start_balance)
