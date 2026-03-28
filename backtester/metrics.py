import numpy as np
import pandas as pd
from .engine import Trade


class PerformanceMetrics:
    """백테스트 결과로부터 성과 지표를 계산합니다."""

    def __init__(self, result: dict):
        self.trades: list[Trade] = result["trades"]
        self.equity_curve: pd.DataFrame = result["equity_curve"]
        self.initial_capital: float = result["equity_curve"]["equity"].iloc[0] if not result["equity_curve"].empty else 0
        self.final_capital: float = result["final_capital"]
        self.strategy = result["strategy"]

    def summary(self) -> dict:
        trades = self.trades
        equity = self.equity_curve["equity"]

        if not trades:
            return {"error": "거래 없음 — 시그널이 발생하지 않았습니다."}

        pnl_list = [t.pnl for t in trades if t.pnl is not None]
        pnl_pct_list = [t.pnl_pct for t in trades if t.pnl_pct is not None]
        winning = [p for p in pnl_list if p > 0]
        losing = [p for p in pnl_list if p <= 0]

        total_return_pct = (self.final_capital / self.initial_capital - 1) * 100
        win_rate = len(winning) / len(trades) * 100 if trades else 0
        avg_win = np.mean(winning) if winning else 0
        avg_loss = np.mean(losing) if losing else 0
        profit_factor = abs(sum(winning) / sum(losing)) if losing else float("inf")
        max_dd = self._max_drawdown(equity)
        sharpe = self._sharpe_ratio(equity)

        return {
            "strategy": str(self.strategy),
            "total_trades": len(trades),
            "winning_trades": len(winning),
            "losing_trades": len(losing),
            "win_rate_pct": round(win_rate, 2),
            "total_return_pct": round(total_return_pct, 2),
            "profit_factor": round(profit_factor, 3),
            "avg_win_usdt": round(avg_win, 2),
            "avg_loss_usdt": round(avg_loss, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "sharpe_ratio": round(sharpe, 3),
            "initial_capital": round(self.initial_capital, 2),
            "final_capital": round(self.final_capital, 2),
        }

    @staticmethod
    def _max_drawdown(equity: pd.Series) -> float:
        peak = equity.cummax()
        drawdown = (equity - peak) / peak * 100
        return drawdown.min()

    @staticmethod
    def _sharpe_ratio(equity: pd.Series, risk_free: float = 0.0) -> float:
        returns = equity.pct_change().dropna()
        if returns.std() == 0:
            return 0.0
        # 일간 기준으로 연환산 (252거래일)
        return (returns.mean() - risk_free) / returns.std() * np.sqrt(252)
