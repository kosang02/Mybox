"""성과 지표 계산."""

import numpy as np
import pandas as pd
from .engine import Trade


def calc_metrics(trades: list[Trade], equity_curve: pd.DataFrame,
                 initial_capital: float, strategy_name: str, interval: str) -> dict:
    if not trades:
        return _empty(strategy_name, interval)

    pnls      = [t.pnl for t in trades if t.pnl is not None]
    wins      = [p for p in pnls if p > 0]
    losses    = [p for p in pnls if p <= 0]
    equity    = equity_curve["equity"]
    final_cap = equity.iloc[-1]

    total_return = (final_cap / initial_capital - 1) * 100
    win_rate     = len(wins) / len(pnls) * 100 if pnls else 0
    avg_win      = np.mean(wins)   if wins   else 0
    avg_loss     = np.mean(losses) if losses else 0
    profit_factor = (abs(sum(wins)) / abs(sum(losses))
                     if losses and sum(losses) != 0 else float("inf"))

    # 최대 낙폭
    peak  = equity.cummax()
    mdd   = ((equity - peak) / peak * 100).min()

    # 샤프 비율 (일 수익률 기준 연환산)
    rets  = equity.pct_change().dropna()
    sharpe = (rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0

    # 청산 횟수
    liq_count = sum(1 for t in trades if t.exit_reason == "LIQ")

    # 평균 보유 시간 (봉 단위)
    hold_times = []
    for t in trades:
        if t.exit_time and t.entry_time:
            hold_times.append((t.exit_time - t.entry_time).total_seconds())
    avg_hold_sec = np.mean(hold_times) if hold_times else 0

    return {
        "strategy":       strategy_name,
        "interval":       interval,
        "total_return":   round(total_return, 2),
        "final_capital":  round(final_cap, 2),
        "mdd":            round(mdd, 2),
        "sharpe":         round(sharpe, 3),
        "win_rate":       round(win_rate, 1),
        "total_trades":   len(pnls),
        "winning":        len(wins),
        "losing":         len(losses),
        "liq_count":      liq_count,
        "avg_win":        round(avg_win, 2),
        "avg_loss":       round(avg_loss, 2),
        "profit_factor":  round(profit_factor, 3) if np.isfinite(profit_factor) else "∞",
        "avg_hold_sec":   round(avg_hold_sec),
    }


def _empty(strategy_name, interval):
    return {
        "strategy": strategy_name, "interval": interval,
        "total_return": 0, "final_capital": 0, "mdd": 0, "sharpe": 0,
        "win_rate": 0, "total_trades": 0, "winning": 0, "losing": 0,
        "liq_count": 0, "avg_win": 0, "avg_loss": 0,
        "profit_factor": 0, "avg_hold_sec": 0,
    }
