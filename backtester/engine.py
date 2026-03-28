import pandas as pd
import numpy as np
from .strategy import Strategy


class Trade:
    """단일 거래 기록."""
    __slots__ = ["entry_time", "exit_time", "entry_price", "exit_price",
                 "quantity", "pnl", "pnl_pct"]

    def __init__(self, entry_time, entry_price, quantity):
        self.entry_time = entry_time
        self.entry_price = entry_price
        self.quantity = quantity
        self.exit_time = None
        self.exit_price = None
        self.pnl = None
        self.pnl_pct = None

    def close(self, exit_time, exit_price):
        self.exit_time = exit_time
        self.exit_price = exit_price
        self.pnl = (exit_price - self.entry_price) * self.quantity
        self.pnl_pct = (exit_price - self.entry_price) / self.entry_price * 100


class BacktestEngine:
    """
    시그널을 받아 포지션을 관리하고 거래를 시뮬레이션합니다.

    Args:
        initial_capital: 초기 자본 (USDT)
        fee_rate:        거래 수수료 (예: 0.001 = 0.1%)
        position_pct:    1회 거래 시 자본 사용 비율 (예: 1.0 = 전체)
    """

    def __init__(
        self,
        initial_capital: float = 10_000.0,
        fee_rate: float = 0.001,
        position_pct: float = 1.0,
    ):
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        self.position_pct = position_pct

    def run(self, df: pd.DataFrame, strategy: Strategy) -> dict:
        """백테스트를 실행하고 결과를 반환합니다."""
        signals = strategy.generate_signals(df)

        capital = self.initial_capital
        position: Trade | None = None
        trades: list[Trade] = []
        equity_curve: list[tuple] = []

        for ts, row in df.iterrows():
            price = row["close"]
            signal = signals[ts]

            # 매수 시그널 & 포지션 없음
            if signal == 1 and position is None:
                spend = capital * self.position_pct
                fee = spend * self.fee_rate
                spend_after_fee = spend - fee
                quantity = spend_after_fee / price
                capital -= spend
                position = Trade(ts, price, quantity)

            # 매도 시그널 & 포지션 있음
            elif signal == -1 and position is not None:
                proceeds = position.quantity * price
                fee = proceeds * self.fee_rate
                net_proceeds = proceeds - fee
                position.close(ts, price)
                position.pnl = net_proceeds - (position.entry_price * position.quantity)
                trades.append(position)
                capital += net_proceeds
                position = None

            # 현재 자산 가치 계산
            unrealized = (position.quantity * price) if position else 0
            equity_curve.append((ts, capital + unrealized))

        # 기간 종료 시 열린 포지션 강제 청산
        if position is not None:
            last_price = df["close"].iloc[-1]
            proceeds = position.quantity * last_price
            fee = proceeds * self.fee_rate
            net_proceeds = proceeds - fee
            position.close(df.index[-1], last_price)
            position.pnl = net_proceeds - (position.entry_price * position.quantity)
            trades.append(position)
            capital += net_proceeds

        equity_df = pd.DataFrame(equity_curve, columns=["time", "equity"]).set_index("time")

        return {
            "trades": trades,
            "equity_curve": equity_df,
            "final_capital": capital,
            "strategy": strategy,
            "price_data": df,
        }
