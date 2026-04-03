"""
벡터화 선물 백테스팅 엔진
===========================
핵심 아이디어:
- 모든 캔들을 순회하지 않고 시그널 발생 봉만 처리
- TP/SL/청산 도달 탐색은 numpy 벡터 연산으로 처리
- 10만봉 기준 기존 대비 ~100배 빠름
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class Trade:
    direction:   int
    entry_time:  object
    entry_price: float
    exit_time:   object
    exit_price:  float
    exit_reason: str       # TP / SL / LIQ / FORCE
    pnl:         float
    pnl_pct:     float     # 초기자본 대비
    quantity:    float


def _calc_atr(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    high  = df["high"].values
    low   = df["low"].values
    close = df["close"].values
    prev  = np.roll(close, 1); prev[0] = close[0]

    tr = np.maximum.reduce([
        high - low,
        np.abs(high - prev),
        np.abs(low  - prev),
    ])
    # Wilder smoothing (EWM with com=period-1)
    alpha = 1.0 / period
    atr   = np.empty_like(tr)
    atr[:period] = np.nan
    if period <= len(tr):
        atr[period - 1] = tr[:period].mean()
        for i in range(period, len(tr)):
            atr[i] = atr[i-1] * (1 - alpha) + tr[i] * alpha
    return atr


class BacktestEngine:
    def __init__(
        self,
        initial_capital: float = 10_000.0,
        leverage:        int   = 10,
        risk_per_trade:  float = 0.01,
        sl_atr_mult:     float = 1.5,
        tp_atr_mult:     float = 2.5,
        fee_rate:        float = 0.0005,
        atr_period:      int   = 14,
        maint_margin:    float = 0.005,
        max_hold_bars:   int   = 500,   # 최대 보유 봉 수 (스캘핑 상한)
    ):
        self.initial_capital = initial_capital
        self.leverage        = leverage
        self.risk_per_trade  = risk_per_trade
        self.sl_atr_mult     = sl_atr_mult
        self.tp_atr_mult     = tp_atr_mult
        self.fee_rate        = fee_rate
        self.atr_period      = atr_period
        self.maint_margin    = maint_margin
        self.max_hold_bars   = max_hold_bars

    def run(self, df: pd.DataFrame, signals: pd.Series) -> dict:
        closes   = df["close"].values
        highs    = df["high"].values
        lows     = df["low"].values
        times    = df.index
        atr      = _calc_atr(df, self.atr_period)
        sig_vals = signals.values
        n        = len(closes)

        capital   = self.initial_capital
        trades    = []
        equity    = np.full(n, self.initial_capital, dtype=float)

        # 시그널 발생 인덱스만 추출
        sig_indices = np.where(sig_vals != 0)[0]
        last_exit_idx = -1

        for sig_i in sig_indices:
            if sig_i <= last_exit_idx:
                continue   # 이전 포지션 아직 열려 있음

            direction = int(sig_vals[sig_i])
            price     = closes[sig_i]
            curr_atr  = atr[sig_i]

            if np.isnan(curr_atr) or curr_atr <= 0:
                continue

            # ── 손절/익절/청산가 계산 ──
            sl_dist = curr_atr * self.sl_atr_mult
            tp_dist = curr_atr * self.tp_atr_mult

            if direction == 1:   # 롱
                sl_price  = price - sl_dist
                tp_price  = price + tp_dist
                liq_price = price * (1 - 1/self.leverage + self.maint_margin)
                sl_price  = max(sl_price, liq_price * 1.001)
            else:                # 숏
                sl_price  = price + sl_dist
                tp_price  = price - tp_dist
                liq_price = price * (1 + 1/self.leverage - self.maint_margin)
                sl_price  = min(sl_price, liq_price * 0.999)

            # ── 포지션 크기 (리스크 기반) ──
            risk_amount = capital * self.risk_per_trade
            quantity    = risk_amount / sl_dist
            margin      = quantity * price / self.leverage

            if margin > capital * 0.9 or quantity <= 0 or capital <= 0:
                continue

            # ── 벡터 탐색: 최대 max_hold_bars 범위 내에서 TP/SL/LIQ 탐색 ──
            search_end = min(sig_i + 1 + self.max_hold_bars, n)
            future_h  = highs[sig_i + 1: search_end]
            future_l  = lows[sig_i  + 1: search_end]
            N_future   = len(future_h)

            if direction == 1:
                tp_arr  = np.where(future_h >= tp_price)[0]
                sl_arr  = np.where(future_l <= sl_price)[0]
                liq_arr = np.where(future_l <= liq_price)[0]
            else:
                tp_arr  = np.where(future_l <= tp_price)[0]
                sl_arr  = np.where(future_h >= sl_price)[0]
                liq_arr = np.where(future_h >= liq_price)[0]

            first_tp  = tp_arr[0]  if len(tp_arr)  else N_future
            first_sl  = sl_arr[0]  if len(sl_arr)  else N_future
            first_liq = liq_arr[0] if len(liq_arr) else N_future

            min_offset = min(first_tp, first_sl, first_liq)

            if min_offset >= N_future:
                exit_idx    = n - 1
                exit_price  = closes[-1]
                exit_reason = "FORCE"
            else:
                exit_idx = sig_i + 1 + min_offset
                # 청산이 SL보다 먼저 혹은 동시에 도달
                if first_liq <= min(first_tp, first_sl):
                    exit_price  = liq_price
                    exit_reason = "LIQ"
                elif first_sl <= first_tp:
                    exit_price  = sl_price
                    exit_reason = "SL"
                else:
                    exit_price  = tp_price
                    exit_reason = "TP"

            last_exit_idx = exit_idx

            # ── 손익 계산 ──
            entry_fee = quantity * price       * self.fee_rate
            exit_fee  = quantity * exit_price  * self.fee_rate
            raw_pnl   = (exit_price - price) * direction * quantity
            net_pnl   = raw_pnl - entry_fee - exit_fee
            capital  += net_pnl

            trades.append(Trade(
                direction   = direction,
                entry_time  = times[sig_i],
                entry_price = price,
                exit_time   = times[exit_idx],
                exit_price  = exit_price,
                exit_reason = exit_reason,
                pnl         = net_pnl,
                pnl_pct     = net_pnl / self.initial_capital * 100,
                quantity    = quantity,
            ))

            # ── 에쿼티 업데이트 ──
            equity[sig_i:exit_idx + 1] = capital - net_pnl + np.linspace(
                0, net_pnl, exit_idx - sig_i + 1
            )
            if exit_idx + 1 < n:
                equity[exit_idx + 1:] = capital

        equity_df = pd.DataFrame({"equity": equity}, index=df.index)
        return {
            "trades":        trades,
            "equity_curve":  equity_df,
            "final_capital": capital,
        }
