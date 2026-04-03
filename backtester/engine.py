"""
선물 백테스팅 엔진 (롱/숏, 레버리지, ATR 손절/익절)
=====================================================
- 롱(1)  진입: 가격 상승 시 수익
- 숏(-1) 진입: 가격 하락 시 수익
- 레버리지 반영한 증거금 계산
- ATR 기반 손절(SL) / 익절(TP) 자동 설정
- 청산가 도달 시 강제 손절 처리
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class Trade:
    direction:   int       # 1=롱, -1=숏
    entry_time:  pd.Timestamp
    entry_price: float
    quantity:    float     # 코인 수량 (마진 × 레버리지 / 진입가)
    sl_price:    float     # 손절가
    tp_price:    float     # 익절가
    liq_price:   float     # 청산가
    exit_time:   pd.Timestamp = None
    exit_price:  float = None
    exit_reason: str   = ""   # "TP" / "SL" / "LIQ" / "FORCE"
    pnl:         float = None
    pnl_pct:     float = None # 마진 대비 수익률 (레버리지 반영)


class BacktestEngine:
    """
    Args:
        initial_capital: 초기 자본 (USDT)
        leverage:        레버리지 배율
        risk_per_trade:  거래당 리스크 비율 (자본의 X%)
        sl_atr_mult:     손절 = ATR × 이 배수
        tp_atr_mult:     익절 = ATR × 이 배수
        fee_rate:        편도 수수료 (테이커 기준 0.05%)
        atr_period:      ATR 계산 기간
        maint_margin:    유지증거금율 (청산 기준, Binance BTC = 0.5%)
    """

    def __init__(
        self,
        initial_capital: float = 10_000.0,
        leverage:        int   = 10,
        risk_per_trade:  float = 0.01,      # 자본의 1%
        sl_atr_mult:     float = 1.5,
        tp_atr_mult:     float = 2.5,
        fee_rate:        float = 0.0005,    # 바이낸스 선물 테이커
        atr_period:      int   = 14,
        maint_margin:    float = 0.005,
    ):
        self.initial_capital = initial_capital
        self.leverage        = leverage
        self.risk_per_trade  = risk_per_trade
        self.sl_atr_mult     = sl_atr_mult
        self.tp_atr_mult     = tp_atr_mult
        self.fee_rate        = fee_rate
        self.atr_period      = atr_period
        self.maint_margin    = maint_margin

    def run(self, df: pd.DataFrame, signals: pd.Series) -> dict:
        atr = self._calc_atr(df)
        capital  = self.initial_capital
        position: Trade | None = None
        trades   = []
        equity   = []

        for i, (ts, row) in enumerate(df.iterrows()):
            price  = row["close"]
            high   = row["high"]
            low    = row["low"]
            signal = signals.iloc[i]
            curr_atr = atr.iloc[i]

            # ── 열린 포지션 청산 체크 ──
            if position is not None:
                exit_price, exit_reason = self._check_exit(position, high, low, price)
                if exit_price is not None:
                    capital = self._close_trade(position, ts, exit_price, exit_reason, capital)
                    trades.append(position)
                    position = None

            # ── 새 포지션 진입 ──
            if signal != 0 and position is None and not pd.isna(curr_atr):
                position = self._open_trade(ts, price, signal, curr_atr, capital)
                if position is not None:
                    fee = position.quantity * price * self.fee_rate
                    capital -= fee  # 진입 수수료

            # ── 에쿼티 계산 ──
            unrealized = 0.0
            if position is not None:
                unrealized = (price - position.entry_price) * position.direction * position.quantity
            equity.append({"time": ts, "equity": capital + unrealized})

        # ── 기간 종료 강제 청산 ──
        if position is not None:
            last_price = df["close"].iloc[-1]
            capital = self._close_trade(position, df.index[-1], last_price, "FORCE", capital)
            trades.append(position)

        equity_df = pd.DataFrame(equity).set_index("time")
        return {
            "trades":        trades,
            "equity_curve":  equity_df,
            "final_capital": capital,
        }

    # ──────────────────────────────────────────
    def _open_trade(self, ts, price, direction, atr, capital) -> Trade | None:
        sl_dist = atr * self.sl_atr_mult
        tp_dist = atr * self.tp_atr_mult

        if direction == 1:   # 롱
            sl_price  = price - sl_dist
            tp_price  = price + tp_dist
            liq_price = price * (1 - 1 / self.leverage + self.maint_margin)
            if sl_price <= liq_price:  # 손절 전에 청산되면 진입 안 함
                sl_price = liq_price * 1.005
        else:                # 숏
            sl_price  = price + sl_dist
            tp_price  = price - tp_dist
            liq_price = price * (1 + 1 / self.leverage - self.maint_margin)
            if sl_price >= liq_price:
                sl_price = liq_price * 0.995

        # 리스크 기반 포지션 크기 계산
        risk_amount = capital * self.risk_per_trade
        quantity    = (risk_amount / sl_dist) if sl_dist > 0 else 0
        margin      = (quantity * price) / self.leverage

        if margin > capital * 0.5 or quantity <= 0:  # 마진이 자본 50% 초과 방지
            return None

        return Trade(
            direction   = direction,
            entry_time  = ts,
            entry_price = price,
            quantity    = quantity,
            sl_price    = sl_price,
            tp_price    = tp_price,
            liq_price   = liq_price,
        )

    def _check_exit(self, pos: Trade, high: float, low: float, close: float):
        """봉 내에서 TP/SL/청산 도달 여부 체크. (exit_price, reason) 반환."""
        if pos.direction == 1:  # 롱
            if low <= pos.liq_price:
                return pos.liq_price, "LIQ"
            if low <= pos.sl_price:
                return pos.sl_price, "SL"
            if high >= pos.tp_price:
                return pos.tp_price, "TP"
        else:                   # 숏
            if high >= pos.liq_price:
                return pos.liq_price, "LIQ"
            if high >= pos.sl_price:
                return pos.sl_price, "SL"
            if low <= pos.tp_price:
                return pos.tp_price, "TP"
        return None, None

    def _close_trade(self, pos: Trade, ts, price: float, reason: str, capital: float) -> float:
        fee = pos.quantity * price * self.fee_rate
        raw_pnl = (price - pos.entry_price) * pos.direction * pos.quantity
        net_pnl = raw_pnl - fee

        pos.exit_time   = ts
        pos.exit_price  = price
        pos.exit_reason = reason
        pos.pnl         = net_pnl
        pos.pnl_pct     = net_pnl / self.initial_capital * 100

        return capital + net_pnl

    @staticmethod
    def _calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high  = df["high"]
        low   = df["low"]
        prev  = df["close"].shift(1)
        tr    = pd.concat([high - low,
                           (high - prev).abs(),
                           (low  - prev).abs()], axis=1).max(axis=1)
        return tr.ewm(com=period - 1, min_periods=period).mean()
