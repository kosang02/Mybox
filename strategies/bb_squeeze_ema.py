"""
전략 A: BB 스퀴즈 + EMA 방향
==============================
볼린저밴드가 좁아진 상태(스퀴즈)에서
EMA 기울기가 위로 꺾이면 롱, 아래로 꺾이면 숏 진입.

직관: 변동성이 낮아져 에너지가 압축된 상태에서
      EMA 방향이 바뀌는 순간을 포착 → 방향성 있는 돌파 예상
"""

import numpy as np
import pandas as pd


class BBSqueezeEMA:
    name = "BB 스퀴즈+EMA"

    def __init__(
        self,
        bb_period: int = 20,
        bb_std: float = 2.0,
        ema_period: int = 20,
        squeeze_percentile: float = 20.0,   # BB 폭이 최근 N봉 중 하위 X%일 때 스퀴즈
        squeeze_lookback: int = 100,
        slope_period: int = 3,               # EMA 기울기 측정 기간 (봉 수)
    ):
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.ema_period = ema_period
        self.squeeze_percentile = squeeze_percentile
        self.squeeze_lookback = squeeze_lookback
        self.slope_period = slope_period

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Returns:
             1 = 롱 진입
            -1 = 숏 진입
             0 = 유지
        """
        close = df["close"]

        # ── 볼린저 밴드 ──
        bb_mid   = close.rolling(self.bb_period).mean()
        bb_std   = close.rolling(self.bb_period).std()
        bb_upper = bb_mid + self.bb_std * bb_std
        bb_lower = bb_mid - self.bb_std * bb_std
        bb_width = (bb_upper - bb_lower) / bb_mid  # 정규화된 밴드 폭

        # ── 스퀴즈 판정: 현재 폭이 최근 N봉 하위 X% 이하 ──
        squeeze_threshold = bb_width.rolling(self.squeeze_lookback).quantile(
            self.squeeze_percentile / 100
        )
        is_squeeze = bb_width <= squeeze_threshold

        # ── EMA + 기울기 ──
        ema = close.ewm(span=self.ema_period, adjust=False).mean()
        ema_slope = ema - ema.shift(self.slope_period)  # 양수=상승, 음수=하락

        # ── 기울기 전환 감지 ──
        slope_turned_up   = (ema_slope > 0) & (ema_slope.shift(1) <= 0)
        slope_turned_down = (ema_slope < 0) & (ema_slope.shift(1) >= 0)

        signals = pd.Series(0, index=df.index)
        signals[is_squeeze & slope_turned_up]   =  1   # 롱
        signals[is_squeeze & slope_turned_down] = -1   # 숏

        return signals

    def __str__(self):
        return (f"{self.name} "
                f"(BB:{self.bb_period}/{self.bb_std}, "
                f"EMA:{self.ema_period}, "
                f"스퀴즈:{self.squeeze_percentile}%)")
