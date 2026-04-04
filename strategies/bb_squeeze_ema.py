"""
전략 A: BB 스퀴즈 + EMA 방향
==============================
볼린저밴드가 좁아진 상태(스퀴즈)에서
EMA 기울기가 위로 꺾이면 롱, 아래로 꺾이면 숏 진입.

직관: 변동성이 낮아져 에너지가 압축된 상태에서
      EMA 방향이 바뀌는 순간을 포착 → 방향성 있는 돌파 예상

[v2] RSI 모멘텀 확인 추가: EMA 방향과 RSI가 일치할 때만 진입
"""

import numpy as np
import pandas as pd


def _calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss  = (-delta).clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


class BBSqueezeEMA:
    name = "BB 스퀴즈+EMA"

    def __init__(
        self,
        bb_period: int = 20,
        bb_std: float = 2.0,            # 수학적으로 결과에 영향 없음 (폭 비율로 정규화)
        ema_period: int = 20,
        squeeze_percentile: float = 20.0,   # BB 폭이 최근 N봉 중 하위 X%일 때 스퀴즈
        squeeze_lookback: int = 100,
        slope_period: int = 3,               # EMA 기울기 측정 기간 (봉 수)
        rsi_period: int = 14,
        rsi_long_min: float = 50.0,          # 롱: RSI 이 값 이상일 때만 (상승 모멘텀)
        rsi_short_max: float = 50.0,         # 숏: RSI 이 값 이하일 때만 (하락 모멘텀)
        use_rsi: bool = False,               # 기본 off (비교용)
    ):
        self.bb_period           = bb_period
        self.bb_std              = bb_std
        self.ema_period          = ema_period
        self.squeeze_percentile  = squeeze_percentile
        self.squeeze_lookback    = squeeze_lookback
        self.slope_period        = slope_period
        self.rsi_period          = rsi_period
        self.rsi_long_min        = rsi_long_min
        self.rsi_short_max       = rsi_short_max
        self.use_rsi             = use_rsi

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
        bb_std_s = close.rolling(self.bb_period).std()
        bb_upper = bb_mid + self.bb_std * bb_std_s
        bb_lower = bb_mid - self.bb_std * bb_std_s
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

        long_signal  = is_squeeze & slope_turned_up
        short_signal = is_squeeze & slope_turned_down

        # ── RSI 모멘텀 확인 ──
        if self.use_rsi:
            rsi = _calc_rsi(close, self.rsi_period)
            long_signal  = long_signal  & (rsi >= self.rsi_long_min)
            short_signal = short_signal & (rsi <= self.rsi_short_max)

        signals = pd.Series(0, index=df.index)
        signals[long_signal]  =  1   # 롱
        signals[short_signal] = -1   # 숏

        return signals

    def __str__(self):
        rsi_str = f"+RSI{self.rsi_period}" if self.use_rsi else ""
        return (f"{self.name} "
                f"(BB:{self.bb_period}, "
                f"EMA:{self.ema_period}, "
                f"스퀴즈:{self.squeeze_percentile}%"
                f"{rsi_str})")
