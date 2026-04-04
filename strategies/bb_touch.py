"""
전략 B: 캔들 BB 터치 (밴드 반전)
====================================
캔들이 볼린저밴드 하단을 터치 → 롱 (반등 기대)
캔들이 볼린저밴드 상단을 터치 → 숏 (하락 기대)

confirm_candle=True: 터치 다음 봉이 밴드 안으로 복귀할 때 진입
                     (False면 터치하는 봉에서 즉시 진입)

직관: 밴드 경계는 통계적 극값. 터치 후 평균회귀(Mean Reversion) 발생 기대.

[v2] EMA 필터 수정: slope 기반으로 변경 (강한 추세 역방향 진입 차단)
     RSI 필터 추가: 과매도/과매수 확인
"""

import pandas as pd
import numpy as np


def _calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss  = (-delta).clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


class BBTouch:
    name = "BB 터치"

    def __init__(
        self,
        bb_period: int = 20,
        bb_std: float = 2.0,
        confirm_candle: bool = True,    # 다음 봉 확인 후 진입
        ema_period: int = 50,
        ema_slope_period: int = 5,      # EMA slope 측정 기간
        rsi_period: int = 14,
        rsi_long_max: float = 50.0,     # 롱: RSI 이 값 이하일 때만 (과매도)
        rsi_short_min: float = 50.0,    # 숏: RSI 이 값 이상일 때만 (과매수)
        use_rsi: bool = True,
        use_ema_slope: bool = True,     # True: EMA 기울기 방향 체크
    ):
        self.bb_period        = bb_period
        self.bb_std           = bb_std
        self.confirm_candle   = confirm_candle
        self.ema_period       = ema_period
        self.ema_slope_period = ema_slope_period
        self.rsi_period       = rsi_period
        self.rsi_long_max     = rsi_long_max
        self.rsi_short_min    = rsi_short_min
        self.use_rsi          = use_rsi
        self.use_ema_slope    = use_ema_slope

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Returns:
             1 = 롱 진입
            -1 = 숏 진입
             0 = 유지
        """
        close = df["close"]
        high  = df["high"]
        low   = df["low"]

        # ── 볼린저 밴드 ──
        bb_mid   = close.rolling(self.bb_period).mean()
        bb_std_s = close.rolling(self.bb_period).std()
        bb_upper = bb_mid + self.bb_std * bb_std_s
        bb_lower = bb_mid - self.bb_std * bb_std_s

        # ── 터치 판정 ──
        touched_lower = low  <= bb_lower   # 하단 터치 (롱 후보)
        touched_upper = high >= bb_upper   # 상단 터치 (숏 후보)

        if self.confirm_candle:
            # 터치한 다음 봉이 밴드 안으로 돌아오는 것 확인
            long_signal  = touched_lower.shift(1) & (close > bb_lower)
            short_signal = touched_upper.shift(1) & (close < bb_upper)
        else:
            long_signal  = touched_lower
            short_signal = touched_upper

        # ── EMA slope 필터: 강한 역방향 추세 차단 ──
        # 평균회귀 전략이므로 강한 하락 추세에서 롱, 강한 상승 추세에서 숏 차단
        if self.use_ema_slope:
            ema   = close.ewm(span=self.ema_period, adjust=False).mean()
            slope = ema - ema.shift(self.ema_slope_period)
            # 롱: EMA가 너무 강하게 하락 중이면 차단 (강한 downtrend는 반등 실패 가능성 높음)
            not_strong_down = slope >= 0  # EMA가 하락 아닌 경우만 롱 허용
            # 숏: EMA가 너무 강하게 상승 중이면 차단
            not_strong_up   = slope <= 0  # EMA가 상승 아닌 경우만 숏 허용
            long_signal  = long_signal  & not_strong_down
            short_signal = short_signal & not_strong_up

        # ── RSI 필터 ──
        if self.use_rsi:
            rsi = _calc_rsi(close, self.rsi_period)
            long_signal  = long_signal  & (rsi <= self.rsi_long_max)
            short_signal = short_signal & (rsi >= self.rsi_short_min)

        signals = pd.Series(0, index=df.index)
        signals[long_signal]  =  1
        signals[short_signal] = -1

        # 같은 봉에서 롱/숏 동시 발생 시 무효화
        conflict = (long_signal & short_signal)
        signals[conflict] = 0

        return signals

    def __str__(self):
        confirm = "확인봉O" if self.confirm_candle else "즉시"
        rsi_str = f"RSI<{self.rsi_long_max:.0f}" if self.use_rsi else "RSI없음"
        slope_str = f"slope{self.ema_slope_period}" if self.use_ema_slope else "slope없음"
        return (f"{self.name} "
                f"(BB:{self.bb_period}/{self.bb_std}, {confirm}, "
                f"EMA{self.ema_period}/{slope_str}, {rsi_str})")
