"""
전략 C: EMA 스택 트렌드 풀백
==============================
추세 방향으로만 진입 (눌림목 매수 / 반등 매도)

진입 조건 (롱):
1. EMA21 > EMA50 (중기 상승 추세)
2. 가격이 EMA21 위에서 아래로 터치 후 반등 (눌림목)
3. RSI < 55 (과매수 아님, 적당히 눌렸을 때)
4. 이전 봉이 하락 → 현재 봉 상승 (반전 캔들)

숏은 반대 조건.

직관: 추세 중 일시적 되돌림을 포착
       EMA21 = "동적 지지선" 역할
"""
import numpy as np
import pandas as pd


def _calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss  = (-delta).clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


class EMAPullback:
    name = "EMA 풀백"

    def __init__(
        self,
        fast_ema:    int   = 9,
        mid_ema:     int   = 21,
        slow_ema:    int   = 50,
        rsi_period:  int   = 14,
        rsi_long_max:  float = 55.0,  # 롱: RSI 이 값 이하 (과매수 아닐 때)
        rsi_short_min: float = 45.0,  # 숏: RSI 이 값 이상
        require_bounce: bool = True,   # 캔들 반전 확인 (이전 봉 역방향 + 현재 봉 진입 방향)
    ):
        self.fast_ema      = fast_ema
        self.mid_ema       = mid_ema
        self.slow_ema      = slow_ema
        self.rsi_period    = rsi_period
        self.rsi_long_max  = rsi_long_max
        self.rsi_short_min = rsi_short_min
        self.require_bounce = require_bounce

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        high  = df["high"]
        low   = df["low"]

        ema_fast = close.ewm(span=self.fast_ema, adjust=False).mean()
        ema_mid  = close.ewm(span=self.mid_ema,  adjust=False).mean()
        ema_slow = close.ewm(span=self.slow_ema, adjust=False).mean()
        rsi      = _calc_rsi(close, self.rsi_period)

        # ── 추세 조건 ──
        uptrend   = ema_mid > ema_slow    # 중기 상승
        downtrend = ema_mid < ema_slow    # 중기 하락

        # ── EMA21 터치 판정 ──
        # 이전 봉: close가 EMA21 위에 있었음 (추세 진행 중)
        # 현재 봉: low가 EMA21 아래까지 눌렸다가 close가 다시 EMA21 위로 회복
        touched_mid_from_above = (
            (close.shift(1) > ema_mid.shift(1)) &  # 이전: EMA21 위
            (low <= ema_mid) &                       # 현재: EMA21 아래까지 눌림
            (close >= ema_mid)                       # 현재 마감: EMA21 위에서 회복
        )
        touched_mid_from_below = (
            (close.shift(1) < ema_mid.shift(1)) &  # 이전: EMA21 아래
            (high >= ema_mid) &                      # 현재: EMA21 위까지 반등
            (close <= ema_mid)                       # 현재 마감: EMA21 아래에서 마감
        )

        # ── 기본 시그널 ──
        long_signal  = uptrend   & touched_mid_from_above & (rsi <= self.rsi_long_max)
        short_signal = downtrend & touched_mid_from_below & (rsi >= self.rsi_short_min)

        # ── 반전 캔들 확인 ──
        if self.require_bounce:
            # 롱: 이전 봉이 하락봉, 현재 봉이 상승봉
            prev_down = close.shift(1) < close.shift(2)
            curr_up   = close > close.shift(1)
            # 숏: 이전 봉이 상승봉, 현재 봉이 하락봉
            prev_up   = close.shift(1) > close.shift(2)
            curr_down = close < close.shift(1)
            long_signal  = long_signal  & prev_down & curr_up
            short_signal = short_signal & prev_up   & curr_down

        signals = pd.Series(0, index=df.index)
        signals[long_signal]  =  1
        signals[short_signal] = -1
        signals[long_signal & short_signal] = 0
        return signals

    def __str__(self):
        return (f"{self.name} "
                f"(EMA{self.fast_ema}/{self.mid_ema}/{self.slow_ema}, "
                f"RSI<{self.rsi_long_max}/>>{self.rsi_short_min})")
