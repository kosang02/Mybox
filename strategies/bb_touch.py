"""
전략 B: 캔들 BB 터치 (밴드 반전)
====================================
캔들이 볼린저밴드 하단을 터치 → 롱 (반등 기대)
캔들이 볼린저밴드 상단을 터치 → 숏 (하락 기대)

confirm_candle=True: 터치 다음 봉이 밴드 안으로 복귀할 때 진입
                     (False면 터치하는 봉에서 즉시 진입)

직관: 밴드 경계는 통계적 극값. 터치 후 평균회귀(Mean Reversion) 발생 기대.
"""

import pandas as pd


class BBTouch:
    name = "BB 터치"

    def __init__(
        self,
        bb_period: int = 20,
        bb_std: float = 2.0,
        confirm_candle: bool = True,    # 다음 봉 확인 후 진입
        ema_filter: bool = True,        # EMA 방향 필터 (역추세 진입 차단)
        ema_period: int = 50,
    ):
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.confirm_candle = confirm_candle
        self.ema_filter = ema_filter
        self.ema_period = ema_period

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
        bb_std   = close.rolling(self.bb_period).std()
        bb_upper = bb_mid + self.bb_std * bb_std
        bb_lower = bb_mid - self.bb_std * bb_std

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

        # ── EMA 방향 필터 ──
        if self.ema_filter:
            ema = close.ewm(span=self.ema_period, adjust=False).mean()
            # 롱은 가격이 EMA 위에 있거나 EMA가 상승 중일 때만
            ema_up   = close >= ema
            ema_down = close <= ema
            long_signal  = long_signal  & ema_up
            short_signal = short_signal & ema_down

        signals = pd.Series(0, index=df.index)
        signals[long_signal]  =  1
        signals[short_signal] = -1

        # 같은 봉에서 롱/숏 동시 발생 시 무효화
        conflict = (long_signal & short_signal)
        signals[conflict] = 0

        return signals

    def __str__(self):
        confirm = "확인봉O" if self.confirm_candle else "즉시"
        ema_f   = f"EMA{self.ema_period}필터" if self.ema_filter else "필터없음"
        return f"{self.name} (BB:{self.bb_period}/{self.bb_std}, {confirm}, {ema_f})"
