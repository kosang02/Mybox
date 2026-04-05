"""
전략 D: BB 스퀴즈 → 돌파 방향 진입
======================================
변동성이 압축된 상태(스퀴즈)에서 밴드를 돌파하는 방향으로 진입.
(현재 전략 A는 EMA 기울기 변환을 사용하지만, 이는 실제 가격 돌파를 사용)

진입 조건 (롱):
1. BB 폭이 최근 N봉 하위 X% = 스퀴즈 상태
2. 직전 봉의 close가 BB 상단 위로 돌파
3. 현재 봉도 BB 상단 위 (돌파 지속)
4. 거래량 급증 확인 (선택)

숏은 반대 (하단 하향 돌파).

직관: 낮은 변동성(에너지 압축) 후 방향성 있는 돌파 = 추세 시작
      'Squeeze Momentum' 개념의 순수 가격 돌파 버전
"""
import numpy as np
import pandas as pd


class BBBreakout:
    name = "BB 돌파"

    def __init__(
        self,
        bb_period:          int   = 20,
        bb_std:             float = 2.0,
        squeeze_pct:        float = 20.0,  # 하위 X%일 때 스퀴즈
        squeeze_lookback:   int   = 100,
        confirm_candles:    int   = 1,      # 돌파 후 몇 봉 지속 확인 (1=같은 봉)
        volume_mult:        float = 0.0,    # 0이면 거래량 필터 비활성
        trend_filter:       bool  = False,  # True이면 EMA 추세 방향과 일치할 때만
        trend_ema:          int   = 50,
    ):
        self.bb_period        = bb_period
        self.bb_std           = bb_std
        self.squeeze_pct      = squeeze_pct
        self.squeeze_lookback = squeeze_lookback
        self.confirm_candles  = confirm_candles
        self.volume_mult      = volume_mult
        self.trend_filter     = trend_filter
        self.trend_ema        = trend_ema

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close  = df["close"]
        high   = df["high"]
        low    = df["low"]

        # ── 볼린저 밴드 + 스퀴즈 ──
        bb_mid   = close.rolling(self.bb_period).mean()
        bb_std_s = close.rolling(self.bb_period).std()
        bb_upper = bb_mid + self.bb_std * bb_std_s
        bb_lower = bb_mid - self.bb_std * bb_std_s
        bb_width = (bb_upper - bb_lower) / bb_mid

        squeeze_threshold = bb_width.rolling(self.squeeze_lookback).quantile(
            self.squeeze_pct / 100
        )
        # 직전 봉이 스퀴즈 상태였어야 함 (압축 → 돌파)
        was_squeeze = bb_width.shift(1) <= squeeze_threshold.shift(1)

        # ── 돌파 조건 ──
        # 직전 봉이 BB 상단/하단 안에 있다가, 현재 봉이 외부로 돌파
        broke_upper = (close.shift(1) <= bb_upper.shift(1)) & (close > bb_upper)
        broke_lower = (close.shift(1) >= bb_lower.shift(1)) & (close < bb_lower)

        long_signal  = was_squeeze & broke_upper
        short_signal = was_squeeze & broke_lower

        # ── 거래량 필터 ──
        if self.volume_mult > 0 and "volume" in df.columns:
            vol_ma  = df["volume"].rolling(self.bb_period).mean()
            high_vol = df["volume"] > vol_ma * self.volume_mult
            long_signal  = long_signal  & high_vol
            short_signal = short_signal & high_vol

        # ── 추세 필터 ──
        if self.trend_filter:
            ema   = close.ewm(span=self.trend_ema, adjust=False).mean()
            slope = ema - ema.shift(3)
            in_uptrend   = slope > 0
            in_downtrend = slope < 0
            long_signal  = long_signal  & in_uptrend
            short_signal = short_signal & in_downtrend

        signals = pd.Series(0, index=df.index)
        signals[long_signal]  =  1
        signals[short_signal] = -1
        signals[long_signal & short_signal] = 0
        return signals

    def __str__(self):
        trend = f"+EMA{self.trend_ema}필터" if self.trend_filter else ""
        vol   = f"+Vol×{self.volume_mult}" if self.volume_mult > 0 else ""
        return (f"{self.name} "
                f"(BB{self.bb_period}/{self.bb_std}, "
                f"스퀴즈{self.squeeze_pct}%{trend}{vol})")
