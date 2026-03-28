from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


class Strategy(ABC):
    """전략 기본 클래스. 서브클래스에서 generate_signals()를 구현합니다."""

    name: str = "Strategy"

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        각 캔들에 대한 시그널을 반환합니다.
          1  = 매수 (long)
         -1  = 매도 (short / 청산)
          0  = 유지
        """

    def __str__(self):
        return self.name


# ──────────────────────────────────────────────
# 1. SMA 크로스오버
# ──────────────────────────────────────────────
class SMACrossover(Strategy):
    """단기 SMA가 장기 SMA를 상향 돌파 → 매수, 하향 돌파 → 매도."""

    def __init__(self, short: int = 20, long: int = 50):
        self.short = short
        self.long = long
        self.name = f"SMA Crossover ({short}/{long})"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        sma_short = df["close"].rolling(self.short).mean()
        sma_long = df["close"].rolling(self.long).mean()

        signals = pd.Series(0, index=df.index)
        # 골든 크로스 → 매수
        signals[(sma_short > sma_long) & (sma_short.shift(1) <= sma_long.shift(1))] = 1
        # 데드 크로스 → 매도
        signals[(sma_short < sma_long) & (sma_short.shift(1) >= sma_long.shift(1))] = -1
        return signals


# ──────────────────────────────────────────────
# 2. RSI 전략
# ──────────────────────────────────────────────
class RSIStrategy(Strategy):
    """RSI 과매도 구간에서 매수, 과매수 구간에서 매도."""

    def __init__(self, period: int = 14, oversold: float = 30, overbought: float = 70):
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        self.name = f"RSI ({period}, {oversold}/{overbought})"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        rsi = self._calc_rsi(df["close"], self.period)

        signals = pd.Series(0, index=df.index)
        # 과매도 → 반등 기대 매수
        signals[(rsi < self.oversold) & (rsi.shift(1) >= self.oversold)] = 1
        # 과매수 → 하락 기대 매도
        signals[(rsi > self.overbought) & (rsi.shift(1) <= self.overbought)] = -1
        return signals

    @staticmethod
    def _calc_rsi(series: pd.Series, period: int) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))


# ──────────────────────────────────────────────
# 3. 볼린저 밴드
# ──────────────────────────────────────────────
class BollingerBands(Strategy):
    """가격이 하단 밴드 아래로 떨어지면 매수, 상단 밴드 위로 오르면 매도."""

    def __init__(self, period: int = 20, std_dev: float = 2.0):
        self.period = period
        self.std_dev = std_dev
        self.name = f"Bollinger Bands ({period}, {std_dev}σ)"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        sma = df["close"].rolling(self.period).mean()
        std = df["close"].rolling(self.period).std()
        upper = sma + self.std_dev * std
        lower = sma - self.std_dev * std

        signals = pd.Series(0, index=df.index)
        # 하단 밴드 하향 돌파 후 복귀 → 매수
        signals[(df["close"] > lower) & (df["close"].shift(1) <= lower.shift(1))] = 1
        # 상단 밴드 상향 돌파 후 복귀 → 매도
        signals[(df["close"] < upper) & (df["close"].shift(1) >= upper.shift(1))] = -1
        return signals
