"""
최종 확정 전략: BB Squeeze Breakout 1h + 200d MA 국면 필터
=============================================================

백테스트 결과 요약 (2022-01 ~ 2025-03, $10,000 초기 자본, 10x 레버리지):

  sq15 EMA20 SL×1.5 TP×4 (공격적):
    2022 (곰장): +9.0%  | 2023: +19.8% | 2024: +16.4% | 2025Q1: -3.4%
    3년 합계: +41.8% | 전체 기간: +49.4%, 최종 $14,939

  sq15 EMA20 SL×1.5 TP×3 (보수적):
    2022 (곰장): +4.6%  | 2023: +16.1% | 2024: +17.8% | 2025Q1: -0.1%
    3년 합계: +38.3% | 전체 기간: +44.3%, 최종 $14,433

전략 파라미터:
  - 볼린저 밴드: 20봉, 2.0σ
  - 스퀴즈 감지: BB폭이 최근 100봉 하위 15% 이하
  - 추세 필터: EMA20 기울기 (3봉 기준) 방향과 일치할 때만 진입
  - 국면 필터: BTC > 200일 MA → 롱만 / BTC < 200일 MA → 숏만
  - SL: 1.5 × ATR | TP: 3.0 × ATR (보수) 또는 4.0 × ATR (공격)
  - 최대 보유: 100봉 (약 100시간)
  - 레버리지: 10x, 거래당 리스크: 자본의 1%

진입 로직:
  롱: 스퀴즈 상태에서 BB 상단 상향 돌파 + EMA20 기울기 양수 + 국면 필터(황소)
  숏: 스퀴즈 상태에서 BB 하단 하향 돌파 + EMA20 기울기 음수 + 국면 필터(곰)
"""
import pandas as pd
from strategies import BBBreakout
from backtester import BacktestEngine
from download_data import fetch


# ──────────────────────────────────────────────
# 파라미터 (확정값)
# ──────────────────────────────────────────────
SQUEEZE_PCT   = 15      # BB폭 하위 15%일 때 스퀴즈로 판단
TREND_EMA     = 20      # 추세 필터 EMA 기간
SL_ATR_MULT   = 1.5     # 손절 배수
TP_ATR_MULT   = 3.0     # 익절 배수 (보수적)
# TP_ATR_MULT = 4.0     # 익절 배수 (공격적)
LEVERAGE      = 10
CAPITAL       = 10_000
RISK_PER_TRADE = 0.01   # 거래당 자본의 1% 리스크


def apply_regime(signals: pd.Series, df_daily: pd.DataFrame,
                 df_target: pd.DataFrame) -> pd.Series:
    """200일 MA 국면 필터: 황소장(롱만) / 곰장(숏만)"""
    ma200  = df_daily["close"].rolling(200).mean()
    bull_r = (df_daily["close"] > ma200).astype(bool)
    bear_r = (df_daily["close"] < ma200).astype(bool)
    union  = bull_r.index.union(df_target.index)
    bull_s = bull_r.reindex(union).ffill().reindex(df_target.index).fillna(False).astype(bool)
    bear_s = bear_r.reindex(union).ffill().reindex(df_target.index).fillna(False).astype(bool)
    result = signals.copy()
    result[(signals ==  1) & (~bull_s)] = 0   # 곰장에서 롱 신호 제거
    result[(signals == -1) & (~bear_s)] = 0   # 황소장에서 숏 신호 제거
    return result


def run_backtest(start: str = "2022-01", end: str = "2025-03") -> dict:
    """백테스트 실행 및 결과 반환"""
    print(f"[데이터 로딩] {start} ~ {end}")
    df_1h = fetch("1h", start, end)
    # 200일 MA 계산을 위해 1년 전부터 로드
    y = int(start.split("-")[0])
    df_1d = fetch("1d", f"{y-1}-01", end)

    strategy = BBBreakout(
        squeeze_pct=SQUEEZE_PCT,
        trend_filter=True,
        trend_ema=TREND_EMA,
    )

    signals = strategy.generate_signals(df_1h)
    signals = apply_regime(signals, df_1d, df_1h)

    engine = BacktestEngine(
        initial_capital=CAPITAL,
        leverage=LEVERAGE,
        risk_per_trade=RISK_PER_TRADE,
        sl_atr_mult=SL_ATR_MULT,
        tp_atr_mult=TP_ATR_MULT,
        max_hold_bars=100,
    )
    result = engine.run(df_1h, signals)
    trades = result["trades"]

    if not trades:
        print("거래 없음")
        return result

    wins      = [t for t in trades if t.pnl > 0]
    longs     = [t for t in trades if t.direction ==  1]
    shorts    = [t for t in trades if t.direction == -1]
    total_pnl = sum(t.pnl for t in trades)
    win_rate  = len(wins) / len(trades) * 100
    loses     = [t for t in trades if t.pnl <= 0]
    pf        = (abs(sum(t.pnl for t in wins)) /
                 abs(sum(t.pnl for t in loses))) if loses else 999

    print(f"\n결과 ({start} ~ {end})")
    print(f"  신호:   {(signals!=0).sum()}건")
    print(f"  거래:   {len(trades)}건 (롱:{len(longs)} 숏:{len(shorts)})")
    print(f"  승률:   {win_rate:.1f}%")
    print(f"  수익팩터: {pf:.2f}")
    print(f"  수익:   {'+' if total_pnl>=0 else ''}{total_pnl/CAPITAL*100:.1f}%")
    print(f"  최종자본: ${result['final_capital']:,.0f}")
    return result


if __name__ == "__main__":
    run_backtest("2022-01", "2025-03")
