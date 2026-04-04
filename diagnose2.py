"""
진단 2: TP비율 극대화 + 새 전략 아이디어 테스트
- 33% 승률에서 TP비율만 올리면 수익 가능한지 확인
- 추세 추종 전략 (Trend Pullback) 테스트
"""
import numpy as np
import pandas as pd
from download_data import fetch
from strategies import BBSqueezeEMA, BBTouch
from backtester import BacktestEngine, calc_metrics


def run_test(label, strategy, df, sl_mult, tp_mult, max_hold, leverage=10, capital=10000):
    sigs = strategy.generate_signals(df)
    n_sig = (sigs != 0).sum()

    engine = BacktestEngine(
        initial_capital=capital, leverage=leverage, risk_per_trade=0.01,
        sl_atr_mult=sl_mult, tp_atr_mult=tp_mult, max_hold_bars=max_hold,
    )
    result = engine.run(df, sigs)
    trades = result["trades"]

    if not trades:
        print(f"  {label}: 거래 없음 (시그널 {n_sig}개)")
        return

    wins  = [t for t in trades if t.pnl > 0]
    loses = [t for t in trades if t.pnl <= 0]
    win_rate = len(wins) / len(trades) * 100
    total_pnl = sum(t.pnl for t in trades)
    pf = abs(sum(t.pnl for t in wins)) / abs(sum(t.pnl for t in loses)) if loses else 999
    tps = sum(1 for t in trades if t.exit_reason == "TP")
    sls = sum(1 for t in trades if t.exit_reason == "SL")

    sign = "+" if total_pnl >= 0 else ""
    flag = "★" if total_pnl > 0 else "  "
    print(f"  {flag} {label:<55} | "
          f"시그:{n_sig:5} 거래:{len(trades):4} | "
          f"승:{win_rate:4.1f}% TP:{tps} SL:{sls} | "
          f"PF:{pf:.2f} | {sign}{total_pnl/capital*100:.1f}%")


def main():
    print("[데이터 로딩]...")
    df5  = fetch("5m",  "2023-01", "2024-12")
    df15 = fetch("15m", "2023-01", "2024-12")
    df1h = fetch("1h",  "2022-01", "2024-12")

    print(f"\n{'='*120}")
    print("실험 1: TP 비율 극대화 (전략A, 승률~33%에서 수익 조건 탐색)")
    print(f"{'='*120}")
    strat_a = BBSqueezeEMA(ema_period=50, squeeze_percentile=20, slope_period=3)
    for tp in [2.0, 3.0, 4.0, 5.0, 6.0, 8.0]:
        for sl in [1.0, 1.5]:
            for hold in [100, 200, 500]:
                run_test(
                    f"5m A SL×{sl} TP×{tp:.0f} hold:{hold}",
                    strat_a, df5, sl, tp, hold
                )

    print(f"\n{'='*120}")
    print("실험 2: BB 터치 + RSI 필터 변형 (TP 비율 변화)")
    print(f"{'='*120}")
    for rsi_th in [35, 40, 45, 50]:
        for tp in [3.0, 4.0, 5.0, 6.0]:
            for hold in [100, 200]:
                run_test(
                    f"5m B RSI<{rsi_th} SL×1.5 TP×{tp:.0f} hold:{hold}",
                    BBTouch(confirm_candle=True, ema_period=50, ema_slope_period=5,
                            use_ema_slope=True, use_rsi=True,
                            rsi_long_max=float(rsi_th), rsi_short_min=float(100-rsi_th)),
                    df5, 1.5, tp, hold
                )

    print(f"\n{'='*120}")
    print("실험 3: 1시간봉 (노이즈 제거, 더 의미있는 시그널)")
    print(f"{'='*120}")
    for strat, label in [
        (BBSqueezeEMA(ema_period=50, squeeze_percentile=20, slope_period=3), "A EMA50 sq20"),
        (BBSqueezeEMA(ema_period=20, squeeze_percentile=20, slope_period=3), "A EMA20 sq20"),
        (BBTouch(confirm_candle=True, ema_period=50, ema_slope_period=5,
                 use_ema_slope=True, use_rsi=True, rsi_long_max=40.0, rsi_short_min=60.0), "B RSI40 slope"),
        (BBTouch(confirm_candle=True, ema_period=50, ema_slope_period=5,
                 use_ema_slope=False, use_rsi=True, rsi_long_max=50.0, rsi_short_min=50.0), "B RSI50 no-slope"),
    ]:
        for tp in [3.0, 4.0, 5.0]:
            for sl in [1.0, 1.5]:
                run_test(f"1h {label} SL×{sl} TP×{tp:.0f}",
                         strat, df1h, sl, tp, 100)

    print(f"\n{'='*120}")
    print("실험 4: 15분봉 (5m 보다 덜 노이즈)")
    print(f"{'='*120}")
    for strat, label in [
        (BBSqueezeEMA(ema_period=50, squeeze_percentile=20, slope_period=3), "A EMA50 sq20"),
        (BBSqueezeEMA(ema_period=20, squeeze_percentile=20, slope_period=3), "A EMA20 sq20"),
        (BBTouch(confirm_candle=True, ema_period=50, ema_slope_period=5,
                 use_ema_slope=True, use_rsi=True, rsi_long_max=40.0, rsi_short_min=60.0), "B RSI40 slope"),
    ]:
        for tp in [3.0, 4.0, 5.0]:
            for sl in [1.0, 1.5]:
                run_test(f"15m {label} SL×{sl} TP×{tp:.0f}",
                         strat, df15, sl, tp, 100)

    print(f"\n{'='*120}")
    print("실험 5: 추세추종 필터 (1h 추세 방향만 거래)")
    print(f"{'='*120}")
    from optimize import build_trend_filter, apply_trend_filter
    trend_5m  = build_trend_filter(df5,  "1h",  "2023-01", "2024-12")
    trend_15m = build_trend_filter(df15, "1h",  "2023-01", "2024-12")

    for strat, df, tf, label in [
        (BBSqueezeEMA(ema_period=50, squeeze_percentile=20, slope_period=3), df5,  trend_5m,  "5m A EMA50 sq20"),
        (BBSqueezeEMA(ema_period=20, squeeze_percentile=20, slope_period=3), df5,  trend_5m,  "5m A EMA20 sq20"),
        (BBTouch(confirm_candle=True, ema_period=50, ema_slope_period=5,
                 use_ema_slope=True, use_rsi=True, rsi_long_max=40.0, rsi_short_min=60.0),
         df5, trend_5m, "5m B RSI40"),
        (BBSqueezeEMA(ema_period=50, squeeze_percentile=20, slope_period=3), df15, trend_15m, "15m A EMA50 sq20"),
        (BBTouch(confirm_candle=True, ema_period=50, ema_slope_period=5,
                 use_ema_slope=True, use_rsi=True, rsi_long_max=40.0, rsi_short_min=60.0),
         df15, trend_15m, "15m B RSI40"),
    ]:
        sigs = strat.generate_signals(df)
        sigs = apply_trend_filter(sigs, tf)
        for tp in [3.0, 4.0, 5.0]:
            for sl in [1.0, 1.5]:
                engine = BacktestEngine(
                    initial_capital=10000, leverage=10, risk_per_trade=0.01,
                    sl_atr_mult=sl, tp_atr_mult=tp, max_hold_bars=200,
                )
                result = engine.run(df, sigs)
                trades = result["trades"]
                if not trades:
                    continue
                wins  = [t for t in trades if t.pnl > 0]
                loses = [t for t in trades if t.pnl <= 0]
                win_rate = len(wins) / len(trades) * 100
                total_pnl = sum(t.pnl for t in trades)
                pf = abs(sum(t.pnl for t in wins)) / abs(sum(t.pnl for t in loses)) if loses else 999
                flag = "★" if total_pnl > 0 else "  "
                sign = "+" if total_pnl >= 0 else ""
                print(f"  {flag} {label:<28} SL×{sl} TP×{tp:.0f} | "
                      f"거래:{len(trades):4} 승:{win_rate:4.1f}% PF:{pf:.2f} | {sign}{total_pnl/10000*100:.1f}%")


if __name__ == "__main__":
    main()
