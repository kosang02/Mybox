"""
진단 3: BB Touch RSI 전략 심화 검증
- RSI<40 + TP×6 조합이 유일하게 수익 → 더 파고들기
- 2022 약세장 vs 2023-2024 강세장 분리 테스트
- RSI threshold, TP 비율, 추세필터 조합 세밀 탐색
"""
import numpy as np
import pandas as pd
from download_data import fetch
from strategies import BBTouch
from backtester import BacktestEngine


def run(label, sigs, df, sl, tp, hold, capital=10000, leverage=10):
    engine = BacktestEngine(
        initial_capital=capital, leverage=leverage, risk_per_trade=0.01,
        sl_atr_mult=sl, tp_atr_mult=tp, max_hold_bars=hold,
    )
    result = engine.run(df, sigs)
    trades = result["trades"]
    if not trades:
        return None

    wins  = [t for t in trades if t.pnl > 0]
    loses = [t for t in trades if t.pnl <= 0]
    win_rate = len(wins) / len(trades) * 100
    total_pnl = sum(t.pnl for t in trades)
    pf = abs(sum(t.pnl for t in wins)) / abs(sum(t.pnl for t in loses)) if loses else 999
    tps = sum(1 for t in trades if t.exit_reason == "TP")
    sls = sum(1 for t in trades if t.exit_reason == "SL")

    flag = "★" if total_pnl > 0 else "  "
    sign = "+" if total_pnl >= 0 else ""
    print(f"  {flag} {label:<65} 거래:{len(trades):3} 승:{win_rate:4.1f}% TP:{tps} SL:{sls} PF:{pf:.2f} {sign}{total_pnl/capital*100:.1f}%")
    return total_pnl


def main():
    print("[데이터 로딩]...")
    df_2022     = fetch("5m", "2022-01", "2022-12")
    df_2023     = fetch("5m", "2023-01", "2023-12")
    df_2024     = fetch("5m", "2024-01", "2024-12")
    df_all      = fetch("5m", "2022-01", "2024-12")
    df_15m_all  = fetch("15m", "2022-01", "2024-12")
    df_1h_all   = fetch("1h", "2022-01", "2024-12")
    print(f"  5m: {len(df_all):,}개 | 15m: {len(df_15m_all):,}개 | 1h: {len(df_1h_all):,}개")

    # ─────────────────────────────────
    # 실험 A: 연도별 분리 검증
    # ─────────────────────────────────
    print(f"\n{'='*100}")
    print("실험 A: 연도별 분리 (2022=약세장, 2023=회복, 2024=강세장)")
    print("기준 전략: 5m BB터치 RSI<40 EMA slope SL×1.5 TP×6 hold:200")
    print(f"{'='*100}")

    best_strat = BBTouch(
        confirm_candle=True, ema_period=50, ema_slope_period=5,
        use_ema_slope=True, use_rsi=True,
        rsi_long_max=40.0, rsi_short_min=60.0
    )
    for year, df in [("2022", df_2022), ("2023", df_2023), ("2024", df_2024), ("2022-2024", df_all)]:
        sigs = best_strat.generate_signals(df)
        run(f"{year}", sigs, df, sl=1.5, tp=6.0, hold=200)

    # ─────────────────────────────────
    # 실험 B: EMA slope 필터 제거
    # ─────────────────────────────────
    print(f"\n{'='*100}")
    print("실험 B: EMA slope 필터 ON/OFF 비교 (RSI<40, TP×6)")
    print(f"{'='*100}")
    for use_slope in [True, False]:
        for confirm in [True, False]:
            strat = BBTouch(
                confirm_candle=confirm, ema_period=50, ema_slope_period=5,
                use_ema_slope=use_slope, use_rsi=True,
                rsi_long_max=40.0, rsi_short_min=60.0
            )
            sigs = strat.generate_signals(df_all)
            label = f"slope:{'O' if use_slope else 'X'} 확인봉:{'O' if confirm else 'X'} [2022-24]"
            run(label, sigs, df_all, sl=1.5, tp=6.0, hold=200)

    # ─────────────────────────────────
    # 실험 C: RSI threshold 세밀 탐색
    # ─────────────────────────────────
    print(f"\n{'='*100}")
    print("실험 C: RSI threshold 세밀 탐색 (2022-2024)")
    print(f"{'='*100}")
    for rsi_long in [30, 35, 38, 40, 42, 45, 48, 50]:
        for use_slope in [True, False]:
            strat = BBTouch(
                confirm_candle=True, ema_period=50, ema_slope_period=5,
                use_ema_slope=use_slope, use_rsi=True,
                rsi_long_max=float(rsi_long), rsi_short_min=float(100-rsi_long)
            )
            sigs = strat.generate_signals(df_all)
            n_sig = (sigs != 0).sum()
            label = f"RSI<{rsi_long} slope:{'O' if use_slope else 'X'} sig:{n_sig:5}"
            run(label, sigs, df_all, sl=1.5, tp=6.0, hold=200)

    # ─────────────────────────────────
    # 실험 D: TP 비율 세밀 탐색 (최적 RSI)
    # ─────────────────────────────────
    print(f"\n{'='*100}")
    print("실험 D: TP 비율 세밀 탐색 (RSI<40 slope=False, 2022-2024)")
    print(f"{'='*100}")
    strat_d = BBTouch(
        confirm_candle=True, ema_period=50, ema_slope_period=5,
        use_ema_slope=False, use_rsi=True,
        rsi_long_max=40.0, rsi_short_min=60.0
    )
    sigs_d = strat_d.generate_signals(df_all)
    for sl in [1.0, 1.5, 2.0]:
        for tp in [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]:
            for hold in [100, 200, 500]:
                label = f"SL×{sl} TP×{tp:.0f} hold:{hold}"
                run(label, sigs_d, df_all, sl=sl, tp=tp, hold=hold)

    # ─────────────────────────────────
    # 실험 E: 추세 필터 적용 (1h EMA50)
    # ─────────────────────────────────
    print(f"\n{'='*100}")
    print("실험 E: 1h 추세 필터 적용 (RSI<40 최적 파라미터)")
    print(f"{'='*100}")
    from optimize import build_trend_filter, apply_trend_filter
    trend_5m = build_trend_filter(df_all, "1h", "2022-01", "2024-12")

    for use_slope in [True, False]:
        for rsi in [40, 45]:
            strat = BBTouch(
                confirm_candle=True, ema_period=50, ema_slope_period=5,
                use_ema_slope=use_slope, use_rsi=True,
                rsi_long_max=float(rsi), rsi_short_min=float(100-rsi)
            )
            sigs_base = strat.generate_signals(df_all)
            sigs_trend = apply_trend_filter(sigs_base, trend_5m)
            n_base  = (sigs_base != 0).sum()
            n_trend = (sigs_trend != 0).sum()

            for tp in [5.0, 6.0, 7.0]:
                label = f"RSI<{rsi} slope:{'O' if use_slope else 'X'} 추세O sig:{n_trend} TP×{tp:.0f}"
                run(label, sigs_trend, df_all, sl=1.5, tp=tp, hold=200)

            label = f"RSI<{rsi} slope:{'O' if use_slope else 'X'} 추세X sig:{n_base} TP×6"
            run(label, sigs_base, df_all, sl=1.5, tp=6.0, hold=200)

    # ─────────────────────────────────
    # 실험 F: 15m / 1h 타임프레임
    # ─────────────────────────────────
    print(f"\n{'='*100}")
    print("실험 F: 15m / 1h 타임프레임 (RSI<40 slope=False)")
    print(f"{'='*100}")
    for tf_label, df_tf in [("15m", df_15m_all), ("1h", df_1h_all)]:
        for rsi in [40, 45, 50]:
            strat = BBTouch(
                confirm_candle=True, ema_period=50, ema_slope_period=5,
                use_ema_slope=False, use_rsi=True,
                rsi_long_max=float(rsi), rsi_short_min=float(100-rsi)
            )
            sigs = strat.generate_signals(df_tf)
            for tp in [4.0, 5.0, 6.0, 8.0]:
                label = f"{tf_label} RSI<{rsi} SL×1.5 TP×{tp:.0f}"
                run(label, sigs, df_tf, sl=1.5, tp=tp, hold=100)


if __name__ == "__main__":
    main()
