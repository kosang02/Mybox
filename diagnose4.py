"""
진단 4: 시장 국면 필터 + 최종 전략 검증
- 200d MA 기반 국면 분류: 강세장(롱만) vs 약세장(숏만)
- 방향별 필터링: bull=롱, bear=숏
- 2022-2024 통합 검증
"""
import numpy as np
import pandas as pd
from download_data import fetch
from strategies import BBTouch, BBSqueezeEMA
from backtester import BacktestEngine


def calc_rsi(close, period=14):
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(com=period-1, adjust=False).mean()
    loss  = (-delta).clip(lower=0).ewm(com=period-1, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def apply_regime_filter(signals: pd.Series, df_daily: pd.DataFrame, df_target: pd.DataFrame) -> pd.Series:
    """200일 MA 기반 국면 필터
    - 일봉 종가 > 200일 MA → 강세장 (롱만)
    - 일봉 종가 < 200일 MA → 약세장 (숏만)
    """
    ma200  = df_daily["close"].rolling(200).mean()
    bull_raw = (df_daily["close"] > ma200)  # Boolean Series
    # 타겟 타임프레임에 ffill 리샘플 - 반드시 bool로 변환
    union_idx = bull_raw.index.union(df_target.index)
    regime = bull_raw.reindex(union_idx).ffill().reindex(df_target.index).fillna(True).astype(bool)

    result = signals.copy()
    result[(signals ==  1) & (~regime)] = 0   # 약세장에서 롱 제거
    result[(signals == -1) & ( regime)] = 0   # 강세장에서 숏 제거
    return result


def run_test(label, sigs, df, sl, tp, hold, capital=10000, leverage=10, verbose=False):
    engine = BacktestEngine(
        initial_capital=capital, leverage=leverage, risk_per_trade=0.01,
        sl_atr_mult=sl, tp_atr_mult=tp, max_hold_bars=hold,
    )
    result = engine.run(df, sigs)
    trades = result["trades"]
    if not trades:
        print(f"    {label}: 거래 없음")
        return 0

    wins  = [t for t in trades if t.pnl > 0]
    loses = [t for t in trades if t.pnl <= 0]
    longs  = [t for t in trades if t.direction ==  1]
    shorts = [t for t in trades if t.direction == -1]
    win_rate = len(wins) / len(trades) * 100
    total_pnl = sum(t.pnl for t in trades)
    pf = abs(sum(t.pnl for t in wins)) / abs(sum(t.pnl for t in loses)) if loses else 999
    tps = sum(1 for t in trades if t.exit_reason == "TP")
    sls = sum(1 for t in trades if t.exit_reason == "SL")
    final_cap = result["final_capital"]

    flag = "★" if total_pnl > 0 else "  "
    sign = "+" if total_pnl >= 0 else ""
    print(f"  {flag} {label:<65} 거래:{len(trades):3}(L:{len(longs)}/S:{len(shorts)}) 승:{win_rate:4.1f}% TP:{tps}/SL:{sls} PF:{pf:.2f} {sign}{total_pnl/capital*100:.1f}% 최종:${final_cap:,.0f}")

    if verbose and trades:
        # 월별
        df_t = pd.DataFrame([{"month": t.entry_time.strftime("%Y-%m"), "pnl": t.pnl, "win": t.pnl > 0} for t in trades])
        monthly = df_t.groupby("month").agg(n=("pnl","count"), wins=("win","sum"), pnl=("pnl","sum"))
        monthly["wr"] = monthly["wins"] / monthly["n"] * 100
        print(f"\n     월별 상세:")
        for m, r in monthly.iterrows():
            s = "+" if r["pnl"] >= 0 else ""
            bar = "█" * int(r["wr"] / 10)
            print(f"       {m} | {r['n']:3.0f}건 승:{r['wr']:4.1f}% {bar:<10} | {s}${r['pnl']:,.0f}")
        print()

    return total_pnl


def main():
    print("[데이터 로딩]...")
    df_5m  = fetch("5m",  "2022-01", "2024-12")
    df_15m = fetch("15m", "2022-01", "2024-12")
    df_1d  = fetch("1d",  "2021-01", "2024-12")  # 200일 MA 계산을 위해 2021 포함
    print(f"  5m:{len(df_5m):,} | 15m:{len(df_15m):,} | 1d:{len(df_1d):,}")

    # ─────────────────────────────────
    # 실험 A: 200일 MA 국면 필터
    # ─────────────────────────────────
    print(f"\n{'='*110}")
    print("실험 A: 200일 MA 국면 필터 (강세=롱, 약세=숏)")
    print(f"{'='*110}")

    best_strat = BBTouch(
        confirm_candle=True, ema_period=50, ema_slope_period=5,
        use_ema_slope=True, use_rsi=True,
        rsi_long_max=40.0, rsi_short_min=60.0
    )

    sigs_base = best_strat.generate_signals(df_5m)
    sigs_regime = apply_regime_filter(sigs_base, df_1d, df_5m)

    n_base   = (sigs_base != 0).sum()
    n_regime = (sigs_regime != 0).sum()
    n_bull   = (sigs_regime == 1).sum()
    n_bear   = (sigs_regime == -1).sum()
    print(f"  원본 시그널: {n_base}개 → 국면필터 후: {n_regime}개 (롱:{n_bull}, 숏:{n_bear})")

    for tp in [5.0, 6.0, 7.0, 8.0]:
        run_test(f"5m BB터치 RSI40 국면필터 TP×{tp:.0f}", sigs_regime, df_5m, 1.5, tp, 200)

    print()
    for tp in [5.0, 6.0, 7.0, 8.0]:
        run_test(f"5m BB터치 RSI40 국면X TP×{tp:.0f}", sigs_base, df_5m, 1.5, tp, 200)

    # ─────────────────────────────────
    # 실험 B: 15m + 국면 필터
    # ─────────────────────────────────
    print(f"\n{'='*110}")
    print("실험 B: 15m 타임프레임 + 국면 필터")
    print(f"{'='*110}")
    sigs_15m = best_strat.generate_signals(df_15m)
    sigs_15m_regime = apply_regime_filter(sigs_15m, df_1d, df_15m)
    n = (sigs_15m_regime != 0).sum()
    print(f"  15m 국면필터 시그널: {n}개")
    for tp in [4.0, 5.0, 6.0, 7.0]:
        run_test(f"15m RSI40 국면 TP×{tp:.0f}", sigs_15m_regime, df_15m, 1.5, tp, 100)

    # ─────────────────────────────────
    # 실험 C: RSI threshold + 국면필터
    # ─────────────────────────────────
    print(f"\n{'='*110}")
    print("실험 C: RSI 기준 + 국면 필터 최적 탐색 (5m)")
    print(f"{'='*110}")
    for rsi in [35, 38, 40, 42, 45]:
        strat = BBTouch(
            confirm_candle=True, ema_period=50, ema_slope_period=5,
            use_ema_slope=True, use_rsi=True,
            rsi_long_max=float(rsi), rsi_short_min=float(100-rsi)
        )
        sigs = strat.generate_signals(df_5m)
        sigs_r = apply_regime_filter(sigs, df_1d, df_5m)
        n = (sigs_r != 0).sum()
        for tp in [5.0, 6.0, 7.0]:
            run_test(f"RSI<{rsi} 국면 sig:{n} TP×{tp:.0f}", sigs_r, df_5m, 1.5, tp, 200)

    # ─────────────────────────────────
    # 실험 D: 최적 조합 연도별 상세 검증
    # ─────────────────────────────────
    print(f"\n{'='*110}")
    print("실험 D: 최적 조합 연도별 상세 검증")
    print(f"{'='*110}")
    for year_start, year_end in [("2022-01", "2022-12"), ("2023-01", "2023-12"), ("2024-01", "2024-12")]:
        df_y = df_5m[(df_5m.index >= pd.Timestamp(year_start, tz="UTC")) &
                     (df_5m.index <= pd.Timestamp(year_end, tz="UTC"))]
        sigs_y = best_strat.generate_signals(df_y)
        sigs_yr = apply_regime_filter(sigs_y, df_1d, df_y)
        year = year_start[:4]
        print(f"\n  [{year}] 원본:{(sigs_y!=0).sum()}건 → 국면필터:{(sigs_yr!=0).sum()}건")
        run_test(f"{year} 국면필터 SL1.5 TP×6", sigs_yr, df_y, 1.5, 6.0, 200)
        run_test(f"{year} 국면필터 SL1.5 TP×7", sigs_yr, df_y, 1.5, 7.0, 200)
        run_test(f"{year} 국면X SL1.5 TP×6",    sigs_y,  df_y, 1.5, 6.0, 200)

    # ─────────────────────────────────
    # 실험 E: 최유망 조합 상세 출력
    # ─────────────────────────────────
    print(f"\n{'='*110}")
    print("실험 E: 최유망 조합 월별 상세 (5m RSI40 국면필터 TP×7)")
    print(f"{'='*110}")
    sigs_best = apply_regime_filter(
        best_strat.generate_signals(df_5m), df_1d, df_5m
    )
    run_test("5m RSI40 국면 TP×7 [2022-2024 전체]", sigs_best, df_5m, 1.5, 7.0, 200, verbose=True)


if __name__ == "__main__":
    main()
