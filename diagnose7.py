"""
Phase 2 후속: BB 돌파 1h 전략 심화 분석
- 연도별 (2022/2023/2024) 분리 검증
- 파라미터 세밀 탐색
- 2025 데이터 검증
- EMA 풀백 1h도 함께 비교
"""
import numpy as np
import pandas as pd
from download_data import fetch
from strategies import BBBreakout, EMAPullback, BBTouch
from backtester import BacktestEngine


def apply_regime(signals, df_daily, df_target, bull_mult=1.0, bear_mult=1.0):
    ma200  = df_daily["close"].rolling(200).mean()
    bull_r = (df_daily["close"] > ma200 * bull_mult).astype(bool)
    bear_r = (df_daily["close"] < ma200 * bear_mult).astype(bool)
    union  = bull_r.index.union(df_target.index)
    bull_s = bull_r.reindex(union).ffill().reindex(df_target.index).fillna(False).astype(bool)
    bear_s = bear_r.reindex(union).ffill().reindex(df_target.index).fillna(False).astype(bool)
    result = signals.copy()
    result[(signals ==  1) & (~bull_s)] = 0
    result[(signals == -1) & (~bear_s)] = 0
    return result


def run(label, sigs, df, sl, tp, hold=100, capital=10000, leverage=10, verbose=False):
    engine = BacktestEngine(
        initial_capital=capital, leverage=leverage, risk_per_trade=0.01,
        sl_atr_mult=sl, tp_atr_mult=tp, max_hold_bars=hold,
    )
    result = engine.run(df, sigs)
    trades = result["trades"]
    if not trades:
        print(f"    {label}: 거래 없음 (신호:{(sigs!=0).sum()}개)")
        return None

    wins   = [t for t in trades if t.pnl > 0]
    loses  = [t for t in trades if t.pnl <= 0]
    longs  = [t for t in trades if t.direction ==  1]
    shorts = [t for t in trades if t.direction == -1]
    win_rate  = len(wins) / len(trades) * 100
    total_pnl = sum(t.pnl for t in trades)
    pf = abs(sum(t.pnl for t in wins)) / abs(sum(t.pnl for t in loses)) if loses else 999
    hold_t = [(t.exit_time - t.entry_time).total_seconds() / 3600 for t in trades]
    final_cap = result["final_capital"]

    flag = "★" if total_pnl > 0 else "  "
    sign = "+" if total_pnl >= 0 else ""
    print(f"  {flag} {label:<72} 거래:{len(trades):3}(L:{len(longs)}/S:{len(shorts)}) "
          f"승:{win_rate:4.1f}% PF:{pf:.2f} 보유:{np.mean(hold_t):.0f}h "
          f"{sign}{total_pnl/capital*100:.1f}% 최종:${final_cap:,.0f}")

    if verbose:
        df_t = pd.DataFrame([{"month": t.entry_time.strftime("%Y-%m"),
                               "pnl": t.pnl, "win": t.pnl > 0} for t in trades])
        mon  = df_t.groupby("month").agg(n=("pnl","count"), wins=("win","sum"), pnl=("pnl","sum"))
        mon["wr"] = (mon["wins"] / mon["n"] * 100).round(1)
        for m, r in mon.iterrows():
            s = "+" if r["pnl"] >= 0 else ""
            bar = "█" * int(r["wr"] / 10)
            print(f"       {m} | {r['n']:3.0f}건 승:{r['wr']:4.1f}% {bar:<10} | {s}${r['pnl']:,.0f}")
    return {"pnl": total_pnl, "wr": win_rate, "n": len(trades), "pf": pf,
            "final": final_cap, "longs": len(longs), "shorts": len(shorts)}


def main():
    print("[데이터 로딩]...")
    df_1h_full = fetch("1h",  "2022-01", "2025-03")  # 2025 포함
    df_1d_full = fetch("1d",  "2021-01", "2025-03")
    df_1h_old  = fetch("1h",  "2022-01", "2024-12")
    df_1d_old  = fetch("1d",  "2021-01", "2024-12")
    print(f"  1h(2022-2025.03):{len(df_1h_full):,} | 1h(2022-2024):{len(df_1h_old):,}")

    sep = "="*115

    # ─────────────────────────────────────────────
    # 실험 1: BB 돌파 1h 연도별 상세 검증
    # ─────────────────────────────────────────────
    print(f"\n{sep}")
    print("실험 1: BB 돌파 1h 최적 조합 - 연도별 (2022/2023/2024/2025Q1)")
    print(sep)

    configs = [
        ("sq20 EMA50", BBBreakout(squeeze_pct=20, trend_filter=True, trend_ema=50)),
        ("sq10 EMA50", BBBreakout(squeeze_pct=10, trend_filter=True, trend_ema=50)),
        ("sq20 noEMA", BBBreakout(squeeze_pct=20, trend_filter=False)),
        ("sq10 noEMA", BBBreakout(squeeze_pct=10, trend_filter=False)),
        ("sq30 EMA50", BBBreakout(squeeze_pct=30, trend_filter=True, trend_ema=50)),
        ("sq20 EMA20", BBBreakout(squeeze_pct=20, trend_filter=True, trend_ema=20)),
    ]

    for name, strat in configs:
        print(f"\n  [{name}]")
        for year_s, year_e, label in [
            ("2022-01", "2022-12", "2022"),
            ("2023-01", "2023-12", "2023"),
            ("2024-01", "2024-12", "2024"),
            ("2025-01", "2025-03", "2025Q1"),
            ("2022-01", "2025-03", "전체"),
        ]:
            ts = pd.Timestamp(year_s, tz="UTC")
            te = pd.Timestamp(year_e, tz="UTC") + pd.Timedelta(days=31)
            df_y = df_1h_full[(df_1h_full.index >= ts) & (df_1h_full.index <= te)]
            if len(df_y) < 100: continue

            sigs_base   = strat.generate_signals(df_y)
            sigs_regime = apply_regime(sigs_base, df_1d_full, df_y)
            n_base  = (sigs_base != 0).sum()
            n_reg   = (sigs_regime != 0).sum()

            for sl, tp in [(1.0, 3.0), (1.5, 3.0), (1.0, 4.0)]:
                run(f"{label} 국면O SL×{sl} TP×{tp:.0f} sig:{n_reg}", sigs_regime, df_y, sl, tp)

    # ─────────────────────────────────────────────
    # 실험 2: 파라미터 세밀 최적화 (SL/TP/스퀴즈)
    # ─────────────────────────────────────────────
    print(f"\n{sep}")
    print("실험 2: 1h BB 돌파 파라미터 세밀 탐색 (전체 2022-2025)")
    print(sep)

    for sq_pct in [10, 15, 20, 25, 30]:
        for trend_ema in [20, 50, 100]:
            strat = BBBreakout(squeeze_pct=sq_pct, trend_filter=True, trend_ema=trend_ema)
            sigs  = strat.generate_signals(df_1h_full)
            sigs_r = apply_regime(sigs, df_1d_full, df_1h_full)
            n = (sigs_r != 0).sum()
            for sl, tp in [(1.0, 2.0), (1.0, 3.0), (1.5, 2.0), (1.5, 3.0), (1.0, 4.0), (1.5, 4.0)]:
                run(f"sq{sq_pct} EMA{trend_ema} SL×{sl} TP×{tp:.0f} sig:{n}", sigs_r, df_1h_full, sl, tp)

    # ─────────────────────────────────────────────
    # 실험 3: 최유망 조합 월별 상세 (전체 기간)
    # ─────────────────────────────────────────────
    print(f"\n{sep}")
    print("실험 3: 최유망 조합 월별 상세 (1h BB돌파 2022-2025)")
    print(sep)

    for name, strat, sl, tp in [
        ("sq20 EMA50 SL1.5 TP3", BBBreakout(squeeze_pct=20, trend_filter=True, trend_ema=50), 1.5, 3.0),
        ("sq10 EMA50 SL1.5 TP3", BBBreakout(squeeze_pct=10, trend_filter=True, trend_ema=50), 1.5, 3.0),
        ("sq20 EMA50 SL1.0 TP3", BBBreakout(squeeze_pct=20, trend_filter=True, trend_ema=50), 1.0, 3.0),
    ]:
        sigs = strat.generate_signals(df_1h_full)
        sigs_r = apply_regime(sigs, df_1d_full, df_1h_full)
        run(f"{name} [2022-2025 전체 월별]", sigs_r, df_1h_full, sl, tp, verbose=True)

    # ─────────────────────────────────────────────
    # 실험 4: EMA 풀백 1h도 같은 조건으로 비교
    # ─────────────────────────────────────────────
    print(f"\n{sep}")
    print("실험 4: EMA 풀백 1h 연도별 + BB 돌파 1h vs RSI 딥매수 비교")
    print(sep)

    strategies = {
        "BB돌파 sq20 EMA50": BBBreakout(squeeze_pct=20, trend_filter=True, trend_ema=50),
        "BB돌파 sq10 EMA50": BBBreakout(squeeze_pct=10, trend_filter=True, trend_ema=50),
        "EMA풀백 9/21/50":   EMAPullback(fast_ema=9, mid_ema=21, slow_ema=50, rsi_long_max=55, rsi_short_min=45),
        "EMA풀백 9/21/100":  EMAPullback(fast_ema=9, mid_ema=21, slow_ema=100, rsi_long_max=55, rsi_short_min=45),
        "BB터치 RSI40":      BBTouch(confirm_candle=True, ema_period=50, ema_slope_period=5,
                                     use_ema_slope=True, use_rsi=True, rsi_long_max=40.0, rsi_short_min=60.0),
    }

    for strat_name, strat in strategies.items():
        print(f"\n  [{strat_name}]")
        cumulative = 0
        for year, ydf in [("2022", df_1h_full), ("2023", df_1h_full),
                           ("2024", df_1h_full), ("2025Q1", df_1h_full)]:
            if year == "2025Q1":
                ts = pd.Timestamp("2025-01-01", tz="UTC")
                te = pd.Timestamp("2025-03-31", tz="UTC")
            else:
                ts = pd.Timestamp(f"{year}-01-01", tz="UTC")
                te = pd.Timestamp(f"{year}-12-31", tz="UTC")
            df_y = ydf[(ydf.index >= ts) & (ydf.index <= te)]
            if len(df_y) < 50: continue

            sigs = strat.generate_signals(df_y)
            sigs_r = apply_regime(sigs, df_1d_full, df_y)
            res = run(f"{year}", sigs_r, df_y, 1.5, 3.0)
            if res:
                cumulative += res["pnl"] / 10000 * 100

        print(f"      → 연간 합계: {'+' if cumulative>=0 else ''}{cumulative:.1f}%")


if __name__ == "__main__":
    main()
