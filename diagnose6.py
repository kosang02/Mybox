"""
Phase 2: 새 전략 테스트
- 전략 C: EMA 풀백 (추세 내 눌림목)
- 전략 D: BB 스퀴즈 돌파
"""
import numpy as np
import pandas as pd
from download_data import fetch
from strategies import EMAPullback, BBBreakout, BBTouch
from backtester import BacktestEngine


def apply_regime(signals, df_daily, df_target, bull_mult=1.0, bear_mult=1.0):
    ma200    = df_daily["close"].rolling(200).mean()
    bull_r   = (df_daily["close"] > ma200 * bull_mult).astype(bool)
    bear_r   = (df_daily["close"] < ma200 * bear_mult).astype(bool)
    union    = bull_r.index.union(df_target.index)
    bull_s   = bull_r.reindex(union).ffill().reindex(df_target.index).fillna(False).astype(bool)
    bear_s   = bear_r.reindex(union).ffill().reindex(df_target.index).fillna(False).astype(bool)
    result   = signals.copy()
    result[(signals ==  1) & (~bull_s)] = 0
    result[(signals == -1) & (~bear_s)] = 0
    return result


def run(label, sigs, df, sl, tp, hold, capital=10000, leverage=10, show_monthly=False):
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
    hold_times = [(t.exit_time - t.entry_time).total_seconds() / 60 for t in trades]

    flag = "★" if total_pnl > 0 else "  "
    sign = "+" if total_pnl >= 0 else ""
    print(f"  {flag} {label:<72} 거래:{len(trades):4}(L:{len(longs)}/S:{len(shorts)}) "
          f"승:{win_rate:4.1f}% PF:{pf:.2f} 보유:{np.mean(hold_times):.0f}m "
          f"{sign}{total_pnl/capital*100:.1f}%")

    if show_monthly and trades:
        df_t = pd.DataFrame([{"month": t.entry_time.strftime("%Y-%m"),
                               "pnl": t.pnl, "win": t.pnl > 0} for t in trades])
        mon  = df_t.groupby("month").agg(n=("pnl","count"), wins=("win","sum"), pnl=("pnl","sum"))
        mon["wr"] = (mon["wins"] / mon["n"] * 100).round(1)
        for m, r in mon.iterrows():
            s = "+" if r["pnl"] >= 0 else ""
            print(f"       {m} | {r['n']:3.0f}건 승:{r['wr']:4.1f}% | {s}${r['pnl']:,.0f}")
    return {"pnl": total_pnl, "wr": win_rate, "n": len(trades), "pf": pf}


def main():
    print("[데이터 로딩]...")
    df_5m  = fetch("5m",  "2022-01", "2024-12")
    df_15m = fetch("15m", "2022-01", "2024-12")
    df_1h  = fetch("1h",  "2022-01", "2024-12")
    df_1d  = fetch("1d",  "2021-01", "2024-12")
    print(f"  5m:{len(df_5m):,} | 15m:{len(df_15m):,} | 1h:{len(df_1h):,} | 1d:{len(df_1d):,}")

    sep = "="*115

    # ═══════════════════════════════════════════════════════
    # 전략 C: EMA 풀백
    # ═══════════════════════════════════════════════════════
    print(f"\n{sep}")
    print("전략 C: EMA 풀백 (추세 내 EMA21 눌림목 진입)")
    print(sep)

    configs_c = [
        EMAPullback(fast_ema=9, mid_ema=21, slow_ema=50, rsi_long_max=55, rsi_short_min=45, require_bounce=True),
        EMAPullback(fast_ema=9, mid_ema=21, slow_ema=50, rsi_long_max=55, rsi_short_min=45, require_bounce=False),
        EMAPullback(fast_ema=9, mid_ema=21, slow_ema=50, rsi_long_max=60, rsi_short_min=40, require_bounce=True),
        EMAPullback(fast_ema=9, mid_ema=21, slow_ema=50, rsi_long_max=50, rsi_short_min=50, require_bounce=True),
        EMAPullback(fast_ema=9, mid_ema=21, slow_ema=100, rsi_long_max=55, rsi_short_min=45, require_bounce=True),
        EMAPullback(fast_ema=9, mid_ema=21, slow_ema=200, rsi_long_max=55, rsi_short_min=45, require_bounce=True),
    ]

    for df, tf in [(df_5m, "5m"), (df_15m, "15m"), (df_1h, "1h")]:
        print(f"\n  [{tf}]")
        for strat in configs_c:
            sigs = strat.generate_signals(df)
            n_sig = (sigs != 0).sum()
            for sl, tp in [(1.0, 2.0), (1.0, 3.0), (1.5, 3.0), (1.5, 4.0)]:
                label = f"{tf} {str(strat)[:50]} SL×{sl} TP×{tp:.0f} sig:{n_sig}"
                run(label, sigs, df, sl, tp, 200)

    # ═══════════════════════════════════════════════════════
    # 전략 C + 국면 필터
    # ═══════════════════════════════════════════════════════
    print(f"\n{sep}")
    print("전략 C + 200d MA 국면 필터")
    print(sep)
    best_c = EMAPullback(fast_ema=9, mid_ema=21, slow_ema=50,
                         rsi_long_max=55, rsi_short_min=45, require_bounce=True)
    for df, tf in [(df_5m, "5m"), (df_15m, "15m"), (df_1h, "1h")]:
        sigs_base  = best_c.generate_signals(df)
        sigs_regime = apply_regime(sigs_base, df_1d, df)
        n = (sigs_regime != 0).sum()
        print(f"\n  [{tf}] 국면필터 후 신호: {n}건")
        for sl, tp in [(1.0, 2.0), (1.0, 3.0), (1.5, 3.0), (1.5, 4.0), (1.0, 4.0)]:
            label = f"{tf} EMA풀백 국면O SL×{sl} TP×{tp:.0f}"
            run(label, sigs_regime, df, sl, tp, 200)

    # ═══════════════════════════════════════════════════════
    # 전략 D: BB 스퀴즈 돌파
    # ═══════════════════════════════════════════════════════
    print(f"\n{sep}")
    print("전략 D: BB 스퀴즈 돌파 (변동성 압축 후 방향성 돌파)")
    print(sep)

    configs_d = [
        BBBreakout(squeeze_pct=20, trend_filter=False),
        BBBreakout(squeeze_pct=20, trend_filter=True, trend_ema=50),
        BBBreakout(squeeze_pct=10, trend_filter=False),
        BBBreakout(squeeze_pct=10, trend_filter=True, trend_ema=50),
        BBBreakout(squeeze_pct=30, trend_filter=True, trend_ema=50),
        BBBreakout(squeeze_pct=20, trend_filter=True, trend_ema=20),
    ]

    for df, tf in [(df_5m, "5m"), (df_15m, "15m"), (df_1h, "1h")]:
        print(f"\n  [{tf}]")
        for strat in configs_d:
            sigs  = strat.generate_signals(df)
            n_sig = (sigs != 0).sum()
            for sl, tp in [(1.0, 2.0), (1.0, 3.0), (1.5, 2.0), (1.5, 3.0)]:
                label = f"{tf} {str(strat)[:50]} SL×{sl} TP×{tp:.0f} sig:{n_sig}"
                run(label, sigs, df, sl, tp, 200)

    # ═══════════════════════════════════════════════════════
    # 전략 D + 국면 필터
    # ═══════════════════════════════════════════════════════
    print(f"\n{sep}")
    print("전략 D + 국면 필터 연도별 검증")
    print(sep)
    for strat_d in [
        BBBreakout(squeeze_pct=20, trend_filter=True, trend_ema=50),
        BBBreakout(squeeze_pct=10, trend_filter=True, trend_ema=50),
    ]:
        for df, tf in [(df_5m, "5m"), (df_15m, "15m"), (df_1h, "1h")]:
            sigs_base   = strat_d.generate_signals(df)
            sigs_regime = apply_regime(sigs_base, df_1d, df)
            n = (sigs_regime != 0).sum()
            print(f"\n  [{tf}] {strat_d} 국면필터 신호:{n}건")
            for sl, tp in [(1.0, 2.0), (1.0, 3.0), (1.5, 3.0)]:
                run(f"{tf} SL×{sl} TP×{tp:.0f}", sigs_regime, df, sl, tp, 200)

    # ═══════════════════════════════════════════════════════
    # 연도별 상세 (가장 유망한 조합들)
    # ═══════════════════════════════════════════════════════
    print(f"\n{sep}")
    print("연도별 검증: EMA 풀백 vs BB 돌파 (최유망 조합)")
    print(sep)

    candidates = [
        ("EMA풀백 5m", EMAPullback(fast_ema=9, mid_ema=21, slow_ema=50,
                                    rsi_long_max=55, rsi_short_min=45, require_bounce=True),
         df_5m, 1.5, 3.0),
        ("EMA풀백 15m", EMAPullback(fast_ema=9, mid_ema=21, slow_ema=50,
                                    rsi_long_max=55, rsi_short_min=45, require_bounce=True),
         df_15m, 1.5, 3.0),
        ("BB돌파 15m sq20", BBBreakout(squeeze_pct=20, trend_filter=True, trend_ema=50),
         df_15m, 1.0, 3.0),
        ("BB돌파 1h sq10", BBBreakout(squeeze_pct=10, trend_filter=True, trend_ema=50),
         df_1h, 1.0, 3.0),
    ]

    for name, strat, df, sl, tp in candidates:
        sigs_base   = strat.generate_signals(df)
        sigs_regime = apply_regime(sigs_base, df_1d, df)
        print(f"\n  [{name}] 전체 신호:{(sigs_base!=0).sum()} 국면필터:{(sigs_regime!=0).sum()}")
        for year in ["2022", "2023", "2024"]:
            mask = df.index.year == int(year)
            df_y = df[mask]
            sy   = strat.generate_signals(df_y)
            sy_r = apply_regime(sy, df_1d, df_y)
            res  = run(f"{year} 국면O", sy_r, df_y, sl, tp, 200)
            if res is None:
                res = run(f"{year} 국면X", sy, df_y, sl, tp, 200)


if __name__ == "__main__":
    main()
