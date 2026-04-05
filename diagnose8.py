"""
Phase 3: sq15 집중 최적화
- sq15 조합이 전체 기간 +35~49%로 최고 성능 확인
- 연도별 상세 검증 (2022/2023/2024/2025Q1)
- EMA 필터 비교 (EMA20 vs EMA50 vs EMA100)
- SL/TP 최적화 (1.5×3 vs 1.5×4 vs 1.0×4)
- 월별 verbose 로그
"""
import numpy as np
import pandas as pd
from download_data import fetch
from strategies import BBBreakout
from backtester import BacktestEngine


def apply_regime(signals, df_daily, df_target):
    ma200  = df_daily["close"].rolling(200).mean()
    bull_r = (df_daily["close"] > ma200).astype(bool)
    bear_r = (df_daily["close"] < ma200).astype(bool)
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
    df_1h = fetch("1h", "2022-01", "2025-03")
    df_1d = fetch("1d", "2021-01", "2025-03")
    print(f"  1h: {len(df_1h):,} | 1d: {len(df_1d):,}")

    sep = "=" * 115

    years = [
        ("2022-01", "2022-12", "2022"),
        ("2023-01", "2023-12", "2023"),
        ("2024-01", "2024-12", "2024"),
        ("2025-01", "2025-03", "2025Q1"),
        ("2022-01", "2025-03", "전체"),
    ]

    # ─────────────────────────────────────────────
    # 실험 1: sq15 연도별 상세 (최유망 조합)
    # ─────────────────────────────────────────────
    print(f"\n{sep}")
    print("실험 1: sq15 연도별 검증 (EMA20/50/100 × SL1.5 × TP3/4)")
    print(sep)

    configs = [
        ("sq15 EMA20",  BBBreakout(squeeze_pct=15, trend_filter=True, trend_ema=20)),
        ("sq15 EMA50",  BBBreakout(squeeze_pct=15, trend_filter=True, trend_ema=50)),
        ("sq15 EMA100", BBBreakout(squeeze_pct=15, trend_filter=True, trend_ema=100)),
    ]

    for name, strat in configs:
        print(f"\n  [{name}]")
        cumulative_3 = 0
        cumulative_4 = 0
        for year_s, year_e, label in years:
            ts = pd.Timestamp(year_s, tz="UTC")
            te = pd.Timestamp(year_e, tz="UTC") + pd.Timedelta(days=31)
            df_y = df_1h[(df_1h.index >= ts) & (df_1h.index <= te)]
            if len(df_y) < 100:
                continue
            sigs = strat.generate_signals(df_y)
            sigs_r = apply_regime(sigs, df_1d, df_y)
            res3 = run(f"{label} SL×1.5 TP×3 sig:{(sigs_r!=0).sum()}", sigs_r, df_y, 1.5, 3.0)
            res4 = run(f"{label} SL×1.5 TP×4 sig:{(sigs_r!=0).sum()}", sigs_r, df_y, 1.5, 4.0)
            if res3 and label != "전체":
                cumulative_3 += res3["pnl"] / 10000 * 100
            if res4 and label != "전체":
                cumulative_4 += res4["pnl"] / 10000 * 100
        print(f"      → 연간 합계(TP3): {'+' if cumulative_3>=0 else ''}{cumulative_3:.1f}% | "
              f"(TP4): {'+' if cumulative_4>=0 else ''}{cumulative_4:.1f}%")

    # ─────────────────────────────────────────────
    # 실험 2: sq15 EMA100 SL×1.5 TP×4 월별 상세
    # ─────────────────────────────────────────────
    print(f"\n{sep}")
    print("실험 2: sq15 EMA100 SL×1.5 TP×4 전체 기간 월별 상세")
    print(sep)

    strat_best = BBBreakout(squeeze_pct=15, trend_filter=True, trend_ema=100)
    sigs = strat_best.generate_signals(df_1h)
    sigs_r = apply_regime(sigs, df_1d, df_1h)
    run("sq15 EMA100 SL×1.5 TP×4 [월별]", sigs_r, df_1h, 1.5, 4.0, verbose=True)

    # ─────────────────────────────────────────────
    # 실험 3: sq15 EMA20 SL×1.5 TP×4 월별 상세
    # ─────────────────────────────────────────────
    print(f"\n{sep}")
    print("실험 3: sq15 EMA20 SL×1.5 TP×4 전체 기간 월별 상세")
    print(sep)

    strat_ema20 = BBBreakout(squeeze_pct=15, trend_filter=True, trend_ema=20)
    sigs = strat_ema20.generate_signals(df_1h)
    sigs_r = apply_regime(sigs, df_1d, df_1h)
    run("sq15 EMA20 SL×1.5 TP×4 [월별]", sigs_r, df_1h, 1.5, 4.0, verbose=True)

    # ─────────────────────────────────────────────
    # 실험 4: sq15 세밀 파라미터 (연도별 일관성 확인)
    # ─────────────────────────────────────────────
    print(f"\n{sep}")
    print("실험 4: sq15 전체 파라미터 조합 연도별 합계 비교")
    print(sep)

    combos = [
        ("sq15 EMA20  SL1.5 TP3", BBBreakout(squeeze_pct=15, trend_filter=True, trend_ema=20),  1.5, 3.0),
        ("sq15 EMA20  SL1.5 TP4", BBBreakout(squeeze_pct=15, trend_filter=True, trend_ema=20),  1.5, 4.0),
        ("sq15 EMA50  SL1.5 TP3", BBBreakout(squeeze_pct=15, trend_filter=True, trend_ema=50),  1.5, 3.0),
        ("sq15 EMA50  SL1.5 TP4", BBBreakout(squeeze_pct=15, trend_filter=True, trend_ema=50),  1.5, 4.0),
        ("sq15 EMA100 SL1.5 TP3", BBBreakout(squeeze_pct=15, trend_filter=True, trend_ema=100), 1.5, 3.0),
        ("sq15 EMA100 SL1.5 TP4", BBBreakout(squeeze_pct=15, trend_filter=True, trend_ema=100), 1.5, 4.0),
    ]

    print(f"\n  {'조합':<28} {'2022':>8} {'2023':>8} {'2024':>8} {'2025Q1':>8} {'합계':>8} {'전체':>8}")
    print(f"  {'-'*80}")

    for name, strat, sl, tp in combos:
        yearly = {}
        total_pnl = 0
        for year_s, year_e, label in years:
            ts = pd.Timestamp(year_s, tz="UTC")
            te = pd.Timestamp(year_e, tz="UTC") + pd.Timedelta(days=31)
            df_y = df_1h[(df_1h.index >= ts) & (df_1h.index <= te)]
            if len(df_y) < 100:
                yearly[label] = None
                continue
            sigs = strat.generate_signals(df_y)
            sigs_r = apply_regime(sigs, df_1d, df_y)
            res = run(f"", sigs_r, df_y, sl, tp) if False else None  # suppress output

            engine = BacktestEngine(
                initial_capital=10000, leverage=10, risk_per_trade=0.01,
                sl_atr_mult=sl, tp_atr_mult=tp, max_hold_bars=100,
            )
            result = engine.run(df_y, sigs_r)
            trades = result["trades"]
            pnl_pct = sum(t.pnl for t in trades) / 10000 * 100 if trades else 0
            yearly[label] = pnl_pct
            if label != "전체":
                total_pnl += pnl_pct

        def fmt(v):
            if v is None: return "  N/A"
            s = "+" if v >= 0 else ""
            return f"{s}{v:.1f}%"

        print(f"  {name:<28} {fmt(yearly.get('2022')):>8} {fmt(yearly.get('2023')):>8} "
              f"{fmt(yearly.get('2024')):>8} {fmt(yearly.get('2025Q1')):>8} "
              f"{fmt(total_pnl):>8} {fmt(yearly.get('전체')):>8}")

    # ─────────────────────────────────────────────
    # 실험 5: 최종 최적 조합 롱/숏 분리 분석
    # ─────────────────────────────────────────────
    print(f"\n{sep}")
    print("실험 5: sq15 EMA100 SL×1.5 TP×4 롱/숏 별도 분석")
    print(sep)

    strat_final = BBBreakout(squeeze_pct=15, trend_filter=True, trend_ema=100)

    for year_s, year_e, label in years:
        ts = pd.Timestamp(year_s, tz="UTC")
        te = pd.Timestamp(year_e, tz="UTC") + pd.Timedelta(days=31)
        df_y = df_1h[(df_1h.index >= ts) & (df_1h.index <= te)]
        if len(df_y) < 100:
            continue
        sigs = strat_final.generate_signals(df_y)
        sigs_r = apply_regime(sigs, df_1d, df_y)

        engine = BacktestEngine(
            initial_capital=10000, leverage=10, risk_per_trade=0.01,
            sl_atr_mult=1.5, tp_atr_mult=4.0, max_hold_bars=100,
        )
        result = engine.run(df_y, sigs_r)
        trades = result["trades"]
        if not trades:
            continue

        longs  = [t for t in trades if t.direction ==  1]
        shorts = [t for t in trades if t.direction == -1]
        l_wr   = sum(1 for t in longs  if t.pnl > 0) / len(longs)  * 100 if longs  else 0
        s_wr   = sum(1 for t in shorts if t.pnl > 0) / len(shorts) * 100 if shorts else 0
        l_pnl  = sum(t.pnl for t in longs)
        s_pnl  = sum(t.pnl for t in shorts)
        tot_pnl = sum(t.pnl for t in trades)

        print(f"\n  [{label}] 총:{len(trades)}건 {'+' if tot_pnl>=0 else ''}{tot_pnl/10000*100:.1f}%")
        if longs:
            print(f"     롱  {len(longs):3}건 승:{l_wr:4.1f}% PnL:{'+' if l_pnl>=0 else ''}{l_pnl/10000*100:.1f}%")
        if shorts:
            print(f"     숏  {len(shorts):3}건 승:{s_wr:4.1f}% PnL:{'+' if s_pnl>=0 else ''}{s_pnl/10000*100:.1f}%")


if __name__ == "__main__":
    main()
