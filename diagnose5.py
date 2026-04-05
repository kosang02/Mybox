"""
Phase 1: 현재 최적 전략 심화
- 강세장 강도 필터 (BTC > 200d MA × N배)
- 약세장 숏 RSI threshold 조정
- 2025 데이터 검증
- EMA 기울기 없이 regime만 사용한 버전 vs 현재 최적
"""
import numpy as np
import pandas as pd
from download_data import fetch
from strategies import BBTouch
from backtester import BacktestEngine


def apply_regime_v2(signals, df_daily, df_target, bull_mult=1.0, bear_mult=1.0):
    """강화된 국면 필터
    bull_mult > 1.0: 200d MA × N배 위에서만 롱 (예: 1.1 = 10% 위)
    bear_mult < 1.0 or 1.0: 200d MA × N배 아래에서만 숏
    """
    ma200 = df_daily["close"].rolling(200).mean()
    strong_bull = (df_daily["close"] > ma200 * bull_mult).astype(bool)
    strong_bear = (df_daily["close"] < ma200 * bear_mult).astype(bool)

    union_idx   = strong_bull.index.union(df_target.index)
    bull_regime = strong_bull.reindex(union_idx).ffill().reindex(df_target.index).fillna(False).astype(bool)
    bear_regime = strong_bear.reindex(union_idx).ffill().reindex(df_target.index).fillna(False).astype(bool)

    result = signals.copy()
    result[(signals ==  1) & (~bull_regime)] = 0   # 강세장 아닐 때 롱 제거
    result[(signals == -1) & (~bear_regime)] = 0   # 약세장 아닐 때 숏 제거
    return result


def run(label, sigs, df, sl, tp, hold, capital=10000, leverage=10):
    engine = BacktestEngine(
        initial_capital=capital, leverage=leverage, risk_per_trade=0.01,
        sl_atr_mult=sl, tp_atr_mult=tp, max_hold_bars=hold,
    )
    result = engine.run(df, sigs)
    trades = result["trades"]
    if not trades:
        print(f"    {label}: 거래 없음")
        return None

    wins   = [t for t in trades if t.pnl > 0]
    loses  = [t for t in trades if t.pnl <= 0]
    longs  = [t for t in trades if t.direction ==  1]
    shorts = [t for t in trades if t.direction == -1]
    win_rate  = len(wins) / len(trades) * 100
    total_pnl = sum(t.pnl for t in trades)
    pf = abs(sum(t.pnl for t in wins)) / abs(sum(t.pnl for t in loses)) if loses else 999
    tps = sum(1 for t in trades if t.exit_reason == "TP")
    sls = sum(1 for t in trades if t.exit_reason == "SL")

    flag = "★" if total_pnl > 0 else "  "
    sign = "+" if total_pnl >= 0 else ""
    print(f"  {flag} {label:<70} 거래:{len(trades):3}(L:{len(longs)}/S:{len(shorts)}) "
          f"승:{win_rate:4.1f}% PF:{pf:.2f} {sign}{total_pnl/capital*100:.1f}%")
    return {"pnl": total_pnl, "win_rate": win_rate, "trades": len(trades), "pf": pf}


def main():
    print("[데이터 로딩]...")
    df_5m  = fetch("5m",  "2022-01", "2024-12")
    df_15m = fetch("15m", "2022-01", "2024-12")
    df_1d  = fetch("1d",  "2021-01", "2024-12")
    print(f"  5m:{len(df_5m):,} | 15m:{len(df_15m):,} | 1d:{len(df_1d):,}")

    # ─────────────────────────────────────────────
    # 실험 1: 강세장 강도 필터 변화
    # ─────────────────────────────────────────────
    print(f"\n{'='*115}")
    print("실험 1: 강세장 강도 필터 (bull_mult) - 200d MA × N배 이상일 때만 롱")
    print(f"{'='*115}")

    # RSI 엄격 (slope=True) vs 완화 (slope=False) 비교
    for use_slope in [True, False]:
        for rsi_l, rsi_s in [(40, 60), (42, 58), (45, 55)]:
            strat = BBTouch(
                confirm_candle=True, ema_period=50, ema_slope_period=5,
                use_ema_slope=use_slope, use_rsi=True,
                rsi_long_max=float(rsi_l), rsi_short_min=float(rsi_s)
            )
            base_sigs = strat.generate_signals(df_5m)
            for bull_m in [1.0, 1.05, 1.10, 1.15, 1.20]:
                for bear_m in [1.0, 0.95, 0.90]:
                    sigs = apply_regime_v2(base_sigs, df_1d, df_5m, bull_m, bear_m)
                    n_l = (sigs==1).sum(); n_s = (sigs==-1).sum()
                    if n_l + n_s < 5:
                        continue
                    label = (f"slope:{'O' if use_slope else 'X'} RSI<{rsi_l}/>{rsi_s} "
                             f"bull×{bull_m:.2f} bear×{bear_m:.2f} sig:{n_l}L/{n_s}S")
                    run(label, sigs, df_5m, 1.5, 7.0, 200)

    # ─────────────────────────────────────────────
    # 실험 2: 약세장 숏 전략 최적화 (RSI threshold)
    # ─────────────────────────────────────────────
    print(f"\n{'='*115}")
    print("실험 2: 약세장 숏 최적화 - RSI>N 상단 터치만 (slope=False, bear only)")
    print(f"{'='*115}")

    for rsi_short in [55, 60, 65, 70]:
        strat = BBTouch(
            confirm_candle=True, ema_period=50, ema_slope_period=5,
            use_ema_slope=False, use_rsi=True,
            rsi_long_max=40.0, rsi_short_min=float(rsi_short)
        )
        base_sigs = strat.generate_signals(df_5m)
        for bear_m in [1.0, 0.95, 0.90]:
            sigs = apply_regime_v2(base_sigs, df_1d, df_5m, bull_mult=99.0, bear_mult=bear_m)
            n_s = (sigs==-1).sum()
            if n_s < 5: continue
            for tp in [4.0, 5.0, 6.0, 7.0]:
                label = f"숏전용 RSI>{rsi_short} bear×{bear_m:.2f} sig:{n_s}S TP×{tp:.0f}"
                run(label, sigs, df_5m, 1.5, tp, 200)

    # ─────────────────────────────────────────────
    # 실험 3: 강세장 롱 전략 최적화 (slope=False + 강세장 강도)
    # ─────────────────────────────────────────────
    print(f"\n{'='*115}")
    print("실험 3: 강세장 롱 최적화 - slope=False + bull 강도 필터 (롱만)")
    print(f"{'='*115}")

    for rsi_l in [38, 40, 42, 45]:
        strat = BBTouch(
            confirm_candle=True, ema_period=50, ema_slope_period=5,
            use_ema_slope=False, use_rsi=True,
            rsi_long_max=float(rsi_l), rsi_short_min=100.0  # 숏 없음
        )
        base_sigs = strat.generate_signals(df_5m)
        # 롱 신호만 남김
        base_sigs_long = base_sigs.copy()
        base_sigs_long[base_sigs_long == -1] = 0

        for bull_m in [1.0, 1.05, 1.10, 1.15]:
            sigs = apply_regime_v2(base_sigs_long, df_1d, df_5m, bull_m, 0.0)
            n_l = (sigs==1).sum()
            if n_l < 5: continue
            for tp in [5.0, 6.0, 7.0, 8.0]:
                label = f"롱전용 RSI<{rsi_l} bull×{bull_m:.2f} sig:{n_l}L TP×{tp:.0f}"
                run(label, sigs, df_5m, 1.5, tp, 200)

    # ─────────────────────────────────────────────
    # 실험 4: 상위 조합 연도별 검증
    # ─────────────────────────────────────────────
    print(f"\n{'='*115}")
    print("실험 4: slope=True 최적 조합 연도별 (RSI<40 bull×1.05 TP×7)")
    print(f"{'='*115}")
    strat_best = BBTouch(
        confirm_candle=True, ema_period=50, ema_slope_period=5,
        use_ema_slope=True, use_rsi=True, rsi_long_max=40.0, rsi_short_min=60.0
    )
    for year in ["2022", "2023", "2024"]:
        mask = df_5m.index.year == int(year)
        df_y = df_5m[mask]
        base_y = strat_best.generate_signals(df_y)
        sigs_y = apply_regime_v2(base_y, df_1d, df_y, 1.05, 0.95)
        n_l = (sigs_y==1).sum(); n_s = (sigs_y==-1).sum()
        print(f"\n  [{year}] 신호: 롱:{n_l} 숏:{n_s}")
        for tp in [6.0, 7.0, 8.0]:
            run(f"{year} SL1.5 TP×{tp:.0f}", sigs_y, df_y, 1.5, tp, 200)

    print(f"\n{'='*115}")
    print("실험 4b: slope=False 롱전용 RSI<40 bull×1.10 연도별")
    print(f"{'='*115}")
    strat_long = BBTouch(
        confirm_candle=True, ema_period=50, ema_slope_period=5,
        use_ema_slope=False, use_rsi=True, rsi_long_max=40.0, rsi_short_min=100.0
    )
    for year in ["2022", "2023", "2024"]:
        mask = df_5m.index.year == int(year)
        df_y = df_5m[mask]
        base_y = strat_long.generate_signals(df_y)
        base_y[base_y == -1] = 0
        sigs_y = apply_regime_v2(base_y, df_1d, df_y, 1.10, 0.0)
        n_l = (sigs_y==1).sum()
        print(f"\n  [{year}] 롱 신호: {n_l}건")
        for tp in [5.0, 6.0, 7.0]:
            run(f"{year} 롱전용 TP×{tp:.0f}", sigs_y, df_y, 1.5, tp, 200)


if __name__ == "__main__":
    main()
