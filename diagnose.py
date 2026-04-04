"""
전략 진단 스크립트
- 현재 전략이 왜 지는지 분석
- 승률, 손익비, 시그널 통계 출력
"""
import numpy as np
import pandas as pd
from download_data import fetch
from strategies import BBSqueezeEMA, BBTouch
from backtester import BacktestEngine, calc_metrics


def diagnose_strategy(name, strategy, df, engine_params, label=""):
    sigs = strategy.generate_signals(df)
    n_sig = (sigs != 0).sum()
    n_long = (sigs == 1).sum()
    n_short = (sigs == -1).sum()

    engine = BacktestEngine(**engine_params)
    result = engine.run(df, sigs)
    trades = result["trades"]

    if not trades:
        print(f"  {name}: 거래 0건 (시그널: {n_sig}개)")
        return

    wins  = [t for t in trades if t.pnl > 0]
    loses = [t for t in trades if t.pnl <= 0]
    tps   = [t for t in trades if t.exit_reason == "TP"]
    sls   = [t for t in trades if t.exit_reason == "SL"]
    liq   = [t for t in trades if t.exit_reason == "LIQ"]
    force = [t for t in trades if t.exit_reason == "FORCE"]

    total_pnl = sum(t.pnl for t in trades)
    win_rate  = len(wins) / len(trades) * 100
    avg_win   = np.mean([t.pnl for t in wins]) if wins else 0
    avg_loss  = np.mean([t.pnl for t in loses]) if loses else 0
    pf        = abs(sum(t.pnl for t in wins)) / abs(sum(t.pnl for t in loses)) if loses else 999

    hold_times = [(t.exit_time - t.entry_time).total_seconds() / 60 for t in trades]

    print(f"\n{'='*60}")
    print(f"  {name} {label}")
    print(f"{'='*60}")
    print(f"  시그널: {n_sig}개 (롱:{n_long}, 숏:{n_short})")
    print(f"  실행 거래: {len(trades)}건")
    print(f"  승률: {win_rate:.1f}%  ({len(wins)}승 / {len(loses)}패)")
    print(f"  평균 수익: ${avg_win:,.1f}  |  평균 손실: ${avg_loss:,.1f}")
    print(f"  수익 팩터: {pf:.2f}")
    print(f"  총 PnL: ${total_pnl:,.0f}  ({total_pnl/10000*100:.1f}%)")
    print(f"  종료 사유: TP={len(tps)}, SL={len(sls)}, LIQ={len(liq)}, FORCE={len(force)}")
    print(f"  평균 보유시간: {np.mean(hold_times):.0f}분  (중앙값: {np.median(hold_times):.0f}분)")

    # 월별 승률
    if trades:
        df_t = pd.DataFrame([{
            "month": t.entry_time.strftime("%Y-%m"),
            "win": int(t.pnl > 0),
            "pnl": t.pnl
        } for t in trades])
        monthly = df_t.groupby("month").agg(trades=("win","count"), wins=("win","sum"), pnl=("pnl","sum"))
        monthly["win_rate"] = (monthly["wins"] / monthly["trades"] * 100).round(1)
        print(f"\n  월별 성과 (최근 12개월):")
        for m, row in monthly.tail(12).iterrows():
            bar = "█" * int(row["win_rate"] / 10)
            sign = "+" if row["pnl"] >= 0 else ""
            print(f"    {m}: 승률 {row['win_rate']:5.1f}% {bar:<10} | {row['trades']:3.0f}건 | {sign}${row['pnl']:,.0f}")


def main():
    print("\n[데이터 로딩] 5m 2023-01~2024-12...")
    df_5m = fetch("5m", "2023-01", "2024-12")
    print(f"  {len(df_5m)}개 캔들")

    print("\n[데이터 로딩] 15m 2023-01~2024-12...")
    df_15m = fetch("15m", "2023-01", "2024-12")
    print(f"  {len(df_15m)}개 캔들")

    base_engine = {
        "initial_capital": 10_000,
        "leverage": 10,
        "risk_per_trade": 0.01,
        "sl_atr_mult": 1.5,
        "tp_atr_mult": 3.0,
        "max_hold_bars": 200,
    }

    # ── 전략 A: 원본 (RSI 없음) ──
    diagnose_strategy(
        "전략A 원본",
        BBSqueezeEMA(ema_period=50, squeeze_percentile=20, slope_period=3, use_rsi=False),
        df_5m, base_engine, "[5m]"
    )

    # ── 전략 A: RSI 추가 ──
    diagnose_strategy(
        "전략A + RSI",
        BBSqueezeEMA(ema_period=50, squeeze_percentile=20, slope_period=3, use_rsi=True, rsi_long_min=50.0, rsi_short_max=50.0),
        df_5m, base_engine, "[5m]"
    )

    # ── 전략 B: 구버전 EMA 필터 (close vs ema) 시뮬레이션 ──
    # use_ema_slope=False, rsi 없음 → 구버전과 유사
    diagnose_strategy(
        "전략B 구버전 (EMA slope없음, RSI없음)",
        BBTouch(confirm_candle=True, ema_period=50, use_ema_slope=False,
                use_rsi=False, rsi_long_max=100, rsi_short_min=0),
        df_5m, base_engine, "[5m]"
    )

    # ── 전략 B: 개선 (RSI + EMA slope) ──
    diagnose_strategy(
        "전략B 개선 (RSI<40, EMA slope)",
        BBTouch(confirm_candle=True, ema_period=50, ema_slope_period=5,
                use_ema_slope=True, use_rsi=True,
                rsi_long_max=40.0, rsi_short_min=60.0),
        df_5m, base_engine, "[5m]"
    )

    diagnose_strategy(
        "전략B 개선 (RSI<40, EMA slope)",
        BBTouch(confirm_candle=True, ema_period=50, ema_slope_period=5,
                use_ema_slope=True, use_rsi=True,
                rsi_long_max=40.0, rsi_short_min=60.0),
        df_15m, base_engine, "[15m]"
    )

    # ── 전략 B: RSI<50, 확인봉 없음 ──
    diagnose_strategy(
        "전략B (RSI<50, 즉시진입, EMA slope)",
        BBTouch(confirm_candle=False, ema_period=50, ema_slope_period=5,
                use_ema_slope=True, use_rsi=True,
                rsi_long_max=50.0, rsi_short_min=50.0),
        df_5m, base_engine, "[5m]"
    )

    # ── 15m 비교 ──
    diagnose_strategy(
        "전략A 원본",
        BBSqueezeEMA(ema_period=50, squeeze_percentile=20, slope_period=3, use_rsi=False),
        df_15m, base_engine, "[15m]"
    )


if __name__ == "__main__":
    main()
