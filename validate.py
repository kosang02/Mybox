"""
최적 전략 심층 검증
===================
- 연도별 / 월별 성과 분석
- 하락장(2022) vs 상승장(2024) 비교
- BB std 파라미터 무관성 원인 분석
- 거래 분포 시각화 (텍스트)
- 연속 손실 최대치, 회복 기간 분석

사용:
  python validate.py
"""

import pandas as pd
import numpy as np
from tabulate import tabulate
from download_data import fetch
from strategies import BBSqueezeEMA, BBTouch
from backtester import BacktestEngine, calc_metrics
from optimize import apply_trend_filter, build_trend_filter


# ── 최적 조합 (optimize.py 결과 기반) ──
BEST = dict(
    bb_period=20, bb_std=2.0, ema_period=50,
    squeeze_percentile=20, slope_period=3,   # ← 3이 핵심 (2는 과매매, 5는 중간)
)
ENGINE_PARAMS = dict(
    initial_capital=10_000,
    leverage=10,
    risk_per_trade=0.01,
    sl_atr_mult=1.0,
    tp_atr_mult=4.0,
    max_hold_bars=200,
)

PERIODS = {
    "2022 (하락장)": ("2022-01", "2022-12"),
    "2023":          ("2023-01", "2023-12"),
    "2024 (상승장)": ("2024-01", "2024-12"),
    "2023-2024 전체":("2023-01", "2024-12"),
}


def run_period(label, start, end):
    try:
        df = fetch("5m", start, end)
    except Exception as e:
        return None, f"데이터 없음: {e}"

    strategy = BBSqueezeEMA(**BEST)
    engine   = BacktestEngine(**ENGINE_PARAMS)
    signals  = strategy.generate_signals(df)
    result   = engine.run(df, signals)
    trades   = result["trades"]
    equity   = result["equity_curve"]
    metrics  = calc_metrics(trades, equity, ENGINE_PARAMS["initial_capital"],
                            str(strategy), "5m")
    return trades, metrics


def monthly_breakdown(trades: list, start_year: int, end_year: int) -> pd.DataFrame:
    """월별 손익 집계."""
    if not trades:
        return pd.DataFrame()

    rows = []
    for t in trades:
        if t.pnl is not None and t.exit_time is not None:
            rows.append({
                "year":  t.exit_time.year,
                "month": t.exit_time.month,
                "pnl":   t.pnl,
                "win":   1 if t.pnl > 0 else 0,
            })
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    monthly = df.groupby(["year", "month"]).agg(
        pnl=("pnl", "sum"),
        trades=("pnl", "count"),
        wins=("win", "sum"),
    ).reset_index()
    monthly["win_rate"] = monthly["wins"] / monthly["trades"] * 100
    return monthly


def max_consecutive_loss(trades: list) -> tuple[int, float]:
    """최대 연속 손실 횟수, 최대 연속 손실 금액."""
    max_streak = 0
    max_loss   = 0
    streak = 0
    streak_loss = 0
    for t in trades:
        if t.pnl is not None and t.pnl <= 0:
            streak      += 1
            streak_loss += t.pnl
            if streak > max_streak:
                max_streak = streak
            if streak_loss < max_loss:
                max_loss = streak_loss
        else:
            streak = 0
            streak_loss = 0
    return max_streak, max_loss


def bb_std_analysis(df):
    """BB std 파라미터가 신호에 영향이 없는 이유 분석."""
    print("\n[BB std 파라미터 무관성 분석]")
    print("BB width = (upper - lower) / middle = 2 × bb_std × rolling_std / middle")
    print("squeeze_threshold = bb_width.rolling(100).quantile(p)")
    print("→ bb_width와 threshold 모두 bb_std에 비례 → 비율 비교 시 bb_std 약분됨")
    print("→ 결론: bb_std는 이 전략에서 실질적으로 무관한 파라미터 (제거 가능)\n")

    # 실제로 확인
    for std_val in [1.5, 2.0, 2.5]:
        s = BBSqueezeEMA(bb_std=std_val, ema_period=50, squeeze_percentile=20, slope_period=2)
        sigs = s.generate_signals(df)
        print(f"  bb_std={std_val}: 시그널 수 = {(sigs != 0).sum()} "
              f"(롱 {(sigs == 1).sum()}, 숏 {(sigs == -1).sum()})")


def print_period_comparison(results: list):
    rows = []
    for label, metrics in results:
        if metrics is None:
            continue
        if isinstance(metrics, str):
            rows.append([label, metrics, "-", "-", "-", "-", "-", "-"])
            continue
        sign = "+" if metrics["total_return"] >= 0 else ""
        rows.append([
            label,
            f"{sign}{metrics['total_return']}%",
            f"{metrics['win_rate']}%",
            f"{metrics['mdd']}%",
            f"{metrics['sharpe']:.3f}",
            metrics["total_trades"],
            metrics["profit_factor"],
            f"${metrics['final_capital']:,.0f}",
        ])

    print("\n" + "=" * 90)
    print("  기간별 성과 비교 (BB 스퀴즈+EMA, EMA50, 스퀴즈20%, SL×1.0/TP×4.0, 레버리지10x)")
    print("=" * 90)
    headers = ["기간", "수익률", "승률", "MDD", "샤프", "거래수", "수익팩터", "최종자본"]
    print(tabulate(rows, headers=headers, tablefmt="simple"))
    print("=" * 90)


def print_monthly(monthly: pd.DataFrame, label: str):
    if monthly.empty:
        return
    print(f"\n  [{label}] 월별 손익:")
    rows = []
    for _, r in monthly.iterrows():
        sign = "+" if r.pnl >= 0 else ""
        bar = "█" * int(abs(r.pnl) / 100) if abs(r.pnl) < 5000 else "█" * 50
        color = "▲" if r.pnl >= 0 else "▼"
        rows.append([
            f"{int(r.year)}-{int(r.month):02d}",
            f"{color} {sign}${r.pnl:,.0f}",
            f"{int(r.trades)}건 / {r.win_rate:.0f}%승",
            bar[:30],
        ])
    print(tabulate(rows, headers=["월", "손익", "거래", "규모"], tablefmt="simple"))


def print_trade_dist(trades: list, label: str):
    if not trades:
        return
    max_seq, max_loss = max_consecutive_loss(trades)
    pnls = [t.pnl for t in trades if t.pnl is not None]
    exits = [t.exit_reason for t in trades]

    print(f"\n  [{label}] 거래 분포:")
    print(f"    최대 연속 손실: {max_seq}건 연속 / 누적 ${max_loss:,.2f}")
    print(f"    최대 단일 수익: ${max(pnls):,.2f}  /  최대 단일 손실: ${min(pnls):,.2f}")
    from collections import Counter
    exit_counts = Counter(exits)
    print(f"    청산 유형: {dict(exit_counts)}")


def main():
    print("BTC 5m BB 스퀴즈+EMA 전략 심층 검증")
    print("=" * 60)

    # 기간별 비교
    results = []
    all_trades = {}
    for label, (start, end) in PERIODS.items():
        print(f"  {label} 테스트 중...", end=" ", flush=True)
        trades, metrics = run_period(label, start, end)
        results.append((label, metrics))
        if trades:
            all_trades[label] = trades
        if isinstance(metrics, dict):
            print(f"완료 ({metrics['total_trades']}건)")
        else:
            print(f"실패 ({metrics})")

    print_period_comparison(results)

    # 월별 분석
    for label in ["2022 (하락장)", "2023-2024 전체"]:
        if label in all_trades:
            monthly = monthly_breakdown(all_trades[label], 2022, 2024)
            print_monthly(monthly, label)

    # 거래 분포
    for label, trades in all_trades.items():
        print_trade_dist(trades, label)

    # BB std 무관성 분석
    df_sample = fetch("5m", "2023-01", "2023-03")
    bb_std_analysis(df_sample)

    # 핵심 결론
    print("\n" + "=" * 60)
    print("  검증 요약")
    print("=" * 60)
    for label, metrics in results:
        if isinstance(metrics, dict) and metrics["total_trades"] > 0:
            verdict = "✓ 수익" if metrics["total_return"] > 0 else "✗ 손실"
            print(f"  {verdict}  {label}: {'+' if metrics['total_return']>0 else ''}"
                  f"{metrics['total_return']}% / {metrics['total_trades']}거래")


if __name__ == "__main__":
    main()
