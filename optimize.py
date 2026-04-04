"""
파라미터 그리드 서치 최적화
============================
전략 A (BB 스퀴즈+EMA) / 전략 B (BB 터치+RSI)의
모든 파라미터 조합을 자동 탐색, 수익 나는 조합을 랭킹으로 출력.

최적화 기법:
- 신호를 전략 파라미터별로 한 번만 생성 후 엔진 파라미터에 재사용
- multiprocessing 으로 CPU 코어 수만큼 병렬 실행
- max_hold_bars 제한으로 탐색 범위를 스캘핑 범위로 한정

사용:
  python optimize.py
  python optimize.py --interval 5m 15m --top 30
  python optimize.py --interval 15m 1h --leverage 10
  python optimize.py --interval 5m --no-trend-filter
"""

import argparse
import itertools
import multiprocessing as mp
import numpy as np
import pandas as pd
from tabulate import tabulate
from tqdm import tqdm

from download_data import fetch
from strategies import BBSqueezeEMA, BBTouch
from backtester import BacktestEngine, calc_metrics


# ──────────────────────────────────────────────
# 파라미터 그리드 (bb_std 제거 - 수학적으로 무의미)
# ──────────────────────────────────────────────
GRID_A = {
    "bb_period":          [20],
    "ema_period":         [10, 20, 50],
    "squeeze_percentile": [10, 20, 30],
    "slope_period":       [2, 3, 5],
    "use_rsi":            [False, True],   # RSI 모멘텀 필터 ON/OFF
    "rsi_long_min":       [50.0],
    "rsi_short_max":      [50.0],
}

GRID_B = {
    "bb_period":        [20],
    "confirm_candle":   [True, False],
    "ema_period":       [20, 50],
    "ema_slope_period": [3, 5],
    "use_rsi":          [True],
    "rsi_long_max":     [40.0, 50.0],   # 롱: RSI 이하일 때 (낮을수록 엄격)
    "rsi_short_min":    [50.0, 60.0],   # 숏: RSI 이상일 때 (높을수록 엄격)
    "use_ema_slope":    [True, False],
}

GRID_ENGINE = {
    "sl_atr_mult":   [1.0, 1.5, 2.0],
    "tp_atr_mult":   [2.0, 3.0, 4.0],
    "max_hold_bars": [100, 200, 500],
}

TREND_TF_MAP  = {"1m": "15m", "3m": "15m", "5m": "1h", "15m": "1h", "30m": "4h", "1h": "4h"}
TREND_EMA     = 50


# ──────────────────────────────────────────────
# 추세 필터
# ──────────────────────────────────────────────
def build_trend_filter(df_target: pd.DataFrame, trend_tf: str, start: str, end: str) -> pd.Series:
    df_trend = fetch(trend_tf, start, end)
    ema   = df_trend["close"].ewm(span=TREND_EMA, adjust=False).mean()
    slope = ema - ema.shift(3)
    trend = pd.Series(0, index=df_trend.index, dtype=int)
    trend[slope > 0] = 1
    trend[slope < 0] = -1
    return trend.reindex(trend.index.union(df_target.index)).ffill().reindex(df_target.index)


def apply_trend_filter(signals: pd.Series, trend: pd.Series) -> pd.Series:
    f = signals.copy()
    f[(signals ==  1) & (trend == -1)] = 0
    f[(signals == -1) & (trend ==  1)] = 0
    return f


# ──────────────────────────────────────────────
# 단일 백테스트 (multiprocessing worker용)
# ──────────────────────────────────────────────
def _worker(task):
    df, sig_vals, ep, capital, leverage, interval, strat_name = task

    signals = pd.Series(sig_vals, index=df.index)

    engine = BacktestEngine(
        initial_capital = capital,
        leverage        = leverage,
        risk_per_trade  = 0.01,
        sl_atr_mult     = ep["sl_atr_mult"],
        tp_atr_mult     = ep["tp_atr_mult"],
        max_hold_bars   = ep["max_hold_bars"],
    )
    result  = engine.run(df, signals)
    metrics = calc_metrics(
        result["trades"], result["equity_curve"],
        capital, strat_name, interval,
    )
    metrics.update({
        "interval":  interval,
        "sl_mult":   ep["sl_atr_mult"],
        "tp_mult":   ep["tp_atr_mult"],
        "max_hold":  ep["max_hold_bars"],
        "leverage":  leverage,
    })
    return metrics


# ──────────────────────────────────────────────
# 그리드 서치 (신호 재사용 + 병렬)
# ──────────────────────────────────────────────
def grid_search(df, trend, interval, capital, leverage,
                use_trend_filter, min_trades, workers):

    combos_A = [dict(zip(GRID_A.keys(), v)) for v in itertools.product(*GRID_A.values())]
    combos_B = [dict(zip(GRID_B.keys(), v)) for v in itertools.product(*GRID_B.values())]
    combos_E = [dict(zip(GRID_ENGINE.keys(), v)) for v in itertools.product(*GRID_ENGINE.values())]

    trend_label = (f"추세O({TREND_TF_MAP.get(interval,'1h')} EMA{TREND_EMA})"
                   if use_trend_filter else "추세X")

    # ── 신호 사전 생성 (전략 파라미터별 1회) ──
    n_total = len(combos_A) + len(combos_B)
    print(f"  [{interval}/{trend_label}] 신호 생성 중 ({n_total}개 전략)...", end=" ", flush=True)
    strategy_signals = []  # [(strat_name, sig_vals)]

    for sp in combos_A:
        strat = BBSqueezeEMA(**sp)
        sigs  = strat.generate_signals(df)
        if use_trend_filter and trend is not None:
            sigs = apply_trend_filter(sigs, trend)
        if (sigs != 0).sum() > 0:
            strategy_signals.append((str(strat), sigs.values.copy()))

    for sp in combos_B:
        strat = BBTouch(**sp)
        sigs  = strat.generate_signals(df)
        if use_trend_filter and trend is not None:
            sigs = apply_trend_filter(sigs, trend)
        if (sigs != 0).sum() > 0:
            strategy_signals.append((str(strat), sigs.values.copy()))

    print(f"완료 ({len(strategy_signals)}개 유효)")

    # ── 엔진 파라미터 × 전략 신호 → 태스크 목록 ──
    tasks = []
    for strat_name, sig_vals in strategy_signals:
        for ep in combos_E:
            tasks.append((df, sig_vals, ep, capital, leverage, interval, strat_name))

    # ── 병렬 실행 ──
    results = []
    desc = f"  [{interval}/{trend_label}]"
    with mp.Pool(processes=workers) as pool:
        for m in tqdm(pool.imap_unordered(_worker, tasks), total=len(tasks),
                      desc=desc, ncols=90):
            if m and m.get("total_trades", 0) >= min_trades:
                m["trend_filter"] = trend_label
                results.append(m)

    return results


# ──────────────────────────────────────────────
# 결과 출력
# ──────────────────────────────────────────────
def print_results(all_results, top_n, initial_capital):
    if not all_results:
        print("\n수익 나는 조합이 없습니다.")
        return

    all_results.sort(key=lambda x: x["total_return"], reverse=True)
    positive = [r for r in all_results if r["total_return"] > 0]
    top      = all_results[:top_n]

    print(f"\n총 {len(all_results)}개 조합 | ★수익: {len(positive)}개 | ✗손실: {len(all_results)-len(positive)}개\n")
    print("=" * 150)
    print(f"  상위 {min(top_n, len(top))}개 결과 (수익률 기준 정렬)")
    print("=" * 150)

    rows = []
    for i, r in enumerate(top, 1):
        sign = "+" if r["total_return"] >= 0 else ""
        flag = "★" if r["total_return"] > 0 else "  "
        rows.append([
            f"{flag}{i}",
            r["strategy"][:55],
            r["interval"],
            r["trend_filter"],
            f"{r['sl_mult']}/{r['tp_mult']}",
            r["max_hold"],
            f"{sign}{r['total_return']}%",
            f"{r['sharpe']:.3f}",
            f"{r['win_rate']}%",
            f"{r['mdd']}%",
            r["total_trades"],
            r["liq_count"],
            r["profit_factor"],
            f"${r['final_capital']:,.0f}",
        ])

    headers = ["#", "전략", "봉", "추세필터", "SL/TP×", "최대보유봉",
               "수익률", "샤프", "승률", "MDD", "거래수", "청산수", "수익팩터", "최종자본"]
    print(tabulate(rows, headers=headers, tablefmt="simple"))
    print("=" * 150)

    if positive:
        best = positive[0]
        print(f"\n★  최적 조합 발견!")
        print(f"   전략      : {best['strategy']}")
        print(f"   타임프레임 : {best['interval']} / {best['trend_filter']}")
        print(f"   SL×{best['sl_mult']} / TP×{best['tp_mult']} / 최대{best['max_hold']}봉 / 레버리지 {best['leverage']}x")
        print(f"   수익률    : {'+' if best['total_return']>0 else ''}{best['total_return']}%")
        print(f"   승률      : {best['win_rate']}%  |  샤프: {best['sharpe']}  |  MDD: {best['mdd']}%")
        print(f"   거래수    : {best['total_trades']}  |  청산수: {best['liq_count']}")
    else:
        print("\n⚠  수익 나는 조합 없음.")
        print(f"   최고 수익률: {all_results[0]['total_return']}% ({all_results[0]['strategy'][:40]})")
        print("   → 전략 개선이 필요합니다.")


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--interval",        nargs="+", default=["5m", "15m"])
    p.add_argument("--start",           default="2023-01")
    p.add_argument("--end",             default="2024-12")
    p.add_argument("--leverage",        type=int,   default=10)
    p.add_argument("--capital",         type=float, default=10_000.0)
    p.add_argument("--top",             type=int,   default=30)
    p.add_argument("--min-trades",      type=int,   default=30)
    p.add_argument("--workers",         type=int,   default=max(1, mp.cpu_count() - 1),
                   help="병렬 프로세스 수")
    p.add_argument("--no-trend-filter", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"\nBTC 선물 파라미터 최적화 v2")
    print(f"기간: {args.start}~{args.end} | 레버리지: {args.leverage}x | "
          f"자본: ${args.capital:,.0f} | 워커: {args.workers}개")

    combos_A = list(itertools.product(*GRID_A.values()))
    combos_B = list(itertools.product(*GRID_B.values()))
    combos_E = list(itertools.product(*GRID_ENGINE.values()))
    n_strat  = len(combos_A) + len(combos_B)
    print(f"전략 A: {len(combos_A)}개, 전략 B: {len(combos_B)}개 = 총 {n_strat}개")
    print(f"엔진 조합: {len(combos_E)}개 → 최대 {n_strat * len(combos_E)}개 백테스트")

    all_results = []

    for interval in args.interval:
        print(f"\n[{interval}] 데이터 로딩...")
        df = fetch(interval, args.start, args.end)

        trend    = None
        trend_tf = TREND_TF_MAP.get(interval)
        if not args.no_trend_filter and trend_tf:
            try:
                trend = build_trend_filter(df, trend_tf, args.start, args.end)
                up = (trend == 1).sum(); dn = (trend == -1).sum()
                print(f"  추세 필터 ({trend_tf} EMA{TREND_EMA}): 상승 {up:,}봉 / 하락 {dn:,}봉")
            except Exception as e:
                print(f"  추세 필터 실패 ({e})")

        # 추세 필터 ON
        r1 = grid_search(df, trend, interval, args.capital, args.leverage,
                         use_trend_filter=True,  min_trades=args.min_trades,
                         workers=args.workers)
        all_results.extend(r1)

        # 추세 필터 OFF (비교용)
        if args.no_trend_filter:
            r2 = grid_search(df, None, interval, args.capital, args.leverage,
                             use_trend_filter=False, min_trades=args.min_trades,
                             workers=args.workers)
            all_results.extend(r2)

    print_results(all_results, args.top, args.capital)


if __name__ == "__main__":
    mp.freeze_support()
    main()
