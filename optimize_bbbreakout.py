"""
BBBreakout 전략 파라미터 그리드 서치
=====================================
전략 조건(BB 스퀴즈 돌파 + EMA 추세 + 200일 MA 국면 필터)은 고정.
다양한 파라미터 조합으로 백테스트해 최적 조합 탐색.

사용:
  python optimize_bbbreakout.py
"""
import itertools
import multiprocessing as mp
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from tabulate import tabulate
from tqdm import tqdm

from strategies import BBBreakout
from backtester import BacktestEngine
from download_data import fetch

# ── 고정값 ────────────────────────────────────────────────────
CAPITAL        = 10_000
LEVERAGE       = 10
RISK_PER_TRADE = 0.01
START          = "2022-01"
END            = "2025-03"

# ── 탐색 그리드 ───────────────────────────────────────────────
GRID = {
    "squeeze_pct":      [5, 10, 15, 20, 25, 30],
    "trend_ema":        [10, 20, 50],
    "squeeze_lookback": [50, 100, 200],
    "sl_atr_mult":      [1.0, 1.5, 2.0],
    "tp_atr_mult":      [2.0, 3.0, 4.0, 5.0],
    "max_hold_bars":    [50, 100, 200],
}
# 총 조합수: 6×3×3×3×4×3 = 1944


# ── 200일 MA 국면 필터 ────────────────────────────────────────
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


# ── 단일 백테스트 (worker) ────────────────────────────────────
def _worker(args):
    df_1h_vals, df_1h_idx, df_1d_vals, df_1d_idx, params = args

    df_1h = pd.DataFrame(df_1h_vals, index=df_1h_idx, columns=["open","high","low","close","volume"])
    df_1d = pd.DataFrame(df_1d_vals, index=df_1d_idx, columns=["open","high","low","close","volume"])

    sp = params["squeeze_pct"]
    te = params["trend_ema"]
    sl = params["sl_atr_mult"]
    tp = params["tp_atr_mult"]
    lb = params["squeeze_lookback"]
    mh = params["max_hold_bars"]

    strategy = BBBreakout(
        squeeze_pct=sp, trend_filter=True, trend_ema=te, squeeze_lookback=lb
    )
    signals = strategy.generate_signals(df_1h)
    signals = apply_regime(signals, df_1d, df_1h)

    n_signals = (signals != 0).sum()
    if n_signals == 0:
        return None

    engine = BacktestEngine(
        initial_capital=CAPITAL, leverage=LEVERAGE,
        risk_per_trade=RISK_PER_TRADE,
        sl_atr_mult=sl, tp_atr_mult=tp, max_hold_bars=mh,
    )
    result = engine.run(df_1h, signals)
    trades = result["trades"]
    if not trades:
        return None

    wins   = [t for t in trades if t.pnl > 0]
    loses  = [t for t in trades if t.pnl <= 0]
    pf     = (sum(t.pnl for t in wins) / abs(sum(t.pnl for t in loses))) if loses else 999
    ret    = (result["final_capital"] - CAPITAL) / CAPITAL * 100

    # 연도별 수익률
    yearly = {}
    for yr in [2022, 2023, 2024, 2025]:
        yr_trades = [t for t in trades if t.entry_time.year == yr]
        yearly[yr] = round(sum(t.pnl for t in yr_trades) / CAPITAL * 100, 1) if yr_trades else 0.0

    # MDD
    equity = result["equity_curve"]["equity"]
    roll_max = equity.cummax()
    mdd = ((equity - roll_max) / roll_max).min() * 100

    return {
        "squeeze_pct": sp, "trend_ema": te, "squeeze_lookback": lb,
        "sl": sl, "tp": tp, "max_hold": mh,
        "trades": len(trades),
        "win_rate": round(len(wins) / len(trades) * 100, 1),
        "pf": round(pf, 2),
        "ret": round(ret, 1),
        "mdd": round(mdd, 1),
        "final": round(result["final_capital"], 0),
        "y2022": yearly[2022],
        "y2023": yearly[2023],
        "y2024": yearly[2024],
        "y2025": yearly[2025],
    }


def main():
    print(f"BBBreakout 그리드 서치 | {START}~{END} | 레버리지 {LEVERAGE}x")
    print("데이터 로딩 중...")

    df_1h = fetch("1h", START, END)
    df_1d = fetch("1d", f"{int(START.split('-')[0])-1}-01", END)

    # numpy 배열로 변환 (multiprocessing pickle 최적화)
    h_vals = df_1h.values; h_idx = df_1h.index
    d_vals = df_1d.values; d_idx = df_1d.index

    combos = [dict(zip(GRID.keys(), v)) for v in itertools.product(*GRID.values())]
    print(f"총 조합: {len(combos)}개 | CPU: {mp.cpu_count()}코어\n")

    tasks = [(h_vals, h_idx, d_vals, d_idx, p) for p in combos]

    workers = max(1, mp.cpu_count() - 1)
    results = []
    with mp.Pool(processes=workers) as pool:
        for r in tqdm(pool.imap_unordered(_worker, tasks), total=len(tasks), ncols=80):
            if r:
                results.append(r)

    if not results:
        print("수익 조합 없음")
        return

    results.sort(key=lambda x: x["ret"], reverse=True)
    pos = [r for r in results if r["ret"] > 0]
    print(f"\n총 {len(results)}개 유효 조합 | 수익: {len(pos)}개 | 손실: {len(results)-len(pos)}개\n")

    # 상위 30개 출력
    top = results[:30]
    rows = []
    for i, r in enumerate(top, 1):
        flag = "★" if r["ret"] > 0 else " "
        rows.append([
            f"{flag}{i}",
            f"sq{r['squeeze_pct']}%/lb{r['squeeze_lookback']}/ema{r['trend_ema']}",
            f"SL{r['sl']}/TP{r['tp']}/mh{r['max_hold']}",
            f"{r['ret']:+.1f}%",
            f"{r['win_rate']}%",
            f"{r['pf']}",
            f"{r['mdd']:.1f}%",
            r["trades"],
            f"{r['y2022']:+.1f}%",
            f"{r['y2023']:+.1f}%",
            f"{r['y2024']:+.1f}%",
            f"{r['y2025']:+.1f}%",
            f"${r['final']:,.0f}",
        ])

    headers = ["#", "전략 파라미터", "엔진 파라미터", "총수익", "승률", "PF", "MDD",
               "거래수", "2022", "2023", "2024", "2025Q1", "최종자본"]
    print(tabulate(rows, headers=headers, tablefmt="simple"))

    if pos:
        best = pos[0]
        print(f"\n★ 최적 조합:")
        print(f"   squeeze_pct={best['squeeze_pct']}  squeeze_lookback={best['squeeze_lookback']}  trend_ema={best['trend_ema']}")
        print(f"   SL×{best['sl']}  TP×{best['tp']}  max_hold={best['max_hold']}봉")
        print(f"   수익: {best['ret']:+.1f}%  승률: {best['win_rate']}%  PF: {best['pf']}  MDD: {best['mdd']:.1f}%")
        print(f"   연도별: 2022={best['y2022']:+.1f}%  2023={best['y2023']:+.1f}%  2024={best['y2024']:+.1f}%  2025Q1={best['y2025']:+.1f}%")


if __name__ == "__main__":
    mp.freeze_support()
    main()
