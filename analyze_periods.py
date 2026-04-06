"""
현재 봇 설정으로 기간/분봉 변경 분석
======================================
고정 파라미터: squeeze_pct=15, trend_ema=20, SL×1.5, TP×3.0
변수: 기간, 타임프레임
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from tabulate import tabulate
from download_data import fetch
from strategies import BBBreakout
from backtester import BacktestEngine

CAPITAL        = 10_000
LEVERAGE       = 10
RISK_PER_TRADE = 0.01

# ── 고정 전략 파라미터 (현재 봇 설정) ────────────────────────
STRATEGY = BBBreakout(squeeze_pct=15, trend_filter=True, trend_ema=20)
SL = 1.5
TP = 3.0

# ── 테스트할 기간 ─────────────────────────────────────────────
PERIODS = [
    ("2019-01", "2020-12", "2019~2020 (상승→코로나 폭락)"),
    ("2020-01", "2021-12", "2020~2021 (코로나 회복+불장)"),
    ("2021-01", "2021-12", "2021 (사상최고가)"),
    ("2022-01", "2022-12", "2022 (곰장)"),
    ("2023-01", "2023-12", "2023 (회복)"),
    ("2024-01", "2024-12", "2024 (불장)"),
    ("2022-01", "2024-12", "2022~2024 (3년)"),
    ("2022-01", "2025-03", "2022~2025Q1 (전체)"),
    ("2023-01", "2025-03", "2023~2025Q1 (최근 2년)"),
    ("2024-01", "2025-03", "2024~2025Q1 (최근)"),
]

# ── 테스트할 타임프레임 ───────────────────────────────────────
INTERVALS = ["15m", "30m", "1h", "2h", "4h"]

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


def run_one(interval, start, end):
    try:
        df = fetch(interval, start, end)
        yr = int(start.split("-")[0])
        df_1d = fetch("1d", f"{yr-1}-01", end)
    except Exception as e:
        return None

    signals = STRATEGY.generate_signals(df)
    signals = apply_regime(signals, df_1d, df)

    n_sig = (signals != 0).sum()
    if n_sig == 0:
        return {"trades": 0, "ret": 0, "win_rate": 0, "pf": 0, "mdd": 0, "final": CAPITAL}

    engine = BacktestEngine(
        initial_capital=CAPITAL, leverage=LEVERAGE,
        risk_per_trade=RISK_PER_TRADE,
        sl_atr_mult=SL, tp_atr_mult=TP, max_hold_bars=100,
    )
    result = engine.run(df, signals)
    trades = result["trades"]
    if not trades:
        return {"trades": 0, "ret": 0, "win_rate": 0, "pf": 0, "mdd": 0, "final": CAPITAL}

    wins  = [t for t in trades if t.pnl > 0]
    loses = [t for t in trades if t.pnl <= 0]
    pf    = (sum(t.pnl for t in wins) / abs(sum(t.pnl for t in loses))) if loses else 999
    ret   = (result["final_capital"] - CAPITAL) / CAPITAL * 100

    equity   = result["equity_curve"]["equity"]
    roll_max = equity.cummax()
    mdd      = ((equity - roll_max) / roll_max).min() * 100

    return {
        "trades":   len(trades),
        "ret":      round(ret, 1),
        "win_rate": round(len(wins) / len(trades) * 100, 1),
        "pf":       round(pf, 2),
        "mdd":      round(mdd, 1),
        "final":    round(result["final_capital"], 0),
    }


def main():
    print(f"현재 봇 설정 분석: sq15 / ema20 / SL×{SL} / TP×{TP} / 10x\n")

    # ── 1. 기간별 분석 (1h 고정) ─────────────────────────────
    print("=" * 85)
    print("  [1] 타임프레임 1h 고정 — 기간별 성과")
    print("=" * 85)
    rows = []
    for start, end, label in PERIODS:
        r = run_one("1h", start, end)
        if not r:
            continue
        flag = "★" if r["ret"] > 0 else " "
        rows.append([
            f"{flag} {label}",
            f"{r['ret']:+.1f}%",
            f"{r['win_rate']}%",
            f"{r['pf']}",
            f"{r['mdd']:.1f}%",
            r["trades"],
            f"${r['final']:,.0f}",
        ])
        print(f"  {label} ... {r['ret']:+.1f}%")

    print()
    print(tabulate(rows,
        headers=["기간", "수익률", "승률", "PF", "MDD", "거래수", "최종자본"],
        tablefmt="simple"))

    # ── 2. 타임프레임별 분석 (2022~2025 고정) ────────────────
    print("\n" + "=" * 85)
    print("  [2] 기간 2022~2025Q1 고정 — 타임프레임별 성과")
    print("=" * 85)
    rows2 = []
    for iv in INTERVALS:
        print(f"  {iv} ...", end=" ", flush=True)
        r = run_one(iv, "2022-01", "2025-03")
        if not r:
            print("실패")
            continue
        print(f"{r['ret']:+.1f}%")
        flag = "★" if r["ret"] > 0 else " "
        rows2.append([
            f"{flag} {iv}",
            f"{r['ret']:+.1f}%",
            f"{r['win_rate']}%",
            f"{r['pf']}",
            f"{r['mdd']:.1f}%",
            r["trades"],
            f"${r['final']:,.0f}",
        ])

    print()
    print(tabulate(rows2,
        headers=["봉", "수익률", "승률", "PF", "MDD", "거래수", "최종자본"],
        tablefmt="simple"))

    # ── 3. 타임프레임 × 기간 매트릭스 ───────────────────────
    print("\n" + "=" * 85)
    print("  [3] 수익률 매트릭스 (행=기간, 열=타임프레임)")
    print("=" * 85)
    short_labels = [label.split(" ")[0] for _, _, label in PERIODS]
    header = ["기간"] + INTERVALS
    matrix = []
    for (start, end, label), sl in zip(PERIODS, short_labels):
        row = [sl]
        for iv in INTERVALS:
            r = run_one(iv, start, end)
            if not r:
                row.append("—")
            else:
                sign = "+" if r["ret"] >= 0 else ""
                row.append(f"{sign}{r['ret']:.1f}%")
        matrix.append(row)

    print(tabulate(matrix, headers=header, tablefmt="simple"))


if __name__ == "__main__":
    main()
