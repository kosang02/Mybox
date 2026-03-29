#!/usr/bin/env python3
"""
Bitcoin Backtesting Bot
=======================
사용법:
  python main.py
  python main.py --strategy sma --start 2024-01-01 --end 2024-12-31
  python main.py --strategy rsi --interval 4h --capital 5000
  python main.py --strategy bb  --plot

  # 파라미터 자동 최적화 (전체 전략)
  python main.py --optimize
  python main.py --optimize --strategy sma
  python main.py --optimize --sort sharpe
"""

import argparse
import itertools
from datetime import date, timedelta
from tabulate import tabulate

from backtester import DataFetcher, BacktestEngine, PerformanceMetrics
from backtester import SMACrossover, RSIStrategy, BollingerBands


# ──────────────────────────────────────────────
# 최적화용 파라미터 그리드
# ──────────────────────────────────────────────
OPTIMIZE_GRID = {
    "sma": {
        "short": [5, 10, 20, 50],
        "long":  [20, 50, 100, 200],
    },
    "rsi": {
        "period":     [7, 14, 21],
        "oversold":   [20, 30],
        "overbought": [70, 80],
    },
    "bb": {
        "period":  [10, 20, 30],
        "std_dev": [1.5, 2.0, 2.5],
    },
}

SORT_KEYS = {
    "return":  ("total_return_pct",   True),
    "sharpe":  ("sharpe_ratio",        True),
    "winrate": ("win_rate_pct",        True),
    "mdd":     ("max_drawdown_pct",    False),  # 낮을수록 좋음 → ascending
}

STRATEGIES = {
    "sma": lambda args: SMACrossover(short=args.sma_short, long=args.sma_long),
    "rsi": lambda args: RSIStrategy(period=args.rsi_period,
                                    oversold=args.rsi_oversold,
                                    overbought=args.rsi_overbought),
    "bb":  lambda args: BollingerBands(period=args.bb_period, std_dev=args.bb_std),
}


def parse_args():
    today = date.today()
    default_end   = today.strftime("%Y-%m-%d")
    default_start = (today - timedelta(days=365)).strftime("%Y-%m-%d")

    parser = argparse.ArgumentParser(
        description="Bitcoin Backtesting Bot",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--symbol",   default="BTCUSDT", help="거래 심볼")
    parser.add_argument("--start",    default=default_start, help="백테스트 시작일 (YYYY-MM-DD)")
    parser.add_argument("--end",      default=default_end,   help="백테스트 종료일 (YYYY-MM-DD)")
    parser.add_argument("--interval", default="1d", help="캔들 간격 (1m/1h/4h/1d 등)")
    parser.add_argument("--capital",  type=float, default=10_000.0, help="초기 자본 (USDT)")
    parser.add_argument("--fee",      type=float, default=0.001,    help="거래 수수료 비율")
    parser.add_argument("--strategy", default="sma",
                        choices=list(STRATEGIES.keys()),
                        help="사용할 전략 (sma / rsi / bb)")
    parser.add_argument("--plot",     action="store_true", help="차트 저장 (chart.png)")

    # 최적화 모드
    opt = parser.add_argument_group("최적화 옵션")
    opt.add_argument("--optimize",    action="store_true",
                     help="파라미터 그리드 서치로 최적 조합 탐색")
    opt.add_argument("--optimize-all",action="store_true",
                     help="모든 전략을 한번에 최적화해서 비교")
    opt.add_argument("--sort",        default="return",
                     choices=list(SORT_KEYS.keys()),
                     help="최적화 결과 정렬 기준 (return/sharpe/winrate/mdd)")
    opt.add_argument("--top",         type=int, default=10,
                     help="최적화 결과 상위 N개 출력")

    # SMA 파라미터
    sma = parser.add_argument_group("SMA Crossover 파라미터")
    sma.add_argument("--sma-short", type=int,   default=20)
    sma.add_argument("--sma-long",  type=int,   default=50)

    # RSI 파라미터
    rsi = parser.add_argument_group("RSI 파라미터")
    rsi.add_argument("--rsi-period",     type=int,   default=14)
    rsi.add_argument("--rsi-oversold",   type=float, default=30.0)
    rsi.add_argument("--rsi-overbought", type=float, default=70.0)

    # Bollinger Bands 파라미터
    bb = parser.add_argument_group("Bollinger Bands 파라미터")
    bb.add_argument("--bb-period", type=int,   default=20)
    bb.add_argument("--bb-std",    type=float, default=2.0)

    return parser.parse_args()


# ──────────────────────────────────────────────
# 단일 백테스트
# ──────────────────────────────────────────────
def run_single(df, strategy, capital, fee):
    engine = BacktestEngine(initial_capital=capital, fee_rate=fee)
    result = engine.run(df, strategy)
    return result, PerformanceMetrics(result).summary()


# ──────────────────────────────────────────────
# 그리드 서치
# ──────────────────────────────────────────────
def build_strategies(strategy_name: str) -> list:
    """전략 이름에 해당하는 모든 파라미터 조합의 전략 객체를 반환."""
    grid = OPTIMIZE_GRID[strategy_name]
    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))

    strategies = []
    for combo in combos:
        params = dict(zip(keys, combo))
        if strategy_name == "sma":
            if params["short"] >= params["long"]:
                continue
            strategies.append(SMACrossover(**params))
        elif strategy_name == "rsi":
            strategies.append(RSIStrategy(**params))
        elif strategy_name == "bb":
            strategies.append(BollingerBands(**params))
    return strategies


def run_optimize(df, strategy_names: list, capital: float, fee: float,
                 sort_by: str, top_n: int):
    engine = BacktestEngine(initial_capital=capital, fee_rate=fee)
    sort_key, descending = SORT_KEYS[sort_by]

    all_results = []
    total = sum(len(build_strategies(s)) for s in strategy_names)
    done = 0

    for name in strategy_names:
        strategies = build_strategies(name)
        for strategy in strategies:
            done += 1
            print(f"\r  진행 중... {done}/{total} ({strategy})" + " " * 20, end="", flush=True)
            result = engine.run(df, strategy)
            metrics = PerformanceMetrics(result).summary()
            if "error" not in metrics:
                all_results.append(metrics)

    print(f"\r  완료: {done}개 조합 테스트" + " " * 40)

    # 정렬
    all_results.sort(key=lambda x: x[sort_key], reverse=descending)

    print_optimize_results(all_results[:top_n], sort_by)


def print_optimize_results(results: list, sort_by: str):
    if not results:
        print("\n결과 없음 — 시그널이 발생한 조합이 없습니다.")
        return

    sort_label = {"return": "수익률", "sharpe": "샤프비율",
                  "winrate": "승률", "mdd": "MDD"}[sort_by]

    print(f"\n{'=' * 90}")
    print(f"  최적화 결과 (정렬 기준: {sort_label})")
    print(f"{'=' * 90}")

    rows = []
    for i, m in enumerate(results, 1):
        rows.append([
            i,
            m["strategy"],
            f"{m['total_return_pct']:+.2f}%",
            f"{m['sharpe_ratio']:.3f}",
            f"{m['win_rate_pct']:.1f}%",
            f"{m['max_drawdown_pct']:.2f}%",
            m["total_trades"],
            f"{m['profit_factor']:.3f}",
            f"${m['final_capital']:,.0f}",
        ])

    headers = ["순위", "전략", "수익률", "샤프비율", "승률", "MDD", "거래수", "수익팩터", "최종자본"]
    print(tabulate(rows, headers=headers, tablefmt="simple"))
    print(f"{'=' * 90}")


# ──────────────────────────────────────────────
# 단일 실행 출력
# ──────────────────────────────────────────────
def print_results(metrics: dict):
    if "error" in metrics:
        print(f"\n[오류] {metrics['error']}")
        return

    print("\n" + "=" * 60)
    print(f"  백테스트 결과: {metrics['strategy']}")
    print("=" * 60)
    rows = [
        ["초기 자본",       f"${metrics['initial_capital']:,.2f}"],
        ["최종 자본",       f"${metrics['final_capital']:,.2f}"],
        ["총 수익률",       f"{metrics['total_return_pct']:+.2f}%"],
        ["최대 낙폭 (MDD)", f"{metrics['max_drawdown_pct']:.2f}%"],
        ["샤프 비율",       f"{metrics['sharpe_ratio']:.3f}"],
        ["─" * 20,          "─" * 20],
        ["총 거래 수",      metrics["total_trades"]],
        ["승리 거래",       metrics["winning_trades"]],
        ["패배 거래",       metrics["losing_trades"]],
        ["승률",            f"{metrics['win_rate_pct']:.1f}%"],
        ["평균 수익 (승)",  f"${metrics['avg_win_usdt']:,.2f}"],
        ["평균 손실 (패)",  f"${metrics['avg_loss_usdt']:,.2f}"],
        ["수익 팩터",       f"{metrics['profit_factor']:.3f}"],
    ]
    print(tabulate(rows, tablefmt="simple"))
    print("=" * 60)


def print_trade_log(trades, max_show: int = 10):
    if not trades:
        return
    print(f"\n거래 내역 (최근 {min(max_show, len(trades))}건):")
    rows = []
    for t in trades[-max_show:]:
        pnl_str = f"${t.pnl:+,.2f} ({t.pnl_pct:+.2f}%)" if t.pnl is not None else "-"
        rows.append([
            str(t.entry_time)[:10],
            f"${t.entry_price:,.2f}",
            str(t.exit_time)[:10] if t.exit_time else "-",
            f"${t.exit_price:,.2f}" if t.exit_price else "-",
            pnl_str,
        ])
    headers = ["매수일", "매수가", "매도일", "매도가", "손익"]
    print(tabulate(rows, headers=headers, tablefmt="simple"))


def plot_results(result: dict, filename: str = "chart.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        print("[경고] matplotlib이 없어 차트를 저장할 수 없습니다.")
        return

    df     = result["price_data"]
    equity = result["equity_curve"]
    trades = result["trades"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1]})
    fig.suptitle(f"Bitcoin Backtest — {result['strategy']}", fontsize=14)

    ax1.plot(df.index, df["close"], color="#888", linewidth=1)
    ax1.set_ylabel("Price (USDT)")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    for trade in trades:
        ax1.axvline(trade.entry_time, color="green", alpha=0.4, linewidth=0.8)
        if trade.exit_time:
            ax1.axvline(trade.exit_time, color="red", alpha=0.4, linewidth=0.8)

    ax2.plot(equity.index, equity["equity"], color="#2196F3", linewidth=1.5)
    ax2.fill_between(equity.index, equity["equity"].min(), equity["equity"],
                     alpha=0.1, color="#2196F3")
    ax2.set_ylabel("Equity (USDT)")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"\n차트 저장 완료: {filename}")


# ──────────────────────────────────────────────
# 진입점
# ──────────────────────────────────────────────
def main():
    args = parse_args()

    print(f"\nBitcoin Backtesting Bot")
    print(f"기간: {args.start} ~ {args.end} | 간격: {args.interval}")

    # 데이터 한 번만 수집
    fetcher = DataFetcher(symbol=args.symbol)
    df = fetcher.fetch(start=args.start, end=args.end, interval=args.interval)

    # ── 최적화 모드 ──
    if args.optimize_all:
        print(f"\n[전략 전체 최적화] 정렬: {args.sort} | 상위 {args.top}개")
        run_optimize(df, list(OPTIMIZE_GRID.keys()),
                     args.capital, args.fee, args.sort, args.top)
        return

    if args.optimize:
        print(f"\n[{args.strategy.upper()} 최적화] 정렬: {args.sort} | 상위 {args.top}개")
        run_optimize(df, [args.strategy],
                     args.capital, args.fee, args.sort, args.top)
        return

    # ── 단일 실행 모드 ──
    strategy = STRATEGIES[args.strategy](args)
    print(f"전략: {strategy}")

    result, metrics = run_single(df, strategy, args.capital, args.fee)
    print_results(metrics)
    print_trade_log(result["trades"])

    if args.plot:
        plot_results(result)


if __name__ == "__main__":
    main()
