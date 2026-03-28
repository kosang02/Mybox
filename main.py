#!/usr/bin/env python3
"""
Bitcoin Backtesting Bot
=======================
사용법:
  python main.py
  python main.py --strategy sma --start 2023-01-01 --end 2023-12-31
  python main.py --strategy rsi --interval 4h --capital 5000
  python main.py --strategy bb  --start 2022-01-01 --end 2024-01-01 --plot
"""

import argparse
import sys
from tabulate import tabulate

from backtester import DataFetcher, BacktestEngine, PerformanceMetrics
from backtester import SMACrossover, RSIStrategy, BollingerBands


STRATEGIES = {
    "sma": lambda args: SMACrossover(short=args.sma_short, long=args.sma_long),
    "rsi": lambda args: RSIStrategy(period=args.rsi_period,
                                    oversold=args.rsi_oversold,
                                    overbought=args.rsi_overbought),
    "bb":  lambda args: BollingerBands(period=args.bb_period, std_dev=args.bb_std),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Bitcoin Backtesting Bot",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    from datetime import date, timedelta
    today = date.today()
    default_end = today.strftime("%Y-%m-%d")
    default_start = (today - timedelta(days=365)).strftime("%Y-%m-%d")

    parser.add_argument("--symbol",   default="BTCUSDT", help="거래 심볼")
    parser.add_argument("--start",    default=default_start, help="백테스트 시작일 (YYYY-MM-DD)")
    parser.add_argument("--end",      default=default_end, help="백테스트 종료일 (YYYY-MM-DD)")
    parser.add_argument("--interval", default="1d", help="캔들 간격 (1m/1h/4h/1d 등)")
    parser.add_argument("--capital",  type=float, default=10_000.0, help="초기 자본 (USDT)")
    parser.add_argument("--fee",      type=float, default=0.001, help="거래 수수료 비율")
    parser.add_argument("--strategy", default="sma",
                        choices=list(STRATEGIES.keys()),
                        help="사용할 전략 (sma / rsi / bb)")
    parser.add_argument("--plot",     action="store_true", help="차트 저장 (chart.png)")

    # SMA 파라미터
    sma = parser.add_argument_group("SMA Crossover 파라미터")
    sma.add_argument("--sma-short", type=int, default=20)
    sma.add_argument("--sma-long",  type=int, default=50)

    # RSI 파라미터
    rsi = parser.add_argument_group("RSI 파라미터")
    rsi.add_argument("--rsi-period",    type=int,   default=14)
    rsi.add_argument("--rsi-oversold",  type=float, default=30.0)
    rsi.add_argument("--rsi-overbought",type=float, default=70.0)

    # Bollinger Bands 파라미터
    bb = parser.add_argument_group("Bollinger Bands 파라미터")
    bb.add_argument("--bb-period", type=int,   default=20)
    bb.add_argument("--bb-std",    type=float, default=2.0)

    return parser.parse_args()


def print_results(metrics: dict):
    if "error" in metrics:
        print(f"\n[오류] {metrics['error']}")
        return

    print("\n" + "=" * 60)
    print(f"  백테스트 결과: {metrics['strategy']}")
    print("=" * 60)

    rows = [
        ["초기 자본",        f"${metrics['initial_capital']:,.2f}"],
        ["최종 자본",        f"${metrics['final_capital']:,.2f}"],
        ["총 수익률",        f"{metrics['total_return_pct']:+.2f}%"],
        ["최대 낙폭 (MDD)",  f"{metrics['max_drawdown_pct']:.2f}%"],
        ["샤프 비율",        f"{metrics['sharpe_ratio']:.3f}"],
        ["─" * 20,           "─" * 20],
        ["총 거래 수",       metrics['total_trades']],
        ["승리 거래",        metrics['winning_trades']],
        ["패배 거래",        metrics['losing_trades']],
        ["승률",             f"{metrics['win_rate_pct']:.1f}%"],
        ["평균 수익 (승)",   f"${metrics['avg_win_usdt']:,.2f}"],
        ["평균 손실 (패)",   f"${metrics['avg_loss_usdt']:,.2f}"],
        ["수익 팩터",        f"{metrics['profit_factor']:.3f}"],
    ]
    print(tabulate(rows, tablefmt="simple"))
    print("=" * 60)


def plot_results(result: dict, filename: str = "chart.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        print("[경고] matplotlib이 설치되지 않아 차트를 저장할 수 없습니다.")
        return

    df = result["price_data"]
    equity = result["equity_curve"]
    trades = result["trades"]
    strategy_name = str(result["strategy"])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1]})
    fig.suptitle(f"Bitcoin Backtest — {strategy_name}", fontsize=14)

    # 가격 차트
    ax1.plot(df.index, df["close"], color="#888", linewidth=1, label="BTC Price")
    ax1.set_ylabel("Price (USDT)")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    for trade in trades:
        ax1.axvline(trade.entry_time, color="green", alpha=0.4, linewidth=0.8)
        if trade.exit_time:
            ax1.axvline(trade.exit_time, color="red", alpha=0.4, linewidth=0.8)

    # 에쿼티 커브
    ax2.plot(equity.index, equity["equity"], color="#2196F3", linewidth=1.5)
    ax2.fill_between(equity.index, equity["equity"].min(), equity["equity"],
                     alpha=0.1, color="#2196F3")
    ax2.set_ylabel("Equity (USDT)")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"\n차트 저장 완료: {filename}")


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


def main():
    args = parse_args()

    print(f"\nBitcoin Backtesting Bot")
    print(f"심볼: {args.symbol} | 기간: {args.start} ~ {args.end} | 간격: {args.interval}")

    # 1. 데이터 수집
    fetcher = DataFetcher(symbol=args.symbol)
    df = fetcher.fetch(start=args.start, end=args.end, interval=args.interval)

    # 2. 전략 선택
    strategy = STRATEGIES[args.strategy](args)
    print(f"전략: {strategy}")

    # 3. 백테스트 실행
    engine = BacktestEngine(
        initial_capital=args.capital,
        fee_rate=args.fee,
    )
    result = engine.run(df, strategy)

    # 4. 성과 지표 출력
    metrics = PerformanceMetrics(result).summary()
    print_results(metrics)
    print_trade_log(result["trades"])

    # 5. 차트 (선택)
    if args.plot:
        plot_results(result)


if __name__ == "__main__":
    main()
