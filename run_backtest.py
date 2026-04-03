"""
백테스팅 실행 진입점
====================
전략 A (BB 스퀴즈+EMA) vs 전략 B (BB 터치)를
여러 타임프레임에서 비교합니다.

사용:
  python run_backtest.py
  python run_backtest.py --interval 5m 15m --leverage 10 --start 2023-01 --end 2024-12
  python run_backtest.py --strategy A --interval 5m --detail
"""

import argparse
import pandas as pd
from tabulate import tabulate

from download_data import fetch
from strategies import BBSqueezeEMA, BBTouch
from backtester import BacktestEngine, calc_metrics


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--interval",  nargs="+", default=["5m", "15m"],
                   help="테스트할 타임프레임 (예: 1m 5m 15m 1h)")
    p.add_argument("--start",     default="2023-01", help="시작 월 (YYYY-MM)")
    p.add_argument("--end",       default="2024-12", help="종료 월 (YYYY-MM)")
    p.add_argument("--leverage",  type=int,   default=10)
    p.add_argument("--capital",   type=float, default=10_000.0)
    p.add_argument("--risk",      type=float, default=0.01,  help="거래당 리스크 비율")
    p.add_argument("--sl-mult",   type=float, default=1.5,   help="손절 = ATR × 배수")
    p.add_argument("--tp-mult",   type=float, default=2.5,   help="익절 = ATR × 배수")
    p.add_argument("--strategy",  default="both", choices=["A", "B", "both"],
                   help="A=BB스퀴즈+EMA, B=BB터치, both=둘다")
    p.add_argument("--detail",    action="store_true", help="거래 내역 상세 출력")
    return p.parse_args()


def fmt_hold(seconds: float) -> str:
    if seconds < 3600:
        return f"{int(seconds//60)}분"
    elif seconds < 86400:
        return f"{seconds/3600:.1f}시간"
    else:
        return f"{seconds/86400:.1f}일"


def print_comparison(results: list[dict], initial_capital: float):
    if not results:
        print("결과 없음")
        return

    # 수익률 기준 정렬
    results.sort(key=lambda x: x["total_return"], reverse=True)

    rows = []
    for r in results:
        ret_sign = "+" if r["total_return"] >= 0 else ""
        rows.append([
            r["strategy"],
            r["interval"],
            f"{ret_sign}{r['total_return']}%",
            f"{r['sharpe']:.3f}",
            f"{r['win_rate']}%",
            f"{r['mdd']}%",
            r["total_trades"],
            r["liq_count"],
            r["profit_factor"],
            fmt_hold(r["avg_hold_sec"]),
            f"${r['final_capital']:,.0f}",
        ])

    headers = ["전략", "타임프레임", "수익률", "샤프", "승률", "MDD",
               "거래수", "청산수", "수익팩터", "평균보유", "최종자본"]

    print(f"\n{'='*100}")
    print(f"  BTC 선물 백테스팅 비교  |  레버리지: {results[0].get('leverage','?')}x  |  "
          f"초기자본: ${initial_capital:,.0f}")
    print(f"{'='*100}")
    print(tabulate(rows, headers=headers, tablefmt="simple"))
    print(f"{'='*100}\n")


def print_detail(trades, strategy_name: str, interval: str, max_show: int = 20):
    print(f"\n  [{strategy_name} / {interval}] 거래 내역 (최근 {max_show}건)")
    rows = []
    for t in trades[-max_show:]:
        if t.pnl is None:
            continue
        sign = "+" if t.pnl >= 0 else ""
        dir_str = "롱" if t.direction == 1 else "숏"
        rows.append([
            dir_str,
            str(t.entry_time)[:16],
            f"${t.entry_price:,.1f}",
            str(t.exit_time)[:16] if t.exit_time else "-",
            f"${t.exit_price:,.1f}" if t.exit_price else "-",
            t.exit_reason,
            f"{sign}${t.pnl:,.2f}",
        ])
    print(tabulate(rows, headers=["방향","진입시간","진입가","청산시간","청산가","사유","손익"],
                   tablefmt="simple"))


def main():
    args = parse_args()

    engine = BacktestEngine(
        initial_capital = args.capital,
        leverage        = args.leverage,
        risk_per_trade  = args.risk,
        sl_atr_mult     = args.sl_mult,
        tp_atr_mult     = args.tp_mult,
    )

    strategies = []
    if args.strategy in ("A", "both"):
        strategies.append(BBSqueezeEMA())
    if args.strategy in ("B", "both"):
        strategies.append(BBTouch())

    all_results = []

    for interval in args.interval:
        print(f"\n[{interval}] 데이터 로딩...")
        try:
            df = fetch(interval, args.start, args.end)
        except RuntimeError as e:
            print(f"  [건너뜀] {e}")
            continue

        for strategy in strategies:
            print(f"  → {strategy} 백테스팅 중...", end=" ")
            signals = strategy.generate_signals(df)
            result  = engine.run(df, signals)
            metrics = calc_metrics(
                result["trades"],
                result["equity_curve"],
                args.capital,
                str(strategy),
                interval,
            )
            metrics["leverage"] = args.leverage
            all_results.append((metrics, result["trades"]))
            print(f"완료 (거래: {metrics['total_trades']}건)")

            if args.detail:
                print_detail(result["trades"], str(strategy), interval)

    if all_results:
        print_comparison([m for m, _ in all_results], args.capital)
    else:
        print("\n결과 없음. 데이터를 먼저 다운받아 주세요:")
        print("  python download_data.py --interval 5m 15m --start 2023-01 --end 2024-12")


if __name__ == "__main__":
    main()
