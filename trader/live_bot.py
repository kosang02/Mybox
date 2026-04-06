"""
Binance Futures 라이브 봇
==========================
확정 전략: BB Squeeze Breakout 1h + 200일 MA 국면 필터

동작 흐름:
  1. 시작 시 기존 포지션/SL/TP 복구 (재시작 대비)
  2. WebSocket으로 1h 봉 마감 이벤트 수신
  3. REST API로 최근 캔들 수집 (1h: 신호 계산, 1d: 국면 필터)
  4. BBBreakout.generate_signals() + apply_regime() — iloc[-2] 기준
  5. 포지션 없을 때만 진입 (중복 방지)
  6. minNotional/stepSize 검증 후 시장가 진입 → SL/TP 자동 설정
"""
import asyncio
import logging
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies import BBBreakout
from .binance_futures import BinanceFutures
from .db import (
    init_db, save_position, load_position, clear_position,
    save_trade, save_equity,
)

log = logging.getLogger("live_bot")

# ── 설정 (환경변수 우선, 기본값 폴백) ────────────────────────
SYMBOL       = os.getenv("SYMBOL",      "BTCUSDT")
LEVERAGE     = int(os.getenv("LEVERAGE", "10"))
RISK_PCT     = float(os.getenv("RISK_PCT",   "0.01"))
SL_ATR_MULT  = float(os.getenv("SL_ATR_MULT","1.5"))
TP_ATR_MULT  = float(os.getenv("TP_ATR_MULT","3.0"))
ATR_PERIOD   = int(os.getenv("ATR_PERIOD",   "14"))
CANDLES_1H   = int(os.getenv("CANDLES_1H",  "300"))
CANDLES_1D   = int(os.getenv("CANDLES_1D",  "300"))


# ── 보조 함수 ─────────────────────────────────────────────────

def _calc_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> float:
    high  = df["high"].values
    low   = df["low"].values
    close = df["close"].values
    prev  = np.roll(close, 1); prev[0] = close[0]
    tr = np.maximum.reduce([
        high - low,
        np.abs(high - prev),
        np.abs(low  - prev),
    ])
    alpha = 1.0 / period
    atr   = float("nan")
    for i in range(len(tr)):
        if i < period - 1:
            continue
        if i == period - 1:
            atr = float(tr[:period].mean())
        else:
            atr = atr * (1 - alpha) + float(tr[i]) * alpha
    return atr


def apply_regime(
    signals:   pd.Series,
    df_daily:  pd.DataFrame,
    df_target: pd.DataFrame,
) -> pd.Series:
    """200일 MA 국면 필터: 황소장 → 롱만 / 곰장 → 숏만."""
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


def _floor_step(quantity: float, step_size: float) -> float:
    """stepSize 단위로 내림."""
    precision = max(0, round(-math.log10(step_size)))
    return math.floor(quantity / step_size) * step_size


# ── 봇 본체 ──────────────────────────────────────────────────

class LiveBot:
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        self.client   = BinanceFutures(api_key, api_secret, testnet)
        self.strategy = BBBreakout(squeeze_pct=15, trend_filter=True, trend_ema=20)
        self._busy    = False
        self._sym_info: dict | None = None   # exchange_info 캐시

    async def start(self):
        mode = "TESTNET" if self.client.testnet else "MAINNET"
        log.info(f"봇 시작 [{mode}] | {SYMBOL} | 레버리지 {LEVERAGE}x | SL×{SL_ATR_MULT} TP×{TP_ATR_MULT}")

        init_db()

        # 심볼 제약 캐시
        self._sym_info = await self.client.get_symbol_info(SYMBOL)
        log.info(f"심볼 제약: {self._sym_info}")

        # 레버리지 / 마진 타입 (포지션 없을 때만 변경 가능)
        existing = await self.client.get_position(SYMBOL)
        if not existing:
            await self.client.set_leverage(SYMBOL, LEVERAGE)
            await self.client.set_margin_type(SYMBOL, "ISOLATED")
            log.info(f"레버리지 {LEVERAGE}x, ISOLATED 마진 설정 완료")

        # 재시작 복구
        await self._recover_on_startup()

        # 시작 직후 즉시 한 번 체크
        await self._check_and_trade()

        # 1h 봉 마감 이벤트 구독 (블로킹)
        await self.client.ws_kline(SYMBOL, "1h", self._on_kline_close)

    async def stop(self):
        await self.client.close()
        log.info("봇 종료")

    # ── 재시작 복구 ───────────────────────────────────────────

    async def _recover_on_startup(self):
        """
        봇 재시작 시: 거래소 포지션 + DB 포지션 비교.
        포지션 있는데 SL/TP 주문 없으면 DB 저장값으로 재설정.
        """
        live_pos = await self.client.get_position(SYMBOL)
        db_pos   = load_position()

        if not live_pos:
            if db_pos:
                log.warning("DB에 포지션 기록 있지만 거래소에 포지션 없음 → DB 클리어")
                clear_position()
            return

        log.info(f"기존 포지션 감지: {live_pos['side']} qty={live_pos['qty']:.3f}")

        if not db_pos:
            log.error(
                "거래소에 포지션 있지만 DB 기록 없음 → SL/TP 복구 불가. "
                "수동으로 확인 필요."
            )
            return

        # SL/TP 주문 살아있는지 확인
        open_orders = await self.client.get_open_orders(SYMBOL)
        order_types = {o["type"] for o in open_orders}
        has_sl = "STOP_MARKET"        in order_types
        has_tp = "TAKE_PROFIT_MARKET" in order_types

        if has_sl and has_tp:
            log.info("SL/TP 주문 정상 확인 → 복구 불필요")
            return

        log.warning(f"SL/TP 누락 (sl={has_sl}, tp={has_tp}) → DB 값으로 재설정")
        await self.client.cancel_all_orders(SYMBOL)
        await self.client.place_sl_tp(
            symbol   = SYMBOL,
            pos_side = db_pos["side"],
            quantity = db_pos["quantity"],
            sl_price = db_pos["sl_price"],
            tp_price = db_pos["tp_price"],
        )
        log.info(
            f"SL/TP 재설정 완료 | SL={db_pos['sl_price']} TP={db_pos['tp_price']}"
        )

    # ── 이벤트 핸들러 ─────────────────────────────────────────

    async def _on_kline_close(self, k: dict):
        ts = datetime.fromtimestamp(k["T"] / 1000, tz=timezone.utc)
        log.info(f"1h 봉 마감: {ts.strftime('%Y-%m-%d %H:%M UTC')}")

        if self._busy:
            log.warning("이전 처리 진행 중 — 스킵")
            return
        self._busy = True
        try:
            await asyncio.sleep(2)  # REST 캔들 반영 대기
            await self._check_and_trade()
        except Exception as e:
            log.error(f"체크 중 오류: {e}", exc_info=True)
        finally:
            self._busy = False

    # ── 핵심 로직 ─────────────────────────────────────────────

    async def _check_and_trade(self):
        df_1h, df_1d = await asyncio.gather(
            self.client.get_klines(SYMBOL, "1h", CANDLES_1H),
            self.client.get_klines(SYMBOL, "1d", CANDLES_1D),
        )

        # 신호 계산: iloc[-2] = 방금 닫힌 봉 (iloc[-1]은 현재 진행 중인 미완성 봉)
        signals = self.strategy.generate_signals(df_1h)
        signals = apply_regime(signals, df_1d, df_1h)
        signal  = int(signals.iloc[-2])

        price = float(df_1h["close"].iloc[-2])
        atr   = _calc_atr(df_1h.iloc[:-1])   # 미완성 봉 제외

        signal_str = {1: "롱 ▲", -1: "숏 ▼", 0: "없음"}.get(signal, "?")
        log.info(f"신호: {signal_str} | 가격: ${price:,.2f} | ATR: {atr:.2f}")

        # 에쿼티 기록
        balance = await self.client.get_balance("USDT")
        live_pos = await self.client.get_position(SYMBOL)
        total = balance
        if live_pos:
            cur_price = float(df_1h["close"].iloc[-1])
            total += live_pos["qty"] * cur_price
        save_equity(total)

        # 포지션 중복 방지
        if live_pos:
            log.info(
                f"포지션 유지 중 ({live_pos['side']}, "
                f"qty={live_pos['qty']:.3f}, "
                f"미실현={live_pos['unrealized']:+.2f}$) — 진입 스킵"
            )
            return

        # DB 포지션 불일치 클리어 (거래소에 포지션 없는데 DB에 남아있는 경우)
        db_pos = load_position()
        if db_pos and not live_pos:
            log.info("거래소 포지션 없음 → DB 포지션 클리어")
            clear_position()

        if signal == 0:
            return

        await self._enter(signal, price, atr)

    # ── 진입 ─────────────────────────────────────────────────

    async def _enter(self, direction: int, price: float, atr: float):
        balance = await self.client.get_balance("USDT")

        sl_dist = atr * SL_ATR_MULT
        tp_dist = atr * TP_ATR_MULT

        # 리스크 1% 기반 수량 계산
        risk_usdt = balance * RISK_PCT
        raw_qty   = risk_usdt / sl_dist

        # stepSize 단위 절사
        step_size = self._sym_info["stepSize"] if self._sym_info else 0.001
        min_qty   = self._sym_info["minQty"]   if self._sym_info else 0.001
        min_notional = self._sym_info["minNotional"] if self._sym_info else 5.0

        quantity = _floor_step(raw_qty, step_size)

        # 최소 수량 / 최소 명목가치 검증
        if quantity < min_qty:
            log.error(f"수량 미달: {quantity:.4f} < minQty {min_qty} → 진입 스킵")
            return
        if quantity * price < min_notional:
            log.error(
                f"명목가치 미달: {quantity * price:.2f}$ < minNotional {min_notional}$ → 진입 스킵"
            )
            return
        if balance <= 0:
            log.error(f"잔고 없음: {balance:.2f} → 진입 스킵")
            return

        side     = "BUY"  if direction == 1 else "SELL"
        side_str = "롱 ▲" if direction == 1 else "숏 ▼"
        pos_side = "long" if direction == 1 else "short"

        if direction == 1:
            sl_price = round(price - sl_dist, 2)
            tp_price = round(price + tp_dist, 2)
        else:
            sl_price = round(price + sl_dist, 2)
            tp_price = round(price - tp_dist, 2)

        log.info(
            f"[진입 시도] {side_str} | qty={quantity:.3f} BTC | "
            f"진입=${price:,.2f} | SL=${sl_price:,.2f} | TP=${tp_price:,.2f}"
        )

        entry_time = datetime.now(timezone.utc).isoformat()

        await self.client.cancel_all_orders(SYMBOL)
        order = await self.client.market_order(SYMBOL, side, quantity)
        log.info(f"[주문 체결] orderId={order.get('orderId')} status={order.get('status')}")

        # DB에 포지션 저장 (재시작 복구용)
        save_position(
            side       = pos_side,
            entry_price= price,
            sl_price   = sl_price,
            tp_price   = tp_price,
            quantity   = quantity,
            entry_time = entry_time,
        )

        await self.client.place_sl_tp(SYMBOL, pos_side, quantity, sl_price, tp_price)
        log.info("[SL/TP 설정 완료]")
