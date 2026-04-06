"""
Binance Futures REST + WebSocket 클라이언트
=============================================
Binance USDⓈ-M 선물 전용.
- REST: 잔고/포지션/주문
- WebSocket: 1h 봉 마감 이벤트
"""
import asyncio
import hashlib
import hmac
import json
import time
from urllib.parse import urlencode

import aiohttp
import pandas as pd


MAINNET_REST = "https://fapi.binance.com"
TESTNET_REST = "https://testnet.binancefuture.com"
MAINNET_WS   = "wss://fstream.binance.com/ws"
TESTNET_WS   = "wss://stream.binancefuture.com/ws"


class BinanceFutures:
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        self.api_key    = api_key
        self.api_secret = api_secret
        self.rest_base  = TESTNET_REST if testnet else MAINNET_REST
        self.ws_base    = TESTNET_WS   if testnet else MAINNET_WS
        self.testnet    = testnet
        self._session: aiohttp.ClientSession | None = None

    async def _sess(self) -> aiohttp.ClientSession:
        if not self._session or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    # ── 서명 / 헤더 ──────────────────────────────────────────

    def _sign(self, params: dict) -> dict:
        params["timestamp"] = int(time.time() * 1000)
        qs  = urlencode(params)
        sig = hmac.new(self.api_secret.encode(), qs.encode(), hashlib.sha256).hexdigest()
        params["signature"] = sig
        return params

    @property
    def _headers(self) -> dict:
        return {"X-MBX-APIKEY": self.api_key}

    # ── HTTP 공통 ─────────────────────────────────────────────

    async def _get(self, path: str, params: dict | None = None, signed: bool = False):
        sess = await self._sess()
        p = dict(params or {})
        if signed:
            p = self._sign(p)
        async with sess.get(
            f"{self.rest_base}{path}", params=p,
            headers=self._headers, timeout=aiohttp.ClientTimeout(total=10)
        ) as r:
            data = await r.json()
        if isinstance(data, dict) and data.get("code", 0) < 0:
            raise RuntimeError(f"Binance API 오류: {data}")
        return data

    async def _post(self, path: str, params: dict):
        sess = await self._sess()
        p = self._sign(dict(params))
        async with sess.post(
            f"{self.rest_base}{path}", params=p,
            headers=self._headers, timeout=aiohttp.ClientTimeout(total=10)
        ) as r:
            data = await r.json()
        if isinstance(data, dict) and data.get("code", 0) < 0:
            raise RuntimeError(f"Binance API 오류: {data}")
        return data

    async def _delete(self, path: str, params: dict):
        sess = await self._sess()
        p = self._sign(dict(params))
        async with sess.delete(
            f"{self.rest_base}{path}", params=p,
            headers=self._headers, timeout=aiohttp.ClientTimeout(total=10)
        ) as r:
            return await r.json()

    # ── Public ───────────────────────────────────────────────

    async def get_klines(self, symbol: str, interval: str, limit: int = 300) -> pd.DataFrame:
        raw = await self._get("/fapi/v1/klines", {
            "symbol": symbol, "interval": interval, "limit": limit,
        })
        df = pd.DataFrame(raw, columns=[
            "time","open","high","low","close","volume",
            "close_time","qvol","trades","tb_base","tb_quote","ignore",
        ])
        df["time"] = pd.to_datetime(df["time"].astype(int), unit="ms", utc=True)
        df = df.set_index("time")
        for c in ["open","high","low","close","volume"]:
            df[c] = df[c].astype(float)
        return df[["open","high","low","close","volume"]]

    # ── Signed ───────────────────────────────────────────────

    async def set_leverage(self, symbol: str, leverage: int):
        await self._post("/fapi/v1/leverage", {"symbol": symbol, "leverage": leverage})

    async def set_margin_type(self, symbol: str, margin_type: str = "ISOLATED"):
        try:
            await self._post("/fapi/v1/marginType", {
                "symbol": symbol, "marginType": margin_type,
            })
        except RuntimeError as e:
            # 이미 해당 마진 타입이면 -4046 오류 → 무시
            if "-4046" not in str(e):
                raise

    async def get_balance(self, asset: str = "USDT") -> float:
        data = await self._get("/fapi/v2/balance", signed=True)
        for b in data:
            if b["asset"] == asset:
                return float(b["availableBalance"])
        return 0.0

    async def get_position(self, symbol: str) -> dict | None:
        """현재 포지션. 없으면 None."""
        data = await self._get("/fapi/v2/positionRisk", {"symbol": symbol}, signed=True)
        for p in data:
            if p["symbol"] == symbol:
                amt = float(p["positionAmt"])
                if amt != 0:
                    return {
                        "side":      "long" if amt > 0 else "short",
                        "qty":       abs(amt),
                        "entry":     float(p["entryPrice"]),
                        "unrealized": float(p["unRealizedProfit"]),
                    }
        return None

    async def market_order(self, symbol: str, side: str, quantity: float) -> dict:
        """시장가 주문. side: 'BUY' | 'SELL'"""
        return await self._post("/fapi/v1/order", {
            "symbol":   symbol,
            "side":     side,
            "type":     "MARKET",
            "quantity": f"{quantity:.3f}",
        })

    async def place_sl_tp(
        self,
        symbol:    str,
        pos_side:  str,   # 'long' | 'short'
        quantity:  float,
        sl_price:  float,
        tp_price:  float,
    ):
        """SL(STOP_MARKET) + TP(TAKE_PROFIT_MARKET) 주문 설정."""
        close_side = "SELL" if pos_side == "long" else "BUY"
        qty = f"{quantity:.3f}"

        await self._post("/fapi/v1/order", {
            "symbol":      symbol,
            "side":        close_side,
            "type":        "STOP_MARKET",
            "stopPrice":   f"{sl_price:.2f}",
            "quantity":    qty,
            "reduceOnly":  "true",
            "timeInForce": "GTC",
            "workingType": "CONTRACT_PRICE",
        })

        await self._post("/fapi/v1/order", {
            "symbol":      symbol,
            "side":        close_side,
            "type":        "TAKE_PROFIT_MARKET",
            "stopPrice":   f"{tp_price:.2f}",
            "quantity":    qty,
            "reduceOnly":  "true",
            "timeInForce": "GTC",
            "workingType": "CONTRACT_PRICE",
        })

    async def get_open_orders(self, symbol: str) -> list:
        """현재 미체결 주문 목록."""
        return await self._get("/fapi/v1/openOrders", {"symbol": symbol}, signed=True)

    async def get_symbol_info(self, symbol: str) -> dict:
        """minNotional, stepSize 등 심볼 거래 제약 조회."""
        data = await self._get("/fapi/v1/exchangeInfo")
        for s in data.get("symbols", []):
            if s["symbol"] == symbol:
                info = {"stepSize": 0.001, "minNotional": 5.0, "minQty": 0.001}
                for f in s.get("filters", []):
                    if f["filterType"] == "LOT_SIZE":
                        info["stepSize"] = float(f["stepSize"])
                        info["minQty"]   = float(f["minQty"])
                    elif f["filterType"] == "MIN_NOTIONAL":
                        info["minNotional"] = float(f.get("notional", 5.0))
                return info
        return {"stepSize": 0.001, "minNotional": 5.0, "minQty": 0.001}

    async def cancel_all_orders(self, symbol: str):
        await self._delete("/fapi/v1/allOpenOrders", {"symbol": symbol})

    # ── WebSocket ────────────────────────────────────────────

    async def ws_kline(self, symbol: str, interval: str, on_close):
        """
        봉이 마감될 때마다 on_close(kline_dict) 호출.
        연결 끊기면 자동 재연결.
        """
        stream = f"{symbol.lower()}@kline_{interval}"
        while True:
            try:
                sess = await self._sess()
                async with sess.ws_connect(
                    f"{self.ws_base}/{stream}",
                    heartbeat=30,
                    timeout=aiohttp.ClientTimeout(total=None),
                ) as ws:
                    print(f"[WS] 연결됨: {stream}")
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            data = json.loads(msg.data)
                            k = data.get("k", {})
                            if k.get("x"):  # x=True → 봉 마감
                                await on_close(k)
                        elif msg.type in (
                            aiohttp.WSMsgType.CLOSED,
                            aiohttp.WSMsgType.ERROR,
                        ):
                            break
            except Exception as e:
                print(f"[WS] 연결 끊김: {e} — 5초 후 재연결")
            await asyncio.sleep(5)
