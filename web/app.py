"""
웹 대시보드 백엔드
==================
LiveBot과 별도 프로세스로 실행.
SQLite(data/bot.db)를 읽기 전용으로 접근해 상태를 제공.

실행:
    uvicorn web.app:app --host 0.0.0.0 --port 8000
"""
import asyncio
import json
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).parent.parent))
from trader.db import init_db, load_position, load_trades, load_equity

LOG_PATH = Path(__file__).parent.parent / "bot.log"


def read_logs(lines: int = 100) -> list[str]:
    if not LOG_PATH.exists():
        return []
    with open(LOG_PATH, encoding="utf-8") as f:
        all_lines = f.readlines()
    return [l.rstrip() for l in all_lines[-lines:]]

app = FastAPI(title="BTC Bot Dashboard")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_static = Path(__file__).parent / "static"


@app.on_event("startup")
async def startup():
    init_db()


# ── REST 엔드포인트 ───────────────────────────────────────────

@app.get("/api/status")
def get_status():
    position = load_position()
    trades   = load_trades(limit=50)
    equity   = load_equity(limit=500)

    total_pnl = sum(t["pnl"] for t in trades)
    wins      = sum(1 for t in trades if t["pnl"] > 0)
    win_rate  = round(wins / len(trades) * 100, 1) if trades else 0.0

    latest_equity = load_equity(limit=1)
    balance = latest_equity[0]["value"] if latest_equity else 0.0

    return {
        "position": position,
        "trades":   trades,
        "equity":   equity,
        "summary": {
            "total_trades": len(trades),
            "total_pnl":    round(total_pnl, 2),
            "win_rate":     win_rate,
            "balance":      round(balance, 2),
        },
    }


@app.get("/api/position")
def get_position():
    return load_position() or {}


@app.get("/api/trades")
def get_trades(limit: int = 50):
    return load_trades(limit=limit)


@app.get("/api/equity")
def get_equity(limit: int = 500):
    return load_equity(limit=limit)


@app.get("/api/logs")
def get_logs(lines: int = 100):
    return {"logs": read_logs(lines)}


# ── SSE — 2초마다 상태 push ──────────────────────────────────

@app.get("/api/stream")
async def stream():
    async def event_generator():
        while True:
            all_trades = load_trades(limit=1000)
            data = {
                "position": load_position(),
                "equity":   load_equity(limit=1),
                "summary": {
                    "total_trades": len(all_trades),
                    "total_pnl":    round(sum(t["pnl"] for t in all_trades), 2),
                    "win_rate":     round(
                        sum(1 for t in all_trades if t["pnl"] > 0) / len(all_trades) * 100, 1
                    ) if all_trades else 0.0,
                    "balance":      round(load_equity(limit=1)[0]["value"], 2) if load_equity(limit=1) else 0.0,
                },
                "logs": read_logs(50),
            }
            yield f"data: {json.dumps(data)}\n\n"
            await asyncio.sleep(2)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ── 정적 파일 (프론트엔드) ────────────────────────────────────
if _static.exists():
    app.mount("/", StaticFiles(directory=str(_static), html=True), name="static")
