"""
SQLite 상태 영속화 모듈
========================
봇 재시작 후에도 포지션/거래 이력 복구 가능하도록 저장.

테이블:
- position  : 현재 열린 포지션 (단일 행)
- trades    : 거래 히스토리
- equity    : 에쿼티 커브
"""
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "bot.db"


@contextmanager
def _conn():
    con = sqlite3.connect(DB_PATH, timeout=10)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _conn() as con:
        con.executescript("""
            CREATE TABLE IF NOT EXISTS position (
                id          INTEGER PRIMARY KEY CHECK (id = 1),
                side        TEXT    NOT NULL,          -- 'long' | 'short'
                entry_price REAL    NOT NULL,
                sl_price    REAL    NOT NULL,
                tp_price    REAL    NOT NULL,
                quantity    REAL    NOT NULL,
                entry_time  TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS trades (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                side        TEXT,
                entry_time  TEXT,
                entry_price REAL,
                exit_time   TEXT,
                exit_price  REAL,
                quantity    REAL,
                pnl         REAL,
                pnl_pct     REAL,
                exit_reason TEXT
            );

            CREATE TABLE IF NOT EXISTS equity (
                ts    INTEGER PRIMARY KEY,   -- unix timestamp
                value REAL    NOT NULL
            );
        """)


# ── Position ──────────────────────────────────────────────────

def save_position(
    side: str,
    entry_price: float,
    sl_price: float,
    tp_price: float,
    quantity: float,
    entry_time: str,
):
    with _conn() as con:
        con.execute("""
            INSERT INTO position (id, side, entry_price, sl_price, tp_price, quantity, entry_time)
            VALUES (1, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                side        = excluded.side,
                entry_price = excluded.entry_price,
                sl_price    = excluded.sl_price,
                tp_price    = excluded.tp_price,
                quantity    = excluded.quantity,
                entry_time  = excluded.entry_time
        """, (side, entry_price, sl_price, tp_price, quantity, entry_time))


def load_position() -> dict | None:
    with _conn() as con:
        row = con.execute("SELECT * FROM position WHERE id = 1").fetchone()
    return dict(row) if row else None


def clear_position():
    with _conn() as con:
        con.execute("DELETE FROM position WHERE id = 1")


# ── Trades ────────────────────────────────────────────────────

def save_trade(
    side: str,
    entry_time: str,
    entry_price: float,
    exit_time: str,
    exit_price: float,
    quantity: float,
    pnl: float,
    pnl_pct: float,
    exit_reason: str,
):
    with _conn() as con:
        con.execute("""
            INSERT INTO trades
                (side, entry_time, entry_price, exit_time, exit_price,
                 quantity, pnl, pnl_pct, exit_reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (side, entry_time, entry_price, exit_time, exit_price,
              quantity, pnl, pnl_pct, exit_reason))


def load_trades(limit: int = 100) -> list[dict]:
    with _conn() as con:
        rows = con.execute(
            "SELECT * FROM trades ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


# ── Equity ────────────────────────────────────────────────────

def save_equity(value: float):
    ts = int(datetime.now(timezone.utc).timestamp())
    with _conn() as con:
        con.execute(
            "INSERT OR REPLACE INTO equity (ts, value) VALUES (?, ?)",
            (ts, value)
        )


def load_equity(limit: int = 500) -> list[dict]:
    with _conn() as con:
        rows = con.execute(
            "SELECT ts, value FROM equity ORDER BY ts DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in reversed(rows)]
