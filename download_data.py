"""
Binance Vision 과거 데이터 다운로더
=====================================
data.binance.vision 에서 BTCUSDT 선물 OHLCV 데이터를 다운받아
data/ 폴더에 parquet으로 캐싱합니다.

사용:
  python download_data.py --interval 5m --start 2023-01 --end 2024-12
  python download_data.py --interval 1m 5m 15m --start 2022-01 --end 2024-12
"""

import io
import os
import zipfile
import argparse
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

BASE_URL = "https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT"
DATA_DIR = Path("data")

COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "count",
    "taker_buy_volume", "taker_buy_quote_volume", "ignore",
]


def download_month(interval: str, year: int, month: int) -> pd.DataFrame | None:
    """월별 zip 파일 다운로드 → DataFrame 반환 (실패 시 None)."""
    filename = f"BTCUSDT-{interval}-{year}-{month:02d}.zip"
    url = f"{BASE_URL}/{interval}/{filename}"

    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
    except Exception as e:
        print(f"  [오류] {filename}: {e}")
        return None

    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        csv_name = z.namelist()[0]
        with z.open(csv_name) as f:
            raw = pd.read_csv(f, header=None, names=COLUMNS)
            # 첫 행이 헤더 문자열인 경우 제거
            if str(raw.iloc[0]["open_time"]).lower() in ("open_time", "timestamp"):
                raw = raw.iloc[1:].reset_index(drop=True)
            df = raw

    df["open_time"] = pd.to_datetime(df["open_time"].astype(np.int64), unit="ms", utc=True)
    df = df.set_index("open_time")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df[["open", "high", "low", "close", "volume"]]


def load_or_download(interval: str, year: int, month: int) -> pd.DataFrame | None:
    """캐시에 있으면 로드, 없으면 다운로드 후 저장."""
    cache_path = DATA_DIR / interval / f"{year}-{month:02d}.parquet"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        return pd.read_parquet(cache_path)

    df = download_month(interval, year, month)
    if df is not None:
        df.to_parquet(cache_path)
    return df


def fetch(interval: str, start: str, end: str) -> pd.DataFrame:
    """
    start ~ end 기간의 데이터를 반환합니다.

    Args:
        interval: "1m", "5m", "15m", "1h" 등
        start:    "2023-01"
        end:      "2024-12"
    """
    start_y, start_m = map(int, start.split("-"))
    end_y, end_m = map(int, end.split("-"))

    months = []
    y, m = start_y, start_m
    while (y, m) <= (end_y, end_m):
        months.append((y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1

    frames = []
    for y, m in tqdm(months, desc=f"[{interval}] 데이터 로딩"):
        df = load_or_download(interval, y, m)
        if df is not None:
            frames.append(df)

    if not frames:
        raise RuntimeError(f"데이터 없음: {interval} {start}~{end}")

    result = pd.concat(frames).sort_index()
    result = result[~result.index.duplicated(keep="first")]
    print(f"  [{interval}] 총 {len(result):,}개 캔들 ({result.index[0].date()} ~ {result.index[-1].date()})")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", nargs="+", default=["5m"],
                        help="캔들 간격 (예: 1m 5m 15m)")
    parser.add_argument("--start", default="2023-01", help="시작 월 (YYYY-MM)")
    parser.add_argument("--end",   default="2024-12", help="종료 월 (YYYY-MM)")
    args = parser.parse_args()

    for iv in args.interval:
        print(f"\n{iv} 데이터 다운로드 중...")
        df = fetch(iv, args.start, args.end)
        print(f"  저장 완료: data/{iv}/")
