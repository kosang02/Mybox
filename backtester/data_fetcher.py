import time
import requests
import pandas as pd
from datetime import datetime, timedelta


KRAKEN_OHLC_URL = "https://api.kraken.com/0/public/OHLC"

# Kraken interval (minutes) → 하루치 캔들 수
KRAKEN_INTERVALS = {
    "1m":  1,
    "5m":  5,
    "15m": 15,
    "30m": 30,
    "1h":  60,
    "4h":  240,
    "1d":  1440,
    "1w":  10080,
}

# Kraken API 최대 반환 수: 720 캔들
MAX_CANDLES = 720


class DataFetcher:
    """
    Kraken 공개 API로 BTC OHLCV 데이터를 수집합니다.

    제약: 공개 API는 interval당 최대 720 캔들을 반환합니다.
      - 1d  → 최대 720일 (~2년)
      - 4h  → 최대 120일
      - 1h  → 최대 30일
    """

    PAIR = "XBTUSD"

    def __init__(self, symbol: str = "BTCUSDT"):
        # Kraken은 XBTUSD 심볼 고정 (BTCUSDT 호환 유지)
        self.symbol = symbol

    def fetch(
        self,
        start: str,
        end: str,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        지정된 기간의 OHLCV 데이터를 반환합니다.

        Args:
            start: 시작일 (예: "2024-01-01")
            end:   종료일 (예: "2024-12-31")
            interval: 캔들 간격 (1m/5m/15m/30m/1h/4h/1d/1w)

        Returns:
            columns: open, high, low, close, volume
            index: datetime (UTC)
        """
        if interval not in KRAKEN_INTERVALS:
            raise ValueError(
                f"지원하지 않는 interval: {interval}. "
                f"가능한 값: {sorted(KRAKEN_INTERVALS)}"
            )

        start_dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d").replace(
            hour=23, minute=59, second=59
        )

        # 가능한 기간 검증
        interval_min = KRAKEN_INTERVALS[interval]
        max_days = (MAX_CANDLES * interval_min) / 1440
        available_from = datetime.utcnow() - timedelta(days=max_days)

        if start_dt < available_from:
            available_str = available_from.strftime("%Y-%m-%d")
            print(
                f"  [경고] {interval} 간격은 최근 {int(max_days)}일({available_str} 이후) "
                f"데이터만 제공됩니다.\n"
                f"  더 긴 기간이 필요하면 --interval 1d (최대 ~2년)를 사용하세요."
            )

        print(f"  데이터 수집 중: BTC/USD {interval} ({start} ~ {end})")
        since_ts = int(start_dt.timestamp())
        all_candles = self._fetch_all(since_ts, interval_min)

        if not all_candles:
            raise RuntimeError(
                "데이터를 가져오지 못했습니다. 기간/간격을 확인해주세요."
            )

        df = self._to_dataframe(all_candles)

        # 요청 범위로 필터링
        start_ts = pd.Timestamp(start, tz="UTC")
        end_ts = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
        df = df[(df.index >= start_ts) & (df.index < end_ts)]

        if df.empty:
            raise RuntimeError(
                f"지정한 기간({start} ~ {end})에 데이터가 없습니다.\n"
                f"공개 API 제약: {interval} 간격은 최근 {int(max_days)}일 이내만 가능합니다."
            )

        print(f"  총 {len(df)}개 캔들 수집 완료")
        return df

    def _fetch_all(self, since_ts: int, interval_min: int) -> list:
        """720 캔들 한계 내에서 페이지네이션으로 데이터 수집."""
        all_candles = []
        current_since = since_ts

        while True:
            data, last_ts = self._request(current_since, interval_min)
            if not data:
                break
            all_candles.extend(data)
            # Kraken은 마지막 미완성 캔들을 포함하므로 next_since로 이동
            if last_ts is None or last_ts <= current_since:
                break
            current_since = last_ts
            if len(data) < MAX_CANDLES:
                break
            time.sleep(0.5)

        return all_candles

    def _request(self, since: int, interval_min: int) -> tuple[list, int | None]:
        params = {
            "pair": self.PAIR,
            "interval": interval_min,
            "since": since,
        }
        resp = requests.get(KRAKEN_OHLC_URL, params=params, timeout=10)
        resp.raise_for_status()
        body = resp.json()

        if body.get("error"):
            raise RuntimeError(f"Kraken API 오류: {body['error']}")

        result = body["result"]
        # 페어 키는 'XXBTZUSD' 또는 'XBTUSD'
        pair_key = next(k for k in result if k != "last")
        candles = result[pair_key]
        last_ts = result.get("last")
        return candles, last_ts

    @staticmethod
    def _to_dataframe(candles: list) -> pd.DataFrame:
        # Kraken OHLCV: [time, open, high, low, close, vwap, volume, count]
        df = pd.DataFrame(candles, columns=[
            "time", "open", "high", "low", "close", "vwap", "volume", "count"
        ])
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.set_index("time")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        return df[["open", "high", "low", "close", "volume"]]
