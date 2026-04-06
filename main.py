"""
봇 실행 진입점
==============
사용법:
    cp .env.example .env
    # .env 에 API 키 입력
    python main.py
"""
import asyncio
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(Path(__file__).parent / "bot.log", encoding="utf-8"),
    ],
)

sys.path.insert(0, str(Path(__file__).parent))
from trader.live_bot import LiveBot


async def main():
    api_key    = os.getenv("BINANCE_API_KEY", "")
    api_secret = os.getenv("BINANCE_API_SECRET", "")
    testnet    = os.getenv("TESTNET", "true").lower() != "false"

    if not api_key or not api_secret:
        logging.error(".env에 BINANCE_API_KEY / BINANCE_API_SECRET 설정 필요")
        sys.exit(1)

    bot = LiveBot(api_key=api_key, api_secret=api_secret, testnet=testnet)
    try:
        await bot.start()
    except KeyboardInterrupt:
        pass
    finally:
        await bot.stop()


if __name__ == "__main__":
    asyncio.run(main())
