# Bitcoin Futures Auto-Trading Bot - 프로젝트 인수인계

## 현재 상태: **라이브 봇 + 웹 대시보드 구현 완료, 테스트넷 검증 대기 중**

---

## 확정 전략: BB Squeeze Breakout 1h

### 파라미터
```python
BBBreakout(
    squeeze_pct=15,      # BB폭 하위 15% = 스퀴즈
    trend_filter=True,
    trend_ema=20,        # EMA20 기울기 방향 필터
)
# SL × 1.5 ATR
# TP × 3.0 ATR (보수) 또는 4.0 ATR (공격) — .env에서 설정
# 200일 MA 국면 필터 필수
# 타임프레임: 1h
# 레버리지: 10x, 거래당 리스크: 자본의 1%
```

### 백테스트 성과 (2022~2024, $10,000 초기자본, 10x)
| 연도 | SL1.5/TP3 | SL1.5/TP4 | 비고 |
|------|-----------|-----------|------|
| 2022 | +4.6% | +9.0% | 베어마켓 - 숏만 진입 |
| 2023 | +16.1% | +19.8% | 회복장 |
| 2024 | +17.8% | +16.4% | 불장 |
| 3년 합계 | **+38.3%** | **+41.8%** | |
| 2025Q1 | -0.1% | -3.4% | 불확실 |
| 전체 기간 | +44.3% → $14,433 | +49.4% → $14,939 | |

- 승률: ~43%, 수익팩터(PF): 1.25
- 월 평균 5~6건 거래

### 200일 MA 국면 필터 (핵심)
```python
# 주의: .astype(bool) 필수! 없으면 ~regime이 -2/-1을 반환하는 dtype 버그 있음
result[(signals ==  1) & (~bull_s)] = 0   # 곰장에서 롱 제거
result[(signals == -1) & (~bear_s)] = 0   # 황소장에서 숏 제거
```

---

## 코드 구조 (현재)

```
Mybox/
├── strategies/
│   └── bb_breakout.py      # ★ 최종 전략 (BBBreakout 클래스)
├── backtester/
│   └── engine.py           # 벡터화 백테스팅 엔진
├── trader/
│   ├── __init__.py         # LiveBot, BinanceFutures, init_db export
│   ├── binance_futures.py  # Binance Futures REST + WebSocket 클라이언트
│   ├── live_bot.py         # ★ 라이브 봇 본체
│   └── db.py               # SQLite 영속화 (position/trades/equity)
├── web/
│   ├── app.py              # FastAPI 대시보드 백엔드 (SSE 포함)
│   └── static/
│       └── index.html      # 라이브 봇 대시보드 UI
├── main.py                 # 봇 실행 진입점
├── .env.example            # 환경변수 템플릿
├── btc-bot.service         # systemd 서비스 파일
└── requirements.txt
```

---

## 이번 세션에서 구현한 것

### 핵심 버그 수정
- **`iloc[-1]` → `iloc[-2]`**: WebSocket 봉 마감 후 REST가 반환하는 마지막 봉은 항상 현재 진행 중인 미완성 봉. 신호 계산은 방금 닫힌 봉(`iloc[-2]`) 기준으로 해야 정확.

### trader/db.py (신규)
SQLite 영속화 모듈. 봇 재시작 시 포지션 복구에 핵심.
- `save_position(side, entry_price, sl_price, tp_price, quantity, entry_time)`
- `load_position()` → dict or None
- `clear_position()`
- `save_trade(...)` / `load_trades(limit)`
- `save_equity(value)` / `load_equity(limit)`
- DB 경로: `data/bot.db`

### trader/live_bot.py (전면 재작성)
1. **재시작 복구 로직** (`_recover_on_startup`):
   - 거래소 포지션 확인 → open orders 조회
   - SL/TP 주문 없으면 DB 저장값으로 재설정
   - DB에 포지션 없으면 수동 확인 요청 로그
2. **Binance 주문 제약 검증** (`_enter`):
   - `get_symbol_info()`로 minNotional, stepSize, minQty 동적 조회
   - stepSize 단위 절사 (`_floor_step`)
   - 조건 미달 시 스킵 + 로그
3. **진입 후 DB 저장**: `save_position()`으로 sl_price, tp_price, quantity 보존
4. **파라미터 전부 .env화**: SYMBOL, LEVERAGE, RISK_PCT, SL_ATR_MULT, TP_ATR_MULT 등

### trader/binance_futures.py (메서드 추가)
- `get_symbol_info(symbol)`: minNotional, stepSize, minQty 반환
- `get_open_orders(symbol)`: 미체결 주문 목록

### main.py (신규)
`.env` 읽어서 `LiveBot` 실행. 로그 파일(`bot.log`) + stdout 동시 출력.

### web/app.py (전면 재작성)
- LiveBot embed 없이 SQLite 읽기만
- `/api/status`, `/api/position`, `/api/trades`, `/api/equity`
- `/api/stream` SSE — 2초마다 포지션/요약 push

### web/static/index.html (전면 재작성)
- 실시간 포지션 박스 (SL/TP/수량/진입시각)
- 에쿼티 커브 (canvas, 그라디언트)
- 거래 히스토리 테이블
- SSE 연결 상태 표시

### btc-bot.service (신규)
systemd 서비스. `Restart=always`, `journalctl` 로깅.

### 삭제된 레거시
- `trader/live_trader.py` (현물 API 기반 구식)
- `trader/binance_client.py` (현물 API)

---

## 설계 결정사항 (웹 Claude와 합의)

| 항목 | 결정 |
|------|------|
| 신호 기준 봉 | `iloc[-2]` (닫힌 봉) |
| TP 기본값 | 3.0 (보수적), .env에서 변경 가능 |
| 재시작 복구 | DB의 sl_price/tp_price 그대로 사용 (현재 ATR 재계산 금지) |
| 레거시 삭제 순서 | 웹 재작성 완료 후 삭제 ✅ |
| 주문 제약 | exchange_info API 동적 조회 |
| 프로세스 구조 | LiveBot 독립 프로세스 + SQLite 공유 |
| 서비스 관리 | systemd (Debian) |

---

## 다음 세션이 해야 할 일

### 즉시 (테스트넷 API 키 발급 후)
```bash
# 1. Binance Futures 테스트넷에서 API 키 발급
#    https://testnet.binancefuture.com → 로그인 → API 발급

# 2. .env 설정
cp .env.example .env
# BINANCE_API_KEY, BINANCE_API_SECRET 입력
# TESTNET=true 유지

# 3. 의존성 설치
pip install -r requirements.txt

# 4. 봇 실행
python main.py

# 5. 대시보드 실행 (별도 터미널)
uvicorn web.app:app --host 0.0.0.0 --port 8000
```

### systemd 등록 (검증 후)
```bash
sudo cp btc-bot.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable btc-bot
sudo systemctl start btc-bot
sudo journalctl -u btc-bot -f
```

### 확인해야 할 것들
1. 테스트넷에서 레버리지 설정 정상 작동 여부
2. WebSocket 봉 마감 이벤트 수신 정상 여부
3. SL/TP 브라켓 주문 체결 확인
4. 봇 강제 종료 후 재시작 시 복구 로직 동작 확인
5. 대시보드 SSE 실시간 업데이트 확인

---

## 소통 채널
- **CHAT.md**: CLI Claude ↔ 웹 Claude 실시간 대화 파일
- **CLAUDE.md**: 인수인계 문서 (이 파일)

웹 Claude에게 확인이 필요하면 CHAT.md에 `[CLI]` 태그로 작성 후 push → 사용자가 웹 세션에서 확인 요청.

## Git
- 브랜치: `claude/bitcoin-backtesting-bot-qEVwm`
- 최신 커밋: Phase 1+2 완료 (87709f8)

## 주요 교훈 (이전 세션)
- **평균회귀 전략(BB 터치)은 실패**: 하락장에서 "싸다"고 롱 → 더 폭락
- **BB 돌파 전략이 성공**: 방향 확정 후 추세 동승
- **국면 필터 없으면 절반 이상 손실**: 2022 곰장에서 롱 신호가 전부 날아감
- **TP×2는 수수료도 못 벌음**: 최소 TP×3 이상 필요
- **dtype 버그**: pandas reindex 후 bool에 NaN 섞이면 `~regime`이 -2/-1 반환 → `.astype(bool)` 필수
