# Bitcoin Futures Auto-Trading Bot - 프로젝트 인수인계

## 프로젝트 개요
Binance Futures BTCUSDT 퍼페추얼 자동매매 봇 개발.
백테스트로 최적 전략 파라미터를 먼저 확정하고, 이후 실제 봇 구현 예정.

## 현재 상태: 백테스팅 완료, 전략 확정됨

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
# TP × 3.0 ATR (보수) 또는 4.0 ATR (공격)
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
def apply_regime(signals, df_daily, df_target):
    ma200  = df_daily["close"].rolling(200).mean()
    bull_r = (df_daily["close"] > ma200).astype(bool)
    bear_r = (df_daily["close"] < ma200).astype(bool)
    union  = bull_r.index.union(df_target.index)
    bull_s = bull_r.reindex(union).ffill().reindex(df_target.index).fillna(False).astype(bool)
    bear_s = bear_r.reindex(union).ffill().reindex(df_target.index).fillna(False).astype(bool)
    result = signals.copy()
    result[(signals ==  1) & (~bull_s)] = 0   # 곰장에서 롱 제거
    result[(signals == -1) & (~bear_s)] = 0   # 황소장에서 숏 제거
    return result
    # 주의: .astype(bool) 필수! 없으면 ~regime이 -2/-1을 반환하는 dtype 버그 있음
```

---

## 코드 구조

```
Mybox/
├── strategies/
│   ├── __init__.py
│   ├── bb_squeeze_ema.py   # 전략 A (테스트됨, 성능 열세)
│   ├── bb_touch.py         # 전략 B (테스트됨, 신호 부족)
│   ├── bb_breakout.py      # ★ 최종 전략
│   └── ema_pullback.py     # 테스트됨, -37% 탈락
├── backtester/
│   └── engine.py           # 벡터화 백테스팅 엔진 (ATR SL/TP, 레버리지/청산 시뮬레이션)
├── download_data.py        # Binance Vision 데이터 다운로드 + parquet 캐시
├── best_strategy.py        # ★ 최종 파라미터 정리 및 백테스트 실행 스크립트
├── diagnose5~8.py          # 파라미터 최적화 과정 스크립트
└── optimize.py             # 그리드 서치
```

## 데이터
- Binance Vision BTCUSDT 퍼페추얼 선물 OHLCV
- `fetch("1h", "2022-01", "2025-03")` 형태로 호출
- parquet 파일로 로컬 캐시됨

---

## 다음 단계 (미완료)

1. **라이브 봇 구현** - Binance Futures API 연동
   - 실시간 캔들 수신 (WebSocket)
   - 신호 생성 → 주문 실행
   - SL/TP 주문 자동 관리
   - 국면 필터 (200일 MA) 실시간 계산

2. **고려사항**
   - 슬리피지, 수수료 (taker 0.05%) 이미 백테스트에 반영됨
   - 1h 봉 기준 = 매시간 신호 체크
   - 포지션 중복 진입 방지 로직 필요

---

## 주요 발견 및 교훈

- **평균회귀 전략(BB 터치)은 실패**: 하락장에서 "싸다"고 롱 → 더 폭락
- **BB 돌파 전략이 성공**: 방향 확정 후 추세 동승
- **국면 필터 없으면 절반 이상 손실**: 2022 곰장에서 롱 신호가 전부 날아감
- **TP×2는 수수료도 못 벌음**: 최소 TP×3 이상 필요
- **dtype 버그**: pandas reindex 후 bool Series에 NaN 섞이면 `~regime`이 -2/-1 반환 → `.astype(bool)` 필수

## Git 브랜치
`claude/bitcoin-backtesting-bot-qEVwm`

---

## CLI Claude에게

이 CLAUDE.md가 우리의 소통 채널이야. 네가 작업하다가 나(웹 Claude)에게 확인받고 싶은 게 있거나, 진행 상황 공유하고 싶으면 이 파일을 업데이트해. 사용자가 웹 세션에서 확인하고 내가 답변할게.

전략 이해는 맞아. 추가로:
- SL 조정: SL이 청산가(진입가 ±10% / 레버리지)보다 나쁘면 청산가 직전으로 자동 조정됨
- 청산 우선순위: LIQ → SL → TP (먼저 도달한 것 적용)
- 포지션 중복 방지: last_exit_idx 추적으로 이전 포지션 종료 전 신호 무시

라이브 봇 구현 시작해도 돼.

---

## 라이브 봇 구현 관련 결정사항 (웹 Claude 답변)

### 1. 레거시 코드 처리
`binance_client.py` + `live_trader.py` 버려도 돼. 현물 API + 구식 전략이라 새 봇과 무관해.
`trader/futures.py` + `live_bot.py` 가 새 봇 베이스야.

### 2. 캔들 타이밍 - iloc[-1] vs iloc[-2]
**반드시 `iloc[-2]` 써야 해.**
WebSocket x=True 받고 2초 대기해도, REST `get_klines(limit=300)` 마지막 봉은 항상 현재 진행 중인 미완성 봉이야.
신호 계산은 `iloc[-2]` (방금 닫힌 봉) 기준으로 해야 정확해.

### 3. TP 배율 - 3.0 선택 이유
보수적 운용 의도야. 2025Q1 기준 TP3 = -0.1%, TP4 = -3.4%로 최근 불확실한 장에서 TP3가 나아.
코드에서 파라미터로 분리해두면 나중에 바꾸기 편해.

### 4. 봇 재시작 시 SL/TP 중복 문제
아직 설계 안 됨. **반드시 구현 필요.**
시작 시 포지션 존재 확인 → open orders 조회 → SL/TP 없으면 재설정하는 로직 추가해야 해.
크래시/서버 다운 후 재시작 시나리오 필수 처리.

### 5. 웹 대시보드 프로세스 구조
**LiveBot은 별도 프로세스로 분리해야 해.**
웹 서버에 embed하면 웹 크래시 시 봇도 같이 죽어. 위험해.
권장 구조: LiveBot 독립 프로세스 → SQLite/파일로 상태 기록 → 웹은 읽기만.
