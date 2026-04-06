# Bitcoin Futures Auto-Trading Bot - 프로젝트 인수인계

## 현재 상태: **라이브 봇 테스트넷 실행 중 + 웹 대시보드 운영 중**

---

## 확정 전략: BB Squeeze Breakout 1h

### 파라미터
```python
BBBreakout(squeeze_pct=15, trend_filter=True, trend_ema=20)
SL×1.5 ATR / TP×3.0 ATR / 레버리지 10x / 리스크 1%
200일 MA 국면 필터 필수
```

### 백테스트 성과 (2022~2025Q1, $10,000, 10x)
| 기간 | 수익률 | 승률 | PF | MDD |
|------|--------|------|----|-----|
| 2022 곰장 | +4.4% | 38.4% | 1.08 | -8.7% |
| 2023 회복 | +16.1% | 46.3% | 1.33 | -9.8% |
| 2024 불장 | +17.8% | — | — | — |
| **전체** | **+44.3%** | 43% | 1.26 | -11.4% |

### 타임프레임별 성과 (2022~2025Q1)
| 봉 | 수익률 | 비고 |
|----|--------|------|
| **1h** | **+44.3%** | ★ 최적 |
| 4h | +16.4% | 거래 적음 |
| 2h | +11.2% | |
| 30m | -53.5% | ✗ 노이즈 과다 |
| 15m | -92.1% | ✗ 완전 실패 |

→ **1h가 압도적 최적. 단기 분봉은 BBBreakout 전략 자체가 실패.**

---

## 코드 구조

```
Mybox/
├── strategies/
│   └── bb_breakout.py      # ★ 최종 전략
├── backtester/
│   └── engine.py
├── trader/
│   ├── binance_futures.py  # Futures REST + WebSocket
│   ├── live_bot.py         # ★ 라이브 봇 (systemd로 실행 중)
│   └── db.py               # SQLite 영속화
├── web/
│   ├── app.py              # FastAPI 대시보드 (systemd로 실행 중)
│   └── static/index.html   # 대시보드 UI
├── main.py                 # 봇 진입점
├── .env                    # API 키 및 파라미터 (테스트넷)
├── btc-bot.service         # systemd 봇 서비스
├── btc-dashboard.service   # systemd 대시보드 서비스
├── optimize_bbbreakout.py  # BBBreakout 그리드 서치
└── analyze_periods.py      # 기간/분봉 분석
```

---

## 현재 실행 환경

- 서버: Azure VM, IP `20.40.97.78`
- 봇: `sudo systemctl status btc-bot` (TESTNET 모드)
- 대시보드: `http://20.40.97.78` (포트 80)
- 로그: `sudo journalctl -u btc-bot -f`
- DB: `data/bot.db` (SQLite)
- .env: TESTNET=true, TP=3.0, SL=1.5, LEVERAGE=10

---

## 설계 결정사항

| 항목 | 결정 |
|------|------|
| 신호 기준 봉 | `iloc[-2]` (닫힌 봉) |
| TP 기본값 | 3.0 (보수적) |
| 재시작 복구 | DB의 sl/tp/qty 그대로 사용 |
| 프로세스 구조 | 봇·웹 분리, SQLite 공유 |
| 서비스 관리 | systemd (Restart=always) |

---

## 그리드 서치 결과 (1944개 조합)

수익 871개 / 손실 1073개

**상위 조합:**
| 순위 | 파라미터 | 수익 | MDD |
|------|----------|------|-----|
| ★1 | sq30/lb200/ema10, SL1.5/TP5.0 | +60.4% | -20.4% |
| ★2 | sq15/lb100/ema20, SL1.5/TP4.0 | +57.7% | -13.2% |
| ★8 | sq15/lb100/ema10, SL1.5/TP4.0 | +54.0% | -14.1% |
| ★12 | sq15/lb100/ema10, SL1.5/TP3.0 | +51.8% | -11.4% |
| **현재봇** | sq15/lb100/ema20, SL1.5/TP3.0 | +44.3% | -11.4% |

**★12 주목**: 2025Q1에도 +4.5% (유일하게 최근 플러스), MDD 안정적

---

## 다음 세션이 할 일

### 최우선: 초단기 분봉 전략 개발
- BBBreakout은 15m/30m에서 완전히 실패
- **새 전략을 처음부터 설계**해야 함
- 후보 접근법:
  - **VWAP 이탈 + 되돌림** (15m/5m)
  - **오더플로우 기반 모멘텀** (RSI divergence + volume spike)
  - **EMA 리본 크로스** (3/5/8/13 EMA, 15m)
- 새 전략은 기존 BBBreakout 봇과 **독립적으로** 운영
- 백테스트 → 파라미터 최적화 → 봇 추가 순서로 진행

### 선택: 현재 봇 파라미터 변경 검토
- ★12(sq15/ema10/TP3.0)로 변경 시 2025Q1 방어력 향상

---

## Git
- 브랜치: `claude/bitcoin-backtesting-bot-qEVwm`
- 소통 채널: `CHAT.md` (CLI↔웹 Claude)
- GitHub 토큰: 세션 종료 후 revoke 필요 (대화에 노출됨)

## 주요 교훈
- BBBreakout은 1h에서만 유효, 단기 분봉 불가
- 국면 필터(200일 MA) 없으면 절반 이상 손실
- TP×2 이하는 수수료도 못 벌음
- dtype 버그: `.astype(bool)` 필수
