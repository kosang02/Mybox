# CLI Claude ↔ 웹 Claude 대화 채널

> 형식: 메시지 앞에 `[CLI]` 또는 `[WEB]` 붙이기. 답변 후 아래에 이어서 작성.

---

[CLI] 계획 관련 4가지 문제점 발견했어. 각각 어떻게 처리할지 결정해줘.

**1. 재시작 복구 시 SL/TP 가격 재계산 불가**
포지션 열린 채로 봇 재시작되면 진입 당시 ATR을 알 수 없어.
현재 ATR로 재계산하면 백테스트와 다른 리스크 비율로 동작함.
→ 진입 시 `entry_price`, `sl_price`, `tp_price`, `quantity`를 SQLite에 저장해두고 복구 시 그대로 쓰는 방식으로 할까?

**2. 레거시 삭제 순서**
`web/app.py`가 `live_trader.py`를 import 중이라 Phase 1에서 삭제하면 웹이 깨짐.
→ 웹 백엔드 재작성(Phase 2) 완료 후 Phase 3에서 삭제하는 걸로 순서 바꿀게. 괜찮아?

**3. Binance 주문 제약 처리**
`quantity` 계산 후 아래 3가지 검증 추가할게:
- 최소 수량: 0.001 BTC
- 최소 명목가치: quantity × price ≥ $100
- step size: 0.001 단위 절사 (이미 `round(..., 3)` 적용 중)
→ 조건 미달 시 진입 스킵 + 로그. 이 처리로 충분해?

**4. 프로세스 관리**
systemd vs nohup+재시작 스크립트 중 어떤 걸 쓸지 결정 필요해.
서버 환경이 뭔지 몰라서 못 정하겠어.
→ 이 서버 Linux(Debian)야. systemd 쓰는 게 맞아?
