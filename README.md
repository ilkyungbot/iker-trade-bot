# Signal Bot v2.0

Bybit 선물 시장 기반 **포지션 관리 봇**. 자동 주문 없이, 사용자가 등록한 포지션을 실시간 모니터링하고 행동 제안을 Telegram으로 푸시합니다.

## 핵심 철학

> 3~7배 레버리지 스윙 트레이딩에서 생존율을 결정하는 건 "어떤 지표를 보는가"가 아니라 "리스크를 얼마나 통제하는가"이다.

- 손절가 없으면 포지션 등록 불가
- R:R 1.5 미만이면 경고 후 확인
- 일일 손실 5% 도달 시 자동 거래 중단
- 모든 알림에 "무슨 일 / 내 영향 / 행동 제안" 3단 구조

## 기능

### 포지션 등록 (8단계)

```
신규 포지션 → 코인 → 롱/숏 → 평단가 → 레버리지 → 마진(USDT)
  → 손절가 (ATR 가이드 제공) → 익절가 (R:R 가이드 제공) → 진입 논리
```

### 실시간 모니터링 (5분 주기)

| 이벤트 | 조건 | 쿨다운 |
|--------|------|--------|
| SL 접근 경고 | 손절가까지 3% 이내 | 30분 (critical: 10분) |
| 1차 익절 도달 | 1.5R 도달 | 1회만 |
| 트레일링 스탑 | 1차 익절 후 2 ATR 이탈 | 30분 |
| 시간 스탑 | 12시간 내 0.5R 미도달 | 30분 |
| OI/펀딩 이상 | 펀딩 ±0.05%, OI ±3% | 30분 |
| 레짐 변화 | ATR 비율 변동 | 변화 시에만 |

### 포트폴리오 제어

- 동시 포지션 최대 3개
- 일일 손실 한도 5% (계좌 잔고 기준)
- 월간 드로다운 20% 도달 시 봇 정지

### 시장 레짐 분류

| 레짐 | ATR(14)/ATR(50) | 봇 행동 |
|------|-----------------|---------|
| 고변동 추세 | > 1.3 | 넓은 스탑, 트레일링 활용 |
| 저변동 횡보 | < 0.7 | 진입 자제 경고, 포지션 50% 축소 |
| 전환기 | 0.7 ~ 1.3 | 포지션 사이즈 50% 축소 |

### Telegram 명령어

| 명령어 | 설명 |
|--------|------|
| `신규 포지션` | 8단계 포지션 등록 |
| `청산` | 포지션 청산 (PnL 계산 + 저널 기록) |
| `취소` | 등록 흐름 취소 |
| `현황` | 포지션 대시보드 + 시장 브리핑 |
| `상태` | 봇 시스템 상태 |
| `성과` | 주간 트레이딩 리포트 (레짐별 승률) |
| `BTC`, `ETH` 등 | 코인 심층 분석 |
| `도움말` | 명령어 목록 |

## 아키텍처

```
Layer 0: 매크로 필터      ← MarketRegimeClassifier (ATR 레짐 + BTC 200EMA)
Layer 1: 엣지 확인        ← EdgeDetector (펀딩 극단값 + OI 변화율)
Layer 2: 진입 트리거      ← TrendFollowing + FundingRate 전략 (참고 시그널)
Layer 3: 리스크 프레임    ← RiskCalculator (SL/TP 추천 + R:R 검증)
Layer 4: 출구 관리        ← ExitManager (분할 익절 + 트레일링 + 시간 스탑)
Layer 5: 포트폴리오 제어  ← PortfolioGuard (동시 3개 + 일일/월간 한도)
```

### 디렉토리 구조

```
src/
├── core/
│   ├── config.py              # 환경변수 설정
│   ├── types.py               # ManualPosition, Side, Signal 등
│   └── safety.py              # 안전 상수
├── data/
│   ├── collector.py           # Bybit API (캔들, 펀딩비, OI)
│   ├── validator.py           # 데이터 품질 검증
│   ├── features.py            # 기술 지표 (EMA, RSI, ATR, ADX 등)
│   └── storage.py             # SQLAlchemy 모델
├── strategy/
│   ├── market_regime.py       # ATR 레짐 분류 + BTC 200EMA
│   ├── edge_detector.py       # 펀딩 극단값 + OI 이상치
│   ├── position_monitor.py    # PositionMonitorV2 (통합 모니터)
│   ├── trend_following.py     # 추세추종 전략 (참고 시그널)
│   ├── funding_rate.py        # 펀딩레이트 역전 전략
│   └── pair_selector.py       # 페어 자동 선택
├── execution/
│   ├── risk_calculator.py     # SL/TP 추천, R:R 검증, 슬리피지 버퍼
│   ├── exit_manager.py        # 분할 익절 + 트레일링 + 시간 스탑
│   ├── portfolio_guard.py     # 포지션 수/손실 한도 제어
│   └── circuit_breaker.py     # 시그널 쿨다운 + API 에러 추적
├── conversation/
│   ├── position_manager.py    # 포지션 CRUD (SQLite, WAL)
│   ├── state_machine.py       # IDLE ↔ MONITORING 상태
│   └── signal_tracker.py      # 시그널 정확도 추적
├── review/
│   ├── reporter.py            # Telegram 메시지 포맷 (3단 구조)
│   ├── telegram_commands.py   # 대화 흐름 (8단계 등록, 청산, 현황)
│   └── trading_journal.py     # 자동 거래 기록 + 성과 리포트
└── main.py                    # SignalBot 오케스트레이터
```

## 설치

### 요구사항

- Python 3.12+
- Telegram Bot Token ([BotFather](https://t.me/BotFather)에서 생성)
- Bybit API Key (선택 — 없어도 시세 조회 가능)

### 로컬 실행

```bash
git clone https://github.com/ilkyungbot/iker-trade-bot.git
cd iker-trade-bot

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# .env 파일에 TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID 입력

python src/main.py
```

### Docker

```bash
cp .env.example .env
# .env 편집

docker compose up -d
```

## 환경변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `TELEGRAM_BOT_TOKEN` | (필수) | Telegram 봇 토큰 |
| `TELEGRAM_CHAT_ID` | (필수) | 알림 받을 채팅 ID |
| `BYBIT_API_KEY` | (선택) | Bybit API 키 |
| `BYBIT_API_SECRET` | (선택) | Bybit API 시크릿 |
| `DATABASE_URL` | `sqlite:///signal_bot.db` | DB 경로 |
| `SIGNAL_COOLDOWN_MINUTES` | `30` | 시그널 간 최소 간격 |
| `PRIMARY_INTERVAL` | `240` | 기본 봉 (4시간=240분) |
| `MAX_PAIRS` | `5` | 관찰 페어 수 |
| `MIN_SIGNAL_QUALITY` | `moderate` | 최소 시그널 품질 |

## 스케줄러

| 작업 | 주기 | 설명 |
|------|------|------|
| 시그널 사이클 | 4시간 (0/4/8/12/16/20시) | 참고용 시그널 자동 발송 |
| 시장 브리핑 | 매시 정각 | Top 10 코인 + 스코어링 |
| 포지션 모니터링 | 5분 | 활성 포지션 이벤트 감지 |
| 시그널 결과 추적 | 30분 | 과거 시그널 4h/8h/24h 가격 |
| 일간 리포트 | 매일 00:05 UTC | 시그널 정확도 |
| 주간 리포트 | 월요일 00:10 UTC | 시그널 정확도 |
| 일일 가드 리셋 | 매일 00:00 UTC | 일일 손실 카운터 초기화 |
| 월간 가드 리셋 | 매월 1일 00:00 UTC | 월간 드로다운 카운터 초기화 |

## 테스트

```bash
# 전체 테스트
python -m pytest tests/ -v

# 특정 모듈
python -m pytest tests/execution/ -v
python -m pytest tests/strategy/test_position_monitor_v2.py -v
```

## 알림 예시

### 포지션 등록

```
✅ 포지션 등록 완료

종목: BTCUSDT
방향: 롱
평단: 67,500.00
레버리지: 5x
손절: 66,000.00
익절: 70,000.00
R:R = 1:1.67

모니터링을 시작합니다. 청산 시 '청산'을 입력하세요.
```

### 1차 익절 도달

```
💰 1차 익절 도달 | BTCUSDT

[무슨 일이 일어났는가]
1차 목표(1.5R) 도달! 현재 PnL +11.1%

[내 포지션에 어떤 영향인가]
롱 5x | 마진 500 USDT
예상 손익: +55.6 USDT

[어떻게 하면 되는가]
포지션 50% 익절 + 손절가를 본전으로 이동 권장
```

### 현황 대시보드

```
📊 내 포지션 현황

1. BTCUSDT 롱 5x
   마진: 500 USDT
   평단 67,500.00 → 현재 69,200.00
   PnL: ✅ +12.6% (+63.0 USDT)
   SL 66,000.00 (-4.6%) | TP 70,000.00 (+1.2%)
   보유 8h 23m
   논리: "200EMA 반등 + 펀딩 음전환"

────────────────────
총 PnL: +63.0 USDT
활성 포지션: 1/3
```

## License

Private repository.
