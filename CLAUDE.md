# 업비트 퀀트 자동매매 봇

## 전체 기획서
docs/upbit_quant_v9.md 참조.
코드 작성 전 해당 Phase 섹션 먼저 확인할 것.

## 개발자 배경
- CS 석사 과정 / Python·PyTorch·LSTM 숙련
- 기초 ML/Python 설명 생략
- 프로젝트 특화 구현 + 실전 고려사항 집중

## 현재 진행 단계
Phase A 진행 중.
Phase A → B → C → D 순서 엄수.

## 핵심 원칙 (코드 작성 시 반드시 준수)
1. FinBERT 등 무거운 연산은 ProcessPoolExecutor로 메인 루프와 분리
2. REST API는 초기 히스토리 로딩 1회만 — 이후 WebSocket CandleBuilder 사용
3. 서킷브레이커는 모든 레이어보다 최우선 (circuit_breaker.py 항상 import)
4. 일봉 피처는 df_daily.shift(1) 적용 필수 (Lookahead Bias 방지)
5. 주문은 반드시 SmartOrderRouter 경유
6. 부분체결은 PartialFillHandler 처리 (이중 주문 방지)
7. trade_count < 200 이면 DRY_RUN=True 강제
8. 모든 함수에 type hint 작성
9. 단위 테스트 파일을 항상 같이 작성 (test_파일명.py)

## 기술 스택
Python 3.11+
pyupbit / pandas / numpy / pandas-ta
PyTorch (device=cpu 강제) / XGBoost / LightGBM
SQLite / APScheduler / python-telegram-bot 20.x
asyncio / concurrent.futures

## 하드웨어 제약
Mac Mini M4 16GB RAM
device=cpu 강제 (MPS 사용 금지)
배치사이즈 최대 32
코인당 인메모리 캔들 최대 500개

## 폴더 구조
upbit_bot/
├── docs/                  # 기획서
├── data/
│   ├── collector.py       # WebSocket + 동적 페어리스트
│   ├── candle_builder.py  # WebSocket 캔들 합성
│   ├── cache.py           # SQLite 캐시
│   └── quality.py         # DataQualityChecker 7단계
├── layers/
│   ├── layer1_filter.py
│   └── layer2_ensemble.py
├── strategies/
│   ├── selector.py
│   ├── trend.py
│   ├── grid.py
│   ├── dca.py
│   └── decay_monitor.py
├── execution/
│   ├── engine.py          # 메인 루프 (APScheduler)
│   ├── order.py           # SmartOrderRouter + PartialFillHandler
│   └── paper_trading.py
├── risk/
│   ├── circuit_breaker.py
│   ├── kelly.py
│   └── trailing_stop.py
├── backtest/
│   ├── walk_forward.py
│   ├── lookahead.py
│   └── monte_carlo.py
├── monitoring/
│   ├── telegram_bot.py
│   ├── storage_manager.py
│   └── dashboard.py
├── models/                # 체크포인트
├── utils/
│   ├── logger.py
│   └── helpers.py
├── config.yaml
├── .env                   # API 키 (git 제외)
├── CLAUDE.md
└── main.py