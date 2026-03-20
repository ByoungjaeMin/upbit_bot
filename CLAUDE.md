# 업비트 퀀트 자동매매 봇
## 전체 기획서
docs/upbit_quant_v9.md 참조.
코드 작성 전 해당 Phase 섹션 먼저 확인할 것.

## 개발자 배경
- CS 석사 과정 / Python·PyTorch·LSTM 숙련
- 기초 ML/Python 설명 생략
- 프로젝트 특화 구현 + 실전 고려사항 집중

## 현재 진행 단계
Phase A 완료. Phase B 코드 완성 + 전체 코드리뷰 완료. 맥미니 도착 후 데이터 수집 시작.
Phase A → B → C → D 순서 엄수.

---

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

---

## 기술 스택

Python 3.11+
pyupbit / pandas / numpy / pandas-ta
PyTorch (device=cpu 강제) / XGBoost / LightGBM
SQLite / APScheduler / python-telegram-bot 20.x
asyncio / concurrent.futures

---

## 하드웨어 제약

Mac Mini M4 16GB RAM
device=cpu 강제 (MPS 사용 금지)
배치사이즈 최대 32
코인당 인메모리 캔들 최대 500개

---

## SQLite 스키마 (data/bot.db 주요 테이블)

코드 작성 시 테이블·컬럼명 참조용. 실제 코드와 다를 경우 아래를 먼저 수정할 것.

```sql
-- 거래 기록 (성과 분석·세금 기준)
trades(id, coin, strategy, side, entry_price, exit_price, amount,
       pnl, pnl_pct, fee, slippage, reason, dry_run, timestamp)

-- 캔들 데이터 (타임프레임별 분리)
candles_5m(coin, timestamp, open, high, low, close, volume)
candles_1h(coin, timestamp, open, high, low, close, volume)
candles_1d(coin, timestamp, open, high, low, close, volume)

-- ML 예측 결과
predictions(id, coin, timestamp, ensemble_prob, signal,
            xgb_prob, lgbm_prob, lstm_prob, gru_prob, model_version)

-- 서킷브레이커 이력
circuit_breaker_log(id, level, reason, triggered_at, resolved_at, action_taken)

-- 생존편향 처리용 스냅샷 (매일 00:05 수집)
coin_history(id, snapshot_date, coin, volume_24h_krw, rank,
    market_cap_krw, included_in_pairlist)

-- 전략 성과 추적 (StrategyDecayMonitor용)
strategy_log(id, strategy_type, pnl_pct, pnl, timestamp)
strategy_decay_log(id, week_start, strategy_type, rolling_sharpe, win_rate,
                   profit_loss_ratio, trade_count, is_dormant, dormant_since, revival_date)

-- 백테스트 결과
backtest_results(id, run_id, strategy, sharpe, max_drawdown,
                 win_rate, oos_sharpe, p_value, lookahead_clean,
                 params_json, created_at)

-- 페이퍼 트레이딩 비교
paper_trades(id, coin, strategy, signal_match_rate, price_deviation, timing_slippage, timestamp)

-- 디스크·스토리지 모니터링
storage_log(id, db_size_mb, disk_free_gb, checked_at)
```

> 실제 스키마 확인: `data/cache.py` — 불일치 발견 시 이 섹션을 업데이트할 것

---

## 핵심 모듈 인터페이스

모듈 간 연동 코드 작성 시 참조. 실제 시그니처와 다를 경우 이 섹션을 먼저 수정할 것.

```python
# risk/circuit_breaker.py
CircuitBreaker.is_buy_blocked() -> bool
CircuitBreaker.is_sell_blocked() -> bool
CircuitBreaker.is_all_blocked() -> boolCircuitBreaker.check() -> bool
# False = 매수 전면 차단. 모든 진입 판단 전 최우선 호출 필수.
CircuitBreaker.record_api_error() -> None
CircuitBreaker.get_level() -> int  # 0=정상, 1~5=차단 수준

# execution/order.py
SmartOrderRouter.execute(coin: str, side: str, amount: float) -> OrderResult
# side: "buy" | "sell". 직접 pyupbit.buy_limit() 호출 금지 — 반드시 이 경유.
PartialFillHandler.handle(order_id: str) -> FillResult
# 부분체결 감지 → 잔여 수량 추적 → 이중 주문 방지.

# layers/layer1_filter.py
Layer1MarketFilter.check(ms: MarketState, coin: str) -> FilterResult
# FilterResult.tradeable=False 시 해당 코인 진입 스킵.

# layers/layer2_ensemble.py
Layer2Ensemble.predict(ms: MarketState) -> EnsemblePrediction
# EnsemblePrediction.prob: 0.0~1.0 매수 확률. 임계값 0.62 이상 시 진입 신호.
# Phase B 이전: _PHASE_B_ACTIVE=False → XGBoost+LightGBM만 사용.

# risk/kelly.py
KellySizer.compute(coin: str, win_rate: float, profit_loss_ratio: float, ...) -> RiskBudget
# RiskBudget.fraction: 0.0~0.5 포지션 비율. 0 이하 반환 시 포지션 진입 금지.

# risk/trailing_stop.py
TrailingStopManager.update(coin: str, current_price: float, atr: float) -> bool
# True 반환 시 즉시 시장가 청산.

# data/collector.py
async UpbitDataCollector.get_market_state(coin: str) -> MarketState | None
# 반환 MarketState: 35개 피처 포함, 일봉 피처 shift(1) 적용 완료.
# 호출 전 circuit_breaker.check() 통과 확인 필수.
```

---

## 디버깅 원칙

### 에러 처리 규칙
- 에러는 숨기지 말고 즉시 터뜨릴 것
- fallback은 의도적인 것만 허용 — 반드시 주석으로 이유 명시
- 서킷브레이커·주문 관련 코드는 예외 절대 삼키지 말 것
- 환경변수 누락 시 기본값 대신 즉시 종료
- `except Exception: pass` / `except Exception: continue` / `except Exception: return None` 패턴 금지

### 코드 수정 워크플로우
- 수정 전 반드시 파일명 + 줄 번호 + 변경 내용 먼저 보고할 것
- 멀쩡한 코드는 절대 건드리지 말 것
- 한 번에 하나의 파일(또는 하나의 카테고리)만 수정할 것
- 수정 후 pytest로 기존 테스트 깨진 것 없는지 확인할 것

### 감사(audit) 우선 원칙
- 누락 확인 요청 시: 목록만 정리, 코드는 짜지 말 것
- 누락 목록 확인 후 항목별로 하나씩 요청할 것
- "빠진 거 다 구현해" 방식 금지 (기존 코드 덮어씀·컨텍스트 초과 위험)

---

## 안티패턴 감지 목록
코드 리뷰 또는 감사 시 아래 패턴을 우선 탐색할 것.

### 에러 처리 문제
- `except Exception: pass` — 예외 완전 무시
- `except Exception: continue` — 예외 삼키고 진행
- `except Exception: return None` — 조용한 실패
- 예외 잡고 로그만 찍고 진행 (에러 상태 전파 없음)
- except 범위가 너무 넓은 것
- finally 블록 없이 리소스 열고 닫는 코드

### 무한 루프 / 재시도 문제
- while True에 종료 조건 없는 것
- 재시도 횟수 제한 없는 것
- 재시도 간격(backoff) 없는 것
- WebSocket 재연결 무한 루프

### None / 타입 안전성 문제
- None 체크 없이 바로 속성 접근
- 딕셔너리 키 확인 없이 바로 접근
- 타입 힌트 없는 함수
- 반환값 무시하는 함수 호출

### 비동기 문제
- async 함수 내 `time.sleep()` → `asyncio.sleep()` 써야 함
- async 함수 내 `requests.get()` → aiohttp 써야 함
- await 없이 코루틴 호출
- run_in_executor 없이 CPU 집약 작업 메인 루프에서 실행

### LLM 특유의 설계 문제
- 쓸데없는 fallback (오류 시 기본값으로 조용히 대체)
- 과잉 방어 코드 (불필요한 try-except 중첩)
- 일관성 없는 에러 처리
- 죽어야 할 코드를 살려놓는 패턴

### 퀀트 봇 특화 문제
- 서킷브레이커 우회 가능한 경로
- DRY_RUN=True인데 실제 주문 나가는 경로
- 일봉 피처 shift(1) 누락 (Lookahead Bias)
- 주문 실패 시 조용히 넘어가는 코드
- Kelly 계산 결과 음수/0 처리 없는 것
- API 레이트 리밋 초과 처리 없는 것

---

## 감사용 프롬프트 템플릿

### 전체 누락 항목 감사
```
~/quant 폴더 전체 파일 구조를 확인하고,
docs/upbit_quant_v9.md의 Phase 1~9 요구사항과 대조해서

1. 구현된 항목
2. 누락된 항목
3. 파일은 있지만 기획서 스펙과 다른 항목

이 세 가지를 목록으로만 정리해줘. 코드는 아직 짜지 마.
```

### 안티패턴 감지 감사
```
~/quant/upbit_bot 전체 코드를 분석해서
아래 패턴을 찾아 목록으로만 정리해줘. 코드 수정은 하지 마.

1. 삼켜진 예외 (except Exception: pass)
2. 쓸데없는 fallback (오류 시 기본값 대체)
3. 죽어야 할 코드를 살려놓는 패턴
4. 몰래 들어간 기본값

파일명과 줄 번호까지 같이 알려줘.
```

### 스키마·인터페이스 불일치 확인 감사
```
CLAUDE.md의 "SQLite 스키마"와 "핵심 모듈 인터페이스" 섹션을
실제 코드와 대조해줘.

1. 스키마가 다른 테이블 (컬럼명·타입 불일치)
2. 시그니처가 다른 함수 (파라미터·반환타입 불일치)
3. CLAUDE.md에 없는 주요 테이블·함수

목록만 정리해줘. 코드 수정하지 마.
```

### 수정 요청 시 공통 원칙 (모든 프롬프트 앞에 붙이기)
```
수정 원칙:
- 한 번에 하나의 카테고리만 수정할 것
- 수정 전 파일명 + 줄 번호 + 변경 내용 먼저 보고할 것
- 멀쩡한 코드는 절대 건드리지 말 것
- 수정 후 pytest로 기존 테스트 깨진 것 없는지 확인할 것
```

---

## 폴더 구조

```
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
├── scripts/
│   └── coin_history_collector.py  # 생존편향 처리용 스냅샷 수집
├── utils/
│   ├── logger.py
│   └── helpers.py
├── config.yaml
├── .env                   # API 키 (git 제외)
├── CLAUDE.md
└── main.py
```

---

## Phase별 완료 기준

### Phase A (완료)
- pytest 983개 0 failed
- mypy 타입 에러 0건
- DRY_RUN 기동/종료 EXIT 0
- 서킷브레이커 우회 없음
- DRY_RUN 우회 없음
- 비의도적 예외 삼킴 없음
- None 체크 누락 없음

### Phase B 완료 기준 (맥미니 도착 후)
- ✅ Walk-Forward 코드 완성 (backtest/walk_forward.py)
- ✅ Monte Carlo 코드 완성 (backtest/monte_carlo.py)
- ✅ Lookahead Bias 검사 코드 완성 (backtest/lookahead.py)
- [ ] Walk-Forward 샤프비율 > 1.5 (데이터 수집 후 검증)
- [ ] 최대낙폭 < 20%
- [ ] 승률 > 55%
- [ ] Monte Carlo p-value < 0.05
- [ ] Lookahead Bias 오염 피처 0개

### 실전 전환 7가지 체크리스트 (모두 통과 전 실거래 금지)
- [ ] 샤프비율 > 1.5
- [ ] 최대낙폭 < 20%
- [ ] 승률 > 55%
- [ ] 하락장 낙폭 < 10%
- [ ] Lookahead Bias 오염 피처 0개
- [ ] Monte Carlo p-value < 0.05
- [ ] DRY_RUN 48시간 정상 확인