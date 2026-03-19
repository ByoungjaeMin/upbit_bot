# 업비트 AI 퀀트 자동매매 봇 — 전체 구현 프롬프트 설계서
> Master Prompt + Phase 1~9 완전판 (전략 다양성 + 데이터 품질 개선 반영)
> **v9 업데이트: 전략 포화 방어 + 실전 검증 강화 + 동적 페어리스트 + 엔지니어링 안정성 + 엣지 강화 + 주문 무결성 + 진입 트리거 명확화 + 실전 운용 세부화 + 신호 품질 강화**

---

## v3 개선안 전체 요약

### [v2] 전략 포화 방어 5개
| # | 개선안 | 적용 위치 | Phase | 핵심 효과 |
|---|---|---|---|---|
| 1 | **StrategyDecayMonitor** | strategies/decay_monitor.py | C | 전략 포화 자동 감지 + 휴면 전환 |
| 2 | **동적 전략 가중치** | strategies/selector.py | C | 성과 기반 자본 배분 자동 조정 |
| 3 | **김치프리미엄 피처** | data/collector.py + 피처 +2 | A(수집) / B(활용) | 업비트 특화 엣지, 해외봇과 비경쟁 |
| 4 | **오더북 불균형(OBI) 피처** | data/collector.py + 피처 +1 | A(수집) / B(활용) | 공개 기술지표보다 포화 저항성 높음 |
| 5 | **실행 타이밍 노이즈** | execution/engine.py | A | front-running 방어, 즉시 적용 가능 |

### [v3] 실전 검증 강화 + 구조 개선 6개
| # | 개선안 | 적용 위치 | Phase | 핵심 효과 |
|---|---|---|---|---|
| 6 | **Lookahead Bias 검증** | backtest/lookahead.py | B | 백테스트 미래참조 오염 방지 |
| 7 | **생존 편향 처리** | backtest/walk_forward.py | B | 시점별 실제 코인 목록으로 백테스트 |
| 8 | **일봉 지표 타임 원칙** | data/collector.py | A | 백테스트↔실거래 지표 괴리 제거 |
| 9 | **Monte Carlo 검증** | backtest/monte_carlo.py | B | 전략 엣지와 운을 통계적으로 분리 |
| 10 | **동적 페어리스트** | config.yaml + collector.py | A | 10개 고정→상위 30개 동적, 기회 3배 확대 |
| 11 | **Phase D 피처 자동생성 방향** | 설계 문서 | D | FreqAI식 대규모 피처 엔지니어링 예고 |

### [v4] 엔지니어링 안정성 2개
| # | 개선안 | 적용 위치 | Phase | 핵심 효과 |
|---|---|---|---|---|
| 12 | **FinBERT 프로세스 격리** | data/collector.py + execution/engine.py | A | asyncio 블로킹 → 서킷브레이커 지연 방지 |
| 13 | **WebSocket 캔들 합성** | data/candle_builder.py (신규) | A | REST 90회/루프 → 0회, IP 밴 위험 제거 |

### [v5] 엣지 강화 2개
| # | 개선안 | 적용 위치 | Phase | 핵심 효과 |
|---|---|---|---|---|
| 14 | **체결 강도 피처 3개** | data/candle_builder.py + 피처 +3 | A(수집) / B(활용) | WebSocket 체결 데이터 → 실시간 마이크로스트럭처 엣지 |
| 15 | **조건부 지정가/시장가 선택** | execution/order.py | A | 슬리피지 최소화 + maker 수수료 절감 |

### [v6] 실전 주문 안정성 1개
| # | 개선안 | 적용 위치 | Phase | 핵심 효과 |
|---|---|---|---|---|
| 16 | **부분 체결(Partial Fill) 처리** | execution/order.py (PartialFillHandler) | A | 이중 주문 방지 + Kelly 포지션 사이즈 정확성 보장 |

### [v7] 주문 무결성 + 진입 명확화 2개
| # | 개선안 | 적용 위치 | Phase | 핵심 효과 |
|---|---|---|---|---|
| 17 | **DELETE 레이스 컨디션 예외 처리** | execution/order.py | A | 취소 찰나 100% 체결 시 봇 크래시 방지 |
| 18 | **최종 진입 트리거 6조건 명시** | execution/engine.py (메인 루프 13번) | A | "언제 산다"를 코드 수준으로 명확화 |

### [v9] 신호 품질 강화 2개
| # | 개선안 | 적용 위치 | Phase | 핵심 효과 |
|---|---|---|---|---|
| 24 | **앙상블 임계값 0.55→0.62** | layer2_ensemble.py + engine.py | A | 노이즈 신호 차단, 승률·손익비 개선 |
| 25 | **일 거래 상한 5→10회 + Optuna ensemble_threshold** | engine.py + hyperopt.py | A/B | 좋은 신호 기회 손실 방지 + 최적값 자동 탐색 |
| # | 개선안 | 적용 위치 | Phase | 핵심 효과 |
|---|---|---|---|---|
| 19 | **청산 주문 실행 방식 명세** | execution/engine.py (포지션 모니터) | A | 손절 중 슬리피지 폭발 방지 + 상황별 분기 |
| 20 | **그리드 최소 자본 조건** | strategies/selector.py | A | $300 미만 시 GRID→HOLD 자동 대체 |
| 21 | **페이퍼↔실거래 정량 비교 3지표** | execution/paper_trading.py | A | 신호일치율·체결가괴리·타이밍슬리피지 정량 추적 |
| 22 | **재진입 조건 강화** | execution/engine.py | A | 손절 후 동일 추세 재진입 방지 |
| 23 | **콜드스타트 DRY_RUN 강제** | execution/engine.py | A | 200건 미만 구간 실거래 원천 차단 |

> **[신규]** 태그로 문서 전체에 표시됨. 피처 총수: 32개 → 35개 → **38개**, RL State: 19 → 20, 스캔 대상: 10개 → 최대 30개
> **[v9 핵심]** 앙상블 임계값 0.55→0.62(초기값) / 일 거래 상한 5→10회 / Optuna에 ensemble_threshold(0.55~0.75) 추가해 실전 최적값 자동 탐색

---


| 항목 | 내용 |
|---|---|
| 플랫폼 | 업비트(Upbit) 현물 거래소 |
| 언어 | Python 3.11+ |
| ML 스택 | XGBoost, LightGBM, LSTM, GRU, HMM, DQN |
| 하드웨어 | Mac Mini M4 (16GB RAM, 256GB) |
| 초기 자본 | $100~500 |
| 법률 검토 | 한국 가상자산이용자보호법 준수 확인 |
| 벤치마크 | Freqtrade, VishvaAlgo, Alex K Bybit 봇 |

---

## 단계적 구현 원칙

**Phase A — 즉시 구현 (작동 우선)**
- Layer 1 룰 기반 필터 + XGBoost 단일 모델
- 고정 포지션 2% + 단순 손절/익절
- 서킷브레이커 5단계 + 텔레그램 알림
- WebSocket + SQLite 캐시 데이터 수집
- DataQualityChecker(7단계) + StorageManager

**Phase B — 안정화 후**
- LSTM + GRU 앙상블 추가 (Google Colab 초기 학습)
- ADX + SuperTrend + ATR 트레일링 스탑
- Kelly Fractional + ATR 변동성 그룹화
- Walk-Forward 백테스트 + Optuna Hyperopt
- **[신규-v3] Lookahead Bias 검증 (backtest/lookahead.py)**
- **[신규-v3] 생존 편향 처리 — 시점별 실제 코인 목록 백테스트**
- **[신규-v3] Monte Carlo 검증 1,000회 (backtest/monte_carlo.py)**

**Phase C — 검증 후**
- HMM 레짐 감지 + VaR 오버레이
- 다중자산 공분산 Kelly + 코인 상관 클러스터링
- 3전략 자동 전환 (추세추종/그리드/DCA) 실전 적용
- Streamlit 대시보드
- RL 에이전트 실거래 전환 (4가지 기준 충족 시)
- **[신규] StrategyDecayMonitor — 전략별 롤링 성과 감시 + 자동 휴면**
- **[신규] 동적 전략 가중치 — 고정 60/20/15 → 성과 기반 자동 배분**

**Phase D — 장기**
- RL 에이전트 고도화 (PPO 전환)
- Incremental Learning 자동화
- 모델 드리프트 다중 지표 감지
- **[신규-v3] 자동 피처 엔지니어링 — FreqAI식 대규모 피처 생성 (현재 35개 → 수백개 목표)**
  - 기술지표 파라미터 자동 조합 (RSI 기간 5~30, EMA 기간 다수 등)
  - 타임프레임 간 교차 피처 자동 생성
  - feature importance 기반 자동 가지치기

## 레이어 구조

| 레이어 | 역할 | 단계 |
|---|---|---|
| 데이터 수집 | WebSocket + REST + SQLite 캐시 + 7단계 품질검증 | Phase A |
| Layer 0 | HMM 시장 레짐 감지 (4개 레짐) | Phase C |
| Layer 0.5 | 코인 상관 클러스터링 + ATR 변동성 그룹화 | Phase C |
| Layer 1 | 룰 기반 시장 필터 (10개 조건) | Phase A |
| Layer 2 | ML 앙상블 예측기 (레짐별 전문 모델) | Phase A~B |
| Layer 3 | RL 에이전트 DQN→PPO | Phase C~D |
| 전략 선택기 | 3전략 자동 전환 (추세추종/그리드/DCA) | Phase A~C |
| 서킷브레이커 | 5단계 블랙스완 대응 (최우선) | Phase A |
| 텔레그램 봇 | 실시간 알림 + 원격 제어 | Phase A |
| StorageManager | SQLite 자동 정리 + VACUUM + 디스크 모니터링 | Phase A |
| Streamlit 대시보드 | 실시간 성과 시각화 | Phase C |

---

# Master Prompt

```
## 역할 설정
너는 암호화폐 퀀트 트레이딩 전문가이자 Python/PyTorch 시니어 개발자다.
멀티타임프레임 기술적 분석, 퀀트 전략, ML/DL 앙상블, HMM 레짐 감지,
온체인 분석, RL, 다중자산 Kelly+VaR, 포트폴리오 클러스터링,
3전략 자동 전환 시스템에 깊은 이해를 갖추고 있다.

## 프로젝트 컨텍스트
업비트(Upbit) 거래소 Python 기반 퀀트 자동매매 봇.
텔레그램 봇으로 실시간 모니터링 및 원격 제어.

## 개발자 배경
- CS Bachelor's + Master's 과정 중
- Python 숙련, pandas/numpy/PyTorch 능숙
- LSTM 구현 경험 있음 / RL 학습 중
→ 기초 ML/Python 설명 생략
→ 프로젝트 특화 구현 + 실전 고려사항 집중

## 핵심 설계 원칙
[단계적 구현] Phase A→B→C→D 순서 준수
[모듈 폴백] HMM오류→ADX / FinBERT오류→VADER / RL오류→룰기반 / VaR오류→고정2%
[페이퍼 트레이딩] 실전 봇과 동일 신호 병렬 실행 (매일 비교)
[거래 빈도 제한] 진입 후 최소 15분 보유 / 동일 코인 재진입 10분 대기 / 일 최대 10회
  [v9 변경] 5회 → 10회: 진입 조건 강화(앙상블 0.62+)로 신호 품질 확보 → 횟수 상한 완화
  비용 상한선 유지: 왕복 0.1% × 10회 = 일 1.0% → 월 목표 수익 5~10% 기준 허용 범위 내
[신규-v8] 재진입 조건 강화 — 쿨타임만으로는 부족, 아래 조건 추가 필요:
  손절 후 재진입 시 추가 확인 조건 (쿨타임 10분 경과 후):
    ① tick_imbalance > 0.15 (매수 체결 우세 회복 확인)
    ② SuperTrend 방향이 손절 당시와 반대로 전환됨
    ③ OBI > 0.1 (오더북 매수 압력 최소 확인)
  3조건 중 2개 이상 미충족 시 재진입 보류 (최대 1시간 추가 대기)
  이유: 쿨타임만 기다리면 동일 하락 추세에 재진입하는 연속 손절 패턴 발생 가능
[비용 반영] 수수료 0.1% + 슬리피지 0.15% = 0.25% 전 단계 반영
[신규-실행 타이밍 노이즈] 매수 신호 확정 후 0~90초 균등 랜덤 지연 후 주문 실행
  목적: front-running 방어, 동일 전략 사용자 간 동시 체결 회피
  import random; delay = random.uniform(0, 90); await asyncio.sleep(delay)
  예외: 서킷브레이커 발동 시 즉시 실행 (지연 없음)
  예외: 익절/손절 청산 시 즉시 실행 (지연 없음)

## 시스템 아키텍처

[서킷브레이커] 최우선 — 모든 레이어보다 우선
  Level 1: 1분 내 -3% → 매수 5분 정지, 트레일링 스탑 타이트 조정
  Level 2: 10분 내 -8% → 전량 USDT 전환 + 30분 중단
  Level 3: 일일 -10% → 당일 전체 중단
  Level 4: API 오류 연속 3회 OR WebSocket vs REST 괴리 >3% → 전체 즉시 중단
  Level 5: 24h 누적 -15% → 수동 확인 요청

[Layer 0] HMM 레짐 감지 (Phase C)
  hmmlearn GaussianHMM 4레짐:
    0: 강한 상승 / 1: 약한 상승·횡보 / 2: 약한 하락·횡보 / 3: 강한 하락
  ADX와 병행: HMM확률 × ADX신호 = 최종 신뢰도
  폴백: HMM 오류 시 ADX 단독 사용

[Layer 0.5] 코인 클러스터링 (Phase C)
  30일 수익률 상관계수 행렬 (numpy.corrcoef)
  상관 > 0.8: 동일 클러스터 → 1개만 보유 (동반 폭락 방지)
  ATR 변동성 그룹:
    High (ATR/Price>5%): 손절-10%, 익절1차+15%, Kelly×0.6
    Low  (ATR/Price<2%): 손절-5%,  익절1차+8%,  Kelly×1.0
  매주 재계산 + SQLite 저장

[전략 선택기] StrategySelector — HMM 레짐별 자동 전환 + 동적 가중치
  레짐 0 (ADX>30):         TREND_STRONG  → 추세 추종 + 피라미딩 (자본 60% 기본)
  레짐 1 (ADX 20~30):      TREND_NORMAL  → 추세 추종 보수적 (자본 60% 기본)
  레짐 2 (ADX<20):         GRID          → 그리드 전략 (자본 20% 기본)
  레짐 3 (F&G<30):         DCA           → 적응형 DCA (자본 15% 기본)
  레짐 3 (F&G>=30):        HOLD          → USDT 유지 (자본 5%)
  Phase A,B: ADX 기반 간소화 버전 사용 (HMM 없이)

  [신규] 동적 자본 배분 (Phase C, StrategyDecayMonitor 연동):
    각 전략의 최근 4주 롤링 샤프비율을 가중치로 정규화
    w_i = max(sharpe_i, 0) / Σ max(sharpe_j, 0)  # 음수 전략 제외
    최종 배분 = 기본 배분 × 0.5 + 동적 배분 × 0.5  # 완전 동적 전환 방지
    예: 추세 샤프0.8, 그리드 샤프1.6, DCA 샤프0.4 → 그리드 가중치 상승
    StrategyDecayMonitor.get_weights() 호출로 매 주기 갱신

[신규] StrategyDecayMonitor (strategies/decay_monitor.py, Phase C)
  감시 항목: 전략별 4주 롤링 샤프비율, 승률, 손익비 (SQLite strategy_log 기반)
  휴면 트리거: 샤프비율 < 0.5 로 4주 연속 → 해당 전략 DORMANT 상태 전환
  DORMANT 동작: 해당 전략 자본 배분 0% + 텔레그램 경고 전송
  복귀 조건: 2주 연속 샤프비율 > 0.8 (백테스트 재검증 통과 시)
  파라미터 재탐색: DORMANT 진입 시 Optuna 해당 전략 구간만 n_trials=50 재실행
  SQLite strategy_decay_log 별도 저장 (전략 수명 추적)
  APScheduler: 매주 일요일 05:00 전체 전략 성과 재계산

[Layer 1] 룰 기반 시장 필터 (Phase A)
  조건 0: 서킷브레이커 상태 확인 (최우선)
  조건 1: HMM+ADX 신뢰도 (Phase C 이전은 ADX만, ADX<15 시 매수 보류)
  조건 2: Fear&Greed 15~85 범위
  조건 3: BTC 도미넌스 < 60% (알트 매수 시, 초과 시 신호강도 ×0.7)
  조건 4: 일봉 EMA50 vs EMA200 추세 (데드크로스: 강도 ×0.8)
  조건 5: 멀티타임프레임 추세 합의 (3개 동시 충족 시만 진입 허용)
    5a: 5분봉 — EMA7 > EMA25 (단기 방향 확인)
    5b: 1시간봉 — EMA20 > EMA50 (중기 방향 확인)
    5c: 일봉 — EMA50 > EMA200 (장기 방향 확인, 데드크로스 시 매수 차단)
    3개 중 2개 이상 불일치 시 매수 보류 / 3개 불일치 시 즉시 차단
  조건 6: 5분봉 거래량 20기간 평균 50% 이상
  조건 7: 온체인 거래소 유입 전일比 +50% → 매도 압력 경고
  조건 8: 감성 비토 (VADER+FinBERT+시장지수 3방향 일치 + 레짐 연동)
  조건 9: 코인 클러스터 중복 포지션 확인 (Phase C)
  조건 10: API 레이턴시 < 500ms (초과 시 해당 코인 스킵)
  저유동성 자동 제외: 24h 거래량 < 1억원

[Layer 2] ML 앙상블 예측기
  레짐 0,1: XGBoost(1.0) + LightGBM(1.0) + LSTM(0.8) + GRU(0.8) 가중 평균
  레짐 2,3: 보수적 파라미터 앙상블
  신호 확정: 가중 평균 0.62 이상 AND 3모델 이상 합의
  [v9 변경] 0.55 → 0.62: 애매한 신호 차단, 승률·손익비 개선 목적
    실전 최적값은 Optuna ensemble_threshold(0.55~0.75) 탐색으로 자동 결정
  Incremental: 매 거래 후 XGBoost/LGB 소폭 업데이트 (continue_training, 가중치 ×2.0)
  Optuna: 핵심 파라미터 20~30개 자동 최적화 (목적함수: Walk-Forward OOS 샤프비율)

[Layer 3] RL 에이전트 (Phase C 이후)
  Phase A,B: 시뮬레이션 모드만 실행
  실거래 전환 기준 (4가지 동시 충족):
    시뮬레이션 누적 ≥ 1,000 에피소드
    시뮬레이션 샤프비율 ≥ 1.5
    최근 4주 실거래(룰기반) 승률 ≥ 55%
    자본 ≥ $1,000

## 데이터 소스
[가격] pyupbit: 5분봉+1시간봉+일봉
       [신규-v3] 동적 페어리스트: 업비트 KRW 마켓 24h 거래량 상위 30개 (매일 04:00 갱신)
         → 저유동성 자동 제외: 24h 거래량 < 1억원
         → 레버리지 토큰 블랙리스트 자동 제외 (*BULL, *BEAR, *UP, *DOWN, *L, *S)
         → 실제 스캔 대상: 약 20~25개 / 동시 보유 상한: 최대 5개 포지션
         → 페어리스트 갱신 시 기존 포지션 보유 코인은 유지 (강제 청산 없음)
         → config.yaml의 COINS 리스트는 블랙리스트 추가 전용으로 역할 변경
       WebSocket 우선 (최대 30개 코인 단일 연결 동시 구독)
       REST 증분 업데이트 (새 캔들만), RateLimiter(max=8/초), sleep(0.1)
       Remaining-Req 헤더 모니터링 (잔여<5 시 sleep(1))
       [주의] 30코인 × 3타임프레임 REST 업데이트 = 최소 12초 소요 → 5분 루프 내 처리 보장
[지표] pandas-ta:
       5분봉: RSI(14), MACD(12,26,9), BB(20,2), EMA(7,25,99), ADX(14), SuperTrend(10,3), ATR(14)
       1시간봉: RSI, EMA(20,50), MACD, ADX
       일봉: EMA(50,200), RSI, 추세 인코딩
       [신규-v3] 일봉 지표 타임 원칙: 모든 일봉 피처(EMA_50d, EMA_200d, RSI_daily)는
         전일(D-1) 종가 확정값만 사용. 당일 미확정 캔들 사용 금지.
         구현: df_daily.shift(1) 적용 후 피처 추출 → 백테스트↔실거래 지표 괴리 원천 차단
[시장] Fear&Greed(Alternative.me, 일1회), BTC도미넌스+시총(CoinGecko, 1시간)
       펀딩비(CoinGlass, 8시간), Altcoin Season(Blockchain Center)
[온체인] CryptoQuant 무료 티어: 거래소 유입/유출량 (1시간)
[감성] VADER(1차, ~50MB, ~50ms) + FinBERT(2차 ±0.3미만만, ~1.2GB, 상시 로드)
       감성 비토: VADER + FinBERT + 시장지수 3방향 일치 시만 신호 확정
       레짐 연동: 레짐0,1+긍정→풀포지션 / 레짐0,1+부정→×0.5 / 레짐2,3+부정→USDT / 레짐2,3+긍정→반등대기
       RSS: CoinDesk, CoinTelegraph (1시간)
[신규-김치프리미엄] 업비트 BTC 가격 vs Binance BTC/USDT 가격 + 환율(한국은행 OpenAPI)
       kimchi_premium = (upbit_price_krw / (binance_price_usd × usd_krw_rate) - 1) × 100
       수집 주기: 5분 / SQLite 저장 / 피처로 ML 앙상블 입력 (+1개 피처, 누적 36개)
       해석: 프리미엄 급등(>3%) → 국내 매수세 과열 경고 / 급락(<-1%) → 국내 매도 압력
       환율 API: https://www.bok.or.kr/openapi (무료, 일 1회 캐시 사용)
[신규-오더북불균형] 업비트 REST /v1/orderbook (15개 호가, 5분 주기)
       OBI = (bid_total_size - ask_total_size) / (bid_total_size + ask_total_size)  # -1~+1
       상위 5호가 집중도: top5_concentration = top5_volume / total_volume
       [신규-v5] orderbook_wall_ratio = 최대단일호가잔량 / 평균호가잔량
       피처로 ML 앙상블 입력 (+3개 피처, 누적 38개) — OBI + top5 + wall_ratio
       해석: OBI > 0.3 → 매수 압력 / OBI < -0.3 → 매도 압력
[신규-v5 마이크로스트럭처] WebSocket trade 이벤트 실시간 계산 (CandleBuilder 연동)
       tick_imbalance = 5분 단위 (매수체결량 - 매도체결량) / 전체체결량  # -1~+1
       trade_velocity = 최근30초체결건수 / 직전30초체결건수               # 가속도
       피처로 ML 앙상블 입력 (+2개 피처 → 총 38개에 이미 포함, 피처 목록 참조)

## 기술 스택
Python 3.11+ / pyupbit / pandas, numpy, pandas-ta
PyTorch (LSTM, GRU, DQN) / XGBoost, LightGBM / hmmlearn
Optuna / PyPortfolioOpt / scipy / stable-baselines3
python-telegram-bot 20.x / Streamlit
SQLite / python-dotenv / APScheduler

## 하드웨어 제약
Mac Mini M4 (16GB RAM, 256GB SSD) — 봇 전용 운용
device=cpu 강제 (MPS 대신 CPU 사용, 안정성 우선)
배치사이즈: 32 / Transformer→GRU 교체 (메모리 절약)
인메모리: 코인당 최근 500개 캔들만 유지
초기 대규모 학습: Google Colab → 맥미니 fine-tuning만
[신규-v3] 30코인 동적 페어리스트 기준 메모리 예산:
  캔들 데이터(30코인×3TF×500개): ~1.2GB
  ML 앙상블 모델: ~700MB
  FinBERT: ProcessPoolExecutor 별도 프로세스 실행 (추론 중 ~420MB / 유휴 시 ~0MB)
  SQLite 캐시 + 기타: ~700MB
  OS + Python 런타임: ~2GB
  합계 예상: ~5.1GB (유휴) / ~5.5GB (FinBERT 추론 중) → 16GB 기준 여유 충분
  [필수-v4] FinBERT는 메인 asyncio 루프와 반드시 프로세스 분리
    → concurrent.futures.ProcessPoolExecutor(max_workers=1) 전용 프로세스 사용
    → 1시간 배치 처리 완료 후 del model; gc.collect() 로 메모리 즉시 해제
    → 메인 루프(서킷브레이커, WebSocket 수신)와 완전 독립 보장
    → 미분리 시: 1.2GB 모델 로딩+추론 과정에서 메인 루프 수 초간 블로킹 → 서킷브레이커 지연 위험

## 학습 데이터
목표: 25,000~30,000 타임스텝
  5분봉 6개월: ~52,560개 / 1시간봉 2년: ~17,520개 / 일봉 3년: ~1,095개
HMM 학습: 일봉 3년 + 1시간봉 2년 (피처: 수익률, 변동성, 거래량변화율)
레짐별 앙상블 분리: 레짐0,1 데이터→상승 전문 모델 / 레짐2,3→하락 전문 모델
Walk-Forward: 6개월 학습→1개월 테스트→전진 반복
Pessimistic: 수익 -0.25%, 손실 +0.25%
콜드스타트: <200건 룰기반 / 200건+ XGBoost·LGB / 500건+ LSTM·GRU / 1000건+ HMM
재학습: 매주 일요일 자정, 2주 정확도<55% OR 예측기반수익<실거래수익×0.8 즉시 트리거
[신규-v8] 콜드스타트 페이퍼 트레이딩 전제 조건:
  페이퍼 트레이딩 시작 시점: 룰기반 거래 200건 누적 이후
  이유: 200건 미만 구간은 XGBoost 미작동 → 신호 품질 최저 → 페이퍼 결과 신뢰 불가
  실거래 투입: 페이퍼 트레이딩 4주 이상 + 7가지 체크리스트 통과 후
  콜드스타트 구간 DRY_RUN 강제: <200건 동안 DRY_RUN=true 자동 유지
    → 코드로 강제: if trade_count < 200: assert DRY_RUN == True

## ML 앙상블 입력 피처 (총 38개)
5분봉(14): OHLCV(5), RSI, MACD, MACD_signal, BB_upper, BB_lower, EMA(7,25,99), 거래량변화율
1시간봉(5): RSI_1h, EMA20_1h, EMA50_1h, MACD_1h, 1h추세방향
일봉(4): EMA_50d, EMA_200d, RSI_daily, 일봉추세인코딩
시장지수(3): fear_greed, btc_dominance, altcoin_season
온체인(2): exchange_inflow, exchange_outflow
감성(2): sentiment_score(-1~1), sentiment_confidence(0~1)
추세·레짐(4): ADX_5m, ADX_1h, supertrend_signal, hmm_regime(0~3)
업비트 특화(3): kimchi_premium(-5~+10%), obi(-1~+1), top5_concentration(0~1)
[신규-v5] 마이크로스트럭처(3): tick_imbalance(-1~+1), orderbook_wall_ratio(0~20+), trade_velocity(0~5+)
  tick_imbalance: 5분 단위 매수/매도 체결량 불균형 → 단기 방향성 선행 지표
  orderbook_wall_ratio: 호가벽 강도 → 저항/지지 레벨 탐지
  trade_velocity: 체결 속도 가속도 → 변동성 폭발 직전 신호

## 3전략 상세 설계

### 전략 1 — 추세 추종 (레짐 0,1 / 자본 60%)
트레일링 스탑: 손절선 = 현재가 - ATR(14)×2.5 (상승 시 위로 이동)
부분 익절: +10% 1/3매도 → +20% 1/3매도 → 잔여 1/3 트레일링
피라미딩: ADX>30 + 조정확인 + RSI40~50 + 포지션+5% → 기존50% 추가 (상한 자본10%)
조정 vs 반전:
  조정→홀딩: ADX>25 AND SuperTrend미반전 AND EMA99위 AND 거래량감소 AND RSI≥40
  반전→청산: ADX<20 OR SuperTrend반전 OR EMA99이탈 OR RSI다이버전스 OR 거래량급증+하락

### 전략 2 — 그리드 전략 (레짐 2 / 자본 20%)
[신규-v8] 최소 자본 조건: 총 자본 >= $300 (약 40만원) 시에만 활성화
  미만 시: GRID 레짐 진입해도 HOLD로 자동 대체
  이유: 그리드 자본 20% × 10단계 = 단계당 자본의 2%
        $100 기준 단계당 $2 → 업비트 최소 주문금액 5,000원($3.8) 미달
범위: 현재가 ± ATR(14)×3 자동 계산
레벨: 10단계 균등 분할 / 단위금액: 그리드자본/10
주문: 하단5개 지정가매수 + 상단5개 지정가매도 (업비트 지정가 주문 사용)
체결: 매수체결→바로위레벨 매도주문 / 매도체결→바로아래레벨 매수주문
종료: ADX > 25 감지 시 그리드 즉시 종료 → 추세 추종 전환
재계산: 매 4시간 ATR 변화에 따라 그리드 자동 재설정 (APScheduler)

### 전략 3 — 적응형 DCA (레짐 3+F&G<30 / 자본 15%)
트리거: HMM 레짐 3 + Fear&Greed < 30
기본 매수: 자본 2% 첫 매수
Safety Order: -3% 하락마다 추가 매수 (최대 5회)
금액 증가: base × 1.5^n (n=현재 오더 수)
익절: 평균매수가 +3% 달성 시 전량 매도
F&G 연동: F&G<15 → 매수금액 ×1.5 / F&G<30 → ×1.0 / F&G≥30 → ×0.5
레짐 호전: DCA 포지션 유지 + 추세 추종 병행 시작
DCA 모니터링: 매 5분 Safety Order 조건 체크 (APScheduler)

## Kelly + VaR 포지션 사이징

### 다중자산 Kelly (PyPortfolioOpt)
F* = Σ⁻¹×μ (Ledoit-Wolf 수축 추정)
Fractional: f*×0.1 시작 → 점진적 증가 (크립토 극변동성 대응)
상한: 자본 5% / 하한: 자본 1%
매 20거래마다 재계산
음수 Kelly → 해당 코인 스킵 / 극단값 >20% → 5%로 클리핑

### HMM 신뢰도 연동 (동적 리스크 예산)
실제 크기 = Kelly × HMM 레짐 신뢰도 확률
레짐 0 신뢰도 90%: f*×0.25×0.9 / 레짐 0 신뢰도 60%: f*×0.25×0.6

### VaR 오버레이
역사적 VaR (95%, 1일, 최근 60일)
VaR>자본3%: 포지션×0.5 / VaR>자본5%: 신규매수 중단

### 연속 손실 패널티
3연속: ×0.7 / 5연속: ×0.5 / 7연속+: ×0.3 / 5연속수익: 정상복구

### 레짐별 Kelly 조정
레짐 0: f*×0.25 / 레짐 1: f*×0.15 / 레짐 2: f*×0.05 / 레짐 3: 0

## RL State(20) / Action(18)
State(20): 앙상블확률, 모델합의도, HMM레짐(0~3), HMM신뢰도,
           F&G, BTC도미넌스, RSI(5분봉), ADX(5분봉), SuperTrend방향,
           일봉추세, risk_level, 포지션수/최대, 미실현손익, 잔여자본비율,
           최근10거래승률, 연속손실횟수, 시간대(주기성), VaR값, Kelly_f*,
           [신규] kimchi_premium  # 업비트 특화 시장 온도계

Action(18):
  0:관망 / 1:소액매수(Kelly×0.5) / 2:중액매수(Kelly×1.0) / 3:대액매수(Kelly×1.5, 상한5%)
  4:1/3매도 / 5:1/2매도 / 6:전량매도 / 7:USDT전환 / 8:코인교체
  9:트레일링스탑활성화 / 10:피라미딩매수 / 11:1차부분익절 / 12:2차부분익절
  13:그리드시작(레짐2진입) / 14:그리드종료(레짐전환)
  15:DCA첫매수(레짐3진입) / 16:DCA Safety Order추가 / 17:DCA전략청산

## 하드가드 우선순위 (RL보다 항상 우선)
(1) 서킷브레이커 Level 1~5
(2) VaR 오버레이
(3) 일일 손실 한도 -10%
(4) 개별 손절 강제 (-7% 또는 ATR 변동성 그룹 기준)
(5) 연속 손실 패널티
(6) 레짐 3 신규 매수 차단
(7) 코인 클러스터 중복 포지션 차단
(8) API 레이턴시 이상 스킵
(9) RL 결정 (최후)
Private Key: .env 전용 / 비용: 0.25% 전 단계 반영

## 법률 준수 (가상자산이용자보호법)
본인 계정/자산만 운용 (타인 위탁 금지)
공개 데이터만 활용 (미공개 정보 이용 금지)
저유동성 코인 자동 제외 (24h거래량 < 1억원)
동시 다중 코인 매수 간격: 최소 30초
거래 기록 영구 보관 (trades 테이블 삭제 금지, 최소 5년)

이 컨텍스트를 기억하고 모든 코드를 이 구조로.
기초 개념 설명 생략, 프로젝트 특화 구현 집중.
Phase A부터 단계적으로 구현.
지금은 확인만 하고 다음 지시를 기다려.
```

---

# Phase 1 — 프로젝트 설계 (설계 문서)

```
Master Prompt 컨텍스트 기반으로 전체 프로젝트 설계 문서를 작성해줘.
코드 작성 말고 설계 문서 형태로만.

[요청 1] 폴더/파일 구조
upbit_bot/
├── main.py                   # 메인 진입점
├── config.yaml               # 설정 (코인 목록, 파라미터)
├── .env                      # API 키 (git 제외)
├── data/
│   ├── collector.py          # WebSocket + REST 수집기
│   ├── cache.py              # SQLite 캐시 + 증분 업데이트
│   ├── candle_builder.py     # [필수-v4] WebSocket 캔들 합성 (REST 대체)
│   └── quality.py            # DataQualityChecker (7단계)
├── layers/
│   ├── layer0_hmm.py         # HMM 레짐 감지
│   ├── layer0_5_cluster.py   # 코인 상관 클러스터링
│   ├── layer1_filter.py      # 룰 기반 필터 (10개 조건)
│   ├── layer2_ensemble.py    # ML 앙상블 (XGB+LGB+LSTM+GRU)
│   └── layer3_rl.py          # RL 에이전트 (DQN→PPO)
├── strategies/
│   ├── selector.py           # StrategySelector (레짐→전략 전환 + 동적 가중치)
│   ├── trend.py              # 추세 추종 + 트레일링 + 피라미딩
│   ├── grid.py               # GridStrategy (횡보장)
│   ├── dca.py                # AdaptiveDCAStrategy (하락장)
│   └── decay_monitor.py      # [신규] StrategyDecayMonitor (전략 포화 감시)
├── models/                   # 체크포인트 저장
├── risk/
│   ├── kelly.py              # 다중자산 Kelly + VaR
│   ├── circuit_breaker.py    # 서킷브레이커 5단계
│   └── trailing_stop.py      # ATR 트레일링 스탑
├── execution/
│   ├── engine.py             # 메인 실행 엔진
│   ├── order.py              # 업비트 주문 (지정가/시장가)
│   └── paper_trading.py      # 페이퍼 트레이딩 병렬
├── monitoring/
│   ├── telegram_bot.py       # 텔레그램 봇 (18개 명령어)
│   ├── storage_manager.py    # 스토리지 자동 관리
│   └── dashboard.py          # Streamlit 대시보드
├── backtest/
│   ├── walk_forward.py       # Walk-Forward 검증 + 생존 편향 처리
│   ├── hyperopt.py           # Optuna 파라미터 최적화
│   ├── lookahead.py          # [신규-v3] Lookahead Bias 검증
│   └── monte_carlo.py        # [신규-v3] Monte Carlo 시뮬레이션 (1,000회)
└── utils/
    ├── logger.py             # 로깅 설정
    └── helpers.py            # 공통 유틸

[요청 2] 레이어 간 데이터 스키마 (dataclass)
아래 8가지 dataclass 상세 정의:
  RawMarketData: WebSocket 수신 원본
  MarketState: 멀티타임프레임 + 온체인 + 감성 통합 출력
  FilterResult: Layer 1 출력 (tradeable, regime_strategy, signal_multiplier,
                adx_value, supertrend_direction, atr_value, active_warnings,
                pullback_detected, reversal_detected, api_latency_ok, daily_loss_pct)
  EnsemblePrediction: Layer 2 출력 (per_model_probs, weighted_avg,
                      consensus_count, signal_confirmed, hmm_regime, hmm_confidence)
  StrategyDecision: 전략 선택기 출력 (strategy_type, capital_allocation,
                    grid_params, dca_params)
  RiskBudget: Kelly+VaR 출력 (kelly_f, hmm_adjusted_f, var_adjusted_f,
              final_position_size, coin_group)
  TradeDecision: Layer 3 최종 출력 (action 0~17, target_coin, position_size,
                 trailing_stop_price, partial_exit_ratio)
  TelegramEvent: 메시지 큐 구조

[요청 3] SQLite 테이블 설계 (15개 테이블)
candles_5m, candles_1h, candles_1d
market_indices, onchain_data, sentiment_log
ensemble_predictions, trades (영구보관), layer1_log
coin_scan_results, strategy_log, storage_audit_log
[신규-v2] kimchi_premium_log  # 5분 주기 김치프리미엄 + 환율 (보관 3개월)
[신규-v2] strategy_decay_log  # 전략별 주간 샤프/승률/손익비 + DORMANT 이력 (영구보관)
[신규-v3] coin_history         # 날짜별 업비트 KRW 마켓 코인 목록 스냅샷 (생존 편향 처리용, 영구보관)
각 테이블 컬럼 + 인덱스 + 보관 기간 명시

[요청 4] APScheduler 전체 스케줄 목록
메인루프(5분), 포지션모니터(1분), 서킷브레이커(10초),
그리드범위재계산(4시간), DCA모니터(5분),
데이터수집 각 주기, 재학습(주1회), 스토리지관리 등
[신규-v2] 김치프리미엄 수집: 5분
[신규-v2] 오더북 OBI 수집: 5분 (메인루프와 동기화)
[신규-v2] 전략 성과 재계산 (StrategyDecayMonitor): 매주 일요일 05:00
[신규-v3] 동적 페어리스트 갱신 (PairlistManager): 매일 04:00
[신규-v3] 코인 목록 스냅샷 저장 (생존 편향용): 매일 00:00 → coin_history 테이블

[요청 5] 개발 우선순위 로드맵
Phase A~D 각 단계별 구현 항목 + 예상 소요 시간 + 전제 조건

[요청 6] M4 16GB 최적화 전략
3전략 동시 실행 시 메모리 사용량 예측 + Google Colab 연동 계획
동적 페어리스트 30개 코인 기준 메모리 예산 계획 포함
```

---

# Phase 2 — 데이터 수집기 + DataQualityChecker (7단계)

```
Phase 1 설계 기반으로 데이터 수집 모듈과
DataQualityChecker를 구현해줘.
이전 Phase 설계의 dataclass와 폴더 구조를 유지해줘.

[수집 대상 1] 업비트 가격 (pyupbit + WebSocket)
대상: [신규-v3] 동적 페어리스트 — 업비트 KRW 마켓 24h 거래량 상위 30개
  (기존 config.yaml 고정 10개 → 매일 04:00 자동 갱신)
  제외 규칙: 24h 거래량 < 1억원 / 레버리지 토큰 / config.yaml 블랙리스트
  PairlistManager 클래스: get_active_pairs() → 현재 유효 코인 목록 반환
타임프레임: 5분봉 + 1시간봉 + 일봉

WebSocket:
  wss://api.upbit.com/websocket/v1
  Origin 헤더 제거 (rate limit 완화)
  최대 30개 코인 단일 연결로 동시 구독
  수신: ticker(현재가), trade(체결)
  수신 → asyncio.Queue → 메인 루프
  연결 끊김 시 자동 재연결 (exponential backoff)
  [신규-v3] 페어리스트 갱신 시 WebSocket 재구독 자동 처리
  [신규-v5] trade 이벤트에서 실시간 체결 강도 피처 계산 (CandleBuilder 연동):
    tick_imbalance = (매수체결량 - 매도체결량) / 전체체결량  # -1~+1, 5분 단위 집계
    trade_velocity = 최근30초체결건수 / 직전30초체결건수     # 체결 속도 가속도, 1분 롤링
    업비트 trade 이벤트의 ask_bid 필드 활용 (매수=ASK, 매도=BID)

REST 캔들 업데이트 — 최초 1회 히스토리 로딩에만 사용:
  초기 기동 시: SQLite 없을 때 코인당 200개 풀 요청 (1회성)
  이후 실시간: REST 호출 중단 → WebSocket 캔들 합성으로 완전 대체
  일봉: 매일 04:00 하루 1회 REST 업데이트 (30코인 × 1요청 = 30회, 문제없음)
  RateLimiter(max_per_second=8): 초기 로딩 시에만 적용
  [필수-v4] REST → WebSocket 합성 전환으로 5분 루프당 REST 호출: 90회 → 0회
    이유: 30코인 × 3타임프레임 REST 증분 업데이트 = 루프당 최소 90회 요청
          → 초당 8회 제한 소진 시 HTTP 429 에러 + IP 수 분간 밴 위험
          → 밴 기간 중 서킷브레이커/신호 감지 불가 → 계좌 리스크

[필수-v4] WebSocket 캔들 합성 (CandleBuilder 클래스, data/candle_builder.py):
  WebSocket trade 이벤트(체결 데이터) 실시간 누적
  5분봉 합성: 매 5분 경계에서 누적 trade → OHLCV 완성 → MarketState 갱신
  1시간봉 합성: 5분봉 12개 집계 → 1시간봉 자동 완성
  캔들 완성 시 pandas-ta 지표 자동 재계산 트리거
  갭 발생 시 (WebSocket 끊김): REST 폴백으로 해당 구간만 보완 후 재개
  DataQualityChecker step7 cross-validate 연동 (합성 캔들 무결성 검증)
  SQLite 저장: 합성 완성된 캔들만 저장 (실시간 미완성 캔들 저장 금지)
  [신규-v5] 5분 캔들 완성 시 마이크로스트럭처 피처 동시 계산:
    tick_imbalance = (매수체결량합 - 매도체결량합) / 전체체결량합  # ask_bid 필드 기반
    trade_velocity = 마지막30초_체결건수 / 직전30초_체결건수       # 급등 직전 가속도 감지
    → 두 값 모두 MarketState에 통합, ML 피처로 전달
  [신규-v5] WebSocket 침묵(Silent Drop) 감지:
    5분 경계 도달 시 해당 구간 체결 건수 < 10건 → stale 의심 플래그
    REST 폴백 가동 → 해당 캔들 1개만 보완 (전체 REST 재개 아님)

기술지표 자동 계산 (pandas-ta):
  5분봉: RSI(14), MACD(12,26,9), BB(20,2), EMA(7,25,99),
         ADX(14), SuperTrend(10,3), ATR(14), 거래량변화율
  1시간봉: RSI, EMA(20,50), MACD, ADX
  일봉: EMA(50,200), RSI, 추세인코딩
  [신규-v3] 일봉 지표 타임 원칙 적용:
    df_daily_shifted = df_daily.shift(1)  # 전일 종가 확정값만 사용
    EMA_50d, EMA_200d, RSI_daily 모두 shift(1) 적용 후 피처 추출
    당일 미확정 일봉 캔들 사용 금지 → 백테스트↔실거래 괴리 원천 차단

[수집 대상 2] 시장 지수
Fear&Greed (Alternative.me): 일 1회 / BTC도미넌스+시총 (CoinGecko): 1시간
펀딩비 (CoinGlass): 8시간 / Altcoin Season: 일 1회

[수집 대상 3] 온체인 (CryptoQuant 무료)
거래소 유입/유출량: 1시간

[수집 대상 4] 감성 (VADER + FinBERT)
RSS: CoinDesk, CoinTelegraph (1시간)
1단계: VADER 전체 분석 (compound ±0.3 이상 → 즉시 확정)
2단계: FinBERT ±0.3 미만 케이스만 2차 분석
  [필수-v4] 실행 방식: ProcessPoolExecutor 별도 프로세스에서 실행
    이유: FinBERT 로딩(~1.2GB) + 추론 시 Python GIL 특성상
          메인 asyncio 루프가 수 초간 블로킹됨
          → 이 구간에 서킷브레이커 발동 타이밍 누락, WebSocket 수신 지연 위험
    구현:
      process_pool = ProcessPoolExecutor(max_workers=1)  # FinBERT 전용
      result = await loop.run_in_executor(process_pool, finbert_batch, texts)
      # 완료 시 결과를 asyncio.Queue에 put → 메인 루프에서 비차단 get
    완료 후: del model; gc.collect() → 메모리 즉시 해제
    폴백: FinBERT 프로세스 오류 시 VADER 단독 결과로 자동 대체
감성 비토: VADER + FinBERT + 시장지수 3방향 일치 시만
1시간 평균 집계 → SQLite sentiment_log

[신규 수집 대상 5] 김치프리미엄 (Phase A부터 수집, Phase B부터 피처 활용)
  업비트 BTC 현재가(KRW) vs Binance BTC/USDT 현재가 × USD/KRW 환율
  환율: 한국은행 OpenAPI (https://www.bok.or.kr/openapi, 일 1회 캐시)
  Binance 가격: https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT (무료)
  수집 주기: 5분 / SQLite kimchi_premium_log 저장
  Layer 1 연동: 프리미엄 > 5% → 과열 경고 플래그 (신호강도 ×0.8)
  Layer 1 연동: 프리미엄 < -2% → 역프리미엄 경고 (매수 보류)

[신규 수집 대상 6] 오더북 불균형 OBI + 호가벽 탐지 (Phase A부터 수집, Phase B부터 피처 활용)
  업비트 REST /v1/orderbook (30개 코인, 15호가)
  OBI = (bid_total_size - ask_total_size) / (bid_total_size + ask_total_size)
  top5_concentration = 상위5호가합계 / 전체호가합계
  [신규-v5] orderbook_wall_ratio = 최대단일호가잔량 / 평균호가잔량  # 호가벽 강도 (>5 = 강한 벽)
    해석: 높은 값 → 특정 가격대에 대형 지정가 주문 존재 → 저항/지지 신호
    호가벽 방향: 매도벽(ask) 강할 때 상승 저항 / 매수벽(bid) 강할 때 하락 지지
  피처로 ML 앙상블 입력 (+3개 피처: obi, top5_concentration, orderbook_wall_ratio, 총 38개)
  API 레이턴시 포함 측정 → Layer 1 조건10 연동
  [주의] 오더북 원본 저장 금지 — 계산된 float 값만 MarketState에 통합 (디스크 절약)

[DataQualityChecker — 7단계 파이프라인]

def validate_pipeline(df, interval, coin):
  """7단계 순차 검증 — 순서 반드시 유지"""
  df, report = step1_ohlcv_logic(df)
  df, report = step2_timestamp(df, report)
  df, report = step3_volume_outliers(df, report)
  df, report = step4_price_outliers(df, report)
  df, report = step5_anomaly_detection(df, report)  # Isolation Forest
  df, report = step6_freshness(df, interval, report)
  report     = step7_cross_validate(df, report)
  score = compute_score(report)
  save_report(coin, interval, report, score)
  return df, score, report

1단계 step1_ohlcv_logic:
  invalid_mask = (High<Low) | (Close>High) | (Close<Low) | (Open<=0) | (Close<=0) | (Volume<0)
  처리: 해당 캔들 즉시 제거 (dropna 아님, drop)
  결과: 제거 개수 report['ohlcv_errors']에 기록

2단계 step2_timestamp:
  중복: df[~df.index.duplicated(keep='last')]
  역순: df.sort_index()
  미래: df[df.index <= pd.Timestamp.now(tz='UTC')]
  갭: expected_freq={'5m':'5T','1h':'1H','1d':'1D'} 기준 비교
      갭 발견 → 선형 보간 (최대 3개 연속) / 3개 초과 → 제외 플래그

3단계 step3_volume_outliers:
  Z-score > 4: 이상 거래량 플래그 (제거 아님, 학습 가중치 조정용)
  0 거래량 연속 3개+: 보간
  거래량 급증은 실제 이벤트일 수 있으므로 플래그만 설정

4단계 step4_price_outliers (강화):
  방법 3가지 중 2가지 이상 해당 시 이상치 확정:
    IQR 1.5배 초과
    Z-score > 3
    순간 변화율 > 15% (캔들 간)
  처리: 중간값(median)으로 대체

5단계 step5_anomaly_detection (핵심 신규):
  from sklearn.ensemble import IsolationForest
  features = ['close_pct_change','volume_pct_change','high_low_ratio','candle_body_ratio']
  iso_forest = IsolationForest(contamination=0.05, random_state=42, n_estimators=100)
  predictions = iso_forest.fit_predict(df[features])
  anomaly_mask = predictions == -1
  df['is_anomaly'] = anomaly_mask        # 제거 아님
  df['exclude_from_training'] = anomaly_mask
  이상 구간 → SQLite 별도 저장 (블랙스완 DB)
  연속 이상치 3개+ → 서킷브레이커 Level 2 연동
  anomaly_pct > 10%: 데이터 품질 심각 경고

6단계 step6_freshness:
  max_lag = {'5m': 10분, '1h': 2시간, '1d': 26시간}
  lag = now - df.index[-1]
  lag > max_lag: stale_data=True → 서킷브레이커 Level 4 연동
  lag > max_lag×2: 즉시 Level 4 발동

7단계 step7_cross_validate:
  WebSocket 현재가 vs REST 최신 종가 비교
  괴리 > 3%: source_mismatch → 서킷브레이커 Level 4
  괴리 > 1%: source_warning → 경고 로그 + REST 폴백

품질 점수:
  score = 1.0
  - ohlcv_errors × 0.02
  - anomaly_pct × 0.01
  - 0.3 if stale_data
  - 0.2 if source_mismatch
  - gap_count × 0.01
  점수 기준: ≥0.9 정상 / 0.7~0.9 경고 / 0.5~0.7 학습보류 / <0.5 중단

LSTM 학습 연동:
  exclude_from_training==True 행 제외
  이상 구간 전후 5개 캔들 추가 제외
  anomaly_score 높은 캔들: sample_weight 낮게
  최신 데이터: sample_weight × 2.0

SQLite 캐시: 5분봉 6개월 / 1시간봉 2년 / 일봉 3년
APScheduler 추가:
  매일 01:00 전체 코인 품질 리포트 생성 + 텔레그램 전송
  매주 이상 구간 DB 분석 → 패턴 발견 시 알림

[구현 요구사항]
각 수집기 독립 클래스 / 전체 결과 MarketState dataclass 통합
API 실패 시 재시도 3회 + 캐시 폴백
asyncio.gather()로 최대 30개 코인 병렬 처리
헬스체크 함수 포함
[신규-v3] PairlistManager 클래스: 매일 04:00 거래량 상위 30개 자동 갱신
[신규-v3] 일봉 shift(1) 적용 검증: 단위테스트로 미래참조 없음 확인
```

---

# Phase 3 — Layer 1 룰 기반 시장 필터

```
Phase 2 MarketState를 입력받는 Layer 1 필터를 구현해줘.

[10개 조건 — 순서대로 체크, 하나라도 실패 시 즉시 중단]
조건 0: 서킷브레이커 상태 확인 (최우선)
조건 1: HMM+ADX 신뢰도 (Phase C 이전은 ADX만, ADX<15 매수 보류)
조건 2: Fear&Greed 15~85 범위 확인
조건 3: BTC 도미넌스 < 60% (알트 매수 시, 초과 → 강도 ×0.7)
조건 4: 일봉 EMA50 vs EMA200 (데드크로스 → 강도 ×0.8)
조건 5: [신규-v2] 멀티타임프레임 추세 합의
  5a: 5분봉 — EMA7 > EMA25 (단기 방향)
  5b: 1시간봉 — EMA20 > EMA50 (중기 방향)
  5c: 일봉 — EMA50 > EMA200 (장기 방향, shift(1) 적용값 사용)
  3개 중 2개 이상 불일치 → 매수 보류 / 3개 불일치 → 즉시 차단
조건 6: 5분봉 거래량 20기간 평균 50% 이상
조건 7: 온체인 유입 전일比 +50% → 매도 압력 경고
조건 8: 감성 비토 (VADER+FinBERT+시장지수 3방향 일치 + 레짐 연동)
조건 9: 코인 클러스터 중복 포지션 확인 (Phase C)
조건 10: API 레이턴시 < 500ms
저유동성 자동 제외: 24h 거래량 < 1억원

[FilterResult dataclass 출력]
tradeable, market_regime(bull/bear/sideways),
regime_strategy(trend_following/grid/dca/wait),
risk_level(low/medium/high), signal_multiplier(0~1.5),
adx_value, supertrend_direction(1/-1), atr_value,
active_warnings, pullback_detected, reversal_detected,
api_latency_ok, daily_loss_pct

[구현 요구사항]
각 조건 독립 async 함수 (단락 평가)
모든 결과 SQLite layer1_log 저장
Phase C 이전 HMM 조건 자동 스킵
FinBERT 오류 시 VADER 단독 폴백
상세 로깅 (조건별 통과/실패 이유)
```

---

# Phase 4 — Layer 2 ML 앙상블 예측기

```
Layer 2 ML 앙상블 예측기를 구현해줘.
LSTM 구현 경험 있으므로 기초 설명 생략.

[앙상블 구조]
XGBoost(1.0) + LightGBM(1.0) + LSTM(0.8) + GRU(0.8) 가중 평균
레짐 0,1: 공격적 파라미터 / 레짐 2,3: 보수적 파라미터
신호 확정: 0.62 이상 AND 3모델 이상 합의  # [v9] 0.55→0.62, Optuna로 최적화

[입력] LSTM/GRU: 5분봉 60 timestep × 35 features
       XGBoost/LGB: 최신 1 timestep × 35 features (플랫)
       [신규-v3] 일봉 피처(EMA_50d, EMA_200d, RSI_daily)는 전일 shift(1) 적용값만 입력

[모델별]
XGBoost/LGB: device=cpu, Phase A 200건+ 시작
  Incremental: continue_training, 새 샘플 가중치 ×2.0
LSTM/GRU: device=cpu 강제 (M1 MPS 이슈)
  2레이어, hidden=128, Dropout(0.2), BCELoss, Adam(lr=0.001)
  Phase A 500건+ 시작

[Optuna Hyperopt]
탐색: RSI기간(5~30), EMA조합, ADX임계값(15~35),
      손절/익절비율, Kelly계수(0.1~0.5), ATR배수(1.5~4.0),
      ensemble_threshold(0.55~0.75)  # [신규-v9] 앙상블 임계값 자동 최적화
목적함수: Walk-Forward OOS 샤프비율 최대화
n_trials=200, MedianPruner 조기종료
Phase B 완성 후 최초 실행, 이후 월 1회 자동화

[데이터 품질 연동]
DataQualityChecker 점수 < 0.7 → 해당 코인 날짜 학습 제외
exclude_from_training==True 행 제외
이상 구간 전후 5개 캔들 추가 제외

[모델 드리프트 감지]
2주 정확도 < 55% OR 예측기반수익 < 실거래수익×0.8 → 즉시 재학습
피처 중요도 변화 추적 (SHAP)
예측 신뢰도 분포 모니터링

[콜드스타트]
<200건: 룰기반 스코어 / 200건+: XGBoost·LGB / 500건+: LSTM·GRU

[Google Colab 초기 학습]
Colab 노트북 코드 함께 작성 (드라이브 마운트 + 체크포인트 저장)

[구현 요구사항]
EnsemblePredictor 클래스 (4모델 통합)
모델별 독립 전처리 파이프라인
predict(market_state) → EnsemblePrediction
Walk-Forward + Optuna 연동 구조
예측 결과 SQLite ensemble_predictions 저장
```

---

# Phase 5 — Layer 3 RL 에이전트 + 3전략 시스템

```
Layer 3 RL 에이전트와 3전략 자동 전환 시스템을 구현해줘.
RL 학습 중이므로 핵심 컴포넌트에 '왜 필요한지' 주석 포함.

[Phase A,B: 시뮬레이션 모드만]
실거래 전환 기준 (4가지 동시 충족):
  시뮬레이션 ≥ 1,000 에피소드 + 샤프비율 ≥ 1.5
  최근 4주 실거래 승률 ≥ 55% + 자본 ≥ $1,000
미충족 시 ValueError 발생 (안전장치)

[StrategySelector 클래스]
select_strategy(hmm_regime, adx, fear_greed) → StrategyDecision
  레짐0(ADX>30): TREND_STRONG / 레짐1(ADX20~30): TREND_NORMAL
  레짐2(ADX<20): GRID / 레짐3(F&G<30): DCA / 레짐3(F&G≥30): HOLD
자본 배분 기본값: TREND 60% / GRID 20% / DCA 15% / HOLD 5%

[신규] 동적 자본 배분 연동 (Phase C):
  decay_monitor.get_weights() 호출 → 성과 기반 가중치 반환
  최종 배분 = 기본 배분 × 0.5 + 동적 배분 × 0.5
  모든 전략 DORMANT 시 → HOLD 100% 폴백

[신규] StrategyDecayMonitor 클래스 (Phase C):
  update_weekly_stats(): 매주 전략별 샤프/승률/손익비 계산 → strategy_decay_log 저장
  get_weights() → Dict[strategy_name, float]: 정규화된 가중치 반환
  check_dormant(): 샤프 < 0.5 4주 연속 → status='DORMANT', 텔레그램 알림
  check_revival(): DORMANT 전략 중 2주 연속 샤프 > 0.8 → 재활성화 후보
  trigger_reoptimize(strategy): Optuna 해당 전략 구간 n_trials=50 재실행
  get_status_report() → /decay 명령어 응답용 리포트 생성

[GridStrategy 클래스]
__init__(capital, current_price, atr):
  range_upper = current_price + atr*3
  range_lower = current_price - atr*3
  grid_levels=10, unit_size=grid_capital/10
  grid_prices = np.linspace(range_lower, range_upper, 10)
place_grid_orders(): 하단5개 지정가매수 + 상단5개 지정가매도
on_order_filled(level, side): 체결 시 반대 레벨에 자동 재주문
should_close(): ADX > 25 → True

[AdaptiveDCAStrategy 클래스]
__init__(capital):
  base_amount=capital*0.02, max_safety=5
  step_pct=-0.03, volume_scale=1.5, take_profit_pct=0.03
add_safety_order(current_price): 매수 금액 base × 1.5^n
avg_entry_price(): 가중 평균 매수가
check_take_profit(current_price): 평균가 +3% 시 전량 청산
fear_greed_multiplier(fg): fg<15→1.5 / fg<30→1.0 / else→0.5

[RL State(19) / Action(18)] Master Prompt 참조

[Reward 설계]
익절: +수익률 / 손절: -손실률×1.5
USDT전환 후 하락회피: +0.05 / 관망 후 기회손실(>10%): -0.05
그리드 레벨 체결: +체결수익 / 그리드 범위이탈 손실: -손실×1.5
DCA 익절: +수익률 / DCA 레짐3 매수회피: -0.03
피라미딩 성공: +추가수익 / 손절 강제발동: -0.1 추가
수수료 0.25% 매 거래마다 차감

[DQN 구현]
Experience Replay Buffer(capacity=10,000): # 상관관계 제거, 학습 안정화
Target Network(100스텝 동기화): # 이동 타겟 문제 방지
Epsilon-Greedy(1.0→0.1, decay=0.995): # 탐험vs활용 균형
device=cpu

TradingEnvironment (Gym 인터페이스):
  reset() / step(action) → next_state, reward, done, info
  DQN/PPO 모두 재사용 가능

PPO 전환 예시 주석:
  # from stable_baselines3 import PPO
  # model = PPO("MlpPolicy", env); model.learn(100_000)

[APScheduler 추가]
그리드 범위 재계산: 매 4시간
DCA Safety Order 모니터링: 매 5분

[구현 요구사항]
StrategySelector, GridStrategy, AdaptiveDCAStrategy 클래스
DQN + TradingEnvironment
시뮬레이션 vs 실거래 모드 분리
실거래 전환 기준 체크 함수
에피소드별 누적 보상 로깅
체크포인트 저장/불러오기
```

---

# Phase 6 — 실행 엔진 + 업비트 연동

```
전체 파이프라인 통합 실행 엔진을 구현해줘.
이전 모든 Phase 코드와 연결되도록.

[메인 루프 — APScheduler 5분마다]
1. WebSocket 버퍼에서 최신 현재가 읽기
2. [신규-v3] PairlistManager → 현재 활성 코인 목록 확인 (최대 30개)
3. [필수-v4] CandleBuilder → WebSocket 합성 캔들로 MarketState 갱신 (REST 호출 없음)
   일봉만 예외: 매일 04:00 REST 1회 업데이트
4. DataQualityChecker 7단계 품질 검증
5. 서킷브레이커 상태 확인 (최우선)
6. 코인 상관 클러스터링 확인 (Phase C)
7. Layer 0 HMM 레짐 감지 (Phase C)
8. StrategySelector → 전략 결정
9. 활성 코인 Layer 1 필터 병렬 (asyncio.gather)
10. 통과 코인 Layer 2 앙상블 예측 병렬
11. Kelly + VaR 리스크 예산 계산
12. Layer 3 RL 결정 (Phase C+) or 룰기반 (Phase A,B)
13. 최강 신호 코인 선정 (동시 보유 상한 5개 초과 시 최고 신호만)
    [신규-v7] 최종 진입 트리거 조건 (모든 조건 동시 충족 시에만 주문 실행)
    ① Layer 1 통과 (10개 조건 전부)
    ② Layer 2 앙상블 가중 평균 >= 0.62 AND 3모델 이상 합의
    ③ Kelly 포지션 사이즈 > 0 (음수 Kelly → 스킵)
    ④ VaR <= 자본 3% (초과 시 포지션 ×0.5 후 재판단)
    ⑤ tick_imbalance > 0.1 OR OBI > 0.2 (마이크로스트럭처 최소 확인)
    ⑥ 동시 보유 포지션 < 5개
    → 6개 조건 중 하나라도 미충족 시 해당 코인 해당 루프 스킵 (관망)
    → 충족 코인이 복수일 때: 앙상블 확률 × Kelly 사이즈 곱이 가장 큰 코인 1개 우선 선정
14. 전략별 주문 실행:
    TREND → [신규-v2] 0~90초 랜덤 지연 후 시장가 매수/매도 (order.py)
    GRID → 지정가 그리드 주문 세팅 (grid.py), 즉시 실행
    DCA → 시장가 Safety Order (dca.py), 즉시 실행
    단, 서킷브레이커/손절/익절 청산은 항상 즉시 실행 (지연 없음)
15. SQLite 저장 → RL 보상 피드백
16. 텔레그램 이벤트 큐 알림

[포지션 모니터링 루프 — 1분마다]
보유 포지션별:
  ATR 트레일링 스탑 업데이트
  SuperTrend 반전 → 즉시 청산
  조정/반전 판별 실행
  부분 익절 조건 체크 (+10%, +20%)
  피라미딩 조건 체크
그리드 체결 확인 + 자동 재주문
DCA Safety Order 조건 체크

[신규-v8] 청산 주문 실행 방식 명세 (손절/익절 모두 적용)
  일반 청산 (익절, 조정→반전 감지):
    SmartOrderRouter 경유 — spread/잔량 체크 후 지정가 우선
    지정가 미체결 10초 → 시장가 전환 (진입보다 빠른 전환, 수익 보호 우선)
  강제 손절 (서킷브레이커, -7% 하드 손절, Level 1~3):
    무조건 시장가 즉시 실행 (SmartOrderRouter 우회)
    이유: 폭락 중 지정가 대기는 더 큰 손실 위험
  폭락 중 시장가 슬리피지 대비:
    손절 주문 수량을 2회 분할 실행 (50% + 50%, 3초 간격)
    → 단일 대형 시장가 주문의 호가 충격 최소화
    → 단, 서킷브레이커 Level 2 이상(전량 USDT 전환)은 분할 없이 즉시 전량

[서킷브레이커 상시 감시 — 10초마다]
Level 1~5 조건 체크 + 발동 시 즉시 실행

[업비트 주문]
TREND/DCA: [신규-v5] 조건부 지정가/시장가 선택 (SmartOrderRouter)
  호가창 유동성 체크 후 주문 방식 결정:
    조건: spread < 0.1% AND 1호가_잔량 > 주문금액×3 AND trade_velocity < 2.0
    → 지정가 매수 (maker 수수료 0.05%, 슬리피지 0%)
    조건 미충족 (호가 얇음 or 급등 중):
    → 시장가 (즉각 체결 우선, taker 수수료 0.05%)
  이유: 호가창이 얇을 때 시장가 진입 시 슬리피지 0.3~0.5% 발생 가능
        → 수수료+슬리피지 합산 시 예상 비용 초과, 백테스트와 실거래 괴리 원인

  [신규-v6] 부분 체결(Partial Fill) 처리 — PartialFillHandler
  문제: 지정가 주문 후 일부만 체결되고 가격이 이탈할 경우
        → 미체결 잔량 무시 시: 포지션 사이즈 부족 (Kelly 계산 어긋남)
        → 원래 금액 전체를 시장가 재발주 시: 이중 체결 → 목표 포지션의 1.3~1.5배 초과
  처리 절차:
    Step 1: 지정가 주문 후 30초 대기
    Step 2: 부분 체결 확인
      → 체결률 >= 80%: 미체결 잔량 취소 후 완료 처리 (허용 오차 내)
      → 체결률 30~80%: 기존 지정가 주문 즉시 취소(cancel_order) → 미체결 잔량만큼만 시장가 Sweep
      → 체결률 < 30%: 기존 지정가 주문 즉시 취소 → 전략 신호 재평가 후 재진입 여부 결정
    Step 3: 최종 체결 수량을 Kelly 포지션 사이징에 역산 반영
    핵심: 항상 "취소 후 재발주" 순서 보장 — 취소 확인 전 시장가 발주 금지 (이중 주문 방지)
    업비트 API: DELETE /v1/order (취소) → GET /v1/order (취소 확인) → POST /v1/orders (재발주)

  [신규-v7] DELETE 레이스 컨디션 예외 처리
  문제: DELETE 요청이 서버에 도달하는 0.1초 사이에 타인의 시장가로 내 지정가가 100% 체결되는 경우
        → 업비트 API는 이미 체결된 주문 취소 시 HTTP 400 에러 반환
        → 예외 처리 없으면 봇 크래시 또는 메인 루프 중단 위험
  처리 방식:
    try:
        await cancel_order(order_id)
    except UpbitAPIError as e:
        if e.code == 400 and "already done" in e.message:
            # 취소 실패가 아닌 100% 체결로 인한 자연 종료
            filled_qty = await get_order(order_id).executed_volume
            update_position_state(filled_qty)  # 상태를 완전 체결로 업데이트
            return OrderResult.FULLY_FILLED   # 정상 처리로 분기
        else:
            raise  # 진짜 에러는 재raise → 서킷브레이커 Level 4 연동
  결과: 취소 찰나 체결 → 봇 크래시 없이 "100% 체결 완료"로 상태 우아하게 업데이트
  SmartOrderRouter + PartialFillHandler 통합 클래스 (execution/order.py)

GRID: 지정가 필수 (그리드 특성, 기존 유지)
주문 전 잔고 확인 / 실패 시 3회 재시도
API 오류 → 서킷브레이커 Level 4 연동
저유동성(24h<1억원) 자동 제외
동시 다중 매수 간격: 최소 30초

[페이퍼 트레이딩 병렬 실행]
실전 봇과 동일 신호로 가상 거래 병렬 실행
[신규-v8] 페이퍼↔실거래 정량 비교 지표 3개 (매일 SQLite 저장):
  ① 신호 일치율: 동일 루프에서 페이퍼/실거래 모두 진입한 비율
     → 85% 미만 시 알림 (실거래에서만 스킵되는 경우 원인 분석)
  ② 체결가 괴리: (실거래 평균 체결가 - 페이퍼 기준가) / 기준가 × 100
     → 지속적으로 -0.2% 초과 시 슬리피지 과다 → SmartOrderRouter 조건 재조정
  ③ 타이밍 슬리피지: 신호 발생 타임스탬프 → 실제 체결 완료 타임스탬프 차이 (초)
     → 평균 5초 초과 시 네트워크/API 지연 점검
비교 결과 주간 리포트 + /paper 텔레그램 명령어로 조회 가능

[DRY_RUN 모드]
DRY_RUN=true: 전체 시뮬레이션 (실제 주문 없음)
DRY_RUN=false: 실제 주문

[하드가드 우선순위 강제 적용]
Master Prompt 하드가드 (1)~(9) 순서대로
모든 가드가 RL/룰 결정보다 우선

[필수-v4] 비동기 격리 원칙 (execution/engine.py 구현 시 필수 준수)
FinBERT 추론:
  process_pool = ProcessPoolExecutor(max_workers=1)  # 봇 시작 시 1회 초기화
  loop = asyncio.get_event_loop()
  sentiment_result = await loop.run_in_executor(process_pool, finbert_batch, texts)
  # 메인 루프는 FinBERT 작업 중에도 서킷브레이커/WebSocket 계속 처리
HMM 재학습, Walk-Forward 등 무거운 연산: 동일 패턴 적용 (별도 프로세스 격리)
원칙: 메인 asyncio 루프에서 1초 이상 소요되는 동기 작업은 반드시 run_in_executor로 분리
위반 시: 서킷브레이커 발동 타이밍 누락, WebSocket 버퍼 overflow, 포지션 손실 위험
```

---

# Phase 7 — 백테스트 + Walk-Forward + Hyperopt

```
실전 투입 전 검증 시스템을 구현해줘.

[데이터]
pyupbit: 동적 페어리스트 기준 (백테스트 시점별 실제 코인 목록 사용)
  5분봉 6개월, 1시간봉 2년, 일봉 3년
시장지수 히스토리: CoinGecko, Alternative.me
Pessimistic: 수익-0.25%, 손실+0.25%
[신규-v3] 일봉 피처는 전 구간에 걸쳐 shift(1) 적용 검증 후 사용

[신규-v3] Lookahead Bias 검증 (backtest/lookahead.py)
LookaheadBiasChecker:
  검증 방법: 전체 백테스트 실행 후 각 신호 시점에서
    해당 시점 이후 데이터가 피처에 사용되었는지 역추적 검사
  검사 대상: 모든 ML 피처, 특히 일봉 EMA/RSI (shift 누락 위험)
  검사 대상: LSTM/GRU의 시퀀스 입력이 미래 캔들을 포함하지 않는지
  발견 시: 즉시 ValueError 발생 + 오염된 피처명 로그 출력
  통과 기준: 0개 오염 피처
  APScheduler: 매 Walk-Forward 사이클 시작 전 자동 실행

[신규-v3] 생존 편향(Survivorship Bias) 처리 (backtest/walk_forward.py)
SurvivourshipHandler:
  백테스트 각 시점(t)에서 해당 시점의 실제 업비트 상장 코인 목록 사용
  현재 상위 코인 목록을 과거에 그대로 적용하는 것 금지
  구현: SQLite coin_history 테이블 — 날짜별 업비트 KRW 마켓 코인 목록 저장
  데이터 수집: pyupbit.get_tickers() 결과를 매일 스냅샷으로 저장 (Phase A부터)
  백테스트 시 해당 날짜 스냅샷 조회 → 해당 시점 유효 코인만 대상으로 평가
  최소 6개월 스냅샷 누적 후 Walk-Forward 백테스트 실행 가능

[신규-v3] Monte Carlo 검증 (backtest/monte_carlo.py)
MonteCarloValidator:
  방법: 실제 거래 결과(손익 시퀀스)의 순서를 1,000회 무작위 셔플
  각 셔플에서 샤프비율, 최대낙폭, 최종 수익률 계산
  통과 기준:
    실제 샤프비율이 셔플 분포의 상위 5% 이상 (p-value < 0.05)
    즉, 1,000회 셔플 중 실제보다 높은 샤프 나온 횟수 < 50회
  실패 시: "엣지가 통계적으로 랜덤과 구분 불가" 경고 + 실전 투입 차단
  출력: 신뢰구간 95% 샤프비율 범위, p-value, 엣지 신뢰도 점수
  Phase B Walk-Forward 완료 후 자동 실행

[전략별 구간 분리 백테스트]
강세장 (2024.10~2025.01): 추세 추종 전략 평가
횡보장 (2023.01~2023.06): 그리드 전략 평가
하락장 (2022.05~2022.12): DCA 전략 평가
각 구간별 샤프비율, 낙폭, 승률 별도 계산
하락장 구간 낙폭 < 10% 필수 (하락 방어력)
[신규-v3] 각 구간의 코인 목록은 해당 시점 생존 편향 처리 적용

[Walk-Forward 최적화]
WalkForwardOptimizer:
  6개월 학습 → 1개월 테스트 → 전진 반복
  각 구간 파라미터 재최적화
  OOS 성과 집계
  과적합 판별: IS 샤프 vs OOS 샤프 비율 < 0.5 → 경고
  [신규-v3] 각 사이클 시작 전 Lookahead Bias 검증 자동 실행

[Optuna Hyperopt]
탐색: RSI기간(5~30), EMA조합, ADX임계값(15~35),
      손절/익절비율, Kelly계수(0.1~0.5), ATR배수(1.5~4.0),
      그리드레벨수(5~20), DCA step_pct(-0.02~-0.05),
      ensemble_threshold(0.55~0.75)  # [신규-v9] 앙상블 임계값 자동 최적화
목적함수: Walk-Forward OOS 샤프비율 최대화
n_trials=200 (M4에서 1~2시간)
Pruner: MedianPruner (성능 낮은 trial 조기 종료)

[검증 지표]
승률 / 샤프비율 / 최대낙폭 / 손익비
전략별 기여도 분석 (추세/그리드/DCA 각각)
Layer 1 필터 기여도 (손실 거래 차단율)
앙상블 모델별 정확도 비교
[신규-v3] Monte Carlo p-value / 엣지 신뢰도 점수
[신규-v3] Lookahead Bias 오염 피처 수 (0개 목표)

[실전 전환 기준 — 7가지 모두 통과]
샤프비율 > 1.5 (전 구간 평균)
최대낙폭 < 20%
승률 > 55%
하락장 구간 낙폭 < 10%
Optuna 최적 파라미터 적용 후 재검증 통과
[신규-v3] Lookahead Bias 오염 피처 0개 확인
[신규-v3] Monte Carlo p-value < 0.05 (엣지 통계 유의성 확인)

[시각화 — matplotlib]
자본 곡선 (구간별 색상 구분)
전략별 기여도 비교 차트
Walk-Forward OOS 성과 분포
Optuna 파라미터 중요도 차트
RL 학습 곡선
[신규-v3] Monte Carlo 분포도 (실제 샤프 vs 셔플 분포 히스토그램)
```

---

# Phase 8 — 텔레그램 봇

```
퀀트 봇과 연동되는 텔레그램 봇을 구현해줘.

[알림 메시지 포맷]
매수: 🟢 매수체결 / 코인,가격,금액(자본%) / 앙상블확률,합의수 / 전략타입 / 포지션N/3
매도: 🔴 매도체결 / 매수가→매도가 / 손익%,원 / 사유
그리드: ⚡ 그리드체결 / 코인,레벨 / 누적수익
DCA:   💧 DCA매수 / Safety Order N / 평균단가,현재가격
서킷: 🚨 서킷브레이커 Level N / 사유 / 조치 / 재개예정
재학습: 🔄 앙상블재학습완료 / 정확도 / Walk-Forward샤프
일간리포트(밤11시): 📊 총거래/실현손익/승률/자본 / 전략별기여도 / 페이퍼vs실전 / 디스크잔여

[전체 명령어 20개]
/status    봇상태+포지션 / /balance 잔고+미실현손익
/scan      활성코인 스캔결과 (최대 30개) / /strategy 현재전략+자본배분
/stop      신규매수중단 / /emergency 전량매도+중단(/confirm필요)
/report    오늘거래내역 / /layer1 현재시장필터
/ensemble  앙상블예측(모델별) / /hmm 현재레짐+신뢰도
/kelly     Kelly f*+VaR값 / /grid 현재그리드상태
/dca       현재DCA포지션 / /retrain 수동재학습트리거
/hyperopt  수동Hyperopt실행 / /quality 데이터품질리포트
/storage   디스크사용량 / /vacuum 수동VACUUM
/cleanup   전체정리즉시 / /paper 페이퍼vs실전비교
/mode dry|live DRY_RUN전환 / /phase 현재구현Phase확인
[신규-v2] /decay   전략별 롤링샤프 + DORMANT 전략 목록
[신규-v2] /kimchi  현재 김치프리미엄 + 최근 24h 추이
[신규-v3] /pairs   현재 활성 페어리스트 + 다음 갱신 예정시각
[신규-v3] /montecarlo  최근 Monte Carlo 검증 결과 요약

[구현 요구사항]
python-telegram-bot 20.x (async)
asyncio.Queue로 퀀트 봇과 통신
봇토큰+ChatID: .env 전용
내 ChatID만 수신 (화이트리스트)
/emergency 이중확인 (/confirm)
자동재연결 + 3회 재시도
인라인 키보드 메뉴화
퀀트 봇과 같은 asyncio 루프
```

---

# Phase 9 — StorageManager + Streamlit 대시보드

```
스토리지 자동 관리 + Streamlit 대시보드를 구현해줘.

[StorageManager 클래스]
1. cleanup_candles(): 5분봉>6개월, 1h>2년, 1d>3년, 예측>6개월, 로그>3개월 DELETE
2. vacuum_database(): VACUUM (파일 크기 실제 회수, DB크기×2 임시공간 필요, 실행 중 봇 일시중단)
3. archive_old_trades(): 거래기록 1년+ → archive.db 이동 (영구보관, 세금용)
4. cleanup_model_checkpoints(): 모델별 최신 3개만 보관
5. cleanup_logs(): logs/ 30일 초과 삭제
6. cleanup_backtest_results(): 최신10개+90일치만 보관
7. check_disk_usage():
   여유<50GB: ⚠️ 텔레그램 경고
   여유<20GB: 🚨 텔레그램 긴급
   DB>10GB: ⚠️ VACUUM 권장

[APScheduler 스케줄]
매일 03:00: cleanup_candles + cleanup_logs
매주 일요일 03:30: vacuum_database
매주 일요일 04:00: cleanup_model_checkpoints
매월 1일 04:30: cleanup_backtest_results
매시간 :00: check_disk_usage
매주 일요일 02:00: archive_old_trades

[Streamlit 대시보드 — localhost만]
streamlit run monitoring/dashboard.py
자동새로고침: 30초, 데이터소스: SQLite 실시간 쿼리

화면 구성:
  상단: 자본+오늘손익+승률+현재전략 카드
  차트1: 자본 곡선 (전체기간)
  차트2: 오늘 거래 타임라인 (전략별 색상)
  차트3: 전략별 기여도 (추세/그리드/DCA) + [신규-v2] 동적 가중치 현황
  차트4: 앙상블 모델별 정확도
  차트5: HMM 레짐 히스토리
  차트6: 데이터 품질 점수 트렌드
  [신규-v2] 차트7: 김치프리미엄 24h 추이 + 과열/역프리미엄 경계선 표시
  [신규-v2] 차트8: 전략별 롤링 샤프비율 (4주) + DORMANT 전략 하이라이트
  [신규-v3] 차트9: Monte Carlo 분포 + 실제 샤프 위치 + p-value 표시
  [신규-v3] 차트10: 활성 페어리스트 (현재 스캔 코인 + 거래량 순위)
  테이블: 현재 포지션 목록
  테이블: 최근 20거래
  하단: 디스크 사용량 게이지 + 다음 정리 예정시각
```

---

# 실전 전환 체크리스트

```
DRY_RUN=false 전환 기준 (7가지 모두 충족 필수):
  ✅ Walk-Forward 샤프비율 > 1.5 (전 구간 평균)
  ✅ 최대 낙폭 < 20%
  ✅ 승률 > 55%
  ✅ 하락장 구간 낙폭 < 10%
  ✅ DRY_RUN 48시간 페이퍼 트레이딩 정상 작동 확인
  ✅ [신규-v3] Lookahead Bias 오염 피처 0개 확인
  ✅ [신규-v3] Monte Carlo p-value < 0.05 (엣지 통계 유의성 확인)

Phase A 시작 전 준비:
  ✅ 업비트 API 키 발급 (Open API)
  ✅ 텔레그램 봇 생성 (BotFather 토큰)
  ✅ .env 파일 생성 (UPBIT_ACCESS_KEY, UPBIT_SECRET_KEY, TELEGRAM_TOKEN, CHAT_ID)
  ✅ Python 3.11+ 설치
  ✅ Google Colab 계정 준비 (Phase B LSTM 학습)
  ✅ [신규-v3] coin_history 스냅샷 수집 시작 (생존 편향 처리용 — Phase B 이전 최소 1개월치 필요)
```

---

# 최종 벤치마크 점수표

| 항목 | 우리 설계 | Freqtrade | VishvaAlgo | 3Commas | 점수 |
|---|---|---|---|---|---|
| 리스크 관리 | Kelly×HMM×VaR+서킷브레이커5단계 | 기본 | 중간 | 중간 | ⭐⭐⭐⭐⭐ |
| 레짐 감지 | HMM+ADX+폴백 | 없음 | HMM | 없음 | ⭐⭐⭐⭐⭐ |
| 앙상블 모델 | 4모델 레짐별 분리 | ML단일 | 복합 | 없음 | ⭐⭐⭐⭐⭐ |
| 전략 다양성 | 추세추종+그리드+DCA 자동전환 | 다수전략 | 다수전략 | DCA+그리드 | ⭐⭐⭐⭐⭐ |
| **전략 포화 방어** | **DecayMonitor+동적가중치+타이밍노이즈** | **없음** | **없음** | **없음** | **⭐⭐⭐⭐⭐** |
| **업비트 특화 피처** | **김치프리미엄+OBI+마이크로스트럭처 (38개 피처)** | **해당없음** | **해당없음** | **해당없음** | **⭐⭐⭐⭐⭐** |
| 데이터 품질 | 7단계+IsolationForest | 있음 | 있음 | 해당없음 | ⭐⭐⭐⭐⭐ |
| Hyperopt | Optuna+Walk-Forward | 완비 | 있음 | 없음 | ⭐⭐⭐⭐⭐ |
| **백테스트 신뢰도** | **Lookahead검증+생존편향+MonteCarlo** | **부분** | **없음** | **없음** | **⭐⭐⭐⭐⭐** |
| **주문 실행 품질** | **SmartOrderRouter + PartialFillHandler** | **없음** | **없음** | **있음** | **⭐⭐⭐⭐⭐** |
| **스캔 커버리지** | **동적 30개 페어리스트 (KRW 상위)** | **수동설정** | **수동설정** | **수동설정** | **⭐⭐⭐⭐⭐** |
| API 안정성 | WebSocket+캐시+레이턴시감시 | 있음 | 중간 | 해당없음 | ⭐⭐⭐⭐⭐ |
| 스토리지 관리 | 자동VACUUM+모니터링 | 수동 | 미반영 | 해당없음 | ⭐⭐⭐⭐⭐ |
| 법률 준수 | 가상자산이용자보호법 반영 | 해당없음 | 해당없음 | 해당없음 | ⭐⭐⭐⭐⭐ |
| **종합** | | | | | **⭐⭐⭐⭐⭐ (5/5)** |
