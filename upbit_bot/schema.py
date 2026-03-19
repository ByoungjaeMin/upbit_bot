"""
schema.py — 공통 dataclass + SQLite 15개 테이블 DDL

Phase 1 설계서 기반 (docs/upbit_quant_v9.md)
레이어 간 데이터 계약 + SQLite 스키마 단일 관리.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

# ---------------------------------------------------------------------------
# 공통 dataclass 8개
# ---------------------------------------------------------------------------

@dataclass
class RawMarketData:
    """WebSocket 수신 원본.

    ticker 이벤트(현재가)와 trade 이벤트(체결) 모두 이 구조로 정규화.
    ask_bid: 'ASK'(매수 체결) | 'BID'(매도 체결) — tick_imbalance 계산용.
    stream_type: 'ticker' | 'trade'
    """
    coin: str
    timestamp: datetime
    trade_price: float
    trade_volume: float
    ask_bid: str                   # 'ASK' | 'BID'
    sequential_id: int
    stream_type: str               # 'ticker' | 'trade'
    # ticker 전용 (trade 이벤트에는 None)
    acc_trade_volume_24h: float | None = None
    acc_trade_price_24h: float | None = None
    change_rate: float | None = None


@dataclass
class MarketState:
    """멀티타임프레임 + 온체인 + 감성 통합 출력 (ML 입력 38개 피처 포함).

    CandleBuilder가 캔들 완성 시 생성. 총 38개 피처 커버.
    일봉 피처는 반드시 df_daily.shift(1) 적용 후 채워야 함 (Lookahead Bias 방지).
    """
    coin: str
    timestamp: datetime

    # 5분봉 OHLCV (5)
    open_5m: float = 0.0
    high_5m: float = 0.0
    low_5m: float = 0.0
    close_5m: float = 0.0
    volume_5m: float = 0.0

    # 5분봉 기술지표 (9)
    rsi_5m: float = 50.0
    macd_5m: float = 0.0
    macd_signal_5m: float = 0.0
    bb_upper_5m: float = 0.0
    bb_lower_5m: float = 0.0
    ema7_5m: float = 0.0
    ema25_5m: float = 0.0
    ema99_5m: float = 0.0
    volume_ratio_5m: float = 1.0   # 현재 거래량 / 20기간 평균

    # 1시간봉 피처 (5)
    rsi_1h: float = 50.0
    ema20_1h: float = 0.0
    ema50_1h: float = 0.0
    macd_1h: float = 0.0
    trend_dir_1h: int = 0          # +1 상승 / -1 하락 / 0 중립

    # 일봉 피처 (4) — shift(1) 적용 필수
    ema50_1d: float = 0.0
    ema200_1d: float = 0.0
    rsi_1d: float = 50.0
    trend_encoding_1d: int = 0     # +1 골든크로스 / -1 데드크로스 / 0 중립

    # 시장지수 (3)
    fear_greed: float = 50.0       # 0~100
    btc_dominance: float = 50.0    # %
    altcoin_season: float = 50.0   # 0~100

    # 온체인 (2)
    exchange_inflow: float = 0.0
    exchange_outflow: float = 0.0

    # 감성 (2)
    sentiment_score: float = 0.0   # -1~+1
    sentiment_confidence: float = 0.0  # 0~1

    # 추세·레짐 (4)
    adx_5m: float = 0.0
    adx_1h: float = 0.0
    supertrend_signal: int = 0     # +1 | -1
    hmm_regime: int = -1           # 0~3, Phase C 이전은 -1

    # 업비트 특화 (3)
    kimchi_premium: float = 0.0    # -5~+10 %
    obi: float = 0.0               # -1~+1 오더북 불균형
    top5_concentration: float = 0.5  # 0~1 상위5호가 집중도

    # 마이크로스트럭처 (3) — CandleBuilder WebSocket 체결 집계
    tick_imbalance: float = 0.0    # -1~+1
    orderbook_wall_ratio: float = 1.0  # 0~20+
    trade_velocity: float = 1.0    # 0~5+ 체결 속도 가속도

    # 메타 (서킷브레이커·필터용)
    atr_5m: float = 0.0
    supertrend_dir: int = 0        # +1 | -1
    api_latency_ms: float = 0.0


@dataclass
class FilterResult:
    """Layer 1 룰 기반 필터 출력.

    tradeable=False 이면 Layer 2 호출 불필요.
    active_warnings: 발동된 경고 코드 목록 (예: 'CIRCUIT_L1', 'LOW_ADX').
    """
    coin: str
    timestamp: datetime
    tradeable: bool
    regime_strategy: str           # 'TREND_STRONG' | 'TREND_NORMAL' | 'GRID' | 'DCA' | 'HOLD'
    signal_multiplier: float       # 0.5 | 0.7 | 0.8 | 1.0
    adx_value: float
    supertrend_direction: int      # +1 | -1
    atr_value: float
    active_warnings: list[str] = field(default_factory=list)
    pullback_detected: bool = False
    reversal_detected: bool = False
    api_latency_ok: bool = True
    daily_loss_pct: float = 0.0    # 당일 누적 손실률 (음수)
    circuit_breaker_level: int = 0  # 0: 정상, 1~5: 발동 레벨


@dataclass
class EnsemblePrediction:
    """Layer 2 ML 앙상블 예측기 출력.

    per_model_probs: {'xgb': 0.7, 'lgb': 0.65, 'lstm': 0.6, 'gru': 0.58}
    consensus_count: threshold 초과 모델 수 (≥3 이어야 signal_confirmed=True)
    [v9] 초기 threshold=0.62, Optuna로 0.55~0.75 탐색.
    """
    coin: str
    timestamp: datetime
    per_model_probs: dict[str, float] = field(default_factory=dict)
    weighted_avg: float = 0.0
    consensus_count: int = 0
    signal_confirmed: bool = False
    hmm_regime: int = -1
    hmm_confidence: float = 0.0
    threshold_used: float = 0.62   # 실제 적용된 임계값


@dataclass
class StrategyDecision:
    """StrategySelector 출력.

    capital_allocation: 0.0~1.0 (총 자본 대비 비율)
    grid_params / dca_params: 해당 전략 미선택 시 None.
    """
    coin: str
    timestamp: datetime
    strategy_type: str             # 'TREND_STRONG' | 'TREND_NORMAL' | 'GRID' | 'DCA' | 'HOLD'
    capital_allocation: float      # 0.0~1.0
    grid_params: dict[str, Any] | None = None
    # grid_params 예: {'lower': float, 'upper': float, 'levels': int, 'unit_krw': float}
    dca_params: dict[str, Any] | None = None
    # dca_params 예: {'base_amount': float, 'safety_orders': int, 'step_pct': float}
    dynamic_weight: float = 1.0    # StrategyDecayMonitor 연동 동적 가중치


@dataclass
class RiskBudget:
    """Kelly + VaR 포지션 사이징 출력.

    kelly_f: 순수 Kelly 분수 (다중자산 PyPortfolioOpt)
    hmm_adjusted_f: HMM 레짐 신뢰도 곱한 값
    var_adjusted_f: VaR 오버레이 후 최종 분수
    final_position_size: 실제 투입 KRW 금액
    coin_group: ATR 변동성 그룹 ('HIGH' | 'LOW' | 'NORMAL')
    """
    coin: str
    timestamp: datetime
    kelly_f: float                 # 원시 Kelly
    hmm_adjusted_f: float          # × HMM 신뢰도
    var_adjusted_f: float          # VaR 오버레이 후
    final_position_size: float     # KRW 금액
    coin_group: str = 'NORMAL'     # 'HIGH' | 'LOW' | 'NORMAL'
    consecutive_losses: int = 0    # 연속 손실 횟수 (패널티 반영 용도)
    var_95: float = 0.0            # 역사적 VaR 95% (자본 대비 %)


@dataclass
class TradeDecision:
    """Layer 3 / 메인 루프 최종 매매 결정.

    action: 0~17 (Master Prompt Action 정의와 동일 인덱스)
    trailing_stop_price: 트레일링 스탑 발동 가격 (None=미사용)
    partial_exit_ratio: 부분 익절 비율 (None=전량)
    is_dry_run: trade_count<200 이면 True 강제
    entry_delay_sec: 0~90초 랜덤 지연 (front-running 방어, 서킷·청산 시 0)
    """
    coin: str
    timestamp: datetime
    action: int                    # 0~17
    target_coin: str               # 매수 대상 코인 (action에 따라 현재 보유 코인과 다를 수 있음)
    position_size: float           # KRW 금액
    trailing_stop_price: float | None = None
    partial_exit_ratio: float | None = None
    is_dry_run: bool = True
    entry_delay_sec: float = 0.0   # 랜덤 지연 (import random; random.uniform(0, 90))
    strategy_type: str = ''
    ensemble_score: float = 0.0    # 참조용 앙상블 점수


@dataclass
class TelegramEvent:
    """텔레그램 메시지 큐 항목.

    priority: 1(즉시) > 2(일반) > 3(집계 후 배치)
    parse_mode: 'HTML' | 'Markdown'
    """
    event_type: str                # 'TRADE' | 'CIRCUIT' | 'ALERT' | 'REPORT' | 'ERROR'
    message: str
    timestamp: datetime
    priority: int = 2              # 1=즉시, 2=일반, 3=배치
    chat_id: int | None = None     # None → 기본 설정 chat_id 사용
    parse_mode: str = 'HTML'
    disable_notification: bool = False  # 야간 저우선순위 시 True


# ---------------------------------------------------------------------------
# SQLite 15개 테이블 DDL
# ---------------------------------------------------------------------------

# 각 항목: (테이블명, DDL 문자열, 인덱스 목록, 보관 기간 설명)
# cache.py의 init_db()에서 순서대로 실행할 것.

SCHEMA_VERSION = 9  # upbit_quant_v9 기준

TABLE_DDLS: dict[str, str] = {

    # ------------------------------------------------------------------
    # 1. candles_5m — 5분봉 완성 캔들 (WebSocket 합성, REST 아님)
    #    보관: 3개월 (StorageManager 자동 삭제)
    # ------------------------------------------------------------------
    "candles_5m": """
    CREATE TABLE IF NOT EXISTS candles_5m (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        coin            TEXT    NOT NULL,
        timestamp       TEXT    NOT NULL,           -- ISO8601 UTC
        open            REAL    NOT NULL,
        high            REAL    NOT NULL,
        low             REAL    NOT NULL,
        close           REAL    NOT NULL,
        volume          REAL    NOT NULL,
        -- pandas-ta 지표
        rsi             REAL,
        macd            REAL,
        macd_signal     REAL,
        macd_hist       REAL,
        bb_upper        REAL,
        bb_lower        REAL,
        bb_mid          REAL,
        ema7            REAL,
        ema25           REAL,
        ema99           REAL,
        adx             REAL,
        supertrend      REAL,
        supertrend_dir  INTEGER,                    -- +1 | -1
        atr             REAL,
        volume_ratio    REAL,                       -- volume / 20기간 평균
        -- 마이크로스트럭처 (CandleBuilder WebSocket 집계)
        tick_imbalance  REAL,                       -- -1~+1
        trade_velocity  REAL,                       -- 0~5+
        UNIQUE(coin, timestamp)
    )
    """,

    # ------------------------------------------------------------------
    # 2. candles_1h — 1시간봉 (5분봉 12개 집계 또는 REST 초기 로딩)
    #    보관: 1년
    # ------------------------------------------------------------------
    "candles_1h": """
    CREATE TABLE IF NOT EXISTS candles_1h (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        coin            TEXT    NOT NULL,
        timestamp       TEXT    NOT NULL,
        open            REAL    NOT NULL,
        high            REAL    NOT NULL,
        low             REAL    NOT NULL,
        close           REAL    NOT NULL,
        volume          REAL    NOT NULL,
        rsi             REAL,
        ema20           REAL,
        ema50           REAL,
        macd            REAL,
        macd_signal     REAL,
        adx             REAL,
        trend_dir       INTEGER,                    -- +1 | -1 | 0
        UNIQUE(coin, timestamp)
    )
    """,

    # ------------------------------------------------------------------
    # 3. candles_1d — 일봉 (매일 04:00 REST 갱신, shift(1) 적용 소스)
    #    보관: 영구 (학습 데이터 소스)
    # ------------------------------------------------------------------
    "candles_1d": """
    CREATE TABLE IF NOT EXISTS candles_1d (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        coin            TEXT    NOT NULL,
        timestamp       TEXT    NOT NULL,           -- 날짜 (UTC 00:00)
        open            REAL    NOT NULL,
        high            REAL    NOT NULL,
        low             REAL    NOT NULL,
        close           REAL    NOT NULL,
        volume          REAL    NOT NULL,
        ema50           REAL,
        ema200          REAL,
        rsi             REAL,
        trend_encoding  INTEGER,                    -- +1 골든크로스 / -1 데드크로스 / 0
        UNIQUE(coin, timestamp)
    )
    """,

    # ------------------------------------------------------------------
    # 4. market_indices — Fear&Greed, BTC 도미넌스, 환율 등 전역 지표
    #    보관: 1년
    # ------------------------------------------------------------------
    "market_indices": """
    CREATE TABLE IF NOT EXISTS market_indices (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp           TEXT    NOT NULL UNIQUE,
        fear_greed          REAL,                   -- 0~100
        btc_dominance       REAL,                   -- %
        btc_market_cap_usd  REAL,
        altcoin_season      REAL,                   -- 0~100
        funding_rate        REAL,                   -- 8시간 펀딩비 (CoinGlass)
        usd_krw_rate        REAL                    -- 한국은행 환율 (일1회)
    )
    """,

    # ------------------------------------------------------------------
    # 5. onchain_data — CryptoQuant 거래소 유입/유출량
    #    보관: 6개월
    # ------------------------------------------------------------------
    "onchain_data": """
    CREATE TABLE IF NOT EXISTS onchain_data (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        coin             TEXT    NOT NULL,
        timestamp        TEXT    NOT NULL,
        exchange_inflow  REAL,
        exchange_outflow REAL,
        net_flow         REAL,                      -- outflow - inflow (양수=거래소 이탈)
        UNIQUE(coin, timestamp)
    )
    """,

    # ------------------------------------------------------------------
    # 6. sentiment_log — VADER + FinBERT 감성 분석 결과
    #    보관: 3개월
    # ------------------------------------------------------------------
    "sentiment_log": """
    CREATE TABLE IF NOT EXISTS sentiment_log (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        coin                TEXT    NOT NULL,
        timestamp           TEXT    NOT NULL,
        vader_score         REAL,                   -- -1~+1
        finbert_score       REAL,                   -- -1~+1 (±0.3 미만 시만 호출)
        combined_score      REAL,                   -- 최종 감성 점수
        confidence          REAL,                   -- 0~1
        news_count          INTEGER,
        veto_triggered      INTEGER DEFAULT 0,      -- 감성 비토 발동 여부 (0/1)
        UNIQUE(coin, timestamp)
    )
    """,

    # ------------------------------------------------------------------
    # 7. ensemble_predictions — Layer 2 ML 앙상블 예측 로그
    #    보관: 6개월
    # ------------------------------------------------------------------
    "ensemble_predictions": """
    CREATE TABLE IF NOT EXISTS ensemble_predictions (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        coin             TEXT    NOT NULL,
        timestamp        TEXT    NOT NULL,
        xgb_prob         REAL,
        lgb_prob         REAL,
        lstm_prob        REAL,
        gru_prob         REAL,
        weighted_avg     REAL    NOT NULL,
        consensus_count  INTEGER NOT NULL,
        signal_confirmed INTEGER NOT NULL DEFAULT 0,
        hmm_regime       INTEGER DEFAULT -1,        -- -1: Phase A/B (HMM 미적용)
        hmm_confidence   REAL    DEFAULT 0.0,
        threshold_used   REAL    DEFAULT 0.62,
        UNIQUE(coin, timestamp)
    )
    """,

    # ------------------------------------------------------------------
    # 8. trades — 체결 거래 기록 (영구보관, 법률 5년 의무)
    #    절대 삭제 금지. StorageManager VACUUM 대상에서 제외.
    # ------------------------------------------------------------------
    "trades": """
    CREATE TABLE IF NOT EXISTS trades (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        coin            TEXT    NOT NULL,
        timestamp       TEXT    NOT NULL,
        action          INTEGER NOT NULL,           -- TradeDecision.action (0~17)
        side            TEXT    NOT NULL,           -- 'BUY' | 'SELL'
        price           REAL    NOT NULL,
        volume          REAL    NOT NULL,           -- 코인 수량
        krw_amount      REAL    NOT NULL,           -- 주문 KRW 금액
        fee             REAL    DEFAULT 0.0,        -- 수수료 KRW
        slippage        REAL    DEFAULT 0.0,        -- 체결가 괴리율 (%)
        strategy_type   TEXT,
        kelly_f         REAL,
        position_size   REAL,
        pnl             REAL,                       -- 실현 손익 KRW (매도 시)
        pnl_pct         REAL,                       -- 실현 손익 % (매도 시)
        is_dry_run      INTEGER NOT NULL DEFAULT 1, -- DRY_RUN 여부
        order_id        TEXT,                       -- 업비트 주문 UUID
        paper_trade     INTEGER NOT NULL DEFAULT 0  -- 페이퍼 트레이딩 여부
    )
    """,

    # ------------------------------------------------------------------
    # 9. layer1_log — Layer 1 룰 기반 필터 매 사이클 출력
    #    보관: 3개월
    # ------------------------------------------------------------------
    "layer1_log": """
    CREATE TABLE IF NOT EXISTS layer1_log (
        id                      INTEGER PRIMARY KEY AUTOINCREMENT,
        coin                    TEXT    NOT NULL,
        timestamp               TEXT    NOT NULL,
        tradeable               INTEGER NOT NULL,   -- 0/1
        regime_strategy         TEXT,
        signal_multiplier       REAL,
        adx_value               REAL,
        supertrend_direction    INTEGER,
        atr_value               REAL,
        active_warnings         TEXT,               -- JSON 배열 문자열
        pullback_detected       INTEGER DEFAULT 0,
        reversal_detected       INTEGER DEFAULT 0,
        api_latency_ms          REAL,
        daily_loss_pct          REAL,
        circuit_breaker_level   INTEGER DEFAULT 0,
        UNIQUE(coin, timestamp)
    )
    """,

    # ------------------------------------------------------------------
    # 10. coin_scan_results — 동적 페어리스트 스캔 결과
    #     보관: 3개월
    # ------------------------------------------------------------------
    "coin_scan_results": """
    CREATE TABLE IF NOT EXISTS coin_scan_results (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp        TEXT    NOT NULL,          -- 스캔 실행 시각
        coin             TEXT    NOT NULL,
        rank_by_volume   INTEGER,                   -- 거래량 순위
        volume_24h_krw   REAL,
        is_leverage_token INTEGER DEFAULT 0,
        is_blacklisted   INTEGER DEFAULT 0,
        included         INTEGER NOT NULL,          -- 최종 페어리스트 포함 여부
        reason_excluded  TEXT,                      -- 제외 사유 (포함 시 NULL)
        UNIQUE(timestamp, coin)
    )
    """,

    # ------------------------------------------------------------------
    # 11. strategy_log — 전략별 거래 성과 상세 (StrategyDecayMonitor 소스)
    #     보관: 영구 (decay_monitor 입력 소스)
    # ------------------------------------------------------------------
    "strategy_log": """
    CREATE TABLE IF NOT EXISTS strategy_log (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        coin            TEXT    NOT NULL,
        timestamp       TEXT    NOT NULL,           -- 진입 시각
        strategy_type   TEXT    NOT NULL,
        action          INTEGER NOT NULL,
        entry_price     REAL    NOT NULL,
        exit_price      REAL,                       -- 청산 시 기록
        pnl             REAL,
        pnl_pct         REAL,
        hold_minutes    REAL,                       -- 보유 시간 (분)
        regime          INTEGER DEFAULT -1,
        kelly_f         REAL
    )
    """,

    # ------------------------------------------------------------------
    # 12. storage_audit_log — StorageManager 실행 기록
    #     보관: 1년
    # ------------------------------------------------------------------
    "storage_audit_log": """
    CREATE TABLE IF NOT EXISTS storage_audit_log (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp       TEXT    NOT NULL,
        db_size_mb      REAL,
        total_rows      INTEGER,
        vacuum_triggered INTEGER DEFAULT 0,
        rows_deleted    INTEGER DEFAULT 0,
        tables_pruned   TEXT,                       -- JSON 배열 (삭제된 테이블명)
        disk_free_gb    REAL
    )
    """,

    # ------------------------------------------------------------------
    # 13. kimchi_premium_log — 5분 주기 김치프리미엄 + 환율
    #     보관: 3개월 (StorageManager 자동 삭제)
    #     [신규-v2]
    # ------------------------------------------------------------------
    "kimchi_premium_log": """
    CREATE TABLE IF NOT EXISTS kimchi_premium_log (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp           TEXT    NOT NULL UNIQUE,
        upbit_btc_krw       REAL    NOT NULL,       -- 업비트 BTC/KRW
        binance_btc_usd     REAL    NOT NULL,       -- 바이낸스 BTC/USDT
        usd_krw_rate        REAL    NOT NULL,       -- 원달러 환율
        kimchi_premium_pct  REAL    NOT NULL        -- 김치프리미엄 % (-5~+10)
    )
    """,

    # ------------------------------------------------------------------
    # 14. strategy_decay_log — 전략별 주간 성과 + DORMANT 이력
    #     보관: 영구 (전략 수명 추적)
    #     [신규-v2] StrategyDecayMonitor 전용
    # ------------------------------------------------------------------
    "strategy_decay_log": """
    CREATE TABLE IF NOT EXISTS strategy_decay_log (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        week_start          TEXT    NOT NULL,       -- ISO8601 (월요일 00:00 UTC)
        strategy_type       TEXT    NOT NULL,
        rolling_sharpe      REAL,                   -- 4주 롤링 샤프비율
        win_rate            REAL,                   -- 해당 주 승률 (0~1)
        profit_loss_ratio   REAL,                   -- 손익비
        trade_count         INTEGER DEFAULT 0,
        is_dormant          INTEGER DEFAULT 0,      -- DORMANT 상태 여부
        dormant_since       TEXT,                   -- DORMANT 진입 시각 (NULL=정상)
        revival_date        TEXT,                   -- DORMANT 해제 시각 (NULL=미해제)
        UNIQUE(week_start, strategy_type)
    )
    """,

    # ------------------------------------------------------------------
    # 15. coin_history — 날짜별 업비트 KRW 마켓 코인 목록 스냅샷
    #     보관: 영구 (생존 편향 처리용 Walk-Forward 백테스트 소스)
    #     [신규-v3]
    # ------------------------------------------------------------------
    "coin_history": """
    CREATE TABLE IF NOT EXISTS coin_history (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        snapshot_date       TEXT    NOT NULL,       -- 'YYYY-MM-DD' (UTC)
        coin                TEXT    NOT NULL,
        volume_24h_krw      REAL,
        rank                INTEGER,                -- 거래량 순위
        market_cap_krw      REAL,
        included_in_pairlist INTEGER DEFAULT 0,     -- 해당 날짜 페어리스트 포함 여부
        UNIQUE(snapshot_date, coin)
    )
    """,

    # ------------------------------------------------------------------
    # 16. paper_comparison — 페이퍼 vs 실거래 신호 일치율 추적
    #     보관: 영구 (전략 검증 소스)
    #     [신규-v4]
    # ------------------------------------------------------------------
    "paper_comparison": """
    CREATE TABLE IF NOT EXISTS paper_comparison (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp           TEXT    NOT NULL,       -- ISO8601 UTC
        coin                TEXT    NOT NULL,
        strategy_type       TEXT,
        paper_action        INTEGER NOT NULL,       -- 페이퍼 결정 action (0~17)
        live_action         INTEGER,                -- 실거래 결정 action (NULL=미실행)
        paper_price         REAL,                   -- 페이퍼 체결가
        live_price          REAL,                   -- 실거래 체결가 (NULL=미실행)
        signal_match        INTEGER NOT NULL DEFAULT 0,  -- 신호 일치 여부 (0/1)
        paper_pnl           REAL,                   -- 페이퍼 손익 KRW (매도 시)
        live_pnl            REAL,                   -- 실거래 손익 KRW (매도 시)
        ensemble_score      REAL,
        week_start          TEXT                    -- 주간 집계용 (ISO 'YYYY-MM-DD')
    )
    """,
}

# ---------------------------------------------------------------------------
# 인덱스 DDL (init_db에서 CREATE TABLE 이후 실행)
# ---------------------------------------------------------------------------

INDEX_DDLS: list[str] = [
    # candles_5m
    "CREATE INDEX IF NOT EXISTS idx_candles_5m_coin_ts  ON candles_5m  (coin, timestamp DESC)",

    # candles_1h
    "CREATE INDEX IF NOT EXISTS idx_candles_1h_coin_ts  ON candles_1h  (coin, timestamp DESC)",

    # candles_1d
    "CREATE INDEX IF NOT EXISTS idx_candles_1d_coin_ts  ON candles_1d  (coin, timestamp DESC)",

    # market_indices
    "CREATE INDEX IF NOT EXISTS idx_market_indices_ts   ON market_indices (timestamp DESC)",

    # onchain_data
    "CREATE INDEX IF NOT EXISTS idx_onchain_coin_ts     ON onchain_data (coin, timestamp DESC)",

    # sentiment_log
    "CREATE INDEX IF NOT EXISTS idx_sentiment_coin_ts   ON sentiment_log (coin, timestamp DESC)",

    # ensemble_predictions
    "CREATE INDEX IF NOT EXISTS idx_ensemble_coin_ts    ON ensemble_predictions (coin, timestamp DESC)",

    # trades — 법률 조회 대비 다중 인덱스
    "CREATE INDEX IF NOT EXISTS idx_trades_coin_ts      ON trades (coin, timestamp DESC)",
    "CREATE INDEX IF NOT EXISTS idx_trades_strategy     ON trades (strategy_type)",
    "CREATE INDEX IF NOT EXISTS idx_trades_dry_run      ON trades (is_dry_run)",

    # layer1_log
    "CREATE INDEX IF NOT EXISTS idx_layer1_coin_ts      ON layer1_log (coin, timestamp DESC)",

    # coin_scan_results
    "CREATE INDEX IF NOT EXISTS idx_scan_ts_coin        ON coin_scan_results (timestamp DESC, coin)",

    # strategy_log
    "CREATE INDEX IF NOT EXISTS idx_strat_log_coin_ts   ON strategy_log (coin, timestamp DESC)",
    "CREATE INDEX IF NOT EXISTS idx_strat_log_type      ON strategy_log (strategy_type)",

    # storage_audit_log
    "CREATE INDEX IF NOT EXISTS idx_audit_ts            ON storage_audit_log (timestamp DESC)",

    # kimchi_premium_log
    "CREATE INDEX IF NOT EXISTS idx_kimchi_ts           ON kimchi_premium_log (timestamp DESC)",

    # strategy_decay_log
    "CREATE INDEX IF NOT EXISTS idx_decay_week_type     ON strategy_decay_log (week_start DESC, strategy_type)",

    # coin_history
    "CREATE INDEX IF NOT EXISTS idx_coin_hist_date      ON coin_history (snapshot_date DESC)",
    "CREATE INDEX IF NOT EXISTS idx_coin_hist_coin      ON coin_history (coin)",

    # paper_comparison
    "CREATE INDEX IF NOT EXISTS idx_paper_cmp_ts        ON paper_comparison (timestamp DESC)",
    "CREATE INDEX IF NOT EXISTS idx_paper_cmp_coin      ON paper_comparison (coin, timestamp DESC)",
    "CREATE INDEX IF NOT EXISTS idx_paper_cmp_week      ON paper_comparison (week_start DESC)",
]

# ---------------------------------------------------------------------------
# 보관 기간 정책 (StorageManager가 참조)
# ---------------------------------------------------------------------------

RETENTION_DAYS: dict[str, int | None] = {
    "candles_5m":           90,    # 3개월
    "candles_1h":           365,   # 1년
    "candles_1d":           None,  # 영구
    "market_indices":       365,   # 1년
    "onchain_data":         180,   # 6개월
    "sentiment_log":        90,    # 3개월
    "ensemble_predictions": 180,   # 6개월
    "trades":               None,  # 영구 (법률 5년 의무 — 절대 삭제 금지)
    "layer1_log":           90,    # 3개월
    "coin_scan_results":    90,    # 3개월
    "strategy_log":         None,  # 영구
    "storage_audit_log":    365,   # 1년
    "kimchi_premium_log":   90,    # 3개월
    "strategy_decay_log":   None,  # 영구
    "coin_history":         None,  # 영구 (생존 편향 처리용)
    "paper_comparison":     None,  # 영구 (전략 검증 소스)
}

# trades 테이블은 StorageManager VACUUM 대상에서 제외
IMMUTABLE_TABLES: frozenset[str] = frozenset({"trades"})
