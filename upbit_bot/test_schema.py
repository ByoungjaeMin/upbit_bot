"""
test_schema.py — schema.py 단위 테스트

테스트 항목:
  - 8개 dataclass 기본 생성 + 타입
  - 15개 테이블 DDL 파싱 가능 여부 (SQLite in-memory)
  - 15개 인덱스 DDL 실행 성공
  - RETENTION_DAYS 키 완전성 (15개 테이블 전부 존재)
  - IMMUTABLE_TABLES 포함 여부
"""

import sqlite3
from datetime import datetime, timezone

import pytest

from schema import (
    IMMUTABLE_TABLES,
    INDEX_DDLS,
    RETENTION_DAYS,
    SCHEMA_VERSION,
    TABLE_DDLS,
    EnsemblePrediction,
    FilterResult,
    MarketState,
    RawMarketData,
    RiskBudget,
    StrategyDecision,
    TelegramEvent,
    TradeDecision,
)

NOW = datetime.now(timezone.utc)
EXPECTED_TABLES = {
    "candles_5m", "candles_1h", "candles_1d",
    "market_indices", "onchain_data", "sentiment_log",
    "ensemble_predictions", "trades", "layer1_log",
    "coin_scan_results", "strategy_log", "storage_audit_log",
    "kimchi_premium_log", "strategy_decay_log", "coin_history",
}


# ---------------------------------------------------------------------------
# dataclass 생성 테스트
# ---------------------------------------------------------------------------

class TestDataclasses:
    def test_raw_market_data_defaults(self):
        d = RawMarketData(
            coin="KRW-BTC",
            timestamp=NOW,
            trade_price=90_000_000.0,
            trade_volume=0.001,
            ask_bid="ASK",
            sequential_id=1,
            stream_type="trade",
        )
        assert d.coin == "KRW-BTC"
        assert d.ask_bid in ("ASK", "BID")
        assert d.acc_trade_volume_24h is None

    def test_market_state_38_fields(self):
        ms = MarketState(coin="KRW-ETH", timestamp=NOW)
        # 38개 ML 피처: 5m OHLCV(5) + 5m TA(9) + 1h(5) + 1d(4) + 시장(3) + 온체인(2)
        #               + 감성(2) + 추세레짐(4) + 업비트특화(3) + 마이크로(3) = 40
        # (open/high/low/close/volume 5 + rsi/macd/macd_signal/bb_upper/bb_lower/ema7/ema25/ema99/volume_ratio 9 = 14)
        assert ms.rsi_5m == 50.0
        assert ms.hmm_regime == -1       # Phase A: HMM 미적용
        assert ms.tick_imbalance == 0.0

    def test_filter_result(self):
        fr = FilterResult(
            coin="KRW-BTC",
            timestamp=NOW,
            tradeable=True,
            regime_strategy="TREND_STRONG",
            signal_multiplier=1.0,
            adx_value=32.0,
            supertrend_direction=1,
            atr_value=500_000.0,
        )
        assert fr.circuit_breaker_level == 0
        assert fr.active_warnings == []

    def test_ensemble_prediction_threshold(self):
        ep = EnsemblePrediction(
            coin="KRW-BTC",
            timestamp=NOW,
            per_model_probs={"xgb": 0.7, "lgb": 0.65},
            weighted_avg=0.68,
            consensus_count=3,
            signal_confirmed=True,
        )
        assert ep.threshold_used == 0.62   # v9 기본값

    def test_strategy_decision_defaults(self):
        sd = StrategyDecision(
            coin="KRW-BTC",
            timestamp=NOW,
            strategy_type="GRID",
            capital_allocation=0.2,
        )
        assert sd.grid_params is None
        assert sd.dynamic_weight == 1.0

    def test_risk_budget(self):
        rb = RiskBudget(
            coin="KRW-BTC",
            timestamp=NOW,
            kelly_f=0.04,
            hmm_adjusted_f=0.036,
            var_adjusted_f=0.03,
            final_position_size=150_000.0,
        )
        assert rb.coin_group == "NORMAL"
        assert rb.consecutive_losses == 0

    def test_trade_decision_dry_run_default(self):
        td = TradeDecision(
            coin="KRW-BTC",
            timestamp=NOW,
            action=2,
            target_coin="KRW-BTC",
            position_size=100_000.0,
        )
        assert td.is_dry_run is True   # 기본값 True (콜드스타트 보호)
        assert td.trailing_stop_price is None

    def test_telegram_event(self):
        te = TelegramEvent(
            event_type="TRADE",
            message="<b>BUY</b> KRW-BTC",
            timestamp=NOW,
        )
        assert te.priority == 2
        assert te.parse_mode == "HTML"


# ---------------------------------------------------------------------------
# SQLite DDL 테스트
# ---------------------------------------------------------------------------

@pytest.fixture
def db():
    """인메모리 SQLite 연결."""
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA journal_mode=WAL")
    yield conn
    conn.close()


class TestSQLiteDDL:
    def test_all_15_tables_defined(self):
        assert TABLE_DDLS.keys() == EXPECTED_TABLES, (
            f"누락 또는 초과 테이블: {TABLE_DDLS.keys() ^ EXPECTED_TABLES}"
        )

    def test_table_creation(self, db):
        for name, ddl in TABLE_DDLS.items():
            try:
                db.execute(ddl)
            except sqlite3.Error as exc:
                pytest.fail(f"테이블 '{name}' DDL 오류: {exc}")

    def test_index_creation(self, db):
        for ddl in TABLE_DDLS.values():
            db.execute(ddl)
        for idx_ddl in INDEX_DDLS:
            try:
                db.execute(idx_ddl)
            except sqlite3.Error as exc:
                pytest.fail(f"인덱스 DDL 오류: {idx_ddl!r} → {exc}")

    def test_table_count_after_creation(self, db):
        for ddl in TABLE_DDLS.values():
            db.execute(ddl)
        cur = db.execute(
            "SELECT count(*) FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        count = cur.fetchone()[0]
        assert count == 15, f"테이블 수 불일치: {count}"

    def test_trades_unique_constraint(self, db):
        """trades 테이블에는 UNIQUE 제약이 없어야 함 — 동일 코인 다중 거래 허용."""
        db.execute(TABLE_DDLS["trades"])
        # 동일 coin+timestamp 두 번 삽입 가능해야 한다 (UNIQUE 없음)
        db.execute(
            "INSERT INTO trades (coin, timestamp, action, side, price, volume, krw_amount, is_dry_run, paper_trade) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("KRW-BTC", "2026-01-01T00:00:00", 2, "BUY", 90000000, 0.001, 90000, 1, 0),
        )
        db.execute(
            "INSERT INTO trades (coin, timestamp, action, side, price, volume, krw_amount, is_dry_run, paper_trade) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("KRW-BTC", "2026-01-01T00:00:00", 6, "SELL", 91000000, 0.001, 91000, 1, 0),
        )
        cur = db.execute("SELECT count(*) FROM trades")
        assert cur.fetchone()[0] == 2

    def test_candles_unique_constraint(self, db):
        """candles_5m은 (coin, timestamp) UNIQUE — 중복 삽입 시 오류."""
        db.execute(TABLE_DDLS["candles_5m"])
        db.execute(
            "INSERT INTO candles_5m (coin, timestamp, open, high, low, close, volume) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("KRW-BTC", "2026-01-01T00:00:00", 100, 110, 90, 105, 10),
        )
        with pytest.raises(sqlite3.IntegrityError):
            db.execute(
                "INSERT INTO candles_5m (coin, timestamp, open, high, low, close, volume) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("KRW-BTC", "2026-01-01T00:00:00", 100, 110, 90, 105, 10),
            )


# ---------------------------------------------------------------------------
# 정책 테스트
# ---------------------------------------------------------------------------

class TestRetentionPolicy:
    def test_all_tables_have_retention(self):
        missing = EXPECTED_TABLES - RETENTION_DAYS.keys()
        assert not missing, f"RETENTION_DAYS 누락 테이블: {missing}"

    def test_immutable_tables_are_none_retention(self):
        for tbl in IMMUTABLE_TABLES:
            assert RETENTION_DAYS[tbl] is None, (
                f"IMMUTABLE 테이블 '{tbl}'의 보관 기간이 None이 아님"
            )

    def test_trades_immutable(self):
        assert "trades" in IMMUTABLE_TABLES

    def test_schema_version(self):
        assert SCHEMA_VERSION == 9
