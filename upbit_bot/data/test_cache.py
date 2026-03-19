"""
test_cache.py — CandleCache + init_db 단위 테스트 (in-memory SQLite)
"""

import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from data.cache import CandleCache, init_db


# ---------------------------------------------------------------------------
# 픽스처
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    db = tmp_path / "test.db"
    init_db(db)
    return db


@pytest.fixture
def cache(tmp_db: Path) -> CandleCache:
    c = CandleCache(tmp_db)
    yield c
    c.close()


# ---------------------------------------------------------------------------
# init_db
# ---------------------------------------------------------------------------

class TestInitDb:
    def test_creates_all_15_tables(self, tmp_db: Path):
        conn = sqlite3.connect(str(tmp_db))
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        tables = {r[0] for r in cur.fetchall()}
        conn.close()
        assert len(tables) == 15

    def test_idempotent(self, tmp_db: Path):
        """두 번 호출해도 오류 없어야 함."""
        init_db(tmp_db)
        init_db(tmp_db)

    def test_wal_mode(self, tmp_db: Path):
        conn = sqlite3.connect(str(tmp_db))
        cur = conn.execute("PRAGMA journal_mode")
        mode = cur.fetchone()[0]
        conn.close()
        assert mode == "wal"


# ---------------------------------------------------------------------------
# CandleCache.upsert_candle + get_recent_candles
# ---------------------------------------------------------------------------

SAMPLE_CANDLE = {
    "coin": "KRW-BTC",
    "timestamp": "2026-01-01T00:00:00+00:00",
    "open": 90_000_000.0,
    "high": 91_000_000.0,
    "low": 89_500_000.0,
    "close": 90_500_000.0,
    "volume": 1.5,
}


class TestUpsert:
    def test_upsert_single(self, cache: CandleCache):
        cache.upsert_candle("candles_5m", "KRW-BTC", SAMPLE_CANDLE)
        df = cache.get_recent_candles("candles_5m", "KRW-BTC", limit=10)
        assert len(df) == 1
        assert float(df.iloc[0]["close"]) == 90_500_000.0

    def test_upsert_replace_duplicate(self, cache: CandleCache):
        """같은 (coin, timestamp) UPSERT → 1개만 존재해야 함."""
        cache.upsert_candle("candles_5m", "KRW-BTC", SAMPLE_CANDLE)
        updated = {**SAMPLE_CANDLE, "close": 91_000_000.0}
        cache.upsert_candle("candles_5m", "KRW-BTC", updated)
        df = cache.get_recent_candles("candles_5m", "KRW-BTC", limit=10)
        assert len(df) == 1
        assert float(df.iloc[0]["close"]) == 91_000_000.0

    def test_bulk_upsert(self, cache: CandleCache):
        rows = []
        for i in range(10):
            ts = f"2026-01-01T{i:02d}:00:00+00:00"
            rows.append({**SAMPLE_CANDLE, "timestamp": ts})
        cache.bulk_upsert("candles_5m", rows)
        df = cache.get_recent_candles("candles_5m", "KRW-BTC", limit=20)
        assert len(df) == 10

    def test_get_recent_candles_ascending_order(self, cache: CandleCache):
        """반환 DataFrame은 시간 오름차순."""
        rows = [
            {**SAMPLE_CANDLE, "timestamp": "2026-01-01T00:05:00+00:00"},
            {**SAMPLE_CANDLE, "timestamp": "2026-01-01T00:00:00+00:00"},
            {**SAMPLE_CANDLE, "timestamp": "2026-01-01T00:10:00+00:00"},
        ]
        cache.bulk_upsert("candles_5m", rows)
        df = cache.get_recent_candles("candles_5m", "KRW-BTC", limit=10)
        ts_list = df["timestamp"].tolist()
        assert ts_list == sorted(ts_list)

    def test_get_latest_timestamp(self, cache: CandleCache):
        cache.upsert_candle("candles_5m", "KRW-BTC", SAMPLE_CANDLE)
        ts = cache.get_latest_timestamp("candles_5m", "KRW-BTC")
        assert ts is not None
        assert ts.tzinfo is not None

    def test_get_latest_timestamp_none_when_empty(self, cache: CandleCache):
        ts = cache.get_latest_timestamp("candles_5m", "KRW-ETH")
        assert ts is None


# ---------------------------------------------------------------------------
# prune_old_rows
# ---------------------------------------------------------------------------

class TestPrune:
    def test_prune_removes_old_rows(self, cache: CandleCache):
        # 오래된 타임스탬프
        old_row = {**SAMPLE_CANDLE, "timestamp": "2020-01-01T00:00:00+00:00"}
        cache.upsert_candle("candles_5m", "KRW-BTC", old_row)
        deleted = cache.prune_old_rows("candles_5m", days=90)
        assert deleted >= 1
        df = cache.get_recent_candles("candles_5m", "KRW-BTC", limit=10)
        assert df.empty

    def test_prune_keeps_recent_rows(self, cache: CandleCache):
        now_str = datetime.now(timezone.utc).isoformat()
        cache.upsert_candle("candles_5m", "KRW-BTC", {**SAMPLE_CANDLE, "timestamp": now_str})
        deleted = cache.prune_old_rows("candles_5m", days=90)
        assert deleted == 0

    def test_prune_immutable_table_raises(self, cache: CandleCache):
        with pytest.raises(ValueError, match="영구보관"):
            cache.prune_old_rows("trades")

    def test_prune_permanent_table_skips(self, cache: CandleCache):
        """None 보관 기간 테이블은 자동 스킵 (0 반환)."""
        deleted = cache.prune_old_rows("candles_1d")
        assert deleted == 0


# ---------------------------------------------------------------------------
# insert_row (trades — UNIQUE 없음)
# ---------------------------------------------------------------------------

class TestInsertRow:
    def test_insert_trade(self, cache: CandleCache):
        row = {
            "coin": "KRW-BTC",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "action": 2,
            "side": "BUY",
            "price": 90_000_000.0,
            "volume": 0.001,
            "krw_amount": 90_000.0,
            "is_dry_run": 1,
            "paper_trade": 0,
        }
        cache.insert_row("trades", row)
        cache.insert_row("trades", row)   # 중복 삽입 가능
        assert cache.count_rows("trades") == 2
