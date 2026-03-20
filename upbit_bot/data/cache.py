"""
data/cache.py — SQLite 캐시 관리 + DB 초기화

책임:
  - init_db(): TABLE_DDLS + INDEX_DDLS 실행 (최초 1회)
  - CandleCache: 5분/1시간/일봉 UPSERT + 최근 N개 조회
  - thread-safe: WAL 모드 + threading.Lock
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from schema import IMMUTABLE_TABLES, INDEX_DDLS, RETENTION_DAYS, TABLE_DDLS

logger = logging.getLogger(__name__)


def init_db(db_path: str | Path) -> None:
    """DB 초기화 — 테이블 + 인덱스 CREATE IF NOT EXISTS."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=-32000")   # 32 MB
        conn.execute("PRAGMA temp_store=MEMORY")

        for name, ddl in TABLE_DDLS.items():
            conn.execute(ddl)
            logger.debug("테이블 생성/확인: %s", name)

        for ddl in INDEX_DDLS:
            conn.execute(ddl)

        conn.commit()
        logger.info("DB 초기화 완료: %s", db_path)
    finally:
        conn.close()


class CandleCache:
    """캔들 데이터 SQLite UPSERT + 조회.

    - 단일 Connection 재사용 (WAL 모드, check_same_thread=False)
    - UPSERT: INSERT OR REPLACE (UNIQUE(coin, timestamp) 활용)
    - bulk_upsert: 초기 히스토리 로딩 전용 (executemany)
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA cache_size=-32000")
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # UPSERT
    # ------------------------------------------------------------------

    def upsert_candle(self, table: str, coin: str, row: dict) -> None:
        """단일 행 UPSERT (캔들 완성 시마다 호출)."""
        if "coin" not in row:
            row = {"coin": coin, **row}
        cols = list(row.keys())
        placeholders = ", ".join(["?"] * len(cols))
        col_str = ", ".join(cols)
        sql = f"INSERT OR REPLACE INTO {table} ({col_str}) VALUES ({placeholders})"
        with self._lock:
            self._conn.execute(sql, list(row.values()))
            self._conn.commit()

    def bulk_upsert(self, table: str, rows: list[dict]) -> None:
        """다수 행 일괄 UPSERT — 초기 히스토리 로딩 전용."""
        if not rows:
            return
        cols = list(rows[0].keys())
        placeholders = ", ".join(["?"] * len(cols))
        col_str = ", ".join(cols)
        sql = f"INSERT OR REPLACE INTO {table} ({col_str}) VALUES ({placeholders})"
        values = [list(r.values()) for r in rows]
        with self._lock:
            self._conn.executemany(sql, values)
            self._conn.commit()

    def insert_row(self, table: str, row: dict) -> None:
        """단순 INSERT (UNIQUE 없는 테이블 — trades 등)."""
        cols = list(row.keys())
        placeholders = ", ".join(["?"] * len(cols))
        col_str = ", ".join(cols)
        sql = f"INSERT INTO {table} ({col_str}) VALUES ({placeholders})"
        with self._lock:
            self._conn.execute(sql, list(row.values()))
            self._conn.commit()

    # ------------------------------------------------------------------
    # 조회
    # ------------------------------------------------------------------

    def get_recent_candles(
        self,
        table: str,
        coin: str,
        limit: int = 500,
    ) -> pd.DataFrame:
        """최근 N개 캔들 → DataFrame (시간 오름차순 반환)."""
        sql = f"""
            SELECT * FROM {table}
            WHERE coin = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """
        with self._lock:
            df = pd.read_sql_query(sql, self._conn, params=(coin, limit))
        if df.empty:
            return df
        # format='ISO8601': REST(tz-naive)·WebSocket(+00:00 tz-aware) 혼합 형식 처리
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601", utc=True)
        return df.sort_values("timestamp").reset_index(drop=True)

    def get_latest_timestamp(self, table: str, coin: str) -> datetime | None:
        """최신 캔들 타임스탬프 (초기 로딩 여부 확인용)."""
        sql = f"SELECT MAX(timestamp) AS ts FROM {table} WHERE coin = ?"
        with self._lock:
            cur = self._conn.execute(sql, (coin,))
            row = cur.fetchone()
        if row and row["ts"]:
            return datetime.fromisoformat(row["ts"]).replace(tzinfo=timezone.utc)
        return None

    def get_latest_row(self, table: str, timestamp_col: str = "timestamp") -> dict | None:
        """특정 테이블의 최신 단일 행 반환 (market_indices 등)."""
        sql = f"SELECT * FROM {table} ORDER BY {timestamp_col} DESC LIMIT 1"
        with self._lock:
            cur = self._conn.execute(sql)
            row = cur.fetchone()
        return dict(row) if row else None

    def count_rows(self, table: str) -> int:
        """테이블 전체 행 수."""
        with self._lock:
            cur = self._conn.execute(f"SELECT COUNT(*) FROM {table}")
            return cur.fetchone()[0]

    # ------------------------------------------------------------------
    # 보관 기간 초과 행 삭제 (StorageManager 호출용)
    # ------------------------------------------------------------------

    def prune_old_rows(
        self,
        table: str,
        days: int | None = None,
        timestamp_col: str = "timestamp",
    ) -> int:
        """보관 기간 초과 행 삭제.

        days=None → RETENTION_DAYS 기본값 사용.
        trades 테이블은 호출 불가 (IMMUTABLE_TABLES).
        """
        if table in IMMUTABLE_TABLES:
            raise ValueError(f"'{table}'은 영구보관 테이블 — 삭제 금지")

        effective_days = days if days is not None else RETENTION_DAYS.get(table)
        if effective_days is None:
            logger.debug("'%s' 영구보관 — prune 건너뜀", table)
            return 0

        cutoff = (datetime.now(timezone.utc) - timedelta(days=effective_days)).isoformat()
        sql = f"DELETE FROM {table} WHERE {timestamp_col} < ?"
        with self._lock:
            cur = self._conn.execute(sql, (cutoff,))
            self._conn.commit()
        deleted = cur.rowcount
        if deleted:
            logger.info("'%s' 오래된 행 %d개 삭제 (>%d일)", table, deleted, effective_days)
        return deleted

    def vacuum(self) -> None:
        """VACUUM 실행 — 디스크 회수 (StorageManager 주간 호출)."""
        with self._lock:
            self._conn.execute("VACUUM")
        logger.info("VACUUM 완료")

    # ------------------------------------------------------------------
    # context manager
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "CandleCache":
        return self

    def __exit__(self, *_) -> None:
        self.close()
