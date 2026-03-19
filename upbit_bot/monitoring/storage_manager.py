"""
monitoring/storage_manager.py — SQLite 스토리지 자동 관리

[기능]
  1. cleanup_candles()           — 타임프레임별 보관 기간 초과 행 DELETE
  2. vacuum_database()           — VACUUM (파일 크기 실제 회수)
  3. archive_old_trades()        — 1년+ 거래기록 → archive.db 이동 (세금용)
  4. cleanup_model_checkpoints() — 모델별 최신 3개만 보관
  5. cleanup_logs()              — logs/ 30일 초과 파일 삭제
  6. cleanup_backtest_results()  — 최신 10개 + 90일치만 보관
  7. check_disk_usage()          — 여유<50GB 경고 / <20GB 긴급 / DB>10GB VACUUM 권장

[APScheduler 스케줄]
  매일 03:00          — cleanup_candles + cleanup_logs
  매주 일요일 02:00   — archive_old_trades
  매주 일요일 03:30   — vacuum_database
  매주 일요일 04:00   — cleanup_model_checkpoints
  매월 1일  04:30     — cleanup_backtest_results
  매시간 :00          — check_disk_usage
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# 상수
# ─────────────────────────────────────────────────────────────────

DISK_WARN_GB   = 50.0   # 여유 < 50 GB → 경고
DISK_CRIT_GB   = 20.0   # 여유 < 20 GB → 긴급
DB_VACUUM_GB   = 10.0   # DB 크기 > 10 GB → VACUUM 권장
MAX_CHECKPOINTS = 3     # 모델별 최신 보관 개수
BACKTEST_MAX_FILES = 10 # 최신 N개 보관
BACKTEST_DAYS   = 90    # + 90일치 보관
LOG_RETENTION_DAYS = 30 # 로그 파일 보관 일수
ARCHIVE_RETENTION_DAYS = 365  # 1년 이상 거래 → archive.db


# ─────────────────────────────────────────────────────────────────
# StorageManager
# ─────────────────────────────────────────────────────────────────

class StorageManager:
    """SQLite 자동 정리 + 디스크 감시 + APScheduler 스케줄 등록.

    사용법:
        mgr = StorageManager(db_path="data/bot.db", base_dir=Path("."))
        mgr.set_telegram_callback(bot.send)   # 알림 콜백 (optional)
        mgr.schedule(scheduler)               # APScheduler 등록
    """

    def __init__(
        self,
        db_path: str | Path,
        base_dir: Path | None = None,
        archive_db_path: str | Path | None = None,
    ) -> None:
        self._db_path = Path(db_path)
        self._base_dir = base_dir or self._db_path.parent.parent
        self._archive_db = Path(archive_db_path) if archive_db_path else (
            self._db_path.parent / "archive.db"
        )
        self._lock = threading.Lock()
        self._telegram_cb: Callable[[str], Any] | None = None

    # ──────────────────────────────────────────────────────────────
    # 텔레그램 알림 콜백
    # ──────────────────────────────────────────────────────────────

    def set_telegram_callback(self, cb: Callable[[str], Any]) -> None:
        """텔레그램 send 함수(또는 asyncio coroutine 함수) 등록."""
        self._telegram_cb = cb

    def _notify(self, message: str) -> None:
        """텔레그램 알림 전송. coroutine이면 새 이벤트 루프에서 실행."""
        logger.warning("[StorageManager] %s", message)
        if self._telegram_cb is None:
            return
        try:
            result = self._telegram_cb(message)
            if asyncio.iscoroutine(result):
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.ensure_future(result)
                    else:
                        loop.run_until_complete(result)
                except RuntimeError:
                    asyncio.run(result)
        except Exception as exc:
            logger.error("[StorageManager] 텔레그램 알림 실패: %s", exc)

    # ──────────────────────────────────────────────────────────────
    # DB 연결 헬퍼
    # ──────────────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    # ──────────────────────────────────────────────────────────────
    # 1. cleanup_candles
    # ──────────────────────────────────────────────────────────────

    def cleanup_candles(self) -> dict[str, int]:
        """타임프레임별 보관 기간 초과 캔들 행 DELETE.

        보관 기간:
          candles_5m          : 6개월 (180일)
          candles_1h          : 2년  (730일)
          candles_1d          : 3년  (1095일)
          ensemble_predictions: 6개월
          sentiment_log       : 3개월 (90일)
          layer1_log          : 3개월
          coin_scan_results   : 3개월

        Returns:
            {테이블명: 삭제 행수}
        """
        cutoffs: dict[str, int] = {
            "candles_5m":           180,
            "candles_1h":           730,
            "candles_1d":           1095,
            "ensemble_predictions": 180,
            "sentiment_log":        90,
            "layer1_log":           90,
            "coin_scan_results":    90,
            "kimchi_premium_log":   90,
            "onchain_data":         180,
            "market_indices":       365,
        }
        deleted: dict[str, int] = {}
        with self._lock:
            conn = self._connect()
            try:
                for table, days in cutoffs.items():
                    cutoff_ts = (
                        datetime.now(timezone.utc) - timedelta(days=days)
                    ).isoformat()
                    try:
                        cur = conn.execute(
                            f"DELETE FROM {table} WHERE timestamp < ?",  # noqa: S608
                            (cutoff_ts,),
                        )
                        deleted[table] = cur.rowcount
                        if cur.rowcount:
                            logger.info(
                                "[StorageManager] %s: %d행 삭제 (보관=%d일)",
                                table, cur.rowcount, days,
                            )
                    except sqlite3.OperationalError:
                        logger.debug("[StorageManager] 테이블 없음 — 건너뜀: %s", table)
                        deleted[table] = 0
                conn.commit()
                self._write_audit(conn, rows_deleted=sum(deleted.values()),
                                  tables_pruned=list(deleted.keys()))
                conn.commit()
            finally:
                conn.close()
        return deleted

    # ──────────────────────────────────────────────────────────────
    # 2. vacuum_database
    # ──────────────────────────────────────────────────────────────

    def vacuum_database(self) -> float:
        """VACUUM — 파일 크기 실제 회수.

        주의: DB 크기 × 2 임시공간 필요. 실행 중 봇 일시중단 필요.
        Returns:
            VACUUM 후 DB 크기 MB.
        """
        size_before = self._db_size_mb()
        logger.info("[StorageManager] VACUUM 시작 (전: %.1f MB)", size_before)

        # VACUUM은 WAL 체크포인트 후 단독 연결 필요
        with self._lock:
            conn = sqlite3.connect(str(self._db_path), isolation_level=None)
            try:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                conn.execute("VACUUM")
            finally:
                conn.close()

        size_after = self._db_size_mb()
        freed = size_before - size_after
        logger.info(
            "[StorageManager] VACUUM 완료: %.1f MB → %.1f MB (%.1f MB 회수)",
            size_before, size_after, freed,
        )

        with self._lock:
            conn = self._connect()
            try:
                self._write_audit(conn, vacuum_triggered=1, rows_deleted=0,
                                  tables_pruned=[])
                conn.commit()
            finally:
                conn.close()
        return size_after

    # ──────────────────────────────────────────────────────────────
    # 3. archive_old_trades
    # ──────────────────────────────────────────────────────────────

    def archive_old_trades(self) -> int:
        """1년 이상된 거래 기록 → archive.db 이동 (영구 보관, 세금용).

        Returns:
            이동된 행수.
        """
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=ARCHIVE_RETENTION_DAYS)
        ).isoformat()

        with self._lock:
            # 원본 DB에서 조회
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT * FROM trades WHERE timestamp < ?", (cutoff,)
                ).fetchall()
                if not rows:
                    logger.info("[StorageManager] archive 대상 거래 없음")
                    conn.close()
                    return 0

                # archive.db에 upsert
                arch_conn = sqlite3.connect(str(self._archive_db))
                try:
                    # archive.db에 동일 테이블 생성 (schema 가져오기)
                    col_info = conn.execute(
                        "SELECT sql FROM sqlite_master WHERE type='table' AND name='trades'"
                    ).fetchone()
                    if col_info:
                        arch_conn.execute(col_info[0])
                    arch_conn.executemany(
                        f"INSERT OR IGNORE INTO trades VALUES ({','.join(['?']*len(rows[0]))})",
                        rows,
                    )
                    arch_conn.commit()
                finally:
                    arch_conn.close()

                # 원본 삭제 (절대 trades 삭제 금지 — archive 확인 후만)
                ids = [r[0] for r in rows]
                conn.execute(
                    f"DELETE FROM trades WHERE id IN ({','.join('?'*len(ids))})",
                    ids,
                )
                conn.commit()
                count = len(rows)
                logger.info("[StorageManager] archive 이동: %d건", count)
                return count
            finally:
                conn.close()

    # ──────────────────────────────────────────────────────────────
    # 4. cleanup_model_checkpoints
    # ──────────────────────────────────────────────────────────────

    def cleanup_model_checkpoints(self) -> int:
        """models/ 폴더에서 모델별 최신 3개만 보관, 나머지 삭제.

        파일명 패턴: {model_type}_{timestamp}.pt / .pkl / .json
        Returns:
            삭제된 파일 수.
        """
        models_dir = self._base_dir / "models"
        if not models_dir.exists():
            logger.debug("[StorageManager] models/ 폴더 없음 — 건너뜀")
            return 0

        from collections import defaultdict
        groups: dict[str, list[Path]] = defaultdict(list)
        for f in models_dir.iterdir():
            if f.is_file() and f.suffix in {".pt", ".pkl", ".json", ".pth"}:
                # 모델 타입: 파일명에서 첫 '_' 앞 부분
                model_type = f.stem.split("_")[0]
                groups[model_type].append(f)

        deleted = 0
        for model_type, files in groups.items():
            # 수정시간 내림차순 정렬
            files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            to_delete = files[MAX_CHECKPOINTS:]
            for f in to_delete:
                try:
                    f.unlink()
                    logger.info("[StorageManager] 체크포인트 삭제: %s", f.name)
                    deleted += 1
                except OSError as exc:
                    logger.warning("[StorageManager] 삭제 실패: %s — %s", f.name, exc)

        logger.info("[StorageManager] 체크포인트 정리 완료: %d개 삭제", deleted)
        return deleted

    # ──────────────────────────────────────────────────────────────
    # 5. cleanup_logs
    # ──────────────────────────────────────────────────────────────

    def cleanup_logs(self) -> int:
        """logs/ 폴더에서 30일 초과 로그 파일 삭제.

        Returns:
            삭제된 파일 수.
        """
        logs_dir = self._base_dir / "logs"
        if not logs_dir.exists():
            logger.debug("[StorageManager] logs/ 폴더 없음 — 건너뜀")
            return 0

        cutoff_ts = (
            datetime.now(timezone.utc) - timedelta(days=LOG_RETENTION_DAYS)
        ).timestamp()
        deleted = 0
        for f in logs_dir.rglob("*"):
            if f.is_file() and f.stat().st_mtime < cutoff_ts:
                try:
                    f.unlink()
                    logger.info("[StorageManager] 로그 삭제: %s", f.name)
                    deleted += 1
                except OSError as exc:
                    logger.warning("[StorageManager] 로그 삭제 실패: %s — %s", f.name, exc)

        logger.info("[StorageManager] 로그 정리 완료: %d개 삭제", deleted)
        return deleted

    # ──────────────────────────────────────────────────────────────
    # 6. cleanup_backtest_results
    # ──────────────────────────────────────────────────────────────

    def cleanup_backtest_results(self) -> int:
        """backtest/results/ 폴더에서 최신 10개 + 90일치만 보관.

        Returns:
            삭제된 파일 수.
        """
        results_dir = self._base_dir / "backtest" / "results"
        if not results_dir.exists():
            logger.debug("[StorageManager] backtest/results/ 없음 — 건너뜀")
            return 0

        cutoff_ts = (
            datetime.now(timezone.utc) - timedelta(days=BACKTEST_DAYS)
        ).timestamp()
        files = sorted(
            [f for f in results_dir.iterdir() if f.is_file()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        # 최신 BACKTEST_MAX_FILES개는 무조건 보관
        keep = set(files[:BACKTEST_MAX_FILES])
        # 90일 이내도 보관
        for f in files:
            if f.stat().st_mtime >= cutoff_ts:
                keep.add(f)

        deleted = 0
        for f in files:
            if f not in keep:
                try:
                    f.unlink()
                    logger.info("[StorageManager] 백테스트 결과 삭제: %s", f.name)
                    deleted += 1
                except OSError as exc:
                    logger.warning("[StorageManager] 삭제 실패: %s — %s", f.name, exc)

        logger.info("[StorageManager] 백테스트 결과 정리 완료: %d개 삭제", deleted)
        return deleted

    # ──────────────────────────────────────────────────────────────
    # 7. check_disk_usage
    # ──────────────────────────────────────────────────────────────

    def check_disk_usage(self) -> dict[str, float]:
        """디스크 여유 공간 및 DB 크기 점검.

        Returns:
            {'free_gb': float, 'db_mb': float, 'alert_level': int}
            alert_level: 0=정상, 1=경고, 2=긴급
        """
        disk = shutil.disk_usage(str(self._db_path.parent))
        free_gb = disk.free / (1024 ** 3)
        db_mb = self._db_size_mb()
        db_gb = db_mb / 1024

        alert_level = 0

        if free_gb < DISK_CRIT_GB:
            msg = f"🚨 디스크 긴급: 여유 공간 {free_gb:.1f}GB — 즉시 정리 필요"
            self._notify(msg)
            alert_level = 2
        elif free_gb < DISK_WARN_GB:
            msg = f"⚠️ 디스크 경고: 여유 공간 {free_gb:.1f}GB"
            self._notify(msg)
            alert_level = 1

        if db_gb > DB_VACUUM_GB:
            msg = f"⚠️ DB 크기 {db_gb:.1f}GB 초과 — VACUUM 권장"
            self._notify(msg)

        logger.info(
            "[StorageManager] 디스크 점검: 여유=%.1f GB, DB=%.1f MB",
            free_gb, db_mb,
        )
        return {"free_gb": free_gb, "db_mb": db_mb, "alert_level": float(alert_level)}

    # ──────────────────────────────────────────────────────────────
    # APScheduler 등록
    # ──────────────────────────────────────────────────────────────

    def schedule(self, scheduler: Any) -> None:
        """APScheduler에 모든 정리 작업 등록.

        Args:
            scheduler: APScheduler AsyncIOScheduler 또는 BackgroundScheduler 인스턴스.
        """
        from apscheduler.triggers.cron import CronTrigger

        # 매일 03:00 — cleanup_candles + cleanup_logs
        scheduler.add_job(
            self._daily_cleanup,
            CronTrigger(hour=3, minute=0),
            id="storage_daily_cleanup",
            replace_existing=True,
            misfire_grace_time=300,
        )

        # 매주 일요일 02:00 — archive_old_trades
        scheduler.add_job(
            self.archive_old_trades,
            CronTrigger(day_of_week="sun", hour=2, minute=0),
            id="storage_archive_trades",
            replace_existing=True,
            misfire_grace_time=600,
        )

        # 매주 일요일 03:30 — vacuum_database
        scheduler.add_job(
            self.vacuum_database,
            CronTrigger(day_of_week="sun", hour=3, minute=30),
            id="storage_vacuum",
            replace_existing=True,
            misfire_grace_time=600,
        )

        # 매주 일요일 04:00 — cleanup_model_checkpoints
        scheduler.add_job(
            self.cleanup_model_checkpoints,
            CronTrigger(day_of_week="sun", hour=4, minute=0),
            id="storage_cleanup_models",
            replace_existing=True,
            misfire_grace_time=300,
        )

        # 매월 1일 04:30 — cleanup_backtest_results
        scheduler.add_job(
            self.cleanup_backtest_results,
            CronTrigger(day=1, hour=4, minute=30),
            id="storage_cleanup_backtest",
            replace_existing=True,
            misfire_grace_time=300,
        )

        # 매시간 :00 — check_disk_usage
        scheduler.add_job(
            self.check_disk_usage,
            CronTrigger(minute=0),
            id="storage_check_disk",
            replace_existing=True,
            misfire_grace_time=120,
        )

        logger.info("[StorageManager] APScheduler 등록 완료 (6개 작업)")

    def _daily_cleanup(self) -> None:
        """cleanup_candles + cleanup_logs 묶음."""
        self.cleanup_candles()
        self.cleanup_logs()

    # ──────────────────────────────────────────────────────────────
    # 감사 로그
    # ──────────────────────────────────────────────────────────────

    def _write_audit(
        self,
        conn: sqlite3.Connection,
        vacuum_triggered: int = 0,
        rows_deleted: int = 0,
        tables_pruned: list[str] | None = None,
    ) -> None:
        """storage_audit_log 테이블에 작업 기록."""
        disk = shutil.disk_usage(str(self._db_path.parent))
        conn.execute(
            """
            INSERT INTO storage_audit_log
                (timestamp, db_size_mb, vacuum_triggered, rows_deleted, tables_pruned, disk_free_gb)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                self._db_size_mb(),
                vacuum_triggered,
                rows_deleted,
                json.dumps(tables_pruned or []),
                disk.free / (1024 ** 3),
            ),
        )

    # ──────────────────────────────────────────────────────────────
    # 내부 유틸
    # ──────────────────────────────────────────────────────────────

    def _db_size_mb(self) -> float:
        """DB 파일 크기 (MB)."""
        if not self._db_path.exists():
            return 0.0
        return self._db_path.stat().st_size / (1024 * 1024)

    def get_stats(self) -> dict[str, Any]:
        """현재 상태 요약 (대시보드용)."""
        disk = shutil.disk_usage(str(self._db_path.parent))
        return {
            "db_path": str(self._db_path),
            "db_size_mb": self._db_size_mb(),
            "disk_free_gb": disk.free / (1024 ** 3),
            "disk_total_gb": disk.total / (1024 ** 3),
            "archive_db_path": str(self._archive_db),
        }
