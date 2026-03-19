"""test_storage_manager.py — StorageManager 단위 테스트."""

from __future__ import annotations

import json
import sqlite3
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from upbit_bot.monitoring.storage_manager import StorageManager


# ─────────────────────────────────────────────────────────────────
# 픽스처
# ─────────────────────────────────────────────────────────────────

def _create_test_db(db_path: Path) -> None:
    """최소한의 스키마 + 샘플 데이터 생성."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")

    # candles_5m
    conn.execute("""
        CREATE TABLE IF NOT EXISTS candles_5m (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            coin TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            open REAL, high REAL, low REAL, close REAL, volume REAL,
            UNIQUE(coin, timestamp)
        )
    """)

    # candles_1h
    conn.execute("""
        CREATE TABLE IF NOT EXISTS candles_1h (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            coin TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            open REAL, high REAL, low REAL, close REAL, volume REAL,
            UNIQUE(coin, timestamp)
        )
    """)

    # trades
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            coin TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            action INTEGER DEFAULT 0,
            side TEXT NOT NULL,
            price REAL NOT NULL,
            volume REAL DEFAULT 0,
            krw_amount REAL NOT NULL,
            fee REAL DEFAULT 0,
            slippage REAL DEFAULT 0,
            strategy_type TEXT,
            kelly_f REAL,
            position_size REAL,
            pnl REAL,
            pnl_pct REAL,
            is_dry_run INTEGER DEFAULT 1,
            order_id TEXT,
            paper_trade INTEGER DEFAULT 0
        )
    """)

    # ensemble_predictions
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ensemble_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            coin TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            xgb_prob REAL, lgb_prob REAL, lstm_prob REAL, gru_prob REAL,
            weighted_avg REAL DEFAULT 0,
            consensus_count INTEGER DEFAULT 0,
            signal_confirmed INTEGER DEFAULT 0,
            hmm_regime INTEGER DEFAULT -1,
            UNIQUE(coin, timestamp)
        )
    """)

    # sentiment_log
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sentiment_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            coin TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            vader_score REAL,
            UNIQUE(coin, timestamp)
        )
    """)

    # layer1_log
    conn.execute("""
        CREATE TABLE IF NOT EXISTS layer1_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            coin TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            tradeable INTEGER DEFAULT 1,
            regime_strategy TEXT,
            UNIQUE(coin, timestamp)
        )
    """)

    # coin_scan_results
    conn.execute("""
        CREATE TABLE IF NOT EXISTS coin_scan_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            coin TEXT NOT NULL,
            rank_by_volume INTEGER,
            volume_24h_krw REAL,
            is_leverage_token INTEGER DEFAULT 0,
            is_blacklisted INTEGER DEFAULT 0,
            included INTEGER NOT NULL,
            reason_excluded TEXT,
            UNIQUE(timestamp, coin)
        )
    """)

    # kimchi_premium_log
    conn.execute("""
        CREATE TABLE IF NOT EXISTS kimchi_premium_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL UNIQUE,
            upbit_btc_krw REAL NOT NULL,
            binance_btc_usd REAL NOT NULL,
            usd_krw_rate REAL NOT NULL,
            kimchi_premium_pct REAL NOT NULL
        )
    """)

    # onchain_data
    conn.execute("""
        CREATE TABLE IF NOT EXISTS onchain_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            coin TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            exchange_inflow REAL,
            exchange_outflow REAL,
            net_flow REAL,
            UNIQUE(coin, timestamp)
        )
    """)

    # market_indices
    conn.execute("""
        CREATE TABLE IF NOT EXISTS market_indices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL UNIQUE,
            fear_greed REAL
        )
    """)

    # storage_audit_log
    conn.execute("""
        CREATE TABLE IF NOT EXISTS storage_audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            db_size_mb REAL,
            total_rows INTEGER,
            vacuum_triggered INTEGER DEFAULT 0,
            rows_deleted INTEGER DEFAULT 0,
            tables_pruned TEXT,
            disk_free_gb REAL
        )
    """)

    conn.commit()
    conn.close()


def _insert_old_candles(db_path: Path, days_ago: int = 200) -> None:
    """테스트용 오래된 캔들 삽입."""
    old_ts = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT OR IGNORE INTO candles_5m (coin, timestamp, open, high, low, close, volume)"
        " VALUES (?, ?, 1, 1, 1, 1, 1)",
        ("KRW-BTC", old_ts),
    )
    conn.execute(
        "INSERT OR IGNORE INTO sentiment_log (coin, timestamp, vader_score)"
        " VALUES (?, ?, 0.1)",
        ("KRW-BTC", old_ts),
    )
    conn.commit()
    conn.close()


def _insert_old_trades(db_path: Path, days_ago: int = 400) -> None:
    """테스트용 오래된 거래 삽입."""
    old_ts = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """INSERT INTO trades
           (coin, timestamp, side, price, krw_amount, is_dry_run)
           VALUES (?, ?, 'BUY', 1000000, 50000, 0)""",
        ("KRW-BTC", old_ts),
    )
    conn.commit()
    conn.close()


@pytest.fixture
def tmp_db(tmp_path: Path) -> tuple[Path, StorageManager]:
    db_path = tmp_path / "bot.db"
    _create_test_db(db_path)
    mgr = StorageManager(
        db_path=db_path,
        base_dir=tmp_path,
        archive_db_path=tmp_path / "archive.db",
    )
    return db_path, mgr


# ─────────────────────────────────────────────────────────────────
# StorageManager 초기화
# ─────────────────────────────────────────────────────────────────

class TestStorageManagerInit:
    def test_init(self, tmp_db: tuple) -> None:
        db_path, mgr = tmp_db
        assert mgr._db_path == db_path
        assert mgr._archive_db.name == "archive.db"

    def test_db_size_mb_zero_if_tiny(self, tmp_db: tuple) -> None:
        _, mgr = tmp_db
        size = mgr._db_size_mb()
        assert size >= 0

    def test_db_size_mb_nonexistent(self, tmp_path: Path) -> None:
        mgr = StorageManager(db_path=tmp_path / "nonexistent.db")
        assert mgr._db_size_mb() == 0.0

    def test_get_stats(self, tmp_db: tuple) -> None:
        _, mgr = tmp_db
        stats = mgr.get_stats()
        assert "db_size_mb" in stats
        assert "disk_free_gb" in stats
        assert stats["disk_free_gb"] > 0


# ─────────────────────────────────────────────────────────────────
# 1. cleanup_candles
# ─────────────────────────────────────────────────────────────────

class TestCleanupCandles:
    def test_deletes_old_5m_candles(self, tmp_db: tuple) -> None:
        db_path, mgr = tmp_db
        _insert_old_candles(db_path, days_ago=200)  # 200일 > 180일 한도

        result = mgr.cleanup_candles()

        assert result.get("candles_5m", 0) >= 1

    def test_keeps_recent_candles(self, tmp_db: tuple) -> None:
        db_path, mgr = tmp_db
        # 최근 10일 데이터 삽입
        recent_ts = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "INSERT OR IGNORE INTO candles_5m (coin, timestamp, open, high, low, close, volume)"
            " VALUES (?, ?, 1, 1, 1, 1, 1)",
            ("KRW-ETH", recent_ts),
        )
        conn.commit()
        conn.close()

        result = mgr.cleanup_candles()
        assert result.get("candles_5m", 0) == 0  # 최근 데이터는 유지

    def test_deletes_old_sentiment(self, tmp_db: tuple) -> None:
        db_path, mgr = tmp_db
        _insert_old_candles(db_path, days_ago=200)  # sentiment_log도 삽입됨

        result = mgr.cleanup_candles()
        assert result.get("sentiment_log", 0) >= 1

    def test_returns_dict(self, tmp_db: tuple) -> None:
        _, mgr = tmp_db
        result = mgr.cleanup_candles()
        assert isinstance(result, dict)
        assert "candles_5m" in result


# ─────────────────────────────────────────────────────────────────
# 2. vacuum_database
# ─────────────────────────────────────────────────────────────────

class TestVacuumDatabase:
    def test_vacuum_returns_float(self, tmp_db: tuple) -> None:
        _, mgr = tmp_db
        size_after = mgr.vacuum_database()
        assert isinstance(size_after, float)
        assert size_after >= 0

    def test_vacuum_writes_audit(self, tmp_db: tuple) -> None:
        db_path, mgr = tmp_db
        mgr.vacuum_database()
        conn = sqlite3.connect(str(db_path))
        row = conn.execute(
            "SELECT vacuum_triggered FROM storage_audit_log ORDER BY id DESC LIMIT 1"
        ).fetchone()
        conn.close()
        assert row is not None
        assert row[0] == 1


# ─────────────────────────────────────────────────────────────────
# 3. archive_old_trades
# ─────────────────────────────────────────────────────────────────

class TestArchiveOldTrades:
    def test_moves_old_trades(self, tmp_db: tuple) -> None:
        db_path, mgr = tmp_db
        _insert_old_trades(db_path, days_ago=400)

        count = mgr.archive_old_trades()
        assert count == 1

        # archive.db에 저장됐는지 확인
        arch_conn = sqlite3.connect(str(mgr._archive_db))
        rows = arch_conn.execute("SELECT COUNT(*) FROM trades").fetchone()
        arch_conn.close()
        assert rows[0] == 1

    def test_no_old_trades_returns_zero(self, tmp_db: tuple) -> None:
        _, mgr = tmp_db
        count = mgr.archive_old_trades()
        assert count == 0

    def test_original_db_row_deleted(self, tmp_db: tuple) -> None:
        db_path, mgr = tmp_db
        _insert_old_trades(db_path, days_ago=400)

        mgr.archive_old_trades()

        conn = sqlite3.connect(str(db_path))
        rows = conn.execute("SELECT COUNT(*) FROM trades").fetchone()
        conn.close()
        assert rows[0] == 0


# ─────────────────────────────────────────────────────────────────
# 4. cleanup_model_checkpoints
# ─────────────────────────────────────────────────────────────────

class TestCleanupModelCheckpoints:
    def test_no_models_dir(self, tmp_db: tuple) -> None:
        _, mgr = tmp_db
        result = mgr.cleanup_model_checkpoints()
        assert result == 0

    def test_keeps_latest_3(self, tmp_path: Path) -> None:
        db_path = tmp_path / "bot.db"
        _create_test_db(db_path)
        mgr = StorageManager(db_path=db_path, base_dir=tmp_path)

        models_dir = tmp_path / "models"
        models_dir.mkdir()

        # 5개 파일 생성 (수정시간 다르게)
        for i in range(5):
            f = models_dir / f"xgb_{i:04d}.pt"
            f.write_bytes(b"x")
            time.sleep(0.01)

        deleted = mgr.cleanup_model_checkpoints()
        remaining = list(models_dir.glob("*.pt"))
        assert deleted == 2
        assert len(remaining) == 3

    def test_different_model_types(self, tmp_path: Path) -> None:
        db_path = tmp_path / "bot.db"
        _create_test_db(db_path)
        mgr = StorageManager(db_path=db_path, base_dir=tmp_path)

        models_dir = tmp_path / "models"
        models_dir.mkdir()

        # xgb 5개, lgb 4개 생성
        for i in range(5):
            (models_dir / f"xgb_{i}.pt").write_bytes(b"x")
            time.sleep(0.01)
        for i in range(4):
            (models_dir / f"lgb_{i}.pkl").write_bytes(b"x")
            time.sleep(0.01)

        deleted = mgr.cleanup_model_checkpoints()
        assert deleted == 3  # xgb 2 + lgb 1


# ─────────────────────────────────────────────────────────────────
# 5. cleanup_logs
# ─────────────────────────────────────────────────────────────────

class TestCleanupLogs:
    def test_no_logs_dir(self, tmp_db: tuple) -> None:
        _, mgr = tmp_db
        result = mgr.cleanup_logs()
        assert result == 0

    def test_deletes_old_logs(self, tmp_path: Path) -> None:
        db_path = tmp_path / "bot.db"
        _create_test_db(db_path)
        mgr = StorageManager(db_path=db_path, base_dir=tmp_path)

        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()

        old_log = logs_dir / "old.log"
        old_log.write_text("old")
        # 수정시간을 31일 전으로
        old_ts = (datetime.now() - timedelta(days=31)).timestamp()
        import os
        os.utime(str(old_log), (old_ts, old_ts))

        recent_log = logs_dir / "recent.log"
        recent_log.write_text("recent")

        deleted = mgr.cleanup_logs()
        assert deleted == 1
        assert not old_log.exists()
        assert recent_log.exists()


# ─────────────────────────────────────────────────────────────────
# 6. cleanup_backtest_results
# ─────────────────────────────────────────────────────────────────

class TestCleanupBacktestResults:
    def test_no_results_dir(self, tmp_db: tuple) -> None:
        _, mgr = tmp_db
        result = mgr.cleanup_backtest_results()
        assert result == 0

    def test_keeps_latest_10_and_recent_90days(self, tmp_path: Path) -> None:
        db_path = tmp_path / "bot.db"
        _create_test_db(db_path)
        mgr = StorageManager(db_path=db_path, base_dir=tmp_path)

        results_dir = tmp_path / "backtest" / "results"
        results_dir.mkdir(parents=True)

        import os
        # 15개 파일: 5개는 오래됨(100일), 10개는 최신
        for i in range(15):
            f = results_dir / f"result_{i:03d}.json"
            f.write_text("{}")
            if i < 5:
                old_ts = (datetime.now() - timedelta(days=100)).timestamp()
                os.utime(str(f), (old_ts, old_ts))
            else:
                time.sleep(0.01)  # 수정시간 차이

        deleted = mgr.cleanup_backtest_results()
        remaining = list(results_dir.glob("*.json"))
        # 최신 10개는 무조건 보관, 오래된 5개 중 일부 삭제
        assert deleted > 0
        assert len(remaining) >= 10


# ─────────────────────────────────────────────────────────────────
# 7. check_disk_usage
# ─────────────────────────────────────────────────────────────────

class TestCheckDiskUsage:
    def test_returns_dict(self, tmp_db: tuple) -> None:
        _, mgr = tmp_db
        result = mgr.check_disk_usage()
        assert "free_gb" in result
        assert "db_mb" in result
        assert "alert_level" in result

    def test_normal_disk_no_alert(self, tmp_db: tuple) -> None:
        _, mgr = tmp_db
        result = mgr.check_disk_usage()
        # 테스트 환경은 보통 50GB 이상 여유 있음
        assert result["free_gb"] >= 0

    def test_warn_when_low_disk(self, tmp_db: tuple) -> None:
        _, mgr = tmp_db
        notified: list[str] = []
        mgr.set_telegram_callback(lambda msg: notified.append(msg))

        # 디스크 사용량 mock
        import shutil
        fake_usage = shutil.disk_usage("/")._replace(free=int(30e9))  # 30 GB
        with patch("upbit_bot.monitoring.storage_manager.shutil.disk_usage",
                   return_value=fake_usage):
            result = mgr.check_disk_usage()

        assert result["alert_level"] == 1
        assert any("경고" in m for m in notified)

    def test_critical_when_very_low_disk(self, tmp_db: tuple) -> None:
        _, mgr = tmp_db
        notified: list[str] = []
        mgr.set_telegram_callback(lambda msg: notified.append(msg))

        import shutil
        fake_usage = shutil.disk_usage("/")._replace(free=int(10e9))  # 10 GB
        with patch("upbit_bot.monitoring.storage_manager.shutil.disk_usage",
                   return_value=fake_usage):
            result = mgr.check_disk_usage()

        assert result["alert_level"] == 2
        assert any("긴급" in m for m in notified)


# ─────────────────────────────────────────────────────────────────
# 텔레그램 알림 콜백
# ─────────────────────────────────────────────────────────────────

class TestNotify:
    def test_callback_called(self, tmp_db: tuple) -> None:
        _, mgr = tmp_db
        messages: list[str] = []
        mgr.set_telegram_callback(lambda msg: messages.append(msg))

        import shutil
        fake_usage = shutil.disk_usage("/")._replace(free=int(10e9))
        with patch("upbit_bot.monitoring.storage_manager.shutil.disk_usage",
                   return_value=fake_usage):
            mgr.check_disk_usage()

        assert len(messages) >= 1

    def test_no_callback_no_error(self, tmp_db: tuple) -> None:
        """콜백 미설정 시 예외 없어야 함."""
        _, mgr = tmp_db
        import shutil
        fake_usage = shutil.disk_usage("/")._replace(free=int(10e9))
        with patch("upbit_bot.monitoring.storage_manager.shutil.disk_usage",
                   return_value=fake_usage):
            mgr.check_disk_usage()  # 예외 없음


# ─────────────────────────────────────────────────────────────────
# APScheduler 등록
# ─────────────────────────────────────────────────────────────────

class TestSchedule:
    def test_schedule_registers_6_jobs(self, tmp_db: tuple) -> None:
        _, mgr = tmp_db
        mock_scheduler = MagicMock()

        try:
            from apscheduler.triggers.cron import CronTrigger
            mgr.schedule(mock_scheduler)
            assert mock_scheduler.add_job.call_count == 6
        except ImportError:
            pytest.skip("apscheduler 미설치")
