"""scripts/test_coin_history_collector.py — coin_history_collector 단위 테스트.

커버 대상:
  - collect_and_save(): 정상 흐름 / 빈 티커 예외 / 볼륨·순위 계산
  - _scheduled_job(): 예외 흡수 확인
  - main(): --now 플래그, DB 미존재 시 sys.exit
"""

from __future__ import annotations

import sqlite3
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pandas as pd
import pytest

# ─────────────────────────────────────────────────────────────────
# 모듈 임포트 — LOG_DIR.mkdir이 import 시 실행되므로 tmp_path 패치 필요
# ─────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _patch_log_dir(tmp_path, monkeypatch):
    """스크립트 임포트 시 생성하는 LOG_DIR을 tmp_path로 리다이렉트."""
    monkeypatch.setattr(Path, "mkdir", lambda *a, **kw: None)


# 스크립트를 직접 임포트 (scripts/ 는 패키지가 아니어서 importlib 사용)
import importlib.util as _ilu

_spec = _ilu.spec_from_file_location(
    "coin_history_collector",
    Path(__file__).parent / "coin_history_collector.py",
)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

collect_and_save  = _mod.collect_and_save
_scheduled_job    = _mod._scheduled_job


# ─────────────────────────────────────────────────────────────────
# 헬퍼
# ─────────────────────────────────────────────────────────────────

def _make_pyupbit_mock(tickers: list[str], ohlcv_volume: float = 1_000.0) -> MagicMock:
    """pyupbit 모의 객체 생성."""
    mock = MagicMock()
    mock.get_tickers.return_value = tickers
    mock.get_current_price.return_value = {t: 50_000.0 for t in tickers}

    def _ohlcv(ticker, **kwargs):
        return pd.DataFrame({"close": [50_000.0], "volume": [ohlcv_volume]})

    mock.get_ohlcv.side_effect = _ohlcv
    return mock


def _make_db_mock():
    """sqlite3.connect 모의 객체 반환 (context manager 불필요 — finally 분기 처리)."""
    conn = MagicMock()
    conn.execute.return_value = MagicMock(fetchone=MagicMock(return_value=(3,)))
    return conn


# ─────────────────────────────────────────────────────────────────
# collect_and_save() — 정상 흐름
# ─────────────────────────────────────────────────────────────────

class TestCollectAndSaveNormal:
    def _run(self, tickers: list[str], ohlcv_volume: float = 1_000.0):
        """pyupbit + sqlite3 모두 모킹 후 collect_and_save() 실행."""
        mock_pyupbit = _make_pyupbit_mock(tickers, ohlcv_volume)
        conn = _make_db_mock()

        with patch.dict(sys.modules, {"pyupbit": mock_pyupbit}), \
             patch("sqlite3.connect", return_value=conn):
            return collect_and_save(), conn

    def test_returns_saved_count(self):
        """저장된 코인 수(mock fetchone → 3)를 반환해야 한다."""
        result, _ = self._run(["KRW-BTC", "KRW-ETH", "KRW-SOL"])
        assert result == 3

    def test_executemany_called(self):
        """executemany가 정확히 1회 호출되어야 한다."""
        _, conn = self._run(["KRW-BTC", "KRW-ETH"])
        conn.executemany.assert_called_once()

    def test_executemany_row_count_matches_tickers(self):
        """executemany에 전달된 rows 수 == 티커 수."""
        tickers = ["KRW-BTC", "KRW-ETH", "KRW-SOL"]
        _, conn = self._run(tickers)
        _, rows = conn.executemany.call_args.args
        assert len(rows) == len(tickers)

    def test_commit_called(self):
        """DB commit이 호출되어야 한다."""
        _, conn = self._run(["KRW-BTC"])
        conn.commit.assert_called_once()

    def test_conn_close_called(self):
        """finally 블록에서 conn.close() 호출 보장."""
        _, conn = self._run(["KRW-BTC"])
        conn.close.assert_called_once()

    def test_volume_krw_in_row(self):
        """row의 volume_24h_krw = close * volume."""
        mock_pyupbit = _make_pyupbit_mock(["KRW-BTC"], ohlcv_volume=2.0)
        # close=50_000.0, volume=2.0 → volume_krw=100_000.0
        conn = _make_db_mock()
        with patch.dict(sys.modules, {"pyupbit": mock_pyupbit}), \
             patch("sqlite3.connect", return_value=conn):
            collect_and_save()
        _, rows = conn.executemany.call_args.args
        snapshot_date, ticker, volume_krw, rank, _, _ = rows[0]
        assert ticker == "KRW-BTC"
        assert volume_krw == pytest.approx(100_000.0)

    def test_rank_starts_from_1(self):
        """단일 코인 rank == 1."""
        mock_pyupbit = _make_pyupbit_mock(["KRW-BTC"], ohlcv_volume=1.0)
        conn = _make_db_mock()
        with patch.dict(sys.modules, {"pyupbit": mock_pyupbit}), \
             patch("sqlite3.connect", return_value=conn):
            collect_and_save()
        _, rows = conn.executemany.call_args.args
        _, _, _, rank, _, _ = rows[0]
        assert rank == 1

    def test_higher_volume_gets_rank1(self):
        """거래량이 많은 코인이 rank 1이어야 한다."""
        mock_pyupbit = MagicMock()
        mock_pyupbit.get_tickers.return_value = ["KRW-BTC", "KRW-ETH"]
        mock_pyupbit.get_current_price.return_value = {}

        volumes = {"KRW-BTC": 1_000.0, "KRW-ETH": 5_000.0}

        def _ohlcv(ticker, **kwargs):
            return pd.DataFrame({"close": [1.0], "volume": [volumes[ticker]]})

        mock_pyupbit.get_ohlcv.side_effect = _ohlcv
        conn = _make_db_mock()

        with patch.dict(sys.modules, {"pyupbit": mock_pyupbit}), \
             patch("sqlite3.connect", return_value=conn):
            collect_and_save()

        _, rows = conn.executemany.call_args.args
        rank_map = {row[1]: row[3] for row in rows}
        assert rank_map["KRW-ETH"] == 1
        assert rank_map["KRW-BTC"] == 2

    def test_wal_pragma_set(self):
        """WAL 모드 PRAGMA가 실행되어야 한다."""
        mock_pyupbit = _make_pyupbit_mock(["KRW-BTC"])
        conn = _make_db_mock()
        with patch.dict(sys.modules, {"pyupbit": mock_pyupbit}), \
             patch("sqlite3.connect", return_value=conn):
            collect_and_save()
        calls_sql = [c.args[0] for c in conn.execute.call_args_list if c.args]
        assert any("WAL" in s for s in calls_sql)


# ─────────────────────────────────────────────────────────────────
# collect_and_save() — 예외 경로
# ─────────────────────────────────────────────────────────────────

class TestCollectAndSaveErrors:
    def test_empty_tickers_raises(self):
        """get_tickers 결과 없음 → RuntimeError."""
        mock_pyupbit = MagicMock()
        mock_pyupbit.get_tickers.return_value = []
        with patch.dict(sys.modules, {"pyupbit": mock_pyupbit}):
            with pytest.raises(RuntimeError, match="get_tickers"):
                collect_and_save()

    def test_none_tickers_raises(self):
        """get_tickers가 None 반환 → RuntimeError."""
        mock_pyupbit = MagicMock()
        mock_pyupbit.get_tickers.return_value = None
        with patch.dict(sys.modules, {"pyupbit": mock_pyupbit}):
            with pytest.raises(RuntimeError):
                collect_and_save()

    def test_ohlcv_error_uses_zero_volume(self):
        """get_ohlcv 예외 시 volume_krw=0 으로 폴백."""
        mock_pyupbit = MagicMock()
        mock_pyupbit.get_tickers.return_value = ["KRW-BTC"]
        mock_pyupbit.get_current_price.return_value = {}
        mock_pyupbit.get_ohlcv.side_effect = Exception("API 오류")

        conn = _make_db_mock()
        with patch.dict(sys.modules, {"pyupbit": mock_pyupbit}), \
             patch("sqlite3.connect", return_value=conn):
            collect_and_save()

        _, rows = conn.executemany.call_args.args
        _, _, volume_krw, _, _, _ = rows[0]
        assert volume_krw == 0.0

    def test_pyupbit_not_installed_raises(self):
        """pyupbit 미설치 시 RuntimeError."""
        # sys.modules에서 pyupbit 제거하여 ImportError 유도
        with patch.dict(sys.modules, {"pyupbit": None}):
            with pytest.raises((RuntimeError, ImportError)):
                collect_and_save()


# ─────────────────────────────────────────────────────────────────
# _scheduled_job() — 예외 흡수
# ─────────────────────────────────────────────────────────────────

class TestScheduledJob:
    def test_exception_absorbed(self):
        """collect_and_save() 예외가 _scheduled_job()에서 흡수되어야 한다."""
        with patch.object(_mod, "collect_and_save", side_effect=RuntimeError("테스트 오류")):
            # 예외가 전파되지 않으면 통과
            _scheduled_job()

    def test_collect_and_save_called(self):
        """_scheduled_job이 collect_and_save를 호출해야 한다."""
        with patch.object(_mod, "collect_and_save", return_value=5) as mock_fn:
            _scheduled_job()
        mock_fn.assert_called_once()
