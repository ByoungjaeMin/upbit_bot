"""
scripts/test_run_walk_forward.py — run_walk_forward.py 단위 테스트

실행:
    pytest upbit_bot/scripts/test_run_walk_forward.py -v
"""

from __future__ import annotations

import json
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ─────────────────────────────────────────────────────────────────
# sys.path 설정
# ─────────────────────────────────────────────────────────────────

_SCRIPT_DIR = Path(__file__).parent
_BOT_DIR    = _SCRIPT_DIR.parent

if str(_BOT_DIR) not in sys.path:
    sys.path.insert(0, str(_BOT_DIR))

from scripts.run_walk_forward import (
    _build_telegram_message,
    parse_args,
    run_lookahead_check,
    save_json,
    save_sqlite,
    send_telegram,
)


# ─────────────────────────────────────────────────────────────────
# 픽스처 — 가짜 WalkForwardResult
# ─────────────────────────────────────────────────────────────────

@dataclass
class _FakeCycle:
    cycle_idx: int = 0
    is_start: str = "2022-01-01"
    is_end: str = "2022-07-01"
    oos_start: str = "2022-07-01"
    oos_end: str = "2022-08-01"
    is_sharpe: float = 1.8
    oos_sharpe: float = 1.6
    overfitting_flag: bool = False
    lookahead_passed: bool = True
    n_oos_trades: int = 42

    @property
    def is_oos_ratio(self) -> float:
        return self.oos_sharpe / self.is_sharpe if self.is_sharpe else 0.0


@dataclass
class _FakeWFResult:
    cycles: list[_FakeCycle] = field(default_factory=lambda: [_FakeCycle()])
    avg_oos_sharpe: float = 1.6
    avg_is_sharpe: float = 1.8
    overfitting_cycles: int = 0
    all_oos_pnls: list[float] = field(default_factory=list)
    regime_metrics: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        return f"[WalkForward] {len(self.cycles)}개 사이클 avg_oos={self.avg_oos_sharpe:.3f}"


# ─────────────────────────────────────────────────────────────────
# 1. parse_args
# ─────────────────────────────────────────────────────────────────

class TestParseArgs:
    def test_required_args(self) -> None:
        args = parse_args([
            "--coins", "BTC,ETH",
            "--start", "2022-01-01",
            "--end",   "2024-12-31",
        ])
        assert args.coins == "BTC,ETH"
        assert args.start == "2022-01-01"
        assert args.end   == "2024-12-31"

    def test_default_output_dir(self) -> None:
        args = parse_args([
            "--coins", "BTC",
            "--start", "2022-01-01",
            "--end",   "2024-12-31",
        ])
        assert args.output_dir == "results/walk_forward"

    def test_optional_flags(self) -> None:
        args = parse_args([
            "--coins", "BTC",
            "--start", "2022-01-01",
            "--end",   "2024-12-31",
            "--skip-lookahead",
            "--no-telegram",
            "--is-months", "3",
            "--oos-months", "2",
        ])
        assert args.skip_lookahead is True
        assert args.no_telegram is True
        assert args.is_months == 3
        assert args.oos_months == 2

    def test_missing_coins_raises(self) -> None:
        with pytest.raises(SystemExit):
            parse_args(["--start", "2022-01-01", "--end", "2024-12-31"])

    def test_coins_with_spaces_parsed(self) -> None:
        args = parse_args([
            "--coins", "BTC, ETH, XRP",
            "--start", "2022-01-01",
            "--end",   "2024-12-31",
        ])
        # main()에서 strip()을 하므로 argparse 단계에선 그대로
        assert "BTC" in args.coins


# ─────────────────────────────────────────────────────────────────
# 2. run_lookahead_check
# ─────────────────────────────────────────────────────────────────

class TestRunLookaheadCheck:
    def test_pass_does_not_raise(self) -> None:
        """오염 피처 없으면 예외 없음."""
        mock_report = MagicMock()
        mock_report.passed = True
        mock_report.contamination_count = 0
        mock_report.summary.return_value = "[LookaheadBiasChecker] PASS"

        MockChecker = MagicMock()
        MockChecker.return_value.check.return_value = mock_report

        df = MagicMock()
        df.index = list(range(200))

        run_lookahead_check(df, MockChecker)  # 예외 없어야 함

    def test_fail_raises_value_error(self) -> None:
        """오염 피처 있으면 ValueError 발생."""
        mock_report = MagicMock()
        mock_report.passed = False
        mock_report.contamination_count = 2
        mock_report.contaminated_features = ["ema50_1d", "rsi_1d"]
        mock_report.summary.return_value = "[LookaheadBiasChecker] FAIL"

        MockChecker = MagicMock()
        MockChecker.return_value.check.return_value = mock_report

        df = MagicMock()
        df.index = list(range(200))

        with pytest.raises(ValueError, match="Lookahead Bias"):
            run_lookahead_check(df, MockChecker)


# ─────────────────────────────────────────────────────────────────
# 3. save_json
# ─────────────────────────────────────────────────────────────────

class TestSaveJson:
    def test_creates_file(self, tmp_path: Path) -> None:
        result = _FakeWFResult()
        path = save_json(
            result=result,
            output_dir=tmp_path,
            run_id="20240101_120000",
            coins=["BTC", "ETH"],
            start="2022-01-01",
            end="2024-12-31",
            failures=[],
        )
        assert path.exists()
        assert path.suffix == ".json"

    def test_json_content(self, tmp_path: Path) -> None:
        result = _FakeWFResult()
        path = save_json(
            result=result,
            output_dir=tmp_path,
            run_id="test_run",
            coins=["BTC"],
            start="2022-01-01",
            end="2024-12-31",
            failures=["샤프비율 미달"],
        )
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["run_id"] == "test_run"
        assert data["coins"] == ["BTC"]
        assert data["avg_oos_sharpe"] == round(result.avg_oos_sharpe, 4)
        assert data["live_ready"] is False
        assert "샤프비율 미달" in data["failures"]
        assert len(data["cycles"]) == 1

    def test_live_ready_true_when_no_failures(self, tmp_path: Path) -> None:
        result = _FakeWFResult()
        path = save_json(
            result=result,
            output_dir=tmp_path,
            run_id="ok_run",
            coins=["BTC"],
            start="2022-01-01",
            end="2024-12-31",
            failures=[],
        )
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["live_ready"] is True

    def test_output_dir_created_if_missing(self, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b" / "c"
        result = _FakeWFResult()
        save_json(
            result=result,
            output_dir=nested,
            run_id="nested_run",
            coins=["BTC"],
            start="2022-01-01",
            end="2024-12-31",
            failures=[],
        )
        assert nested.exists()


# ─────────────────────────────────────────────────────────────────
# 4. save_sqlite
# ─────────────────────────────────────────────────────────────────

class TestSaveSqlite:
    def test_creates_tables(self, tmp_path: Path) -> None:
        db_path = str(tmp_path / "test.db")
        result = _FakeWFResult()
        save_sqlite(
            result=result,
            db_path=db_path,
            run_id="sql_run",
            coins=["BTC", "ETH"],
            start="2022-01-01",
            end="2024-12-31",
            failures=[],
        )
        conn = sqlite3.connect(db_path)
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        conn.close()
        assert "wf_results" in tables
        assert "wf_summary" in tables

    def test_cycle_rows_inserted(self, tmp_path: Path) -> None:
        db_path = str(tmp_path / "test.db")
        result = _FakeWFResult(
            cycles=[_FakeCycle(cycle_idx=i) for i in range(3)]
        )
        save_sqlite(
            result=result,
            db_path=db_path,
            run_id="multi_cycle",
            coins=["BTC"],
            start="2022-01-01",
            end="2024-12-31",
            failures=[],
        )
        conn = sqlite3.connect(db_path)
        count = conn.execute(
            "SELECT COUNT(*) FROM wf_results WHERE run_id=?", ("multi_cycle",)
        ).fetchone()[0]
        conn.close()
        assert count == 3

    def test_summary_live_ready_flag(self, tmp_path: Path) -> None:
        db_path = str(tmp_path / "test.db")
        result = _FakeWFResult()

        save_sqlite(
            result=result,
            db_path=db_path,
            run_id="fail_run",
            coins=["BTC"],
            start="2022-01-01",
            end="2024-12-31",
            failures=["샤프 미달"],
        )
        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT live_ready FROM wf_summary WHERE run_id=?", ("fail_run",)
        ).fetchone()
        conn.close()
        assert row[0] == 0  # live_ready=False


# ─────────────────────────────────────────────────────────────────
# 5. main() 통합 (mock 기반)
# ─────────────────────────────────────────────────────────────────

class TestMain:
    def test_main_returns_1_on_data_error(self, tmp_path: Path) -> None:
        """데이터 없으면 exit 1."""
        from scripts.run_walk_forward import main

        with patch("scripts.run_walk_forward._import_backtest_modules") as mock_import, \
             patch("scripts.run_walk_forward.load_ohlcv_data",
                   side_effect=RuntimeError("데이터 없음")):
            mock_import.return_value = (MagicMock(), MagicMock(), MagicMock(), MagicMock())
            result = main([
                "--coins", "BTC",
                "--start", "2022-01-01",
                "--end",   "2024-12-31",
                "--output-dir", str(tmp_path),
                "--no-telegram",
            ])
        assert result == 1

    def test_main_returns_1_on_lookahead_fail(self, tmp_path: Path) -> None:
        """Lookahead 실패 시 exit 1."""
        from scripts.run_walk_forward import main
        import pandas as pd

        mock_df = MagicMock()
        mock_df.index = pd.date_range("2022-01-01", periods=200, freq="5min")

        with patch("scripts.run_walk_forward._import_backtest_modules") as mock_import, \
             patch("scripts.run_walk_forward.load_ohlcv_data", return_value=mock_df), \
             patch("scripts.run_walk_forward.run_lookahead_check",
                   side_effect=ValueError("오염 피처 발견")):
            mock_import.return_value = (MagicMock(), MagicMock(), MagicMock(), MagicMock())
            result = main([
                "--coins", "BTC",
                "--start", "2022-01-01",
                "--end",   "2024-12-31",
                "--output-dir", str(tmp_path),
                "--no-telegram",
            ])
        assert result == 1

    def test_main_returns_0_on_success(self, tmp_path: Path) -> None:
        """정상 완료 시 exit 0."""
        from scripts.run_walk_forward import main
        import pandas as pd

        mock_df = MagicMock()
        mock_df.index = pd.date_range("2022-01-01", periods=200, freq="5min")

        fake_result = _FakeWFResult()

        MockWFOptimizer  = MagicMock()
        MockEngine       = MagicMock()
        MockSurv         = MagicMock()
        MockLookahead    = MagicMock()

        optimizer_instance = MagicMock()
        optimizer_instance.run.return_value = fake_result
        optimizer_instance.check_live_readiness.return_value = []
        MockWFOptimizer.return_value = optimizer_instance

        with patch("scripts.run_walk_forward._import_backtest_modules",
                   return_value=(MockWFOptimizer, MockEngine, MockSurv, MockLookahead)), \
             patch("scripts.run_walk_forward.load_ohlcv_data", return_value=mock_df), \
             patch("scripts.run_walk_forward.run_lookahead_check"), \
             patch("scripts.run_walk_forward.save_sqlite"), \
             patch("backtest.walk_forward.BacktestParams", MagicMock()):
            result = main([
                "--coins", "BTC",
                "--start", "2022-01-01",
                "--end",   "2024-12-31",
                "--output-dir", str(tmp_path),
                "--no-telegram",
            ])
        assert result == 0

    def test_telegram_message_format(self) -> None:
        """텔레그램 메시지에 필수 필드 포함 확인."""
        result = _FakeWFResult()
        msg = _build_telegram_message(
            result=result,
            run_id="test_run",
            coins=["BTC", "ETH"],
            start="2022-01-01",
            end="2024-12-31",
            failures=[],
            elapsed_sec=120.5,
        )
        assert "Walk-Forward" in msg
        assert "test_run" in msg
        assert "1.600" in msg  # avg_oos_sharpe
        assert "실전 전환 가능" in msg

    def test_telegram_message_with_failures(self) -> None:
        result = _FakeWFResult(avg_oos_sharpe=0.8)
        msg = _build_telegram_message(
            result=result,
            run_id="fail_run",
            coins=["BTC"],
            start="2022-01-01",
            end="2024-12-31",
            failures=["샤프비율 0.800 < 1.5"],
            elapsed_sec=60.0,
        )
        assert "미통과" in msg
        assert "샤프비율" in msg
