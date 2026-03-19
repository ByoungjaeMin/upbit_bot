"""
scripts/test_run_hyperopt.py — run_hyperopt.py 단위 테스트

실행:
    pytest upbit_bot/scripts/test_run_hyperopt.py -v
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch, mock_open

import pytest

# ─────────────────────────────────────────────────────────────────
# sys.path 설정
# ─────────────────────────────────────────────────────────────────

_SCRIPT_DIR = Path(__file__).parent
_BOT_DIR    = _SCRIPT_DIR.parent

if str(_BOT_DIR) not in sys.path:
    sys.path.insert(0, str(_BOT_DIR))

from scripts.run_hyperopt import (
    VALID_STRATEGIES,
    _build_telegram_message,
    _get_strategy_fn,
    parse_args,
    save_json,
    update_config_yaml,
)


# ─────────────────────────────────────────────────────────────────
# 픽스처 — 가짜 HyperoptResult / BacktestParams
# ─────────────────────────────────────────────────────────────────

@dataclass
class _FakeParams:
    rsi_period: int = 14
    ema_short: int = 7
    ema_long: int = 25
    adx_threshold: float = 20.0
    stop_loss_pct: float = 0.07
    take_profit_pct: float = 0.10
    kelly_fraction: float = 0.25
    atr_multiplier: float = 2.0
    grid_levels: int = 10
    dca_step_pct: float = -0.03
    ensemble_threshold: float = 0.62


@dataclass
class _FakeHyperoptResult:
    best_params: _FakeParams = field(default_factory=_FakeParams)
    best_oos_sharpe: float = 1.75
    n_trials_completed: int = 200
    study_name: str = "wf_hyperopt"
    param_importances: dict[str, float] = field(
        default_factory=lambda: {
            "ensemble_threshold": 0.35,
            "adx_threshold": 0.20,
            "kelly_fraction": 0.15,
        }
    )

    def summary(self) -> str:
        return f"[Hyperopt] 최적 OOS 샤프: {self.best_oos_sharpe:.4f} trials={self.n_trials_completed}"


# ─────────────────────────────────────────────────────────────────
# 1. parse_args
# ─────────────────────────────────────────────────────────────────

class TestParseArgs:
    def test_default_n_trials(self) -> None:
        args = parse_args([])
        assert args.n_trials == 200

    def test_custom_n_trials(self) -> None:
        args = parse_args(["--n-trials", "50"])
        assert args.n_trials == 50

    def test_default_strategy(self) -> None:
        args = parse_args([])
        assert args.strategy == "all"

    def test_valid_strategies(self) -> None:
        for s in VALID_STRATEGIES:
            args = parse_args(["--strategy", s])
            assert args.strategy == s

    def test_invalid_strategy_raises(self) -> None:
        with pytest.raises(SystemExit):
            parse_args(["--strategy", "unknown_strategy"])

    def test_no_telegram_flag(self) -> None:
        args = parse_args(["--no-telegram"])
        assert args.no_telegram is True

    def test_no_update_config_flag(self) -> None:
        args = parse_args(["--no-update-config"])
        assert args.no_update_config is True

    def test_output_dir_default(self) -> None:
        args = parse_args([])
        assert args.output_dir == "results/hyperopt"

    def test_timeout_default(self) -> None:
        args = parse_args([])
        assert args.timeout == 7200


# ─────────────────────────────────────────────────────────────────
# 2. update_config_yaml
# ─────────────────────────────────────────────────────────────────

class TestUpdateConfigYaml:
    def test_creates_backtest_params_section(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        # 최소 config 작성
        config_file.write_text("general:\n  phase: A\n", encoding="utf-8")

        params = _FakeParams(ensemble_threshold=0.65, adx_threshold=22.0)
        update_config_yaml(params, config_path=config_file)

        import yaml
        data = yaml.safe_load(config_file.read_text(encoding="utf-8"))
        assert "backtest_params" in data
        assert "all" in data["backtest_params"]
        assert data["backtest_params"]["all"]["ensemble_threshold"] == pytest.approx(0.65, abs=1e-4)

    def test_preserves_existing_sections(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "general:\n  phase: B\ntrading:\n  dry_run: true\n",
            encoding="utf-8",
        )

        params = _FakeParams()
        update_config_yaml(params, config_path=config_file)

        import yaml
        data = yaml.safe_load(config_file.read_text(encoding="utf-8"))
        # 기존 섹션 보존 확인
        assert data["general"]["phase"] == "B"
        assert data["trading"]["dry_run"] is True

    def test_strategy_key_used(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text("{}\n", encoding="utf-8")

        params = _FakeParams()
        update_config_yaml(params, config_path=config_file, strategy="trend")

        import yaml
        data = yaml.safe_load(config_file.read_text(encoding="utf-8"))
        assert "trend" in data["backtest_params"]

    def test_creates_config_if_not_exists(self, tmp_path: Path) -> None:
        config_file = tmp_path / "new_config.yaml"
        assert not config_file.exists()

        params = _FakeParams()
        update_config_yaml(params, config_path=config_file)

        assert config_file.exists()

    def test_last_updated_field_written(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text("{}\n", encoding="utf-8")

        params = _FakeParams()
        update_config_yaml(params, config_path=config_file)

        import yaml
        data = yaml.safe_load(config_file.read_text(encoding="utf-8"))
        assert "_last_updated" in data["backtest_params"]

    def test_raises_import_error_without_yaml(self, tmp_path: Path) -> None:
        """PyYAML 없으면 ImportError 발생 확인 (builtins import mock)."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("{}\n", encoding="utf-8")

        params = _FakeParams()
        with patch("builtins.__import__",
                   side_effect=lambda name, *a, **kw: (_ for _ in ()).throw(ImportError("yaml"))
                   if name == "yaml" else __import__(name, *a, **kw)):
            with pytest.raises(ImportError):
                update_config_yaml(params, config_path=config_file)


# ─────────────────────────────────────────────────────────────────
# 3. save_json
# ─────────────────────────────────────────────────────────────────

class TestSaveJson:
    def test_creates_file(self, tmp_path: Path) -> None:
        result = _FakeHyperoptResult()
        path = save_json(
            result=result,
            output_dir=tmp_path,
            run_id="20240101_120000",
            strategy="all",
            n_trials=200,
        )
        assert path.exists()
        assert path.suffix == ".json"

    def test_json_contains_best_params(self, tmp_path: Path) -> None:
        result = _FakeHyperoptResult()
        path = save_json(
            result=result,
            output_dir=tmp_path,
            run_id="param_run",
            strategy="trend",
            n_trials=100,
        )
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["run_id"] == "param_run"
        assert data["strategy"] == "trend"
        assert data["best_oos_sharpe"] == pytest.approx(1.75, abs=1e-3)
        assert "best_params" in data
        assert data["best_params"]["ensemble_threshold"] == pytest.approx(0.62, abs=1e-4)

    def test_param_importances_included(self, tmp_path: Path) -> None:
        result = _FakeHyperoptResult()
        path = save_json(
            result=result,
            output_dir=tmp_path,
            run_id="imp_run",
            strategy="all",
            n_trials=200,
        )
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "param_importances" in data
        assert "ensemble_threshold" in data["param_importances"]

    def test_output_dir_created_if_missing(self, tmp_path: Path) -> None:
        nested = tmp_path / "x" / "y"
        result = _FakeHyperoptResult()
        save_json(
            result=result,
            output_dir=nested,
            run_id="nested",
            strategy="all",
            n_trials=200,
        )
        assert nested.exists()

    def test_filename_includes_strategy(self, tmp_path: Path) -> None:
        result = _FakeHyperoptResult()
        path = save_json(
            result=result,
            output_dir=tmp_path,
            run_id="s_run",
            strategy="grid",
            n_trials=50,
        )
        assert "grid" in path.name


# ─────────────────────────────────────────────────────────────────
# 4. _get_strategy_fn
# ─────────────────────────────────────────────────────────────────

class TestGetStrategyFn:
    def test_all_strategies_return_none_or_callable(self) -> None:
        """현재 Phase B 이전에는 모든 전략이 None 반환."""
        for s in VALID_STRATEGIES:
            fn = _get_strategy_fn(s)
            assert fn is None or callable(fn)

    def test_all_returns_none(self) -> None:
        assert _get_strategy_fn("all") is None

    def test_trend_returns_none(self) -> None:
        assert _get_strategy_fn("trend") is None


# ─────────────────────────────────────────────────────────────────
# 5. main() 통합 (mock 기반)
# ─────────────────────────────────────────────────────────────────

class TestMain:
    def test_main_returns_1_on_data_error(self, tmp_path: Path) -> None:
        """데이터 없으면 exit 1."""
        from scripts.run_hyperopt import main

        with patch("scripts.run_hyperopt._import_hyperopt_modules") as mock_import, \
             patch("scripts.run_hyperopt.load_ohlcv_data",
                   side_effect=RuntimeError("데이터 없음")):
            mock_import.return_value = (MagicMock(), MagicMock())
            result = main([
                "--output-dir", str(tmp_path),
                "--no-telegram",
                "--no-update-config",
            ])
        assert result == 1

    def test_main_returns_0_on_success(self, tmp_path: Path) -> None:
        """정상 완료 시 exit 0."""
        from scripts.run_hyperopt import main
        import pandas as pd

        mock_df = MagicMock()
        mock_df.index = pd.date_range("2022-01-01", periods=500, freq="5min")

        fake_result = _FakeHyperoptResult()

        MockHyperoptEngine = MagicMock()
        MockBacktestEngine = MagicMock()

        engine_instance = MagicMock()
        engine_instance.optimize_with_walkforward.return_value = fake_result
        MockHyperoptEngine.return_value = engine_instance

        with patch("scripts.run_hyperopt._import_hyperopt_modules",
                   return_value=(MockHyperoptEngine, MockBacktestEngine)), \
             patch("scripts.run_hyperopt.load_ohlcv_data", return_value=mock_df), \
             patch("scripts.run_hyperopt.update_config_yaml"), \
             patch("scripts.run_hyperopt.send_telegram"):
            result = main([
                "--output-dir", str(tmp_path),
                "--no-telegram",
            ])
        assert result == 0

    def test_main_skips_config_update_with_flag(self, tmp_path: Path) -> None:
        """--no-update-config 플래그 시 update_config_yaml 미호출."""
        from scripts.run_hyperopt import main
        import pandas as pd

        mock_df = MagicMock()
        mock_df.index = pd.date_range("2022-01-01", periods=500, freq="5min")

        fake_result = _FakeHyperoptResult()
        MockHyperoptEngine = MagicMock()
        MockBacktestEngine = MagicMock()
        engine_instance = MagicMock()
        engine_instance.optimize_with_walkforward.return_value = fake_result
        MockHyperoptEngine.return_value = engine_instance

        with patch("scripts.run_hyperopt._import_hyperopt_modules",
                   return_value=(MockHyperoptEngine, MockBacktestEngine)), \
             patch("scripts.run_hyperopt.load_ohlcv_data", return_value=mock_df), \
             patch("scripts.run_hyperopt.update_config_yaml") as mock_update, \
             patch("scripts.run_hyperopt.send_telegram"):
            main([
                "--output-dir", str(tmp_path),
                "--no-telegram",
                "--no-update-config",
            ])

        mock_update.assert_not_called()

    def test_telegram_message_format(self) -> None:
        result = _FakeHyperoptResult()
        msg = _build_telegram_message(
            result=result,
            run_id="tg_run",
            strategy="all",
            elapsed_sec=3600.0,
            config_updated=True,
        )
        assert "Hyperopt" in msg
        assert "tg_run" in msg
        assert "1.7500" in msg  # best_oos_sharpe
        assert "config.yaml 업데이트: 완료" in msg

    def test_telegram_message_config_not_updated(self) -> None:
        result = _FakeHyperoptResult()
        msg = _build_telegram_message(
            result=result,
            run_id="no_cfg",
            strategy="trend",
            elapsed_sec=100.0,
            config_updated=False,
        )
        assert "config.yaml 업데이트: 건너뜀" in msg

    def test_n_trials_passed_to_engine(self, tmp_path: Path) -> None:
        """--n-trials 값이 HyperoptEngine 생성자에 전달되는지 확인."""
        from scripts.run_hyperopt import main
        import pandas as pd

        mock_df = MagicMock()
        mock_df.index = pd.date_range("2022-01-01", periods=500, freq="5min")

        fake_result = _FakeHyperoptResult()
        MockHyperoptEngine = MagicMock()
        MockBacktestEngine = MagicMock()
        engine_instance = MagicMock()
        engine_instance.optimize_with_walkforward.return_value = fake_result
        MockHyperoptEngine.return_value = engine_instance

        with patch("scripts.run_hyperopt._import_hyperopt_modules",
                   return_value=(MockHyperoptEngine, MockBacktestEngine)), \
             patch("scripts.run_hyperopt.load_ohlcv_data", return_value=mock_df), \
             patch("scripts.run_hyperopt.update_config_yaml"), \
             patch("scripts.run_hyperopt.send_telegram"):
            main([
                "--n-trials", "42",
                "--output-dir", str(tmp_path),
                "--no-telegram",
            ])

        # HyperoptEngine이 n_trials=42로 생성됐는지 확인
        _, kwargs = MockHyperoptEngine.call_args
        assert kwargs.get("n_trials") == 42
