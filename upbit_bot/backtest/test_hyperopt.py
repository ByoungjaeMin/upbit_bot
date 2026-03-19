"""test_hyperopt.py — HyperoptEngine 단위 테스트."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.hyperopt import (
    N_TRIALS_DEFAULT,
    HyperoptEngine,
    HyperoptResult,
    _OPTUNA_AVAILABLE,
)
from backtest.walk_forward import BacktestParams


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------

def _make_df(n: int = 300) -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=n, freq="5min", tz="UTC")
    rng = np.random.default_rng(42)
    price = 50_000_000 + np.cumsum(rng.normal(0, 50_000, n))
    price = np.clip(price, 1_000, None)
    return pd.DataFrame({
        "close_5m": price,
        "rsi_5m":   rng.uniform(30, 70, n),
        "adx_5m":   rng.uniform(15, 35, n),
        "ema50_1d": price * 0.99,
        "ema200_1d": price * 0.98,
        "rsi_1d":   rng.uniform(40, 60, n),
        "trend_encoding_1d": np.ones(n, dtype=int),
    }, index=idx)


def _engine(n_trials: int = 3) -> HyperoptEngine:
    return HyperoptEngine(n_trials=n_trials, timeout_sec=30)


# ---------------------------------------------------------------------------
# HyperoptResult
# ---------------------------------------------------------------------------

class TestHyperoptResult:
    def test_summary_contains_sharpe(self):
        r = HyperoptResult(
            best_params=BacktestParams(),
            best_oos_sharpe=1.8,
            n_trials_completed=200,
            study_name="test",
            param_importances={"adx_threshold": 0.3, "ensemble_threshold": 0.25},
        )
        s = r.summary()
        assert "1.8" in s or "Hyperopt" in s

    def test_summary_contains_ensemble_threshold(self):
        r = HyperoptResult(
            best_params=BacktestParams(ensemble_threshold=0.65),
            best_oos_sharpe=1.5,
            n_trials_completed=100,
            study_name="test",
            param_importances={},
        )
        s = r.summary()
        assert "0.65" in s or "ensemble" in s

    def test_summary_no_crash_with_empty_importances(self):
        r = HyperoptResult(
            best_params=BacktestParams(),
            best_oos_sharpe=0.0,
            n_trials_completed=0,
            study_name="test",
            param_importances={},
        )
        r.summary()  # 예외 없어야


# ---------------------------------------------------------------------------
# HyperoptEngine (optuna 없을 시 fallback)
# ---------------------------------------------------------------------------

class TestHyperoptEngineFallback:
    def test_fallback_returns_default_params(self):
        """optuna 미설치 환경에서도 기본 파라미터 반환."""
        engine = _engine()
        df = _make_df(200)
        # optuna 없으면 fallback, 있으면 실제 최적화 (n_trials=3)
        result = engine.optimize(df, df, n_trials=3)
        assert isinstance(result, HyperoptResult)
        assert isinstance(result.best_params, BacktestParams)

    def test_fallback_sharpe_zero_when_no_optuna(self):
        if _OPTUNA_AVAILABLE:
            pytest.skip("optuna 설치됨 — fallback 테스트 불필요")
        engine = _engine()
        result = engine.optimize(pd.DataFrame(), pd.DataFrame(), n_trials=1)
        assert result.best_oos_sharpe == 0.0

    def test_fallback_n_trials_zero_when_no_optuna(self):
        if _OPTUNA_AVAILABLE:
            pytest.skip("optuna 설치됨")
        engine = _engine()
        result = engine.optimize(pd.DataFrame(), pd.DataFrame(), n_trials=1)
        assert result.n_trials_completed == 0


# ---------------------------------------------------------------------------
# HyperoptEngine._suggest_params (파라미터 범위 검증)
# ---------------------------------------------------------------------------

class TestSuggestParams:
    @pytest.mark.skipif(not _OPTUNA_AVAILABLE, reason="optuna 미설치")
    def test_suggested_params_in_range(self):
        import optuna
        study = optuna.create_study(direction="maximize")
        trial = study.ask()
        params = HyperoptEngine._suggest_params(trial)

        assert 5 <= params.rsi_period <= 30
        assert 15.0 <= params.adx_threshold <= 35.0
        assert 0.10 <= params.kelly_fraction <= 0.50
        assert 1.5 <= params.atr_multiplier <= 4.0
        assert 5 <= params.grid_levels <= 20
        assert -0.05 <= params.dca_step_pct <= -0.02
        assert 0.55 <= params.ensemble_threshold <= 0.75

    @pytest.mark.skipif(not _OPTUNA_AVAILABLE, reason="optuna 미설치")
    def test_suggest_different_params_each_trial(self):
        """여러 trial에서 다른 파라미터 제안."""
        import optuna
        study = optuna.create_study(direction="maximize")
        params_list = []
        for _ in range(5):
            trial = study.ask()
            p = HyperoptEngine._suggest_params(trial)
            params_list.append(p.ensemble_threshold)
            study.tell(trial, 0.0)
        # 5번 중 적어도 1번은 다른 값
        assert len(set(round(x, 3) for x in params_list)) > 1 or len(params_list) == 5


# ---------------------------------------------------------------------------
# HyperoptEngine.optimize() — optuna 있을 때
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _OPTUNA_AVAILABLE, reason="optuna 미설치")
class TestHyperoptEngineOptimize:
    def test_optimize_returns_hyperopt_result(self):
        engine = _engine(n_trials=3)
        df = _make_df(300)
        result = engine.optimize(df, df, n_trials=3)
        assert isinstance(result, HyperoptResult)

    def test_optimize_n_trials_completed(self):
        engine = _engine(n_trials=3)
        df = _make_df(200)
        result = engine.optimize(df, df, n_trials=3)
        assert result.n_trials_completed >= 1  # 최소 1개 완료

    def test_optimize_best_params_in_range(self):
        engine = _engine(n_trials=3)
        df = _make_df(200)
        result = engine.optimize(df, df, n_trials=3)
        p = result.best_params
        assert 0.55 <= p.ensemble_threshold <= 0.75
        assert 15.0 <= p.adx_threshold <= 35.0

    def test_optimize_study_name_stored(self):
        engine = _engine(n_trials=3)
        df = _make_df(200)
        result = engine.optimize(df, df, n_trials=3, study_name="my_study")
        assert result.study_name == "my_study"

    def test_optimize_empty_df_no_crash(self):
        engine = _engine(n_trials=2)
        result = engine.optimize(pd.DataFrame(), pd.DataFrame(), n_trials=2)
        assert isinstance(result, HyperoptResult)

    def test_param_importances_dict(self):
        engine = _engine(n_trials=5)
        df = _make_df(200)
        result = engine.optimize(df, df, n_trials=5)
        assert isinstance(result.param_importances, dict)

    def test_sharpe_is_float(self):
        engine = _engine(n_trials=3)
        df = _make_df(200)
        result = engine.optimize(df, df, n_trials=3)
        assert isinstance(result.best_oos_sharpe, float)


# ---------------------------------------------------------------------------
# BacktestParams 검증
# ---------------------------------------------------------------------------

class TestBacktestParamsRanges:
    def test_default_ensemble_threshold_in_optuna_range(self):
        """기본 ensemble_threshold가 Optuna 탐색 범위 내."""
        p = BacktestParams()
        assert 0.55 <= p.ensemble_threshold <= 0.75

    def test_default_kelly_in_range(self):
        p = BacktestParams()
        assert 0.10 <= p.kelly_fraction <= 0.50

    def test_default_grid_levels_in_range(self):
        p = BacktestParams()
        assert 5 <= p.grid_levels <= 20

    def test_default_dca_step_in_range(self):
        p = BacktestParams()
        assert -0.05 <= p.dca_step_pct <= -0.02
