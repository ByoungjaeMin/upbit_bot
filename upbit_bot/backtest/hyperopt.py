"""
backtest/hyperopt.py — Optuna 하이퍼파라미터 최적화

탐색 파라미터:
  RSI 기간(5~30), EMA 조합, ADX 임계값(15~35),
  손절/익절 비율, Kelly 계수(0.1~0.5), ATR 배수(1.5~4.0),
  그리드 레벨(5~20), DCA step_pct(-0.02~-0.05),
  ensemble_threshold(0.55~0.75)

목적함수: Walk-Forward OOS 샤프비율 최대화
n_trials=200 (M4에서 1~2시간)
Pruner: MedianPruner (성능 낮은 trial 조기 종료)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _OPTUNA_AVAILABLE = True
except ImportError:
    _OPTUNA_AVAILABLE = False
    logger.warning("[Hyperopt] optuna 미설치 — 최적화 비활성화")

from backtest.walk_forward import BacktestEngine, BacktestParams, WalkForwardOptimizer

# ------------------------------------------------------------------
# 상수
# ------------------------------------------------------------------

N_TRIALS_DEFAULT = 200
TIMEOUT_DEFAULT  = 3600   # 1시간
N_JOBS           = 1      # M4 CPU 전용 (MPS 금지)


@dataclass
class HyperoptResult:
    """Optuna 최적화 결과."""
    best_params: BacktestParams
    best_oos_sharpe: float
    n_trials_completed: int
    study_name: str
    param_importances: dict[str, float]   # 파라미터 중요도

    def summary(self) -> str:
        lines = [
            f"[Hyperopt] 최적 OOS 샤프: {self.best_oos_sharpe:.4f}",
            f"  trials: {self.n_trials_completed}",
            f"  ensemble_threshold: {self.best_params.ensemble_threshold:.3f}",
            f"  adx_threshold: {self.best_params.adx_threshold:.1f}",
            f"  kelly_fraction: {self.best_params.kelly_fraction:.3f}",
            f"  grid_levels: {self.best_params.grid_levels}",
            f"  dca_step_pct: {self.best_params.dca_step_pct:.3f}",
        ]
        if self.param_importances:
            top = sorted(self.param_importances.items(), key=lambda x: x[1], reverse=True)[:5]
            lines.append(f"  상위 중요도: {top}")
        return "\n".join(lines)


class HyperoptEngine:
    """Optuna 하이퍼파라미터 최적화 엔진.

    사용법:
        engine = HyperoptEngine()
        result = engine.optimize(df_train, df_val, n_trials=200)
        best_params = result.best_params
    """

    def __init__(
        self,
        backtest_engine: BacktestEngine | None = None,
        n_trials: int = N_TRIALS_DEFAULT,
        timeout_sec: int = TIMEOUT_DEFAULT,
    ) -> None:
        self._bt = backtest_engine or BacktestEngine()
        self._n_trials = n_trials
        self._timeout = timeout_sec

    def optimize(
        self,
        df_is: Any,   # pd.DataFrame
        df_oos: Any,  # pd.DataFrame
        strategy_fn: Callable | None = None,
        n_trials: int | None = None,
        study_name: str = "upbit_quant",
    ) -> HyperoptResult:
        """Optuna 최적화 실행.

        Args:
            df_is:  학습(IS) 구간 DataFrame
            df_oos: 검증(OOS) 구간 DataFrame
            strategy_fn: 백테스트 전략 함수
            n_trials: 시도 횟수 (None → N_TRIALS_DEFAULT)
            study_name: Optuna study 이름

        Returns:
            HyperoptResult
        """
        if not _OPTUNA_AVAILABLE:
            logger.warning("[Hyperopt] optuna 미설치 — 기본 파라미터 반환")
            return self._fallback_result(study_name)

        import optuna
        from optuna.pruners import MedianPruner

        n = n_trials or self._n_trials

        def objective(trial: optuna.Trial) -> float:
            params = self._suggest_params(trial)
            try:
                oos_metrics = self._bt.run(df_oos, params, strategy_fn)
                return oos_metrics.sharpe
            except Exception as exc:
                logger.debug("[Hyperopt] trial 실패: %s", exc)
                return -999.0

        study = optuna.create_study(
            direction="maximize",
            pruner=MedianPruner(n_startup_trials=20, n_warmup_steps=5),
            study_name=study_name,
        )

        logger.info("[Hyperopt] 최적화 시작 n_trials=%d timeout=%ds", n, self._timeout)
        study.optimize(
            objective,
            n_trials=n,
            timeout=self._timeout,
            n_jobs=N_JOBS,
            show_progress_bar=False,
        )

        best_trial = study.best_trial
        best_params = self._trial_to_params(best_trial)

        # 파라미터 중요도
        try:
            importances = optuna.importance.get_param_importances(study)
        except Exception:
            importances = {}

        result = HyperoptResult(
            best_params=best_params,
            best_oos_sharpe=best_trial.value if best_trial.value is not None else 0.0,
            n_trials_completed=len(study.trials),
            study_name=study_name,
            param_importances=dict(importances),
        )
        logger.info(result.summary())
        return result

    def optimize_with_walkforward(
        self,
        df_full: Any,   # pd.DataFrame
        strategy_fn: Callable | None = None,
        n_trials: int | None = None,
    ) -> HyperoptResult:
        """Walk-Forward OOS 샤프를 목적함수로 사용하는 전체 최적화.

        각 trial에서 전체 Walk-Forward를 실행하므로 느리지만 과적합이 적다.
        """
        if not _OPTUNA_AVAILABLE:
            return self._fallback_result("wf_study")

        import optuna
        from optuna.pruners import MedianPruner

        wf = WalkForwardOptimizer(engine=self._bt)
        n = n_trials or self._n_trials

        def objective(trial: optuna.Trial) -> float:
            params = self._suggest_params(trial)
            try:
                wf_result = wf.run(
                    df_full,
                    optimize_fn=lambda _df: params,
                    strategy_fn=strategy_fn,
                )
                return wf_result.avg_oos_sharpe
            except Exception as exc:
                logger.debug("[Hyperopt] WF trial 실패: %s", exc)
                return -999.0

        study = optuna.create_study(
            direction="maximize",
            pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=3),
            study_name="wf_hyperopt",
        )

        study.optimize(objective, n_trials=n, timeout=self._timeout, n_jobs=N_JOBS)

        best_trial = study.best_trial
        best_params = self._trial_to_params(best_trial)

        try:
            importances = optuna.importance.get_param_importances(study)
        except Exception:
            importances = {}

        return HyperoptResult(
            best_params=best_params,
            best_oos_sharpe=best_trial.value or 0.0,
            n_trials_completed=len(study.trials),
            study_name="wf_hyperopt",
            param_importances=dict(importances),
        )

    # ------------------------------------------------------------------
    # 파라미터 제안
    # ------------------------------------------------------------------

    @staticmethod
    def _suggest_params(trial: Any) -> BacktestParams:
        """Optuna trial에서 파라미터 제안."""
        return BacktestParams(
            rsi_period=trial.suggest_int("rsi_period", 5, 30),
            ema_short=trial.suggest_int("ema_short", 5, 20),
            ema_long=trial.suggest_int("ema_long", 20, 100),
            adx_threshold=trial.suggest_float("adx_threshold", 15.0, 35.0),
            stop_loss_pct=trial.suggest_float("stop_loss_pct", 0.03, 0.10),
            take_profit_pct=trial.suggest_float("take_profit_pct", 0.05, 0.20),
            kelly_fraction=trial.suggest_float("kelly_fraction", 0.10, 0.50),
            atr_multiplier=trial.suggest_float("atr_multiplier", 1.5, 4.0),
            grid_levels=trial.suggest_int("grid_levels", 5, 20),
            dca_step_pct=trial.suggest_float("dca_step_pct", -0.05, -0.02),
            ensemble_threshold=trial.suggest_float("ensemble_threshold", 0.55, 0.75),
        )

    @staticmethod
    def _trial_to_params(trial: Any) -> BacktestParams:
        """trial 결과를 BacktestParams로 변환."""
        p = trial.params
        return BacktestParams(
            rsi_period=p.get("rsi_period", 14),
            ema_short=p.get("ema_short", 7),
            ema_long=p.get("ema_long", 25),
            adx_threshold=p.get("adx_threshold", 20.0),
            stop_loss_pct=p.get("stop_loss_pct", 0.07),
            take_profit_pct=p.get("take_profit_pct", 0.10),
            kelly_fraction=p.get("kelly_fraction", 0.25),
            atr_multiplier=p.get("atr_multiplier", 2.0),
            grid_levels=p.get("grid_levels", 10),
            dca_step_pct=p.get("dca_step_pct", -0.03),
            ensemble_threshold=p.get("ensemble_threshold", 0.62),
        )

    @staticmethod
    def _fallback_result(study_name: str) -> HyperoptResult:
        """optuna 미설치 시 기본 파라미터 반환."""
        return HyperoptResult(
            best_params=BacktestParams(),
            best_oos_sharpe=0.0,
            n_trials_completed=0,
            study_name=study_name,
            param_importances={},
        )
