"""
backtest/monte_carlo.py — Monte Carlo 검증기

MonteCarloValidator:
  - 실제 거래 손익 시퀀스를 1,000회 무작위 셔플
  - 각 셔플에서 샤프비율, 최대낙폭, 최종 수익률 계산
  - p-value < 0.05 이어야 "통계적으로 유의한 엣지" 판정
  - 실패 시 실전 투입 차단 플래그 반환
  - 출력: 95% 신뢰구간, p-value, 엣지 신뢰도 점수
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# 상수
# ------------------------------------------------------------------

N_SHUFFLES         = 1_000
P_VALUE_THRESHOLD  = 0.05   # 상위 5% (p < 0.05 이어야 통과)
ANNUALIZE_FACTOR   = np.sqrt(252 * 24 * 12)  # 5분봉 연율화 (1년 = 252일 × 24h × 12)
RISK_FREE_RATE     = 0.035  # 무위험 수익률 3.5% (연)
MIN_TRADES         = 30     # Monte Carlo 최소 거래 수


@dataclass
class MonteCarloResult:
    """Monte Carlo 검증 결과."""
    n_shuffles: int
    actual_sharpe: float
    shuffle_sharpe_mean: float
    shuffle_sharpe_std: float
    sharpe_ci_lower: float          # 95% 신뢰구간 하단
    sharpe_ci_upper: float          # 95% 신뢰구간 상단
    p_value: float                  # 실제 샤프 > 셔플 샤프 비율 (낮을수록 좋음)
    edge_confidence: float          # 1 - p_value (높을수록 좋음)
    actual_max_drawdown: float      # 실제 최대낙폭 (음수)
    actual_final_return: float      # 최종 수익률
    passed: bool                    # p_value < 0.05
    n_trades: int = 0

    def summary(self) -> str:
        status = "PASS ✓" if self.passed else "FAIL ✗"
        return (
            f"[MonteCarlo] {status} | "
            f"실제 샤프={self.actual_sharpe:.3f} | "
            f"셔플 평균={self.shuffle_sharpe_mean:.3f}±{self.shuffle_sharpe_std:.3f} | "
            f"p-value={self.p_value:.4f} | "
            f"엣지신뢰도={self.edge_confidence:.1%} | "
            f"MDD={self.actual_max_drawdown:.1%} | "
            f"95%CI=[{self.sharpe_ci_lower:.3f}, {self.sharpe_ci_upper:.3f}]"
        )


class MonteCarloValidator:
    """Monte Carlo 검증기.

    사용법:
        validator = MonteCarloValidator()
        result = validator.validate(pnl_series)
        if not result.passed:
            raise ValueError("엣지가 통계적으로 유의하지 않음")
    """

    def __init__(
        self,
        n_shuffles: int = N_SHUFFLES,
        p_value_threshold: float = P_VALUE_THRESHOLD,
        risk_free_rate: float = RISK_FREE_RATE,
    ) -> None:
        self._n_shuffles = n_shuffles
        self._p_threshold = p_value_threshold
        self._rf = risk_free_rate
        # 의도적 고정 seed: 검증 재현성 확보 목적.
        # 통계적 유의성은 n=1000 셔플 수로 확보.
        self._rng = np.random.default_rng(42)

    def validate(
        self,
        pnl_pcts: Sequence[float],
        n_shuffles: int | None = None,
    ) -> MonteCarloResult:
        """Monte Carlo 검증 실행.

        Args:
            pnl_pcts: 거래별 손익률 시퀀스 (예: [0.02, -0.01, 0.03, ...])
            n_shuffles: 셔플 횟수 (None → self._n_shuffles)

        Returns:
            MonteCarloResult
        """
        arr = np.array(pnl_pcts, dtype=np.float64)
        n_trades = len(arr)

        if n_trades < MIN_TRADES:
            logger.warning(
                "[MonteCarlo] 거래 수 부족 (%d < %d) — 검증 신뢰도 낮음",
                n_trades, MIN_TRADES,
            )

        n = n_shuffles or self._n_shuffles

        # 실제 지표
        actual_sharpe  = self._compute_sharpe(arr)
        actual_mdd     = self._compute_max_drawdown(arr)
        actual_final   = float(np.prod(1 + arr) - 1)

        # 셔플 분포
        shuffle_sharpes = np.empty(n, dtype=np.float64)
        for i in range(n):
            shuffled = self._rng.permutation(arr)
            shuffle_sharpes[i] = self._compute_sharpe(shuffled)

        # p-value: 셔플 샤프 > 실제 샤프인 비율
        p_value = float(np.mean(shuffle_sharpes >= actual_sharpe))

        shuffle_mean = float(np.mean(shuffle_sharpes))
        shuffle_std  = float(np.std(shuffle_sharpes))
        ci_lower = float(np.percentile(shuffle_sharpes, 2.5))
        ci_upper = float(np.percentile(shuffle_sharpes, 97.5))

        passed = p_value < self._p_threshold

        result = MonteCarloResult(
            n_shuffles=n,
            actual_sharpe=actual_sharpe,
            shuffle_sharpe_mean=shuffle_mean,
            shuffle_sharpe_std=shuffle_std,
            sharpe_ci_lower=ci_lower,
            sharpe_ci_upper=ci_upper,
            p_value=p_value,
            edge_confidence=1.0 - p_value,
            actual_max_drawdown=actual_mdd,
            actual_final_return=actual_final,
            passed=passed,
            n_trades=n_trades,
        )

        if passed:
            logger.info(result.summary())
        else:
            logger.warning(
                "[MonteCarlo] FAIL: 엣지가 통계적으로 랜덤과 구분 불가. "
                "p-value=%.4f (기준 %.2f)", p_value, self._p_threshold
            )

        return result

    def assert_edge_significance(self, pnl_pcts: Sequence[float]) -> MonteCarloResult:
        """유의성 미달 시 ValueError 발생 (실전 투입 차단)."""
        result = self.validate(pnl_pcts)
        if not result.passed:
            raise ValueError(
                f"Monte Carlo 검증 실패: p-value={result.p_value:.4f} >= {self._p_threshold}.\n"
                f"엣지가 통계적으로 랜덤과 구분되지 않음. 실전 투입 불가.\n"
                f"{result.summary()}"
            )
        return result

    # ------------------------------------------------------------------
    # 지표 계산 헬퍼
    # ------------------------------------------------------------------

    def _compute_sharpe(self, returns: np.ndarray) -> float:
        """연율화 샤프비율 (무위험 수익률 차감)."""
        if len(returns) < 2:
            return 0.0
        mean_r = float(np.mean(returns))
        std_r  = float(np.std(returns, ddof=1))
        if std_r < 1e-10:
            return 0.0
        rf_per_period = self._rf / (252 * 24 * 12)   # 5분봉 기준
        excess = mean_r - rf_per_period
        return float(excess / std_r * ANNUALIZE_FACTOR)

    @staticmethod
    def _compute_max_drawdown(returns: np.ndarray) -> float:
        """최대낙폭 (음수로 반환)."""
        if len(returns) == 0:
            return 0.0
        cum = np.cumprod(1 + returns)
        rolling_max = np.maximum.accumulate(cum)
        drawdowns = (cum - rolling_max) / rolling_max
        return float(np.min(drawdowns))

    # ------------------------------------------------------------------
    # 전략별 분리 백테스트 편의 메서드
    # ------------------------------------------------------------------

    def validate_by_regime(
        self,
        pnl_by_regime: dict[str, Sequence[float]],
    ) -> dict[str, MonteCarloResult]:
        """전략별 / 시장 구간별 Monte Carlo 검증.

        Args:
            pnl_by_regime: {'bull': [...], 'bear': [...], 'sideways': [...]}

        Returns:
            dict[str, MonteCarloResult]
        """
        results: dict[str, MonteCarloResult] = {}
        for regime, pnls in pnl_by_regime.items():
            logger.info("[MonteCarlo] 구간 '%s' 검증 중 (n=%d)...", regime, len(pnls))
            results[regime] = self.validate(pnls, n_shuffles=min(self._n_shuffles, 200))
        return results
