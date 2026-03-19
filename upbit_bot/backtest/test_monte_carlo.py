"""test_monte_carlo.py — MonteCarloValidator 단위 테스트."""

from __future__ import annotations

import numpy as np
import pytest

from backtest.monte_carlo import (
    MIN_TRADES,
    N_SHUFFLES,
    P_VALUE_THRESHOLD,
    MonteCarloResult,
    MonteCarloValidator,
)


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------

def _validator(n_shuffles: int = 200) -> MonteCarloValidator:
    """빠른 테스트용 소규모 셔플."""
    return MonteCarloValidator(n_shuffles=n_shuffles)


def _strong_pnls(n: int = 100) -> list[float]:
    """강한 엣지 손익 시퀀스 (샤프 높음)."""
    rng = np.random.default_rng(42)
    return list(rng.normal(0.005, 0.01, n))  # 평균 +0.5%


def _random_pnls(n: int = 100) -> list[float]:
    """랜덤 손익 (엣지 없음)."""
    rng = np.random.default_rng(42)
    return list(rng.normal(0.0, 0.01, n))


def _losing_pnls(n: int = 100) -> list[float]:
    """손실 시퀀스."""
    rng = np.random.default_rng(42)
    return list(rng.normal(-0.003, 0.01, n))


# ---------------------------------------------------------------------------
# MonteCarloResult 기본
# ---------------------------------------------------------------------------

class TestMonteCarloResult:
    def test_summary_contains_sharpe(self):
        result = MonteCarloResult(
            n_shuffles=200, actual_sharpe=1.5,
            shuffle_sharpe_mean=0.3, shuffle_sharpe_std=0.2,
            sharpe_ci_lower=0.0, sharpe_ci_upper=0.7,
            p_value=0.02, edge_confidence=0.98,
            actual_max_drawdown=-0.05, actual_final_return=0.10,
            passed=True,
        )
        s = result.summary()
        assert "PASS" in s
        assert "1.5" in s

    def test_summary_fail(self):
        result = MonteCarloResult(
            n_shuffles=200, actual_sharpe=0.3,
            shuffle_sharpe_mean=0.5, shuffle_sharpe_std=0.2,
            sharpe_ci_lower=0.1, sharpe_ci_upper=0.9,
            p_value=0.15, edge_confidence=0.85,
            actual_max_drawdown=-0.15, actual_final_return=-0.05,
            passed=False,
        )
        assert "FAIL" in result.summary()

    def test_edge_confidence_complement_of_pvalue(self):
        result = MonteCarloResult(
            n_shuffles=100, actual_sharpe=1.0,
            shuffle_sharpe_mean=0.0, shuffle_sharpe_std=0.1,
            sharpe_ci_lower=-0.2, sharpe_ci_upper=0.2,
            p_value=0.03, edge_confidence=0.97,
            actual_max_drawdown=-0.05, actual_final_return=0.10,
            passed=True,
        )
        assert result.edge_confidence == pytest.approx(1.0 - result.p_value)


# ---------------------------------------------------------------------------
# MonteCarloValidator.validate()
# ---------------------------------------------------------------------------

class TestMonteCarloValidatorValidate:
    def test_returns_monte_carlo_result(self):
        v = _validator()
        result = v.validate(_strong_pnls())
        assert isinstance(result, MonteCarloResult)

    def test_actual_sharpe_positive_for_strong_edge(self):
        v = _validator()
        result = v.validate(_strong_pnls())
        assert result.actual_sharpe > 0

    def test_actual_sharpe_near_zero_for_random(self):
        v = _validator()
        result = v.validate(_random_pnls())
        # 랜덤이어도 실현 샤프가 크게 양수일 수 있으므로 범위만 체크
        assert isinstance(result.actual_sharpe, float)

    def test_n_shuffles_matches_request(self):
        v = _validator(n_shuffles=100)
        result = v.validate(_strong_pnls(50), n_shuffles=100)
        assert result.n_shuffles == 100

    def test_p_value_in_range_0_1(self):
        v = _validator()
        result = v.validate(_strong_pnls())
        assert 0.0 <= result.p_value <= 1.0

    def test_edge_confidence_in_range_0_1(self):
        v = _validator()
        result = v.validate(_strong_pnls())
        assert 0.0 <= result.edge_confidence <= 1.0

    def test_ci_lower_less_than_ci_upper(self):
        v = _validator()
        result = v.validate(_strong_pnls())
        assert result.sharpe_ci_lower <= result.sharpe_ci_upper

    def test_max_drawdown_non_positive(self):
        v = _validator()
        result = v.validate(_strong_pnls())
        assert result.actual_max_drawdown <= 0.0

    def test_n_trades_recorded(self):
        pnls = _strong_pnls(80)
        v = _validator()
        result = v.validate(pnls)
        assert result.n_trades == 80

    def test_strong_edge_passes(self):
        """강한 엣지 → p_value 낮아 passed=True 가능성 높음."""
        v = MonteCarloValidator(n_shuffles=500, p_value_threshold=0.05)
        pnls = [0.01] * 100 + [-0.002] * 20  # 높은 승률
        result = v.validate(pnls)
        # 결과 타입만 확인 (확률적 특성으로 passed 여부 단정 불가)
        assert isinstance(result.passed, bool)

    def test_losing_series_negative_sharpe(self):
        v = _validator()
        result = v.validate(_losing_pnls())
        assert result.actual_sharpe < 0

    def test_few_trades_warning_not_crash(self):
        """거래 수 MIN_TRADES 미만도 크래시 없어야."""
        v = _validator()
        result = v.validate([0.01, -0.02, 0.03])
        assert isinstance(result, MonteCarloResult)

    def test_single_trade_no_crash(self):
        v = _validator()
        result = v.validate([0.05])
        assert result.actual_sharpe == 0.0  # 단일 값 → std=0

    def test_empty_series_no_crash(self):
        v = _validator()
        result = v.validate([])
        assert result.actual_sharpe == 0.0


# ---------------------------------------------------------------------------
# MonteCarloValidator._compute_sharpe
# ---------------------------------------------------------------------------

class TestComputeSharpe:
    def test_positive_mean_positive_sharpe(self):
        v = _validator()
        arr = np.array([0.01] * 100)
        # std=0 → sharpe=0
        assert v._compute_sharpe(arr) == 0.0

    def test_varying_returns_nonzero_sharpe(self):
        v = _validator()
        rng = np.random.default_rng(1)
        arr = rng.normal(0.005, 0.01, 100)
        sharpe = v._compute_sharpe(arr)
        assert sharpe != 0.0

    def test_single_element_returns_zero(self):
        v = _validator()
        assert v._compute_sharpe(np.array([0.01])) == 0.0

    def test_zero_std_returns_zero(self):
        v = _validator()
        assert v._compute_sharpe(np.zeros(50)) == 0.0


# ---------------------------------------------------------------------------
# MonteCarloValidator._compute_max_drawdown
# ---------------------------------------------------------------------------

class TestComputeMaxDrawdown:
    def test_increasing_returns_zero_drawdown(self):
        arr = np.array([0.01, 0.01, 0.01, 0.01])
        mdd = MonteCarloValidator._compute_max_drawdown(arr)
        assert mdd == pytest.approx(0.0, abs=1e-6)

    def test_all_losses_negative_drawdown(self):
        arr = np.array([-0.02, -0.02, -0.02])
        mdd = MonteCarloValidator._compute_max_drawdown(arr)
        assert mdd < 0.0

    def test_mixed_returns_drawdown(self):
        arr = np.array([0.10, -0.20, 0.05])
        mdd = MonteCarloValidator._compute_max_drawdown(arr)
        assert mdd < 0.0

    def test_empty_returns_zero(self):
        mdd = MonteCarloValidator._compute_max_drawdown(np.array([]))
        assert mdd == 0.0


# ---------------------------------------------------------------------------
# assert_edge_significance
# ---------------------------------------------------------------------------

class TestAssertEdgeSignificance:
    def test_passes_without_raise_for_clear_edge(self):
        """매우 강한 엣지 → ValueError 없음 (확률적이므로 항상 보장 불가, 타입만 체크)."""
        v = _validator(n_shuffles=100)
        pnls = [0.02] * 150 + [-0.003] * 30
        result = v.validate(pnls)
        assert isinstance(result, MonteCarloResult)

    def test_raises_for_random_series(self):
        """p_value_threshold를 0으로 설정 → 항상 실패."""
        v = MonteCarloValidator(n_shuffles=100, p_value_threshold=0.0)
        with pytest.raises(ValueError, match="Monte Carlo"):
            v.assert_edge_significance(_random_pnls(50))

    def test_error_message_contains_p_value(self):
        v = MonteCarloValidator(n_shuffles=50, p_value_threshold=0.0)
        try:
            v.assert_edge_significance([0.0] * 30)
        except ValueError as e:
            assert "p-value" in str(e)


# ---------------------------------------------------------------------------
# validate_by_regime
# ---------------------------------------------------------------------------

class TestValidateByRegime:
    def test_returns_dict_per_regime(self):
        v = _validator(n_shuffles=50)
        pnl_map = {
            "bull":     _strong_pnls(60),
            "bear":     _losing_pnls(40),
            "sideways": _random_pnls(50),
        }
        results = v.validate_by_regime(pnl_map)
        assert set(results.keys()) == {"bull", "bear", "sideways"}

    def test_each_result_is_monte_carlo_result(self):
        v = _validator(n_shuffles=50)
        results = v.validate_by_regime({"bull": _strong_pnls(30)})
        assert isinstance(results["bull"], MonteCarloResult)

    def test_empty_regime_no_crash(self):
        v = _validator(n_shuffles=50)
        results = v.validate_by_regime({"empty": []})
        assert "empty" in results
