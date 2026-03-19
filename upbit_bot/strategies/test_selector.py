"""test_selector.py — StrategySelector 단위 테스트."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from schema import FilterResult, StrategyDecision
from strategies.selector import (
    BASE_ALLOCATION,
    ADX_STRONG,
    ADX_NORMAL,
    StrategySelector,
)


def _fr(atr: float = 500.0, mult: float = 1.0) -> FilterResult:
    return FilterResult(
        coin="KRW-BTC",
        timestamp=datetime.now(timezone.utc),
        tradeable=True,
        regime_strategy="TREND_STRONG",
        signal_multiplier=mult,
        adx_value=30.0,
        supertrend_direction=1,
        atr_value=atr,
    )


class TestAdxToStrategy:
    def test_adx_above_strong_is_trend_strong(self):
        sel = StrategySelector()
        d = sel.select_strategy(adx=35.0, fear_greed=50.0)
        assert d.strategy_type == "TREND_STRONG"

    def test_adx_between_normal_and_strong_is_trend_normal(self):
        sel = StrategySelector()
        d = sel.select_strategy(adx=25.0, fear_greed=50.0)
        assert d.strategy_type == "TREND_NORMAL"

    def test_adx_below_normal_fg_high_is_grid(self):
        sel = StrategySelector()
        d = sel.select_strategy(adx=15.0, fear_greed=50.0)
        assert d.strategy_type == "GRID"

    def test_adx_below_normal_fg_low_is_dca(self):
        sel = StrategySelector()
        d = sel.select_strategy(adx=15.0, fear_greed=20.0)
        assert d.strategy_type == "DCA"

    def test_adx_exactly_strong_is_trend_strong(self):
        sel = StrategySelector()
        d = sel.select_strategy(adx=ADX_STRONG, fear_greed=50.0)
        assert d.strategy_type == "TREND_STRONG"

    def test_adx_exactly_normal_is_trend_normal(self):
        sel = StrategySelector()
        d = sel.select_strategy(adx=ADX_NORMAL, fear_greed=50.0)
        assert d.strategy_type == "TREND_NORMAL"


class TestHmmToStrategy:
    def test_hmm0_adx_high_trend_strong(self):
        sel = StrategySelector(phase_c_enabled=True)
        d = sel.select_strategy(adx=35.0, fear_greed=50.0, hmm_regime=0)
        assert d.strategy_type == "TREND_STRONG"

    def test_hmm0_adx_low_trend_normal(self):
        sel = StrategySelector(phase_c_enabled=True)
        d = sel.select_strategy(adx=15.0, fear_greed=50.0, hmm_regime=0)
        assert d.strategy_type == "TREND_NORMAL"

    def test_hmm1_trend_normal(self):
        sel = StrategySelector(phase_c_enabled=True)
        d = sel.select_strategy(adx=25.0, fear_greed=50.0, hmm_regime=1)
        assert d.strategy_type == "TREND_NORMAL"

    def test_hmm2_grid(self):
        sel = StrategySelector(phase_c_enabled=True)
        d = sel.select_strategy(adx=10.0, fear_greed=50.0, hmm_regime=2)
        assert d.strategy_type == "GRID"

    def test_hmm3_fg_low_dca(self):
        sel = StrategySelector(phase_c_enabled=True)
        d = sel.select_strategy(adx=10.0, fear_greed=20.0, hmm_regime=3)
        assert d.strategy_type == "DCA"

    def test_hmm3_fg_high_hold(self):
        sel = StrategySelector(phase_c_enabled=True)
        d = sel.select_strategy(adx=10.0, fear_greed=60.0, hmm_regime=3)
        assert d.strategy_type == "HOLD"

    def test_phase_c_disabled_ignores_hmm(self):
        sel = StrategySelector(phase_c_enabled=False)
        d = sel.select_strategy(adx=35.0, fear_greed=50.0, hmm_regime=3)
        # ADX 기반 → TREND_STRONG (hmm_regime 무시)
        assert d.strategy_type == "TREND_STRONG"


class TestCapitalAllocation:
    def test_base_allocation_applied(self):
        sel = StrategySelector()
        d = sel.select_strategy(adx=35.0, fear_greed=50.0)
        assert d.capital_allocation == pytest.approx(BASE_ALLOCATION["TREND_STRONG"])

    def test_signal_multiplier_reduces_allocation(self):
        sel = StrategySelector()
        d = sel.select_strategy(adx=35.0, fear_greed=50.0, filter_result=_fr(mult=0.8))
        expected = BASE_ALLOCATION["TREND_STRONG"] * 0.8
        assert d.capital_allocation == pytest.approx(expected, rel=1e-4)

    def test_allocation_clamped_0_1(self):
        sel = StrategySelector()
        d = sel.select_strategy(adx=35.0, fear_greed=50.0, filter_result=_fr(mult=2.0))
        assert 0.0 <= d.capital_allocation <= 1.0

    def test_zero_multiplier_zero_allocation(self):
        sel = StrategySelector()
        d = sel.select_strategy(adx=35.0, fear_greed=50.0, filter_result=_fr(mult=0.0))
        assert d.capital_allocation == pytest.approx(0.0)


class TestDynamicWeights:
    def test_decay_monitor_weights_applied(self):
        mock_dm = MagicMock()
        mock_dm.get_weights.return_value = {"TREND_STRONG": 0.8}
        sel = StrategySelector(decay_monitor=mock_dm, phase_c_enabled=True)
        d = sel.select_strategy(adx=35.0, fear_greed=50.0, hmm_regime=0)
        mock_dm.get_weights.assert_called_once()

    def test_all_dormant_returns_hold_allocation(self):
        mock_dm = MagicMock()
        mock_dm.get_weights.return_value = {}  # 모든 전략 DORMANT
        sel = StrategySelector(decay_monitor=mock_dm, phase_c_enabled=True)
        d = sel.select_strategy(adx=35.0, fear_greed=50.0, hmm_regime=0)
        assert d.capital_allocation == pytest.approx(BASE_ALLOCATION["HOLD"])

    def test_decay_monitor_error_falls_back_to_base(self):
        mock_dm = MagicMock()
        mock_dm.get_weights.side_effect = RuntimeError("DB 오류")
        sel = StrategySelector(decay_monitor=mock_dm, phase_c_enabled=True)
        d = sel.select_strategy(adx=35.0, fear_greed=50.0, hmm_regime=0)
        assert d.capital_allocation > 0.0


class TestStrategyDecisionFields:
    def test_returns_strategy_decision(self):
        sel = StrategySelector()
        d = sel.select_strategy(adx=25.0, fear_greed=50.0)
        assert isinstance(d, StrategyDecision)

    def test_coin_propagated(self):
        sel = StrategySelector()
        d = sel.select_strategy(adx=25.0, fear_greed=50.0, coin="KRW-ETH")
        assert d.coin == "KRW-ETH"

    def test_grid_has_params(self):
        sel = StrategySelector()
        d = sel.select_strategy(adx=10.0, fear_greed=50.0, filter_result=_fr(atr=1_000_000))
        assert d.grid_params is not None
        assert "levels" in d.grid_params

    def test_dca_has_params(self):
        sel = StrategySelector()
        d = sel.select_strategy(adx=10.0, fear_greed=20.0)
        assert d.dca_params is not None
        assert "step_pct" in d.dca_params

    def test_trend_no_grid_params(self):
        sel = StrategySelector()
        d = sel.select_strategy(adx=35.0, fear_greed=50.0)
        assert d.grid_params is None

    def test_timestamp_is_utc(self):
        sel = StrategySelector()
        d = sel.select_strategy(adx=25.0, fear_greed=50.0)
        assert d.timestamp.tzinfo is not None
