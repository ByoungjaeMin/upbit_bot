"""test_decay_monitor.py — StrategyDecayMonitor 단위 테스트."""

from __future__ import annotations

import pytest

from strategies.decay_monitor import (
    DORMANT_SHARPE_THRESHOLD,
    DORMANT_WEEKS,
    REVIVAL_SHARPE_THRESHOLD,
    REVIVAL_WEEKS,
    STRATEGY_TYPES,
    StrategyDecayMonitor,
)


def _rows(strategy: str, pnl_pcts: list[float]) -> list[dict]:
    return [{"strategy_type": strategy, "pnl_pct": p, "pnl": p * 100} for p in pnl_pcts]


class TestUpdateWeeklyStats:
    def test_returns_all_strategy_types(self):
        monitor = StrategyDecayMonitor()
        rows = _rows("TREND_STRONG", [0.02, 0.01, -0.01, 0.03])
        stats = monitor.update_weekly_stats(rows)
        for stype in STRATEGY_TYPES:
            assert stype in stats

    def test_trade_count_correct(self):
        monitor = StrategyDecayMonitor()
        rows = _rows("GRID", [0.01, 0.02, -0.01])
        stats = monitor.update_weekly_stats(rows)
        assert stats["GRID"]["trade_count"] == 3

    def test_sharpe_positive_for_profitable(self):
        monitor = StrategyDecayMonitor()
        rows = _rows("TREND_STRONG", [0.05] * 10)
        stats = monitor.update_weekly_stats(rows)
        # 단일 값 → std=0 → sharpe=0 (예외 처리)
        # 여러 값
        rows2 = _rows("TREND_STRONG", [0.01, 0.02, 0.03, 0.04, 0.05])
        stats2 = monitor.update_weekly_stats(rows2)
        assert stats2["TREND_STRONG"]["rolling_sharpe"] > 0

    def test_win_rate_all_positive(self):
        monitor = StrategyDecayMonitor()
        rows = _rows("DCA", [0.01, 0.02, 0.03])
        stats = monitor.update_weekly_stats(rows)
        assert stats["DCA"]["win_rate"] == pytest.approx(1.0)

    def test_win_rate_all_negative(self):
        monitor = StrategyDecayMonitor()
        rows = _rows("GRID", [-0.01, -0.02, -0.03])
        stats = monitor.update_weekly_stats(rows)
        assert stats["GRID"]["win_rate"] == pytest.approx(0.0)

    def test_sharpe_history_grows(self):
        monitor = StrategyDecayMonitor()
        rows = _rows("TREND_NORMAL", [0.01, 0.02])
        monitor.update_weekly_stats(rows)
        monitor.update_weekly_stats(rows)
        assert len(monitor._sharpe_history["TREND_NORMAL"]) == 2

    def test_sharpe_history_max_8(self):
        monitor = StrategyDecayMonitor()
        rows = _rows("TREND_STRONG", [0.01, 0.02])
        for _ in range(12):
            monitor.update_weekly_stats(rows)
        assert len(monitor._sharpe_history["TREND_STRONG"]) <= 8

    def test_insufficient_data_sharpe_zero(self):
        monitor = StrategyDecayMonitor()
        rows = _rows("TREND_STRONG", [0.01])  # 1개 → std=0
        stats = monitor.update_weekly_stats(rows)
        assert stats["TREND_STRONG"]["rolling_sharpe"] == 0.0


class TestGetWeights:
    def test_no_history_returns_empty(self):
        monitor = StrategyDecayMonitor()
        weights = monitor.get_weights()
        assert weights == {}

    def test_single_strategy_gets_all_weight(self):
        monitor = StrategyDecayMonitor()
        monitor._sharpe_history["TREND_STRONG"] = [1.0, 1.2, 0.9, 1.1]
        weights = monitor.get_weights()
        assert weights["TREND_STRONG"] == pytest.approx(1.0)

    def test_negative_sharpe_excluded(self):
        monitor = StrategyDecayMonitor()
        monitor._sharpe_history["TREND_STRONG"] = [1.0]
        monitor._sharpe_history["GRID"] = [-0.5]
        weights = monitor.get_weights()
        assert weights.get("GRID", 0.0) == pytest.approx(0.0)

    def test_weights_sum_to_1(self):
        monitor = StrategyDecayMonitor()
        for stype in STRATEGY_TYPES:
            monitor._sharpe_history[stype] = [0.5, 0.8, 0.3, 1.0]
        weights = monitor.get_weights()
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-4)

    def test_dormant_strategy_excluded(self):
        monitor = StrategyDecayMonitor()
        monitor._sharpe_history["TREND_STRONG"] = [1.0]
        monitor._sharpe_history["GRID"] = [1.0]
        monitor._dormant_status["GRID"] = True
        weights = monitor.get_weights()
        assert weights.get("GRID", 0.0) == pytest.approx(0.0)


class TestCheckDormant:
    def test_4_weeks_below_threshold_triggers_dormant(self):
        monitor = StrategyDecayMonitor()
        low = DORMANT_SHARPE_THRESHOLD - 0.1
        monitor._sharpe_history["TREND_STRONG"] = [low] * DORMANT_WEEKS
        newly = monitor.check_dormant()
        assert "TREND_STRONG" in newly
        assert monitor._dormant_status["TREND_STRONG"] is True

    def test_3_weeks_below_threshold_no_dormant(self):
        monitor = StrategyDecayMonitor()
        low = DORMANT_SHARPE_THRESHOLD - 0.1
        monitor._sharpe_history["GRID"] = [low] * (DORMANT_WEEKS - 1)
        newly = monitor.check_dormant()
        assert "GRID" not in newly

    def test_already_dormant_not_re_triggered(self):
        monitor = StrategyDecayMonitor()
        low = DORMANT_SHARPE_THRESHOLD - 0.1
        monitor._sharpe_history["DCA"] = [low] * DORMANT_WEEKS
        monitor._dormant_status["DCA"] = True  # 이미 DORMANT
        newly = monitor.check_dormant()
        assert "DCA" not in newly

    def test_one_above_threshold_no_dormant(self):
        monitor = StrategyDecayMonitor()
        low = DORMANT_SHARPE_THRESHOLD - 0.1
        high = DORMANT_SHARPE_THRESHOLD + 0.1
        monitor._sharpe_history["TREND_NORMAL"] = [low, low, low, high]
        newly = monitor.check_dormant()
        assert "TREND_NORMAL" not in newly


class TestCheckRevival:
    def test_2_weeks_above_revival_threshold(self):
        monitor = StrategyDecayMonitor()
        monitor._dormant_status["GRID"] = True
        high = REVIVAL_SHARPE_THRESHOLD + 0.1
        monitor._sharpe_history["GRID"] = [high] * REVIVAL_WEEKS
        candidates = monitor.check_revival()
        assert "GRID" in candidates

    def test_non_dormant_not_revival_candidate(self):
        monitor = StrategyDecayMonitor()
        monitor._dormant_status["TREND_STRONG"] = False
        monitor._sharpe_history["TREND_STRONG"] = [2.0, 2.0]
        candidates = monitor.check_revival()
        assert "TREND_STRONG" not in candidates

    def test_1_week_not_enough(self):
        monitor = StrategyDecayMonitor()
        monitor._dormant_status["DCA"] = True
        monitor._sharpe_history["DCA"] = [REVIVAL_SHARPE_THRESHOLD + 0.1]
        candidates = monitor.check_revival()
        assert "DCA" not in candidates


class TestRevive:
    def test_revive_reactivates(self):
        monitor = StrategyDecayMonitor()
        monitor._dormant_status["GRID"] = True
        monitor.revive("GRID")
        assert monitor._dormant_status["GRID"] is False

    def test_revive_unknown_raises(self):
        monitor = StrategyDecayMonitor()
        with pytest.raises(ValueError):
            monitor.revive("UNKNOWN_STRATEGY")


class TestStatusReport:
    def test_report_contains_all_strategies(self):
        monitor = StrategyDecayMonitor()
        report = monitor.get_status_report()
        for stype in STRATEGY_TYPES:
            assert stype in report

    def test_report_contains_dormant_flag(self):
        monitor = StrategyDecayMonitor()
        monitor._dormant_status["GRID"] = True
        report = monitor.get_status_report()
        assert "DORMANT" in report
