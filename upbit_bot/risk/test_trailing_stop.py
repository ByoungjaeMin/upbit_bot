"""test_trailing_stop.py — TrailingStopManager 단위 테스트."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BOT_DIR = Path(__file__).parent.parent
if str(_BOT_DIR) not in sys.path:
    sys.path.insert(0, str(_BOT_DIR))

from risk.trailing_stop import TrailingStopManager


class TestTrailingStopManager:
    def setup_method(self):
        self.ts = TrailingStopManager()

    def test_init_returns_stop_price(self):
        stop = self.ts.init("KRW-BTC", entry_price=50_000_000, atr=750_000)
        assert stop < 50_000_000

    def test_not_triggered_when_price_above_stop(self):
        self.ts.init("KRW-BTC", entry_price=50_000_000, atr=500_000)
        triggered = self.ts.update("KRW-BTC", current_price=51_000_000, atr=500_000)
        assert not triggered

    def test_triggered_when_price_below_stop(self):
        stop = self.ts.init("KRW-BTC", entry_price=50_000_000, atr=500_000)
        triggered = self.ts.update("KRW-BTC", current_price=stop - 1, atr=500_000)
        assert triggered

    def test_stop_rises_with_peak(self):
        self.ts.init("KRW-BTC", entry_price=50_000_000, atr=500_000)
        old_stop = self.ts.get_stop("KRW-BTC")
        # 가격 상승 후 업데이트
        self.ts.update("KRW-BTC", current_price=55_000_000, atr=500_000)
        new_stop = self.ts.get_stop("KRW-BTC")
        assert new_stop > old_stop

    def test_remove_clears_state(self):
        self.ts.init("KRW-BTC", entry_price=50_000_000, atr=500_000)
        self.ts.remove("KRW-BTC")
        assert self.ts.get_stop("KRW-BTC") is None

    def test_update_unknown_coin_returns_false(self):
        result = self.ts.update("KRW-XRP", current_price=1000, atr=10)
        assert result is False

    def test_partial_exit_ratio_when_target_reached(self):
        self.ts.init("KRW-BTC", entry_price=50_000_000, atr=500_000)
        ratio = self.ts.get_partial_ratio("KRW-BTC", current_price=52_600_000)
        assert ratio == pytest.approx(0.5)

    def test_partial_exit_none_before_target(self):
        self.ts.init("KRW-BTC", entry_price=50_000_000, atr=500_000)
        ratio = self.ts.get_partial_ratio("KRW-BTC", current_price=50_500_000)
        assert ratio is None

    def test_partial_exit_none_after_done(self):
        self.ts.init("KRW-BTC", entry_price=50_000_000, atr=500_000)
        self.ts.mark_partial_done("KRW-BTC")
        ratio = self.ts.get_partial_ratio("KRW-BTC", current_price=53_000_000)
        assert ratio is None

    def test_summary_contains_all_coins(self):
        self.ts.init("KRW-BTC", entry_price=50_000_000, atr=500_000)
        self.ts.init("KRW-ETH", entry_price=3_000_000, atr=50_000)
        s = self.ts.summary()
        assert "KRW-BTC" in s
        assert "KRW-ETH" in s

    def test_regime_strong_has_larger_stop_gap(self):
        ts_strong = TrailingStopManager()
        ts_normal = TrailingStopManager()
        stop_strong = ts_strong.init("KRW-BTC", 50_000_000, 500_000, regime="TREND_STRONG")
        stop_normal = ts_normal.init("KRW-BTC", 50_000_000, 500_000, regime="TREND_NORMAL")
        # TREND_STRONG × 3.0 > TREND_NORMAL × 2.5 → stop_strong이 더 낮음
        assert stop_strong < stop_normal

    def test_clear_removes_all(self):
        self.ts.init("KRW-BTC", 50_000_000, 500_000)
        self.ts.init("KRW-ETH", 3_000_000, 50_000)
        self.ts.clear()
        assert self.ts.summary() == {}
