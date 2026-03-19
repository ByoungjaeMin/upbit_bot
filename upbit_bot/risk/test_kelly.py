"""test_kelly.py — KellySizer 단위 테스트."""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime, timezone

import pytest

# 경로 설정
_BOT_DIR = Path(__file__).parent.parent
if str(_BOT_DIR) not in sys.path:
    sys.path.insert(0, str(_BOT_DIR))

from risk.kelly import KellySizer, _atr_group


class TestKellySizer:
    def setup_method(self):
        self.sizer = KellySizer(total_capital=10_000_000)

    def test_basic_compute_returns_risk_budget(self):
        budget = self.sizer.compute(
            coin="KRW-BTC",
            win_rate=0.55,
            profit_loss_ratio=2.0,
        )
        assert budget.coin == "KRW-BTC"
        assert isinstance(budget.final_position_size, float)
        assert 0 <= budget.final_position_size <= 10_000_000

    def test_zero_win_rate_gives_no_position(self):
        budget = self.sizer.compute(
            coin="KRW-BTC",
            win_rate=0.0,
            profit_loss_ratio=2.0,
        )
        assert budget.final_position_size == pytest.approx(0.0)

    def test_kelly_f_positive_when_edge(self):
        budget = self.sizer.compute(
            coin="KRW-ETH",
            win_rate=0.6,
            profit_loss_ratio=2.0,
        )
        assert budget.kelly_f > 0

    def test_max_single_pct_cap(self):
        """최대 30% 하드 캡."""
        budget = self.sizer.compute(
            coin="KRW-BTC",
            win_rate=0.99,  # 극단적 승률
            profit_loss_ratio=10.0,
        )
        assert budget.final_position_size <= 10_000_000 * 0.30

    def test_consecutive_loss_penalty(self):
        budget_no_loss = self.sizer.compute(
            coin="KRW-BTC",
            win_rate=0.55,
            profit_loss_ratio=2.0,
            consecutive_losses=0,
        )
        budget_with_loss = self.sizer.compute(
            coin="KRW-BTC",
            win_rate=0.55,
            profit_loss_ratio=2.0,
            consecutive_losses=3,
        )
        assert budget_with_loss.final_position_size <= budget_no_loss.final_position_size

    def test_var_overlay_with_returns(self):
        returns = [-0.05, -0.03, 0.02, 0.01, -0.02]
        budget = self.sizer.compute(
            coin="KRW-BTC",
            win_rate=0.55,
            profit_loss_ratio=2.0,
            recent_returns=returns,
        )
        assert budget.var_95 > 0

    def test_update_capital(self):
        self.sizer.update_capital(5_000_000)
        budget = self.sizer.compute(
            coin="KRW-BTC",
            win_rate=0.55,
            profit_loss_ratio=2.0,
        )
        # 자본 절반 → 포지션도 절반 수준
        assert budget.final_position_size <= 5_000_000 * 0.30 + 1

    def test_min_position_krw_threshold(self):
        """최소 금액 미달 시 포지션=0."""
        tiny_sizer = KellySizer(total_capital=1_000)  # 극소 자본
        budget = tiny_sizer.compute(
            coin="KRW-BTC",
            win_rate=0.55,
            profit_loss_ratio=1.1,
        )
        assert budget.final_position_size == pytest.approx(0.0)

    def test_atr_group_high(self):
        assert _atr_group(0.04) == "HIGH"

    def test_atr_group_low(self):
        assert _atr_group(0.003) == "LOW"

    def test_atr_group_normal(self):
        assert _atr_group(0.015) == "NORMAL"

    def test_timestamp_utc(self):
        budget = self.sizer.compute("KRW-BTC", 0.55, 2.0)
        assert budget.timestamp.tzinfo is not None
