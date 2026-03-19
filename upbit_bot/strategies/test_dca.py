"""test_dca.py — AdaptiveDCAStrategy 단위 테스트."""

from __future__ import annotations

import pytest

from strategies.dca import (
    DEFAULT_MAX_SAFETY,
    DEFAULT_STEP_PCT,
    DEFAULT_TP_PCT,
    DEFAULT_VOLUME_SCALE,
    AdaptiveDCAStrategy,
    DCAOrder,
)


def _dca(capital: float = 1_000_000) -> AdaptiveDCAStrategy:
    return AdaptiveDCAStrategy(capital=capital)


class TestInit:
    def test_base_amount_is_2pct(self):
        dca = _dca(capital=1_000_000)
        assert dca._base_amount == pytest.approx(20_000)

    def test_initial_order_count_zero(self):
        dca = _dca()
        assert dca.order_count == 0

    def test_safety_remaining_max(self):
        dca = _dca()
        assert dca.safety_orders_remaining() == DEFAULT_MAX_SAFETY


class TestAddSafetyOrder:
    def test_first_order_always_added(self):
        dca = _dca()
        order = dca.add_safety_order(current_price=50_000_000)
        assert order is not None
        assert dca.order_count == 1

    def test_first_order_qty_correct(self):
        dca = _dca(capital=1_000_000)
        order = dca.add_safety_order(current_price=50_000_000, fear_greed=50.0)
        # base=20000, fg_mult=0.5 (fg=50 → else branch)
        expected_amount = 20_000 * 0.5
        assert order.krw_amount == pytest.approx(expected_amount)

    def test_second_order_requires_drop(self):
        dca = _dca()
        dca.add_safety_order(50_000_000)
        # 가격 변화 없음 → 조건 불충족
        order = dca.add_safety_order(50_000_000)
        assert order is None

    def test_second_order_after_3pct_drop(self):
        dca = _dca()
        dca.add_safety_order(50_000_000)
        # -3% 이상 하락
        order = dca.add_safety_order(50_000_000 * 0.96)
        assert order is not None

    def test_amount_scales_geometrically(self):
        dca = AdaptiveDCAStrategy(capital=1_000_000, volume_scale=1.5)
        # 첫 매수 (n=0, scale^0=1)
        o1 = dca.add_safety_order(50_000_000, fear_greed=15.0)
        # 두 번째 매수 (n=1, scale^1=1.5)
        o2 = dca.add_safety_order(50_000_000 * 0.96, fear_greed=15.0)
        if o1 and o2:
            assert o2.krw_amount == pytest.approx(o1.krw_amount * 1.5, rel=0.01)

    def test_max_safety_orders_limit(self):
        dca = AdaptiveDCAStrategy(capital=10_000_000, max_safety=3)
        price = 50_000_000
        for _ in range(10):
            dca.add_safety_order(price, fear_greed=10.0)
            price *= 0.96
        # 초기 1 + safety 3 = 최대 4
        assert dca.order_count <= 4

    def test_coin_qty_correct(self):
        dca = _dca(capital=1_000_000)
        price = 100_000
        o = dca.add_safety_order(price, fear_greed=10.0)
        assert o is not None
        assert o.coin_qty == pytest.approx(o.krw_amount / price)


class TestAvgEntryPrice:
    def test_single_order_avg_is_entry(self):
        dca = _dca()
        dca.add_safety_order(50_000_000, fear_greed=10.0)
        assert dca.avg_entry_price() == pytest.approx(50_000_000)

    def test_no_orders_returns_zero(self):
        dca = _dca()
        assert dca.avg_entry_price() == 0.0

    def test_two_orders_weighted_avg(self):
        dca = AdaptiveDCAStrategy(capital=10_000_000)
        # 첫 매수: 1,000원에 1코인 (10% of capital rule 적용 X, 직접 주입)
        from strategies.dca import DCAOrder
        dca._orders.append(DCAOrder(order_n=0, price=1000, krw_amount=1000, coin_qty=1.0))
        dca._orders.append(DCAOrder(order_n=1, price=2000, krw_amount=2000, coin_qty=1.0))
        # avg = (1000+2000) / (1+1) = 1500
        assert dca.avg_entry_price() == pytest.approx(1500.0)


class TestCheckTakeProfit:
    def test_price_above_avg_plus_tp_triggers(self):
        dca = _dca()
        dca.add_safety_order(50_000_000, fear_greed=10.0)
        tp_price = dca.avg_entry_price() * (1 + DEFAULT_TP_PCT) + 1
        assert dca.check_take_profit(tp_price) is True

    def test_price_below_tp_no_trigger(self):
        dca = _dca()
        dca.add_safety_order(50_000_000, fear_greed=10.0)
        assert dca.check_take_profit(50_000_000 * 1.01) is False

    def test_no_orders_no_trigger(self):
        dca = _dca()
        assert dca.check_take_profit(100_000_000) is False


class TestFearGreedMultiplier:
    def test_extreme_fear_15(self):
        assert AdaptiveDCAStrategy.fear_greed_multiplier(10.0) == pytest.approx(1.5)

    def test_fear_range_10(self):
        assert AdaptiveDCAStrategy.fear_greed_multiplier(25.0) == pytest.approx(1.0)

    def test_normal_range_05(self):
        assert AdaptiveDCAStrategy.fear_greed_multiplier(60.0) == pytest.approx(0.5)

    def test_boundary_exactly_15(self):
        # fg=15 → else(>=15 → not <15) → 공포 구간 아님
        assert AdaptiveDCAStrategy.fear_greed_multiplier(15.0) == pytest.approx(1.0)

    def test_boundary_exactly_30(self):
        assert AdaptiveDCAStrategy.fear_greed_multiplier(30.0) == pytest.approx(0.5)


class TestStatusQueries:
    def test_total_invested_grows(self):
        dca = _dca(capital=10_000_000)
        dca.add_safety_order(50_000_000, fear_greed=10.0)
        inv1 = dca.total_invested_krw
        dca.add_safety_order(50_000_000 * 0.96, fear_greed=10.0)
        assert dca.total_invested_krw > inv1

    def test_unrealized_pnl_positive_when_price_up(self):
        dca = _dca()
        dca.add_safety_order(50_000_000, fear_greed=10.0)
        pnl = dca.unrealized_pnl_pct(55_000_000)
        assert pnl > 0

    def test_unrealized_pnl_zero_no_position(self):
        dca = _dca()
        assert dca.unrealized_pnl_pct(50_000_000) == 0.0

    def test_reset_clears_orders(self):
        dca = _dca()
        dca.add_safety_order(50_000_000, fear_greed=10.0)
        dca.reset()
        assert dca.order_count == 0
        assert dca._last_buy_price == 0.0
