"""test_grid.py — GridStrategy 단위 테스트."""

from __future__ import annotations

import pytest

from strategies.grid import GRID_ADX_CLOSE_THRESHOLD, GridOrder, GridStrategy


def _grid(
    capital: float = 200_000,
    price: float = 50_000_000,
    atr: float = 1_500_000,
    levels: int = 10,
) -> GridStrategy:
    return GridStrategy(capital=capital, current_price=price, atr=atr, levels=levels)


class TestGridInit:
    def test_range_upper_lower(self):
        g = _grid(price=50_000_000, atr=1_500_000)
        assert g.range_upper == pytest.approx(50_000_000 + 1_500_000 * 3)
        assert g.range_lower == pytest.approx(50_000_000 - 1_500_000 * 3)

    def test_grid_prices_length(self):
        g = _grid(levels=10)
        assert len(g.grid_prices) == 10

    def test_unit_size(self):
        g = _grid(capital=200_000, levels=10)
        assert g.unit_size == pytest.approx(20_000)

    def test_invalid_price_raises(self):
        with pytest.raises(ValueError):
            GridStrategy(capital=100_000, current_price=0, atr=1_000)

    def test_invalid_atr_raises(self):
        with pytest.raises(ValueError):
            GridStrategy(capital=100_000, current_price=50_000_000, atr=0)

    def test_lower_bound_not_negative(self):
        g = GridStrategy(capital=100_000, current_price=100, atr=100)
        assert g.range_lower >= 0.0


class TestPlaceGridOrders:
    def test_returns_10_orders(self):
        g = _grid(levels=10)
        orders = g.place_grid_orders()
        assert len(orders) == 10

    def test_lower_half_buy_upper_half_sell(self):
        g = _grid(levels=10)
        orders = g.place_grid_orders()
        buy_orders = [o for o in orders if o.side == "BUY"]
        sell_orders = [o for o in orders if o.side == "SELL"]
        assert len(buy_orders) == 5
        assert len(sell_orders) == 5

    def test_prices_ascending(self):
        g = _grid()
        orders = g.place_grid_orders()
        prices = [o.price for o in orders]
        assert prices == sorted(prices)

    def test_each_order_size_equals_unit(self):
        g = _grid(capital=200_000, levels=10)
        orders = g.place_grid_orders()
        for o in orders:
            assert o.size_krw == pytest.approx(20_000)

    def test_order_levels_0_to_9(self):
        g = _grid(levels=10)
        orders = g.place_grid_orders()
        levels = sorted([o.level for o in orders])
        assert levels == list(range(10))


class TestOnOrderFilled:
    def test_buy_filled_creates_sell_above(self):
        g = _grid(levels=10)
        g.place_grid_orders()
        reorder = g.on_order_filled(level=2, side="BUY")
        assert reorder is not None
        assert reorder.level == 3
        assert reorder.side == "SELL"

    def test_sell_filled_creates_buy_below(self):
        g = _grid(levels=10)
        g.place_grid_orders()
        reorder = g.on_order_filled(level=7, side="SELL")
        assert reorder is not None
        assert reorder.level == 6
        assert reorder.side == "BUY"

    def test_top_sell_filled_creates_buy_below(self):
        """level=9 SELL 체결 → level=8에 BUY 재주문 (범위 내)."""
        g = _grid(levels=10)
        g.place_grid_orders()
        reorder = g.on_order_filled(level=9, side="SELL")
        assert reorder is not None
        assert reorder.level == 8
        assert reorder.side == "BUY"

    def test_bottom_buy_filled_creates_sell_above(self):
        """level=0 BUY 체결 → level=1에 SELL 재주문 (범위 내)."""
        g = _grid(levels=10)
        g.place_grid_orders()
        reorder = g.on_order_filled(level=0, side="BUY")
        assert reorder is not None
        assert reorder.level == 1
        assert reorder.side == "SELL"

    def test_top_buy_filled_no_reorder(self):
        """level=9 BUY 체결 → level=10은 범위 초과 → None."""
        g = _grid(levels=10)
        g.place_grid_orders()
        reorder = g.on_order_filled(level=9, side="BUY")
        assert reorder is None

    def test_bottom_sell_filled_no_reorder(self):
        """level=0 SELL 체결 → level=-1은 범위 초과 → None."""
        g = _grid(levels=10)
        g.place_grid_orders()
        reorder = g.on_order_filled(level=0, side="SELL")
        assert reorder is None

    def test_filled_order_marked(self):
        g = _grid(levels=10)
        g.place_grid_orders()
        g.on_order_filled(level=3, side="BUY")
        assert g._orders[3].filled is True


class TestShouldClose:
    def test_adx_above_threshold_close(self):
        g = _grid()
        assert g.should_close(adx=30.0) is True

    def test_adx_below_threshold_no_close(self):
        g = _grid()
        assert g.should_close(adx=20.0) is False

    def test_adx_exactly_threshold_close(self):
        g = _grid()
        assert g.should_close(adx=GRID_ADX_CLOSE_THRESHOLD + 0.1) is True


class TestRecalculateRange:
    def test_recalculate_updates_range(self):
        g = _grid(price=50_000_000, atr=1_500_000)
        g.recalculate_range(current_price=55_000_000, atr=2_000_000)
        assert g.range_upper == pytest.approx(55_000_000 + 2_000_000 * 3)

    def test_recalculate_clears_orders(self):
        g = _grid()
        g.place_grid_orders()
        g.recalculate_range(current_price=60_000_000, atr=1_000_000)
        assert g.get_open_orders() == []


class TestStatusQueries:
    def test_get_open_orders_initially_empty(self):
        g = _grid()
        assert g.get_open_orders() == []

    def test_get_open_orders_after_place(self):
        g = _grid(levels=10)
        g.place_grid_orders()
        assert len(g.get_open_orders()) == 10

    def test_filled_count(self):
        g = _grid(levels=10)
        g.place_grid_orders()
        g.on_order_filled(2, "BUY")
        g.on_order_filled(5, "SELL")
        assert g.get_filled_count() == 2
