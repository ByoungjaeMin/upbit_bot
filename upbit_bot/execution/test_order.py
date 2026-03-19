"""test_order.py — SmartOrderRouter + PartialFillHandler 단위 테스트."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from execution.order import (
    FILL_RATE_ACCEPT,
    FILL_RATE_SWEEP,
    FILL_WAIT_SEC,
    MAX_ORDER_RETRY,
    OrderRequest,
    OrderResult,
    OrderStatus,
    OrderType,
    PartialFillHandler,
    SmartOrderRouter,
    UpbitAPIError,
    UpbitClient,
)


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------

def _client(dry_run: bool = True) -> UpbitClient:
    return UpbitClient(dry_run=dry_run)


def _router(dry_run: bool = True) -> SmartOrderRouter:
    return SmartOrderRouter(_client(dry_run))


def _req(
    coin: str = "BTC",
    side: str = "BUY",
    amount: float = 100_000,
    price: float = 50_000_000,
    emergency: bool = False,
    force_market: bool = False,
) -> OrderRequest:
    return OrderRequest(
        coin=coin,
        side=side,
        krw_amount=amount,
        current_price=price,
        is_emergency=emergency,
        force_market=force_market,
    )


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# UpbitClient DRY_RUN
# ---------------------------------------------------------------------------

class TestUpbitClientDryRun:
    def test_dry_run_flag_true(self):
        c = _client(dry_run=True)
        assert c.dry_run is True

    def test_get_orderbook_returns_dict(self):
        c = _client()
        ob = run(c.get_orderbook("BTC"))
        assert "ask_price" in ob
        assert "spread_pct" in ob

    def test_place_limit_returns_string(self):
        c = _client()
        oid = run(c.place_limit_order("BTC", "BUY", 50_000_000, 0.002))
        assert isinstance(oid, str) and len(oid) > 0

    def test_place_market_returns_string(self):
        c = _client()
        oid = run(c.place_market_order("BTC", "BUY", krw_amount=100_000))
        assert isinstance(oid, str)

    def test_get_order_dry_run_done(self):
        c = _client()
        info = run(c.get_order("fake-id"))
        assert info["state"] == "done"

    def test_cancel_order_dry_run(self):
        c = _client()
        resp = run(c.cancel_order("fake-id"))
        assert resp["state"] == "cancel"

    def test_get_balance_dry_run(self):
        c = _client()
        bal = run(c.get_balance("KRW"))
        assert bal == 10_000_000.0


# ---------------------------------------------------------------------------
# SmartOrderRouter — 지정가/시장가 선택
# ---------------------------------------------------------------------------

class TestSmartOrderRouterOrderTypeSelection:
    def test_should_use_limit_good_conditions(self):
        router = _router()
        ob = {"ask_price": 50_000_000, "bid_price": 49_975_000,
               "ask_size": 1.0, "bid_size": 1.0, "spread_pct": 0.0005}
        assert router._should_use_limit(ob, 100_000, trade_velocity=1.0) is True

    def test_spread_too_high_uses_market(self):
        router = _router()
        ob = {"ask_price": 50_000_000, "bid_price": 49_900_000,
               "ask_size": 1.0, "spread_pct": 0.002}
        assert router._should_use_limit(ob, 100_000, trade_velocity=1.0) is False

    def test_insufficient_liquidity_uses_market(self):
        router = _router()
        ob = {"ask_price": 50_000_000, "ask_size": 0.000001,
               "spread_pct": 0.0005}
        # ask_size 너무 작아 liquidity 불충분
        assert router._should_use_limit(ob, 100_000, trade_velocity=1.0) is False

    def test_high_velocity_uses_market(self):
        router = _router()
        ob = {"ask_price": 50_000_000, "ask_size": 1.0, "spread_pct": 0.0005}
        assert router._should_use_limit(ob, 100_000, trade_velocity=3.0) is False

    def test_empty_orderbook_uses_market(self):
        router = _router()
        assert router._should_use_limit({}, 100_000, 1.0) is False


class TestSmartOrderRouterExecute:
    def test_dry_run_buy_returns_status(self):
        router = _router()
        status = run(router.execute(_req()))
        assert isinstance(status, OrderStatus)
        assert status.coin == "BTC"

    def test_dry_run_sell_returns_status(self):
        router = _router()
        status = run(router.execute(_req(side="SELL")))
        assert isinstance(status, OrderStatus)

    def test_zero_amount_raises(self):
        router = _router()
        with pytest.raises(ValueError):
            run(router.execute(_req(amount=0)))

    def test_emergency_buy_uses_market(self):
        router = _router()
        status = run(router.execute(_req(emergency=True)))
        assert status is not None  # 비상 주문도 상태 반환

    def test_emergency_split_uses_two_orders(self):
        """긴급 손절 split=True → 50%+50% 두 번 실행."""
        router = _router()
        market_calls = []
        original = router._market_order

        async def mock_market(r):
            market_calls.append(r.krw_amount)
            s = OrderStatus(
                order_id="x", coin=r.coin, side=r.side,
                order_type=OrderType.MARKET,
                requested_krw=r.krw_amount, price=50_000_000,
                executed_volume=r.krw_amount / 50_000_000,
                executed_krw=r.krw_amount,
                result=OrderResult.DRY_RUN,
            )
            return s

        router._market_order = mock_market
        run(router._emergency_market(_req(amount=100_000), split=True))
        assert len(market_calls) == 2
        assert all(abs(x - 50_000) < 1 for x in market_calls)

    def test_emergency_no_split_single_order(self):
        router = _router()
        market_calls = []

        async def mock_market(r):
            market_calls.append(r.krw_amount)
            return OrderStatus(
                order_id="x", coin=r.coin, side=r.side,
                order_type=OrderType.MARKET,
                requested_krw=r.krw_amount, price=50_000_000,
                result=OrderResult.DRY_RUN,
            )

        router._market_order = mock_market
        run(router._emergency_market(_req(amount=100_000), split=False))
        assert len(market_calls) == 1


# ---------------------------------------------------------------------------
# PartialFillHandler
# ---------------------------------------------------------------------------

class TestPartialFillHandler:
    def _handler(self) -> PartialFillHandler:
        return PartialFillHandler(_client())

    def _make_info(self, exec_vol: float, remain_vol: float, price: float = 50_000_000) -> dict:
        return {
            "state": "done" if remain_vol == 0 else "wait",
            "executed_volume": str(exec_vol),
            "remaining_volume": str(remain_vol),
            "avg_price": str(price),
            "paid_fee": "25.0",
        }

    def test_fill_rate_above_80_fully_filled(self):
        """체결률 ≥ 80% → FULLY_FILLED."""
        handler = self._handler()
        # get_order를 stub
        exec_v, remain_v = 0.9, 0.1  # 90% 체결
        handler._client.get_order = AsyncMock(
            return_value=self._make_info(exec_v, remain_v)
        )
        handler._client.cancel_order = AsyncMock(return_value={"state": "cancel"})
        status = run(handler.handle("id", "BTC", "BUY", 100_000, 50_000_000))
        assert status.result == OrderResult.FULLY_FILLED

    def test_fill_rate_30_to_80_partially_filled(self):
        """체결률 30~80% → PARTIALLY_FILLED."""
        handler = self._handler()
        exec_v, remain_v = 0.5, 0.5  # 50% 체결
        handler._client.get_order = AsyncMock(
            return_value=self._make_info(exec_v, remain_v)
        )
        handler._client.cancel_order = AsyncMock(return_value={"state": "cancel"})
        handler._client.place_market_order = AsyncMock(return_value="sweep-id")
        status = run(handler.handle("id", "BTC", "BUY", 100_000, 50_000_000))
        assert status.result == OrderResult.PARTIALLY_FILLED

    def test_fill_rate_below_30_cancelled(self):
        """체결률 < 30% → CANCELLED."""
        handler = self._handler()
        exec_v, remain_v = 0.1, 0.9  # 10% 체결
        handler._client.get_order = AsyncMock(
            return_value=self._make_info(exec_v, remain_v)
        )
        handler._client.cancel_order = AsyncMock(return_value={"state": "cancel"})
        status = run(handler.handle("id", "BTC", "BUY", 100_000, 50_000_000))
        assert status.result == OrderResult.CANCELLED

    def test_race_condition_400_already_done(self):
        """DELETE 400 already done → FULLY_FILLED 처리."""
        handler = self._handler()
        exec_v, remain_v = 0.1, 0.9  # 낮은 체결률이지만 취소 시 race condition
        handler._client.get_order = AsyncMock(
            return_value=self._make_info(exec_v, remain_v)
        )
        # cancel_order → 400 already done
        async def mock_cancel(_oid):
            raise UpbitAPIError(400, "already done")

        handler._client.cancel_order = mock_cancel
        # _safe_cancel → True (FULLY_FILLED로 처리)
        result = run(handler._safe_cancel("order-id"))
        assert result is True

    def test_race_condition_other_error_reraises(self):
        """DELETE 500 에러 → 재raise."""
        handler = self._handler()

        async def mock_cancel(_oid):
            raise UpbitAPIError(500, "server error")

        handler._client.cancel_order = mock_cancel
        with pytest.raises(UpbitAPIError):
            run(handler._safe_cancel("order-id"))

    def test_fill_rate_property(self):
        s = OrderStatus(
            order_id="x", coin="BTC", side="BUY",
            order_type=OrderType.LIMIT,
            requested_krw=100_000, price=50_000_000,
            executed_volume=0.7, remaining_volume=0.3,
        )
        assert s.fill_rate == pytest.approx(0.7)

    def test_fill_rate_zero_when_no_volume(self):
        s = OrderStatus(
            order_id="x", coin="BTC", side="BUY",
            order_type=OrderType.LIMIT,
            requested_krw=100_000, price=50_000_000,
        )
        assert s.fill_rate == 0.0


# ---------------------------------------------------------------------------
# cancel_with_race_guard
# ---------------------------------------------------------------------------

class TestCancelWithRaceGuard:
    def test_normal_cancel_returns_cancelled(self):
        router = _router()
        result = run(router.cancel_with_race_guard("fake-id"))
        assert result == OrderResult.CANCELLED

    def test_400_already_done_returns_fully_filled(self):
        router = _router()

        async def mock_cancel(_):
            raise UpbitAPIError(400, "already done")

        router._client.cancel_order = mock_cancel
        result = run(router.cancel_with_race_guard("fake-id"))
        assert result == OrderResult.FULLY_FILLED

    def test_unexpected_error_returns_failed(self):
        router = _router()

        async def mock_cancel(_):
            raise UpbitAPIError(500, "server error")

        router._client.cancel_order = mock_cancel
        result = run(router.cancel_with_race_guard("fake-id"))
        assert result == OrderResult.FAILED


# ---------------------------------------------------------------------------
# UpbitAPIError
# ---------------------------------------------------------------------------

class TestUpbitAPIError:
    def test_code_and_message(self):
        e = UpbitAPIError(400, "already done")
        assert e.code == 400
        assert "already done" in e.message

    def test_str_contains_code(self):
        e = UpbitAPIError(404, "not found")
        assert "404" in str(e)
