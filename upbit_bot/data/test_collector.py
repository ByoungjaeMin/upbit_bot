"""data/test_collector.py — collector.py 핵심 함수 단위 테스트.

커버 대상:
  - RateLimiter.acquire()
  - PairlistManager._filter_pairs(), _is_leverage_token(), get_active_pairs()
  - UpbitWebSocketCollector._parse_message(), _build_subscribe()
  - KimchiPremiumCollector.__init__ (usd_krw_initial 파라미터)
  - UpbitDataCollector.set_circuit_breaker(), process_ws_queue() 예외 처리
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from data.collector import (
    KimchiPremiumCollector,
    PairlistManager,
    RateLimiter,
    UpbitWebSocketCollector,
)


# ===========================================================================
# RateLimiter
# ===========================================================================

class TestRateLimiter:
    def test_initial_acquire_no_sleep(self):
        """첫 번째 acquire는 즉시 통과해야 함."""
        rl = RateLimiter(max_per_second=8)
        asyncio.run(rl.acquire())
        assert len(rl._timestamps) == 1

    def test_within_limit_no_sleep(self):
        """한도 내 요청은 sleep 없이 통과."""
        rl = RateLimiter(max_per_second=8)
        for _ in range(8):
            asyncio.run(rl.acquire())
        # 8회 모두 성공적으로 완료되어야 함
        assert len(rl._timestamps) <= 8

    def test_timestamps_expire_after_1s(self):
        """1초 전 타임스탬프는 필터링됨."""
        import time
        rl = RateLimiter(max_per_second=2)
        rl._timestamps = [time.monotonic() - 2.0]  # 2초 전
        asyncio.run(rl.acquire())
        # 오래된 타임스탬프 제거 후 새 것 1개만
        assert len(rl._timestamps) == 1


# ===========================================================================
# PairlistManager
# ===========================================================================

class TestPairlistManagerIsLeverageToken:
    @pytest.mark.parametrize("coin", [
        # 4자 이상 + L/S 접미사 or 명시적 접미사
        "BTCUP", "BTCDOWN", "ETHBULL", "ETHBEAR",
        "BTC3L", "BTC3S",  # len > 3 + L/S 접미사 → True
    ])
    def test_leverage_tokens_detected(self, coin: str):
        assert PairlistManager._is_leverage_token(coin) is True

    @pytest.mark.parametrize("coin", ["BTC", "ETH", "SOL", "XRP"])
    def test_normal_coins_not_leverage(self, coin: str):
        assert PairlistManager._is_leverage_token(coin) is False


class TestPairlistManagerFilterPairs:
    def test_filters_blacklist(self):
        mgr = PairlistManager(blacklist=["KRW-USDT"])
        tickers = [
            {"market": "KRW-BTC", "acc_trade_price_24h": 1_000_000},
            {"market": "KRW-USDT", "acc_trade_price_24h": 500_000},
        ]
        result = mgr._filter_pairs(tickers)
        assert "KRW-USDT" not in result

    def test_filters_leverage_tokens(self):
        mgr = PairlistManager()
        # MIN_VOLUME_KRW = 1억 이상이어야 볼륨 필터 통과
        tickers = [
            {"market": "KRW-BTC", "acc_trade_price_24h": 200_000_000},
            {"market": "KRW-BTC3L", "acc_trade_price_24h": 150_000_000},
        ]
        result = mgr._filter_pairs(tickers)
        assert "KRW-BTC3L" not in result
        assert "KRW-BTC" in result

    def test_top_n_by_volume(self):
        mgr = PairlistManager()
        # MAX_PAIRS(30)보다 많은 코인
        tickers = [
            {"market": f"KRW-COIN{i:02d}", "acc_trade_price_24h": i * 1000}
            for i in range(1, 41)
        ]
        result = mgr._filter_pairs(tickers)
        assert len(result) <= 30

    def test_held_coins_always_included(self):
        mgr = PairlistManager()
        mgr.set_held_coins({"KRW-RARE"})
        # 거래량이 0이어도 보유 코인은 포함
        tickers = [
            {"market": "KRW-RARE", "acc_trade_price_24h": 0},
        ]
        result = mgr._filter_pairs(tickers)
        assert "KRW-RARE" in result

    def test_get_active_pairs_empty_initially(self):
        mgr = PairlistManager()
        assert mgr.get_active_pairs() == []

    def test_is_active_after_set(self):
        mgr = PairlistManager()
        mgr._active_pairs = ["KRW-BTC"]
        assert mgr.is_active("KRW-BTC") is True
        assert mgr.is_active("KRW-ETH") is False


# ===========================================================================
# UpbitWebSocketCollector._parse_message()
# ===========================================================================

class TestUpbitWebSocketCollectorParseMessage:
    def _make_queue(self):
        return asyncio.Queue()

    def test_valid_trade_message(self):
        q = self._make_queue()
        col = UpbitWebSocketCollector(q)
        data = {
            "type": "trade",
            "code": "KRW-BTC",
            "trade_price": 50_000_000.0,
            "trade_volume": 0.001,
            "ask_bid": "BID",
            "sequential_id": 12345,
            "trade_timestamp": 1700000000000,
        }
        result = col._parse_message(data)
        assert result is not None
        assert result.coin == "KRW-BTC"
        assert result.trade_price == 50_000_000.0
        assert result.trade_volume == 0.001
        assert result.ask_bid == "BID"

    def test_non_krw_market_filtered(self):
        q = self._make_queue()
        col = UpbitWebSocketCollector(q)
        data = {
            "type": "trade",
            "code": "BTC-ETH",
            "trade_price": 0.05,
            "trade_volume": 1.0,
            "sequential_id": 1,
            "trade_timestamp": 1700000000000,
        }
        result = col._parse_message(data)
        assert result is None

    def test_malformed_message_returns_none(self):
        q = self._make_queue()
        col = UpbitWebSocketCollector(q)
        # 필수 필드 누락 → 예외 삼킴 후 None 반환
        result = col._parse_message({})
        assert result is None

    def test_build_subscribe_structure(self):
        q = self._make_queue()
        col = UpbitWebSocketCollector(q)
        pairs = ["KRW-BTC", "KRW-ETH"]
        msg = col._build_subscribe(pairs)
        assert isinstance(msg, list)
        assert len(msg) >= 3  # ticket + ticker_type + trade_type
        # 첫 번째는 ticket
        assert "ticket" in msg[0]


# ===========================================================================
# KimchiPremiumCollector — usd_krw_initial 파라미터
# ===========================================================================

class TestKimchiPremiumCollectorInit:
    def test_default_usd_krw(self):
        col = KimchiPremiumCollector()
        assert col._usd_krw == 1350.0

    def test_custom_usd_krw_initial(self):
        col = KimchiPremiumCollector(usd_krw_initial=1400.0)
        assert col._usd_krw == 1400.0

    def test_latest_premium_initial_zero(self):
        col = KimchiPremiumCollector()
        assert col.latest_premium == 0.0


# ===========================================================================
# UpbitDataCollector — set_circuit_breaker + process_ws_queue 예외 처리
# ===========================================================================

class TestUpbitDataCollectorCircuitBreaker:
    """UpbitDataCollector의 CB 주입 및 process_ws_queue 예외 처리."""

    def _make_collector(self):
        """UpbitDataCollector를 DB/네트워크 없이 최소 초기화.

        CandleCache/init_db는 __init__ 내부에서 로컬 import되므로
        data.cache 모듈에서 패치해야 한다.
        """
        with patch("data.cache.CandleCache"), \
             patch("data.cache.init_db"):
            from data.collector import UpbitDataCollector
            col = UpbitDataCollector(
                access_key="test",
                secret_key="test",
                db_path=":memory:",
            )
        return col

    def test_set_circuit_breaker(self):
        col = self._make_collector()
        mock_cb = MagicMock()
        col.set_circuit_breaker(mock_cb)
        assert col._cb is mock_cb

    def test_process_ws_queue_calls_cb_on_error(self):
        """on_trade() 예외 발생 시 CB.record_api_error() 호출 확인."""
        col = self._make_collector()
        mock_cb = MagicMock()
        col.set_circuit_breaker(mock_cb)

        # on_trade가 예외를 던지도록 설정
        col._candle_builder = MagicMock()
        col._candle_builder.on_trade = AsyncMock(side_effect=RuntimeError("테스트 오류"))

        async def _run():
            # 큐에 메시지 1개 넣고 루프를 취소
            from schema import RawMarketData
            msg = MagicMock()
            await col._ws_queue.put(msg)
            task = asyncio.create_task(col.process_ws_queue())
            # 큐 소비 대기 후 취소
            await asyncio.sleep(0.05)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        asyncio.run(_run())
        mock_cb.record_api_error.assert_called_once()

    def test_process_ws_queue_continues_after_error(self):
        """예외 발생 후 루프가 계속 실행됨을 확인."""
        col = self._make_collector()
        col.set_circuit_breaker(MagicMock())

        call_count = 0

        async def _flaky_on_trade(msg):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("첫 번째 오류")
            # 두 번째는 성공

        col._candle_builder = MagicMock()
        col._candle_builder.on_trade = _flaky_on_trade

        async def _run():
            msg = MagicMock()
            await col._ws_queue.put(msg)
            await col._ws_queue.put(msg)
            task = asyncio.create_task(col.process_ws_queue())
            await asyncio.sleep(0.05)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        asyncio.run(_run())
        assert call_count == 2  # 두 번 모두 호출됨
