"""
test_candle_builder.py — CandleBuilder + _TradeBucket 단위 테스트
"""

import asyncio
from datetime import datetime, timezone

import pandas as pd
import pytest

from data.candle_builder import (
    CANDLE_5M_SECONDS,
    CandleBuilder,
    _TradeBucket,
)
from schema import RawMarketData


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------

def _trade(
    coin: str = "KRW-BTC",
    price: float = 90_000_000.0,
    volume: float = 0.001,
    ask_bid: str = "ASK",
    ts: datetime | None = None,
) -> RawMarketData:
    if ts is None:
        ts = datetime(2026, 1, 1, 0, 0, 30, tzinfo=timezone.utc)
    return RawMarketData(
        coin=coin,
        timestamp=ts,
        trade_price=price,
        trade_volume=volume,
        ask_bid=ask_bid,
        sequential_id=1,
        stream_type="trade",
    )


def _floor_5m(ts: datetime) -> datetime:
    from datetime import timezone
    t = ts.timestamp()
    return datetime.fromtimestamp(t - t % CANDLE_5M_SECONDS, tz=timezone.utc)


# ---------------------------------------------------------------------------
# _TradeBucket
# ---------------------------------------------------------------------------

class TestTradeBucket:
    def test_initial_state_empty(self):
        b = _TradeBucket()
        assert b.is_empty()

    def test_update_sets_ohlcv(self):
        b = _TradeBucket()
        t = _trade(price=100.0, volume=1.0, ask_bid="ASK")
        b.update(t)
        assert b.open_price == 100.0
        assert b.close_price == 100.0
        assert b.buy_volume == 1.0
        assert b.sell_volume == 0.0

    def test_tick_imbalance_all_buy(self):
        b = _TradeBucket()
        for _ in range(5):
            b.update(_trade(ask_bid="ASK", volume=1.0))
        assert b.tick_imbalance() == pytest.approx(1.0)

    def test_tick_imbalance_all_sell(self):
        b = _TradeBucket()
        for _ in range(5):
            b.update(_trade(ask_bid="BID", volume=1.0))
        assert b.tick_imbalance() == pytest.approx(-1.0)

    def test_tick_imbalance_balanced(self):
        b = _TradeBucket()
        for _ in range(5):
            b.update(_trade(ask_bid="ASK", volume=1.0))
            b.update(_trade(ask_bid="BID", volume=1.0))
        assert b.tick_imbalance() == pytest.approx(0.0)

    def test_high_low_updated_correctly(self):
        b = _TradeBucket()
        prices = [100.0, 110.0, 95.0, 105.0]
        for p in prices:
            b.update(_trade(price=p))
        assert b.high_price == 110.0
        assert b.low_price == 95.0

    def test_to_ohlcv_structure(self):
        b = _TradeBucket()
        b.update(_trade(price=100.0, volume=2.0))
        b.update(_trade(price=105.0, volume=1.0))
        ohlcv = b.to_ohlcv()
        assert set(ohlcv.keys()) == {"open", "high", "low", "close", "volume"}
        assert ohlcv["open"] == 100.0
        assert ohlcv["close"] == 105.0
        assert ohlcv["volume"] == 3.0


# ---------------------------------------------------------------------------
# CandleBuilder.on_trade → 5분봉 완성
# ---------------------------------------------------------------------------

class TestCandleBuilderOnTrade:
    def test_single_bucket_accumulates(self):
        builder = CandleBuilder()
        ts = datetime(2026, 1, 1, 0, 0, 30, tzinfo=timezone.utc)
        asyncio.run(builder.on_trade(_trade(ts=ts, price=100.0, volume=1.0)))
        # 버킷에 1개 체결 쌓임, 캔들은 아직 미완성
        assert len(builder._candles_5m["KRW-BTC"]) == 0
        assert not builder._buckets["KRW-BTC"].is_empty()

    def test_cross_boundary_finalizes_candle(self):
        """5분 경계를 넘으면 이전 버킷의 캔들이 _candles_5m에 추가되어야 함."""
        builder = CandleBuilder()

        async def run():
            ts1 = datetime(2026, 1, 1, 0, 0, 30, tzinfo=timezone.utc)
            ts2 = datetime(2026, 1, 1, 0, 5, 30, tzinfo=timezone.utc)
            await builder.on_trade(_trade(ts=ts1, price=100.0))
            await builder.on_trade(_trade(ts=ts2, price=105.0))

        asyncio.run(run())
        # 경계를 넘으면 첫 번째 버킷 캔들이 추가됨 (데이터 부족 시에도 저장)
        assert len(builder._candles_5m["KRW-BTC"]) == 1

    def test_ohlcv_values_correct(self):
        builder = CandleBuilder()
        from datetime import timedelta

        async def run():
            base = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
            prices = [100.0, 110.0, 90.0, 105.0]
            for i, p in enumerate(prices):
                ts = base + timedelta(seconds=i * 10)
                await builder.on_trade(_trade(ts=ts, price=p, volume=1.0))
            # 5분 경계 트리거
            ts_next = base + timedelta(minutes=5, seconds=30)
            await builder.on_trade(_trade(ts=ts_next, price=106.0))

        asyncio.run(run())
        assert len(builder._candles_5m["KRW-BTC"]) == 1
        c = builder._candles_5m["KRW-BTC"][0]
        assert c["open"] == 100.0
        assert c["high"] == 110.0
        assert c["low"] == 90.0
        assert c["close"] == 105.0
        assert c["volume"] == pytest.approx(4.0)

    def test_tick_imbalance_in_candle(self):
        builder = CandleBuilder()

        async def run():
            ts = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
            # 모두 매수 체결
            for i in range(4):
                await builder.on_trade(_trade(ts=ts, ask_bid="ASK", volume=1.0))
            # 5분 경계 → 완성
            ts2 = datetime(2026, 1, 1, 0, 5, 1, tzinfo=timezone.utc)
            await builder.on_trade(_trade(ts=ts2))

        asyncio.run(run())
        c = builder._candles_5m["KRW-BTC"][0]
        assert c["tick_imbalance"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 1시간봉 합성 (12개 5분봉)
# ---------------------------------------------------------------------------

class TestOneHourSynthesis:
    def test_12_candles_produce_1h(self):
        """12개 5분봉 완성 → 1시간봉 _candles_1h에 추가."""
        builder = CandleBuilder()
        from datetime import timedelta

        async def run():
            base = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
            # 13개 5분 구간 생성 (12번째 완성 시 1h 합성)
            for i in range(13):
                # 현재 5분 구간 시작 + 1초
                ts = base + timedelta(minutes=i * 5, seconds=1)
                await builder.on_trade(_trade(ts=ts, price=100.0 + i))
                # 다음 5분 구간 진입 → 이전 버킷 완성
                ts_next = base + timedelta(minutes=(i + 1) * 5, seconds=1)
                await builder.on_trade(_trade(ts=ts_next, price=100.0 + i))

        asyncio.run(run())
        # 12개 5분봉 완성 후 1시간봉 1개 생성
        assert len(builder._candles_1h["KRW-BTC"]) >= 1


# ---------------------------------------------------------------------------
# 일봉 shift(1) Lookahead Bias 방지
# ---------------------------------------------------------------------------

class TestDailyUpdate:
    def test_first_row_ema50_is_nan(self):
        """shift(1) 적용 시 첫 행의 ema50은 NaN이어야 함."""
        import numpy as np
        builder = CandleBuilder()
        idx = pd.date_range("2026-01-01", periods=10, freq="D", tz="UTC")
        df = pd.DataFrame({
            "open": [100.0] * 10,
            "high": [105.0] * 10,
            "low": [95.0] * 10,
            "close": [102.0] * 10,
            "volume": [1000.0] * 10,
        }, index=idx)
        df_shifted = builder.daily_update("KRW-BTC", df)
        assert pd.isna(df_shifted["ema50"].iloc[0])

    def test_shifted_ema_lags_by_one(self):
        """ema50[i] (shifted) == ema50[i-1] (원본) — 데이터 충분 구간 검증."""
        import numpy as np
        import pandas_ta as ta_

        builder = CandleBuilder()
        n = 60
        idx = pd.date_range("2026-01-01", periods=n, freq="D", tz="UTC")
        close_vals = np.linspace(100.0, 200.0, n)
        df = pd.DataFrame({
            "open": close_vals - 1,
            "high": close_vals + 2,
            "low": close_vals - 2,
            "close": close_vals,
            "volume": [1000.0] * n,
        }, index=idx)
        df_shifted = builder.daily_update("KRW-BTC", df)

        # 원본 EMA50 계산
        orig_ema = ta_.ema(pd.Series(close_vals, index=idx), length=50)

        # 데이터 충분한 구간(i >= 51)에서 shift(1) 적용 여부 검증
        for i in range(52, min(n, 58)):
            sv = df_shifted["ema50"].iloc[i]
            ov = orig_ema.iloc[i - 1]
            if pd.isna(sv) or pd.isna(ov):
                continue
            assert float(sv) == pytest.approx(float(ov), rel=1e-6)


# ---------------------------------------------------------------------------
# 유틸
# ---------------------------------------------------------------------------

class TestUtils:
    def test_floor_5m(self):
        ts = datetime(2026, 1, 1, 0, 3, 45, tzinfo=timezone.utc)
        floored = CandleBuilder._floor_5m(ts)
        assert floored.minute == 0
        assert floored.second == 0

    def test_floor_5m_boundary(self):
        ts = datetime(2026, 1, 1, 0, 5, 0, tzinfo=timezone.utc)
        floored = CandleBuilder._floor_5m(ts)
        assert floored.minute == 5

    def test_check_silent_drop_false_when_no_data(self):
        builder = CandleBuilder()
        assert builder.check_silent_drop("KRW-BTC") is False

    def test_has_enough_data_false_when_empty(self):
        builder = CandleBuilder()
        assert builder.has_enough_data("KRW-BTC", min_candles=100) is False
