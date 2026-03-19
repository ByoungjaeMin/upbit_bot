"""test_helpers.py — utils/helpers.py 단위 테스트."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

_BOT_DIR = Path(__file__).parent.parent
if str(_BOT_DIR) not in sys.path:
    sys.path.insert(0, str(_BOT_DIR))

from utils.helpers import (
    clamp,
    floor_to_5m,
    fmt_krw,
    fmt_pct,
    is_krw_market,
    kst_now,
    pct_change,
    round_price,
    safe_div,
    strip_market_prefix,
    str_to_ts,
    ts_to_str,
    utc_now,
)


class TestTimeUtils:
    def test_utc_now_has_tzinfo(self):
        dt = utc_now()
        assert dt.tzinfo is not None

    def test_kst_now_offset_9h(self):
        from datetime import timedelta
        dt = kst_now()
        assert dt.utcoffset() == timedelta(hours=9)

    def test_floor_to_5m(self):
        dt = datetime(2026, 3, 19, 12, 37, 45, tzinfo=timezone.utc)
        floored = floor_to_5m(dt)
        assert floored.minute == 35
        assert floored.second == 0
        assert floored.microsecond == 0

    def test_ts_to_str_returns_iso(self):
        dt = datetime(2026, 3, 19, 0, 0, 0, tzinfo=timezone.utc)
        s = ts_to_str(dt)
        assert "2026-03-19" in s
        assert "T" in s

    def test_str_to_ts_roundtrip(self):
        dt = datetime(2026, 3, 19, 12, 0, 0, tzinfo=timezone.utc)
        s = ts_to_str(dt)
        dt2 = str_to_ts(s)
        assert dt2.year == 2026
        assert dt2.month == 3


class TestNumericUtils:
    def test_clamp_within_range(self):
        assert clamp(5.0, 0, 10) == 5.0

    def test_clamp_below_lo(self):
        assert clamp(-1.0, 0, 10) == 0.0

    def test_clamp_above_hi(self):
        assert clamp(15.0, 0, 10) == 10.0

    def test_pct_change_positive(self):
        assert pct_change(110, 100) == pytest.approx(10.0)

    def test_pct_change_negative(self):
        assert pct_change(90, 100) == pytest.approx(-10.0)

    def test_pct_change_zero_prev(self):
        assert pct_change(100, 0) == pytest.approx(0.0)

    def test_safe_div_normal(self):
        assert safe_div(10, 4) == pytest.approx(2.5)

    def test_safe_div_zero_denominator(self):
        assert safe_div(10, 0) == pytest.approx(0.0)

    def test_safe_div_custom_default(self):
        assert safe_div(10, 0, default=-1.0) == pytest.approx(-1.0)

    def test_round_price(self):
        assert round_price(50_001_234, 1000) == 50_001_000


class TestCoinUtils:
    def test_strip_market_prefix(self):
        assert strip_market_prefix("KRW-BTC") == "BTC"

    def test_is_krw_market_true(self):
        assert is_krw_market("KRW-ETH") is True

    def test_is_krw_market_false(self):
        assert is_krw_market("BTC-ETH") is False


class TestFormatUtils:
    def test_fmt_krw(self):
        s = fmt_krw(1_234_567)
        assert "1,234,567" in s
        assert "원" in s

    def test_fmt_krw_with_sign(self):
        s = fmt_krw(500, sign=True)
        assert s.startswith("+")

    def test_fmt_pct(self):
        s = fmt_pct(0.1234)
        assert "12.34%" in s

    def test_fmt_pct_with_sign(self):
        s = fmt_pct(0.05, sign=True)
        assert s.startswith("+")
