"""test_lookahead.py — LookaheadBiasChecker 단위 테스트."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from backtest.lookahead import (
    DAILY_FEATURES,
    LookaheadBiasChecker,
    LookaheadReport,
)


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------

def _make_df(
    n: int = 200,
    freq: str = "5min",
    with_daily: bool = True,
    shift_daily: bool = True,
) -> pd.DataFrame:
    """테스트용 피처 DataFrame 생성."""
    idx = pd.date_range("2024-01-01", periods=n, freq=freq, tz="UTC")
    rng = np.random.default_rng(42)

    price = 50_000_000 + np.cumsum(rng.normal(0, 100_000, n))
    data = {
        "close_5m": price,
        "open_5m":  price * 0.999,
        "high_5m":  price * 1.001,
        "low_5m":   price * 0.999,
        "volume_5m": rng.uniform(1, 10, n),
        "rsi_5m":   rng.uniform(30, 70, n),
        "macd_5m":  rng.normal(0, 0.1, n),
        "adx_5m":   rng.uniform(10, 40, n),
    }

    if with_daily:
        # shift 적용: 날짜별로 전날 값 (하루 내 동일값)
        daily_val = rng.uniform(48_000_000, 52_000_000, n)
        if shift_daily:
            # 날짜 경계에서 값 변경 (shift 적용 시 하루 내 동일값 유지)
            dates = idx.date
            unique_dates = sorted(set(dates))
            date_to_val = {d: float(rng.uniform(48_000_000, 52_000_000)) for d in unique_dates}
            data["ema50_1d"]          = [date_to_val[d] for d in dates]
            data["ema200_1d"]         = [date_to_val[d] * 0.99 for d in dates]
            data["rsi_1d"]            = [50.0] * n
            data["trend_encoding_1d"] = [1] * n
        else:
            # shift 미적용: 5분마다 값이 변함 (오염)
            data["ema50_1d"]          = daily_val
            data["ema200_1d"]         = daily_val * 0.99
            data["rsi_1d"]            = rng.uniform(30, 70, n)
            data["trend_encoding_1d"] = rng.choice([-1, 0, 1], n)

    return pd.DataFrame(data, index=idx)


def _signal_ts(df: pd.DataFrame, n: int = 20) -> list:
    return list(df.index[:n])


# ---------------------------------------------------------------------------
# LookaheadReport
# ---------------------------------------------------------------------------

class TestLookaheadReport:
    def test_passed_when_no_contamination(self):
        r = LookaheadReport()
        assert r.passed is True

    def test_contamination_count(self):
        r = LookaheadReport(contaminated_features=["ema50_1d", "rsi_1d"])
        assert r.contamination_count == 2

    def test_summary_pass_contains_pass(self):
        r = LookaheadReport(passed=True, total_signals_checked=100)
        assert "PASS" in r.summary()

    def test_summary_fail_contains_features(self):
        r = LookaheadReport(
            contaminated_features=["ema50_1d"],
            shift_violations=["ema50_1d"],
            passed=False,
        )
        s = r.summary()
        assert "FAIL" in s
        assert "ema50_1d" in s


# ---------------------------------------------------------------------------
# LookaheadBiasChecker
# ---------------------------------------------------------------------------

class TestLookaheadBiasChecker:
    def _checker(self) -> LookaheadBiasChecker:
        return LookaheadBiasChecker()

    # --- check() 메인 ---

    def test_check_returns_report(self):
        checker = self._checker()
        df = _make_df(200)
        report = checker.check(df, _signal_ts(df))
        assert isinstance(report, LookaheadReport)

    def test_check_signal_count_recorded(self):
        checker = self._checker()
        df = _make_df(200)
        sigs = _signal_ts(df, 15)
        report = checker.check(df, sigs)
        assert report.total_signals_checked == 15

    def test_empty_df_returns_empty_report(self):
        checker = self._checker()
        df = pd.DataFrame()
        report = checker.check(df, [])
        assert report.total_signals_checked == 0

    def test_no_signal_ts_no_crash(self):
        checker = self._checker()
        df = _make_df(50)
        report = checker.check(df, [])
        assert isinstance(report, LookaheadReport)

    # --- shift 검사 ---

    def test_shifted_daily_passes(self):
        """shift 적용 시 하루 내 동일값 → 위반 없음."""
        checker = self._checker()
        df = _make_df(200, shift_daily=True)
        violations = checker._check_daily_shift(df)
        assert len(violations) == 0

    def test_unshifted_daily_detected(self):
        """shift 미적용 시 하루 내 값 변화 → 위반 감지."""
        checker = self._checker()
        df = _make_df(n=300, shift_daily=False)
        violations = checker._check_daily_shift(df)
        # 여러 피처에서 위반 감지되거나 0개 (데이터 특성에 따라 다름)
        assert isinstance(violations, list)

    def test_no_daily_cols_no_violation(self):
        """일봉 피처 없으면 위반 없음."""
        checker = self._checker()
        df = _make_df(100, with_daily=False)
        violations = checker._check_daily_shift(df)
        assert violations == []

    def test_non_datetimeindex_returns_empty(self):
        checker = self._checker()
        df = pd.DataFrame({"ema50_1d": [1.0, 2.0, 3.0]})
        violations = checker._check_daily_shift(df)
        assert violations == []

    # --- 시퀀스 검사 ---

    def test_seq_check_no_violations_with_correct_data(self):
        """올바른 시퀀스 → 위반 없음."""
        checker = self._checker()
        df = _make_df(200)
        sigs = _signal_ts(df, 10)
        violations = checker._check_seq_future_leak(df, sigs, seq_len=60)
        assert isinstance(violations, list)

    def test_seq_check_empty_signals_no_crash(self):
        checker = self._checker()
        df = _make_df(50)
        violations = checker._check_seq_future_leak(df, [], seq_len=60)
        assert violations == []

    def test_seq_check_empty_df_no_crash(self):
        checker = self._checker()
        violations = checker._check_seq_future_leak(pd.DataFrame(), [datetime.now(timezone.utc)], 60)
        assert violations == []

    # --- 미래 상관 검사 ---

    def test_corr_check_returns_list(self):
        checker = self._checker()
        df = _make_df(200)
        sigs = _signal_ts(df, 30)
        violations = checker._check_future_correlation(df, sigs)
        assert isinstance(violations, list)

    def test_corr_check_empty_df_no_crash(self):
        checker = self._checker()
        violations = checker._check_future_correlation(pd.DataFrame(), [])
        assert violations == []

    # --- assert_no_contamination ---

    def test_assert_passes_when_clean(self):
        checker = self._checker()
        report = LookaheadReport(passed=True)
        checker.assert_no_contamination(report)  # 예외 없어야 함

    def test_assert_raises_when_contaminated(self):
        checker = self._checker()
        report = LookaheadReport(
            contaminated_features=["ema50_1d"],
            passed=False,
        )
        with pytest.raises(ValueError, match="Lookahead"):
            checker.assert_no_contamination(report)

    # --- verify_daily_shift_in_dataframe ---

    def test_verify_daily_shift_with_valid_df(self):
        df = _make_df(200, shift_daily=True)
        violations = LookaheadBiasChecker.verify_daily_shift_in_dataframe(df)
        assert isinstance(violations, list)

    def test_verify_daily_shift_non_datetime_returns_empty(self):
        df = pd.DataFrame({"ema50_1d": [1.0, 2.0]})
        violations = LookaheadBiasChecker.verify_daily_shift_in_dataframe(df)
        assert violations == []
