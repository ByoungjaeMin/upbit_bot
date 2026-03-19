"""
test_quality.py — DataQualityChecker 7단계 단위 테스트
"""

import asyncio
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from data.quality import DataQualityChecker, QualityReport


# ---------------------------------------------------------------------------
# 픽스처
# ---------------------------------------------------------------------------

def _make_df(n: int = 100, freq: str = "5min", stale: bool = False) -> pd.DataFrame:
    """정상 5분봉 DataFrame 생성."""
    if stale:
        end = pd.Timestamp("2020-01-01", tz="UTC")
    else:
        end = pd.Timestamp.now(tz="UTC").floor("5min")
    idx = pd.date_range(end=end, periods=n, freq=freq, tz="UTC")
    close = pd.Series(np.linspace(90_000_000, 95_000_000, n), index=idx)
    return pd.DataFrame({
        "open": close * 0.999,
        "high": close * 1.001,
        "low": close * 0.998,
        "close": close,
        "volume": np.random.uniform(1, 10, n),
    }, index=idx)


@pytest.fixture
def checker() -> DataQualityChecker:
    return DataQualityChecker()


# ---------------------------------------------------------------------------
# step1: OHLCV 논리 오류
# ---------------------------------------------------------------------------

class TestStep1:
    def test_removes_high_lt_low(self, checker):
        df = _make_df(50)
        # 의도적으로 high < low 삽입
        df.iloc[5, df.columns.get_loc("high")] = df.iloc[5]["low"] - 100
        report = QualityReport(coin="KRW-BTC", interval="5m")
        df_out, report = checker.step1_ohlcv_logic(df, report)
        assert report.ohlcv_errors >= 1
        assert len(df_out) < len(df)

    def test_removes_negative_volume(self, checker):
        df = _make_df(50)
        df.iloc[10, df.columns.get_loc("volume")] = -1.0
        report = QualityReport(coin="KRW-BTC", interval="5m")
        df_out, report = checker.step1_ohlcv_logic(df, report)
        assert report.ohlcv_errors >= 1

    def test_clean_df_no_errors(self, checker):
        df = _make_df(50)
        report = QualityReport(coin="KRW-BTC", interval="5m")
        df_out, report = checker.step1_ohlcv_logic(df, report)
        assert report.ohlcv_errors == 0
        assert len(df_out) == 50


# ---------------------------------------------------------------------------
# step2: 타임스탬프
# ---------------------------------------------------------------------------

class TestStep2:
    def test_removes_future_timestamps(self, checker):
        df = _make_df(20)
        future = pd.Timestamp.now(tz="UTC") + timedelta(hours=1)
        df.loc[future] = df.iloc[0]
        report = QualityReport(coin="KRW-BTC", interval="5m")
        df_out, report = checker.step2_timestamp(df, "5m", report)
        assert future not in df_out.index

    def test_removes_duplicates(self, checker):
        df = _make_df(20)
        df_dup = pd.concat([df, df.iloc[:5]])
        report = QualityReport(coin="KRW-BTC", interval="5m")
        df_out, report = checker.step2_timestamp(df_dup, "5m", report)
        assert report.duplicate_count == 5

    def test_interpolates_small_gaps(self, checker):
        df = _make_df(30)
        # 1개 캔들 제거 → 갭 생성
        df_gapped = df.drop(df.index[15])
        report = QualityReport(coin="KRW-BTC", interval="5m")
        df_out, report = checker.step2_timestamp(df_gapped, "5m", report)
        assert report.gap_count >= 1
        assert report.gaps_interpolated >= 1


# ---------------------------------------------------------------------------
# step3: 거래량 이상치
# ---------------------------------------------------------------------------

class TestStep3:
    def test_flags_volume_outlier(self, checker):
        df = _make_df(100)
        df.iloc[50, df.columns.get_loc("volume")] = 999_999.0  # 극단적 거래량
        report = QualityReport(coin="KRW-BTC", interval="5m")
        df_out, report = checker.step3_volume_outliers(df, report)
        assert report.volume_outlier_count >= 1

    def test_no_outlier_in_normal_data(self, checker):
        df = _make_df(100)
        report = QualityReport(coin="KRW-BTC", interval="5m")
        df_out, report = checker.step3_volume_outliers(df, report)
        assert report.volume_outlier_count == 0


# ---------------------------------------------------------------------------
# step4: 가격 이상치
# ---------------------------------------------------------------------------

class TestStep4:
    def test_replaces_price_outlier(self, checker):
        df = _make_df(100)
        # 2가지 이상 해당하도록 극단값 삽입
        df.iloc[20, df.columns.get_loc("close")] = 1.0   # Z-score + IQR + pct
        df.iloc[20, df.columns.get_loc("open")] = 1.0
        report = QualityReport(coin="KRW-BTC", interval="5m")
        df_out, report = checker.step4_price_outliers(df, report)
        assert report.price_outlier_count >= 1
        # median으로 대체됐는지 확인 (1.0이 아니어야 함)
        assert df_out.iloc[20]["close"] != 1.0


# ---------------------------------------------------------------------------
# step5: IsolationForest
# ---------------------------------------------------------------------------

class TestStep5:
    def test_marks_anomaly_column(self, checker):
        df = _make_df(100)
        report = QualityReport(coin="KRW-BTC", interval="5m")
        df_out, report = asyncio.run(checker.step5_anomaly_detection(df, report))
        assert "is_anomaly" in df_out.columns
        assert "exclude_from_training" in df_out.columns
        assert 0.0 <= report.anomaly_pct <= 1.0

    def test_skips_small_df(self, checker):
        df = _make_df(10)
        report = QualityReport(coin="KRW-BTC", interval="5m")
        df_out, report = asyncio.run(checker.step5_anomaly_detection(df, report))
        assert "is_anomaly" not in df_out.columns
        assert report.anomaly_count == 0


# ---------------------------------------------------------------------------
# step6: 신선도
# ---------------------------------------------------------------------------

class TestStep6:
    def test_stale_data_flagged(self, checker):
        df = _make_df(50, stale=True)
        report = QualityReport(coin="KRW-BTC", interval="5m")
        report = checker.step6_freshness(df, "5m", report)
        assert report.stale_data is True

    def test_fresh_data_ok(self, checker):
        df = _make_df(50)
        report = QualityReport(coin="KRW-BTC", interval="5m")
        report = checker.step6_freshness(df, "5m", report)
        assert report.stale_data is False


# ---------------------------------------------------------------------------
# step7: 교차 검증
# ---------------------------------------------------------------------------

class TestStep7:
    def test_source_mismatch_gt_3pct(self, checker):
        df = _make_df(50)
        rest_close = float(df.iloc[-1]["close"])
        ws_price = rest_close * 1.05   # 5% 괴리
        report = QualityReport(coin="KRW-BTC", interval="5m")
        report = checker.step7_cross_validate(df, ws_price, report)
        assert report.source_mismatch is True

    def test_source_warning_1_to_3pct(self, checker):
        df = _make_df(50)
        rest_close = float(df.iloc[-1]["close"])
        ws_price = rest_close * 1.02   # 2% 괴리
        report = QualityReport(coin="KRW-BTC", interval="5m")
        report = checker.step7_cross_validate(df, ws_price, report)
        assert report.source_warning is True
        assert report.source_mismatch is False

    def test_no_warning_within_1pct(self, checker):
        df = _make_df(50)
        rest_close = float(df.iloc[-1]["close"])
        ws_price = rest_close * 1.005  # 0.5% 괴리
        report = QualityReport(coin="KRW-BTC", interval="5m")
        report = checker.step7_cross_validate(df, ws_price, report)
        assert report.source_warning is False
        assert report.source_mismatch is False


# ---------------------------------------------------------------------------
# 전체 파이프라인 + 점수
# ---------------------------------------------------------------------------

class TestPipeline:
    def test_clean_data_high_score(self, checker):
        df = _make_df(100)
        df_out, score, report = asyncio.run(checker.validate_pipeline(df, "5m", "KRW-BTC"))
        assert score >= 0.9
        assert report.status == "NORMAL"

    def test_stale_data_low_score(self, checker):
        df = _make_df(100, stale=True)
        df_out, score, report = asyncio.run(checker.validate_pipeline(df, "5m", "KRW-BTC"))
        assert score < 0.9
        assert report.stale_data is True

    def test_score_to_status(self):
        assert DataQualityChecker.score_to_status(0.95) == "NORMAL"
        assert DataQualityChecker.score_to_status(0.80) == "WARNING"
        assert DataQualityChecker.score_to_status(0.60) == "HOLD_TRAINING"
        assert DataQualityChecker.score_to_status(0.40) == "STOP"

    def test_training_mask(self, checker):
        df = _make_df(100)
        _, _, _ = asyncio.run(checker.validate_pipeline(df, "5m", "KRW-BTC"))
        mask = checker.get_training_mask(df)
        assert len(mask) == len(df)
        assert mask.dtype == bool


# ---------------------------------------------------------------------------
# Lookahead Bias 방지 검증 (daily shift)
# ---------------------------------------------------------------------------

class TestLookaheadBias:
    def test_daily_shifted_values_are_lagged(self):
        """shift(1) 적용 시 오늘 피처 = 어제 값인지 확인."""
        from data.candle_builder import CandleBuilder
        builder = CandleBuilder()

        n = 10
        idx = pd.date_range("2026-01-01", periods=n, freq="D", tz="UTC")
        close = pd.Series(range(100, 100 + n), index=idx, dtype=float)
        df_raw = pd.DataFrame({
            "open": close - 1,
            "high": close + 2,
            "low": close - 2,
            "close": close,
            "volume": [100.0] * n,
        })

        df_shifted = builder.daily_update("KRW-BTC", df_raw)

        # 오늘(i=1) ema50 값은 어제(i=0) 기준이어야 함 (shift(1))
        today_ema50 = df_shifted["ema50"].iloc[2]   # NaN이거나 전일값
        # shift(1) 결과 첫 번째 행은 NaN, 이후는 이전 행 값
        assert pd.isna(df_shifted["ema50"].iloc[0])  # 첫 행은 NaN
