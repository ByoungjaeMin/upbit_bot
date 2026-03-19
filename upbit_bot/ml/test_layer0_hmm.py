"""test_layer0_hmm.py — HMMRegimeDetector 단위 테스트.

hmmlearn 미설치 환경을 전제로:
  - ADX 폴백 경로 전체 검증
  - RegimeResult 데이터 계약 검증
  - 피처 추출 유틸 검증
  - 체크포인트 경로 로직 검증
  - 학습 실패 시 안전한 폴백 보장
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.layer0_hmm import (
    ADX_NORMAL_TREND,
    ADX_STRONG_TREND,
    CORR_TYPE,
    FEAR_GREED_BEAR,
    HMM_N_ITER,
    MIN_TRAIN_SAMPLES,
    N_REGIMES,
    REGIME_LABELS,
    HMMConfig,
    HMMRegimeDetector,
    RegimeResult,
    _HMM_AVAILABLE,
)


# ─────────────────────────────────────────────────
# 헬퍼
# ─────────────────────────────────────────────────

def _daily_df(n: int = 300) -> pd.DataFrame:
    """학습용 일봉 DataFrame 생성."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2022-01-01", periods=n, freq="D", tz="UTC")
    price = 50_000_000 + np.cumsum(rng.normal(0, 500_000, n))
    price = np.clip(price, 1_000, None)
    vol = np.abs(rng.normal(1e9, 2e8, n))
    return pd.DataFrame({"close": price, "volume": vol}, index=idx)


def _detector() -> HMMRegimeDetector:
    return HMMRegimeDetector()


# ─────────────────────────────────────────────────
# 상수 검증
# ─────────────────────────────────────────────────

class TestConstants:
    def test_n_regimes(self):
        assert N_REGIMES == 4

    def test_regime_labels_cover_all(self):
        assert set(REGIME_LABELS.keys()) == {0, 1, 2, 3}

    def test_regime_labels_not_empty(self):
        for label in REGIME_LABELS.values():
            assert len(label) > 0

    def test_adx_strong_gt_normal(self):
        assert ADX_STRONG_TREND > ADX_NORMAL_TREND

    def test_fear_greed_bear_positive(self):
        assert FEAR_GREED_BEAR > 0

    def test_min_train_samples(self):
        assert MIN_TRAIN_SAMPLES >= 100

    def test_hmm_n_iter_positive(self):
        assert HMM_N_ITER > 0

    def test_corr_type_valid(self):
        assert CORR_TYPE in ("full", "tied", "diag", "spherical")


# ─────────────────────────────────────────────────
# HMMConfig
# ─────────────────────────────────────────────────

class TestHMMConfig:
    def test_default_n_components(self):
        cfg = HMMConfig()
        assert cfg.n_components == N_REGIMES

    def test_default_checkpoint_name(self):
        cfg = HMMConfig()
        assert cfg.checkpoint_name == "hmm_regime.pkl"

    def test_custom_values(self):
        cfg = HMMConfig(n_iter=200, n_components=4)
        assert cfg.n_iter == 200
        assert cfg.n_components == 4

    def test_min_train_samples_field(self):
        cfg = HMMConfig()
        assert cfg.min_train_samples == MIN_TRAIN_SAMPLES


# ─────────────────────────────────────────────────
# RegimeResult
# ─────────────────────────────────────────────────

class TestRegimeResult:
    def _make(self, regime: int = 0, confidence: float = 0.8,
              adx_fallback: bool = False) -> RegimeResult:
        return RegimeResult(
            regime=regime,
            confidence=confidence,
            adx_fallback=adx_fallback,
            regime_label=REGIME_LABELS[regime],
            regime_probs=[0.25] * 4,
        )

    def test_is_bullish_regime_0_1(self):
        assert self._make(0).is_bullish is True
        assert self._make(1).is_bullish is True

    def test_is_bullish_regime_2_3_false(self):
        assert self._make(2).is_bullish is False
        assert self._make(3).is_bullish is False

    def test_is_bearish_regime_2_3(self):
        assert self._make(2).is_bearish is True
        assert self._make(3).is_bearish is True

    def test_is_bearish_regime_0_1_false(self):
        assert self._make(0).is_bearish is False
        assert self._make(1).is_bearish is False

    def test_to_dict_has_required_keys(self):
        d = self._make().to_dict()
        for key in ("regime", "confidence", "adx_fallback", "regime_label"):
            assert key in d

    def test_to_dict_regime_value(self):
        d = self._make(regime=2).to_dict()
        assert d["regime"] == 2

    def test_default_regime_probs_length(self):
        r = RegimeResult(
            regime=1, confidence=0.6, adx_fallback=True,
            regime_label="test",
        )
        assert len(r.regime_probs) == 4


# ─────────────────────────────────────────────────
# HMMRegimeDetector 초기화
# ─────────────────────────────────────────────────

class TestHMMRegimeDetectorInit:
    def test_instantiation(self):
        d = _detector()
        assert isinstance(d, HMMRegimeDetector)

    def test_not_trained_initially(self):
        d = _detector()
        assert d.is_trained is False

    def test_hmm_available_is_bool(self):
        d = _detector()
        assert isinstance(d.hmm_available, bool)

    def test_custom_config(self):
        cfg = HMMConfig(n_components=4, n_iter=50)
        d = HMMRegimeDetector(config=cfg)
        assert d._cfg.n_iter == 50


# ─────────────────────────────────────────────────
# predict_adx_fallback (hmmlearn 불필요, 항상 테스트 가능)
# ─────────────────────────────────────────────────

class TestPredictADXFallback:
    def _d(self) -> HMMRegimeDetector:
        return _detector()

    def test_returns_regime_result(self):
        result = self._d().predict_adx_fallback(adx=35.0)
        assert isinstance(result, RegimeResult)

    def test_adx_fallback_flag_true(self):
        result = self._d().predict_adx_fallback(adx=25.0)
        assert result.adx_fallback is True

    def test_strong_adx_yields_regime_0(self):
        """ADX > 30 → 레짐 0 (강한 상승)."""
        result = self._d().predict_adx_fallback(adx=ADX_STRONG_TREND + 5)
        assert result.regime == 0

    def test_normal_adx_yields_regime_1(self):
        """ADX 20~30 → 레짐 1."""
        adx = (ADX_STRONG_TREND + ADX_NORMAL_TREND) / 2
        result = self._d().predict_adx_fallback(adx=adx)
        assert result.regime == 1

    def test_low_adx_yields_regime_2(self):
        """ADX < 20 → 레짐 2 (횡보)."""
        result = self._d().predict_adx_fallback(adx=ADX_NORMAL_TREND - 5)
        assert result.regime == 2

    def test_fear_greed_below_30_yields_regime_3(self):
        """F&G < 30 → 레짐 3 (하락장)."""
        result = self._d().predict_adx_fallback(adx=35.0, fear_greed=20.0)
        assert result.regime == 3

    def test_confidence_in_range(self):
        for adx in (10.0, 25.0, 35.0):
            result = self._d().predict_adx_fallback(adx=adx)
            assert 0.0 <= result.confidence <= 1.0

    def test_regime_probs_sum_to_1(self):
        result = self._d().predict_adx_fallback(adx=35.0)
        assert sum(result.regime_probs) == pytest.approx(1.0, abs=1e-6)

    def test_regime_probs_length_4(self):
        result = self._d().predict_adx_fallback(adx=25.0)
        assert len(result.regime_probs) == N_REGIMES

    def test_regime_probs_non_negative(self):
        result = self._d().predict_adx_fallback(adx=40.0, fear_greed=80.0)
        assert all(p >= 0.0 for p in result.regime_probs)

    def test_regime_label_matches_regime(self):
        result = self._d().predict_adx_fallback(adx=35.0)
        assert result.regime_label == REGIME_LABELS[result.regime]

    def test_fear_greed_30_boundary(self):
        """F&G 정확히 30 — '< 30' 조건이므로 30은 레짐 3 미해당 → ADX로 결정."""
        # fear_greed == 30.0 → fear_greed < FEAR_GREED_BEAR (30) 조건 불충족
        # adx=35 > ADX_STRONG_TREND → 레짐 0
        result = self._d().predict_adx_fallback(adx=ADX_STRONG_TREND + 5, fear_greed=FEAR_GREED_BEAR)
        assert result.regime == 0

    def test_extreme_adx(self):
        """극단 ADX 값도 안전하게 처리."""
        result = self._d().predict_adx_fallback(adx=100.0)
        assert 0.0 <= result.confidence <= 1.0


# ─────────────────────────────────────────────────
# predict (미학습 → 폴백 동작)
# ─────────────────────────────────────────────────

class TestPredictUntrained:
    def test_untrained_returns_regime_result(self):
        d = _detector()
        result = d.predict(log_return=0.01, realized_vol=0.02, volume_change=0.05)
        assert isinstance(result, RegimeResult)

    def test_untrained_adx_fallback_true(self):
        d = _detector()
        result = d.predict(log_return=0.01, realized_vol=0.02, volume_change=0.05)
        assert result.adx_fallback is True

    def test_untrained_regime_in_valid_range(self):
        d = _detector()
        result = d.predict(log_return=0.0, realized_vol=0.01, volume_change=0.0)
        assert 0 <= result.regime <= 3

    def test_untrained_confidence_positive(self):
        d = _detector()
        result = d.predict(log_return=0.0, realized_vol=0.01, volume_change=0.0)
        assert result.confidence > 0.0


# ─────────────────────────────────────────────────
# train (hmmlearn 있을 때만 실제 학습, 없으면 False 반환 확인)
# ─────────────────────────────────────────────────

class TestTrain:
    def test_train_insufficient_data_returns_false(self):
        """샘플 부족 → False (안전한 폴백)."""
        d = _detector()
        df = _daily_df(n=10)  # MIN_TRAIN_SAMPLES 미만
        result = d.train(df)
        assert result is False

    def test_train_no_close_column_returns_false(self):
        d = _detector()
        df = pd.DataFrame({"open": [1.0, 2.0, 3.0]})
        result = d.train(df)
        assert result is False

    @pytest.mark.skipif(not _HMM_AVAILABLE, reason="hmmlearn 미설치")
    def test_train_success_with_sufficient_data(self):
        d = _detector()
        df = _daily_df(n=MIN_TRAIN_SAMPLES + 50)
        result = d.train(df)
        assert result is True
        assert d.is_trained is True

    @pytest.mark.skipif(not _HMM_AVAILABLE, reason="hmmlearn 미설치")
    def test_train_predict_returns_valid_regime(self):
        d = _detector()
        df = _daily_df(n=MIN_TRAIN_SAMPLES + 50)
        d.train(df)
        result = d.predict(log_return=0.01, realized_vol=0.02, volume_change=0.1)
        assert 0 <= result.regime <= 3
        assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.skipif(not _HMM_AVAILABLE, reason="hmmlearn 미설치")
    def test_train_predict_adx_fallback_false(self):
        d = _detector()
        df = _daily_df(n=MIN_TRAIN_SAMPLES + 50)
        d.train(df)
        result = d.predict(0.01, 0.02, 0.05)
        assert result.adx_fallback is False

    @pytest.mark.skipif(not _HMM_AVAILABLE, reason="hmmlearn 미설치")
    def test_trained_regime_probs_sum_to_1(self):
        d = _detector()
        df = _daily_df(n=MIN_TRAIN_SAMPLES + 50)
        d.train(df)
        result = d.predict(0.01, 0.02, 0.05)
        assert sum(result.regime_probs) == pytest.approx(1.0, abs=1e-4)


# ─────────────────────────────────────────────────
# 체크포인트 save / load
# ─────────────────────────────────────────────────

class TestCheckpoint:
    def test_save_untrained_no_crash(self):
        """미학습 모델 저장 시도 → no-op (예외 없음)."""
        d = _detector()
        with tempfile.TemporaryDirectory() as tmpdir:
            d.save(path=Path(tmpdir) / "hmm_test.pkl")

    def test_load_nonexistent_raises_file_not_found(self):
        d = _detector()
        with pytest.raises(FileNotFoundError):
            d.load(path="/nonexistent/path/hmm.pkl")

    @pytest.mark.skipif(not _HMM_AVAILABLE, reason="hmmlearn 미설치")
    def test_save_load_roundtrip(self):
        d = _detector()
        df = _daily_df(n=MIN_TRAIN_SAMPLES + 50)
        d.train(df)

        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "hmm_test.pkl"
            d.save(path=p)
            assert p.exists()

            d2 = HMMRegimeDetector()
            d2.load(path=p)
            assert d2.is_trained is True

            result = d2.predict(0.01, 0.02, 0.05)
            assert 0 <= result.regime <= 3


# ─────────────────────────────────────────────────
# extract_features 유틸
# ─────────────────────────────────────────────────

class TestExtractFeatures:
    def test_returns_ndarray(self):
        df = _daily_df(50)
        X = HMMRegimeDetector.extract_features(df)
        assert isinstance(X, np.ndarray)

    def test_shape_has_3_columns(self):
        df = _daily_df(50)
        X = HMMRegimeDetector.extract_features(df)
        assert X.shape[1] == 3

    def test_rows_less_than_input_due_to_dropna(self):
        df = _daily_df(50)
        X = HMMRegimeDetector.extract_features(df)
        assert X.shape[0] < 50  # rolling(10) + shift(1) → NaN 제거

    def test_no_nan_in_output(self):
        df = _daily_df(100)
        X = HMMRegimeDetector.extract_features(df)
        assert not np.any(np.isnan(X))

    def test_no_volume_column_no_crash(self):
        df = pd.DataFrame({
            "close": np.linspace(50_000_000, 55_000_000, 50),
        }, index=pd.date_range("2024-01-01", periods=50, freq="D"))
        X = HMMRegimeDetector.extract_features(df)
        assert isinstance(X, np.ndarray)
        assert X.shape[1] == 3

    def test_large_df_sufficient_samples(self):
        df = _daily_df(MIN_TRAIN_SAMPLES + 50)
        X = HMMRegimeDetector.extract_features(df)
        assert len(X) >= MIN_TRAIN_SAMPLES
