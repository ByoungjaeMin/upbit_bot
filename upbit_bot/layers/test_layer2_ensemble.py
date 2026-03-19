"""
test_layer2_ensemble.py — Layer2Ensemble 단위 테스트

커버리지:
  - FEATURE_COLS 35개 검증
  - 피처 추출 shape / dtype
  - 콜드스타트 룰 기반 스코어
  - 레짐 그룹 분류
  - 트리 모델 학습 + 예측 (XGB/LGB)
  - 신호 확정 로직 (weighted_avg + consensus_count)
  - 증분 update() 버퍼 누적
  - 체크포인트 save / load
  - LSTM/GRU 모델 구조 (PyTorch 설치 시)
  - Phase B 시퀀스 예측
  - DB 저장 (mock cache)
  - Optuna 탐색 (소규모 n_trials=2)
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from unittest.mock import MagicMock

import numpy as np
import pytest

from layers.layer2_ensemble import (
    FEATURE_COLS,
    INCREMENTAL_WEIGHT,
    MIN_CONSENSUS,
    MODEL_WEIGHTS,
    SEQ_LEN,
    THRESHOLD_DEFAULT,
    Layer2Ensemble,
    _TORCH_AVAILABLE,
    run_optuna_study,
)
from schema import EnsemblePrediction, MarketState

# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------

def _ms(
    coin: str = "KRW-BTC",
    adx_5m: float = 25.0,
    rsi_5m: float = 55.0,
    macd_5m: float = 0.5,
    macd_signal_5m: float = 0.3,
    supertrend_signal: int = 1,
    hmm_regime: int = -1,
    fear_greed: float = 50.0,
) -> MarketState:
    return MarketState(
        coin=coin,
        timestamp=datetime.now(timezone.utc),
        adx_5m=adx_5m,
        rsi_5m=rsi_5m,
        macd_5m=macd_5m,
        macd_signal_5m=macd_signal_5m,
        supertrend_signal=supertrend_signal,
        hmm_regime=hmm_regime,
        fear_greed=fear_greed,
        # 5m OHLCV
        open_5m=100.0, high_5m=105.0, low_5m=98.0,
        close_5m=103.0, volume_5m=10.0,
        volume_ratio_5m=1.2,
        # 1h
        rsi_1h=55.0, ema20_1h=102.0, ema50_1h=100.0,
        macd_1h=0.3, trend_dir_1h=1,
        # 1d (shift(1) 적용값)
        ema50_1d=101.0, ema200_1d=98.0, rsi_1d=55.0, trend_encoding_1d=1,
        # market
        btc_dominance=45.0, altcoin_season=60.0,
        # microstructure
        tick_imbalance=0.3, obi=0.1, trade_velocity=1.2,
        adx_1h=22.0,
    )


def _make_synthetic(n: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """합성 학습 데이터 (N, 35) + 레이블."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n, len(FEATURE_COLS))).astype(np.float32)
    y = rng.integers(0, 2, n).astype(np.int32)
    return X, y


# ---------------------------------------------------------------------------
# FEATURE_COLS 검증
# ---------------------------------------------------------------------------

class TestFeatureCols:
    def test_length_35(self):
        assert len(FEATURE_COLS) == 35

    def test_no_duplicates(self):
        assert len(FEATURE_COLS) == len(set(FEATURE_COLS))

    def test_all_in_market_state(self):
        ms = _ms()
        missing = [c for c in FEATURE_COLS if not hasattr(ms, c)]
        assert missing == [], f"MarketState에 없는 피처: {missing}"


# ---------------------------------------------------------------------------
# 피처 추출
# ---------------------------------------------------------------------------

class TestExtractFeatures:
    def test_shape(self):
        ms = _ms()
        X = Layer2Ensemble._extract_features(ms)
        assert X.shape == (35,)

    def test_dtype_float32(self):
        ms = _ms()
        X = Layer2Ensemble._extract_features(ms)
        assert X.dtype == np.float32

    def test_no_nan(self):
        ms = _ms()
        X = Layer2Ensemble._extract_features(ms)
        assert not np.any(np.isnan(X))

    def test_values_match(self):
        ms = _ms(adx_5m=30.0, rsi_5m=60.0)
        X = Layer2Ensemble._extract_features(ms)
        adx_idx = FEATURE_COLS.index("adx_5m")
        rsi_idx = FEATURE_COLS.index("rsi_5m")
        assert X[adx_idx] == pytest.approx(30.0)
        assert X[rsi_idx] == pytest.approx(60.0)


# ---------------------------------------------------------------------------
# 룰 기반 스코어 (콜드스타트)
# ---------------------------------------------------------------------------

class TestRuleBasedScore:
    def test_range_0_to_1(self):
        ms = _ms()
        score = Layer2Ensemble._rule_based_score(ms)
        assert 0.0 <= score <= 1.0

    def test_bullish_higher_than_bearish(self):
        bull = _ms(rsi_5m=65.0, macd_5m=1.0, macd_signal_5m=0.0,
                   supertrend_signal=1, adx_5m=40.0)
        bear = _ms(rsi_5m=35.0, macd_5m=-1.0, macd_signal_5m=0.0,
                   supertrend_signal=-1, adx_5m=10.0)
        assert Layer2Ensemble._rule_based_score(bull) > Layer2Ensemble._rule_based_score(bear)

    def test_rsi_30_macd_neg_supertrend_neg_gives_low(self):
        ms = _ms(rsi_5m=30.0, macd_5m=-1.0, macd_signal_5m=0.0,
                 supertrend_signal=-1, adx_5m=0.0)
        # RSI 30→0점, MACD neg→0점, ST neg→0점, ADX 0→0점
        assert Layer2Ensemble._rule_based_score(ms) == pytest.approx(0.0)

    def test_all_bullish_max_score_capped_at_1(self):
        ms = _ms(rsi_5m=70.0, macd_5m=1.0, macd_signal_5m=0.0,
                 supertrend_signal=1, adx_5m=50.0)
        score = Layer2Ensemble._rule_based_score(ms)
        assert score <= 1.0


# ---------------------------------------------------------------------------
# 레짐 그룹
# ---------------------------------------------------------------------------

class TestGetRegimeGroup:
    def test_hmm_0_1_aggressive(self):
        ms0 = _ms(hmm_regime=0)
        ms1 = _ms(hmm_regime=1)
        assert Layer2Ensemble._get_regime_group(ms0) == "aggressive"
        assert Layer2Ensemble._get_regime_group(ms1) == "aggressive"

    def test_hmm_2_3_conservative(self):
        ms2 = _ms(hmm_regime=2)
        ms3 = _ms(hmm_regime=3)
        assert Layer2Ensemble._get_regime_group(ms2) == "conservative"
        assert Layer2Ensemble._get_regime_group(ms3) == "conservative"

    def test_no_hmm_adx_high_aggressive(self):
        ms = _ms(hmm_regime=-1, adx_5m=25.0)
        assert Layer2Ensemble._get_regime_group(ms) == "aggressive"

    def test_no_hmm_adx_low_conservative(self):
        ms = _ms(hmm_regime=-1, adx_5m=10.0)
        assert Layer2Ensemble._get_regime_group(ms) == "conservative"

    def test_adx_exactly_20_aggressive(self):
        ms = _ms(hmm_regime=-1, adx_5m=20.0)
        assert Layer2Ensemble._get_regime_group(ms) == "aggressive"


# ---------------------------------------------------------------------------
# 콜드스타트 경로 (<200건)
# ---------------------------------------------------------------------------

class TestColdStart:
    def test_cold_start_returns_rule_key(self):
        ens = Layer2Ensemble(trade_count_fn=lambda: 0)
        pred = ens.predict(_ms())
        assert "rule" in pred.per_model_probs

    def test_cold_start_never_signal_confirmed(self):
        """rule 1개 → consensus < 3 → signal_confirmed=False."""
        ens = Layer2Ensemble(trade_count_fn=lambda: 199)
        for _ in range(20):
            pred = ens.predict(_ms(rsi_5m=70.0, macd_5m=1.0, macd_signal_5m=0.0,
                                   supertrend_signal=1, adx_5m=50.0))
            assert not pred.signal_confirmed

    def test_cold_start_weighted_avg_in_range(self):
        ens = Layer2Ensemble(trade_count_fn=lambda: 50)
        pred = ens.predict(_ms())
        assert 0.0 <= pred.weighted_avg <= 1.0

    def test_cold_start_threshold_recorded(self):
        ens = Layer2Ensemble(trade_count_fn=lambda: 0, threshold=0.65)
        pred = ens.predict(_ms())
        assert pred.threshold_used == pytest.approx(0.65)

    def test_cold_start_coin_recorded(self):
        ens = Layer2Ensemble(trade_count_fn=lambda: 0)
        pred = ens.predict(_ms(coin="KRW-ETH"))
        assert pred.coin == "KRW-ETH"


# ---------------------------------------------------------------------------
# 트리 모델 학습 + 예측 (XGB/LGB)
# ---------------------------------------------------------------------------

pytest.importorskip("xgboost", reason="xgboost 미설치")
pytest.importorskip("lightgbm", reason="lightgbm 미설치")


class TestTreeModelTrainPredict:
    def test_train_batch_sets_models(self):
        ens = Layer2Ensemble()
        X, y = _make_synthetic(200)
        ens.train_batch(X, y, regime_group="aggressive")
        assert ens.is_trained("aggressive")

    def test_predict_after_train_has_xgb_lgb(self):
        ens = Layer2Ensemble(trade_count_fn=lambda: 300)
        X, y = _make_synthetic(200)
        ens.train_batch(X, y, regime_group="aggressive")
        pred = ens.predict(_ms())
        assert "xgb" in pred.per_model_probs
        assert "lgb" in pred.per_model_probs

    def test_prob_in_0_1_range(self):
        ens = Layer2Ensemble(trade_count_fn=lambda: 300)
        X, y = _make_synthetic(200)
        ens.train_batch(X, y)
        pred = ens.predict(_ms())
        for name, prob in pred.per_model_probs.items():
            assert 0.0 <= prob <= 1.0, f"{name} prob={prob}"

    def test_weighted_avg_in_range(self):
        ens = Layer2Ensemble(trade_count_fn=lambda: 300)
        X, y = _make_synthetic(200)
        ens.train_batch(X, y)
        pred = ens.predict(_ms())
        assert 0.0 <= pred.weighted_avg <= 1.0

    def test_signal_confirmed_requires_consensus_3(self):
        """signal_confirmed=True이면 consensus_count >= 3."""
        ens = Layer2Ensemble(trade_count_fn=lambda: 300, threshold=0.0)
        X, y = _make_synthetic(200)
        # 두 레짐 모두 학습 → xgb + lgb × 2 = 4개 모델 가능하지만
        # Phase A는 xgb + lgb 2개뿐 → consensus_count <= 2 → signal_confirmed=False
        ens.train_batch(X, y, regime_group="aggressive")
        pred = ens.predict(_ms())
        # 2개 모델만 있으면 consensus < 3 → False
        if pred.consensus_count < MIN_CONSENSUS:
            assert not pred.signal_confirmed

    def test_conservative_fallback_to_aggressive(self):
        """conservative 미학습 시 aggressive 모델로 폴백."""
        ens = Layer2Ensemble(trade_count_fn=lambda: 300)
        X, y = _make_synthetic(200)
        ens.train_batch(X, y, regime_group="aggressive")
        ms = _ms(hmm_regime=2)  # conservative 레짐
        pred = ens.predict(ms)
        # aggressive 모델로 폴백하여 예측 가능해야 함
        assert "xgb" in pred.per_model_probs or "lgb" in pred.per_model_probs

    def test_scaler_fitted_after_train(self):
        ens = Layer2Ensemble()
        X, y = _make_synthetic(100)
        ens.train_batch(X, y)
        assert ens._scaler_fitted

    def test_small_sample_skipped(self):
        """학습 샘플 < 10개 시 모델 미생성."""
        ens = Layer2Ensemble()
        X, y = _make_synthetic(5)
        ens.train_batch(X, y)
        assert not ens.is_trained("aggressive")


# ---------------------------------------------------------------------------
# 신호 확정 로직
# ---------------------------------------------------------------------------

class TestSignalConfirmed:
    def test_signal_confirmed_true_when_4_models_above_threshold(self):
        """4개 모델 모두 threshold 초과 시 signal_confirmed=True."""
        ens = Layer2Ensemble(threshold=0.5)
        per_model = {"xgb": 0.8, "lgb": 0.75, "lstm": 0.7, "gru": 0.65}
        pred = ens._build_prediction(per_model, _ms())
        assert pred.signal_confirmed
        assert pred.consensus_count == 4

    def test_signal_confirmed_false_when_2_models(self):
        """2개 모델만 있으면 consensus < 3."""
        ens = Layer2Ensemble(threshold=0.5)
        per_model = {"xgb": 0.9, "lgb": 0.9}
        pred = ens._build_prediction(per_model, _ms())
        assert not pred.signal_confirmed

    def test_signal_confirmed_false_when_avg_below_threshold(self):
        """평균이 threshold 미만이면 False."""
        ens = Layer2Ensemble(threshold=0.62)
        per_model = {"xgb": 0.5, "lgb": 0.5, "lstm": 0.5, "gru": 0.5}
        pred = ens._build_prediction(per_model, _ms())
        assert not pred.signal_confirmed

    def test_empty_per_model(self):
        ens = Layer2Ensemble()
        pred = ens._build_prediction({}, _ms())
        assert not pred.signal_confirmed
        assert pred.weighted_avg == 0.0

    def test_weighted_avg_formula(self):
        """가중 평균 수식 검증: xgb(w=1.0)=0.8, lgb(w=1.0)=0.6 → (0.8+0.6)/2=0.7."""
        ens = Layer2Ensemble()
        per_model = {"xgb": 0.8, "lgb": 0.6}
        pred = ens._build_prediction(per_model, _ms())
        assert pred.weighted_avg == pytest.approx(0.7, abs=1e-4)

    def test_rule_weight_06(self):
        """rule 모델 가중치 0.6 적용 확인."""
        assert MODEL_WEIGHTS["rule"] == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# 증분 업데이트
# ---------------------------------------------------------------------------

class TestIncrementalUpdate:
    def test_update_accumulates_buffer(self):
        ens = Layer2Ensemble()
        for _ in range(10):
            ens.update(_ms(), label=1)
        assert len(ens._X_buf) == 10
        assert len(ens._y_buf) == 10
        assert len(ens._w_buf) == 10

    def test_update_weight_multiplier_stored(self):
        ens = Layer2Ensemble()
        ens.update(_ms(), label=1, sample_weight_multiplier=2.0)
        assert list(ens._w_buf)[-1] == pytest.approx(2.0)

    def test_default_weight_is_incremental_weight(self):
        ens = Layer2Ensemble()
        ens.update(_ms(), label=0)
        assert list(ens._w_buf)[-1] == pytest.approx(INCREMENTAL_WEIGHT)

    def test_retrain_triggered_at_interval(self):
        """retrain_interval=5, 5번 업데이트 후 재학습 카운터 초기화."""
        ens = Layer2Ensemble(retrain_interval=5)
        for _ in range(5):
            ens.update(_ms(), label=1)
        # 재학습 후 카운터는 0
        assert ens._new_since_retrain == 0

    def test_buffer_maxlen_5000(self):
        ens = Layer2Ensemble()
        assert ens._X_buf.maxlen == 5000


# ---------------------------------------------------------------------------
# 체크포인트 저장 / 로딩
# ---------------------------------------------------------------------------

class TestCheckpoint:
    def test_save_load_roundtrip(self):
        ens = Layer2Ensemble(threshold=0.65)
        X, y = _make_synthetic(100)
        ens.train_batch(X, y)

        with tempfile.TemporaryDirectory() as tmp:
            ens.save_checkpoint(tmp)
            ens2 = Layer2Ensemble()
            ens2.load_checkpoint(tmp)
            assert ens2._scaler_fitted
            assert ens2._threshold == pytest.approx(0.65)
            assert ens2.is_trained("aggressive")

    def test_load_nonexistent_path_is_noop(self):
        ens = Layer2Ensemble()
        ens.load_checkpoint("/nonexistent/path")  # 예외 없어야 함
        assert not ens._scaler_fitted

    def test_save_creates_pkl_file(self):
        ens = Layer2Ensemble()
        with tempfile.TemporaryDirectory() as tmp:
            ens.save_checkpoint(tmp)
            import os
            assert os.path.exists(os.path.join(tmp, "ensemble.pkl"))


# ---------------------------------------------------------------------------
# threshold 프로퍼티
# ---------------------------------------------------------------------------

class TestThresholdProperty:
    def test_set_valid_threshold(self):
        ens = Layer2Ensemble(threshold=0.62)
        ens.threshold = 0.70
        assert ens.threshold == pytest.approx(0.70)

    def test_set_invalid_threshold_raises(self):
        ens = Layer2Ensemble()
        with pytest.raises(ValueError):
            ens.threshold = 1.5
        with pytest.raises(ValueError):
            ens.threshold = 0.0


# ---------------------------------------------------------------------------
# DB 저장
# ---------------------------------------------------------------------------

class TestDbSave:
    def test_save_to_db_calls_upsert(self):
        mock_cache = MagicMock()
        ens = Layer2Ensemble(cache=mock_cache, trade_count_fn=lambda: 0)
        ens.predict(_ms())
        mock_cache.upsert_candle.assert_called_once()

    def test_save_to_db_correct_table(self):
        mock_cache = MagicMock()
        ens = Layer2Ensemble(cache=mock_cache, trade_count_fn=lambda: 0)
        ens.predict(_ms())
        args = mock_cache.upsert_candle.call_args[0]
        assert args[0] == "ensemble_predictions"

    def test_no_cache_no_error(self):
        ens = Layer2Ensemble(cache=None, trade_count_fn=lambda: 0)
        pred = ens.predict(_ms())  # 예외 없어야 함
        assert isinstance(pred, EnsemblePrediction)


# ---------------------------------------------------------------------------
# 시퀀스 버퍼 (LSTM/GRU용)
# ---------------------------------------------------------------------------

class TestSeqBuffer:
    def test_seq_buf_grows_with_predict(self):
        ens = Layer2Ensemble(trade_count_fn=lambda: 0)
        coin = "KRW-BTC"
        for _ in range(5):
            ens.predict(_ms(coin=coin))
        assert len(ens._seq_buf[coin]) == 5

    def test_seq_buf_maxlen_is_seq_len(self):
        ens = Layer2Ensemble()
        assert ens._seq_buf["KRW-BTC"].maxlen == SEQ_LEN

    def test_predict_seq_returns_empty_when_insufficient(self):
        ens = Layer2Ensemble(phase_b_enabled=_TORCH_AVAILABLE)
        result = ens._predict_seq_models("KRW-BTC")  # 버퍼 비어있음
        assert result == {}


# ---------------------------------------------------------------------------
# LSTM / GRU 모델 구조 (PyTorch 설치 시)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch 미설치")
class TestSeqModels:
    def test_lstm_model_init(self):
        from layers.layer2_ensemble import _LSTMModel
        model = _LSTMModel()
        assert model is not None

    def test_gru_model_init(self):
        from layers.layer2_ensemble import _GRUModel
        model = _GRUModel()
        assert model is not None

    def test_lstm_forward_shape(self):
        import torch
        from layers.layer2_ensemble import _LSTMModel
        model = _LSTMModel()
        x = torch.randn(1, SEQ_LEN, len(FEATURE_COLS))
        out = model(x)
        assert out.shape == (1, 1)

    def test_gru_forward_shape(self):
        import torch
        from layers.layer2_ensemble import _GRUModel
        model = _GRUModel()
        x = torch.randn(1, SEQ_LEN, len(FEATURE_COLS))
        out = model(x)
        assert out.shape == (1, 1)

    def test_output_in_0_1_range(self):
        import torch
        from layers.layer2_ensemble import _LSTMModel, _GRUModel
        x = torch.randn(2, SEQ_LEN, len(FEATURE_COLS))
        for Cls in (_LSTMModel, _GRUModel):
            out = Cls()(x)
            assert (out >= 0.0).all() and (out <= 1.0).all()

    def test_phase_b_init_creates_models(self):
        ens = Layer2Ensemble(phase_b_enabled=True)
        assert ens._lstm is not None
        assert ens._gru is not None

    def test_phase_b_checkpoint_saves_pth(self):
        ens = Layer2Ensemble(phase_b_enabled=True)
        with tempfile.TemporaryDirectory() as tmp:
            ens.save_checkpoint(tmp)
            import os
            assert os.path.exists(os.path.join(tmp, "lstm.pth"))
            assert os.path.exists(os.path.join(tmp, "gru.pth"))

    def test_phase_b_predict_returns_seq_probs_after_fill(self):
        """시퀀스 버퍼 SEQ_LEN 채우면 lstm/gru 확률 포함."""
        ens = Layer2Ensemble(
            phase_b_enabled=True,
            trade_count_fn=lambda: 600,
        )
        coin = "KRW-BTC"
        for _ in range(SEQ_LEN + 1):
            ens.predict(_ms(coin=coin))
        last_pred = ens.predict(_ms(coin=coin))
        assert "lstm" in last_pred.per_model_probs
        assert "gru" in last_pred.per_model_probs


# ---------------------------------------------------------------------------
# Phase A: LSTM/GRU 비활성화 시
# ---------------------------------------------------------------------------

class TestPhaseANoSeqModels:
    def test_phase_b_disabled_no_seq_models(self):
        ens = Layer2Ensemble(phase_b_enabled=False)
        assert ens._lstm is None
        assert ens._gru is None

    def test_phase_b_disabled_predict_seq_empty(self):
        ens = Layer2Ensemble(phase_b_enabled=False)
        buf = ens._seq_buf["KRW-BTC"]
        for _ in range(SEQ_LEN):
            buf.append(np.zeros(len(FEATURE_COLS), dtype=np.float32))
        assert ens._predict_seq_models("KRW-BTC") == {}


# ---------------------------------------------------------------------------
# EnsemblePrediction 구조
# ---------------------------------------------------------------------------

class TestEnsemblePredictionStructure:
    def test_all_fields_present(self):
        ens = Layer2Ensemble(trade_count_fn=lambda: 0)
        pred = ens.predict(_ms())
        assert hasattr(pred, "coin")
        assert hasattr(pred, "timestamp")
        assert hasattr(pred, "per_model_probs")
        assert hasattr(pred, "weighted_avg")
        assert hasattr(pred, "consensus_count")
        assert hasattr(pred, "signal_confirmed")
        assert hasattr(pred, "hmm_regime")
        assert hasattr(pred, "threshold_used")

    def test_hmm_regime_propagated(self):
        ens = Layer2Ensemble(trade_count_fn=lambda: 0)
        pred = ens.predict(_ms(hmm_regime=2))
        assert pred.hmm_regime == 2

    def test_timestamp_is_utc(self):
        ens = Layer2Ensemble(trade_count_fn=lambda: 0)
        pred = ens.predict(_ms())
        assert pred.timestamp.tzinfo is not None


# ---------------------------------------------------------------------------
# Optuna 탐색 (소규모)
# ---------------------------------------------------------------------------

class TestOptunaStudy:
    def test_run_returns_threshold(self):
        pytest.importorskip("optuna")
        pytest.importorskip("xgboost")
        X, y = _make_synthetic(100)
        result = run_optuna_study(X, y, n_trials=2, timeout_sec=30)
        assert "ensemble_threshold" in result
        assert 0.55 <= result["ensemble_threshold"] <= 0.75

    def test_run_returns_default_when_import_fails(self):
        """optuna 없을 때 기본값 반환 (import 실패 시뮬레이션)."""
        import sys
        import unittest.mock as mock
        X, y = _make_synthetic(50)
        with mock.patch.dict(sys.modules, {"optuna": None}):
            result = run_optuna_study(X, y, n_trials=1)
        assert result.get("ensemble_threshold") == pytest.approx(THRESHOLD_DEFAULT)
