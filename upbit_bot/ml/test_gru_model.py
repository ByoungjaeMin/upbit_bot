"""test_gru_model.py — GRUTrainer + GRUConfig 단위 테스트.

Phase B 비활성 상태(주석 처리)를 전제로 작성:
  - 임포트 성공 여부
  - Config 데이터클래스 동작
  - 스텁 인터페이스 (NotImplementedError / RuntimeError) 확인
  - is_active 플래그 확인
  - LSTM과의 인터페이스 일관성 검증 (앙상블 교환 가능성)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ml.gru_model import (
    BATCH_SIZE,
    DEVICE,
    DROPOUT,
    HIDDEN_SIZE,
    LEARNING_RATE,
    N_FEATURES,
    N_LAYERS,
    SEQ_LEN,
    GRUConfig,
    GRUTrainer,
    _PHASE_B_ACTIVE,
)


# ─────────────────────────────────────────────────
# 상수 검증
# ─────────────────────────────────────────────────

class TestConstants:
    def test_seq_len(self):
        assert SEQ_LEN == 60

    def test_n_features(self):
        assert N_FEATURES == 35

    def test_hidden_size(self):
        assert HIDDEN_SIZE == 128

    def test_n_layers(self):
        assert N_LAYERS == 2

    def test_dropout_range(self):
        assert 0.0 < DROPOUT < 1.0

    def test_batch_size(self):
        assert BATCH_SIZE == 32

    def test_learning_rate_positive(self):
        assert LEARNING_RATE > 0.0

    def test_device_is_cpu(self):
        """MPS 금지 — device=cpu 강제 확인."""
        assert DEVICE == "cpu"

    def test_phase_b_inactive_by_default(self):
        """Phase A 운영 중 — 기본값 False."""
        assert _PHASE_B_ACTIVE is False


# ─────────────────────────────────────────────────
# LSTM vs GRU 상수 일관성 (앙상블 호환성)
# ─────────────────────────────────────────────────

class TestLSTMGRUConsistency:
    """LSTM과 GRU는 동일 입력/출력 형태를 가져야 앙상블 가중치 적용 가능."""

    def test_seq_len_matches_lstm(self):
        from ml.lstm_model import SEQ_LEN as LSTM_SEQ_LEN
        assert SEQ_LEN == LSTM_SEQ_LEN

    def test_n_features_matches_lstm(self):
        from ml.lstm_model import N_FEATURES as LSTM_N_FEATURES
        assert N_FEATURES == LSTM_N_FEATURES

    def test_hidden_size_matches_lstm(self):
        from ml.lstm_model import HIDDEN_SIZE as LSTM_HIDDEN
        assert HIDDEN_SIZE == LSTM_HIDDEN

    def test_batch_size_matches_lstm(self):
        from ml.lstm_model import BATCH_SIZE as LSTM_BATCH
        assert BATCH_SIZE == LSTM_BATCH

    def test_device_matches_lstm(self):
        from ml.lstm_model import DEVICE as LSTM_DEVICE
        assert DEVICE == LSTM_DEVICE


# ─────────────────────────────────────────────────
# GRUConfig
# ─────────────────────────────────────────────────

class TestGRUConfig:
    def test_default_instantiation(self):
        cfg = GRUConfig()
        assert cfg.seq_len == SEQ_LEN
        assert cfg.n_features == N_FEATURES
        assert cfg.hidden_size == HIDDEN_SIZE
        assert cfg.n_layers == N_LAYERS
        assert cfg.dropout == pytest.approx(DROPOUT)
        assert cfg.batch_size == BATCH_SIZE
        assert cfg.learning_rate == pytest.approx(LEARNING_RATE)
        assert cfg.device == DEVICE

    def test_custom_values(self):
        cfg = GRUConfig(hidden_size=64, n_layers=1, max_epochs=20)
        assert cfg.hidden_size == 64
        assert cfg.n_layers == 1
        assert cfg.max_epochs == 20

    def test_checkpoint_fields_gru_specific(self):
        """GRU 체크포인트명이 LSTM과 달라야 충돌 방지."""
        cfg = GRUConfig()
        assert cfg.checkpoint_name == "gru_best.pt"
        assert "gru" in cfg.checkpoint_name.lower()

    def test_checkpoint_dir_default(self):
        cfg = GRUConfig()
        assert cfg.checkpoint_dir == "models"

    def test_patience_positive(self):
        cfg = GRUConfig()
        assert cfg.patience > 0

    def test_incremental_weight_gt_one(self):
        cfg = GRUConfig()
        assert cfg.incremental_weight > 1.0

    def test_dataclass_equality(self):
        cfg1 = GRUConfig(hidden_size=64)
        cfg2 = GRUConfig(hidden_size=64)
        assert cfg1 == cfg2

    def test_dataclass_inequality(self):
        cfg1 = GRUConfig(hidden_size=64)
        cfg2 = GRUConfig(hidden_size=128)
        assert cfg1 != cfg2

    def test_gru_lstm_checkpoint_names_differ(self):
        """GRU와 LSTM 체크포인트 파일명 분리 확인."""
        from ml.lstm_model import LSTMConfig
        gru_cfg = GRUConfig()
        lstm_cfg = LSTMConfig()
        assert gru_cfg.checkpoint_name != lstm_cfg.checkpoint_name


# ─────────────────────────────────────────────────
# GRUTrainer 스텁 인터페이스
# ─────────────────────────────────────────────────

class TestGRUTrainerStub:
    def _trainer(self) -> GRUTrainer:
        return GRUTrainer()

    def test_instantiation_no_error(self):
        """Phase B 비활성 상태에서도 인스턴스화 가능."""
        trainer = self._trainer()
        assert isinstance(trainer, GRUTrainer)

    def test_instantiation_with_config(self):
        cfg = GRUConfig(hidden_size=64)
        trainer = GRUTrainer(config=cfg)
        assert trainer._cfg.hidden_size == 64

    def test_is_active_false(self):
        trainer = self._trainer()
        assert trainer.is_active is False

    def test_train_raises_not_implemented(self):
        """Phase B 비활성 → train() 호출 시 NotImplementedError."""
        trainer = self._trainer()
        with pytest.raises(NotImplementedError, match="Phase B"):
            trainer.train()

    def test_predict_raises_not_implemented(self):
        trainer = self._trainer()
        with pytest.raises(NotImplementedError, match="Phase B"):
            trainer.predict()

    def test_incremental_update_raises_not_implemented(self):
        trainer = self._trainer()
        with pytest.raises(NotImplementedError, match="Phase B"):
            trainer.incremental_update()

    def test_save_no_op(self):
        """save()는 비활성 상태에서 조용히 종료 (no-op)."""
        trainer = self._trainer()
        trainer.save()  # 예외 없음

    def test_save_with_path_no_op(self):
        trainer = self._trainer()
        trainer.save(path="/tmp/test_gru.pt")  # 예외 없음

    def test_load_raises_runtime_error(self):
        """load()는 체크포인트 없을 때 RuntimeError."""
        trainer = self._trainer()
        with pytest.raises(RuntimeError, match="Phase B"):
            trainer.load()

    def test_load_with_nonexistent_path_raises(self):
        trainer = self._trainer()
        with pytest.raises(RuntimeError):
            trainer.load(path="/nonexistent/path/gru.pt")

    def test_train_error_message_contains_phase_b(self):
        trainer = self._trainer()
        try:
            trainer.train()
        except NotImplementedError as e:
            assert "Phase B" in str(e)

    def test_load_error_message_contains_colab_or_phase_b(self):
        trainer = self._trainer()
        try:
            trainer.load()
        except RuntimeError as e:
            assert "Colab" in str(e) or "Phase B" in str(e)

    def test_config_accessible(self):
        cfg = GRUConfig(max_epochs=30)
        trainer = GRUTrainer(config=cfg)
        assert trainer._cfg.max_epochs == 30

    def test_default_config_used_when_none(self):
        """config=None → GRUConfig() 기본값 사용."""
        trainer = GRUTrainer(config=None)
        assert trainer._cfg.seq_len == SEQ_LEN


# ─────────────────────────────────────────────────
# 임포트 수준 검증
# ─────────────────────────────────────────────────

class TestImport:
    def test_module_importable(self):
        """PyTorch 없이도 모듈 임포트 가능 (Phase A 환경 보장)."""
        import ml.gru_model  # noqa: F401

    def test_all_public_names_accessible(self):
        from ml import gru_model
        assert hasattr(gru_model, "GRUConfig")
        assert hasattr(gru_model, "GRUTrainer")
        assert hasattr(gru_model, "SEQ_LEN")
        assert hasattr(gru_model, "N_FEATURES")
        assert hasattr(gru_model, "DEVICE")
        assert hasattr(gru_model, "_PHASE_B_ACTIVE")


# ─────────────────────────────────────────────────
# LSTM / GRU 트레이너 인터페이스 일관성
# ─────────────────────────────────────────────────

class TestTrainerInterfaceConsistency:
    """앙상블 코드에서 LSTM ↔ GRU 교환 가능성을 보장하는 계약 테스트."""

    def test_both_have_train(self):
        from ml.lstm_model import LSTMTrainer
        assert hasattr(LSTMTrainer(), "train")
        assert hasattr(GRUTrainer(), "train")

    def test_both_have_predict(self):
        from ml.lstm_model import LSTMTrainer
        assert hasattr(LSTMTrainer(), "predict")
        assert hasattr(GRUTrainer(), "predict")

    def test_both_have_incremental_update(self):
        from ml.lstm_model import LSTMTrainer
        assert hasattr(LSTMTrainer(), "incremental_update")
        assert hasattr(GRUTrainer(), "incremental_update")

    def test_both_have_save(self):
        from ml.lstm_model import LSTMTrainer
        assert hasattr(LSTMTrainer(), "save")
        assert hasattr(GRUTrainer(), "save")

    def test_both_have_load(self):
        from ml.lstm_model import LSTMTrainer
        assert hasattr(LSTMTrainer(), "load")
        assert hasattr(GRUTrainer(), "load")

    def test_both_have_is_active(self):
        from ml.lstm_model import LSTMTrainer
        assert hasattr(LSTMTrainer(), "is_active")
        assert hasattr(GRUTrainer(), "is_active")

    def test_both_inactive_by_default(self):
        from ml.lstm_model import LSTMTrainer
        assert LSTMTrainer().is_active is False
        assert GRUTrainer().is_active is False

    def test_both_train_raise_same_error_type(self):
        from ml.lstm_model import LSTMTrainer
        with pytest.raises(NotImplementedError):
            LSTMTrainer().train()
        with pytest.raises(NotImplementedError):
            GRUTrainer().train()

    def test_both_load_raise_same_error_type(self):
        from ml.lstm_model import LSTMTrainer
        with pytest.raises(RuntimeError):
            LSTMTrainer().load()
        with pytest.raises(RuntimeError):
            GRUTrainer().load()


# ─────────────────────────────────────────────────
# Phase B 활성화 시나리오 (인터페이스 계약)
# ─────────────────────────────────────────────────

class TestPhaseActivationInterface:
    def test_train_signature_accepts_arrays(self):
        trainer = GRUTrainer()
        try:
            trainer.train([[1, 2], [3, 4]], [0, 1])
        except NotImplementedError:
            pass

    def test_predict_signature_accepts_array(self):
        trainer = GRUTrainer()
        try:
            trainer.predict([[1, 2, 3]])
        except NotImplementedError:
            pass

    def test_incremental_update_signature(self):
        trainer = GRUTrainer()
        try:
            trainer.incremental_update([[1, 2]], [1], weight_multiplier=2.0)
        except NotImplementedError:
            pass

    def test_save_signature_accepts_path(self):
        trainer = GRUTrainer()
        trainer.save(path=Path("models/gru_best.pt"))  # no-op

    def test_load_signature_accepts_path(self):
        trainer = GRUTrainer()
        with pytest.raises(RuntimeError):
            trainer.load(path=Path("models/gru_best.pt"))
