"""test_lstm_model.py — LSTMTrainer + LSTMConfig 단위 테스트.

Phase B 비활성 상태(주석 처리)를 전제로 작성:
  - 임포트 성공 여부
  - Config 데이터클래스 동작
  - 스텁 인터페이스 (NotImplementedError / RuntimeError) 확인
  - is_active 플래그 확인
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ml.lstm_model import (
    BATCH_SIZE,
    DEVICE,
    DROPOUT,
    HIDDEN_SIZE,
    LEARNING_RATE,
    N_FEATURES,
    N_LAYERS,
    SEQ_LEN,
    LSTMConfig,
    LSTMTrainer,
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
# LSTMConfig
# ─────────────────────────────────────────────────

class TestLSTMConfig:
    def test_default_instantiation(self):
        cfg = LSTMConfig()
        assert cfg.seq_len == SEQ_LEN
        assert cfg.n_features == N_FEATURES
        assert cfg.hidden_size == HIDDEN_SIZE
        assert cfg.n_layers == N_LAYERS
        assert cfg.dropout == pytest.approx(DROPOUT)
        assert cfg.batch_size == BATCH_SIZE
        assert cfg.learning_rate == pytest.approx(LEARNING_RATE)
        assert cfg.device == DEVICE

    def test_custom_values(self):
        cfg = LSTMConfig(hidden_size=64, n_layers=1, max_epochs=20)
        assert cfg.hidden_size == 64
        assert cfg.n_layers == 1
        assert cfg.max_epochs == 20

    def test_checkpoint_fields_default(self):
        cfg = LSTMConfig()
        assert cfg.checkpoint_dir == "models"
        assert cfg.checkpoint_name == "lstm_best.pt"

    def test_patience_positive(self):
        cfg = LSTMConfig()
        assert cfg.patience > 0

    def test_incremental_weight_gt_one(self):
        """최신 샘플 가중치는 1보다 커야 함 (Incremental 학습 효과)."""
        cfg = LSTMConfig()
        assert cfg.incremental_weight > 1.0

    def test_dataclass_equality(self):
        cfg1 = LSTMConfig(hidden_size=64)
        cfg2 = LSTMConfig(hidden_size=64)
        assert cfg1 == cfg2

    def test_dataclass_inequality(self):
        cfg1 = LSTMConfig(hidden_size=64)
        cfg2 = LSTMConfig(hidden_size=128)
        assert cfg1 != cfg2


# ─────────────────────────────────────────────────
# LSTMTrainer 스텁 인터페이스
# ─────────────────────────────────────────────────

class TestLSTMTrainerStub:
    def _trainer(self) -> LSTMTrainer:
        return LSTMTrainer()

    def test_instantiation_no_error(self):
        """Phase B 비활성 상태에서도 인스턴스화 가능."""
        trainer = self._trainer()
        assert isinstance(trainer, LSTMTrainer)

    def test_instantiation_with_config(self):
        cfg = LSTMConfig(hidden_size=64)
        trainer = LSTMTrainer(config=cfg)
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
        trainer.save(path="/tmp/test_lstm.pt")  # 예외 없음

    def test_load_raises_runtime_error(self):
        """load()는 체크포인트 없을 때 RuntimeError."""
        trainer = self._trainer()
        with pytest.raises(RuntimeError, match="Phase B"):
            trainer.load()

    def test_load_with_nonexistent_path_raises(self):
        trainer = self._trainer()
        with pytest.raises(RuntimeError):
            trainer.load(path="/nonexistent/path/lstm.pt")

    def test_train_error_message_contains_phase_b(self):
        trainer = self._trainer()
        try:
            trainer.train()
        except NotImplementedError as e:
            assert "Phase B" in str(e)

    def test_load_error_message_contains_colab(self):
        trainer = self._trainer()
        try:
            trainer.load()
        except RuntimeError as e:
            assert "Colab" in str(e) or "Phase B" in str(e)

    def test_config_accessible(self):
        cfg = LSTMConfig(max_epochs=30)
        trainer = LSTMTrainer(config=cfg)
        assert trainer._cfg.max_epochs == 30

    def test_default_config_used_when_none(self):
        """config=None → LSTMConfig() 기본값 사용."""
        trainer = LSTMTrainer(config=None)
        assert trainer._cfg.seq_len == SEQ_LEN


# ─────────────────────────────────────────────────
# 임포트 수준 검증
# ─────────────────────────────────────────────────

class TestImport:
    def test_module_importable(self):
        """PyTorch 없이도 모듈 임포트 가능 (Phase A 환경 보장)."""
        import ml.lstm_model  # noqa: F401

    def test_all_public_names_accessible(self):
        from ml import lstm_model
        assert hasattr(lstm_model, "LSTMConfig")
        assert hasattr(lstm_model, "LSTMTrainer")
        assert hasattr(lstm_model, "SEQ_LEN")
        assert hasattr(lstm_model, "N_FEATURES")
        assert hasattr(lstm_model, "DEVICE")
        assert hasattr(lstm_model, "_PHASE_B_ACTIVE")


# ─────────────────────────────────────────────────
# Phase B 활성화 시나리오 (모킹으로 검증)
# ─────────────────────────────────────────────────

class TestPhaseActivationInterface:
    """Phase B 활성화 후 예상 인터페이스를 확인하는 계약 테스트.

    실제 PyTorch 없이도 인터페이스 형태를 검증.
    """

    def test_train_signature_accepts_arrays(self):
        """train() 시그니처: *args, **kwargs 허용 → 배열 전달 가능."""
        trainer = LSTMTrainer()
        try:
            trainer.train([[1, 2], [3, 4]], [0, 1])
        except NotImplementedError:
            pass  # Phase B 비활성 → 예상된 동작

    def test_predict_signature_accepts_array(self):
        trainer = LSTMTrainer()
        try:
            trainer.predict([[1, 2, 3]])
        except NotImplementedError:
            pass

    def test_incremental_update_signature(self):
        trainer = LSTMTrainer()
        try:
            trainer.incremental_update([[1, 2]], [1], weight_multiplier=2.0)
        except NotImplementedError:
            pass

    def test_save_signature_accepts_path(self):
        trainer = LSTMTrainer()
        trainer.save(path=Path("models/lstm_best.pt"))  # no-op

    def test_load_signature_accepts_path(self):
        trainer = LSTMTrainer()
        with pytest.raises(RuntimeError):
            trainer.load(path=Path("models/lstm_best.pt"))
