"""
ml/gru_model.py — GRU 이진 분류기 (Phase B, 현재 주석 처리 상태)

[활성화 조건]
  trade_count >= 500 이고 Google Colab 초기 학습 완료 후 활성화.
  활성화 방법:
    1. 아래 _PHASE_B_ACTIVE = False → True 변경
    2. Google Colab에서 GRUTrainer.train() 실행, 체크포인트 models/ 저장
    3. layer2_ensemble.py 의 phase_b_enabled=True 변경

[모델 스펙]
  입력: (batch, seq_len=60, n_features=35) — 5분봉 60 타임스텝 × 35 피처
  구조: 2레이어 GRU, hidden=128, Dropout(0.2)
  출력: sigmoid(Linear(128→1)) — 매수 확률 0~1
  손실: BCELoss / 옵티마이저: Adam(lr=0.001)
  device=cpu 강제 (MPS 금지, Mac Mini M4 안정성)

[LSTM vs GRU]
  GRU는 LSTM 대비 파라미터 수 ~25% 적음 → Mac Mini 메모리 절약
  긴 시퀀스(60 타임스텝)에서 LSTM과 성능 유사 → 앙상블 다양성 확보
  레짐 2,3(하락/횡보) 구간에서 GRU가 LSTM보다 안정적 경향

[학습 전략]
  초기 학습: Google Colab (GPU) → 체크포인트 → 맥미니 fine-tuning
  Incremental: 매 거래 후 소폭 업데이트 (새 샘플 가중치 ×2.0)
  배치사이즈: 32 (하드웨어 제약)
  LSTM과 동일 DataLoader / 동일 Incremental 전략 사용
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ─────────────────────────────────────────────────
# Phase B 활성화 플래그
# ─────────────────────────────────────────────────
_PHASE_B_ACTIVE: bool = False  # Phase B 시작 전까지 False 유지

# ─────────────────────────────────────────────────
# 공통 상수 (주석 해제 불필요 — 항상 활성)
# ─────────────────────────────────────────────────
SEQ_LEN: int = 60
N_FEATURES: int = 35
HIDDEN_SIZE: int = 128
N_LAYERS: int = 2
DROPOUT: float = 0.2
BATCH_SIZE: int = 32
LEARNING_RATE: float = 0.001
DEVICE: str = "cpu"  # MPS 금지


@dataclass
class GRUConfig:
    """GRU 하이퍼파라미터 설정."""
    seq_len: int = SEQ_LEN
    n_features: int = N_FEATURES
    hidden_size: int = HIDDEN_SIZE
    n_layers: int = N_LAYERS
    dropout: float = DROPOUT
    batch_size: int = BATCH_SIZE
    learning_rate: float = LEARNING_RATE
    device: str = DEVICE
    checkpoint_dir: str = "models"
    checkpoint_name: str = "gru_best.pt"
    max_epochs: int = 50
    patience: int = 10          # Early stopping
    incremental_weight: float = 2.0  # 새 샘플 가중치 배율


# ─────────────────────────────────────────────────
# [Phase B 주석 처리 시작]
# 아래 코드는 _PHASE_B_ACTIVE = True 로 변경 후 사용
# ─────────────────────────────────────────────────

# import torch
# import torch.nn as nn
# import numpy as np
# from torch.utils.data import Dataset, DataLoader
#
#
# class GRUDataset(Dataset):
#     """GRU 학습용 시계열 Dataset.
#
#     lstm_model.LSTMDataset과 동일 구조 — 두 모델이 같은 데이터로 학습.
#
#     Args:
#         X: (n_samples, seq_len, n_features) — 정규화된 피처
#         y: (n_samples,) — 이진 레이블 (1=매수 시점 이후 수익, 0=그 외)
#         sample_weights: (n_samples,) — 최신 샘플 ×2.0 가중치
#     """
#
#     def __init__(
#         self,
#         X: np.ndarray,
#         y: np.ndarray,
#         sample_weights: np.ndarray | None = None,
#     ) -> None:
#         assert X.shape[0] == y.shape[0], "X, y 길이 불일치"
#         self.X = torch.from_numpy(X.astype(np.float32))
#         self.y = torch.from_numpy(y.astype(np.float32))
#         if sample_weights is not None:
#             self.weights = torch.from_numpy(sample_weights.astype(np.float32))
#         else:
#             self.weights = torch.ones(len(y), dtype=torch.float32)
#
#     def __len__(self) -> int:
#         return len(self.y)
#
#     def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         return self.X[idx], self.y[idx], self.weights[idx]
#
#
# class GRUBinaryClassifier(nn.Module):
#     """2레이어 GRU 이진 분류기.
#
#     입력: (batch, seq_len, n_features)
#     출력: (batch, 1) — sigmoid 확률
#
#     LSTM 대비 파라미터 ~25% 절약 (GRU 셀 구조 차이).
#     device=cpu 강제 (Mac Mini M4 MPS 불안정 이슈 회피).
#     """
#
#     def __init__(self, config: GRUConfig) -> None:
#         super().__init__()
#         self.gru = nn.GRU(
#             input_size=config.n_features,
#             hidden_size=config.hidden_size,
#             num_layers=config.n_layers,
#             batch_first=True,
#             dropout=config.dropout if config.n_layers > 1 else 0.0,
#         )
#         self.fc = nn.Linear(config.hidden_size, 1)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # x: (batch, seq_len, features)
#         out, _ = self.gru(x)           # (batch, seq_len, hidden)
#         last = out[:, -1, :]           # 마지막 타임스텝
#         return self.sigmoid(self.fc(last))  # (batch, 1)
#
#
# class GRUTrainer:
#     """GRU 학습 + Incremental 업데이트 관리.
#
#     LSTMTrainer와 동일 인터페이스 — 앙상블 코드에서 교환 가능.
#
#     사용법 (Google Colab):
#         config = GRUConfig(max_epochs=50)
#         trainer = GRUTrainer(config)
#         trainer.train(X_train, y_train)
#         trainer.save()
#
#     사용법 (맥미니 incremental):
#         trainer = GRUTrainer(config)
#         trainer.load()
#         trainer.incremental_update(X_new, y_new)
#     """
#
#     def __init__(self, config: GRUConfig | None = None) -> None:
#         self._cfg = config or GRUConfig()
#         self._device = torch.device(self._cfg.device)
#         self._model: GRUBinaryClassifier | None = None
#         self._optimizer: torch.optim.Adam | None = None
#         self._criterion = nn.BCELoss(reduction="none")
#
#     def _build_model(self) -> GRUBinaryClassifier:
#         model = GRUBinaryClassifier(self._cfg)
#         model.to(self._device)
#         return model
#
#     def train(
#         self,
#         X: np.ndarray,
#         y: np.ndarray,
#         X_val: np.ndarray | None = None,
#         y_val: np.ndarray | None = None,
#         sample_weights: np.ndarray | None = None,
#     ) -> dict[str, list[float]]:
#         """전체 학습 루프.
#
#         Args:
#             X: (n, seq_len, n_features) 정규화된 피처
#             y: (n,) 이진 레이블
#             X_val / y_val: 검증 데이터 (None이면 X의 마지막 20% 사용)
#             sample_weights: (n,) 샘플 가중치 (최신 ×2.0)
#
#         Returns:
#             {'train_loss': [...], 'val_loss': [...]}
#         """
#         if self._model is None:
#             self._model = self._build_model()
#         self._optimizer = torch.optim.Adam(
#             self._model.parameters(), lr=self._cfg.learning_rate
#         )
#
#         if X_val is None:
#             split = int(len(X) * 0.8)
#             X_train, X_val = X[:split], X[split:]
#             y_train, y_val = y[:split], y[split:]
#             w_train = sample_weights[:split] if sample_weights is not None else None
#         else:
#             X_train, y_train = X, y
#             w_train = sample_weights
#
#         ds_train = GRUDataset(X_train, y_train, w_train)
#         ds_val   = GRUDataset(X_val, y_val)
#         dl_train = DataLoader(ds_train, batch_size=self._cfg.batch_size, shuffle=True)
#         dl_val   = DataLoader(ds_val,   batch_size=self._cfg.batch_size)
#
#         history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
#         best_val_loss = float("inf")
#         patience_counter = 0
#
#         for epoch in range(self._cfg.max_epochs):
#             self._model.train()
#             train_losses = []
#             for X_b, y_b, w_b in dl_train:
#                 X_b = X_b.to(self._device)
#                 y_b = y_b.to(self._device).unsqueeze(1)
#                 w_b = w_b.to(self._device)
#                 self._optimizer.zero_grad()
#                 pred = self._model(X_b)
#                 loss = (self._criterion(pred, y_b).squeeze() * w_b).mean()
#                 loss.backward()
#                 torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
#                 self._optimizer.step()
#                 train_losses.append(loss.item())
#
#             self._model.eval()
#             val_losses = []
#             with torch.no_grad():
#                 for X_b, y_b, _ in dl_val:
#                     X_b = X_b.to(self._device)
#                     y_b = y_b.to(self._device).unsqueeze(1)
#                     pred = self._model(X_b)
#                     val_losses.append(self._criterion(pred, y_b).mean().item())
#
#             t_loss = float(np.mean(train_losses))
#             v_loss = float(np.mean(val_losses))
#             history["train_loss"].append(t_loss)
#             history["val_loss"].append(v_loss)
#
#             if v_loss < best_val_loss:
#                 best_val_loss = v_loss
#                 patience_counter = 0
#                 self.save()
#             else:
#                 patience_counter += 1
#             if patience_counter >= self._cfg.patience:
#                 break
#
#         return history
#
#     def predict(self, X: np.ndarray) -> np.ndarray:
#         """추론. Returns: (n,) 매수 확률 0~1."""
#         if self._model is None:
#             raise RuntimeError("모델 미로드 — load() 또는 train() 먼저 호출")
#         self._model.eval()
#         X_t = torch.from_numpy(X.astype(np.float32)).to(self._device)
#         with torch.no_grad():
#             probs = self._model(X_t).cpu().numpy().flatten()
#         return probs
#
#     def incremental_update(
#         self,
#         X_new: np.ndarray,
#         y_new: np.ndarray,
#         weight_multiplier: float = 2.0,
#     ) -> float:
#         """새 거래 데이터로 소폭 업데이트 (Incremental Learning).
#
#         Returns:
#             업데이트 손실값
#         """
#         if self._model is None:
#             raise RuntimeError("모델 미로드")
#         if self._optimizer is None:
#             self._optimizer = torch.optim.Adam(
#                 self._model.parameters(), lr=self._cfg.learning_rate * 0.1
#             )
#
#         weights = np.full(len(y_new), weight_multiplier, dtype=np.float32)
#         ds = GRUDataset(X_new, y_new, weights)
#         dl = DataLoader(ds, batch_size=min(self._cfg.batch_size, len(ds)))
#
#         self._model.train()
#         total_loss = 0.0
#         for X_b, y_b, w_b in dl:
#             X_b = X_b.to(self._device)
#             y_b = y_b.to(self._device).unsqueeze(1)
#             w_b = w_b.to(self._device)
#             self._optimizer.zero_grad()
#             pred = self._model(X_b)
#             loss = (self._criterion(pred, y_b).squeeze() * w_b).mean()
#             loss.backward()
#             self._optimizer.step()
#             total_loss += loss.item()
#
#         return total_loss / max(len(dl), 1)
#
#     def save(self, path: str | Path | None = None) -> None:
#         """체크포인트 저장."""
#         if self._model is None:
#             return
#         p = Path(path) if path else Path(self._cfg.checkpoint_dir) / self._cfg.checkpoint_name
#         p.parent.mkdir(parents=True, exist_ok=True)
#         torch.save(self._model.state_dict(), p)
#
#     def load(self, path: str | Path | None = None) -> None:
#         """체크포인트 로드."""
#         p = Path(path) if path else Path(self._cfg.checkpoint_dir) / self._cfg.checkpoint_name
#         if not p.exists():
#             raise FileNotFoundError(f"체크포인트 없음: {p}")
#         self._model = self._build_model()
#         self._model.load_state_dict(
#             torch.load(p, map_location=self._device, weights_only=True)
#         )
#         self._model.eval()

# ─────────────────────────────────────────────────
# [Phase B 주석 처리 끝]
# ─────────────────────────────────────────────────


class GRUTrainer:
    """GRUTrainer 스텁 — Phase B 비활성 시 임포트 안전 보장.

    _PHASE_B_ACTIVE = True 변경 및 위 주석 블록 해제 시 실제 구현으로 교체.
    """

    def __init__(self, config: GRUConfig | None = None) -> None:
        self._cfg = config or GRUConfig()
        self._phase_b = _PHASE_B_ACTIVE

    def train(self, *args: Any, **kwargs: Any) -> dict[str, list[float]]:
        """[Phase B 비활성] 호출 시 NotImplementedError."""
        raise NotImplementedError(
            "GRUTrainer.train()은 Phase B 이후 활성화됩니다. "
            "_PHASE_B_ACTIVE = True 및 위 주석 블록을 해제하세요."
        )

    def predict(self, *args: Any, **kwargs: Any) -> Any:
        """[Phase B 비활성] 호출 시 NotImplementedError."""
        raise NotImplementedError(
            "GRUTrainer.predict()은 Phase B 이후 활성화됩니다."
        )

    def incremental_update(self, *args: Any, **kwargs: Any) -> float:
        """[Phase B 비활성] 호출 시 NotImplementedError."""
        raise NotImplementedError(
            "GRUTrainer.incremental_update()은 Phase B 이후 활성화됩니다."
        )

    def save(self, path: str | Path | None = None) -> None:
        """[Phase B 비활성] no-op."""
        pass

    def load(self, path: str | Path | None = None) -> None:
        """[Phase B 비활성] 호출 시 RuntimeError."""
        raise RuntimeError(
            "GRUTrainer.load()은 Phase B 이후 활성화됩니다. "
            "먼저 Google Colab에서 초기 학습 후 체크포인트를 생성하세요."
        )

    @property
    def is_active(self) -> bool:
        """Phase B 활성화 여부."""
        return self._phase_b
