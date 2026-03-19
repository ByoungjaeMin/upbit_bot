"""
layers/layer2_ensemble.py — Layer 2 ML 앙상블 예측기

[Phase A] XGBoost + LightGBM (trade_count >= 200)
[Phase B] LSTM + GRU 추가 (trade_count >= 500, phase_b_enabled=True)

콜드스타트: trade_count < 200 → 룰 기반 스코어 (학습 불필요)
신호 확정: 가중 평균 >= threshold(기본 0.62) AND 합의 모델 수 >= 3

[v9] ensemble_threshold=0.62 초기값, Optuna 0.55~0.75 탐색으로 자동 최적화.
Incremental: 새 샘플 가중치 ×2.0, retrain_interval(기본 50)마다 재학습.

모델 가중치: XGBoost×1.0, LightGBM×1.0, LSTM×0.8, GRU×0.8
레짐 그룹: HMM 0,1 → 'aggressive' / HMM 2,3 → 'conservative'
           Phase A/B: ADX >= 20 → 'aggressive'
"""

from __future__ import annotations

import logging
import pickle
import threading
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.preprocessing import StandardScaler

from schema import EnsemblePrediction, MarketState

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# 하이퍼파라미터 상수
# ------------------------------------------------------------------

SEQ_LEN: int = 60                    # LSTM/GRU 시퀀스 길이 (5분봉 60개 = 5시간)
BATCH_SIZE: int = 32                 # M4 16GB RAM 제약
LSTM_HIDDEN: int = 128
LSTM_LAYERS: int = 2
LSTM_DROPOUT: float = 0.2
THRESHOLD_DEFAULT: float = 0.62      # [v9] 0.55→0.62
INCREMENTAL_WEIGHT: float = 2.0      # 새 샘플 가중치 배율
MIN_CONSENSUS: int = 3               # 신호 확정 최소 합의 모델 수
MAX_BUF_SIZE: int = 5000             # 학습 버퍼 롤링 상한

MODEL_WEIGHTS: dict[str, float] = {
    "xgb":  1.0,
    "lgb":  1.0,
    "lstm": 0.8,
    "gru":  0.8,
    "rule": 0.6,   # 콜드스타트 룰 기반
}

# 35개 피처 목록 (MarketState 필드, 일봉은 shift(1) 적용값)
FEATURE_COLS: list[str] = [
    # 5분봉 OHLCV (5)
    "open_5m", "high_5m", "low_5m", "close_5m", "volume_5m",
    # 5분봉 기술지표 (9)
    "rsi_5m", "macd_5m", "macd_signal_5m",
    "bb_upper_5m", "bb_lower_5m",
    "ema7_5m", "ema25_5m", "ema99_5m",
    "volume_ratio_5m",
    # 1시간봉 (5)
    "rsi_1h", "ema20_1h", "ema50_1h", "macd_1h", "trend_dir_1h",
    # 일봉 — shift(1) 적용 필수 (4)
    "ema50_1d", "ema200_1d", "rsi_1d", "trend_encoding_1d",
    # 시장지수 (3)
    "fear_greed", "btc_dominance", "altcoin_season",
    # 온체인 (2)
    "exchange_inflow", "exchange_outflow",
    # 감성 (2)
    "sentiment_score", "sentiment_confidence",
    # 마이크로스트럭처 (3)
    "tick_imbalance", "obi", "trade_velocity",
    # 추세·레짐 (2)
    "adx_5m", "adx_1h",
]  # len == 35

# XGBoost 파라미터 (레짐별)
_XGB_PARAMS: dict[str, dict[str, Any]] = {
    "aggressive": dict(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=42,
    ),
    "conservative": dict(
        n_estimators=200, max_depth=4, learning_rate=0.03,
        subsample=0.7, colsample_bytree=0.7,
        eval_metric="logloss", random_state=42,
    ),
}

# LightGBM 파라미터 (레짐별)
_LGB_PARAMS: dict[str, dict[str, Any]] = {
    "aggressive": dict(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbose=-1,
    ),
    "conservative": dict(
        n_estimators=200, max_depth=4, learning_rate=0.03,
        subsample=0.7, colsample_bytree=0.7,
        random_state=42, verbose=-1,
    ),
}

# ------------------------------------------------------------------
# PyTorch 모델 (Phase B, device=cpu 강제 — MPS 금지)
# ------------------------------------------------------------------

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    logger.warning("PyTorch 미설치 — LSTM/GRU Phase B 비활성화")


if _TORCH_AVAILABLE:
    class _LSTMModel(nn.Module):  # type: ignore[misc]
        """2레이어 LSTM 이진 분류기 (BCELoss, Adam lr=0.001)."""

        def __init__(
            self,
            input_size: int = len(FEATURE_COLS),
            hidden: int = LSTM_HIDDEN,
            n_layers: int = LSTM_LAYERS,
            dropout: float = LSTM_DROPOUT,
        ) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size, hidden, n_layers,
                batch_first=True,
                dropout=dropout if n_layers > 1 else 0.0,
            )
            self.fc = nn.Linear(hidden, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out, _ = self.lstm(x)
            return self.sigmoid(self.fc(out[:, -1, :]))

    class _GRUModel(nn.Module):  # type: ignore[misc]
        """2레이어 GRU 이진 분류기 (BCELoss, Adam lr=0.001)."""

        def __init__(
            self,
            input_size: int = len(FEATURE_COLS),
            hidden: int = LSTM_HIDDEN,
            n_layers: int = LSTM_LAYERS,
            dropout: float = LSTM_DROPOUT,
        ) -> None:
            super().__init__()
            self.gru = nn.GRU(
                input_size, hidden, n_layers,
                batch_first=True,
                dropout=dropout if n_layers > 1 else 0.0,
            )
            self.fc = nn.Linear(hidden, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out, _ = self.gru(x)
            return self.sigmoid(self.fc(out[:, -1, :]))

else:
    class _LSTMModel:  # type: ignore[no-redef]
        pass

    class _GRUModel:  # type: ignore[no-redef]
        pass


# ------------------------------------------------------------------
# Layer2Ensemble
# ------------------------------------------------------------------

class Layer2Ensemble:
    """ML 앙상블 예측기.

    콜드스타트 전략:
        trade_count <  200 → 룰 기반 스코어 (학습 불필요)
        trade_count >= 200 → XGBoost + LightGBM 활성
        trade_count >= 500 & phase_b_enabled → + LSTM + GRU

    사용법:
        ensemble = Layer2Ensemble(
            cache=cache,
            trade_count_fn=lambda: db.count_trades(),
        )
        pred = ensemble.predict(market_state)
        if pred.signal_confirmed:
            ...  # 진입 신호 처리
    """

    def __init__(
        self,
        cache: Any = None,
        trade_count_fn: Callable[[], int] | None = None,
        threshold: float = THRESHOLD_DEFAULT,
        consensus_threshold: float = 0.5,  # 개별 모델 합의 판정 임계값 (앙상블 임계값과 별개)
        phase_b_enabled: bool = False,     # LSTM/GRU 활성화 (Phase B 이후)
        retrain_interval: int = 50,        # 신규 샘플 N개마다 자동 재학습
    ) -> None:
        self._cache = cache
        self._trade_count_fn: Callable[[], int] = trade_count_fn or (lambda: 0)
        self._threshold = threshold
        # 개별 모델 합의 판정 임계값 — 앙상블 최종 임계값(self._threshold=0.62)과 분리.
        # 기획서: "앙상블 가중 평균 >= 0.62 AND 3모델 이상 합의"에서 합의 기준은 별개.
        self._consensus_threshold = consensus_threshold
        self._phase_b = phase_b_enabled and _TORCH_AVAILABLE
        self._retrain_interval = retrain_interval
        self._lock = threading.Lock()

        # 트리 모델 저장소 (레짐 그룹 → 모델 인스턴스)
        self._xgb: dict[str, Any] = {}
        self._lgb: dict[str, Any] = {}

        # 시퀀스 모델 (Phase B)
        self._lstm: _LSTMModel | None = None
        self._gru: _GRUModel | None = None
        self._lstm_opt: Any = None
        self._gru_opt: Any = None

        # 스케일러
        self._scaler = StandardScaler()
        self._scaler_fitted = False

        # 롤링 학습 버퍼 (최대 5000개)
        self._X_buf: deque[np.ndarray] = deque(maxlen=MAX_BUF_SIZE)
        self._y_buf: deque[int] = deque(maxlen=MAX_BUF_SIZE)
        self._w_buf: deque[float] = deque(maxlen=MAX_BUF_SIZE)
        self._regime_buf: deque[str] = deque(maxlen=MAX_BUF_SIZE)
        self._new_since_retrain: int = 0

        # 코인별 시퀀스 버퍼 (LSTM/GRU 입력용, maxlen=SEQ_LEN)
        self._seq_buf: dict[str, deque[np.ndarray]] = defaultdict(
            lambda: deque(maxlen=SEQ_LEN)
        )

        if self._phase_b:
            self._init_seq_models()

    # ------------------------------------------------------------------
    # 예측 진입점
    # ------------------------------------------------------------------

    def predict(self, ms: MarketState) -> EnsemblePrediction:
        """MarketState → EnsemblePrediction.

        trade_count에 따라 콜드스타트/트리/시퀀스 경로 자동 선택.
        """
        X_raw = self._extract_features(ms)

        with self._lock:
            X_scaled = (
                self._scaler.transform(X_raw.reshape(1, -1))[0]
                if self._scaler_fitted
                else X_raw.copy()
            )

        # 시퀀스 버퍼에 추가 (항상, LSTM/GRU를 위해)
        self._seq_buf[ms.coin].append(X_scaled)

        trade_count = self._trade_count_fn()
        per_model: dict[str, float] = {}

        if trade_count < 200:
            # 콜드스타트: 룰 기반 스코어
            per_model["rule"] = self._rule_based_score(ms)
        else:
            regime_group = self._get_regime_group(ms)
            per_model.update(self._predict_tree_models(X_scaled, regime_group))

            if self._phase_b and trade_count >= 500:
                per_model.update(self._predict_seq_models(ms.coin))
            else:
                # 의도적: Phase B 비활성 시 가중치 0 유지 (분모 고정 목적).
                # per_model에 포함시켜 total_w 분모가 항상 3.6(xgb+lgb+lstm+gru)으로
                # 고정되도록 함 — 모델 추가/제거에 따른 threshold 판정 기준 변동 방지.
                per_model["lstm"] = 0.0
                per_model["gru"] = 0.0

        pred = self._build_prediction(per_model, ms)
        self._save_to_db(pred)
        return pred

    # ------------------------------------------------------------------
    # 증분 업데이트 (거래 완료 후 호출)
    # ------------------------------------------------------------------

    def update(
        self,
        ms: MarketState,
        label: int,
        sample_weight_multiplier: float = INCREMENTAL_WEIGHT,
    ) -> None:
        """새 거래 결과로 학습 버퍼 누적 + 주기적 재학습.

        Args:
            ms: 진입 시점의 MarketState
            label: 수익=1, 손실=0
            sample_weight_multiplier: 새 샘플 가중치 배율 (기본 ×2.0)
        """
        X = self._extract_features(ms)
        regime_group = self._get_regime_group(ms)

        with self._lock:
            self._X_buf.append(X)
            self._y_buf.append(label)
            self._w_buf.append(sample_weight_multiplier)
            self._regime_buf.append(regime_group)
            self._new_since_retrain += 1
            trigger = self._new_since_retrain >= self._retrain_interval

        if trigger:
            self._retrain_all()
            with self._lock:
                self._new_since_retrain = 0

    # ------------------------------------------------------------------
    # 배치 학습 (초기 로딩 or Colab 전이 학습)
    # ------------------------------------------------------------------

    def train_batch(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regime_group: str = "aggressive",
        sample_weights: np.ndarray | None = None,
    ) -> None:
        """배치 학습 — 초기 히스토리 로딩 또는 Walk-Forward 재학습 시 호출.

        Args:
            X: (N, 35) 피처 행렬
            y: (N,) 레이블 (0 or 1)
            regime_group: 'aggressive' | 'conservative'
            sample_weights: None이면 균일 가중치
        """
        try:
            from xgboost import XGBClassifier
            from lightgbm import LGBMClassifier
        except ImportError:
            logger.error("XGBoost/LightGBM 미설치 — 학습 불가")
            return

        if len(X) < 10:
            logger.warning("[Layer2] 학습 샘플 부족 (%d개) — 건너뜀", len(X))
            return

        w = sample_weights if sample_weights is not None else np.ones(len(y))

        with self._lock:
            if not self._scaler_fitted:
                self._scaler.fit(X)
                self._scaler_fitted = True
            # 청크 단위 transform — 전량 적재 시 M4 16GB RAM OOM 방지 (기획서 하드웨어 제약)
            _chunk = 10_000
            X_scaled = np.concatenate(
                [self._scaler.transform(X[i:i + _chunk]) for i in range(0, len(X), _chunk)]
            )

        xgb = XGBClassifier(**_XGB_PARAMS[regime_group])
        xgb.fit(X_scaled, y, sample_weight=w)

        lgb = LGBMClassifier(**_LGB_PARAMS[regime_group])
        lgb.fit(X_scaled, y, sample_weight=w)

        with self._lock:
            self._xgb[regime_group] = xgb
            self._lgb[regime_group] = lgb

        logger.info("[Layer2] %s 배치 학습 완료 (N=%d)", regime_group, len(X))

        if self._phase_b and len(X) >= SEQ_LEN:
            self._train_seq_models(X_scaled, y)

    # ------------------------------------------------------------------
    # 체크포인트 저장 / 로딩
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str | Path) -> None:
        """모델 + 스케일러 + 임계값 직렬화."""
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)

        with self._lock:
            ckpt = {
                "xgb": self._xgb,
                "lgb": self._lgb,
                "scaler": self._scaler,
                "scaler_fitted": self._scaler_fitted,
                "threshold": self._threshold,
            }
        with open(p / "ensemble.pkl", "wb") as f:
            pickle.dump(ckpt, f)

        if self._phase_b and _TORCH_AVAILABLE:
            if self._lstm is not None:
                torch.save(self._lstm.state_dict(), p / "lstm.pth")
            if self._gru is not None:
                torch.save(self._gru.state_dict(), p / "gru.pth")

        logger.info("[Layer2] 체크포인트 저장: %s", p)

    def load_checkpoint(self, path: str | Path) -> None:
        """체크포인트 복원."""
        p = Path(path)
        pkl = p / "ensemble.pkl"
        if not pkl.exists():
            logger.warning("[Layer2] 체크포인트 없음: %s", pkl)
            return

        with open(pkl, "rb") as f:
            ckpt = pickle.load(f)

        with self._lock:
            self._xgb = ckpt.get("xgb", {})
            self._lgb = ckpt.get("lgb", {})
            self._scaler = ckpt.get("scaler", StandardScaler())
            self._scaler_fitted = ckpt.get("scaler_fitted", False)
            self._threshold = ckpt.get("threshold", THRESHOLD_DEFAULT)

        if self._phase_b and _TORCH_AVAILABLE:
            lstm_p, gru_p = p / "lstm.pth", p / "gru.pth"
            if lstm_p.exists() and self._lstm is not None:
                self._lstm.load_state_dict(
                    torch.load(lstm_p, map_location="cpu", weights_only=True)
                )
            if gru_p.exists() and self._gru is not None:
                self._gru.load_state_dict(
                    torch.load(gru_p, map_location="cpu", weights_only=True)
                )

        logger.info("[Layer2] 체크포인트 로딩 완료: %s", p)

    # ------------------------------------------------------------------
    # 공개 유틸
    # ------------------------------------------------------------------

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        """Optuna 최적값 적용 시 호출."""
        if not 0.0 < value < 1.0:
            raise ValueError(f"threshold는 0~1 사이여야 함: {value}")
        self._threshold = value

    def is_trained(self, regime_group: str = "aggressive") -> bool:
        """해당 레짐 모델이 학습 완료 상태인지 확인."""
        return regime_group in self._xgb and regime_group in self._lgb

    # ------------------------------------------------------------------
    # 내부: Phase B 모델 초기화
    # ------------------------------------------------------------------

    def _init_seq_models(self) -> None:
        """LSTM + GRU 초기화 (device=cpu 강제).

        .to("cpu") 명시: load_checkpoint()의 map_location="cpu"와 일관성 유지.
        MPS/CUDA 환경에서 추론 텐서와 device mismatch 방지.
        """
        self._lstm = _LSTMModel().to("cpu")
        self._gru = _GRUModel().to("cpu")
        self._lstm_opt = torch.optim.Adam(self._lstm.parameters(), lr=0.001)
        self._gru_opt = torch.optim.Adam(self._gru.parameters(), lr=0.001)
        logger.info("[Layer2] LSTM + GRU 모델 초기화 (device=cpu)")

    # ------------------------------------------------------------------
    # 내부: 피처 추출
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_features(ms: MarketState) -> np.ndarray:
        """MarketState → np.ndarray (35,).

        누락 피처 발견 시 즉시 ValueError — 0.0은 실제 값(exchange_inflow=0.0)과
        구분 불가하므로 조용한 잘못된 예측보다 명시적 실패가 안전하다.
        """
        vals: list[float] = []
        for col in FEATURE_COLS:
            if not hasattr(ms, col):
                raise ValueError(f"누락 피처: {col}")
            vals.append(float(getattr(ms, col)))
        return np.array(vals, dtype=np.float32)

    # ------------------------------------------------------------------
    # 내부: 레짐 그룹 분류
    # ------------------------------------------------------------------

    @staticmethod
    def _get_regime_group(ms: MarketState) -> str:
        """HMM 레짐(Phase C) 또는 ADX 기반 aggressive/conservative 분류."""
        if ms.hmm_regime in (0, 1):
            return "aggressive"
        if ms.hmm_regime in (2, 3):
            return "conservative"
        # Phase A/B: ADX 기반 (>= 20 → 추세 장세 → aggressive)
        return "aggressive" if ms.adx_5m >= 20.0 else "conservative"

    # ------------------------------------------------------------------
    # 내부: 룰 기반 스코어 (콜드스타트)
    # ------------------------------------------------------------------

    @staticmethod
    def _rule_based_score(ms: MarketState) -> float:
        """기술지표 기반 0~1 확률 추정 (학습 불필요).

        구성 (합계 최대 1.0):
          RSI 위치:   0.00~0.40 (RSI 30~70 정규화)
          MACD 방향:  0.20 (MACD > signal)
          SuperTrend: 0.20 (supertrend_signal == +1)
          ADX 강도:   0.00~0.20 (ADX/50, 상한 0.20)
        """
        score = 0.0

        # RSI 기여: RSI 50 → 0.20, RSI 70 → 0.40, RSI 30 → 0.00
        rsi_clipped = max(30.0, min(70.0, ms.rsi_5m))
        score += (rsi_clipped - 30.0) / 40.0 * 0.40

        if ms.macd_5m > ms.macd_signal_5m:
            score += 0.20

        if ms.supertrend_signal == 1:
            score += 0.20

        score += min(ms.adx_5m / 50.0, 1.0) * 0.20

        return round(min(score, 1.0), 4)

    # ------------------------------------------------------------------
    # 내부: 트리 모델 예측
    # ------------------------------------------------------------------

    def _predict_tree_models(
        self, X_scaled: np.ndarray, regime_group: str
    ) -> dict[str, float]:
        """XGBoost + LightGBM 예측. 미학습 시 빈 dict 반환."""
        X2 = X_scaled.reshape(1, -1)
        result: dict[str, float] = {}

        with self._lock:
            # 의도적 폴백: conservative 모델 미학습 시 aggressive 모델 대체.
            # 이유: 모델 없이 예측 불가보다 보수적 폴백이 안전.
            xgb = self._xgb.get(regime_group) or self._xgb.get("aggressive")
            lgb = self._lgb.get(regime_group) or self._lgb.get("aggressive")

        if xgb is not None:
            try:
                result["xgb"] = float(xgb.predict_proba(X2)[0, 1])
            except Exception as exc:
                logger.error("[Layer2] XGB 예측 오류 — 해당 모델 제외: %s", exc)

        if lgb is not None:
            try:
                result["lgb"] = float(lgb.predict_proba(X2)[0, 1])
            except Exception as exc:
                logger.error("[Layer2] LGB 예측 오류 — 해당 모델 제외: %s", exc)

        return result

    # ------------------------------------------------------------------
    # 내부: 시퀀스 모델 예측 (Phase B)
    # ------------------------------------------------------------------

    def _predict_seq_models(self, coin: str) -> dict[str, float]:
        """LSTM + GRU 예측. 시퀀스 미충족(< SEQ_LEN) 시 빈 dict 반환."""
        if not self._phase_b or not _TORCH_AVAILABLE:
            return {}

        buf = self._seq_buf[coin]
        if len(buf) < SEQ_LEN:
            return {}

        seq = np.array(list(buf), dtype=np.float32)          # (60, 35)
        x = torch.tensor(seq).unsqueeze(0).to("cpu")         # (1, 60, 35) cpu 강제
        result: dict[str, float] = {}

        with torch.no_grad():
            if self._lstm is not None:
                result["lstm"] = float(self._lstm(x).squeeze())
            if self._gru is not None:
                result["gru"] = float(self._gru(x).squeeze())

        return result

    # ------------------------------------------------------------------
    # 내부: 앙상블 결합 → EnsemblePrediction
    # ------------------------------------------------------------------

    def _build_prediction(
        self, per_model: dict[str, float], ms: MarketState
    ) -> EnsemblePrediction:
        """가중 평균 계산 + 신호 확정 판단."""
        now = datetime.now(timezone.utc)

        if not per_model:
            return EnsemblePrediction(
                coin=ms.coin, timestamp=now,
                per_model_probs={}, weighted_avg=0.0,
                consensus_count=0, signal_confirmed=False,
                hmm_regime=ms.hmm_regime,
                threshold_used=self._threshold,
            )

        total_w = 0.0
        weighted_sum = 0.0
        consensus_count = 0

        for name, prob in per_model.items():
            w = MODEL_WEIGHTS.get(name, 0.6)
            weighted_sum += prob * w
            total_w += w
            if prob >= self._consensus_threshold:
                consensus_count += 1

        weighted_avg = weighted_sum / total_w if total_w > 0 else 0.0

        # 신호 확정: 가중 평균 >= threshold AND 합의 모델 >= 3
        # 콜드스타트(rule 1개)는 consensus < 3 → signal_confirmed=False 보장
        signal_confirmed = (
            weighted_avg >= self._threshold
            and consensus_count >= MIN_CONSENSUS
        )

        return EnsemblePrediction(
            coin=ms.coin,
            timestamp=now,
            per_model_probs=per_model,
            weighted_avg=round(weighted_avg, 4),
            consensus_count=consensus_count,
            signal_confirmed=signal_confirmed,
            hmm_regime=ms.hmm_regime,
            hmm_confidence=0.0,
            threshold_used=self._threshold,
        )

    # ------------------------------------------------------------------
    # 내부: SQLite 저장
    # ------------------------------------------------------------------

    def _save_to_db(self, pred: EnsemblePrediction) -> None:
        if not self._cache:
            return
        try:
            row: dict[str, Any] = {
                "coin": pred.coin,
                "timestamp": pred.timestamp.isoformat(),
                "xgb_prob":  pred.per_model_probs.get("xgb"),
                "lgb_prob":  pred.per_model_probs.get("lgb"),
                "lstm_prob": pred.per_model_probs.get("lstm"),
                "gru_prob":  pred.per_model_probs.get("gru"),
                "weighted_avg":    pred.weighted_avg,
                "consensus_count": pred.consensus_count,
                "signal_confirmed": int(pred.signal_confirmed),
                "hmm_regime":      pred.hmm_regime,
                "hmm_confidence":  pred.hmm_confidence,
                "threshold_used":  pred.threshold_used,
            }
            self._cache.upsert_candle("ensemble_predictions", pred.coin, row)
        except Exception as exc:
            logger.warning("[Layer2] DB 저장 실패: %s", exc)
            # 의도적 계속 진행: 예측 결과 DB 저장 실패는 트레이딩 중단 사유 아님

    # ------------------------------------------------------------------
    # 내부: 주기적 전체 재학습
    # ------------------------------------------------------------------

    def _retrain_all(self) -> None:
        """버퍼 전체로 XGBoost + LightGBM 재학습."""
        try:
            from xgboost import XGBClassifier
            from lightgbm import LGBMClassifier
        except ImportError:
            logger.warning("[Layer2] XGBoost/LightGBM 미설치 — 재학습 불가")
            return

        with self._lock:
            X = np.array(list(self._X_buf), dtype=np.float32)
            y = np.array(list(self._y_buf), dtype=np.int32)
            w = np.array(list(self._w_buf), dtype=np.float32)
            regimes = list(self._regime_buf)

        if len(X) < 10:
            return

        if not self._scaler_fitted:
            self._scaler.fit(X)
            self._scaler_fitted = True
        X_scaled = self._scaler.transform(X)

        for rg in ("aggressive", "conservative"):
            mask = np.array([r == rg for r in regimes])
            if mask.sum() < 5:
                continue
            Xi, yi, wi = X_scaled[mask], y[mask], w[mask]
            xgb = XGBClassifier(**_XGB_PARAMS[rg])
            xgb.fit(Xi, yi, sample_weight=wi)
            lgb = LGBMClassifier(**_LGB_PARAMS[rg])
            lgb.fit(Xi, yi, sample_weight=wi)
            with self._lock:
                self._xgb[rg] = xgb
                self._lgb[rg] = lgb

        logger.info("[Layer2] 자동 재학습 완료 (버퍼 %d개)", len(X))

        if self._phase_b and len(X_scaled) >= SEQ_LEN:
            self._train_seq_models(X_scaled, y)

    # ------------------------------------------------------------------
    # 내부: LSTM/GRU 슬라이딩 윈도우 학습
    # ------------------------------------------------------------------

    def _train_seq_models(
        self, X_scaled: np.ndarray, y: np.ndarray
    ) -> None:
        """슬라이딩 윈도우로 시퀀스 생성 → LSTM + GRU 미니배치 학습."""
        if not self._phase_b or not _TORCH_AVAILABLE:
            return

        seqs, labels = [], []
        for i in range(SEQ_LEN, len(X_scaled)):
            seqs.append(X_scaled[i - SEQ_LEN:i])
            labels.append(y[i])

        if not seqs:
            return

        X_seq = torch.tensor(np.array(seqs, dtype=np.float32))
        y_t = torch.tensor(
            np.array(labels, dtype=np.float32)
        ).unsqueeze(1)
        criterion = torch.nn.BCELoss()

        for model, opt in [
            (self._lstm, self._lstm_opt),
            (self._gru, self._gru_opt),
        ]:
            if model is None or opt is None:
                continue
            model.train()
            for start in range(0, len(X_seq), BATCH_SIZE):
                bx = X_seq[start:start + BATCH_SIZE]
                by = y_t[start:start + BATCH_SIZE]
                opt.zero_grad()
                loss = criterion(model(bx), by)
                loss.backward()
                opt.step()
            model.eval()

        logger.info("[Layer2] LSTM/GRU 학습 완료 (%d 시퀀스)", len(seqs))


# ------------------------------------------------------------------
# Optuna 하이퍼파라미터 탐색 (Walk-Forward OOS 샤프비율 최대화)
# ------------------------------------------------------------------

def run_optuna_study(
    X: np.ndarray,
    y: np.ndarray,
    n_trials: int = 200,
    timeout_sec: int = 3600,
) -> dict[str, Any]:
    """Walk-Forward OOS 기반 앙상블 하이퍼파라미터 탐색.

    탐색 범위 (upbit_quant_v9.md 기준):
        ensemble_threshold: 0.55~0.75  [신규-v9]
        n_estimators:       100~500
        max_depth:          3~8
        learning_rate:      0.01~0.1

    목적함수: Walk-Forward OOS 샤프비율 최대화
    MedianPruner 조기종료, n_trials=200, M4 1~2시간.

    Returns:
        best_params dict (ensemble_threshold 포함)
    """
    try:
        import optuna
        from xgboost import XGBClassifier
    except ImportError:
        logger.error("[Optuna] optuna 또는 xgboost 미설치")
        return {"ensemble_threshold": THRESHOLD_DEFAULT}

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    scaler = StandardScaler()

    def objective(trial: Any) -> float:
        threshold = trial.suggest_float("ensemble_threshold", 0.55, 0.75)
        n_est = trial.suggest_int("n_estimators", 100, 500)
        depth = trial.suggest_int("max_depth", 3, 8)
        lr = trial.suggest_float("learning_rate", 0.01, 0.1, log=True)

        # Walk-Forward: 80% 학습 / 20% OOS
        split = int(len(X) * 0.8)
        if split < 10:
            return -999.0

        X_tr = scaler.fit_transform(X[:split])
        X_ts = scaler.transform(X[split:])
        y_tr, y_ts = y[:split], y[split:]

        xgb = XGBClassifier(
            n_estimators=n_est, max_depth=depth, learning_rate=lr,
            eval_metric="logloss", random_state=42,
        )
        xgb.fit(X_tr, y_tr)

        probs = xgb.predict_proba(X_ts)[:, 1]
        signals = probs >= threshold
        # 단순 수익률 모사: 맞추면 +2%, 틀리면 -1%
        rets = np.where(
            signals & (y_ts == 1), 0.02,
            np.where(signals & (y_ts == 0), -0.01, 0.0),
        )

        std = float(rets.std())
        if std < 1e-9:
            return -999.0
        # 5분봉 연환산 (252일 × 24h × 12 = 72576 기간)
        sharpe = float(rets.mean()) / std * np.sqrt(252 * 24 * 12)
        return sharpe

    pruner = optuna.pruners.MedianPruner(n_startup_trials=10)
    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=n_trials, timeout=timeout_sec)

    best = study.best_params
    logger.info(
        "[Optuna] 최적 파라미터: %s (OOS 샤프: %.3f)", best, study.best_value
    )
    return best
